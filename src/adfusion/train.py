"""Nested cross-validation runner.

For each model and each outer fold:

1. the outer test fold is set aside and never used for any decision;
2. an inner validation split is carved out of the outer training set;
3. the model is trained on the rest with class-balanced BCE, Adam and cosine
   annealing; the epoch with the best inner-validation AUC is kept and the
   decision threshold is tuned on the inner validation set;
4. the kept model predicts the outer test fold once.

Outputs written to ``out_dir``:

    results_per_fold.csv   per-fold metrics at 0.5 and at the inner threshold
    results_summary.csv    fold-mean +/- sd and pooled AUC with bootstrap CI
    paired_tests.csv       paired bootstrap tests with Holm adjustment
    oof_predictions.json   out-of-fold labels and probabilities
    learning_curves.json   per-epoch training loss and validation AUC
    run_config.json        configuration, parameter counts, versions
    checkpoints/           one state dict per model, repeat and fold
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import platform
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import __version__
from .data import assert_disjoint, make_loaders, nested_splits
from .metrics import best_threshold, bootstrap_ci, holm, metrics_from_preds, paired_bootstrap
from .models import MODEL_NAMES, build_model, count_params

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training and evaluation protocol. Defaults reproduce the SIST 2026 runs."""

    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    wd: float = 1e-4
    folds: int = 5
    inner_frac: float = 0.2
    repeats: int = 1
    patience: int = 15
    capacity_matched: bool = True
    seed: int = 42
    workers: int = 0
    n_boot: int = 2000
    device: str = "auto"
    save_checkpoints: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        bad = [m for m in self.models if m not in MODEL_NAMES]
        if bad:
            raise ValueError(f"unknown models {bad}; choose from {MODEL_NAMES}")
        if self.folds < 2:
            raise ValueError("folds must be >= 2")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (seeds augmentation, see data.py)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    """Return (labels, probabilities) for every sample in ``loader``."""
    model.eval()
    probs, ys = [], []
    for mri, pet, y in loader:
        p = torch.sigmoid(model(mri.to(device), pet.to(device)))
        probs.append(p.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_fold(name, tr_recs, va_recs, te_recs, device, cfg: TrainConfig):
    """Train on ``tr``, select on ``va`` (inner), predict ``te`` (outer) once."""
    model = build_model(name, capacity_matched=cfg.capacity_matched).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    n_pos = sum(r["y"] for r in tr_recs)
    n_neg = len(tr_recs) - n_pos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    tr_loader, va_loader = make_loaders(
        tr_recs, va_recs, batch_size=cfg.batch_size, num_workers=cfg.workers, augment=True
    )
    _, te_loader = make_loaders(
        tr_recs, te_recs, batch_size=cfg.batch_size, num_workers=cfg.workers, augment=False
    )

    best = {"auc": -1.0, "state": None, "thr": 0.5, "epoch": 0}
    history, bad = [], 0
    for ep in range(cfg.epochs):
        model.train()
        run = 0.0
        for mri, pet, y in tr_loader:
            mri, pet, y = mri.to(device), pet.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = lossf(model(mri, pet), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * y.size(0)
        sched.step()
        tr_loss = run / max(len(tr_recs), 1)

        yv, pv = predict(model, va_loader, device)  # inner validation only
        mv = metrics_from_preds(yv, pv)
        history.append(
            {"epoch": ep + 1, "train_loss": tr_loss, "val_auc": mv["auc"], "val_acc": mv["acc"]}
        )
        log.debug("%s ep %d loss=%.3f val_auc=%.3f", name, ep + 1, tr_loss, mv["auc"])
        if not np.isnan(mv["auc"]) and mv["auc"] > best["auc"] + 1e-4:
            best = {
                "auc": mv["auc"],
                "thr": best_threshold(yv, pv),
                "epoch": ep + 1,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    yt, pt = predict(model, te_loader, device)  # outer test fold, used once
    return model, yt, pt, best["thr"], history


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_cv(records: list[dict], cfg: TrainConfig, out_dir: str | Path) -> list[dict]:
    """Run nested CV for every model in ``cfg.models``; return the summary rows."""
    cfg.validate()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = Path(out_dir)
    ck = out / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    log.info(
        "device=%s subjects=%d balance=%s",
        device,
        len(records),
        dict(Counter(r["label"] for r in records)),
    )

    y_all = np.array([r["y"] for r in records])
    splits = list(nested_splits(y_all, cfg.folds, cfg.inner_frac, cfg.seed, cfg.repeats))
    for s in splits:
        assert_disjoint(s, records)

    per_fold, summary, oof_store, hist_store, pcounts = [], [], {}, {}, {}
    for name in cfg.models:
        pcounts[name] = count_params(build_model(name, capacity_matched=cfg.capacity_matched))
        log.info("model %s (%.3fM params)", name, pcounts[name] / 1e6)
        fold_ms, oof_y, oof_p, oof_id, oof_subj = [], [], [], [], []
        t0 = time.time()
        for s in splits:
            tr = [records[i] for i in s.train]
            va = [records[i] for i in s.val]
            te = [records[i] for i in s.test]
            model, yt, pt, thr, hist = train_fold(name, tr, va, te, device, cfg)
            tag = f"{name}_r{s.repeat}_f{s.fold}"
            if cfg.save_checkpoints:
                torch.save(model.state_dict(), ck / f"{tag}.pt")
            m_fixed = metrics_from_preds(yt, pt, 0.5)
            m_tuned = metrics_from_preds(yt, pt, thr)
            fold_ms.append(m_fixed)
            oof_y.append(yt)
            oof_p.append(pt)
            oof_id.extend([f"r{s.repeat}f{s.fold}"] * len(yt))
            oof_subj.extend(r["subject"] for r in te)
            hist_store[tag] = hist
            per_fold.append(
                {
                    "model": name,
                    "repeat": s.repeat,
                    "fold": s.fold,
                    "thr_inner": round(thr, 3),
                    **{k: round(v, 4) for k, v in m_fixed.items()},
                    "acc_tuned": round(m_tuned["acc"], 4),
                    "sens_tuned": round(m_tuned["sens"], 4),
                    "spec_tuned": round(m_tuned["spec"], 4),
                }
            )
            log.info("  %s auc=%.3f acc=%.3f", tag, m_fixed["auc"], m_fixed["acc"])

        y_cat, p_cat = np.concatenate(oof_y), np.concatenate(oof_p)
        oof_store[name] = {
            "y": y_cat.tolist(),
            "p": p_cat.tolist(),
            "fold": oof_id,
            "subject": oof_subj,
        }
        row = {"model": name, "params_M": round(pcounts[name] / 1e6, 3)}
        for k in ("acc", "auc", "sens", "spec"):
            v = [f[k] for f in fold_ms if not np.isnan(f[k])]
            row[f"{k}_foldmean"] = round(float(np.mean(v)), 4) if v else float("nan")
            row[f"{k}_foldsd"] = round(float(np.std(v)), 4) if v else float("nan")
        lo, hi = bootstrap_ci(y_cat, p_cat, n=cfg.n_boot)
        row["auc_pooled"] = round(float(roc_auc_score(y_cat, p_cat)), 4)
        row["auc_pooled_ci_low"] = round(lo, 4)
        row["auc_pooled_ci_high"] = round(hi, 4)
        row["minutes"] = round((time.time() - t0) / 60, 2)
        summary.append(row)

    tests = []
    for a, b in itertools.combinations([n for n in cfg.models if n in oof_store], 2):
        ya, pa = np.array(oof_store[a]["y"]), np.array(oof_store[a]["p"])
        yb, pb = np.array(oof_store[b]["y"]), np.array(oof_store[b]["p"])
        if oof_store[a]["subject"] != oof_store[b]["subject"] or not np.array_equal(ya, yb):
            continue  # compare only on identical subject order
        p, (dlo, dhi) = paired_bootstrap(ya, pa, pb, n=cfg.n_boot)
        auc_a, auc_b = roc_auc_score(ya, pa), roc_auc_score(yb, pb)
        tests.append(
            {
                "model_a": a,
                "model_b": b,
                "auc_a": round(float(auc_a), 4),
                "auc_b": round(float(auc_b), 4),
                "delta": round(float(auc_a - auc_b), 4),
                "ci_low": round(dlo, 4),
                "ci_high": round(dhi, 4),
                "p_raw": round(p, 4),
            }
        )
    if tests:
        for t, adj in zip(tests, holm([t["p_raw"] for t in tests])):
            t["p_holm"] = round(float(adj), 4)

    _write_csv(out / "results_per_fold.csv", per_fold)
    _write_csv(out / "results_summary.csv", summary)
    _write_csv(out / "paired_tests.csv", tests)
    (out / "oof_predictions.json").write_text(json.dumps(oof_store))
    (out / "learning_curves.json").write_text(json.dumps(hist_store))
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "params": pcounts,
                "n_subjects": len(records),
                "versions": {
                    "adfusion": __version__,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                },
                "device": str(device),
            },
            indent=2,
        )
    )
    return summary
