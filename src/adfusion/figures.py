"""Greyscale figures from the outputs of :func:`adfusion.train.run_cv`.

Fig_ROC.png            ROC curves of pooled out-of-fold predictions
Fig_FoldBoxplot.png    per-fold AUC distributions
Fig_LearningCurves.png training loss and inner-validation AUC
Fig_AUC_matched.png    pooled AUC (own CI) next to fold-mean AUC (sd)
Fig_Confusion.png      confusion matrices at the a priori 0.5 threshold
Fig_FusionWeights.png  learned branch weights of the hybrid model
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})
NICE = {
    "mri": "MRI-only",
    "pet": "PET-only",
    "early": "Early",
    "late": "Late",
    "intermediate": "Intermediate",
    "hybrid": "Hybrid (cross-attn)",
    "hybrid_noattn": "Hybrid w/o attn",
}
_STYLES = ["-", "--", "-.", ":", "-", "--", "-."]
_GREYS = ["0.05", "0.25", "0.40", "0.55", "0.15", "0.35", "0.60"]


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_results(res_dir: str | Path) -> dict:
    res = Path(res_dir)
    with open(res / "results_per_fold.csv") as fh:
        per = list(csv.DictReader(fh))
    with open(res / "results_summary.csv") as fh:
        summ = list(csv.DictReader(fh))
    return {
        "oof": json.loads((res / "oof_predictions.json").read_text()),
        "hist": json.loads((res / "learning_curves.json").read_text()),
        "per": per,
        "summ": summ,
    }


def fig_roc(oof: dict, out: Path, dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=dpi)
    for i, (name, d) in enumerate(oof.items()):
        y, p = np.array(d["y"]), np.array(d["p"])
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(
            fpr,
            tpr,
            _STYLES[i % 7],
            color=_GREYS[i % 7],
            lw=1.3,
            label=f"{NICE.get(name, name)} (AUC {roc_auc_score(y, p):.3f})",
        )
    ax.plot([0, 1], [0, 1], lw=0.7, color="0.7", ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    _clean(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    return _save(fig, out / "Fig_ROC.png")


def fig_box(per: list[dict], out: Path, dpi: int = 300) -> Path:
    models = list(dict.fromkeys(r["model"] for r in per))
    data = [
        [float(r["auc"]) for r in per if r["model"] == m and r["auc"] not in ("", "nan")]
        for m in models
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=dpi)
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
    for b in bp["boxes"]:
        b.set_facecolor("0.85")
        b.set_edgecolor("black")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.05, len(d)), d, s=9, color="0.25", zorder=3)
    ax.axhline(0.5, ls="--", lw=0.7, color="0.5")
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([NICE.get(m, m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("Per-fold AUC")
    _clean(ax)
    return _save(fig, out / "Fig_FoldBoxplot.png")


def fig_curves(hist: dict, out: Path, dpi: int = 300) -> Path:
    models = list(dict.fromkeys(k.rsplit("_r", 1)[0] for k in hist))
    n = len(models)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), dpi=dpi, squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // ncol][i % ncol]
        ax2 = ax.twinx()
        for k, h in hist.items():
            if k.rsplit("_r", 1)[0] != m:
                continue
            ep = [x["epoch"] for x in h]
            ax.plot(ep, [x["train_loss"] for x in h], color="0.25", lw=0.8, alpha=0.7)
            ax2.plot(ep, [x["val_auc"] for x in h], color="0.55", lw=0.8, ls="--", alpha=0.8)
        ax.set_title(NICE.get(m, m), fontsize=8)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("train loss", fontsize=7)
        ax2.set_ylabel("val AUC", fontsize=7)
        ax2.set_ylim(0.0, 1.0)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return _save(fig, out / "Fig_LearningCurves.png")


def _interval(ax, x, lo, hi, cap: float = 0.06) -> None:
    """Draw vertical intervals with caps (avoids matplotlib yerr edge cases)."""
    ax.vlines(x, lo, hi, color="black", lw=0.8)
    ax.hlines(
        np.concatenate([lo, hi]),
        np.concatenate([x - cap, x - cap]),
        np.concatenate([x + cap, x + cap]),
        color="black",
        lw=0.8,
    )


def fig_auc_matched(summ: list[dict], out: Path, dpi: int = 300) -> Path:
    names = [r["model"] for r in summ]
    pooled = np.array([float(r["auc_pooled"]) for r in summ])
    lo = np.array([float(r["auc_pooled_ci_low"]) for r in summ])
    hi = np.array([float(r["auc_pooled_ci_high"]) for r in summ])
    fm = np.array([float(r["auc_foldmean"]) for r in summ])
    fs = np.array([float(r["auc_foldsd"]) for r in summ])
    x, w = np.arange(len(names), dtype=float), 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=dpi)
    ax.bar(
        x - w / 2,
        pooled,
        w,
        color="0.45",
        edgecolor="black",
        lw=0.6,
        label="pooled OOF AUC (95% CI)",
    )
    ax.bar(x + w / 2, fm, w, color="0.80", edgecolor="black", lw=0.6, label="fold-mean AUC (sd)")
    _interval(ax, x - w / 2, lo, hi)
    _interval(ax, x + w / 2, fm - fs, fm + fs)
    ax.axhline(0.5, ls="--", lw=0.7, color="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(n, n) for n in names], rotation=20, ha="right")
    ax.set(ylabel="AUC", ylim=(0.0, 1.05))
    _clean(ax)
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out / "Fig_AUC_matched.png")


def fig_confusion(oof: dict, out: Path, dpi: int = 300) -> Path:
    names = list(oof)
    n = len(names)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.3 * nrow), dpi=dpi, squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        y, p = np.array(oof[name]["y"]), np.array(oof[name]["p"])
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=max(cm.max(), 1))
        for a in range(2):
            for b in range(2):
                ax.text(
                    b,
                    a,
                    cm[a, b],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[a, b] > cm.max() * 0.6 else "black",
                )
        ax.set_xticks([0, 1], ["CN", "AD"], fontsize=7)
        ax.set_yticks([0, 1], ["CN", "AD"], fontsize=7)
        ax.set_title(NICE.get(name, name), fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return _save(fig, out / "Fig_Confusion.png")


def fusion_weights(ckpt_dir: str | Path, model: str = "hybrid_noattn") -> np.ndarray:
    """Softmax branch weights (fused, MRI, PET) from every checkpoint of ``model``.

    Checkpoints are matched on the exact model stem, so ``hybrid`` does not
    pick up ``hybrid_noattn`` files.
    """
    import torch

    rows = []
    for f in sorted(Path(ckpt_dir).glob("*.pt")):
        if f.stem.rsplit("_r", 1)[0] != model:
            continue
        state = torch.load(f, map_location="cpu", weights_only=True)
        if "w" in state:
            rows.append(torch.softmax(state["w"].float(), dim=0).numpy())
    return np.array(rows).reshape(-1, 3)


def fig_fusion_weights(weights: np.ndarray, out: Path, dpi: int = 300) -> Path:
    mu, sd = weights.mean(0), weights.std(0)
    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=dpi)
    x = np.arange(3, dtype=float)
    ax.bar(x, mu, color=["0.3", "0.6", "0.85"], edgecolor="black")
    _interval(ax, x, mu - sd, mu + sd)
    ax.axhline(1 / 3, ls="--", lw=0.7, color="0.5")
    ax.set_xticks(range(3), ["fused", "sMRI", "PET"])
    ax.set_ylabel("softmax weight")
    _clean(ax)
    return _save(fig, out / "Fig_FusionWeights.png")


def make_all(res_dir: str | Path, out_dir: str | Path | None = None, dpi: int = 300) -> list[Path]:
    """Write every figure that the available outputs allow."""
    out = Path(out_dir or res_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = load_results(res_dir)
    paths = [
        fig_roc(r["oof"], out, dpi),
        fig_box(r["per"], out, dpi),
        fig_curves(r["hist"], out, dpi),
        fig_auc_matched(r["summ"], out, dpi),
        fig_confusion(r["oof"], out, dpi),
    ]
    ck = Path(res_dir) / "checkpoints"
    for m in ("hybrid_noattn", "hybrid"):
        w = fusion_weights(ck, m) if ck.exists() else np.empty((0, 3))
        if len(w):
            paths.append(fig_fusion_weights(w, out, dpi))
            break
    return paths
