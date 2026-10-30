"""Command-line interface: ``adfusion <command> [options]``.

Preprocessing (core install)
    synth         write synthetic raw scans (no patient data needed)
    preprocess    co-register, crop, resample and normalise paired T1 and FDG-PET scans
    qc            summarise registration quality from a cache manifest

Cohort, training and evaluation (``pip install "adfusion[train]"``)
    cohort        intersect ADNI MRI and PET search exports
    synth-cohort  write a synthetic preprocessed cohort with a planted AD signal
    roi-cache     build the central-ROI variant of a cache
    mixed         pair ROI MRI with whole-brain PET in one manifest
    train         nested cross-validation of one or more fusion models
    figures       greyscale figures from a results directory
    info          list models and parameter counts
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from . import __version__


def _shape(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or min(parts) < 8:
        raise argparse.ArgumentTypeError("shape must be N or D,H,W with every side >= 8")
    return tuple(parts)  # type: ignore[return-value]


def _model_shape(s: str) -> tuple[int, int, int]:
    shape = _shape(s)
    if min(shape) < 16:
        raise argparse.ArgumentTypeError("model inputs need every side >= 16")
    return shape


_EXTRA_MSG = (
    "this command needs the training dependencies: "
    'pip install "adfusion[train]" (PyTorch, scikit-learn, matplotlib)'
)


def _cmd_preprocess(a) -> int:
    from .preprocessing import PreprocessConfig, run_preprocessing

    cfg = PreprocessConfig(
        download_dir=Path(a.raw),
        output_dir=Path(a.out),
        label_files=tuple(Path(p) for p in a.labels),
        template=Path(a.template) if a.template else None,
        target_shape=a.shape,
        fg_frac=a.fg_frac,
        clip_sigma=a.clip_sigma,
        affine=not a.rigid_only,
    )
    s = run_preprocessing(cfg)
    print(json.dumps({k: v for k, v in s.items() if k != "failures"}, indent=2))
    if s["failures"]:
        print(f"{len(s['failures'])} subjects were not processed; see preprocess_summary.json")
    return 0


def qc_report(manifest: str | Path, threshold: float = 0.5) -> dict:
    """Registration Dice statistics and the subjects below ``threshold``."""
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    vals = [(r["subject"], float(r["reg_dice"])) for r in rows if r.get("reg_dice")]
    dice = [d for _, d in vals]
    return {
        "n_subjects": len(rows),
        "n_with_dice": len(dice),
        "dice_mean": round(sum(dice) / len(dice), 4) if dice else None,
        "dice_min": round(min(dice), 4) if dice else None,
        "dice_max": round(max(dice), 4) if dice else None,
        "below_threshold": sorted(s for s, d in vals if d < threshold),
    }


def _cmd_qc(a) -> int:
    rep = qc_report(a.manifest, a.threshold)
    print(json.dumps(rep, indent=2))
    return 1 if (a.strict and rep["below_threshold"]) else 0


def _cmd_synth(a) -> int:
    from .synthetic import make_synthetic_raw

    raw, labels = make_synthetic_raw(a.out, a.n, a.shape, seed=a.seed)
    print(f"synthetic raw scans for {a.n} subjects -> {raw}; labels -> {labels}")
    return 0


def _cmd_cohort(a) -> int:
    from .cohort import intersect

    res = intersect(a.mri_csv, a.pet_csv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cohort_both_modalities.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "group"])
        w.writerows([[s, res["groups"].get(s, "")] for s in res["subjects"]])
    with open(out / "best_image_per_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "modality", "description"])
        w.writeheader()
        w.writerows(res["best_images"])
    print(
        f"subjects with both modalities: {len(res['subjects'])} "
        f"(MRI only {res['n_mri_only']}, PET only {res['n_pet_only']}) -> {out}"
    )
    return 0


def _cmd_synth_cohort(a) -> int:
    from .synthetic import make_synthetic_cohort

    man = make_synthetic_cohort(a.out, a.n, a.shape, a.effect, a.seed)
    print(f"synthetic cohort: {a.n} subjects -> {man}")
    return 0


def _cmd_roi(a) -> int:
    from .variants import build_variant

    print(build_variant(a.manifest, a.out, mode="roi", target=a.shape))
    return 0


def _cmd_mixed(a) -> int:
    from .variants import build_mixed_manifest

    print(build_mixed_manifest(a.full, a.roi, a.out))
    return 0


_TRAIN_FLAGS = {
    "epochs": int,
    "batch_size": int,
    "lr": float,
    "wd": float,
    "folds": int,
    "inner_frac": float,
    "repeats": int,
    "patience": int,
    "seed": int,
    "workers": int,
    "n_boot": int,
    "device": str,
}


def _cmd_train(a) -> int:
    from .data import load_manifest
    from .train import TrainConfig, run_cv

    cfg = TrainConfig.from_json(a.config) if a.config else TrainConfig()
    for k in _TRAIN_FLAGS:
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.models:
        cfg.models = a.models
    if a.naive_capacity:
        cfg.capacity_matched = False
    if a.no_checkpoints:
        cfg.save_checkpoints = False
    recs = load_manifest(a.manifest, a.data_root)
    summary = run_cv(recs, cfg, a.out)
    print(f"{'model':16s} {'params':>7s} {'pooled AUC [95% CI]':>24s} {'fold-mean AUC':>16s}")
    for r in summary:
        print(
            f"{r['model']:16s} {r['params_M']:6.3f}M "
            f"{r['auc_pooled']:8.3f} [{r['auc_pooled_ci_low']:.3f}-{r['auc_pooled_ci_high']:.3f}]"
            f" {r['auc_foldmean']:9.3f}+/-{r['auc_foldsd']:.3f}"
        )
    print(f"results written to {a.out}")
    return 0


def _cmd_figures(a) -> int:
    from .figures import make_all

    for p in make_all(a.results, a.out, a.dpi):
        print(f"wrote {p}")
    return 0


def _cmd_info(a) -> int:
    from .models import MODEL_NAMES, build_model, count_params

    print(f"adfusion {__version__}")
    print(f"{'model':16s} {'naive':>10s} {'matched':>10s}")
    for n in MODEL_NAMES:
        p0 = count_params(build_model(n)) / 1e6
        p1 = count_params(build_model(n, capacity_matched=True)) / 1e6
        print(f"{n:16s} {p0:9.3f}M {p1:9.3f}M")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adfusion", description="Preprocessing of paired sMRI and FDG-PET scans"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("preprocess", help="build the co-registered cache")
    s.add_argument(
        "--raw", required=True, help="folder with downloaded NIfTI files or DICOM series"
    )
    s.add_argument("--labels", nargs="+", required=True, help="CSV files with Subject and Group")
    s.add_argument("--out", required=True, help="output folder for the cache")
    s.add_argument("--template", default=None, help="optional T1 template for a common space")
    s.add_argument("--shape", type=_shape, default=(96, 96, 96), help="output shape (default 96)")
    s.add_argument("--fg-frac", type=float, default=0.05, help="foreground threshold fraction")
    s.add_argument("--clip-sigma", type=float, default=5.0, help="z-score clipping limit")
    s.add_argument("--rigid-only", action="store_true", help="skip the affine stage")
    s.set_defaults(func=_cmd_preprocess)

    s = sub.add_parser("qc", help="registration quality summary")
    s.add_argument("manifest", help="cache_manifest.csv written by preprocess")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--strict", action="store_true", help="exit with 1 if any subject is below")
    s.set_defaults(func=_cmd_qc)

    s = sub.add_parser("synth", help="write synthetic raw scans")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--shape", type=_shape, default=(40, 40, 40))
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_synth)

    s = sub.add_parser("cohort", help="intersect ADNI MRI and PET search exports")
    s.add_argument("mri_csv")
    s.add_argument("pet_csv")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_cohort)

    s = sub.add_parser("synth-cohort", help="write a synthetic preprocessed cohort")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--shape", type=_model_shape, default=(32, 32, 32))
    s.add_argument("--effect", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_synth_cohort)

    s = sub.add_parser("roi-cache", help="central ROI variant of a cache")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--shape", type=_model_shape, default=(96, 96, 96))
    s.set_defaults(func=_cmd_roi)

    s = sub.add_parser("mixed", help="ROI MRI + whole-brain PET manifest")
    s.add_argument("--full", required=True, help="whole-brain cache_manifest.csv")
    s.add_argument("--roi", required=True, help="ROI cache_manifest.csv")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_mixed)

    s = sub.add_parser("train", help="nested cross-validation")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config", default=None, help="JSON protocol file (see configs/)")
    s.add_argument("--data-root", dest="data_root", default=None)
    s.add_argument("--models", nargs="+", default=None)
    for k, t in _TRAIN_FLAGS.items():
        s.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    s.add_argument("--naive-capacity", action="store_true", help="do not widen 1-encoder models")
    s.add_argument("--no-checkpoints", action="store_true")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("figures", help="figures from a results directory")
    s.add_argument("results")
    s.add_argument("--out", default=None)
    s.add_argument("--dpi", type=int, default=300)
    s.set_defaults(func=_cmd_figures)

    s = sub.add_parser("info", help="models and parameter counts")
    s.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in {"torch", "sklearn", "matplotlib", "pandas"}:
            print(f"adfusion {args.command}: {_EXTRA_MSG}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
