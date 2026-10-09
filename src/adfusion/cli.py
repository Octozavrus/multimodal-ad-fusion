"""Command-line interface: ``adfusion <command> [options]``.

Commands
    preprocess   co-register, crop, resample and normalise paired T1 and FDG-PET scans
    synth        write synthetic raw scans for a demonstration (no patient data needed)
"""

from __future__ import annotations

import argparse
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


def _cmd_synth(a) -> int:
    from .synthetic import make_synthetic_raw

    raw, labels = make_synthetic_raw(a.out, a.n, a.shape, seed=a.seed)
    print(f"synthetic raw scans for {a.n} subjects -> {raw}; labels -> {labels}")
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

    s = sub.add_parser("synth", help="write synthetic raw scans")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--shape", type=_shape, default=(40, 40, 40))
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
