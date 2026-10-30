"""Derived input caches used by the follow-up experiments.

``build_variant`` writes a new cache in which every volume is transformed
(for example the central ROI crop resampled back to the encoder input size).
``build_mixed_manifest`` pairs MRI from one cache with PET from another, so
each modality can receive the spatial extent that suits it.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from .preprocessing import resize_to, roi_crop

log = logging.getLogger(__name__)
FIELDS = ["subject", "label", "mri", "pet"]


def _read(manifest: str | Path) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write(manifest: Path, rows: list[dict]) -> Path:
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return manifest


def build_variant(
    src_manifest: str | Path,
    dst_dir: str | Path,
    mode: str = "roi",
    target: tuple[int, int, int] = (96, 96, 96),
) -> Path:
    """Write a transformed copy of a cache. ``mode`` is ``"roi"`` or ``"resize"``."""
    if mode not in ("roi", "resize"):
        raise ValueError("mode must be 'roi' or 'resize'")
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in _read(src_manifest):
        om, op = dst / f"{r['subject']}_MRI.npy", dst / f"{r['subject']}_PET.npy"
        if not (om.exists() and op.exists()):
            mri, pet = np.load(r["mri"]), np.load(r["pet"])
            if mode == "roi":
                mri, pet = roi_crop(mri), roi_crop(pet)
            np.save(om, resize_to(mri, target))
            np.save(op, resize_to(pet, target))
        out_rows.append({"subject": r["subject"], "label": r["label"], "mri": om, "pet": op})
    return _write(dst / "cache_manifest.csv", out_rows)


def build_mixed_manifest(
    full_manifest: str | Path, roi_manifest: str | Path, dst_dir: str | Path
) -> Path:
    """MRI paths from the ROI cache, PET paths from the whole-brain cache."""
    full = {r["subject"]: r for r in _read(full_manifest)}
    roi = {r["subject"]: r for r in _read(roi_manifest)}
    common = sorted(set(full) & set(roi))
    skipped = len((set(full) | set(roi)) - set(common))
    if skipped:
        log.warning("%d subjects are not in both caches and were skipped", skipped)
    rows = []
    for sid in common:
        if full[sid]["label"] != roi[sid]["label"]:
            raise ValueError(f"label mismatch for {sid}")
        rows.append(
            {
                "subject": sid,
                "label": full[sid]["label"],
                "mri": roi[sid]["mri"],
                "pet": full[sid]["pet"],
            }
        )
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    return _write(dst / "cache_manifest.csv", rows)
