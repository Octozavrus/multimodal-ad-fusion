"""Synthetic raw scans for tests and demonstrations.

``make_synthetic_raw`` writes a small ADNI-like folder tree of NIfTI files:

    <out>/000_S_0001/MPRAGE/t1.nii.gz
    <out>/000_S_0001/Coreg_FDG/pet.nii.gz
    ...
    <out>/labels.csv          (Subject, Group)

Each "T1" is a brain-like ellipsoid with a brighter cortical shell. The
matching "PET" is a blurred copy with different contrast that has been
shifted and slightly rotated, so co-registration has real work to do.
Site code ``000`` does not exist in ADNI, so these IDs cannot be confused
with real participants. The volumes are not anatomically realistic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, shift


def _phantom(shape, rng):
    g = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    radii = np.array([0.75, 0.8, 0.65]) * rng.uniform(0.92, 1.05, 3)
    r = np.sqrt(sum((x / a) ** 2 for x, a in zip(g, radii)))
    brain = (r <= 1.0).astype(np.float32)
    cortex = ((r > 0.8) & (r <= 1.0)).astype(np.float32)
    ventricles = (r < 0.25).astype(np.float32)
    return brain, cortex, ventricles


def synthetic_subject(
    shape: tuple[int, int, int] = (40, 40, 40),
    max_shift: float = 3.0,
    max_angle: float = 6.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a (T1, PET) pair of float32 arrays on the same grid, PET misaligned."""
    rng = rng or np.random.default_rng()
    brain, cortex, ventricles = _phantom(shape, rng)
    t1 = 60 * brain + 40 * cortex - 40 * ventricles
    t1 = np.clip(t1 + rng.normal(0, 3, shape) * brain, 0, None).astype(np.float32)

    pet = 5 * brain + 10 * cortex - 4 * ventricles
    pet = gaussian_filter(pet, sigma=1.5)
    pet = rotate(pet, rng.uniform(-max_angle, max_angle), axes=(1, 2), reshape=False, order=1)
    pet = shift(pet, rng.uniform(-max_shift, max_shift, 3), order=1)
    pet = np.clip(pet + rng.normal(0, 0.2, shape) * (pet > 0.5), 0, None).astype(np.float32)
    return t1, pet


def make_synthetic_raw(
    out_dir: str | Path,
    n_subjects: int = 4,
    shape: tuple[int, int, int] = (40, 40, 40),
    spacing: float = 2.0,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write synthetic NIfTI pairs and a label file; return (raw_dir, labels_csv)."""
    import SimpleITK as sitk

    if n_subjects < 1:
        raise ValueError("n_subjects must be positive")
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n_subjects + 1):
        sid = f"000_S_{i:04d}"
        t1, pet = synthetic_subject(shape, rng=rng)
        for arr, sub, name in ((t1, "MPRAGE", "t1.nii.gz"), (pet, "Coreg_FDG", "pet.nii.gz")):
            d = out / sid / sub
            d.mkdir(parents=True, exist_ok=True)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((spacing,) * 3)
            sitk.WriteImage(img, str(d / name))
        rows.append({"Subject": sid, "Group": "AD" if i % 2 else "CN"})
    labels = out / "labels.csv"
    with open(labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Subject", "Group"])
        w.writeheader()
        w.writerows(rows)
    return out, labels
