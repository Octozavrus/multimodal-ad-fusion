"""Synthetic data for tests and demonstrations (no patient data needed).

Two generators are provided.

``make_synthetic_raw`` writes a small ADNI-like folder tree of raw NIfTI files
for the preprocessing stage:

    <out>/000_S_0001/MPRAGE/t1.nii.gz
    <out>/000_S_0001/Coreg_FDG/pet.nii.gz
    ...
    <out>/labels.csv          (Subject, Group)

Each "T1" is a brain-like ellipsoid with a brighter cortical shell. The
matching "PET" is a blurred copy with different contrast that has been
shifted and slightly rotated, so co-registration has real work to do.
Site code ``000`` does not exist in ADNI, so these IDs cannot be confused
with real participants. The volumes are not anatomically realistic.

``make_synthetic_cohort`` writes an already preprocessed cache (paired
``.npy`` volumes and ``cache_manifest.csv``) for the training stage, with a
planted, class-dependent signal: AD subjects have reduced "MRI" intensity in
two small medial blobs (a stand-in for medial temporal atrophy) and reduced
"PET" uptake in a posterior blob (a stand-in for posterior cingulate and
parietal hypometabolism). Its subject IDs use the prefix ``SYN``.
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


# ---------------------------------------------------------------- preprocessed cohort


def _grid(shape):
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    return zz, yy, xx


def _blob(zz, yy, xx, c, r):
    return np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * r**2)))


def synthetic_pair(
    label: int,
    shape: tuple[int, int, int] = (32, 32, 32),
    effect: float = 0.5,
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one (MRI, PET) pair in [0, 1] for class ``label`` (0 = CN, 1 = AD)."""
    rng = rng or np.random.default_rng()
    zz, yy, xx = _grid(shape)
    scale = rng.uniform(0.9, 1.05, size=3)
    brain = (
        (zz / (0.8 * scale[0])) ** 2 + (yy / (0.85 * scale[1])) ** 2 + (xx / (0.7 * scale[2])) ** 2
    ) <= 1.0
    base = brain.astype(np.float32) * 0.8

    mri = base.copy()
    pet = base.copy() * 0.7 + 0.3 * _blob(zz, yy, xx, (0.0, 0.0, 0.0), 0.5) * brain
    if label == 1:
        sev = effect * rng.uniform(0.6, 1.0)
        for side in (-0.25, 0.25):
            mri -= sev * _blob(zz, yy, xx, (0.0, -0.1, side), 0.12) * brain
        pet -= sev * _blob(zz, yy, xx, (0.2, 0.4, 0.0), 0.2) * brain
    mri += rng.normal(0, noise, shape) * brain
    pet += rng.normal(0, noise, shape) * brain
    return (np.clip(mri, 0, 1).astype(np.float32), np.clip(pet, 0, 1).astype(np.float32))


def make_synthetic_cohort(
    out_dir: str | Path,
    n_subjects: int = 40,
    shape: tuple[int, int, int] = (32, 32, 32),
    effect: float = 0.5,
    seed: int = 0,
) -> Path:
    """Write a balanced synthetic cohort and its ``cache_manifest.csv``."""
    if n_subjects < 4:
        raise ValueError("need at least 4 subjects")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        y = i % 2
        mri, pet = synthetic_pair(y, shape, effect, rng=rng)
        sid = f"SYN{i:04d}"
        pm, pp = out / f"{sid}_MRI.npy", out / f"{sid}_PET.npy"
        np.save(pm, mri)
        np.save(pp, pet)
        rows.append({"subject": sid, "label": "AD" if y else "CN", "mri": pm, "pet": pp})
    manifest = out / "cache_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "label", "mri", "pet"])
        w.writeheader()
        w.writerows(rows)
    return manifest
