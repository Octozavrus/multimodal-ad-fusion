"""Preprocessing of paired T1-weighted MRI and FDG-PET volumes.

For every subject the PET volume is co-registered to the subject's own T1
(rigid initialisation followed by an affine stage, Mattes mutual information),
so that both modalities share one voxel grid. A single brain bounding box is
computed on the T1 and applied to both modalities, the crops are resampled to
a common shape, and each modality is intensity-normalised separately
(z-score over the foreground, clipping, rescaling to [0, 1]).

Registration quality is recorded as the Dice overlap of the two foreground
masks and stored next to the cached arrays, so it survives resumed runs.

The numerical helpers (bounding box, resizing, normalisation, Dice, ROI crop)
depend only on NumPy and SciPy. SimpleITK is imported lazily by the I/O and
registration functions.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

log = logging.getLogger(__name__)

SUBJECT_RE = re.compile(r"\d{3}_S_\d{4}")
PET_KEYWORDS = ("coreg", "fdg", "uniform resolution", "std img", "averaged")
MRI_KEYWORDS = ("mpr", "mprage", "gradwarp", "mt1")
MANIFEST_FIELDS = ["subject", "label", "mri", "pet", "reg_dice"]


@dataclass
class PreprocessConfig:
    """Settings for one preprocessing run."""

    download_dir: Path
    output_dir: Path
    label_files: tuple[Path, ...] = ()
    template: Path | None = None
    target_shape: tuple[int, int, int] = (96, 96, 96)
    fg_frac: float = 0.05
    clip_sigma: float = 5.0
    affine: bool = True
    seed: int = 42


# --------------------------------------------------------------------------
# Numerical helpers (NumPy / SciPy only)
# --------------------------------------------------------------------------
def brain_bbox(a: np.ndarray, frac: float = 0.05) -> tuple[slice, ...]:
    """Bounding box of voxels brighter than ``frac`` of the maximum."""
    mask = a > a.max() * frac
    if not mask.any():
        return tuple(slice(0, s) for s in a.shape)
    idx = np.where(mask)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def resize_to(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Linear resampling of ``a`` to exactly ``shape`` (crop or zero-pad rounding)."""
    if a.ndim != len(shape):
        raise ValueError(f"array has {a.ndim} dims but target shape has {len(shape)}")
    out = zoom(a, [s / d for s, d in zip(shape, a.shape)], order=1)
    out = out[tuple(slice(0, s) for s in shape)]
    pad = [(0, s - o) for s, o in zip(shape, out.shape)]
    if any(p[1] for p in pad):
        out = np.pad(out, pad)
    return out.astype(np.float32)


def normalize(a: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """Foreground z-score, clip to +/- ``clip_sigma``, rescale to [0, 1]."""
    fg = a[a > 0]
    if fg.size == 0:
        return a.astype(np.float32)
    z = (a - fg.mean()) / (fg.std() + 1e-6)
    z = np.clip(z, -clip_sigma, clip_sigma)
    return ((z - z.min()) / (z.max() - z.min() + 1e-6)).astype(np.float32)


def dice(a: np.ndarray, b: np.ndarray, frac: float = 0.05) -> float:
    """Dice overlap of the thresholded foreground masks of two volumes."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ma, mb = a > a.max() * frac, b > b.max() * frac
    s = ma.sum() + mb.sum()
    return float(2 * (ma & mb).sum() / s) if s else 0.0


def roi_crop(
    vol: np.ndarray, frac: float = 0.55, z_lo: float = 0.25, z_hi: float = 0.75
) -> np.ndarray:
    """Central medial-temporal window of a bounding-box-cropped volume.

    Keeps the central ``frac`` of the first two axes and the band
    ``[z_lo, z_hi]`` of the third axis.
    """
    if not (0 < frac <= 1 and 0 <= z_lo < z_hi <= 1):
        raise ValueError("invalid ROI fractions")
    d0, d1, d2 = vol.shape

    def span(n: int, f: float) -> tuple[int, int]:
        c, h = n / 2.0, n * f / 2.0
        return int(round(c - h)), int(round(c + h))

    a0, b0 = span(d0, frac)
    a1, b1 = span(d1, frac)
    a2, b2 = int(round(d2 * z_lo)), int(round(d2 * z_hi))
    return vol[a0:b0, a1:b1, a2:b2]


# --------------------------------------------------------------------------
# File discovery and labels
# --------------------------------------------------------------------------
def classify_modality(path: str) -> str | None:
    """Return ``"PET"``, ``"MRI"`` or ``None`` from an ADNI-style path.

    PET keywords are checked first because processed PET series descriptions
    can also contain MRI-like tokens.
    """
    p = str(path).lower()
    if any(k in p for k in PET_KEYWORDS):
        return "PET"
    if any(k in p for k in MRI_KEYWORDS):
        return "MRI"
    if "pet" in p:
        return "PET"
    if "mri" in p:
        return "MRI"
    return None


def build_file_index(root: str | Path) -> dict[str, dict[str, str]]:
    """Map subject ID -> {"MRI": path, "PET": path} for NIfTI files and DICOM folders."""
    index: dict[str, dict[str, str]] = {}
    dicom_dirs: set[str] = set()
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in files):
            dicom_dirs.add(dirpath)
        for fn in sorted(files):
            if fn.lower().endswith((".nii", ".nii.gz")):
                full = os.path.join(dirpath, fn)
                m, mod = SUBJECT_RE.search(full), classify_modality(full)
                if m and mod:
                    index.setdefault(m.group(0), {}).setdefault(mod, full)
    for d in sorted(dicom_dirs):
        m, mod = SUBJECT_RE.search(d), classify_modality(d)
        if m and mod:
            index.setdefault(m.group(0), {}).setdefault(mod, d)
    log.info("indexed %d subjects under %s", len(index), root)
    return index


def read_labels(files: tuple[Path, ...] | list[Path]) -> dict[str, str]:
    """Read subject -> diagnostic group from ADNI search-export CSV files."""
    labels: dict[str, str] = {}
    for mf in files:
        if not os.path.exists(mf):
            log.warning("label file not found: %s", mf)
            continue
        with open(mf, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                sid = r.get("Subject") or r.get("Subject ID")
                g = r.get("Group") or r.get("Research Group")
                if sid and g and sid not in labels:
                    labels[sid.strip()] = g.strip().upper()
    return labels


# --------------------------------------------------------------------------
# SimpleITK I/O and registration
# --------------------------------------------------------------------------
def _sitk():
    import SimpleITK as sitk

    return sitk


def read_image(path: str | Path):
    """Read a NIfTI file or a DICOM series folder as a float32 SimpleITK image."""
    sitk = _sitk()
    path = str(path)
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(path)
        files = (
            reader.GetGDCMSeriesFileNames(path, ids[0])
            if ids
            else reader.GetGDCMSeriesFileNames(path)
        )
        if not files:
            raise FileNotFoundError(f"no DICOM series in {path}")
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    img = sitk.Cast(img, sitk.sitkFloat32)
    if img.GetDimension() == 4:  # dynamic frames -> mean image
        img = sitk.MeanProjection(img, 3)[:, :, :, 0]
    return img


def register(moving, fixed, affine: bool = True, seed: int = 42, iterations: int = 150):
    """Rigid then optional affine registration of ``moving`` onto ``fixed``.

    Returns ``moving`` resampled onto the grid of ``fixed``.
    """
    sitk = _sitk()
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10, seed=seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(fixed, moving)
    if affine:
        reg.SetInitialTransform(sitk.AffineTransform(3), inPlace=False)
        reg.SetMovingInitialTransform(tx)
        tx2 = reg.Execute(fixed, moving)
        tx = sitk.CompositeTransform([tx, tx2])
    return sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def to_array(img) -> np.ndarray:
    """SimpleITK image -> float32 NumPy array (z, y, x) without NaNs."""
    a = _sitk().GetArrayFromImage(img).astype(np.float32)
    return np.nan_to_num(a, copy=False)


def process_subject(
    mri_path: str | Path,
    pet_path: str | Path,
    cfg: PreprocessConfig,
    template_img=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the full preprocessing chain for one subject.

    Returns the normalised MRI array, the normalised PET array and the
    registration Dice.
    """
    mri = read_image(mri_path)
    pet = read_image(pet_path)
    if template_img is not None:
        mri = register(mri, template_img, affine=True, seed=cfg.seed)
    pet = register(pet, mri, affine=cfg.affine, seed=cfg.seed)

    a_mri, a_pet = to_array(mri), to_array(pet)
    d = dice(a_mri, a_pet, cfg.fg_frac)

    bb = brain_bbox(a_mri, cfg.fg_frac)  # one box, from the T1, for both modalities
    a_mri, a_pet = a_mri[bb], a_pet[bb]
    a_mri = normalize(resize_to(a_mri, cfg.target_shape), cfg.clip_sigma)
    a_pet = normalize(resize_to(a_pet, cfg.target_shape), cfg.clip_sigma)
    return a_mri, a_pet, d


def run_preprocessing(cfg: PreprocessConfig) -> dict:
    """Preprocess every indexed subject and write ``cache_manifest.csv``.

    The run is resumable: subjects whose arrays already exist are skipped and
    their Dice value is read back from the per-subject QC file.
    """
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    template_img = read_image(cfg.template) if cfg.template else None
    index = build_file_index(cfg.download_dir)
    labels = read_labels(cfg.label_files)

    rows, failures = [], []
    for sid in sorted(index):
        rec = index[sid]
        if "MRI" not in rec or "PET" not in rec:
            failures.append((sid, f"missing modality: {sorted(rec)}"))
            continue
        if sid not in labels:
            failures.append((sid, "no label"))
            continue
        om, op, oq = (out / f"{sid}_{s}" for s in ("MRI.npy", "PET.npy", "qc.json"))
        try:
            if om.exists() and op.exists():
                d = json.loads(oq.read_text())["reg_dice"] if oq.exists() else float("nan")
            else:
                a_mri, a_pet, d = process_subject(rec["MRI"], rec["PET"], cfg, template_img)
                np.save(om, a_mri)
                np.save(op, a_pet)
                oq.write_text(json.dumps({"subject": sid, "reg_dice": d}))
            rows.append(
                {
                    "subject": sid,
                    "label": labels[sid],
                    "mri": str(om),
                    "pet": str(op),
                    "reg_dice": round(d, 4) if d == d else "",
                }
            )
        except Exception as exc:  # keep going; failures are reported at the end
            failures.append((sid, repr(exc)))
            log.error("failed %s: %s", sid, exc)

    manifest = out / "cache_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    dices = np.array([r["reg_dice"] for r in rows if r["reg_dice"] != ""], dtype=float)
    summary = {
        "manifest": str(manifest),
        "n_subjects": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "dice_mean": float(dices.mean()) if dices.size else None,
        "dice_sd": float(dices.std()) if dices.size else None,
        "dice_min": float(dices.min()) if dices.size else None,
        "n_dice_below_0_5": int((dices < 0.5).sum()) if dices.size else 0,
        "failures": failures,
    }
    (out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
