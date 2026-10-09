# Design

The tool is one stage of a larger research pipeline (cohort selection →
**preprocessing** → model training → evaluation). Its job is to turn raw
downloads into a model-ready cache with a fixed contract, so that every
downstream model receives spatially aligned, identically shaped inputs.

## Contract

| | Content |
|---|---|
| Input | a folder of NIfTI files or DICOM series whose paths contain an ADNI-style subject ID and a series description; one or more CSV files with `Subject` and `Group` |
| Output | `<subject>_MRI.npy` and `<subject>_PET.npy` (float32, [0, 1], same shape), `<subject>_qc.json`, `cache_manifest.csv` (subject, label, mri, pet, reg_dice), `preprocess_summary.json` |

## Processing chain (per subject)

```
read T1, read PET (NIfTI or DICOM; 4D PET is averaged over frames)
  [optional] register T1 to a template (affine)
register PET to T1: centred rigid initialisation, then affine
  Mattes mutual information (32 bins, 10% random sampling, seed 42)
  regular-step gradient descent, 150 iterations
  3 resolution levels: shrink 4/2/1, smoothing 2/1/0 mm
Dice of the two foreground masks (> 5% of max)        -> QC
bounding box of the T1 foreground, applied to BOTH volumes
linear resampling to the target shape (default 96^3)
per-modality normalisation: foreground z-score, clip +/-5, rescale to [0,1]
```

## Module structure

| Module | Responsibility |
|---|---|
| `preprocessing` | Pure NumPy/SciPy helpers (bounding box, resizing, normalisation, Dice, ROI crop); file discovery and labels; SimpleITK I/O and registration; the per-subject chain; the resumable batch runner |
| `synthetic` | Phantom T1/PET pairs with known misalignment, written as an ADNI-like folder tree |
| `cli` | `preprocess`, `synth` |

SimpleITK is imported only inside the functions that need it, so the
numerical helpers can be tested and reused without it.

## Design decisions

- **One crop for both modalities.** Cropping each modality separately would
  destroy the voxel correspondence that registration created.
- **Per-modality normalisation.** MRI and PET intensities have different
  units and distributions.
- **Resumable and auditable.** Existing outputs are skipped; Dice is stored
  per subject so that resumed runs report complete QC; failures are listed
  instead of stopping the batch.
