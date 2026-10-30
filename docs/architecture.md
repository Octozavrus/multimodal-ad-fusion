# Architecture

The package is a modular monolith organised as a pipes-and-filters pipeline.
Each stage reads the output of the previous one from disk and writes a new,
self-describing output, so stages can be rerun independently and their
contracts can be tested in isolation.

| Stage | Input | Output (contract) |
|---|---|---|
| `cohort` | ADNI MRI and PET search exports (CSV) | `cohort_both_modalities.csv`, `best_image_per_subject.csv` |
| `preprocess` | raw NIfTI or DICOM, label CSVs | `<subject>_MRI.npy`, `<subject>_PET.npy` (float32, [0,1], same shape), `<subject>_qc.json`, `cache_manifest.csv` (subject, label, mri, pet, reg_dice) |
| `qc` | a `cache_manifest.csv` | Dice summary; exit code 1 under `--strict` if a subject is below the threshold |
| `roi-cache`, `mixed` | a `cache_manifest.csv` | a new cache and `cache_manifest.csv` with the same columns |
| `train` | `cache_manifest.csv`, protocol JSON | results CSV/JSON files, checkpoints |
| `figures` | a results directory | PNG figures |

## Dependency direction

```
cli ──► cohort
    ──► preprocessing ◄── variants
    ──► train ──► data, models, metrics
    ──► figures
synthetic (used by cli and tests only)
```

Lower layers never import higher ones. `preprocessing`, `metrics` and
`synthetic` do not import PyTorch; SimpleITK is imported only inside the
functions that need it.

## Preprocessing chain (per subject)

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

## Evaluation protocol

```
for each repeat r:
    outer StratifiedKFold(k=5, seed=42+r)          -> train_outer | test
        inner StratifiedKFold(1/inner_frac) on train_outer -> train | val
        assert no subject in two partitions
        fit on train; keep the epoch with best val AUC; tune threshold on val
        predict test once
pooled AUC + bootstrap CI over all test predictions
fold-mean +/- sd over per-fold AUCs
paired bootstrap for every model pair on identical subjects, Holm-adjusted
```

The test fold is never used to choose an epoch, a threshold or a model.

## Design decisions (preprocessing)

- **One crop for both modalities.** Cropping each modality separately would
  destroy the voxel correspondence that registration created.
- **Per-modality normalisation.** MRI and PET intensities have different
  units and distributions.
- **Resumable and auditable.** Existing outputs are skipped; Dice is stored
  per subject so that resumed runs report complete QC; failures are listed
  instead of stopping the batch.
