# Data policy

The pipeline was developed on data from the Alzheimer's Disease Neuroimaging
Initiative (ADNI). The ADNI Data Use Agreement does not allow the data to be
redistributed, so this repository contains code only.

## What is never committed

| Item | Why | How it is enforced |
|---|---|---|
| NIfTI, DICOM, NumPy arrays (`.nii`, `.dcm`, `.npy`, ...) | patient images | `.gitignore`, data guard |
| Model checkpoints (`.pt`, `.pth`, `.ckpt`) | weights trained on ADNI images | `.gitignore`, data guard |
| ADNI image-search exports and cache manifests | contain subject IDs, age, sex, diagnosis | `.gitignore` (`*manifest*.csv`, `idaSearch*.csv`) |
| Real ADNI subject IDs (`NNN_S_NNNN`) anywhere in text | identifiers of study participants | data guard (regex) |
| Files larger than 5 MB | almost always data | data guard, pre-commit |

The data guard (`scripts/check_no_data.py`) runs in the first CI job, in the
release workflow and, optionally, as a pre-commit hook. A pull request that
adds any of the items above fails CI.

## Test data

All tests use synthetic volumes from `adfusion.synthetic`. Synthetic subject
IDs start with `SYN`, and tests that need an ADNI-shaped ID use site `000`
(for example `000_S_0001`), which does not exist in ADNI and is whitelisted by
the data guard.

## Results

Aggregate results (AUC values, confidence intervals) may be published, as in
the paper. Per-subject predictions (`oof_predictions.json`) contain subject
IDs and stay outside the repository.
