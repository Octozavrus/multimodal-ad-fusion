# Data policy

The pipeline was developed on data from the Alzheimer's Disease Neuroimaging
Initiative (ADNI). The ADNI Data Use Agreement does not allow the data to be
redistributed, so this repository contains code only.

## What is never committed

| Item | Why | How it is enforced |
|---|---|---|
| NIfTI, DICOM, NumPy arrays (`.nii`, `.dcm`, `.npy`, ...) | patient images | `.gitignore`, data guard |
| ADNI image-search exports and cache manifests | contain subject IDs, age, sex, diagnosis | `.gitignore` (`*manifest*.csv`, `idaSearch*.csv`) |
| Real ADNI subject IDs (`NNN_S_NNNN`) anywhere in text | identifiers of study participants | data guard (regex) |
| Files larger than 5 MB | almost always data | data guard, pre-commit |

The data guard (`scripts/check_no_data.py`) runs in the first CI job, in the
release workflow and, optionally, as a pre-commit hook. A pull request that
adds any of the items above fails CI.

## Test data

All tests use synthetic scans from `adfusion.synthetic`. Their subject IDs
use site `000` (for example `000_S_0001`), which does not exist in ADNI and
is whitelisted by the data guard.

## Outputs

Aggregate statistics (for example the mean registration Dice) may be
published, as in the paper. The cache, the manifest and the per-subject QC
files contain subject IDs and stay outside the repository.
