# adfusion: sMRI + FDG-PET fusion for Alzheimer's disease

[![CI](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/ci.yml)
[![Release](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/release.yml/badge.svg)](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-black.svg)

`adfusion` is a tested Python package that implements the full pipeline behind
the conference paper *Comparing Multimodal Fusion Strategies for Alzheimer's
Disease Detection Using Structural MRI and FDG-PET* (SIST 2026, accepted). It
takes paired T1-weighted MRI and FDG-PET scans, aligns them, trains seven
fusion models under one leakage-safe protocol, and reports results with
confidence intervals and corrected significance tests.

The package was written so that the experiments of the paper can be rerun
from a single configuration file, and so that the protocol itself (nested
cross-validation, separate estimators, paired tests) can be reused on other
cohorts.

> **No patient data are included.** ADNI data may not be redistributed. The
> repository contains code only; all tests and the CI pipelines run on
> synthetic volumes. See [docs/data_policy.md](docs/data_policy.md).

## What the pipeline does

```
ADNI search exports ──► cohort ──► preprocess ──► qc ──► (roi-cache / mixed) ──► train ──► figures
   MRI.csv, PET.csv     subjects    PET→T1 affine         optional input           nested CV   ROC, folds,
                        with both   registration,         variants                 7 models    curves,
                        modalities  shared T1 crop,                                bootstrap   confusion,
                                    96³ resampling,                                + Holm      fusion weights
                                    z-score → [0,1],
                                    Dice QC
```

| Stage | Module | What it guarantees |
|---|---|---|
| Cohort | `adfusion.cohort` | One most-processed image per subject and modality; only subjects with both modalities |
| Preprocessing | `adfusion.preprocessing` | MRI and PET on one voxel grid; one T1-derived crop for both; registration Dice stored per subject (survives resumed runs) |
| Variants | `adfusion.variants` | Central ROI crop and modality-specific inputs (ROI MRI + whole-brain PET) |
| Data | `adfusion.data` | Subject-level, stratified nested splits; a check that no subject appears in two partitions |
| Models | `adfusion.models` | MRI-only, PET-only, early, late, intermediate, hybrid cross-attention, hybrid without attention; optional capacity matching |
| Training | `adfusion.train` | Epoch and threshold chosen on inner validation only; each test fold predicted once |
| Metrics | `adfusion.metrics` | Fold-mean ± sd and pooled AUC with its own bootstrap CI, paired bootstrap tests, Holm correction |
| Figures | `adfusion.figures` | Greyscale publication figures |

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Octozavrus/multimodal-ad-fusion.git
cd multimodal-ad-fusion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

Preprocessing only (NumPy, SciPy, SimpleITK, pandas):

```bash
pip install -e .
```

Full pipeline with training and figures. Install PyTorch first so that you
can choose the CPU or CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[train,dev]"
```

## Quick start (synthetic data)

Preprocessing, a few seconds:

```bash
adfusion synth      --out demo/raw --n 6
adfusion preprocess --raw demo/raw --labels demo/raw/labels.csv --out demo/cache --shape 48
adfusion qc         demo/cache/cache_manifest.csv --threshold 0.5 --strict
```

The synthetic raw scans are brain-like phantoms in which the "PET" is
blurred, shifted and rotated relative to the "T1", so registration has real
work to do.

Training and figures, about one minute on a laptop CPU:

```bash
adfusion synth-cohort --out demo/cohort --n 30 --shape 32
adfusion train        --manifest demo/cohort/cache_manifest.csv --out demo/results --config configs/smoke.json
adfusion figures      demo/results
```

The synthetic cohort contains a planted class signal but is not anatomical.
The smoke protocol trains for three epochs only, so its AUC values say
nothing about model quality; it exists to show that every stage runs end to
end.

## Running on ADNI

1. Request access at <https://adni.loni.usc.edu> and download T1 MRI
   (MPR; GradWarp; B1 Correction; N3; Scaled) and FDG-PET
   (Coreg, Avg, Std Img and Vox Siz, Uniform Resolution) for AD and CN
   subjects. Keep the image-search CSV exports.
2. Select the cohort:
   `adfusion cohort mri_export.csv pet_export.csv --out work/cohort`
3. Preprocess and check registration:
   `adfusion preprocess --raw /path/to/ADNI --labels mri_export.csv pet_export.csv --out work/cache`
   `adfusion qc work/cache/cache_manifest.csv`
4. Train with the locked protocol used in the paper:
   `adfusion train --manifest work/cache/cache_manifest.csv --out work/results --config configs/protocol_sist2026.json`
5. Optional follow-up experiments:
   `adfusion roi-cache --manifest work/cache/cache_manifest.csv --out work/cache_roi`
   `adfusion mixed --full work/cache/cache_manifest.csv --roi work/cache_roi/cache_manifest.csv --out work/cache_mixed`
6. Figures: `adfusion figures work/results`

The full sequence is in [docs/reproducing_sist2026.md](docs/reproducing_sist2026.md).
Keep `work/` outside the repository or rely on `.gitignore`; the data guard
in CI rejects imaging files and real subject IDs.

## Command reference

| Command | Needs `[train]` | Purpose |
|---|---|---|
| `adfusion synth` | | Write synthetic raw scans and a label file |
| `adfusion preprocess` | | Co-register, crop, resample, normalise, record Dice. Options: `--shape`, `--template`, `--rigid-only`, `--fg-frac`, `--clip-sigma` |
| `adfusion qc` | | Registration Dice summary; `--strict` exits with 1 if a subject is below `--threshold` |
| `adfusion cohort` | | Intersect ADNI MRI and PET search exports |
| `adfusion synth-cohort` | | Write a synthetic preprocessed cohort with a planted AD signal |
| `adfusion roi-cache` | | Build the central-ROI variant of a cache |
| `adfusion mixed` | | Pair ROI MRI with whole-brain PET |
| `adfusion train` | yes | Nested cross-validation; flags override values from `--config` |
| `adfusion figures` | yes | All figures that the results directory allows |
| `adfusion info` | yes | Model list with naive and capacity-matched parameter counts |

Run `adfusion <command> --help` for every option.

### Output of `adfusion preprocess`

| File | Content |
|---|---|
| `<subject>_MRI.npy`, `<subject>_PET.npy` | float32 arrays in [0, 1], identical shape |
| `<subject>_qc.json` | registration Dice of that subject |
| `cache_manifest.csv` | subject, label, mri, pet, reg_dice |
| `preprocess_summary.json` | counts, label balance, Dice statistics, failures |

### Output of `adfusion train`

| File | Content |
|---|---|
| `results_per_fold.csv` | Per-fold accuracy, AUC, sensitivity, specificity at 0.5 and at the inner-validation threshold |
| `results_summary.csv` | Fold-mean ± sd and pooled out-of-fold AUC with 95% bootstrap CI |
| `paired_tests.csv` | Paired bootstrap ΔAUC, CI, raw and Holm-adjusted p |
| `oof_predictions.json` | Out-of-fold labels, probabilities, fold and subject IDs |
| `learning_curves.json` | Training loss and inner-validation AUC per epoch |
| `run_config.json` | Full configuration, parameter counts, library versions |
| `checkpoints/` | One state dict per model, repeat and fold |

The results reported in the paper cannot be reproduced from this repository
without ADNI access; see the paper for the cohort and the numbers.

## Repository layout

```
src/adfusion/        package source (one module per pipeline stage)
tests/               pytest suite, synthetic data only
configs/             locked protocol (SIST 2026) and smoke protocol
scripts/             repository data guard
docs/                architecture, data policy, reproduction notes
.github/workflows/   CI (lint, tests, smoke runs, build) and release
```

## Development

```bash
ruff check . && ruff format --check .   # style
python scripts/check_no_data.py         # data guard
pytest --cov=adfusion                   # tests and coverage
pytest -m "not slow"                    # fast subset
python -m build                         # sdist and wheel
```

`pre-commit install` runs ruff and the data guard on every commit. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

### Continuous integration and delivery

| Workflow | Trigger | Jobs |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, pull requests, manual | lint + format + data guard → tests on Python 3.10–3.13 with coverage gate (85%) → synthetic preprocessing and training runs and wheel build, all uploaded as artifacts |
| [`release.yml`](.github/workflows/release.yml) | tag `v*.*.*` | tag/version check → tests → build → GitHub Release with the wheel and sdist |

## Citation

If you use this code, please cite the conference paper (see
[CITATION.cff](CITATION.cff)).

## License

Code: MIT ([LICENSE](LICENSE)). ADNI data are governed by the ADNI Data Use
Agreement and are not covered by this license.

## Acknowledgement

Data used in the paper were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI
investigators contributed to the design and implementation of ADNI and/or
provided data but did not participate in the analysis or writing of this
work.
