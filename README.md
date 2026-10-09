# adfusion: sMRI and FDG-PET preprocessing for multimodal fusion

[![CI](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/ci.yml)
[![Release](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/release.yml/badge.svg)](https://github.com/Octozavrus/multimodal-ad-fusion/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-black.svg)

`adfusion` is a small, tested Python tool that prepares paired T1-weighted MRI
and FDG-PET scans for multimodal deep learning. It is the preprocessing stage
of the pipeline used in the conference paper *Comparing Multimodal Fusion
Strategies for Alzheimer's Disease Detection Using Structural MRI and
FDG-PET* (SIST 2026, accepted).

Fusion models that combine MRI and PET voxel by voxel need both scans on the
same grid. For each subject the tool therefore:

1. finds the T1 and PET scan (NIfTI file or DICOM series) in a download folder;
2. co-registers the PET to the subject's own T1 (rigid, then affine;
   Mattes mutual information, multi-resolution);
3. computes **one** brain bounding box on the T1 and applies it to both scans;
4. resamples both crops to a common shape (default 96³);
5. normalises each modality separately (foreground z-score, clipping at
   ±5 sd, rescaling to [0, 1]);
6. records the registration quality as the Dice overlap of the two foreground
   masks.

The result is a cache of paired `.npy` arrays and a `cache_manifest.csv`
(subject, label, mri, pet, reg_dice) that any model can read. Runs are
resumable: subjects already in the cache are skipped and their Dice values
are kept.

> **No patient data are included.** ADNI data may not be redistributed. The
> repository contains code only; tests and CI run on synthetic scans. See
> [docs/data_policy.md](docs/data_policy.md).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Octozavrus/multimodal-ad-fusion.git
cd multimodal-ad-fusion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start (synthetic data, a few seconds)

```bash
adfusion synth      --out demo/raw --n 6
adfusion preprocess --raw demo/raw --labels demo/raw/labels.csv --out demo/cache --shape 48
adfusion qc         demo/cache/cache_manifest.csv
```

The synthetic scans are brain-like phantoms in which the "PET" is blurred,
shifted and rotated relative to the "T1", so registration has real work to
do. They are not anatomical.

## Use on ADNI

1. Download T1 MRI (MPR; GradWarp; B1 Correction; N3; Scaled) and FDG-PET
   (Coreg, Avg, Std Img and Vox Siz, Uniform Resolution) for your subjects,
   and keep the image-search CSV exports (they contain `Subject` and `Group`).
2. Run
   `adfusion preprocess --raw /path/to/ADNI --labels mri_export.csv pet_export.csv --out work/cache`
3. Check registration quality:
   `adfusion qc work/cache/cache_manifest.csv --threshold 0.5`
   and inspect any subject listed below the threshold.

Keep `work/` outside the repository or rely on `.gitignore`; the data guard
in CI rejects imaging files and real subject IDs.

## Command reference

| Command | Purpose |
|---|---|
| `adfusion preprocess` | Co-register, crop, resample, normalise, record Dice. Options: `--shape`, `--template`, `--rigid-only`, `--fg-frac`, `--clip-sigma` |
| `adfusion qc` | Dice statistics and subjects below a threshold; `--strict` exits with 1 if any are below |
| `adfusion synth` | Write synthetic raw scans and a label file |

### Output of `adfusion preprocess`

| File | Content |
|---|---|
| `<subject>_MRI.npy`, `<subject>_PET.npy` | float32 arrays in [0, 1], identical shape |
| `<subject>_qc.json` | registration Dice of that subject |
| `cache_manifest.csv` | subject, label, mri, pet, reg_dice |
| `preprocess_summary.json` | counts, label balance, Dice statistics, failures |

### Python API

```python
from adfusion.preprocessing import PreprocessConfig, run_preprocessing

summary = run_preprocessing(
    PreprocessConfig(
        download_dir="raw",
        output_dir="cache",
        label_files=("labels.csv",),
        target_shape=(96, 96, 96),
    )
)
```

On ADNI (168 subjects) this preprocessing gave a mean registration Dice of
0.734 ± 0.072 (range 0.480 to 0.903), as reported in the paper.

## Repository layout

```
src/adfusion/        preprocessing.py, synthetic.py, cli.py
tests/               pytest suite, synthetic data only
scripts/             repository data guard
docs/                design notes and data policy
.github/workflows/   CI (lint, tests, smoke run, build) and release
```

## Development

```bash
ruff check . && ruff format --check .   # style
python scripts/check_no_data.py         # data guard
pytest --cov=adfusion                   # tests and coverage
python -m build                         # sdist and wheel
```

`pre-commit install` runs ruff and the data guard on every commit. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

### Continuous integration and delivery

| Workflow | Trigger | Jobs |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, pull requests, manual | lint + format + data guard → tests on Python 3.10–3.13 with coverage gate (85%) → synthetic end-to-end run with a registration quality gate, and wheel build; results and wheel uploaded as artifacts |
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
