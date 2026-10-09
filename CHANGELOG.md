# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-09

First release of the preprocessing module used for the SIST 2026 paper.

### Added
- `adfusion preprocess`: PET-to-T1 co-registration (rigid, then affine),
  one T1-derived bounding box for both scans, resampling to a common shape,
  per-modality normalisation, registration Dice, `cache_manifest.csv` and
  `preprocess_summary.json`. Runs are resumable.
- `adfusion synth`: synthetic raw NIfTI scans (T1 and a blurred, shifted and
  rotated PET) with a label file, so the whole module, including
  registration, can run without patient data.
- pytest suite, GitHub Actions CI (lint, data guard, tests on Python
  3.10-3.13 with a coverage gate, smoke run, wheel build), tag-triggered
  release workflow, Dependabot and pre-commit hooks.
- Repository data guard (`scripts/check_no_data.py`).

### Changed (relative to the original research script)
- Hard-coded Google Colab and Drive paths replaced by command-line options
  and a dataclass configuration.
- Registration Dice is stored per subject in `<subject>_qc.json`, so it is
  no longer lost for subjects skipped when a run is resumed.
- Files are sorted before a series is chosen, so the choice is
  deterministic when a subject has several downloads.
- Library code logs through `logging` instead of `print`.
