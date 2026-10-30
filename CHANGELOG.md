# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Dockerfile` with `test` and `runtime` targets (CPU-only PyTorch, non-root
  user, OCI labels, protocol configs in `/app/configs`) and a `.dockerignore`
  that keeps data out of the build context.
- `Jenkinsfile`: data guard, lint and tests in the test image (JUnit and
  coverage reports), runtime image build and a smoke run of preprocessing and
  training inside it; on `main` an optional Docker Hub push, deployment to a
  staging container and an optional Kubernetes Job.
- `ci/`: local Jenkins server (docker compose, plugins, Configuration as Code
  with the multibranch job), `deploy/compose.staging.yml`, `k8s/job.yaml`,
  `docs/ci_jenkins.md`.
- GitHub Actions job that builds and runs the Docker image.

## [1.1.0] - 2026-10-30

The repository now contains the full research pipeline of the SIST 2026
paper, not only the preprocessing stage.

### Added
- `adfusion cohort`: intersection of ADNI MRI and PET search exports, one
  most-processed image per subject and modality.
- `adfusion roi-cache` and `adfusion mixed`: central-ROI cache and the
  modality-specific input (ROI MRI with whole-brain PET).
- `adfusion train`: paired dataset, subject-level stratified nested
  cross-validation with a leakage check, 3D CNN encoder and seven fusion
  configurations (MRI only, PET only, early, late, intermediate, hybrid with
  and without cross-attention), optional capacity matching, inner-validation
  epoch and threshold selection.
- Metrics: fold-mean and pooled AUC with bootstrap intervals, paired
  bootstrap tests and Holm correction; `adfusion figures` (greyscale) and
  `adfusion info`.
- `adfusion synth-cohort`: synthetic preprocessed cohort with a planted AD
  signal for tests and the training smoke run.
- Protocol files `configs/protocol_sist2026.json` and `configs/smoke.json`;
  `docs/reproducing_sist2026.md`.

### Changed
- Training dependencies (PyTorch, scikit-learn, matplotlib) are an optional
  extra: `pip install "adfusion[train]"`. Preprocessing still installs
  without them; pandas is now a core dependency (cohort selection).
- CI installs CPU-only PyTorch, runs the full test suite and adds a
  synthetic training run and figures to the smoke job.

## [1.0.0] - 2026-10-09

First release of the preprocessing module used for the SIST 2026 paper.

### Added
- `adfusion preprocess`: PET-to-T1 co-registration (rigid, then affine),
  one T1-derived bounding box for both scans, resampling to a common shape,
  per-modality normalisation, registration Dice, `cache_manifest.csv` and
  `preprocess_summary.json`. Runs are resumable.
- `adfusion qc`: registration Dice statistics and the subjects below a
  threshold, with a `--strict` exit code for use as a quality gate.
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
