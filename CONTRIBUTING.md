# Contributing

Thank you for considering a contribution. This project uses trunk-based
development with short-lived branches: `main` is always releasable, every
change is made on a short branch and merged through a reviewed pull request
after CI passes, and releases are tags on `main`. `main` is protected: direct
pushes are blocked and the CI jobs must pass before a merge.

## Workflow

1. Open an issue describing the bug or the proposed change.
2. Create a branch from `main` named `fix/<short-name>` or `feat/<short-name>`.
3. Make small commits with messages in the imperative mood
   (`Add DICOM series check`, `Fix Dice on resumed runs`).
4. Run the checks locally (below) and open a pull request that references the
   issue. The CI workflow must pass before merging.
5. Pull requests are merged with squash merge to keep `main` linear.

## Local checks

```bash
pip install -e ".[dev]"
pre-commit install            # optional, runs ruff and the data guard on commit
ruff check . && ruff format --check .
python scripts/check_no_data.py
pytest --cov=adfusion
```

## Rules

- **Never commit data.** No NIfTI, DICOM, NumPy arrays, ADNI CSV exports,
  manifests or subject IDs. Use `adfusion synth` for test data.
  Subject IDs in tests must use site `000` (for example `000_S_0001`).
- Every change in behaviour needs a test. Bug fixes start with a failing test.
- Public functions have docstrings and type hints.
- Changes that alter the preprocessed output (registration settings, crop,
  resampling, normalisation) must be noted in `CHANGELOG.md`, because they
  change every result computed downstream.

## Releasing

1. Update `version` in `pyproject.toml` and `src/adfusion/__init__.py`.
2. Move the `Unreleased` section of `CHANGELOG.md` under the new version.
3. Commit, then tag and push: `git tag v1.1.0 && git push origin v1.1.0`.
   The release workflow builds and publishes the GitHub Release.
