"""Shared fixtures. Every test uses synthetic data; no patient data are needed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adfusion.data import load_manifest
from adfusion.synthetic import make_synthetic_cohort


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)  # noqa: NPY002
    torch.manual_seed(0)
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def cohort_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cohort")
    make_synthetic_cohort(d, n_subjects=24, shape=(16, 16, 16), effect=0.8, seed=1)
    return d


@pytest.fixture(scope="session")
def manifest(cohort_dir):
    return cohort_dir / "cache_manifest.csv"


@pytest.fixture(scope="session")
def records(manifest):
    return load_manifest(manifest)
