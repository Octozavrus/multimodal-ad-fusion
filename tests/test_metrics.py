import math

import numpy as np
import pytest

from adfusion.metrics import (
    best_threshold,
    bootstrap_ci,
    holm,
    metrics_from_preds,
    paired_bootstrap,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P_PERFECT = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])


def test_perfect_classifier():
    m = metrics_from_preds(Y, P_PERFECT)
    assert m["auc"] == 1.0 and m["acc"] == 1.0 and m["sens"] == 1.0 and m["spec"] == 1.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (4, 4, 0, 0)


def test_confusion_counts_at_threshold():
    m = metrics_from_preds(Y, P_PERFECT, thresh=0.35)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (4, 3, 1, 0)
    assert m["spec"] == pytest.approx(0.75)


def test_single_class_gives_nan_auc():
    m = metrics_from_preds([1, 1, 1], [0.2, 0.6, 0.9])
    assert math.isnan(m["auc"]) and math.isnan(m["spec"])


def test_best_threshold_separates_shifted_scores():
    p = P_PERFECT * 0.5  # all scores below 0.5; the default threshold fails
    assert metrics_from_preds(Y, p)["sens"] == 0.0
    t = best_threshold(Y, p)
    m = metrics_from_preds(Y, p, t)
    assert m["sens"] == 1.0 and m["spec"] == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 200), 0, 1)
    from sklearn.metrics import roc_auc_score

    lo, hi = bootstrap_ci(y, p, n=500)
    assert lo < roc_auc_score(y, p) < hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 60), rng.random(60)
    assert bootstrap_ci(y, p, n=200, seed=3) == bootstrap_ci(y, p, n=200, seed=3)


def test_paired_bootstrap_identical_models():
    p, (lo, hi) = paired_bootstrap(Y, P_PERFECT, P_PERFECT, n=200)
    assert p == 1.0 and lo == 0.0 and hi == 0.0


def test_paired_bootstrap_detects_large_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 300), 0, 1)
    bad = rng.random(300)
    p, (lo, hi) = paired_bootstrap(y, good, bad, n=500)
    assert p < 0.01 and lo > 0


def test_paired_bootstrap_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap([0, 1], [0.1, 0.2], [0.3])


def test_holm_known_values():
    # Holm (1979) worked example: p = 0.01, 0.04, 0.03, 0.005 with m = 4
    adj = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_holm_is_monotone_and_capped():
    p = np.array([0.5, 0.001, 0.2, 0.9, 0.04])
    adj = holm(p)
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= 0).all()
    assert (adj <= 1).all() and (adj >= p).all()
