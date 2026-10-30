"""Classification metrics and statistical comparison of models.

Two estimators are kept separate on purpose:

* fold-level: mean and standard deviation of the per-fold metric;
* pooled: one AUC over all out-of-fold predictions, with its own
  percentile bootstrap confidence interval.

Models are compared with a paired bootstrap on the same subjects, and the
family of pairwise p-values is adjusted with the Holm step-down procedure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def metrics_from_preds(y_true, y_prob, thresh: float = 0.5) -> dict:
    """Accuracy, AUC, sensitivity, specificity and confusion counts."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    # AUC is undefined when only one class is present (possible in tiny folds)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float((y_pred == y_true).mean()),
        "auc": float(auc),
        "sens": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "spec": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def best_threshold(y_true, y_prob, grid: Sequence[float] | None = None) -> float:
    """Threshold that maximises sensitivity + specificity (Youden) on a grid."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else np.asarray(grid)
    scores = []
    for t in grid:
        m = metrics_from_preds(y_true, y_prob, t)
        s = m["sens"] + m["spec"]
        scores.append(-np.inf if np.isnan(s) else s)
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y_true, y_prob, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval of the AUC."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    idx, aucs = np.arange(len(y_true)), []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[s], y_prob[s]))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi))


def paired_bootstrap(y_true, pa, pb, n: int = 2000, seed: int = 0):
    """Two-sided paired bootstrap test of AUC(a) - AUC(b) on the same subjects.

    Returns ``(p_value, (diff_ci_low, diff_ci_high))``.
    """
    rng = np.random.default_rng(seed)
    y_true, pa, pb = np.asarray(y_true), np.asarray(pa), np.asarray(pb)
    if not (len(y_true) == len(pa) == len(pb)):
        raise ValueError("inputs must have equal length")
    idx, diffs = np.arange(len(y_true)), []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[s], pa[s]) - roc_auc_score(y_true[s], pb[s]))
    if not diffs:
        return float("nan"), (float("nan"), float("nan"))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(p, 1.0)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(np.argsort(p)):
        prev = max(prev, min(1.0, (m - rank) * p[i]))
        adj[i] = prev
    return adj
