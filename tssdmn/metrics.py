"""Evaluation metrics for dynamic multilayer network experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC between binary edge labels and predicted scores."""
    y = np.asarray(targets).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"AUC requires both positive and negative edges (got pos={positives}, neg={negatives})."
        )
    return float(roc_auc_score(y, s))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error across all entries."""
    diff = np.asarray(a).reshape(-1) - np.asarray(b).reshape(-1)
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))
