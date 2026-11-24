"""Align estimated node-factor matrix C1 up to column permutation and scale."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch


def _as_numpy(cm: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(cm, torch.Tensor):
        return cm.detach().cpu().numpy()
    return np.asarray(cm)


def adjust_c1_paper(
    c_true: torch.Tensor | np.ndarray,
    c_hat: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, float, tuple[int, ...], float]:
    """Align estimated C1 to true C1 over column permutations and positive scale."""
    true = _as_numpy(c_true).astype(np.float64, copy=False)
    hat = _as_numpy(c_hat).astype(np.float64, copy=False)
    if true.shape != hat.shape:
        raise ValueError(f"Shape mismatch: true {true.shape} vs hat {hat.shape}")

    _, n_cols = true.shape
    true_norm_sq = float(np.sum(true * true))
    if true_norm_sq <= 0.0:
        raise ValueError("True C1 has zero Frobenius norm.")

    hat_norm_sq = float(np.sum(hat * hat))
    if hat_norm_sq <= 0.0:
        return np.zeros_like(true), 0.0, tuple(range(n_cols)), 1.0

    best_error = np.inf
    best_aligned: np.ndarray | None = None
    best_scale = 1.0
    best_perm: tuple[int, ...] = tuple(range(n_cols))

    for perm in permutations(range(n_cols)):
        permuted = hat[:, perm]
        denom = float(np.sum(permuted * permuted))
        if denom <= 0.0:
            continue
        scale = max(float(np.sum(true * permuted) / denom), 0.0)
        aligned = scale * permuted
        error = float(np.sum((true - aligned) ** 2))
        if error < best_error:
            best_error = error
            best_aligned = aligned
            best_scale = scale
            best_perm = perm

    if best_aligned is None:
        raise ValueError("Failed to align C1.")

    rel_error = float(np.sqrt(best_error) / np.sqrt(true_norm_sq))
    return best_aligned, best_scale, best_perm, rel_error


def mape_c1_frobenius(
    c_true: torch.Tensor | np.ndarray,
    c_hat: torch.Tensor | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return relative Frobenius error and aligned C1."""
    aligned, _, _, rel_error = adjust_c1_paper(c_true, c_hat)
    return rel_error, aligned
