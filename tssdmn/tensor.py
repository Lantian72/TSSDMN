"""Kronecker product and factor-packing utilities for three-mode tensors."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Sequence

import numpy as np
import torch


def prod_dims(dims: Sequence[int]) -> int:
    return int(reduce(mul, dims, 1))


def otimes(mats: Sequence[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """Recursive Kronecker product: mats[-1] (x) ... (x) mats[0]."""
    if len(mats) == 1:
        return mats[0]
    tail = otimes(mats[:-1])
    return torch.kron(mats[-1], tail) if isinstance(mats[-1], torch.Tensor) else np.kron(mats[-1], tail)


def flatten_cm_params(
    cm0: torch.Tensor,
    cm2: torch.Tensor | None,
) -> torch.Tensor:
    """Pack C1 and C2 factor matrices into a single parameter vector."""
    params: list[torch.Tensor] = [cm0.T.reshape(-1)]
    if cm2 is not None:
        params.append(cm2.T.reshape(-1))
    return torch.cat(params)


def unflatten_cm_params(
    params: torch.Tensor,
    i_dims: Sequence[int],
    j_dims: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    offset = 0
    cm0 = params[offset : offset + i_dims[0] * j_dims[0]].reshape(j_dims[0], i_dims[0]).T
    offset += i_dims[0] * j_dims[0]
    cm2 = None
    if len(i_dims) == 3:
        cm2 = params[offset : offset + i_dims[2] * j_dims[2]].reshape(j_dims[2], i_dims[2]).T
    return cm0, cm2


def flatten_am_params(am_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pack mode-specific transition matrices [A1, A2, A3] into a vector."""
    return torch.cat([am.T.reshape(-1) for am in am_list])


def unflatten_am_params(
    params: torch.Tensor,
    j_dims: Sequence[int],
) -> list[torch.Tensor]:
    """Unpack flattened vector into mode-specific transition matrices."""
    blocks: list[torch.Tensor] = []
    offset = 0
    for j_dim in j_dims:
        count = j_dim * j_dim
        block = params[offset : offset + count].reshape(j_dim, j_dim).T.contiguous()
        offset += count
        blocks.append(block)
    return blocks


def spectral_radius(matrix: torch.Tensor) -> float:
    """Largest absolute eigenvalue magnitude."""
    values = torch.linalg.eigvals(matrix.detach().cpu())
    return float(values.abs().max().item())


def build_cm_list(
    cm0: torch.Tensor,
    cm_mid: torch.Tensor | None,
    cm_last: torch.Tensor | None,
    n_modes: int,
    *,
    share_first_two: bool = True,
) -> list[torch.Tensor]:
    if n_modes == 1:
        return [cm0]
    if n_modes == 2:
        if share_first_two:
            return [cm0, cm0.clone()]
        if cm_mid is None:
            raise ValueError("Two-mode models with distinct factors require cm_mid")
        return [cm0, cm_mid]
    if cm_last is None:
        raise ValueError("Three-mode models require cm_last (C2)")
    second = cm0.clone() if share_first_two else cm_mid
    if second is None:
        raise ValueError("Three-mode models require cm_mid when factors are not shared")
    return [cm0, second, cm_last]
