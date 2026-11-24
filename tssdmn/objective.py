"""Surrogate variational lower bound and M-step objectives for TSSDMN."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tssdmn.model import Model
from tssdmn.tensor import otimes, prod_dims


def _stable_log1pexp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 50, x, F.softplus(x))


def observation_log_term(
    model: Model,
    X: torch.Tensor,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
) -> torch.Tensor:
    """Surrogate observation lower bound (Equation 13 in Section 4.1).

    Args:
        model: TSSDMN parameter set.
        X: Binary network observations of shape (n^2 K, T).
        mut: Variational means [mu_0, mu_1, ..., mu_T] of shape (m^2 K, T + 1).
        sigmat: Variational standard deviations [sigma_0, ..., sigma_T] of shape (T + 1,).
    """
    mu_obs = mut[:, 1:]
    sigma_obs = sigmat[1:]
    logits = model.C @ mu_obs + model.b.unsqueeze(1)
    row_energy = model.C.pow(2).sum(dim=1).unsqueeze(1) * (sigma_obs.pow(2).unsqueeze(0) / 2.0)
    eta = logits + row_energy
    return (X * logits - _stable_log1pexp(eta)).sum()


def transition_log_term(
    model: Model,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
) -> torch.Tensor:
    """Gaussian state prior and variational entropy terms in the surrogate ELBO."""
    t_horizon = mut.shape[1] - 1
    prod_j = prod_dims(model.J)

    omega_sq = model.omega.pow(2).clamp(min=1e-12)
    sigma_sq = model.sigma.pow(2).clamp(min=1e-12)

    init_term = -0.5 / omega_sq * ((mut[:, 0] - model.u0).pow(2).sum() + prod_j * sigmat[0].pow(2))
    init_logdet = -0.5 * prod_j * torch.log(omega_sq)

    residual = mut[:, 1:] - model.A @ mut[:, :-1]
    tr_aat = torch.trace(model.A @ model.A.T)
    var_dyn = prod_j * sigmat[1:].pow(2).sum() + tr_aat * sigmat[:-1].pow(2).sum()
    dyn_term = -0.5 / sigma_sq * (residual.pow(2).sum() + var_dyn)
    dyn_logdet = -0.5 * t_horizon * prod_j * torch.log(sigma_sq)

    entropy_term = prod_j * torch.log(sigmat.clamp(min=1e-12)).sum()
    return init_term + init_logdet + dyn_term + dyn_logdet + entropy_term


def calculate_elbo(
    model: Model,
    X: torch.Tensor,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the tractable surrogate ELBO."""
    return observation_log_term(model, X, mut, sigmat) + transition_log_term(model, mut, sigmat)


def a_mstep_objective(
    mut: torch.Tensor,
    sigmat: torch.Tensor,
    am_list: list[torch.Tensor],
) -> torch.Tensor:
    """A-dependent Gaussian state objective (Appendix D.2)."""
    A = otimes(am_list)
    residual = mut[:, 1:] - A @ mut[:, :-1]
    tr_aat = torch.trace(A @ A.T)
    return -0.5 * residual.pow(2).sum() - 0.5 * tr_aat * sigmat[:-1].pow(2).sum()
