"""Variational EM estimation for TSSDMN (Section 4 and Appendix D)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from tssdmn.model import Model, initialize_model, prepare_em_initialization
from tssdmn.objective import a_mstep_objective, calculate_elbo, observation_log_term
from tssdmn.tensor import (
    build_cm_list,
    flatten_am_params,
    flatten_cm_params,
    otimes,
    prod_dims,
    spectral_radius,
    unflatten_am_params,
    unflatten_cm_params,
)


@dataclass
class EMConfig:
    """Configuration for the variational EM algorithm."""

    max_iter: int = 200
    min_iter: int = 10
    elbo_tol: float = 1.0
    e_steps: int = 40
    e_lr: float = 0.05
    e_optimizer: Literal["adam", "lbfgs"] = "adam"
    m_steps: int = 100
    m_lr: float = 0.01
    update_a: bool = True
    a_steps: int = 100
    a_lr: float = 0.01
    max_factor_rho: float = 0.98
    b_ridge_lambda: float = 0.0
    c1_l1_lambda: float = 0.0
    seed: int = 0
    dtype: torch.dtype = torch.float64
    device: str | torch.device = "cpu"
    verbose: bool = False


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _softplus_pos(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-8


def _project_am_spectral_radius(
    am_list: list[torch.Tensor],
    target_rho: float,
) -> list[torch.Tensor]:
    projected: list[torch.Tensor] = []
    for block in am_list:
        rho = spectral_radius(block)
        if rho > target_rho:
            projected.append(block * (target_rho / rho))
        else:
            projected.append(block)
    return projected


def _closed_form_updates(model: Model, mut: torch.Tensor, sigmat: torch.Tensor) -> None:
    """Closed-form M-step updates for u0, omega^2, and sigma^2 (Section 4.2)."""
    t_horizon = mut.shape[1] - 1
    prod_j = prod_dims(model.J)

    model.u0 = mut[:, 0].detach().clone()
    model.omega = sigmat[0].detach().clamp(min=1e-6).clone()

    residual = mut[:, 1:] - model.A @ mut[:, :-1]
    tr_aat = torch.trace(model.A @ model.A.T)
    var_sum = prod_j * sigmat[1:].pow(2).sum() + tr_aat * sigmat[:-1].pow(2).sum()
    sigma_sq = (residual.pow(2).sum() + var_sum) / (t_horizon * prod_j)
    model.sigma = torch.sqrt(sigma_sq.clamp(min=1e-12)).detach()


def _optimize_latents(
    model: Model,
    X: torch.Tensor,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
    config: EMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    mut_param = mut.clone().detach().requires_grad_(True)
    sigmat_raw = torch.log(torch.expm1(torch.clamp(sigmat, min=1e-6))).detach().requires_grad_(True)

    if config.e_optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [mut_param, sigmat_raw],
            lr=1.0,
            max_iter=config.e_steps,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = -calculate_elbo(model, X, mut_param, _softplus_pos(sigmat_raw))
            loss.backward()
            return loss

        optimizer.step(closure)
        return mut_param.detach(), _softplus_pos(sigmat_raw).detach()

    optimizer = torch.optim.Adam([mut_param, sigmat_raw], lr=config.e_lr)
    best_mut = mut_param.detach().clone()
    best_sigmat = sigmat.detach().clone()
    best_elbo = float("-inf")

    for _ in range(config.e_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = -calculate_elbo(model, X, mut_param, _softplus_pos(sigmat_raw))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            cur_sigmat = _softplus_pos(sigmat_raw)
            cur_elbo = calculate_elbo(model, X, mut_param, cur_sigmat).item()
            if cur_elbo > best_elbo:
                best_elbo = cur_elbo
                best_mut = mut_param.detach().clone()
                best_sigmat = cur_sigmat.detach().clone()

    return best_mut, best_sigmat


def _optimize_transition(
    model: Model,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
    config: EMConfig,
) -> Model:
    updated = model.clone()
    am_params = flatten_am_params(updated.Am).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([am_params], lr=config.a_lr)

    with torch.no_grad():
        init_list = _project_am_spectral_radius(unflatten_am_params(am_params, updated.J), config.max_factor_rho)
        best_params = flatten_am_params(init_list)
        am_params.copy_(best_params)
        best_obj = a_mstep_objective(mut, sigmat, init_list).item()

    for _ in range(config.a_steps):
        optimizer.zero_grad(set_to_none=True)
        am_list = unflatten_am_params(am_params, updated.J)
        obj = a_mstep_objective(mut, sigmat, am_list)
        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            proj_list = _project_am_spectral_radius(unflatten_am_params(am_params, updated.J), config.max_factor_rho)
            am_params.copy_(flatten_am_params(proj_list))
            cur_obj = a_mstep_objective(mut, sigmat, proj_list).item()
            if cur_obj > best_obj:
                best_obj = cur_obj
                best_params = am_params.detach().clone()

    with torch.no_grad():
        updated.Am = _project_am_spectral_radius(unflatten_am_params(best_params, updated.J), config.max_factor_rho)
        updated.rebuild_matrices()
    return updated


def _optimize_emission(
    model: Model,
    X: torch.Tensor,
    mut: torch.Tensor,
    sigmat: torch.Tensor,
    config: EMConfig,
) -> Model:
    updated = model.clone()
    cm2_init = updated.Cm[2].clone() if len(updated.Cm) > 2 else None
    cm_params = flatten_cm_params(updated.Cm[0].clone(), cm2_init).detach().requires_grad_(True)
    b_param = updated.b.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([cm_params, b_param], lr=config.m_lr)
    n_c1 = updated.I[0] * updated.J[0]

    for _ in range(config.m_steps):
        optimizer.zero_grad(set_to_none=True)
        cm0_u, cm2_u = unflatten_cm_params(cm_params, updated.I, updated.J)
        cm0_u = torch.clamp(cm0_u, min=0.0)
        cm_list = build_cm_list(
            cm0_u,
            None,
            cm2_u,
            len(updated.I),
            share_first_two=(updated.I[0] == updated.I[1] and updated.J[0] == updated.J[1]),
        )
        temp = updated.clone()
        temp.Cm = cm_list
        temp.C = otimes(cm_list)
        temp.b = b_param
        obj = observation_log_term(temp, X, mut, sigmat)
        if config.b_ridge_lambda > 0.0:
            obj = obj - 0.5 * config.b_ridge_lambda * b_param.pow(2).sum()
        if config.c1_l1_lambda > 0.0:
            obj = obj - config.c1_l1_lambda * cm0_u.sum()
        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            cm_params[:n_c1].clamp_(min=0.0)

    with torch.no_grad():
        cm0_f, cm2_f = unflatten_cm_params(cm_params, updated.I, updated.J)
        cm0_f = torch.clamp(cm0_f, min=0.0)
        updated.Cm = build_cm_list(
            cm0_f,
            None,
            cm2_f,
            len(updated.I),
            share_first_two=(updated.I[0] == updated.I[1] and updated.J[0] == updated.J[1]),
        )
        updated.b = b_param.detach()
        updated.rebuild_matrices()
    return updated


def em(
    X: torch.Tensor | Sequence[Sequence[float]],
    I: Sequence[int],
    J: Sequence[int],
    config: EMConfig | None = None,
) -> tuple[Model, torch.Tensor, torch.Tensor]:
    """Fit TSSDMN via variational EM (Algorithm 2).

    Args:
        X: Binary observation matrix of shape (n^2 K, T).
        I: Observation dimensions (n, n, K).
        J: Latent core dimensions (m, m, K).
        config: Optional EM hyperparameters.

    Returns:
        (fitted_model, variational_means, variational_stds) where
        variational_means has shape (m^2 K, T + 1) for t = 0, ..., T.
    """
    config = config or EMConfig()
    device = _resolve_device(config.device)
    dtype = config.dtype

    X_t = torch.as_tensor(X, dtype=dtype, device=device)
    t_horizon = X_t.shape[1]
    prod_j = prod_dims(J)

    model = prepare_em_initialization(
        initialize_model(I, J, sigma=0.1, seed=config.seed, device=device, dtype=dtype)
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)
    mut = torch.randn(prod_j, t_horizon + 1, generator=gen, device="cpu", dtype=dtype).to(device)
    sigmat = torch.full((t_horizon + 1,), 1.0, device=device, dtype=dtype)

    last_elbo = float("-inf")

    for iteration in range(1, config.max_iter + 1):
        mut, sigmat = _optimize_latents(model, X_t, mut, sigmat, config)
        if config.update_a:
            model = _optimize_transition(model, mut, sigmat, config)
        model = _optimize_emission(model, X_t, mut, sigmat, config)
        _closed_form_updates(model, mut, sigmat)
        elbo_val = float(calculate_elbo(model, X_t, mut, sigmat).item())

        if config.verbose:
            delta = elbo_val - last_elbo
            print(
                f"iter={iteration:03d}  surrogate_elbo={elbo_val:.4f}  "
                f"delta={delta:.4f}  sigma={model.sigma.item():.4f}"
            )

        if (elbo_val - last_elbo) < config.elbo_tol and iteration >= config.min_iter:
            break
        last_elbo = elbo_val

    return model, mut.detach(), sigmat.detach()


def forecast_one_step_probs(model: Model, mut: torch.Tensor) -> np.ndarray:
    """Compute one-step-ahead edge probabilities at time T + 1."""
    z_next = model.A @ mut[:, -1]
    logits = (model.C @ z_next + model.b).detach().cpu().numpy()
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_a3_obs(model: Model) -> np.ndarray:
    """Compute the identified observed-layer propagation operator C2 A3 C2^{-1} (Remark 2)."""
    if len(model.Cm) < 3 or len(model.Am) < 3:
        raise ValueError("Observed-layer propagation operator requires a three-mode model.")
    c2 = model.Cm[2].detach().cpu().numpy()
    a3 = model.Am[2].detach().cpu().numpy()
    return c2 @ a3 @ np.linalg.inv(c2)
