#!/usr/bin/env python3
"""Synthetic simulation demo for TSSDMN (Section 5, Setting 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tssdmn import (
    EMConfig,
    auc_score,
    compute_a3_obs,
    em,
    forecast_one_step_probs,
    initialize_model,
    mape_c1_frobenius,
    mse,
)


def simulate_tssdmn(
    n: int = 20,
    k: int = 2,
    m: int = 2,
    t_steps: int = 30,
    sigma_sq: float = 0.01,
    seed: int = 0,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate a synthetic dynamic multilayer network with T training + 1 test step."""
    I = (n, n, k)
    J = (m, m, k)
    sigma = float(np.sqrt(sigma_sq))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    true_model = initialize_model(I, J, sigma=sigma, seed=seed)
    total_steps = t_steps + 1

    z_seq = torch.empty(true_model.prod_j, total_steps + 1, dtype=torch.float64)
    x_seq = torch.empty(true_model.prod_i, total_steps, dtype=torch.float64)
    true_probs = np.empty((true_model.prod_i, total_steps), dtype=np.float64)

    z_seq[:, 0] = true_model.u0 + true_model.omega * torch.randn(
        true_model.prod_j, generator=gen, dtype=torch.float64
    )
    for t in range(1, total_steps + 1):
        eps = true_model.sigma * torch.randn(true_model.prod_j, generator=gen, dtype=torch.float64)
        z_seq[:, t] = true_model.A @ z_seq[:, t - 1] + eps
        logits = true_model.C @ z_seq[:, t] + true_model.b
        probs = torch.sigmoid(logits)
        draws = torch.rand(true_model.prod_i, generator=gen, dtype=torch.float64)
        x_seq[:, t - 1] = (draws < probs).to(torch.float64)
        true_probs[:, t - 1] = probs.numpy()

    x_train = x_seq[:, :t_steps]
    x_test = x_seq[:, t_steps]
    return true_model, x_train, x_test, z_seq, true_probs[:, :t_steps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TSSDMN synthetic simulation demo.")
    parser.add_argument("--nodes", type=int, default=20, help="Number of nodes n (default: 20)")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers K (default: 2)")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent factor dimension m (default: 2)")
    parser.add_argument("--steps", type=int, default=30, help="Training time horizon T (default: 30)")
    parser.add_argument("--sigma-sq", type=float, default=0.01, help="State innovation variance sigma^2 (default: 0.01)")
    parser.add_argument("--max-iter", type=int, default=120, help="Maximum EM iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--verbose", action="store_true", help="Print EM progress")
    args = parser.parse_args()

    I = (args.nodes, args.nodes, args.layers)
    J = (args.latent_dim, args.latent_dim, args.layers)

    true_model, x_train, x_test, _, true_train_probs = simulate_tssdmn(
        n=args.nodes,
        k=args.layers,
        m=args.latent_dim,
        t_steps=args.steps,
        sigma_sq=args.sigma_sq,
        seed=args.seed,
    )

    config = EMConfig(
        max_iter=args.max_iter,
        seed=args.seed + 100,
        device=args.device,
        verbose=args.verbose,
    )
    fitted_model, mut, _ = em(x_train, I, J, config=config)

    est_logits = (fitted_model.C @ mut[:, 1:] + fitted_model.b.unsqueeze(1)).detach().cpu().numpy()
    est_train_probs = 1.0 / (1.0 + np.exp(-np.clip(est_logits, -50.0, 50.0)))
    in_sample_rmse = mse(true_train_probs, est_train_probs)
    in_sample_auc = auc_score(x_train.numpy(), est_train_probs)

    pred_test_probs = forecast_one_step_probs(fitted_model, mut)
    test_auc = auc_score(x_test.numpy(), pred_test_probs)
    c1_rel_err, c1_aligned = mape_c1_frobenius(true_model.Cm[0], fitted_model.Cm[0])
    a3_obs_true = compute_a3_obs(true_model)
    a3_obs_hat = compute_a3_obs(fitted_model)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "simulation_results.npz",
        in_sample_rmse=in_sample_rmse,
        in_sample_auc=in_sample_auc,
        one_step_ahead_auc=test_auc,
        c1_true=true_model.Cm[0].detach().cpu().numpy(),
        c1_aligned=c1_aligned,
        c1_rel_error=c1_rel_err,
        a3_obs_true=a3_obs_true,
        a3_obs_hat=a3_obs_hat,
    )

    print("=== TSSDMN Simulation Demo (Figures 2-4) ===")
    print(f"Configuration : n={args.nodes}, K={args.layers}, m={args.latent_dim}, T={args.steps}, sigma^2={args.sigma_sq}")
    print(f"In-sample prob RMSE   : {in_sample_rmse:.4f}")
    print(f"In-sample edge AUC    : {in_sample_auc:.4f}")
    print(f"One-step-ahead AUC    : {test_auc:.4f}")
    print(f"Aligned C1 rel. error : {c1_rel_err:.4f}")
    print("True A_{3,obs}        :")
    print(np.array2string(a3_obs_true, precision=4, suppress_small=True))
    print("Estimated A_{3,obs}   :")
    print(np.array2string(a3_obs_hat, precision=4, suppress_small=True))
    print(f"Saved outputs to      : {out_dir / 'simulation_results.npz'}")


if __name__ == "__main__":
    main()
