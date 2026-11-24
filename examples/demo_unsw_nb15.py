#!/usr/bin/env python3
"""Empirical case study demo on the UNSW-NB15 dynamic multilayer network (Section 6.2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tssdmn import EMConfig, auc_score, compute_a3_obs, em, forecast_one_step_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TSSDMN on the UNSW-NB15 dynamic multilayer network.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "unsw_nb15_40x40x4x61.npz",
        help="Path to the preprocessed UNSW-NB15 .npz archive",
    )
    parser.add_argument("--latent-dim", type=int, default=3, help="Latent factor dimension m (default: 3)")
    parser.add_argument("--max-iter", type=int, default=120, help="Maximum EM iterations (default: 120)")
    parser.add_argument("--seed", type=int, default=4, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--verbose", action="store_true", help="Print EM iteration progress")
    args = parser.parse_args()

    archive = np.load(args.data, allow_pickle=False)
    Y = archive["Y"].astype(np.float64)
    layers = [str(x) for x in archive["layers"]]
    n_nodes = int(archive["nodes"])
    n_layers = int(archive["n_layers"])
    n_train = int(archive["n_train"])

    x_train = Y[:, :n_train]
    x_test = Y[:, n_train]

    I = (n_nodes, n_nodes, n_layers)
    J = (args.latent_dim, args.latent_dim, n_layers)

    config = EMConfig(
        max_iter=args.max_iter,
        min_iter=10,
        elbo_tol=1.0,
        e_optimizer="adam",
        e_steps=40,
        e_lr=0.05,
        m_steps=80,
        m_lr=0.01,
        a_steps=80,
        a_lr=0.01,
        c1_l1_lambda=1.0,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )

    fitted_model, mut, _ = em(x_train, I, J, config=config)
    pred_probs = forecast_one_step_probs(fitted_model, mut)

    print("=== UNSW-NB15 Case Study (n=40, K=4, T=60 train + 1 test) ===")
    block = n_nodes * n_nodes
    layer_aucs = {}
    edge_counts_over_time = np.zeros((n_layers, n_train), dtype=np.int64)
    for k, layer_name in enumerate(layers):
        sl = slice(k * block, (k + 1) * block)
        edge_counts_over_time[k] = x_train[sl, :].sum(axis=0).astype(np.int64)
        layer_auc = auc_score(x_test[sl], pred_probs[sl])
        layer_aucs[layer_name] = layer_auc
        print(f"Layer {k + 1} ({layer_name:16s}) one-step-ahead AUC: {layer_auc:.4f}")

    c1_hat = fitted_model.Cm[0].detach().cpu().numpy()
    c2_hat = fitted_model.Cm[2].detach().cpu().numpy()
    a3_obs = compute_a3_obs(fitted_model)

    m = args.latent_dim
    z_traj = mut[:, 1:].detach().cpu().numpy().reshape((m * m, n_layers, n_train), order="F")
    obs_layer_cores = np.einsum("lk,mkt->mlt", c2_hat, z_traj)
    fuzzers_interactions = obs_layer_cores[:, 1, :]

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "unsw_nb15_results.npz",
        layers=np.array(layers),
        edge_counts_over_time=edge_counts_over_time,
        layer_auc=np.array([layer_aucs[name] for name in layers]),
        c1_hat=c1_hat,
        a3_obs=a3_obs,
        fuzzers_factor_interactions=fuzzers_interactions,
    )

    print("\nEstimated observed-layer propagation operator A_{3,obs} = C_2 A_3 C_2^{-1}:")
    print(np.array2string(a3_obs, precision=4, suppress_small=True))
    print(f"Saved outputs (Table 2 & Figures 8-10 quantities) to: {out_dir / 'unsw_nb15_results.npz'}")


if __name__ == "__main__":
    main()
