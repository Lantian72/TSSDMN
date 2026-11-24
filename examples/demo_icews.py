#!/usr/bin/env python3
"""Empirical case study demo on the ICEWS dynamic multilayer network (Section 6.1)."""

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
    parser = argparse.ArgumentParser(description="Fit TSSDMN on the ICEWS dynamic multilayer network.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "icews_30x30x4x96.npz",
        help="Path to the preprocessed ICEWS .npz archive",
    )
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent factor dimension m (default: 4)")
    parser.add_argument("--max-iter", type=int, default=150, help="Maximum EM iterations (default: 150)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--verbose", action="store_true", help="Print EM iteration progress")
    args = parser.parse_args()

    archive = np.load(args.data, allow_pickle=False)
    Y = archive["Y"].astype(np.float64)
    countries = [str(x) for x in archive["countries"]]
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

    print("=== ICEWS Case Study (n=30, K=4, T=95 train + 1 test) ===")
    block = n_nodes * n_nodes
    layer_aucs = {}
    for k, layer_name in enumerate(layers):
        sl = slice(k * block, (k + 1) * block)
        layer_auc = auc_score(x_test[sl], pred_probs[sl])
        layer_aucs[layer_name] = layer_auc
        print(f"Layer {k + 1} ({layer_name:20s}) one-step-ahead AUC: {layer_auc:.4f}")

    c1_hat = fitted_model.Cm[0].detach().cpu().numpy()
    c2_hat = fitted_model.Cm[2].detach().cpu().numpy()
    a3_obs = compute_a3_obs(fitted_model)

    # Observed-layer factor interaction trajectories Z_{t} x_3 C_2 (Figure 6: Verbal Cooperation = layer 0)
    m = args.latent_dim
    z_traj = mut[:, 1:].detach().cpu().numpy().reshape((m * m, n_layers, n_train), order="F")
    obs_layer_cores = np.einsum("lk,mkt->mlt", c2_hat, z_traj)
    vcoo_interactions = obs_layer_cores[:, 0, :]

    # US -> Russia dynamic & static probabilities across the 4 layers (Figure 7)
    us_idx = countries.index("United States")
    ru_idx = countries.index("Russia")
    est_logits = (fitted_model.C @ mut[:, 1:] + fitted_model.b.unsqueeze(1)).detach().cpu().numpy()
    est_probs = 1.0 / (1.0 + np.exp(-np.clip(est_logits, -50.0, 50.0)))
    b_vec = fitted_model.b.detach().cpu().numpy()
    static_probs = 1.0 / (1.0 + np.exp(-np.clip(b_vec, -50.0, 50.0)))
    dyad_indices = [us_idx + ru_idx * n_nodes + k * block for k in range(n_layers)]
    us_russia_dynamic_probs = est_probs[dyad_indices, :]
    us_russia_static_probs = static_probs[dyad_indices]

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "icews_results.npz",
        countries=np.array(countries),
        layers=np.array(layers),
        layer_auc=np.array([layer_aucs[name] for name in layers]),
        c1_hat=c1_hat,
        a3_obs=a3_obs,
        vcoo_factor_interactions=vcoo_interactions,
        us_russia_dynamic_probs=us_russia_dynamic_probs,
        us_russia_static_probs=us_russia_static_probs,
    )

    print("\nEstimated observed-layer propagation operator A_{3,obs} = C_2 A_3 C_2^{-1}:")
    print(np.array2string(a3_obs, precision=4, suppress_small=True))
    print(f"Saved outputs (Table 1 & Figures 5-7 quantities) to: {out_dir / 'icews_results.npz'}")


if __name__ == "__main__":
    main()
