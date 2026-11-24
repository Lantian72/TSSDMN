# TSSDMN: Tensor State Space-based Dynamic Multilayer Network Modeling

Official Python implementation of **TSSDMN** for the paper:

> **Tensor State Space-based Dynamic Multilayer Network Modeling**  
> Tian Lan, Jie Guo, Manrui Jiang, Yifei Gao, and Chen Zhang

## Overview

TSSDMN models binary dynamic multilayer networks $\mathcal{X}_t \in \{0, 1\}^{n \times n \times K}$ ($t = 1, \dots, T$) through a tensor state-space formulation:

$$\boldsymbol{\Gamma}_t = \mathcal{Z}_t \times_1 \mathbf{C}_1 \times_2 \mathbf{C}_1 \times_3 \mathbf{C}_2 + \mathcal{B}, \qquad \mathcal{Z}_t = \mathcal{Z}_{t-1} \times_1 \mathbf{A}_1 \times_2 \mathbf{A}_2 \times_3 \mathbf{A}_3 + \mathcal{E}_t,$$

where $\mathcal{X}_{t,ijk} \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\gamma_{t,ijk}))$. Model parameters and latent trajectories are estimated via the variational EM algorithm described in Section 4 and Appendix D of the paper.

## Repository Structure

```text
TSSDMN/
├── tssdmn/                  # Core TSSDMN Python package
│   ├── model.py             # Parameter container and initialization
│   ├── objective.py         # Tractable surrogate ELBO and M-step objectives
│   ├── em.py                # Variational EM estimation and forecasting
│   ├── tensor.py            # Kronecker and mode-factor utilities
│   ├── metrics.py           # ROC-AUC and RMSE evaluation metrics
│   └── adjust.py            # Permutation/scale alignment for C_1
├── data/
│   ├── icews_30x30x4x96.npz      # Preprocessed ICEWS network (n=30, K=4, T=95+1)
│   └── unsw_nb15_40x40x4x61.npz  # Preprocessed UNSW-NB15 network (n=40, K=4, T=60+1)
├── examples/
│   ├── demo_simulation.py        # Synthetic simulation and recovery demo (Section 5)
│   ├── demo_icews.py             # Empirical case study on ICEWS (Section 6.1)
│   └── demo_unsw_nb15.py         # Empirical case study on UNSW-NB15 (Section 6.2)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Lantian72/TSSDMN.git
cd TSSDMN
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Synthetic Simulation Demo

Generates a synthetic dynamic multilayer network ($n = 20, K = 2, m = 2, T = 30, \sigma^2 = 0.01$), fits TSSDMN via variational EM, and reports estimation error, one-step-ahead AUC, aligned $\mathbf{C}_1$ error, and the identified observed-layer propagation operator $\hat{\mathbf{A}}_{3,\mathrm{obs}} = \hat{\mathbf{C}}_2 \hat{\mathbf{A}}_3 \hat{\mathbf{C}}_2^{-1}$:

```bash
python examples/demo_simulation.py
```

### 2. Empirical Case Study Demo (ICEWS)

Fits TSSDMN ($m = 4$) on the $T = 95$ monthly snapshots of the ICEWS international diplomatic network ($n = 30$ countries, $K = 4$ event layers) and evaluates one-step-ahead link prediction AUC at time step $T + 1$:

```bash
python examples/demo_icews.py
```

### 3. Empirical Case Study Demo (UNSW-NB15)

Fits TSSDMN ($m = 3$) on the $T = 60$ 10-second windows (600 seconds) of the UNSW-NB15 network ($n = 40$ active IP addresses, $K = 4$ communication layers) and evaluates one-step-ahead link prediction AUC on the subsequent 10-second window at $T + 1$:

```bash
python examples/demo_unsw_nb15.py
```

### 4. Fitting TSSDMN on Custom Data

```python
import numpy as np
from tssdmn import EMConfig, compute_a3_obs, em, forecast_one_step_probs

# X_train: binary array of shape (n * n * K, T) in column-major order
n, K, m, T = 20, 3, 2, 40
X_train = np.random.binomial(1, 0.15, size=(n * n * K, T)).astype(np.float64)

config = EMConfig(max_iter=150, seed=0, device="cpu")
fitted_model, mut, sigmat = em(X_train, I=(n, n, K), J=(m, m, K), config=config)

next_step_probs = forecast_one_step_probs(fitted_model, mut)
a3_obs = compute_a3_obs(fitted_model)
```

## Empirical Data Sources

- **ICEWS Coded Event Data (Section 6.1):** Raw event records are publicly available from Harvard Dataverse at <https://doi.org/10.7910/DVN/28075>. For convenience, the aggregated binary network tensor ($n = 30, K = 4$, 95 training months + 1 test month) is included in `data/icews_30x30x4x96.npz`.
- **UNSW-NB15 Network Traffic Dataset (Section 6.2):** Raw packet and flow records are publicly available from UNSW at <https://research.unsw.edu.au/projects/unsw-nb15-dataset/>. Directed binary snapshots are constructed over the 40 most active IP addresses across 4 communication layers (`Normal`, `Fuzzers`, `Exploits`, `Reconnaissance`) using 10-second windows over 600 seconds ($T = 60$ training snapshots, plus the subsequent 10-second test window at $T + 1$), and the aggregated binary tensor is included in `data/unsw_nb15_40x40x4x61.npz`.

## Reproducibility Note

Because variational EM optimizes a nonconvex objective from randomized initializations, minor numerical variations across hardware platforms, operating systems, or library versions are expected while preserving consistent qualitative and quantitative conclusions.
