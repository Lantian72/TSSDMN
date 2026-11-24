"""Model representation and parameter initialization for TSSDMN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from tssdmn.tensor import build_cm_list, otimes, prod_dims, spectral_radius


@dataclass
class Model:
    """Parameter container for TSSDMN.

    Attributes:
        I: Observation dimensions (n, n, K).
        J: Core tensor dimensions (m, m, K).
        u0: Initial state mean vector in R^{m^2 K}.
        omega: Initial state standard deviation (omega^2 is initial variance).
        sigma: State innovation standard deviation (sigma^2 is innovation variance).
        b: Static bias vector vec(B) in R^{n^2 K}.
        Am: Mode-specific transition matrices [A1, A2, A3].
        Cm: Mode-specific observation matrices [C1, C1, C2].
        A: Composite transition matrix A3 (x) A2 (x) A1.
        C: Composite observation matrix C2 (x) C1 (x) C1.
    """

    I: tuple[int, ...]
    J: tuple[int, ...]
    u0: torch.Tensor
    omega: torch.Tensor
    sigma: torch.Tensor
    b: torch.Tensor
    Am: list[torch.Tensor]
    Cm: list[torch.Tensor]
    A: torch.Tensor
    C: torch.Tensor
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64

    @property
    def prod_i(self) -> int:
        return prod_dims(self.I)

    @property
    def prod_j(self) -> int:
        return prod_dims(self.J)

    def clone(self) -> "Model":
        return Model(
            I=self.I,
            J=self.J,
            u0=self.u0.clone(),
            omega=self.omega.clone(),
            sigma=self.sigma.clone(),
            b=self.b.clone(),
            Am=[m.clone() for m in self.Am],
            Cm=[m.clone() for m in self.Cm],
            A=self.A.clone(),
            C=self.C.clone(),
            device=self.device,
            dtype=self.dtype,
        )

    def rebuild_matrices(self) -> None:
        self.A = otimes(self.Am)
        self.C = otimes(self.Cm)

    def to_dict(self) -> dict:
        return {
            "I": self.I,
            "J": self.J,
            "u0": self.u0.detach().cpu(),
            "omega": self.omega.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "b": self.b.detach().cpu(),
            "Am": [m.detach().cpu() for m in self.Am],
            "Cm": [m.detach().cpu() for m in self.Cm],
            "A": self.A.detach().cpu(),
            "C": self.C.detach().cpu(),
        }


def _initialize_nonsingular_c2(
    k_dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    while True:
        matrix = torch.randn(k_dim, k_dim, generator=generator, device="cpu", dtype=dtype)
        if torch.linalg.matrix_rank(matrix) == k_dim:
            break
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.T
    return (3.0 * (u @ v)).to(device)


def initialize_stable_am(
    J: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    target_rho: float = 0.85,
) -> list[torch.Tensor]:
    """Sample mode-specific transition matrices with spectral radius rho(A) < 1."""
    j_tuple = tuple(int(x) for x in J)
    for _ in range(100):
        am_list: list[torch.Tensor] = []
        for j_dim in j_tuple:
            mat = torch.randn(j_dim, j_dim, generator=generator, device="cpu", dtype=dtype)
            u, _, vh = torch.linalg.svd(mat, full_matrices=False)
            singular = torch.linspace(0.45, 0.85, j_dim, device="cpu", dtype=dtype)
            block = u @ torch.diag(singular) @ vh
            am_list.append(block.to(device))
        rho = spectral_radius(otimes(am_list))
        if rho >= target_rho:
            factor = (target_rho / max(rho, 1e-12)) ** (1.0 / len(j_tuple))
            am_list = [block * factor for block in am_list]
        if spectral_radius(otimes(am_list)) < 1.0:
            return am_list
    raise RuntimeError("Failed to sample stable transition matrices with rho(A) < 1.")


def initialize_model(
    I: Sequence[int],
    J: Sequence[int],
    sigma: float = 0.1,
    *,
    seed: int | None = None,
    am_scale: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Model:
    """Initialize a TSSDMN parameter set."""
    I_tuple = tuple(int(x) for x in I)
    J_tuple = tuple(int(x) for x in J)
    device = device or torch.device("cpu")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    prod_i = prod_dims(I_tuple)
    prod_j = prod_dims(J_tuple)

    u0 = torch.zeros(prod_j, device=device, dtype=dtype)
    omega = torch.tensor(sigma, device=device, dtype=dtype)
    sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
    b = torch.randn(prod_i, generator=generator, device="cpu", dtype=dtype).to(device)

    if am_scale is not None:
        Am = [am_scale * torch.eye(j_dim, device=device, dtype=dtype) for j_dim in J_tuple]
    else:
        Am = initialize_stable_am(J_tuple, generator=generator, device=device, dtype=dtype)

    cm0 = torch.clamp(
        torch.randn(I_tuple[0], J_tuple[0], generator=generator, device="cpu", dtype=dtype) + 1.0,
        min=0.0,
    ).to(device)
    cm_last = (
        _initialize_nonsingular_c2(I_tuple[-1], generator, device, dtype)
        if len(I_tuple) >= 3
        else None
    )
    cm_list = build_cm_list(
        cm0,
        None,
        cm_last,
        len(I_tuple),
        share_first_two=(len(I_tuple) >= 2 and I_tuple[0] == I_tuple[1] and J_tuple[0] == J_tuple[1]),
    )

    return Model(
        I=I_tuple,
        J=J_tuple,
        u0=u0,
        omega=omega,
        sigma=sigma_t,
        b=b,
        Am=Am,
        Cm=cm_list,
        A=otimes(Am),
        C=otimes(cm_list),
        device=device,
        dtype=dtype,
    )


def prepare_em_initialization(model: Model) -> Model:
    """Scale initial observation factors for numerically stable EM warm-up."""
    prepared = model.clone()
    prepared.Cm[0] = prepared.Cm[0] * 0.1
    prepared.Cm[1] = prepared.Cm[0].clone()
    if len(prepared.Cm) > 2:
        prepared.Cm[2] = prepared.Cm[2] * 0.1
    prepared.Am = [0.5 * torch.eye(j_dim, device=prepared.device, dtype=prepared.dtype) for j_dim in prepared.J]
    prepared.rebuild_matrices()
    return prepared
