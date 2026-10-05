"""Numerical core for the released NOD FitzHugh--Nagumo generator.

The formulas mirror ``DiffusionReaction2D_Data_Generator.m``:

* periodic fourth-order centered Laplacian;
* classical fourth-order Runge--Kutta time integration;
* ``du/dt = Du*lap(u) + u - u**3 + k - v``;
* ``dv/dt = Dv*lap(v) + beta*(u-v)``.

NumPy is the dependency-light reference backend.  A PyTorch implementation is
provided for batched GPU generation.  Formal generation defaults to float64,
matching MATLAB's default numeric type; saved learning tensors may be cast to
float32 after integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FHNConfig:
    grid_size: int = 128
    dt: float = 1.0e-3
    steps: int = 10_000
    sample_every: int = 100
    du: float | None = None
    dv: float | None = None

    @property
    def dx(self) -> float:
        return 1.0 / self.grid_size

    @property
    def diffusion_u(self) -> float:
        return 1.0 / self.grid_size**2 if self.du is None else self.du

    @property
    def diffusion_v(self) -> float:
        return 100.0 / self.grid_size**2 if self.dv is None else self.dv

    @property
    def snapshots(self) -> int:
        if self.steps % self.sample_every:
            raise ValueError("steps must be divisible by sample_every")
        return self.steps // self.sample_every + 1


def _wavenumbers(size: int) -> np.ndarray:
    """Return the exact ordering used by the released MATLAB GRF routine."""
    if size % 2:
        raise ValueError("the released generator assumes an even grid size")
    half = size // 2
    return np.concatenate((np.arange(half), np.arange(-half, 0)))


def gaussian_random_fields(
    count: int,
    grid_size: int = 128,
    *,
    alpha: float = 2.0,
    tau: float = 5.0,
    seed: int = 0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Generate the two-component GRF initial conditions described in the paper.

    The implementation follows ``GaussianRF_2D.m`` including its complex
    coefficient construction and real-part projection.  NumPy and MATLAB do not
    share a random-number stream, so a seed provides reproducibility for the
    Python port rather than bitwise identity with an unavailable MATLAB bank.
    """
    rng = np.random.default_rng(seed)
    k = _wavenumbers(grid_size)
    k_x = k[:, None]
    k_y = k[None, :]
    sigma = tau ** (0.5 * (2.0 * alpha - 2.0))
    sqrt_eig = (
        grid_size**2
        * np.sqrt(2.0)
        * sigma
        * (4.0 * np.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2.0)
    )
    sqrt_eig[0, 0] = 0.0
    coeff = rng.standard_normal((count, 2, grid_size, grid_size, 2))
    coeff_c = (coeff[..., 0] + 1j * coeff[..., 1]) * sqrt_eig
    fields = np.fft.ifft2(coeff_c, axes=(-2, -1)).real
    return fields.astype(dtype, copy=False)


def iid_gaussian_initial_conditions(
    count: int,
    grid_size: int = 128,
    *,
    seed: int = 0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Mirror ``RandomInit_2D.m`` for resolving the release's IC ambiguity."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, 2, grid_size, grid_size)).astype(dtype)


def apply_laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Released periodic fourth-order centered Laplacian."""
    out = -5.0 * field
    weights = (4.0 / 3, 4.0 / 3, 4.0 / 3, 4.0 / 3,
               -1.0 / 12, -1.0 / 12, -1.0 / 12, -1.0 / 12)
    shifts = ((0, -1), (-1, 0), (1, 0), (0, 1),
              (0, -2), (-2, 0), (2, 0), (0, 2))
    for weight, (shift_y, shift_x) in zip(weights, shifts):
        out = out + weight * np.roll(
            field, shift=(shift_y, shift_x), axis=(-2, -1)
        )
    return out / dx**2


def temporal_derivative(
    state: np.ndarray, *, k: float, beta: float, config: FHNConfig
) -> np.ndarray:
    u, v = state[..., 0, :, :], state[..., 1, :, :]
    du_dt = (
        config.diffusion_u * apply_laplacian(u, config.dx)
        + u
        - u**3
        + k
        - v
    )
    dv_dt = (
        config.diffusion_v * apply_laplacian(v, config.dx)
        + beta * (u - v)
    )
    return np.stack((du_dt, dv_dt), axis=-3)


def rk4_step(
    state: np.ndarray, *, k: float, beta: float, config: FHNConfig
) -> np.ndarray:
    dt = config.dt
    k1 = temporal_derivative(state, k=k, beta=beta, config=config)
    k2 = temporal_derivative(state + 0.5 * dt * k1, k=k, beta=beta, config=config)
    k3 = temporal_derivative(state + 0.5 * dt * k2, k=k, beta=beta, config=config)
    k4 = temporal_derivative(state + dt * k3, k=k, beta=beta, config=config)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(
    initial_state: np.ndarray,
    *,
    k: float,
    beta: float,
    config: FHNConfig = FHNConfig(),
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Integrate a batch and return ``[batch,time,2,y,x]`` snapshots."""
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 4 or state.shape[1:] != (
        2,
        config.grid_size,
        config.grid_size,
    ):
        raise ValueError(
            f"expected [batch,2,{config.grid_size},{config.grid_size}], got {state.shape}"
        )
    records = np.empty(
        (state.shape[0], config.snapshots, *state.shape[1:]), dtype=output_dtype
    )
    records[:, 0] = state
    snapshot = 1
    for step in range(1, config.steps + 1):
        state = rk4_step(state, k=k, beta=beta, config=config)
        if step % config.sample_every == 0:
            records[:, snapshot] = state
            snapshot += 1
    if not np.isfinite(records).all():
        raise FloatingPointError(f"non-finite state for k={k}, beta={beta}")
    return records


def torch_simulate(
    initial_state: np.ndarray,
    *,
    k: float,
    beta: float,
    config: FHNConfig = FHNConfig(),
    device: str = "cuda",
    solver_dtype: Literal["float32", "float64"] = "float64",
    output_dtype: Literal["float32", "float64"] = "float32",
) -> np.ndarray:
    """Batched GPU implementation of the same released solver."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on generation host
        raise RuntimeError("PyTorch is required for --backend torch") from exc

    dtype = getattr(torch, solver_dtype)
    out_dtype = np.dtype(output_dtype)
    state = torch.as_tensor(initial_state, dtype=dtype, device=device)
    if state.ndim != 4 or tuple(state.shape[1:]) != (
        2,
        config.grid_size,
        config.grid_size,
    ):
        raise ValueError("invalid initial-state shape")

    def lap(field):
        out = -5.0 * field
        weights = (4.0 / 3, 4.0 / 3, 4.0 / 3, 4.0 / 3,
                   -1.0 / 12, -1.0 / 12, -1.0 / 12, -1.0 / 12)
        shifts = ((0, -1), (-1, 0), (1, 0), (0, 1),
                  (0, -2), (-2, 0), (2, 0), (0, 2))
        for weight, shift in zip(weights, shifts):
            out = out + weight * torch.roll(field, shifts=shift, dims=(-2, -1))
        return out / config.dx**2

    def deriv(x):
        u, v = x[:, 0], x[:, 1]
        du_dt = config.diffusion_u * lap(u) + u - u**3 + k - v
        dv_dt = config.diffusion_v * lap(v) + beta * (u - v)
        return torch.stack((du_dt, dv_dt), dim=1)

    records = np.empty(
        (state.shape[0], config.snapshots, 2, config.grid_size, config.grid_size),
        dtype=out_dtype,
    )
    records[:, 0] = state.detach().cpu().numpy().astype(out_dtype, copy=False)
    snapshot = 1
    dt = config.dt
    with torch.no_grad():
        for step in range(1, config.steps + 1):
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if step % config.sample_every == 0:
                records[:, snapshot] = (
                    state.detach().cpu().numpy().astype(out_dtype, copy=False)
                )
                snapshot += 1
    if not np.isfinite(records).all():
        raise FloatingPointError(f"non-finite state for k={k}, beta={beta}")
    return records
