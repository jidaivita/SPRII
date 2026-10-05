"""Physics-informed drag certificate and coverage diagnostics."""

from __future__ import annotations

import numpy as np


def transition_gamma_estimates(
    states: np.ndarray,
    actions: np.ndarray,
    dt: float = 0.05,
    force_tol: float = 1e-10,
    speed_min: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(states, dtype=np.float64)[..., :-1, 2:]
    next_velocity = np.asarray(states, dtype=np.float64)[..., 1:, 2:]
    force = np.asarray(actions, dtype=np.float64)
    denom = np.sum(velocity * velocity, axis=-1)
    alpha = np.divide(
        np.sum(velocity * next_velocity, axis=-1),
        denom,
        out=np.full_like(denom, np.nan),
        where=denom > 0,
    )
    valid = (
        (np.linalg.norm(force, axis=-1) <= force_tol)
        & (np.sqrt(denom) >= speed_min)
        & (alpha > 0.0)
        & (alpha <= 1.0)
    )
    gamma = np.full_like(alpha, np.nan)
    gamma[valid] = -np.log(alpha[valid]) / dt
    return gamma, valid


def system_gamma_estimates(
    states: np.ndarray, actions: np.ndarray, dt: float = 0.05
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate each system from all valid transitions across its rollouts."""
    gamma, valid = transition_gamma_estimates(states, actions, dt=dt)
    estimates = np.full(states.shape[0], np.nan, dtype=np.float64)
    for system in range(states.shape[0]):
        values = gamma[system][valid[system]]
        if values.size:
            estimates[system] = np.median(values)
    valid_per_rollout = valid.sum(axis=-1)
    diagnostics = {
        "valid_transition_fraction": float(valid.mean()),
        "valid_window_fraction": float((valid_per_rollout >= 2).mean()),
        "valid_system_fraction": float(np.isfinite(estimates).mean()),
    }
    return estimates, diagnostics


def r2_score_system(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    truth, pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if truth.size < 2:
        return float("nan")
    denom = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - np.sum((truth - pred) ** 2) / denom)


def pokeworld_drag_estimates(
    states: np.ndarray,
    contact: np.ndarray,
    dt: float = 0.05,
    substeps: int = 20,
    speed_min: float = 1e-3,
    wall_margin: float = 0.12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Privileged-state certificate on contact-free, wall-free object glides."""
    state64 = np.asarray(states, dtype=np.float64)
    velocity = state64[..., :-1, 6:8]
    next_velocity = state64[..., 1:, 6:8]
    position = state64[..., :-1, 4:6]
    next_position = state64[..., 1:, 4:6]
    denominator = np.square(velocity).sum(axis=-1)
    alpha = np.divide(
        (velocity * next_velocity).sum(axis=-1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    away_from_wall = (
        (np.abs(position) < 1.0 - wall_margin).all(axis=-1)
        & (np.abs(next_position) < 1.0 - wall_margin).all(axis=-1)
    )
    valid = (
        ~np.asarray(contact, dtype=bool)
        & (np.sqrt(denominator) >= speed_min)
        & away_from_wall
        & (alpha > 0.0)
        & (alpha <= 1.0)
    )
    gamma = np.full_like(alpha, np.nan)
    gamma[valid] = (1.0 - np.power(alpha[valid], 1.0 / substeps)) / (dt / substeps)
    estimate = np.full(states.shape[0], np.nan, dtype=np.float64)
    for system in range(states.shape[0]):
        values = gamma[system][valid[system]]
        if values.size:
            estimate[system] = np.median(values)
    return estimate, {
        "valid_transition_fraction": float(valid.mean()),
        "valid_system_fraction": float(np.isfinite(estimate).mean()),
    }
