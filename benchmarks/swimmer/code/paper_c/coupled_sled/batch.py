from typing import Iterable

import numpy as np

from .waveforms import Probe


def simulate_batch(
    parameters: np.ndarray,
    probe: Probe,
    actuator_gains: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Vectorized RK4 simulation for many systems under one probe."""

    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 6 or np.any(theta <= 0):
        raise ValueError("parameters must have shape [system, 6] and be positive")
    gain = np.broadcast_to(np.asarray(actuator_gains, dtype=float), (len(theta),))
    if np.any(gain <= 0):
        raise ValueError("actuator gains must be positive")
    state = np.zeros((len(theta), 4), dtype=float)
    states = np.empty((len(theta), len(probe.actions) + 1, 4), dtype=float)
    states[:, 0] = state
    m_L, m_R, b_L, b_R, k_c, d_c = theta.T

    def f(values: np.ndarray, action: np.ndarray) -> np.ndarray:
        x_L, x_R, v_L, v_R = values.T
        coupling = k_c * (x_L - x_R) + d_c * (v_L - v_R)
        return np.column_stack(
            (
                v_L,
                v_R,
                (gain * action[0] - b_L * v_L - coupling) / m_L,
                (gain * action[1] - b_R * v_R + coupling) / m_R,
            )
        )

    dt = probe.dt
    for index, action in enumerate(probe.actions):
        k1 = f(state, action)
        k2 = f(state + 0.5 * dt * k1, action)
        k3 = f(state + 0.5 * dt * k2, action)
        k4 = f(state + dt * k3, action)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        states[:, index + 1] = state
    return states


def batch_position_landmarks(states: np.ndarray, count: int, semantics: str, sides: Iterable[str]) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("states must have shape [system, time, 4]")
    indices = np.linspace(1, values.shape[1] - 1, count).round().astype(int)
    if semantics == "both_positions":
        return values[:, indices, :2].reshape(values.shape[0], -1)
    side_list = list(sides)
    if semantics == "actuated_side_position" and len(side_list) == values.shape[0]:
        result = np.empty((values.shape[0], count), dtype=float)
        for index, side in enumerate(side_list):
            result[index] = values[index, indices, 0 if side == "L" else 1]
        return result
    raise ValueError("unsupported observation semantics")
