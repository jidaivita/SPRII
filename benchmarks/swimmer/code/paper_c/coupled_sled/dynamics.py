from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SledParameters:
    m_L: float
    m_R: float
    b_L: float
    b_R: float
    k_c: float
    d_c: float

    def validate(self) -> None:
        values = np.asarray([self.m_L, self.m_R, self.b_L, self.b_R, self.k_c, self.d_c])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("all coupled-sled parameters must be finite and positive")


def derivative(
    state: np.ndarray,
    action: np.ndarray,
    parameters: SledParameters,
    actuator_gain: float = 1.0,
) -> np.ndarray:
    """Canonical continuous-time two-module dynamics."""

    parameters.validate()
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.shape != (4,) or action.shape != (2,):
        raise ValueError("state/action shapes must be (4,) and (2,)")
    if not np.isfinite(actuator_gain) or actuator_gain <= 0:
        raise ValueError("actuator gain must be finite and positive")
    x_L, x_R, v_L, v_R = state
    delta_x = x_L - x_R
    delta_v = v_L - v_R
    coupling = parameters.k_c * delta_x + parameters.d_c * delta_v
    a_L = (actuator_gain * action[0] - parameters.b_L * v_L - coupling) / parameters.m_L
    a_R = (actuator_gain * action[1] - parameters.b_R * v_R + coupling) / parameters.m_R
    return np.asarray([v_L, v_R, a_L, a_R], dtype=float)


def rk4_step(
    state: np.ndarray,
    action: np.ndarray,
    dt: float,
    parameters: SledParameters,
    actuator_gain: float,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    k1 = derivative(state, action, parameters, actuator_gain)
    k2 = derivative(state + 0.5 * dt * k1, action, parameters, actuator_gain)
    k3 = derivative(state + 0.5 * dt * k2, action, parameters, actuator_gain)
    k4 = derivative(state + dt * k3, action, parameters, actuator_gain)
    return np.asarray(state, dtype=float) + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate_reference(
    parameters: SledParameters,
    actions: np.ndarray,
    dt: float,
    actuator_gain: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the canonical model with piecewise-constant actions.

    Returns times and states including the initial state.  The reference RK4
    integrator is deterministic and is the CPU information-analysis source of
    truth.
    """

    controls = np.asarray(actions, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("actions must have shape [step, 2]")
    state = np.zeros(4, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    if state.shape != (4,):
        raise ValueError("initial_state must have shape (4,)")
    states = np.empty((len(controls) + 1, 4), dtype=float)
    states[0] = state
    for index, action in enumerate(controls):
        state = rk4_step(state, action, dt, parameters, actuator_gain)
        states[index + 1] = state
    return np.arange(len(states), dtype=float) * dt, states


def position_landmarks(states: np.ndarray, count: int, semantics: str = "both_positions", side: str | None = None) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or count < 2:
        raise ValueError("invalid state trajectory or landmark count")
    indices = np.linspace(1, len(values) - 1, count).round().astype(int)
    if semantics == "both_positions":
        return values[indices, :2].reshape(-1)
    if semantics == "actuated_side_position" and side in {"L", "R"}:
        return values[indices, 0 if side == "L" else 1].copy()
    raise ValueError("unsupported observation semantics")
