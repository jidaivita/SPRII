from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Probe:
    probe_id: str
    actions: np.ndarray
    dt: float
    side: str

    @property
    def duration_s(self) -> float:
        return len(self.actions) * self.dt

    @property
    def energy(self) -> float:
        return float(np.sum(self.actions**2) * self.dt)

    @property
    def peak_force(self) -> float:
        return float(np.max(np.abs(self.actions)))


def _times(duration_s: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    steps = int(round(duration_s / dt))
    if not np.isclose(steps * dt, duration_s):
        raise ValueError("duration must be an integer multiple of dt")
    times = (np.arange(steps, dtype=float) + 0.5) * dt
    return times, times / duration_s


def normalize_vector_energy(raw: np.ndarray, dt: float, energy: float) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    current = float(np.sum(values**2) * dt)
    if current <= 0 or energy <= 0:
        raise ValueError("raw action and target energy must be positive")
    return values * np.sqrt(energy / current)


def _history_raw(kind: str, s: np.ndarray, rho: float = 0.2) -> np.ndarray:
    if kind == "slow":
        return np.sin(np.pi * s) ** 2
    if kind == "oscillatory":
        return np.sin(2.0 * np.pi * s) * np.sin(np.pi * s) ** 2
    if kind == "impulse":
        return np.where(s <= rho, np.sin(np.pi * s / rho) ** 2, 0.0)
    raise KeyError(kind)


def history_probe_bank(duration_s: float, dt: float, energy: float, peak_force: float | None = None) -> Dict[str, Probe]:
    _, s = _times(duration_s, dt)
    bank: Dict[str, Probe] = {}
    for side, channel in (("L", 0), ("R", 1)):
        for kind in ("slow", "oscillatory", "impulse"):
            raw = np.zeros((len(s), 2), dtype=float)
            raw[:, channel] = _history_raw(kind, s)
            actions = normalize_vector_energy(raw, dt, energy)
            probe = Probe(f"P_{side}_{kind}", actions, dt, side)
            if peak_force is not None and probe.peak_force > peak_force + 1e-12:
                raise ValueError(f"{probe.probe_id} exceeds peak-force ceiling")
            bank[probe.probe_id] = probe
    return bank


def _triangle(s: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(2.0 * s - 1.0)


def _bump(s: np.ndarray, center: float, width: float) -> np.ndarray:
    start = center - width / 2.0
    local = (s - start) / width
    return np.where((local >= 0.0) & (local <= 1.0), np.sin(np.pi * local) ** 2, 0.0)


def _chirp(s: np.ndarray, duration_s: float, f0: float, f1: float) -> np.ndarray:
    phase = 2.0 * np.pi * (f0 * duration_s * s + 0.5 * (f1 - f0) * duration_s * s**2)
    return np.sin(np.pi * s) ** 2 * np.sin(phase)


def query_bank(
    duration_s: float,
    dt: float,
    energy: float,
    chirp_hz: tuple[float, float],
    peak_force: float | None = None,
) -> Dict[str, Probe]:
    _, s = _times(duration_s, dt)
    triangle = _triangle(s)
    early = _bump(s, 0.25, 0.35)
    late = _bump(s, 0.75, 0.35)
    chirp = _chirp(s, duration_s, *chirp_hz)
    raw = {
        "Q_left_triangular": np.column_stack((triangle, np.zeros_like(s))),
        "Q_right_triangular": np.column_stack((np.zeros_like(s), triangle)),
        "Q_left_then_right": np.column_stack((early, late)),
        "Q_right_then_left": np.column_stack((late, early)),
        "Q_bilateral_in_phase": np.column_stack((chirp, chirp)),
        "Q_bilateral_anti_phase": np.column_stack((chirp, -chirp)),
    }
    bank = {}
    for probe_id, values in raw.items():
        actions = normalize_vector_energy(values, dt, energy)
        probe = Probe(probe_id, actions, dt, "B")
        if peak_force is not None and probe.peak_force > peak_force + 1e-12:
            raise ValueError(f"{probe_id} exceeds peak-force ceiling")
        bank[probe_id] = probe
    return bank
