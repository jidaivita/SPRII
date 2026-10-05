import numpy as np


def _normalize_energy(signal: np.ndarray, target_rms: float = 0.42) -> np.ndarray:
    rms = np.sqrt(np.mean(np.sum(signal ** 2, axis=1)))
    return signal * (target_rms / max(rms, 1e-12))


def banks(duration_s: float, dt: float):
    steps = int(round(duration_s / dt))
    t = (np.arange(steps) + 0.5) * dt
    phase = t / duration_s
    slow = np.sin(2 * np.pi * phase)
    cosine = np.cos(2 * np.pi * phase)
    chirp = np.sin(2 * np.pi * (0.75 * t + 1.75 * t * t / max(duration_s, 1e-9)))
    triangle = 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1
    zero = np.zeros_like(t)

    def pair(a, b):
        return _normalize_energy(np.column_stack([a, b]))

    history = {
        "j1_slow": pair(slow, zero),
        "j1_chirp": pair(chirp, zero),
        "j2_slow": pair(zero, slow),
        "j2_chirp": pair(zero, chirp),
        "in_phase": pair(slow, slow),
        "quadrature": pair(slow, cosine),
    }
    query = {
        "j1_triangle": pair(triangle, zero),
        "j1_chirp": pair(chirp, zero),
        "j2_triangle": pair(zero, triangle),
        "j2_chirp": pair(zero, chirp),
        "in_phase_chirp": pair(chirp, chirp),
        "anti_phase_chirp": pair(chirp, -chirp),
    }
    return history, query


def development_alternatives(duration_s: float, dt: float):
    steps = int(round(duration_s / dt))
    t = (np.arange(steps) + 0.5) * dt
    phase = t / duration_s
    slow = np.sin(2 * np.pi * phase)
    cosine = np.cos(2 * np.pi * phase)
    phase_60 = np.sin(2 * np.pi * phase + np.pi / 3)
    pulse = np.tanh(4 * np.sin(2 * np.pi * phase))
    zero = np.zeros_like(t)

    def pair(a, b):
        return _normalize_energy(np.column_stack([a, b]))

    return {
        "anti_phase": pair(slow, -slow),
        "quadrature": pair(slow, cosine),
        "phase_60": pair(slow, phase_60),
        "j1_pulse": pair(pulse, zero),
        "j2_pulse": pair(zero, pulse),
    }
