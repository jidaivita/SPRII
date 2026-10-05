"""Train-only, coordinate-wise physics sensitivity for constructive routing."""

from __future__ import annotations

import hashlib
import json
import numpy as np

from .poke_simulator import replay


def _system_weights(system_ids):
    ids = np.asarray(system_ids)
    if ids.ndim != 1:
        raise ValueError("system_ids must be one-dimensional")
    _, counts = np.unique(ids, return_counts=True)
    count_map = dict(zip(np.unique(ids).tolist(), counts.tolist()))
    return np.asarray([1.0 / count_map[x] for x in ids], dtype=np.float64)


def coordinate_sensitivity(initial, actions, theta, target_scale, parameter_scale, config,
                           system_ids, *, fraction=0.01, horizon_index=4):
    """Return train-only per-output sensitivity and frozen soft weights.

    The returned sensitivity has shape ``[N, 8]`` and is computed at one
    declared horizon (h16 by default).  Each output coordinate's norm is over
    the three physical parameter derivatives.  Systems, rather than windows,
    receive equal total weight.
    """
    initial = np.asarray(initial, np.float64)
    actions = np.asarray(actions, np.float64)
    theta = np.asarray(theta, np.float64)
    target_scale = np.asarray(target_scale, np.float64)
    parameter_scale = np.asarray(parameter_scale, np.float64)
    if initial.ndim != 2 or initial.shape[1] != 8:
        raise ValueError("initial must have shape [N,8]")
    if actions.ndim != 3 or actions.shape[:2] != (len(initial), 16):
        raise ValueError("actions must have shape [N,16,2]")
    if theta.shape != (len(initial), 3) or np.any(theta <= 0):
        raise ValueError("theta must be positive [N,3]")
    if target_scale.shape != (5, 8) or parameter_scale.shape != (3,):
        raise ValueError("normalization shapes differ")
    if not 0 <= horizon_index < 5 or fraction <= 0:
        raise ValueError("invalid finite-difference configuration")
    if not np.isfinite(target_scale).all() or not np.isfinite(parameter_scale).all():
        raise ValueError("normalization must be finite")
    h = (1, 2, 4, 8, 16)[horizon_index]
    target_scale_h = target_scale[horizon_index].clip(1e-8)
    derivatives = []
    for j in range(3):
        delta = theta[:, j] * fraction
        plus = theta.copy(); minus = theta.copy()
        plus[:, j] += delta; minus[:, j] -= delta
        yp, _ = replay(initial, actions[:, :h], plus, config)
        ym, _ = replay(initial, actions[:, :h], minus, config)
        derivative = (yp[:, -1, :] - ym[:, -1, :]) / (2 * delta[:, None])
        derivatives.append(derivative / target_scale_h[None, :] * parameter_scale[j])
    sensitivity = np.sqrt(np.sum(np.square(np.stack(derivatives, axis=2)), axis=2))
    if not np.isfinite(sensitivity).all():
        raise ValueError("nonfinite coordinate sensitivity")
    weights = _system_weights(system_ids)
    mean_sensitivity = np.average(sensitivity, axis=0, weights=weights)
    w = mean_sensitivity / max(float(mean_sensitivity.max()), 1e-12)
    return sensitivity.astype(np.float32), w.astype(np.float32)


def freeze_record(*, sensitivity, weights, system_ids, config, fraction, horizon_index,
                  target_scale, parameter_scale):
    """Create a hashable Phase-0 receipt before any route selection."""
    sensitivity = np.asarray(sensitivity, np.float32)
    weights = np.asarray(weights, np.float32)
    if sensitivity.ndim != 2 or sensitivity.shape[1] != 8 or weights.shape != (8,):
        raise ValueError("coordinate-wise sensitivity/weights shape mismatch")
    payload = dict(stage='phase0', split='train', horizon_index=int(horizon_index),
                   fraction=float(fraction), system_ids=np.asarray(system_ids).tolist(),
                   target_scale=np.asarray(target_scale).tolist(),
                   parameter_scale=np.asarray(parameter_scale).tolist(),
                   weights=weights.tolist(),
                   sensitivity_sha256=hashlib.sha256(sensitivity.tobytes()).hexdigest(),
                   simulator_config=config, test_read=False)
    payload['config_sha256'] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload
