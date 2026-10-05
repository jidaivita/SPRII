"""Vectorized independent PokeWorld reproduction from published equations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PokeConfig:
    dt: float = 0.05
    substeps: int = 20
    episode_states: int = 64
    rollouts_per_system: int = 4
    train_systems: int = 2000
    val_systems: int = 400
    test_systems: int = 400
    finger_mass: float = 1.0
    finger_radius: float = 0.06
    object_radius: float = 0.09
    damping_ratio: float = 0.25
    mass_low: float = 0.5
    mass_high: float = 3.0
    gamma_low: float = 0.5
    gamma_high: float = 4.0
    stiffness_low: float = 500.0
    stiffness_high: float = 6000.0
    # The fields below are independent-reproduction choices absent from the paper.
    force_max: float = 20.0
    arena_half_extent: float = 1.0
    wall_restitution: float = 0.5
    mode_probabilities: tuple[float, float, float] = (0.45, 0.35, 0.20)
    mode_steps_min: int = 4
    mode_steps_max: int = 12
    launch_probability: float = 0.25
    launch_speed_low: float = 0.5
    launch_speed_high: float = 2.0
    seed: int = 20260820

    @property
    def sub_dt(self) -> float:
        return self.dt / self.substeps


def _log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def sample_uniform_union(
    rng: np.random.Generator,
    intervals: tuple[tuple[float, float], ...],
    size: int,
) -> np.ndarray:
    """Sample uniformly from a union, weighting components by interval length."""
    if not intervals:
        raise ValueError("at least one gamma interval is required")
    bounds = np.asarray(intervals, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"invalid intervals: {intervals}")
    order = np.argsort(bounds[:, 0])
    bounds = bounds[order]
    if np.any(bounds[1:, 0] < bounds[:-1, 1]):
        raise ValueError(f"overlapping intervals: {intervals}")
    widths = bounds[:, 1] - bounds[:, 0]
    components = rng.choice(bounds.shape[0], size=size, p=widths / widths.sum())
    return rng.uniform(bounds[components, 0], bounds[components, 1])


def _reflect(position: np.ndarray, velocity: np.ndarray, limit: float, restitution: float) -> None:
    for axis in (0, 1):
        high = position[:, axis] > limit
        low = position[:, axis] < -limit
        position[high, axis] = limit
        position[low, axis] = -limit
        velocity[high & (velocity[:, axis] > 0), axis] *= -restitution
        velocity[low & (velocity[:, axis] < 0), axis] *= -restitution


def simulate_split(
    systems: int,
    system_offset: int,
    seed_sequence: np.random.SeedSequence,
    cfg: PokeConfig,
    gamma_intervals: tuple[tuple[float, float], ...] | None = None,
    system_parameters: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed_sequence)
    if system_parameters is None:
        mass = _log_uniform(rng, cfg.mass_low, cfg.mass_high, systems).astype(np.float32)
        if gamma_intervals is None:
            gamma = rng.uniform(cfg.gamma_low, cfg.gamma_high, systems).astype(np.float32)
        else:
            gamma = sample_uniform_union(rng, gamma_intervals, systems).astype(np.float32)
        stiffness = _log_uniform(rng, cfg.stiffness_low, cfg.stiffness_high, systems).astype(np.float32)
    else:
        if gamma_intervals is not None:
            raise ValueError("gamma_intervals cannot be combined with fixed system_parameters")
        if len(system_parameters) != 3:
            raise ValueError("system_parameters must be (mass, gamma, stiffness)")
        mass, gamma, stiffness = (
            np.asarray(values, dtype=np.float32).copy() for values in system_parameters
        )
        if any(values.shape != (systems,) for values in (mass, gamma, stiffness)):
            raise ValueError(f"fixed system parameter arrays must all have shape ({systems},)")
    n = systems * cfg.rollouts_per_system
    system_row = np.repeat(np.arange(systems), cfg.rollouts_per_system)
    m = mass[system_row].astype(np.float64)
    g = gamma[system_row].astype(np.float64)
    k = stiffness[system_row].astype(np.float64)

    object_position = rng.uniform(-0.35, 0.35, size=(n, 2))
    angle = rng.uniform(0, 2 * np.pi, size=n)
    distance = rng.uniform(0.35, 0.60, size=n)
    finger_position = object_position + distance[:, None] * np.stack(
        [np.cos(angle), np.sin(angle)], axis=1
    )
    finger_position = np.clip(finger_position, -0.85, 0.85)
    finger_velocity = np.zeros((n, 2), dtype=np.float64)
    object_velocity = np.zeros((n, 2), dtype=np.float64)
    launched = rng.random(n) < cfg.launch_probability
    launch_angle = rng.uniform(0, 2 * np.pi, size=n)
    launch_speed = rng.uniform(cfg.launch_speed_low, cfg.launch_speed_high, size=n)
    object_velocity[launched] = launch_speed[launched, None] * np.stack(
        [np.cos(launch_angle[launched]), np.sin(launch_angle[launched])], axis=1
    )

    states = np.empty((n, cfg.episode_states, 8), dtype=np.float32)
    actions = np.empty((n, cfg.episode_states - 1, 2), dtype=np.float32)
    touch = np.empty((n, cfg.episode_states - 1, 7), dtype=np.float32)
    contact = np.empty((n, cfg.episode_states - 1), dtype=bool)
    states[:, 0] = np.concatenate(
        [finger_position, finger_velocity, object_position, object_velocity], axis=1
    )
    mode = rng.choice(3, size=n, p=cfg.mode_probabilities)
    remaining = rng.integers(cfg.mode_steps_min, cfg.mode_steps_max + 1, size=n)
    duration = remaining.copy()
    ou = np.zeros((n, 2), dtype=np.float64)

    for step in range(cfg.episode_states - 1):
        switch = remaining <= 0
        if switch.any():
            mode[switch] = rng.choice(3, size=switch.sum(), p=cfg.mode_probabilities)
            remaining[switch] = rng.integers(
                cfg.mode_steps_min, cfg.mode_steps_max + 1, size=switch.sum()
            )
            duration[switch] = remaining[switch]
        delta = object_position - finger_position
        direction = delta / np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-8)
        action = np.zeros((n, 2), dtype=np.float64)
        pursuit = mode == 0
        action[pursuit] = direction[pursuit]
        strike = mode == 1
        first_half = remaining > (duration // 2)
        strike_sign = np.where(first_half, 1.0, -1.0)
        action[strike] = direction[strike] * strike_sign[strike, None]
        drift = mode == 2
        ou[drift] = 0.82 * ou[drift] + 0.35 * rng.normal(size=(drift.sum(), 2))
        action[drift] = ou[drift]
        action += rng.normal(scale=0.06, size=(n, 2))
        action = np.clip(action, -1.0, 1.0)
        actions[:, step] = action.astype(np.float32)
        remaining -= 1

        force_sum = np.zeros((n, 2), dtype=np.float64)
        force_end = np.zeros((n, 2), dtype=np.float64)
        force_peak = np.zeros(n, dtype=np.float64)
        contact_count = np.zeros(n, dtype=np.float64)
        for _ in range(cfg.substeps):
            separation = object_position - finger_position
            distance_now = np.linalg.norm(separation, axis=1).clip(1e-8)
            normal = separation / distance_now[:, None]
            overlap = cfg.finger_radius + cfg.object_radius - distance_now
            active = overlap > 0
            relative_normal_velocity = np.sum(
                (object_velocity - finger_velocity) * normal, axis=1
            )
            local_stiffness = 1.5 * k * np.sqrt(np.maximum(overlap, 0.0))
            effective_mass = m / (m + cfg.finger_mass)
            damping = 2.0 * cfg.damping_ratio * np.sqrt(local_stiffness * effective_mass)
            magnitude = np.maximum(
                0.0, k * np.maximum(overlap, 0.0) ** 1.5 - damping * relative_normal_velocity
            )
            magnitude *= active
            object_contact_force = magnitude[:, None] * normal
            finger_force = cfg.force_max * action - object_contact_force
            object_force = object_contact_force - (g * m)[:, None] * object_velocity
            finger_velocity += finger_force / cfg.finger_mass * cfg.sub_dt
            object_velocity += object_force / m[:, None] * cfg.sub_dt
            finger_position += finger_velocity * cfg.sub_dt
            object_position += object_velocity * cfg.sub_dt
            _reflect(
                finger_position, finger_velocity,
                cfg.arena_half_extent - cfg.finger_radius, cfg.wall_restitution,
            )
            _reflect(
                object_position, object_velocity,
                cfg.arena_half_extent - cfg.object_radius, cfg.wall_restitution,
            )
            reaction = -object_contact_force
            force_sum += reaction
            force_end = reaction
            force_peak = np.maximum(force_peak, magnitude)
            contact_count += active
        noisy_mean = force_sum / cfg.substeps + rng.normal(scale=0.02, size=(n, 2))
        noisy_end = force_end + rng.normal(scale=0.02, size=(n, 2))
        touch[:, step] = np.concatenate(
            [
                noisy_mean,
                force_peak[:, None],
                noisy_end,
                (contact_count / cfg.substeps)[:, None],
                (contact_count > 0)[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        contact[:, step] = contact_count > 0
        states[:, step + 1] = np.concatenate(
            [finger_position, finger_velocity, object_position, object_velocity], axis=1
        ).astype(np.float32)

    shape = (systems, cfg.rollouts_per_system)
    return {
        "states": states.reshape(*shape, cfg.episode_states, 8),
        "actions": actions.reshape(*shape, cfg.episode_states - 1, 2),
        "touch": touch.reshape(*shape, cfg.episode_states - 1, 7),
        "contact": contact.reshape(*shape, cfg.episode_states - 1),
        "mass": mass,
        "gamma": gamma,
        "stiffness": stiffness,
        "system_ids": np.arange(system_offset, system_offset + systems, dtype=np.int64),
    }


def generate_pokeworld(cfg: PokeConfig = PokeConfig()) -> dict[str, dict[str, np.ndarray]]:
    root = np.random.SeedSequence(cfg.seed)
    counts = (cfg.train_systems, cfg.val_systems, cfg.test_systems)
    offsets = np.cumsum((0,) + counts[:-1])
    return {
        split: simulate_split(count, int(offset), sequence, cfg)
        for split, count, offset, sequence in zip(
            ("train", "val", "test"), counts, offsets, root.spawn(3), strict=True
        )
    }


def generate_pokeworld_ood(
    cfg: PokeConfig,
    support_intervals: tuple[tuple[float, float], ...],
    ood_intervals: tuple[tuple[float, float], ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Generate support-distribution train/val plus a sealed OOD split."""
    root = np.random.SeedSequence(cfg.seed)
    counts = (cfg.train_systems, cfg.val_systems, cfg.test_systems)
    offsets = np.cumsum((0,) + counts[:-1])
    intervals = (support_intervals, support_intervals, ood_intervals)
    return {
        split: simulate_split(count, int(offset), sequence, cfg, split_intervals)
        for split, count, offset, sequence, split_intervals in zip(
            ("train", "val", "ood"), counts, offsets, root.spawn(3), intervals, strict=True
        )
    }


def save_pokeworld(
    root: Path,
    dataset: dict[str, dict[str, np.ndarray]],
    cfg: PokeConfig,
    metadata: dict | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"config": asdict(cfg), "splits": {}}
    if metadata is not None:
        manifest["metadata"] = metadata
    for split, payload in dataset.items():
        np.savez_compressed(root / f"{split}.npz", **payload)
        manifest["splits"][split] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in payload.items()
        }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = sha256(encoded).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def render_state(state, *, action=None, width=320, height=240, **kwargs):
    """Human-facing display shared with the manuscript; native dynamics unchanged."""
    from sprii_visuals import render_rgb
    return render_rgb('poke', state=state, action=action, width=width, height=height, **kwargs)
