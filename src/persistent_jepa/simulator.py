"""Deterministic D-Clean simulator and system-disjoint dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DCleanConfig:
    dt: float = 0.05
    episode_states: int = 64
    rollouts_per_system: int = 8
    train_systems: int = 1000
    val_systems: int = 200
    test_systems: int = 200
    gamma_low: float = 0.5
    gamma_high: float = 4.0
    mass: float = 1.0
    position_low: float = -1.0
    position_high: float = 1.0
    speed_low: float = 0.0
    speed_high: float = 2.0
    segment_steps_min: int = 3
    segment_steps_max: int = 8
    zero_segment_probability: float = 0.30
    force_magnitude_max: float = 4.0
    seed: int = 20260818

    @property
    def transition_actions(self) -> int:
        return self.episode_states - 1


@dataclass
class DCleanDataset:
    states: dict[Split, np.ndarray]
    actions: dict[Split, np.ndarray]
    gamma: dict[Split, np.ndarray]
    system_ids: dict[Split, np.ndarray]
    rollout_seeds: dict[Split, np.ndarray]
    config: DCleanConfig

    def manifest(self) -> dict:
        arrays = {}
        for split in ("train", "val", "test"):
            arrays[split] = {
                "states_shape": list(self.states[split].shape),
                "actions_shape": list(self.actions[split].shape),
                "gamma_shape": list(self.gamma[split].shape),
                "system_id_min": int(self.system_ids[split].min()),
                "system_id_max": int(self.system_ids[split].max()),
                "states_sha256": _array_hash(self.states[split]),
                "actions_sha256": _array_hash(self.actions[split]),
                "gamma_sha256": _array_hash(self.gamma[split]),
                "rollout_seeds_sha256": _array_hash(self.rollout_seeds[split]),
            }
        payload = {"config": asdict(self.config), "arrays": arrays}
        payload["manifest_sha256"] = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload

    def save(self, root: Path) -> dict:
        root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            np.savez_compressed(
                root / f"{split}.npz",
                states=self.states[split],
                actions=self.actions[split],
                gamma=self.gamma[split],
                system_ids=self.system_ids[split],
                rollout_seeds=self.rollout_seeds[split],
            )
        manifest = self.manifest()
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def _array_hash(x: np.ndarray) -> str:
    h = sha256()
    h.update(str(x.dtype).encode())
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def exact_step(
    state: np.ndarray, force: np.ndarray, gamma: float, dt: float, mass: float = 1.0
) -> np.ndarray:
    """Exact update for dv/dt=F/m-gamma*v under constant F."""
    state64 = np.asarray(state, dtype=np.float64)
    force64 = np.asarray(force, dtype=np.float64)
    position, velocity = state64[..., :2], state64[..., 2:]
    alpha = np.exp(-gamma * dt)
    one_minus = 1.0 - alpha
    next_velocity = alpha * velocity + force64 * one_minus / (mass * gamma)
    next_position = (
        position
        + velocity * one_minus / gamma
        + force64 / (mass * gamma) * (dt - one_minus / gamma)
    )
    return np.concatenate([next_position, next_velocity], axis=-1)


def _initial_state(rng: np.random.Generator, cfg: DCleanConfig) -> np.ndarray:
    position = rng.uniform(cfg.position_low, cfg.position_high, size=2)
    speed = rng.uniform(cfg.speed_low, cfg.speed_high)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)])
    return np.concatenate([position, velocity])


def _actions(rng: np.random.Generator, cfg: DCleanConfig) -> np.ndarray:
    actions = np.zeros((cfg.transition_actions, 2), dtype=np.float64)
    cursor = 0
    while cursor < cfg.transition_actions:
        length = int(rng.integers(cfg.segment_steps_min, cfg.segment_steps_max + 1))
        end = min(cursor + length, cfg.transition_actions)
        if rng.random() >= cfg.zero_segment_probability:
            magnitude = rng.uniform(0.0, cfg.force_magnitude_max)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            actions[cursor:end] = magnitude * np.array([np.cos(angle), np.sin(angle)])
        cursor = end
    return actions


def generate_rollout(
    gamma: float, rollout_seed: int, cfg: DCleanConfig
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rollout_seed)
    actions = _actions(rng, cfg)
    states = np.empty((cfg.episode_states, 4), dtype=np.float64)
    states[0] = _initial_state(rng, cfg)
    for t in range(cfg.transition_actions):
        states[t + 1] = exact_step(states[t], actions[t], gamma, cfg.dt, cfg.mass)
    return states.astype(np.float32), actions.astype(np.float32)


def generate_dataset(cfg: DCleanConfig = DCleanConfig()) -> DCleanDataset:
    root = np.random.SeedSequence(cfg.seed)
    split_counts = {"train": cfg.train_systems, "val": cfg.val_systems, "test": cfg.test_systems}
    split_sequences = root.spawn(3)
    states, actions, gammas, ids, seeds = {}, {}, {}, {}, {}
    system_offset = 0
    for split, split_ss in zip(("train", "val", "test"), split_sequences, strict=True):
        count = split_counts[split]
        system_ss = split_ss.spawn(count)
        split_states = np.empty(
            (count, cfg.rollouts_per_system, cfg.episode_states, 4), dtype=np.float32
        )
        split_actions = np.empty(
            (count, cfg.rollouts_per_system, cfg.transition_actions, 2), dtype=np.float32
        )
        split_gamma = np.empty(count, dtype=np.float32)
        split_seeds = np.empty((count, cfg.rollouts_per_system), dtype=np.uint64)
        for i, ss in enumerate(system_ss):
            gamma_ss, *rollout_ss = ss.spawn(cfg.rollouts_per_system + 1)
            gamma_rng = np.random.default_rng(gamma_ss)
            gamma = float(gamma_rng.uniform(cfg.gamma_low, cfg.gamma_high))
            split_gamma[i] = gamma
            for r, rss in enumerate(rollout_ss):
                rollout_seed = int(rss.generate_state(1, dtype=np.uint64)[0])
                split_seeds[i, r] = rollout_seed
                split_states[i, r], split_actions[i, r] = generate_rollout(
                    gamma, rollout_seed, cfg
                )
        states[split] = split_states
        actions[split] = split_actions
        gammas[split] = split_gamma
        ids[split] = np.arange(system_offset, system_offset + count, dtype=np.int64)
        seeds[split] = split_seeds
        system_offset += count
    return DCleanDataset(states, actions, gammas, ids, seeds, cfg)



def render_state(state, *, action=None, width=320, height=240, **kwargs):
    """Human-facing display shared with the manuscript; native dynamics unchanged."""
    from sprii_visuals import render_rgb
    return render_rgb('dclean', state=state, action=action, width=width, height=height, **kwargs)
