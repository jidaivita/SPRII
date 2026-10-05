"""Window and donor pairing utilities with explicit system boundaries."""

from __future__ import annotations

import numpy as np


VALID_ANCHORS = np.arange(23, 48, dtype=np.int64)


def sample_same_system_pairs(
    num_systems: int,
    rollouts_per_system: int,
    pairs: int,
    seed: int,
) -> np.ndarray:
    if rollouts_per_system < 2:
        raise ValueError("same-system pairing requires at least two rollouts")
    rng = np.random.default_rng(seed)
    systems = rng.choice(num_systems, size=pairs, replace=num_systems < pairs)
    rollout_a = rng.integers(rollouts_per_system, size=pairs)
    offset = rng.integers(1, rollouts_per_system, size=pairs)
    rollout_b = (rollout_a + offset) % rollouts_per_system
    anchors_a = rng.choice(VALID_ANCHORS, size=pairs)
    anchors_b = rng.choice(VALID_ANCHORS, size=pairs)
    return np.stack([systems, rollout_a, anchors_a, rollout_b, anchors_b], axis=1)


def sample_same_rollout_nonoverlap_pairs(
    num_systems: int,
    rollouts_per_system: int,
    pairs: int,
    seed: int,
) -> np.ndarray:
    """Sample the only two disjoint 24-state histories with legal h16 targets.

    For a 64-state rollout and legal anchors 23..47, histories anchored at 23
    and 47 cover states 0..23 and 24..47 respectively.  Randomly swapping the
    two branches avoids assigning a fixed temporal role to either VICReg arm.
    """
    rng = np.random.default_rng(seed)
    systems = rng.choice(num_systems, size=pairs, replace=num_systems < pairs)
    rollouts = rng.integers(rollouts_per_system, size=pairs)
    swap = rng.integers(2, size=pairs).astype(bool)
    anchors_a = np.where(swap, 47, 23)
    anchors_b = np.where(swap, 23, 47)
    return np.stack([systems, rollouts, anchors_a, rollouts, anchors_b], axis=1)


def build_collision_free_pseudo_systems(
    num_systems: int,
    rollouts_per_system: int,
    seed: int,
) -> np.ndarray:
    """Create a fixed, balanced wrong relation with no within-row collision."""
    if num_systems < rollouts_per_system:
        raise ValueError("collision-free pseudo systems require systems >= rollouts")
    rng = np.random.default_rng(seed)
    base = rng.permutation(num_systems)
    mapping = np.stack([np.roll(base, shift) for shift in range(rollouts_per_system)], axis=1)
    if any(np.unique(row).size != rollouts_per_system for row in mapping):
        raise AssertionError("pseudo-system construction produced a collision")
    if any(np.unique(mapping[:, column]).size != num_systems for column in range(rollouts_per_system)):
        raise AssertionError("pseudo-system column is not a permutation")
    return mapping


def sample_pseudo_system_pairs(
    pseudo_systems: np.ndarray,
    pairs: int,
    seed: int,
) -> np.ndarray:
    """Sample two rollout columns from a frozen pseudo-system relation."""
    num_systems, rollouts_per_system = pseudo_systems.shape
    if rollouts_per_system < 2:
        raise ValueError("pseudo-system pairing requires at least two rollouts")
    rng = np.random.default_rng(seed)
    pseudo = rng.choice(num_systems, size=pairs, replace=num_systems < pairs)
    rollout_a = rng.integers(rollouts_per_system, size=pairs)
    offset = rng.integers(1, rollouts_per_system, size=pairs)
    rollout_b = (rollout_a + offset) % rollouts_per_system
    systems_a = pseudo_systems[pseudo, rollout_a]
    systems_b = pseudo_systems[pseudo, rollout_b]
    anchors_a = rng.choice(VALID_ANCHORS, size=pairs)
    anchors_b = rng.choice(VALID_ANCHORS, size=pairs)
    return np.stack(
        [systems_a, rollout_a, anchors_a, systems_b, rollout_b, anchors_b], axis=1
    )


def deterministic_derangement(n: int, seed: int) -> np.ndarray:
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, n))
    return (np.arange(n) + shift) % n
