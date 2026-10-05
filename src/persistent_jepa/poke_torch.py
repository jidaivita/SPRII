"""Torch sampling for the frozen R0 PokeWorld reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .torch_data import HORIZONS


def legal_poke_anchors(history_length: int = 24) -> np.ndarray:
    """Legal query anchors for current-frame + temporal-difference inputs.

    A length-L visual history ending at t contains current states
    [t-L+1, ..., t] and needs the preceding states [t-L, ..., t-1] to
    construct the temporal-difference channel.  Hence t >= L, not L-1.
    The h=16 target additionally requires t <= 47 in a 64-state episode.
    """
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    return np.arange(history_length, 48, dtype=np.int64)


POKE_ANCHORS = legal_poke_anchors()


@dataclass
class PokeBatch:
    history_current: torch.Tensor
    history_previous: torch.Tensor
    history_actions: torch.Tensor
    target_current: torch.Tensor
    target_previous: torch.Tensor
    future_actions: torch.Tensor
    action_masks: torch.Tensor
    mass: torch.Tensor
    gamma: torch.Tensor
    stiffness: torch.Tensor
    system_index: torch.Tensor
    rollout_id: torch.Tensor
    anchor: torch.Tensor

    def to(self, device: torch.device) -> "PokeBatch":
        return PokeBatch(**{key: value.to(device, non_blocking=True) for key, value in self.__dict__.items()})


class PokeSplit:
    def __init__(self, root: Path, split: str, history_length: int = 24) -> None:
        data = np.load(root / f"{split}.npz", mmap_mode="r")
        self.states = data["states"]
        self.actions = data["actions"]
        self.mass = data["mass"]
        self.gamma = data["gamma"]
        self.stiffness = data["stiffness"]
        self.history_length = int(history_length)
        self.anchors = legal_poke_anchors(self.history_length)
        if self.anchors.size == 0:
            raise ValueError(
                f"history_length={self.history_length} has no legal h16 anchors "
                "under the temporal-difference input convention"
            )

    def batch(self, windows: int, seed: int) -> PokeBatch:
        rng = np.random.default_rng(seed)
        systems = rng.integers(self.states.shape[0], size=windows)
        rollouts = rng.integers(self.states.shape[1], size=windows)
        anchors = rng.choice(self.anchors, size=windows)
        return self._from_indices(systems, rollouts, anchors)

    def paired_batch(self, pairs: int, seed: int) -> PokeBatch:
        rng = np.random.default_rng(seed)
        replace = pairs > self.states.shape[0]
        systems = rng.choice(self.states.shape[0], size=pairs, replace=replace)
        rollout_a = rng.integers(self.states.shape[1], size=pairs)
        offset = rng.integers(1, self.states.shape[1], size=pairs)
        rollout_b = (rollout_a + offset) % self.states.shape[1]
        anchor_a = rng.choice(self.anchors, size=pairs)
        anchor_b = rng.choice(self.anchors, size=pairs)
        a = self._from_indices(systems, rollout_a, anchor_a)
        b = self._from_indices(systems, rollout_b, anchor_b)
        return PokeBatch(
            **{
                key: torch.cat([getattr(a, key), getattr(b, key)], dim=0)
                for key in a.__dict__
            }
        )

    def restricted_paired_batch(
        self,
        pairs: int,
        seed: int,
        donor_mode: str,
        min_separation: int = 12,
    ) -> PokeBatch:
        """Matched restricted-anchor pairs for the independent-rollout control.

        Both modes use identical system, query-anchor, donor-anchor, and target
        distributions.  They differ only in whether the donor rollout is the
        query rollout or another rollout from the same system.  Donors always
        precede queries, so no query-future information is used.
        """
        if donor_mode not in {"same_rollout", "independent_rollout"}:
            raise ValueError(f"unknown donor_mode {donor_mode}")
        query_anchors = self.anchors[self.anchors >= self.anchors[0] + min_separation]
        if query_anchors.size == 0:
            raise ValueError("no restricted query anchors for requested separation")
        rng = np.random.default_rng(seed)
        replace = pairs > self.states.shape[0]
        systems = rng.choice(self.states.shape[0], size=pairs, replace=replace)
        query_rollout = rng.integers(self.states.shape[1], size=pairs)
        query_anchor = rng.choice(query_anchors, size=pairs)
        donor_anchor = np.asarray(
            [rng.integers(self.anchors[0], anchor - min_separation + 1) for anchor in query_anchor],
            dtype=np.int64,
        )
        if donor_mode == "same_rollout":
            donor_rollout = query_rollout.copy()
        else:
            offset = rng.integers(1, self.states.shape[1], size=pairs)
            donor_rollout = (query_rollout + offset) % self.states.shape[1]
        donor = self._from_indices(systems, donor_rollout, donor_anchor)
        query = self._from_indices(systems, query_rollout, query_anchor)
        return PokeBatch(
            **{
                key: torch.cat([getattr(donor, key), getattr(query, key)], dim=0)
                for key in donor.__dict__
            }
        )

    def fixed_system_windows(
        self,
        windows_per_system: int,
        seed: int,
        anchor_min: int | None = None,
    ) -> tuple[PokeBatch, np.ndarray]:
        rng = np.random.default_rng(seed)
        systems = np.repeat(np.arange(self.states.shape[0]), windows_per_system)
        rollouts = rng.integers(self.states.shape[1], size=systems.size)
        anchor_pool = self.anchors[self.anchors >= anchor_min] if anchor_min is not None else self.anchors
        if anchor_pool.size == 0:
            raise ValueError(f"no legal anchors at or above {anchor_min}")
        anchors = rng.choice(anchor_pool, size=systems.size)
        return self._from_indices(systems, rollouts, anchors), systems

    def _from_indices(
        self, systems: np.ndarray, rollouts: np.ndarray, anchors: np.ndarray
    ) -> PokeBatch:
        current_time = anchors[:, None] + np.arange(-(self.history_length - 1), 1)[None, :]
        previous_time = current_time - 1
        history_action_time = anchors[:, None] + np.arange(
            -(self.history_length - 1), 0
        )[None, :]
        target_time = anchors[:, None] + np.asarray(HORIZONS)[None, :]
        target_previous_time = target_time - 1
        history_current = self.states[systems[:, None], rollouts[:, None], current_time]
        history_previous = self.states[systems[:, None], rollouts[:, None], previous_time]
        history_actions = self.actions[systems[:, None], rollouts[:, None], history_action_time]
        target_current = self.states[systems[:, None], rollouts[:, None], target_time]
        target_previous = self.states[
            systems[:, None], rollouts[:, None], target_previous_time
        ]
        future = np.zeros((systems.size, 3, 16, 2), dtype=np.float32)
        masks = np.zeros((systems.size, 3, 16), dtype=np.float32)
        for hi, horizon in enumerate(HORIZONS):
            time = anchors[:, None] + np.arange(horizon)[None, :]
            future[:, hi, :horizon] = self.actions[systems[:, None], rollouts[:, None], time]
            masks[:, hi, :horizon] = 1.0
        arrays = (
            history_current, history_previous, history_actions,
            target_current, target_previous, future, masks,
            self.mass[systems], self.gamma[systems], self.stiffness[systems],
            systems.astype(np.int64), rollouts.astype(np.int64), anchors.astype(np.int64),
        )
        return PokeBatch(*(torch.from_numpy(np.asarray(value).copy()) for value in arrays))
