"""Torch batch construction from the frozen NPZ corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .sampling import (
    sample_pseudo_system_pairs,
    sample_same_rollout_nonoverlap_pairs,
    sample_same_system_pairs,
)


HORIZONS = (1, 4, 16)


@dataclass
class PairedBatch:
    history_states: torch.Tensor
    history_actions: torch.Tensor
    target_states: torch.Tensor
    future_actions: torch.Tensor
    action_masks: torch.Tensor
    gamma: torch.Tensor

    def to(self, device: torch.device) -> "PairedBatch":
        return PairedBatch(**{k: v.to(device, non_blocking=True) for k, v in self.__dict__.items()})


class SplitArrays:
    def __init__(self, data_root: Path, split: str) -> None:
        payload = np.load(data_root / f"{split}.npz", mmap_mode="r")
        self.states = payload["states"]
        self.actions = payload["actions"]
        self.gamma = payload["gamma"]

    def paired_batch(
        self,
        pairs: int,
        seed: int,
        pairing_mode: str = "same_system",
        pseudo_systems: np.ndarray | None = None,
    ) -> PairedBatch:
        if pairing_mode == "same_system":
            legacy = sample_same_system_pairs(
                self.states.shape[0], self.states.shape[1], pairs=pairs, seed=seed
            )
            specs = np.stack(
                [legacy[:, 0], legacy[:, 1], legacy[:, 2], legacy[:, 0], legacy[:, 3], legacy[:, 4]],
                axis=1,
            )
        elif pairing_mode == "same_rollout_nonoverlap":
            legacy = sample_same_rollout_nonoverlap_pairs(
                self.states.shape[0], self.states.shape[1], pairs=pairs, seed=seed
            )
            specs = np.stack(
                [legacy[:, 0], legacy[:, 1], legacy[:, 2], legacy[:, 0], legacy[:, 3], legacy[:, 4]],
                axis=1,
            )
        elif pairing_mode == "random_system_fixed":
            if pseudo_systems is None:
                raise ValueError("random_system_fixed requires a frozen pseudo-system mapping")
            specs = sample_pseudo_system_pairs(pseudo_systems, pairs=pairs, seed=seed)
        else:
            raise ValueError(f"unknown pairing mode: {pairing_mode}")
        windows = []
        for system_col, rollout_col, anchor_col in ((0, 1, 2), (3, 4, 5)):
            systems = specs[:, system_col]
            rollouts = specs[:, rollout_col]
            anchors = specs[:, anchor_col]
            row = np.arange(pairs)[:, None]
            history_state_time = anchors[:, None] + np.arange(-23, 1)[None, :]
            history_action_time = anchors[:, None] + np.arange(-23, 0)[None, :]
            target_time = anchors[:, None] + np.asarray(HORIZONS)[None, :]
            history_states = self.states[systems[:, None], rollouts[:, None], history_state_time]
            history_actions = self.actions[systems[:, None], rollouts[:, None], history_action_time]
            targets = self.states[systems[:, None], rollouts[:, None], target_time]
            future = np.zeros((pairs, 3, 16, 2), dtype=np.float32)
            masks = np.zeros((pairs, 3, 16), dtype=np.float32)
            for hi, horizon in enumerate(HORIZONS):
                future_time = anchors[:, None] + np.arange(horizon)[None, :]
                future[:, hi, :horizon] = self.actions[
                    systems[:, None], rollouts[:, None], future_time
                ]
                masks[:, hi, :horizon] = 1.0
            windows.append(
                tuple(
                    map(
                        np.asarray,
                        (history_states, history_actions, targets, future, masks, self.gamma[systems]),
                    )
                )
            )
        merged = [np.concatenate([windows[0][i], windows[1][i]], axis=0) for i in range(6)]
        return PairedBatch(*(torch.from_numpy(x.copy()) for x in merged))
