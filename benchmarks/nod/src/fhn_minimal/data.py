"""Memory-resident FHN dataset preserving the released DR2D sampling rule."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import io as sio
from torch.utils.data import Dataset


TRAIN_SYSTEMS = ((0.03, 0.10), (0.02, 0.15), (0.04, 0.15),
                 (0.01, 0.20), (0.03, 0.20), (0.05, 0.20),
                 (0.02, 0.25), (0.04, 0.25), (0.03, 0.30))
OOD_INTRA_SYSTEMS = ((0.03, 0.15), (0.02, 0.20),
                     (0.04, 0.20), (0.03, 0.25))
OOD_EXTRA_SYSTEMS = ((0.01, 0.10), (0.02, 0.10), (0.04, 0.10), (0.05, 0.10),
                     (0.01, 0.15), (0.05, 0.15), (0.01, 0.25), (0.05, 0.25),
                     (0.01, 0.30), (0.02, 0.30), (0.04, 0.30), (0.05, 0.30))
SYSTEMS = {
    "train": TRAIN_SYSTEMS,
    "ood-intra": OOD_INTRA_SYSTEMS,
    "ood-extra": OOD_EXTRA_SYSTEMS,
}
CONDITION_FRAMES = (0, 25, 50, 75, 100)
HEAD_STEPS = np.asarray((1, 3, 9, 27), dtype=np.int64)
SEQUENCES = np.asarray([
    [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 0, 2, 2],
    [0, 0, 2, 3], [0, 1, 1, 2], [0, 1, 1, 3], [0, 1, 2, 2],
    [0, 1, 2, 3], [0, 2, 2, 3], [1, 1, 2, 2], [1, 1, 2, 3],
    [1, 2, 2, 3], [1, 1, 0, 0], [2, 1, 0, 0], [3, 1, 0, 0],
    [2, 2, 0, 0], [3, 2, 0, 0], [2, 1, 1, 0], [3, 1, 1, 0],
    [2, 2, 1, 0], [3, 2, 1, 0], [3, 2, 2, 0], [2, 2, 1, 1],
    [3, 2, 1, 1], [3, 2, 2, 1],
], dtype=np.int64)
OFFSETS = np.cumsum(HEAD_STEPS[SEQUENCES], axis=1)
LATEST_START = 100 - OFFSETS[:, -1]
SHIFTS = ((32, 32), (32, 96), (96, 32), (96, 96))


def load_trajectory(data_dir: Path, system: tuple[float, float], initial_id: int) -> torch.Tensor:
    k, beta = system
    fields = []
    for name in ("U", "V"):
        path = data_dir / f"{name}_{k:.2f}_{beta:.2f}_{initial_id}.mat"
        array = sio.loadmat(path)[f"{name}_record"]
        if array.shape != (101, 128, 128) or array.dtype != np.float32:
            raise ValueError(f"invalid trajectory {path}: {array.shape} {array.dtype}")
        fields.append(torch.from_numpy(array))
    return torch.stack(fields, dim=1)  # time, field, x, y


def transform_trajectory(x: torch.Tensor, augmentation: int) -> torch.Tensor:
    """Match the ten transformations in the released DR2D.readData implementation."""
    if augmentation == 0:
        return x
    if augmentation == 1:
        return x.transpose(-2, -1)
    if 2 <= augmentation <= 5:
        return torch.rot90(x, augmentation - 1, (-2, -1))
    if 6 <= augmentation <= 9:
        return torch.roll(x, shifts=SHIFTS[augmentation - 6], dims=(-2, -1))
    raise ValueError(f"augmentation must be in [0,9], got {augmentation}")


class FHNHierDataset(Dataset):
    """Released hierarchy with one deterministic independent same-system pair.

    The second conditioning trajectory does not consume Python RNG. Consequently,
    the random start and primary-condition draws are exactly the same when the
    caller switches between NOD and lambda_align=0 SPRII.
    """

    def __init__(
        self,
        data_dir: str | Path,
        initial_ids: Iterable[int] = range(50, 90),
        split: str = "train",
        max_systems: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.initial_ids = tuple(int(x) for x in initial_ids)
        self.systems = tuple(SYSTEMS[split][:max_systems])
        if len(self.initial_ids) < 2:
            raise ValueError("at least two initial conditions are required for an independent pair")
        self.num_init = len(self.initial_ids)
        self.num_systems = len(self.systems)
        self.num_sequences = len(SEQUENCES)
        self.base = torch.empty(
            self.num_init, self.num_systems, 101, 2, 128, 128, dtype=torch.float32
        )
        for i, initial_id in enumerate(self.initial_ids):
            for s, system in enumerate(self.systems):
                self.base[i, s].copy_(load_trajectory(self.data_dir, system, initial_id))

    def __len__(self) -> int:
        return self.num_init * 10 * self.num_systems * self.num_sequences

    def __getitem__(self, index: int):
        samples_per_sequence = self.num_init * 10 * self.num_systems
        sequence_index = index // samples_per_sequence
        within = index % samples_per_sequence
        system_index = within % self.num_systems
        augmented_init = within // self.num_systems
        base_index = augmented_init % self.num_init
        augmentation = augmented_init // self.num_init

        trajectory = self.base[base_index, system_index]
        start = random.randint(0, int(LATEST_START[sequence_index]))
        offsets = torch.from_numpy(OFFSETS[sequence_index])
        # Spatial transforms commute with time indexing. Selecting the five
        # required frames first avoids transforming all 101 frames per item.
        x_input = transform_trajectory(trajectory[start], augmentation)
        y = transform_trajectory(trajectory[start + offsets], augmentation)
        indicators = torch.from_numpy(SEQUENCES[sequence_index])

        condition_index = random.randint(0, self.num_init * 10 - 1)
        condition_base = condition_index % self.num_init
        condition_aug = condition_index // self.num_init
        condition_1 = transform_trajectory(
            self.base[condition_base, system_index][list(CONDITION_FRAMES)], condition_aug
        )
        # Different physical trajectory, same symmetry transform, no extra RNG draw.
        condition_base_2 = (condition_base + 1) % self.num_init
        condition_2 = transform_trajectory(
            self.base[condition_base_2, system_index][list(CONDITION_FRAMES)], condition_aug
        )
        parameters = torch.tensor(self.systems[system_index], dtype=torch.float32)
        return x_input, condition_1, condition_2, y, indicators, parameters


def one_hot_head(indices: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.long(), num_classes=4).float().unsqueeze(-1)
