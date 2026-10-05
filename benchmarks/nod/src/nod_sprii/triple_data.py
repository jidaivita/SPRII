"""Three-view Burgers loader with matched observation budget.

The base A->B NOD pair is kept unchanged. A third same-system trajectory C is
added through a deterministic wrapper so NOD, SPRII, and SPRII-Random can all
read the same A/B/C tuple while only the relation term changes.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ngs.utils import BurgersPairedDataset


class BurgersTripleDataset(Dataset):
    def __init__(self, base: BurgersPairedDataset, seed: int = 0) -> None:
        self.base = base
        self.seed = int(seed)
        self.n_t = base.n_t
        self.n_x = base.n_x
        self.random_relation = False
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    def describe(self) -> dict[str, Any]:
        d = dict(self.base.describe())
        d["alignment_view"] = "independent_same_system_C"
        d["alignment_sampling"] = "deterministic_by_sample_index"
        return d

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        shard_idx, cond_case_idx = self.base.sample_index[index]
        shard = self.base.shards[shard_idx]
        pred_case_idx = int(sample["pred_case_idx"])
        candidates = [
            int(c) for c in shard["allowed_cases"]
            if int(c) not in {int(cond_case_idx), pred_case_idx}
        ]
        if not candidates:
            # This branch is only possible for a shard with fewer than three
            # legal cases; it remains explicit so a run manifest can record it.
            candidates = [int(c) for c in shard["allowed_cases"] if int(c) != int(cond_case_idx)]
        if not candidates:
            raise RuntimeError("No legal independent alignment trajectory C is available.")
        rng = np.random.default_rng(self.seed + 1000003 * int(index))
        c_case_idx = candidates[int(rng.integers(0, len(candidates)))]
        donor_shard_idx = shard_idx
        if self.random_relation:
            # A fixed, seeded system cycle permutes all C assignments over the
            # full epoch. Every system occurs equally often and none maps to itself.
            order = np.random.default_rng(self.seed + 7919 * self.epoch).permutation(len(self.base.shards))
            position = int(np.flatnonzero(order == shard_idx)[0])
            donor_shard_idx = int(order[(position + 1) % len(order)])
            assert donor_shard_idx != shard_idx
            donor_cases = self.base.shards[donor_shard_idx]["allowed_cases"]
            assert len(donor_cases) == len(shard["allowed_cases"])
            c_case_idx = int(donor_cases[list(shard["allowed_cases"]).index(c_case_idx)])
        c_np = self.base._load_trunk_ds(shard_idx=donor_shard_idx, case_idx=c_case_idx)
        sample["align_system_idx"] = int(donor_shard_idx)
        sample["recipient_system_idx"] = int(shard_idx)
        sample["sample_id"] = int(index)
        sample["align_u"] = torch.from_numpy(np.ascontiguousarray(c_np)).unsqueeze(0)
        sample["align_case_idx"] = int(c_case_idx)
        sample["observation_budget"] = "A+B+C"
        return sample


def _loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool,
            drop_last: bool, pin_memory: bool) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=drop_last,
                      persistent_workers=(num_workers > 0), **kwargs)


def create_burgers_triple_dataloaders(
    data_root: str,
    output_add_root: Optional[str],
    batch_size: int,
    num_workers: int,
    cache_mode: str,
    prediction_horizon: int,
    seed: int,
    include_output_add_train: bool = False,
    include_test: bool = False,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    roots = {
        "train": BurgersTripleDataset(BurgersPairedDataset(
            data_root=data_root, split="train",
            include_output_add_train=include_output_add_train,
            output_add_root=output_add_root,
            prediction_horizon=prediction_horizon, cache_mode=cache_mode, seed=seed), seed=seed),
        "eval": BurgersTripleDataset(BurgersPairedDataset(
            data_root=data_root, split="eval", prediction_horizon=prediction_horizon,
            cache_mode=cache_mode, seed=seed), seed=seed + 1),
    }
    if include_test:
        roots["test"] = BurgersTripleDataset(BurgersPairedDataset(
            data_root=data_root, split="test", prediction_horizon=prediction_horizon,
            cache_mode=cache_mode, seed=seed), seed=seed + 2)
    return {
        "train": _loader(roots["train"], batch_size, num_workers, True, False, pin_memory),
        "eval": _loader(roots["eval"], batch_size, num_workers, False, False, pin_memory),
        **({"test": _loader(roots["test"], batch_size, num_workers, False, False, pin_memory)}
           if include_test else {}),
    }
