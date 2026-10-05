"""Frozen RH20T cache reader and causal relation samplers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .rh20t_model import HORIZONS


@dataclass
class RH20TBatch:
    history_image: torch.Tensor | None
    history_lowdim: torch.Tensor
    history_actions: torch.Tensor
    donor_history_image: torch.Tensor | None
    donor_history_lowdim: torch.Tensor | None
    donor_history_actions: torch.Tensor | None
    target_image: torch.Tensor | None
    target_lowdim: torch.Tensor
    target_force: torch.Tensor
    target_tcp_xyz: torch.Tensor
    future_actions: torch.Tensor
    action_masks: torch.Tensor
    task_index: torch.Tensor
    anchor: torch.Tensor
    donor_anchor: torch.Tensor | None

    def to(self, device: torch.device) -> "RH20TBatch":
        values = {}
        for key, value in self.__dict__.items():
            values[key] = value.to(device, non_blocking=True) if value is not None else None
        return RH20TBatch(**values)


class RH20TNormalization:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    @classmethod
    def from_path(cls, path: Path) -> "RH20TNormalization":
        return cls(json.loads(path.read_text()))

    def apply(self, name: str, value: np.ndarray) -> np.ndarray:
        stats = self.payload[name]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return (value.astype(np.float32) - mean) / np.maximum(std, 1e-6)


class RH20TSplit:
    def __init__(
        self,
        cache_root: Path,
        split_manifest: Path,
        pairing_manifest: Path,
        normalization: Path,
        split: str,
        condition: str,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {split}")
        self.cache_root = cache_root
        self.condition = condition
        self.lowdim_only = "LowDim" in condition
        split_data = json.loads(split_manifest.read_text())["splits"][split]
        pair_data = json.loads(pairing_manifest.read_text())["splits"][split]
        self.episode_ids = set(split_data["episode_ids"])
        self.entries = pair_data["entries"]
        if {item["query_episode_id"] for item in self.entries} != self.episode_ids:
            raise ValueError("pairing and split manifests disagree")
        self.by_task: dict[str, list[dict]] = {}
        for entry in self.entries:
            self.by_task.setdefault(entry["task_id"], []).append(entry)
        self.tasks = sorted(self.by_task)
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        self.normalization = RH20TNormalization.from_path(normalization)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _episode(self, episode_id: str) -> dict[str, np.ndarray]:
        if episode_id not in self._cache:
            path = self.cache_root / f"{episode_id}.npz"
            with np.load(path) as data:
                self._cache[episode_id] = {key: np.asarray(data[key]) for key in data.files}
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[episode_id]

    def _image_window(self, episode: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
        gray = episode["rgb_gray"][indices].astype(np.float32) / 255.0
        diff = np.zeros_like(gray)
        diff[1:] = gray[1:] - gray[:-1]
        channels = np.stack([gray, diff], axis=1)
        mean = np.asarray(self.normalization.payload["image_channels"]["mean"], dtype=np.float32)
        std = np.asarray(self.normalization.payload["image_channels"]["std"], dtype=np.float32)
        return (channels - mean[None, :, None, None]) / np.maximum(
            std[None, :, None, None], 1e-6
        )

    def _target_image(self, episode: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
        current = episode["rgb_gray"][indices].astype(np.float32) / 255.0
        previous = episode["rgb_gray"][indices - 1].astype(np.float32) / 255.0
        channels = np.stack([current, current - previous], axis=1)
        mean = np.asarray(self.normalization.payload["image_channels"]["mean"], dtype=np.float32)
        std = np.asarray(self.normalization.payload["image_channels"]["std"], dtype=np.float32)
        return (channels - mean[None, :, None, None]) / np.maximum(
            std[None, :, None, None], 1e-6
        )

    def _window(
        self, episode_id: str, anchor: int, *, include_targets: bool = True
    ) -> dict[str, np.ndarray | None]:
        episode = self._episode(episode_id)
        history_indices = np.arange(anchor - 23, anchor + 1, dtype=np.int64)
        ft_history = self.normalization.apply("force", episode["ft_base_zeroed"][history_indices])
        tcp_history = self.normalization.apply("tcp", episode["tcp_base"][history_indices])
        action_history = self.normalization.apply(
            "action", episode["gripper_command_width"][history_indices]
        )
        history_lowdim = np.concatenate([ft_history, tcp_history, action_history], axis=-1)
        result: dict[str, np.ndarray | None] = {
            "history_image": None
            if self.lowdim_only
            else self._image_window(episode, history_indices),
            "history_lowdim": history_lowdim,
            "history_actions": action_history[:-1],
        }
        if not include_targets:
            return result
        target_indices = anchor + np.asarray(HORIZONS, dtype=np.int64)
        ft_target = self.normalization.apply("force", episode["ft_base_zeroed"][target_indices])
        tcp_target = self.normalization.apply("tcp", episode["tcp_base"][target_indices])
        action_target = self.normalization.apply(
            "action", episode["gripper_command_width"][target_indices]
        )
        target_lowdim = np.concatenate([ft_target, tcp_target, action_target], axis=-1)
        future = np.zeros((len(HORIZONS), 16, 1), dtype=np.float32)
        masks = np.zeros((len(HORIZONS), 16), dtype=np.float32)
        action = self.normalization.apply("action", episode["gripper_command_width"])
        for hi, horizon in enumerate(HORIZONS):
            future[hi, :horizon] = action[anchor : anchor + horizon]
            masks[hi, :horizon] = 1.0
        result.update({
            "target_image": None
            if self.lowdim_only
            else self._target_image(episode, target_indices),
            "target_lowdim": target_lowdim,
            "target_force": ft_target,
            "target_tcp_xyz": tcp_target[:, :3],
            "future_actions": future,
            "action_masks": masks,
        })
        return result

    def _sample_entry(self, rng: np.random.Generator) -> dict:
        task = self.tasks[int(rng.integers(len(self.tasks)))]
        entries = self.by_task[task]
        return entries[int(rng.integers(len(entries)))]

    def _donor_episode(self, entry: dict, anchor: int, rng: np.random.Generator) -> str:
        if self.condition in {"Bx-SameEp", "B3-SameEp"}:
            return entry["query_episode_id"]
        if self.condition in {"Bx-Indep", "B3-Indep", "B3-LowDim-Same"}:
            candidates = entry["independent_candidate_episode_ids"]
        elif self.condition in {"B3-Random", "B3-LowDim-Random"}:
            candidates = entry["random_candidate_episode_ids"]
        else:
            raise ValueError(f"condition {self.condition} has no donor")
        donor_end = anchor - 24
        # Do not decompress every episode in a task just to inspect its length.
        # A fixed RNG permutation preserves unbiased candidate selection while
        # stopping at the first legal history-only donor.  The old all-candidate
        # scan made a single B3 step load roughly a thousand compressed files.
        for index in rng.permutation(len(candidates)):
            episode_id = candidates[int(index)]
            if len(self._episode(episode_id)["rgb_gray"]) > donor_end:
                return episode_id
        raise RuntimeError("frozen relation has no donor at sampled anchor")

    @staticmethod
    def _stack(windows: list[dict], name: str) -> torch.Tensor | None:
        if windows[0][name] is None:
            return None
        return torch.from_numpy(np.stack([item[name] for item in windows]).astype(np.float32))

    def batch(self, pairs: int, seed: int) -> RH20TBatch:
        rng = np.random.default_rng(seed)
        donor_windows, query_windows = [], []
        task_indices, anchors, donor_anchors = [], [], []
        for _ in range(pairs):
            entry = self._sample_entry(rng)
            anchor = int(rng.integers(entry["query_anchor_min"], entry["query_anchor_max"] + 1))
            query = self._window(entry["query_episode_id"], anchor)
            if self.condition == "B0":
                query_windows.append(query)
                task_indices.append(self.task_to_index[entry["task_id"]])
                anchors.append(anchor)
                continue
            donor_id = self._donor_episode(entry, anchor, rng)
            donor_anchor = anchor - 24
            donor_windows.append(self._window(donor_id, donor_anchor, include_targets=False))
            query_windows.append(query)
            task_indices.append(self.task_to_index[entry["task_id"]])
            anchors.append(anchor)
            donor_anchors.append(donor_anchor)
        windows = query_windows
        return RH20TBatch(
            history_image=self._stack(windows, "history_image"),
            history_lowdim=self._stack(windows, "history_lowdim"),
            history_actions=self._stack(windows, "history_actions"),
            donor_history_image=(
                None if self.condition == "B0" else self._stack(donor_windows, "history_image")
            ),
            donor_history_lowdim=(
                None if self.condition == "B0" else self._stack(donor_windows, "history_lowdim")
            ),
            donor_history_actions=(
                None if self.condition == "B0" else self._stack(donor_windows, "history_actions")
            ),
            target_image=self._stack(windows, "target_image"),
            target_lowdim=self._stack(windows, "target_lowdim"),
            target_force=self._stack(windows, "target_force"),
            target_tcp_xyz=self._stack(windows, "target_tcp_xyz"),
            future_actions=self._stack(windows, "future_actions"),
            action_masks=self._stack(windows, "action_masks"),
            task_index=torch.tensor(task_indices, dtype=torch.long),
            anchor=torch.tensor(anchors, dtype=torch.long),
            donor_anchor=(
                None if self.condition == "B0" else torch.tensor(donor_anchors, dtype=torch.long)
            ),
        )
