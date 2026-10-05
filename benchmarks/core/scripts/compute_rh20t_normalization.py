#!/usr/bin/env python3
"""Compute one frozen train-task-only normalization file for all RH20T conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class Moments:
    def __init__(self, dimensions: int) -> None:
        self.count = 0
        self.total = np.zeros(dimensions, dtype=np.float64)
        self.square = np.zeros(dimensions, dtype=np.float64)

    def add(self, value: np.ndarray) -> None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1, self.total.size)
        self.count += flat.shape[0]
        self.total += flat.sum(axis=0)
        self.square += np.square(flat).sum(axis=0)

    def result(self) -> dict:
        mean = self.total / self.count
        variance = np.maximum(self.square / self.count - np.square(mean), 1e-12)
        return {"count": self.count, "mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split_manifest.read_text())
    episode_ids = split["splits"]["train"]["episode_ids"]
    moments = {
        "force": Moments(6),
        "tcp": Moments(7),
        "action": Moments(1),
        "image_channels": Moments(2),
    }
    for index, episode_id in enumerate(episode_ids, start=1):
        with np.load(args.cache_root / f"{episode_id}.npz") as data:
            moments["force"].add(data["ft_base_zeroed"])
            moments["tcp"].add(data["tcp_base"])
            moments["action"].add(data["gripper_command_width"])
            gray = data["rgb_gray"].astype(np.float32) / 255.0
            diff = np.zeros_like(gray)
            diff[1:] = gray[1:] - gray[:-1]
            moments["image_channels"].add(np.stack([gray, diff], axis=-1))
        if index % 100 == 0 or index == len(episode_ids):
            print(json.dumps({"completed": index, "total": len(episode_ids)}), flush=True)
    payload = {
        "schema_version": 1,
        "statistics_source": "train tasks only",
        "task_split_manifest_sha256": sha256(args.split_manifest),
        "train_episode_count": len(episode_ids),
        **{name: value.result() for name, value in moments.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
