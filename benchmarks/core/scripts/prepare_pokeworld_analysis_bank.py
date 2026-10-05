#!/usr/bin/env python3
"""Create the immutable PokeWorld geometry/counterfactual analysis bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from persistent_jepa.runtime import atomic_json, sha256_file


SCHEMA = "pokeworld-analysis-bank-1.0"
PAIRING_SEED = 20260823
GEOMETRY_ANCHORS = [24, 28, 33, 37, 42, 47]
GEOMETRY_ROLLOUTS = [0, 1, 2, 3]
COUNTERFACTUAL_WINDOWS_PER_SYSTEM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def split_size(root: Path, split: str) -> tuple[int, int]:
    data = np.load(root / f"{split}.npz", mmap_mode="r")
    return int(data["states"].shape[0]), int(data["states"].shape[1])


def counterfactual_rows(systems_count: int, rollouts: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    systems = np.repeat(np.arange(systems_count), COUNTERFACTUAL_WINDOWS_PER_SYSTEM)
    query_rollout = rng.integers(rollouts, size=systems.size)
    donor_offset = rng.integers(1, rollouts, size=systems.size)
    donor_rollout = (query_rollout + donor_offset) % rollouts
    query_anchor = rng.choice(np.arange(24, 48), size=systems.size)
    donor_anchor = rng.choice(np.arange(24, 48), size=systems.size)
    return [
        {
            "system_index": int(system),
            "query_rollout": int(query),
            "query_anchor": int(q_anchor),
            "donor_rollout": int(donor),
            "donor_anchor": int(d_anchor),
        }
        for system, query, q_anchor, donor, d_anchor in zip(
            systems, query_rollout, query_anchor, donor_rollout, donor_anchor, strict=True
        )
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_systems, train_rollouts = split_size(args.data_root, "train")
    val_systems, val_rollouts = split_size(args.data_root, "val")
    if (train_rollouts, val_rollouts) != (4, 4):
        raise ValueError("the frozen bank requires the correctly grouped R4 dataset")
    payload = {
        "schema_version": SCHEMA,
        "dataset_root": str(args.data_root),
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "test_read": False,
        "geometry": {
            "train_system_indices": list(range(train_systems)),
            "val_system_indices": list(range(val_systems)),
            "rollout_ids": GEOMETRY_ROLLOUTS,
            "anchors": GEOMETRY_ANCHORS,
            "window_to_rollout": "arithmetic mean over fixed anchors",
            "rollout_to_system": "arithmetic mean over four rollout codes",
            "standardization": "per-checkpoint and branch, correctly-grouped train rollout codes only",
        },
        "counterfactual": {
            "pairing_seed": PAIRING_SEED,
            "windows_per_system": COUNTERFACTUAL_WINDOWS_PER_SYSTEM,
            "train_rows": counterfactual_rows(train_systems, train_rollouts, PAIRING_SEED),
            "val_rows": counterfactual_rows(val_systems, val_rollouts, PAIRING_SEED + 100_000),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "geometry_train_systems": train_systems,
        "geometry_val_systems": val_systems,
        "counterfactual_train_rows": len(payload["counterfactual"]["train_rows"]),
        "counterfactual_val_rows": len(payload["counterfactual"]["val_rows"]),
    }, indent=2))


if __name__ == "__main__":
    main()
