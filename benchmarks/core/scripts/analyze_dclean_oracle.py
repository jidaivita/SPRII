#!/usr/bin/env python3
"""Compare exact D-Clean forecasts with true gamma versus mean gamma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from persistent_jepa.runtime import atomic_json, sha256_file
from persistent_jepa.simulator import exact_step
from persistent_jepa.torch_data import HORIZONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mean-gamma", type=float, default=2.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    train = np.load(args.data_root / "train.npz", mmap_mode="r")
    test = np.load(args.data_root / "test.npz", mmap_mode="r")
    states = np.asarray(test["states"], dtype=np.float64)
    actions = np.asarray(test["actions"], dtype=np.float64)
    gamma = np.asarray(test["gamma"], dtype=np.float64)
    state_std = np.asarray(train["states"], dtype=np.float64).reshape(-1, 4).std(0).clip(1e-8)
    anchors = np.arange(23, 48, dtype=np.int64)
    systems, rollouts = states.shape[:2]
    system_index = np.arange(systems)[:, None, None]
    rollout_index = np.arange(rollouts)[None, :, None]
    anchor_index = anchors[None, None, :]
    initial = states[system_index, rollout_index, anchor_index]

    report = {
        "schema_version": "dclean-oracle-mean-1.0",
        "split": "test",
        "mean_gamma": args.mean_gamma,
        "anchors": [int(anchors[0]), int(anchors[-1])],
        "windows_per_system": int(rollouts * anchors.size),
        "aggregation": "mean windows within system, then equal mean over systems",
        "position_velocity_metric": "mean Euclidean error",
        "normalized_state_metric": "mean squared error using train-state std",
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "horizons": {},
    }
    for horizon in HORIZONS:
        target = states[system_index, rollout_index, anchor_index + horizon]
        forecasts = {}
        for label, local_gamma in (
            ("oracle", gamma[:, None, None, None]),
            ("mean", np.full((systems, 1, 1, 1), args.mean_gamma)),
        ):
            prediction = initial.copy()
            for offset in range(horizon):
                force = actions[system_index, rollout_index, anchor_index + offset]
                prediction = exact_step(prediction, force, local_gamma, dt=0.05, mass=1.0)
            position = np.linalg.norm(prediction[..., :2] - target[..., :2], axis=-1)
            velocity = np.linalg.norm(prediction[..., 2:] - target[..., 2:], axis=-1)
            normalized = np.square((prediction - target) / state_std).mean(axis=-1)
            forecasts[label] = {
                "position_error": float(position.mean(axis=(1, 2)).mean()),
                "velocity_error": float(velocity.mean(axis=(1, 2)).mean()),
                "normalized_state_error": float(normalized.mean(axis=(1, 2)).mean()),
            }
        report["horizons"][f"h{horizon}"] = {
            **forecasts,
            "mean_minus_oracle": {
                key: forecasts["mean"][key] - forecasts["oracle"][key]
                for key in forecasts["oracle"]
            },
        }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
