#!/usr/bin/env python3
"""System-split R0 drag probes; validation only in the first study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from persistent_jepa.evaluation import mlp_probe, r2, select_ridge, system_mean
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeBatch, PokeSplit
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file


@torch.inference_mode()
def extract(model: PokeJEPA, batch: PokeBatch, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    count = batch.history_current.shape[0]
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        chunk = PokeBatch(**{key: value[start:stop].to(device) for key, value in batch.__dict__.items()})
        history_h, _ = model.encode_batch(chunk)
        output.append(model.context(history_h, chunk.history_actions).float().cpu())
    return torch.cat(output).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_deterministic(20260821)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PokeJEPA().to(args.device)
    model.load_state_dict(checkpoint["model"])
    train, val = PokeSplit(args.data_root, "train"), PokeSplit(args.data_root, "val")
    train_batch, train_system = train.fixed_system_windows(8, 20260821)
    val_batch, val_system = val.fixed_system_windows(8, 20260822)
    train_rep = system_mean(extract(model, train_batch, torch.device(args.device)), train_system, train.states.shape[0])
    val_rep = system_mean(extract(model, val_batch, torch.device(args.device)), val_system, val.states.shape[0])
    ridge, alpha, _ = select_ridge(train_rep, np.asarray(train.gamma), val_rep, np.asarray(val.gamma))
    ridge_prediction = ridge.predict(val_rep)
    mlp_prediction = mlp_probe(train_rep, np.asarray(train.gamma), val_rep, seed=20260821)
    report = {
        "track": "R0-Replicate-independent",
        "split": "val",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "drag_ridge_alpha": alpha,
        "drag_ridge_system_r2": r2(np.asarray(val.gamma), ridge_prediction),
        "drag_mlp_system_r2": r2(np.asarray(val.gamma), mlp_prediction),
        "windows_per_system": 8,
        "test_read": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
