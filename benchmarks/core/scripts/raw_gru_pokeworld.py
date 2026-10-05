#!/usr/bin/env python3
"""R0 raw-observation GRU certificate on train/validation systems only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from persistent_jepa.evaluation import r2, system_mean
from persistent_jepa.runtime import atomic_json, set_deterministic


ANCHORS = np.arange(15, 64, dtype=np.int64)


class RawPokeGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(17, 96, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0][:, -1]).squeeze(-1)


class PokeArrays:
    def __init__(self, root: Path, split: str) -> None:
        data = np.load(root / f"{split}.npz", mmap_mode="r")
        self.states = data["states"]
        self.actions = data["actions"]
        self.touch = data["touch"]
        self.gamma = data["gamma"]


def tokens(data: PokeArrays, systems: np.ndarray, rollouts: np.ndarray, anchors: np.ndarray) -> torch.Tensor:
    time_s = anchors[:, None] + np.arange(-15, 1)[None, :]
    time_transition = anchors[:, None] + np.arange(-15, 0)[None, :]
    states = data.states[systems[:, None], rollouts[:, None], time_s]
    actions = data.actions[systems[:, None], rollouts[:, None], time_transition]
    touch = data.touch[systems[:, None], rollouts[:, None], time_transition]
    previous = np.concatenate(
        [np.zeros((systems.size, 1, 9), np.float32), np.concatenate([actions, touch], axis=-1)],
        axis=1,
    )
    return torch.from_numpy(np.concatenate([states, previous], axis=-1).copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed = 20260821
    set_deterministic(seed)
    rng = np.random.default_rng(seed)
    train, val = PokeArrays(args.data_root, "train"), PokeArrays(args.data_root, "val")
    mean, std = float(np.asarray(train.gamma).mean()), float(np.asarray(train.gamma).std())
    device = torch.device(args.device)
    model = RawPokeGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_trace = []
    model.train()
    for step in range(1, args.steps + 1):
        systems = rng.integers(train.states.shape[0], size=256)
        rollouts = rng.integers(train.states.shape[1], size=256)
        anchors = rng.choice(ANCHORS, size=256)
        x = tokens(train, systems, rollouts, anchors).to(device)
        y = torch.from_numpy(((train.gamma[systems] - mean) / std).copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            loss_trace.append(float(loss.detach().cpu()))

    windows = 8
    systems = np.repeat(np.arange(val.states.shape[0]), windows)
    eval_rng = np.random.default_rng(seed + 1)
    rollouts = eval_rng.integers(val.states.shape[1], size=systems.size)
    anchors = eval_rng.choice(ANCHORS, size=systems.size)
    x = tokens(val, systems, rollouts, anchors)
    values = []
    model.eval()
    with torch.inference_mode():
        for chunk in x.split(512):
            values.append(model(chunk.to(device)).cpu().numpy())
    window_prediction = np.concatenate(values) * std + mean
    prediction = system_mean(window_prediction[:, None], systems, val.states.shape[0])[:, 0]
    report = {
        "certificate": "R0-Raw-GRU-L16-state-proprio-touch-action",
        "validation_system_drag_r2": r2(np.asarray(val.gamma), prediction),
        "steps": args.steps,
        "final_training_loss": loss_trace[-1],
        "loss_every_100_steps": loss_trace,
        "test_read": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
