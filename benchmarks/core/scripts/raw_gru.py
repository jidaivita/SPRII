#!/usr/bin/env python3
"""Raw-L24 train/validation certificate; never reads test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from persistent_jepa.evaluation import r2, system_mean
from persistent_jepa.runtime import atomic_json, set_deterministic
from persistent_jepa.sampling import VALID_ANCHORS
from persistent_jepa.torch_data import SplitArrays


class RawGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(6, 96, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(token)[0][:, -1]).squeeze(-1)


def sample(data: SplitArrays, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    system = rng.integers(data.states.shape[0], size=count)
    rollout = rng.integers(data.states.shape[1], size=count)
    anchor = rng.choice(VALID_ANCHORS, size=count)
    time_s = anchor[:, None] + np.arange(-23, 1)[None, :]
    time_a = anchor[:, None] + np.arange(-23, 0)[None, :]
    states = data.states[system[:, None], rollout[:, None], time_s]
    actions = data.actions[system[:, None], rollout[:, None], time_a]
    previous = np.concatenate([np.zeros((count, 1, 2), np.float32), actions], axis=1)
    return torch.from_numpy(np.concatenate([states, previous], axis=-1).copy()), torch.from_numpy(data.gamma[system].copy())


def fixed_eval(data: SplitArrays, seed: int, windows: int = 8) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    systems = np.repeat(np.arange(data.states.shape[0]), windows)
    rollout = rng.integers(data.states.shape[1], size=systems.size)
    anchor = rng.choice(VALID_ANCHORS, size=systems.size)
    time_s = anchor[:, None] + np.arange(-23, 1)[None, :]
    time_a = anchor[:, None] + np.arange(-23, 0)[None, :]
    states = data.states[systems[:, None], rollout[:, None], time_s]
    actions = data.actions[systems[:, None], rollout[:, None], time_a]
    previous = np.concatenate([np.zeros((systems.size, 1, 2), np.float32), actions], axis=1)
    return torch.from_numpy(np.concatenate([states, previous], axis=-1).copy()), systems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_deterministic(20260819)
    device = torch.device(args.device)
    train, val = SplitArrays(args.data_root, "train"), SplitArrays(args.data_root, "val")
    gamma_mean, gamma_std = float(np.asarray(train.gamma).mean()), float(np.asarray(train.gamma).std())
    model = RawGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        x, y = sample(train, 256, 20260819 + step)
        x, y = x.to(device), ((y - gamma_mean) / gamma_std).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            losses.append(float(loss.detach().cpu()))
    x_val, systems = fixed_eval(val, 20260819)
    model.eval()
    values = []
    with torch.inference_mode():
        for chunk in x_val.split(512):
            values.append(model(chunk.to(device)).cpu().numpy())
    window_prediction = np.concatenate(values) * gamma_std + gamma_mean
    prediction = system_mean(window_prediction[:, None], systems, val.states.shape[0])[:, 0]
    report = {
        "certificate": "Raw-GRU-L24",
        "steps": args.steps,
        "validation_system_r2": r2(np.asarray(val.gamma), prediction),
        "final_training_loss": losses[-1],
        "loss_every_100_steps": losses,
        "test_read": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
