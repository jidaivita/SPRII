#!/usr/bin/env python3
"""Raw state/proprio/touch/action multi-factor observability certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from persistent_jepa.evaluation import r2, system_mean
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file


ANCHORS = np.arange(24, 48, dtype=np.int64)


class RawGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(17, 96, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])


class Arrays:
    def __init__(self, root: Path, split: str) -> None:
        data = np.load(root / f"{split}.npz", mmap_mode="r")
        self.states, self.actions, self.touch = data["states"], data["actions"], data["touch"]
        self.mass, self.gamma, self.stiffness = data["mass"], data["gamma"], data["stiffness"]

    def factors(self):
        return np.column_stack([np.log(self.mass), self.gamma, np.log(self.stiffness)]).astype(np.float32)


def tokens(data, systems, rollouts, anchors):
    state_time = anchors[:, None] + np.arange(-23, 1)[None, :]
    transition_time = anchors[:, None] + np.arange(-23, 0)[None, :]
    states = data.states[systems[:, None], rollouts[:, None], state_time]
    actions = data.actions[systems[:, None], rollouts[:, None], transition_time]
    touch = data.touch[systems[:, None], rollouts[:, None], transition_time]
    previous = np.concatenate(
        [np.zeros((systems.size, 1, 9), np.float32), np.concatenate([actions, touch], axis=-1)],
        axis=1,
    )
    return torch.from_numpy(np.concatenate([states, previous], axis=-1).copy())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-bundle-sha256", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed = 20260904
    set_deterministic(seed)
    rng = np.random.default_rng(seed)
    train, val = Arrays(args.data_root, "train"), Arrays(args.data_root, "val")
    train_factor = train.factors()
    mean, std = train_factor.mean(0), train_factor.std(0).clip(1e-8)
    model = RawGRU().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trace = []
    for step in range(1, args.steps + 1):
        systems = rng.integers(train.states.shape[0], size=256)
        rollouts = rng.integers(train.states.shape[1], size=256)
        anchors = rng.choice(ANCHORS, size=256)
        x = tokens(train, systems, rollouts, anchors).to(args.device)
        y = torch.from_numpy(((train_factor[systems] - mean) / std).copy()).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            trace.append(float(loss.detach().cpu()))
    windows = 8
    systems = np.repeat(np.arange(val.states.shape[0]), windows)
    eval_rng = np.random.default_rng(seed + 1)
    rollouts = eval_rng.integers(val.states.shape[1], size=systems.size)
    anchors = eval_rng.choice(ANCHORS, size=systems.size)
    x = tokens(val, systems, rollouts, anchors)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for chunk in x.split(512):
            predictions.append(model(chunk.to(args.device)).cpu().numpy())
    window = np.concatenate(predictions) * std + mean
    prediction = system_mean(window, systems, val.states.shape[0])
    truth = val.factors()
    report = {
        "schema_version": "pokeworld-raw-multifactor-certificate-1.0",
        "certificate": "raw state-proprio-touch-action GRU, history 24",
        "targets": {"mass": "log mass", "drag": "gamma", "stiffness": "log stiffness"},
        "validation_system_r2": {
            name: r2(truth[:, i], prediction[:, i])
            for i, name in enumerate(("mass", "drag", "stiffness"))
        },
        "steps": args.steps,
        "final_training_loss": trace[-1],
        "loss_every_100_steps": trace,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "source_bundle_sha256": args.source_bundle_sha256,
        "test_read": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
