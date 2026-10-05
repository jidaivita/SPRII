#!/usr/bin/env python3
"""Support-selected Raw-L24 certificate for PokeWorld unseen gamma values."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from persistent_jepa.evaluation import r2, system_mean
from persistent_jepa.runtime import atomic_json, set_deterministic


HISTORY = 24
ANCHORS = np.arange(24, 48, dtype=np.int64)
CHECKPOINT_STEPS = (1000, 2000, 3000, 4000, 5000)


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
        self.states, self.actions, self.touch = data["states"], data["actions"], data["touch"]
        self.gamma = data["gamma"]


def tokens(data: PokeArrays, systems: np.ndarray, rollouts: np.ndarray, anchors: np.ndarray) -> torch.Tensor:
    time_s = anchors[:, None] + np.arange(-(HISTORY - 1), 1)[None, :]
    time_transition = anchors[:, None] + np.arange(-(HISTORY - 1), 0)[None, :]
    states = data.states[systems[:, None], rollouts[:, None], time_s]
    actions = data.actions[systems[:, None], rollouts[:, None], time_transition]
    touch = data.touch[systems[:, None], rollouts[:, None], time_transition]
    previous = np.concatenate(
        [np.zeros((systems.size, 1, 9), np.float32), np.concatenate([actions, touch], axis=-1)],
        axis=1,
    )
    return torch.from_numpy(np.concatenate([states, previous], axis=-1).copy())


def evaluate(model: nn.Module, data: PokeArrays, mean: float, std: float, device: torch.device, seed: int) -> tuple[float, np.ndarray]:
    windows = 8
    systems = np.repeat(np.arange(data.states.shape[0]), windows)
    rng = np.random.default_rng(seed)
    rollouts = rng.integers(data.states.shape[1], size=systems.size)
    anchors = rng.choice(ANCHORS, size=systems.size)
    x = tokens(data, systems, rollouts, anchors)
    values = []
    model.eval()
    with torch.inference_mode():
        for chunk in x.split(512):
            values.append(model(chunk.to(device)).cpu().numpy())
    window_prediction = np.concatenate(values) * std + mean
    prediction = system_mean(window_prediction[:, None], systems, data.states.shape[0])[:, 0]
    return r2(np.asarray(data.gamma), prediction), prediction


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - y
    ranks_y = np.argsort(np.argsort(y))
    ranks_p = np.argsort(np.argsort(prediction))
    return {
        "r2": r2(y, prediction),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "nrmse_by_evaluation_gamma_range": float(np.sqrt(np.square(residual).mean()) / (y.max() - y.min())),
        "spearman": float(np.corrcoef(ranks_y, ranks_p)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seed = 20260829
    set_deterministic(seed)
    rng = np.random.default_rng(seed)
    train = PokeArrays(args.data_root, "train")
    val = PokeArrays(args.data_root, "val")
    ood = PokeArrays(args.data_root, "ood")
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    ood_gamma_range = float(sum(
        high - low for low, high in manifest["metadata"]["ood_gamma_intervals"]
    ))
    mean, std = float(np.asarray(train.gamma).mean()), float(np.asarray(train.gamma).std())
    device = torch.device(args.device)
    model = RawPokeGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    candidates = []
    for step in range(1, max(CHECKPOINT_STEPS) + 1):
        systems = rng.integers(train.states.shape[0], size=256)
        rollouts = rng.integers(train.states.shape[1], size=256)
        anchors = rng.choice(ANCHORS, size=256)
        x = tokens(train, systems, rollouts, anchors).to(device)
        y = torch.from_numpy(((train.gamma[systems] - mean) / std).copy()).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            val_r2, _ = evaluate(model, val, mean, std, device, seed + 1)
            candidates.append((val_r2, step, copy.deepcopy(model.state_dict())))
    best_val_r2, best_step, best_state = max(candidates, key=lambda item: (item[0], -item[1]))
    model.load_state_dict(best_state)
    _, val_prediction = evaluate(model, val, mean, std, device, seed + 1)
    # OOD is first evaluated only after the checkpoint has been selected on support validation.
    _, ood_prediction = evaluate(model, ood, mean, std, device, seed + 2)
    report = {
        "schema_version": "pokeworld-ood-raw-l24-1.0",
        "certificate": "Raw-GRU-L24-state-proprio-touch-action",
        "selection": "checkpoint selected on support-distribution validation only",
        "candidate_validation_r2": {str(step): score for score, step, _ in candidates},
        "selected_step": best_step,
        "val_support": metrics(np.asarray(val.gamma), val_prediction),
        "ood": {
            **metrics(np.asarray(ood.gamma), ood_prediction),
            "nrmse_by_evaluation_gamma_range": float(
                np.sqrt(np.square(ood_prediction - np.asarray(ood.gamma)).mean()) / ood_gamma_range
            ),
        },
        "extrapolation_interpretation_boundary": (
            "low Raw-GRU extrapolation does not prove raw trajectories are non-identifiable; "
            "the learned regression head may fail outside its training label range"
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
