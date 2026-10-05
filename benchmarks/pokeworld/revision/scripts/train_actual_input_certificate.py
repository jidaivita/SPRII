#!/usr/bin/env python3
"""Supervised recoverability from the exact pixel/action history visible to PokeJEPA."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file


ANCHORS = np.asarray([24, 32, 40, 47], dtype=np.int64)
ROLLOUTS = np.asarray([0, 1, 2, 3], dtype=np.int64)


def lr_factor(step: int, total: int) -> float:
    if step <= 500:
        return step / 500
    return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total - 500)))


class Certificate(nn.Module):
    def __init__(self, history_length: int = 24) -> None:
        super().__init__()
        # Fresh random initialization.  This reuses architecture only and never
        # loads a world-model checkpoint.
        self.encoder = PokeJEPA("B0_split", history_length=history_length)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, batch) -> torch.Tensor:
        history_h, _ = self.encoder.encode_batch(batch)
        _, z_p, _ = self.encoder.codes(history_h, batch.history_actions)
        return self.head(z_p.float())


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.square(y - y.mean()).sum()
    return float(1.0 - np.square(y - pred).sum() / max(float(denom), 1e-12))


@torch.inference_mode()
def evaluate(model: Certificate, data: PokeSplit, mean: np.ndarray, std: np.ndarray, device, batch_size: int) -> dict:
    systems = data.states.shape[0]
    s = np.repeat(np.arange(systems), ROLLOUTS.size * ANCHORS.size)
    r = np.tile(np.repeat(ROLLOUTS, ANCHORS.size), systems)
    a = np.tile(ANCHORS, systems * ROLLOUTS.size)
    outputs = []
    model.eval()
    for start in range(0, s.size, batch_size):
        stop = min(s.size, start + batch_size)
        batch = data._from_indices(s[start:stop], r[start:stop], a[start:stop]).to(device)
        outputs.append(model(batch).cpu().numpy())
    standardized = np.concatenate(outputs).reshape(systems, ROLLOUTS.size, ANCHORS.size, 3)
    # window -> rollout mean -> system mean
    system_prediction = standardized.mean(2).mean(1) * std + mean
    target = np.column_stack([np.log(data.mass), data.gamma, np.log(data.stiffness)])
    names = ("log_mass", "drag", "log_stiffness")
    return {
        "system_r2": {name: r2(target[:, i], system_prediction[:, i]) for i, name in enumerate(names)},
        "systems": int(systems), "rollouts": ROLLOUTS.tolist(), "anchors": ANCHORS.tolist(),
        "aggregation": "window_prediction_to_rollout_mean_to_system_mean_to_R2",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--source-bundle-sha256", required=True)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    device = torch.device(args.device)
    train = PokeSplit(args.data_root, "train")
    validation = PokeSplit(args.data_root, "val")
    target_train = np.column_stack([np.log(train.mass), train.gamma, np.log(train.stiffness)])
    mean, std = target_train.mean(0), target_train.std(0).clip(1e-8)
    model = Certificate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args.steps))
    config = {
        "schema_version": "paper-a-actual-input-certificate-1.0", "seed": args.seed,
        "steps": args.steps, "batch_interaction_budget": args.batch_size,
        "history_length": 24, "random_initialization": True, "checkpoint_weight_reuse": False,
        "inputs": "rendered frame/difference plus history actions", "targets": ["log_mass", "drag", "log_stiffness"],
        "target_mean": mean.tolist(), "target_std": std.tolist(), "loss": "equal_weight_standardized_MSE",
        "learning_rate": 3e-4, "weight_decay": 0.05, "warmup_steps": 500,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "source_bundle_sha256": args.source_bundle_sha256, "test_read": False,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    for step in range(1, args.steps + 1):
        batch = train.batch(args.batch_size, args.seed * 10_000_000 + step).to(device)
        target = torch.stack([batch.mass.log(), batch.gamma, batch.stiffness.log()], dim=-1)
        target = (target - mean_t) / std_t
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(batch)
            loss = F.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite certificate loss at {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step % 100 == 0:
            append_jsonl(log, {"event": "train", "step": step, "loss": loss, "grad_norm_preclip": gradient, "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started})
    checkpoint = args.run_dir / "checkpoints" / f"step_{args.steps:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "step": args.steps, "config": config}, temporary)
    os.replace(temporary, checkpoint)
    report = {"schema_version": "paper-a-actual-input-certificate-result-1.0", "seed": args.seed, "checkpoint_sha256": sha256_file(checkpoint), "validation": evaluate(model, validation, mean, std, device, 256), "test_read": False}
    atomic_json(args.run_dir / "validation.json", report)
    append_jsonl(log, {"event": "complete", "step": args.steps, "elapsed_seconds": time.monotonic() - started, "validation": report["validation"]})


if __name__ == "__main__":
    main()
