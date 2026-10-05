#!/usr/bin/env python3
"""Train one frozen A1-Baxter relation condition."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

import torch

from persistent_jepa.baxter_data import BaxterSplit, CONDITIONS
from persistent_jepa.baxter_model import BaxterJEPA, BaxterModelConfig, baxter_objective
from persistent_jepa.losses import SIGReg
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file


def factor(step: int, total: int) -> float:
    if step <= 500:
        return step / 500
    return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total - 500)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-pairs", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--source-bundle-sha256")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps != 10000 and not args.smoke:
        raise ValueError("formal A1-Baxter runs are fixed to 10,000 steps")
    if not args.smoke:
        if args.source_bundle is None or args.source_bundle_sha256 is None:
            raise ValueError("formal run requires a frozen source bundle")
        if sha256_file(args.source_bundle) != args.source_bundle_sha256:
            raise ValueError("source bundle SHA-256 mismatch")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    device = torch.device(args.device)
    model = BaxterJEPA(BaxterModelConfig(args.condition)).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: factor(step, args.steps))
    data = BaxterSplit(args.data_root, args.manifest, args.normalization, "train", args.condition)
    config = {
        "schema_version": "paper-a-a1-baxter-training-v1.0",
        "environment": "BaxterHardness-public-v1.0",
        "condition": args.condition,
        "seed": args.seed,
        "steps": args.steps,
        "fixed_checkpoint_step": args.steps,
        "batch_pairs": args.batch_pairs,
        "history_samples": 40,
        "target_offsets": [1, 20, 40],
        "input_channels": 16,
        "lambda_p": 1.0,
        "lambda_x": 0.1,
        "sigreg_weight": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "manifest_sha256": sha256_file(args.manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "pairing_sha256": sha256_file(args.pairing),
        "source_bundle_sha256": args.source_bundle_sha256,
        "test_read": False,
        "confirmation_read": False,
        "smoke": args.smoke,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        batch = data.batch(args.batch_pairs, args.seed * 10_000_000 + step).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = baxter_objective(model, batch, sigreg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite A1 loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step in {1, args.steps}:
            append_jsonl(log, {
                "event": "train",
                "step": step,
                "grad_norm_preclip": gradient,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
                ),
                **metrics,
            })
    checkpoint = args.run_dir / "checkpoints" / f"step_{args.steps:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "step": args.steps, "config": config}, temporary)
    os.replace(temporary, checkpoint)
    append_jsonl(log, {"event": "checkpoint", "step": args.steps, "path": checkpoint, "sha256": sha256_file(checkpoint)})
    append_jsonl(log, {"event": "complete", "step": args.steps, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
