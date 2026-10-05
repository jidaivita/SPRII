#!/usr/bin/env python3
"""Train one frozen early/full input-accessibility certificate."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from persistent_jepa.baxter_data import BaxterSplit
from persistent_jepa.baxter_model import BaxterCertificate
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file


def factor(step: int, total: int) -> float:
    if step <= 200:
        return step / 200
    return 0.5 * (1 + math.cos(math.pi * (step - 200) / (total - 200)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("early8", "full80"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--source-bundle-sha256")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps != 2000 and not args.smoke:
        raise ValueError("formal certificates are fixed to 2,000 steps")
    if not args.smoke:
        if args.source_bundle is None or args.source_bundle_sha256 is None:
            raise ValueError("formal certificate requires a frozen source bundle")
        if sha256_file(args.source_bundle) != args.source_bundle_sha256:
            raise ValueError("source bundle SHA-256 mismatch")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    input_length = 8 if args.variant == "early8" else 80
    set_deterministic(args.seed)
    device = torch.device(args.device)
    model = BaxterCertificate(input_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: factor(step, args.steps))
    data = BaxterSplit(args.data_root, args.manifest, args.normalization, "train", "Random")
    config = {
        "schema_version": "paper-a-a1-baxter-certificate-v1.0",
        "variant": args.variant,
        "input_length": input_length,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "manifest_sha256": sha256_file(args.manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "source_bundle_sha256": args.source_bundle_sha256,
        "confirmation_read": False,
        "smoke": args.smoke,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        value, hardness, shape = data.classification_batch(
            args.batch_size, args.seed * 10_000_000 + step, input_length
        )
        value, hardness, shape = value.to(device), hardness.to(device), shape.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hardness_logits, shape_logits = model(value)
            hardness_loss = F.cross_entropy(hardness_logits, hardness)
            shape_loss = F.cross_entropy(shape_logits, shape)
            loss = hardness_loss + shape_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite certificate loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step in {1, args.steps}:
            append_jsonl(log, {
                "event": "train",
                "step": step,
                "loss": loss,
                "hardness_loss": hardness_loss,
                "shape_loss": shape_loss,
                "grad_norm_preclip": gradient,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
                ),
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
