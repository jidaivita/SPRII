#!/usr/bin/env python3
"""Train the R0 vision-only multi-horizon JEPA baseline."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

from persistent_jepa.losses import SIGReg
from persistent_jepa.poke_model import PokeJEPA, poke_objective
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file


def factor(step: int, total: int) -> float:
    if step <= 500:
        return step / 500
    return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total - 500)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["B0", "B0_split", "B2", "B3", "Bx", "Sup"], default="B0"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lambda-p", type=float, default=0.0)
    parser.add_argument("--lambda-x", type=float, default=0.0)
    parser.add_argument("--history-length", type=int, default=24)
    parser.add_argument(
        "--pairing-mode",
        choices=["cross_rollout", "same_rollout_restricted", "cross_rollout_restricted"],
        default="cross_rollout",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-bundle-sha256")
    args = parser.parse_args()
    if args.variant == "B2" and args.lambda_p <= 0:
        raise ValueError("B2 requires --lambda-p > 0")
    if args.variant == "B3" and (args.lambda_p <= 0 or args.lambda_x <= 0):
        raise ValueError("B3 requires positive --lambda-p and --lambda-x")
    if args.variant == "Bx" and (args.lambda_p != 0 or args.lambda_x <= 0):
        raise ValueError("Bx requires --lambda-p 0 and --lambda-x > 0")
    if args.pairing_mode != "cross_rollout" and args.variant != "B3":
        raise ValueError("restricted pairing modes are registered only for B3")
    if (args.run_dir / "config.json").exists() or (args.run_dir / "train.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    device = torch.device(args.device)
    model = PokeJEPA(args.variant, history_length=args.history_length).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: factor(step, args.steps))
    data = PokeSplit(args.data_root, "train", history_length=args.history_length)
    config = {
        "track": "R0-Replicate-independent",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "lambda_p": args.lambda_p,
        "lambda_x": args.lambda_x,
        "history_length": args.history_length,
        "pairing_mode": args.pairing_mode,
        "batch_windows": 96,
        "sigreg_weight": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "source_bundle_sha256": args.source_bundle_sha256,
        "environment": "PokeWorld",
        "relation": args.pairing_mode,
        "interaction_budget": "R4",
        "smoke": False,
        "test_read": False,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    pairing_totals = {
        "count": 0,
        "separation_sum": 0,
        "separation_min": None,
        "separation_max": None,
        "overlap_sum": 0,
        "query_anchor_sum": 0,
    }
    for step in range(1, args.steps + 1):
        batch_seed = args.seed * 10_000_000 + step
        if args.variant == "B0":
            batch = data.batch(96, batch_seed)
        elif args.pairing_mode == "same_rollout_restricted":
            batch = data.restricted_paired_batch(48, batch_seed, "same_rollout")
        elif args.pairing_mode == "cross_rollout_restricted":
            batch = data.restricted_paired_batch(48, batch_seed, "independent_rollout")
        else:
            batch = data.paired_batch(48, batch_seed)
        if args.pairing_mode != "cross_rollout":
            half = batch.anchor.shape[0] // 2
            separation = batch.anchor[half:] - batch.anchor[:half]
            overlap = torch.clamp(args.history_length - separation, min=0)
            pairing_totals["count"] += int(separation.numel())
            pairing_totals["separation_sum"] += int(separation.sum())
            pairing_totals["overlap_sum"] += int(overlap.sum())
            pairing_totals["query_anchor_sum"] += int(batch.anchor[half:].sum())
            batch_min, batch_max = int(separation.min()), int(separation.max())
            pairing_totals["separation_min"] = (
                batch_min
                if pairing_totals["separation_min"] is None
                else min(pairing_totals["separation_min"], batch_min)
            )
            pairing_totals["separation_max"] = (
                batch_max
                if pairing_totals["separation_max"] is None
                else max(pairing_totals["separation_max"], batch_max)
            )
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, metrics = poke_objective(
                model, batch, sigreg, lambda_p=args.lambda_p, lambda_x=args.lambda_x
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite R0 loss at {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step in {3000, args.steps}:
            pairing_metrics = {}
            if pairing_totals["count"]:
                count = pairing_totals["count"]
                pairing_metrics = {
                    "pair_separation_mean": pairing_totals["separation_sum"] / count,
                    "pair_separation_min": pairing_totals["separation_min"],
                    "pair_separation_max": pairing_totals["separation_max"],
                    "pair_token_overlap_mean": pairing_totals["overlap_sum"] / count,
                    "query_anchor_mean": pairing_totals["query_anchor_sum"] / count,
                }
            append_jsonl(
                log,
                {
                    "event": "train", "step": step, "grad_norm_preclip": gradient,
                    "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started,
                    "cuda_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    **metrics,
                    **pairing_metrics,
                },
            )
        if step in {3000, args.steps}:
            path = args.run_dir / "checkpoints" / f"step_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "step": step, "config": config}, temporary)
            os.replace(temporary, path)
            append_jsonl(log, {"event": "checkpoint", "step": step, "path": path, "sha256": sha256_file(path)})
    append_jsonl(log, {"event": "complete", "step": args.steps, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
