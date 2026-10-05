#!/usr/bin/env python3
"""Train one frozen RH20T formal condition."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

from persistent_jepa.losses import SIGReg
from persistent_jepa.rh20t_data import RH20TSplit
from persistent_jepa.rh20t_model import CONDITIONS, RH20TJEPA, RH20TModelConfig, rh20t_objective
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file


def factor(step: int, total: int) -> float:
    if step <= 500:
        return step / 500
    return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total - 500)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pairing-manifest", type=Path, required=True)
    parser.add_argument("--field-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-pairs", type=int, default=48)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-x", type=float, default=0.1)
    parser.add_argument("--lambda-force", type=float, default=1.0)
    parser.add_argument("--lambda-tcp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--source-bundle-sha256")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        if args.source_bundle is None or args.source_bundle_sha256 is None:
            raise ValueError("formal runs require --source-bundle and --source-bundle-sha256")
        actual_source_sha256 = sha256_file(args.source_bundle)
        if actual_source_sha256 != args.source_bundle_sha256:
            raise ValueError(
                "source bundle hash mismatch: "
                f"expected={args.source_bundle_sha256} actual={actual_source_sha256}"
            )
    if args.condition in {"B0", "B0split", "LowDim-B0"} and (
        args.lambda_p != 1.0 or args.lambda_x != 0.1
    ):
        # Values are recorded but inactive. Keeping defaults avoids per-condition config drift.
        raise ValueError("B0 must retain frozen inactive lambda defaults")
    if (args.run_dir / "config.json").exists() or (args.run_dir / "train.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    device = torch.device(args.device)
    cfg = RH20TModelConfig(args.condition)
    model = RH20TJEPA(cfg).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: factor(step, args.steps)
    )
    data = RH20TSplit(
        args.cache_root,
        args.split_manifest,
        args.pairing_manifest,
        args.normalization,
        "train",
        args.condition,
    )
    config = {
        "environment": "RH20T-cfg1",
        "condition": args.condition,
        "architecture": (
            "monolithic_joint_query_donor_context_128"
            if cfg.monolithic_qd
            else "split_transient64_persistent64"
            if cfg.split
            else "monolithic_query_context_128"
        ),
        "monolithic_query_donor": cfg.monolithic_qd,
        "explicit_persistent_slot": cfg.persistence,
        "seed": args.seed,
        "steps": args.steps,
        "batch_pairs": args.batch_pairs,
        "query_windows": args.batch_pairs,
        "donor_history_windows": (
            0 if cfg.donor_relation is None else args.batch_pairs
        ),
        "donor_relation": cfg.donor_relation,
        "history_frames": 24,
        "horizons": [1, 4, 16],
        "partial_causal_action": "1D gripper command, causal backward as-of",
        "lambda_p": args.lambda_p,
        "lambda_x": args.lambda_x,
        "lambda_force": args.lambda_force,
        "lambda_tcp": args.lambda_tcp,
        "sigreg_weight": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "pairing_manifest_sha256": sha256_file(args.pairing_manifest),
        "field_manifest_sha256": sha256_file(args.field_manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "source_bundle_sha256": args.source_bundle_sha256,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "smoke": args.smoke,
        "test_read": False,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        batch_seed = args.seed * 10_000_000 + step
        batch = data.batch(args.batch_pairs, batch_seed).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = rh20t_objective(
                model,
                batch,
                sigreg,
                lambda_p=args.lambda_p,
                lambda_x=args.lambda_x,
                lambda_force=args.lambda_force,
                lambda_tcp=args.lambda_tcp,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite RH20T loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step in {1, 3000, args.steps}:
            append_jsonl(
                log,
                {
                    "event": "train",
                    "step": step,
                    "grad_norm_preclip": gradient,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "cuda_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                    **metrics,
                },
            )
        if step in ({args.steps} if args.steps < 3000 else {3000, args.steps}):
            path = args.run_dir / "checkpoints" / f"step_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "step": step, "config": config}, temporary)
            os.replace(temporary, path)
            append_jsonl(
                log,
                {"event": "checkpoint", "step": step, "path": path, "sha256": sha256_file(path)},
            )
    append_jsonl(
        log, {"event": "complete", "step": args.steps, "elapsed_seconds": time.monotonic() - started}
    )


if __name__ == "__main__":
    main()
