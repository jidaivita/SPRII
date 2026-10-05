#!/usr/bin/env python3
"""Single-GPU D-Clean training entry point with 3k/20k checkpoints."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from persistent_jepa.losses import SIGReg
from persistent_jepa.model import ModelConfig, PersistentJEPA
from persistent_jepa.objective import compute_objective
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file
from persistent_jepa.torch_data import SplitArrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["B0", "Sup", "B0_split", "B1", "B2", "B3"], required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lambda-p", type=float, default=0.0)
    parser.add_argument("--lambda-x", type=float, default=0.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, nargs="*", default=[3000, 20000])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pairing-mode",
        choices=["same_system", "same_rollout_nonoverlap", "random_system_fixed"],
        default=None,
    )
    parser.add_argument("--relation-map", type=Path)
    parser.add_argument("--source-bundle-sha256")
    return parser.parse_args()


def code_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "uncommitted-worktree"


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    path: Path,
    model: PersistentJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.variant in {"B1", "B2"} and args.lambda_p <= 0:
        raise ValueError(f"{args.variant} requires --lambda-p > 0")
    if args.variant == "B3" and (args.lambda_p <= 0 or args.lambda_x <= 0):
        raise ValueError("B3 requires --lambda-p > 0 and --lambda-x > 0")
    if args.pairing_mode is None:
        args.pairing_mode = "same_rollout_nonoverlap" if args.variant == "B1" else "same_system"
    if args.variant == "B1" and args.pairing_mode != "same_rollout_nonoverlap":
        raise ValueError("B1 requires same_rollout_nonoverlap pairing")
    if args.pairing_mode == "random_system_fixed" and args.relation_map is None:
        raise ValueError("random_system_fixed requires --relation-map")
    if args.pairing_mode != "random_system_fixed" and args.relation_map is not None:
        raise ValueError("--relation-map is only valid with random_system_fixed")
    if not args.resume and ((args.run_dir / "train.jsonl").exists() or (args.run_dir / "config.json").exists()):
        raise FileExistsError(f"refusing to overwrite existing run: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("frozen protocol requires bf16-capable CUDA hardware")

    model_cfg = ModelConfig(variant=args.variant)
    model = PersistentJEPA(model_cfg).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, args.steps, args.warmup_steps)
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous = checkpoint["config"]
        frozen_keys = (
            "variant", "seed", "lambda_p", "lambda_x", "sigreg_weight",
            "learning_rate", "weight_decay", "warmup_steps",
        )
        mismatches = {
            key: (previous[key], getattr(args, key))
            for key in frozen_keys
            if previous[key] != getattr(args, key)
        }
        if mismatches:
            raise ValueError(f"resume would change frozen training configuration: {mismatches}")
        if args.steps <= int(checkpoint["step"]):
            raise ValueError("resume target steps must exceed checkpoint step")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    data = SplitArrays(args.data_root, "train")
    pseudo_systems = None
    if args.relation_map is not None:
        pseudo_systems = __import__("numpy").load(args.relation_map)
        expected_shape = (data.states.shape[0], data.states.shape[1])
        if pseudo_systems.shape != expected_shape:
            raise ValueError(
                f"relation map shape mismatch: expected {expected_shape}, got {pseudo_systems.shape}"
            )
    manifest_path = args.data_root / "manifest.json"
    project_root = Path(__file__).resolve().parents[1]
    run_config = {
        **vars(args),
        "model": asdict(model_cfg),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "relation_map_sha256": sha256_file(args.relation_map) if args.relation_map else None,
        "source_bundle_sha256": args.source_bundle_sha256,
        "environment": "D-Clean",
        "relation": args.pairing_mode,
        "interaction_budget": "R8",
        "smoke": False,
        "test_read": False,
        "code_revision": code_revision(project_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    atomic_json(args.run_dir / "config.json", run_config)
    log_path = args.run_dir / "train.jsonl"
    append_jsonl(log_path, {"event": "start", "step": start_step, "config": run_config})
    checkpoint_steps = set(args.checkpoint_every) | {args.steps}
    model.train()
    started = time.monotonic()
    interval_started = started
    for step in range(start_step + 1, args.steps + 1):
        batch = data.paired_batch(
            pairs=48,
            seed=args.seed * 10_000_000 + step,
            pairing_mode=args.pairing_mode,
            pseudo_systems=pseudo_systems,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = compute_objective(
                model,
                batch,
                sigreg,
                sigreg_weight=args.sigreg_weight,
                lambda_p=args.lambda_p,
                lambda_x=args.lambda_x,
            )
        if not torch.isfinite(loss):
            append_jsonl(log_path, {"event": "fatal_nonfinite_loss", "step": step, **metrics})
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            append_jsonl(log_path, {"event": "fatal_nonfinite_gradient", "step": step})
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step in checkpoint_steps:
            now = time.monotonic()
            append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm_preclip": grad_norm,
                    "steps_per_second_interval": args.log_every / max(now - interval_started, 1e-9),
                    "elapsed_seconds": now - started,
                    "cuda_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                    **metrics,
                },
            )
            interval_started = now
        if step in checkpoint_steps:
            checkpoint_path = args.run_dir / "checkpoints" / f"step_{step:06d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, run_config)
            append_jsonl(
                log_path,
                {
                    "event": "checkpoint",
                    "step": step,
                    "path": checkpoint_path,
                    "sha256": sha256_file(checkpoint_path),
                },
            )
    append_jsonl(log_path, {"event": "complete", "step": args.steps, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
