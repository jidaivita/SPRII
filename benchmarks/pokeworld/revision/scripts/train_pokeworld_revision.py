#!/usr/bin/env python3
"""Frozen Paper-A revision training: Rel-InfoNCE, Refinement, and Fidelity."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from persistent_jepa.losses import SIGReg, canonical_vicreg
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import append_jsonl, atomic_json, set_deterministic, sha256_file
from persistent_jepa.torch_data import HORIZONS


def lr_factor(step: int, total: int) -> float:
    if step <= 500:
        return step / 500
    return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total - 500)))


def rel_infonce(z_p: torch.Tensor, temperature: float) -> torch.Tensor:
    if z_p.ndim != 2 or z_p.shape[0] % 2:
        raise ValueError(f"expected two equal view halves, got {tuple(z_p.shape)}")
    count = z_p.shape[0]
    half = count // 2
    normalized = F.normalize(z_p.float(), dim=-1)
    logits = normalized @ normalized.T / temperature
    logits.fill_diagonal_(float("-inf"))
    target = (torch.arange(count, device=z_p.device) + half) % count
    return F.cross_entropy(logits, target)


def objective(
    model: PokeJEPA,
    batch,
    sigreg: SIGReg,
    objective_name: str,
    lambda_p: float,
    lambda_nce: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_h, target_h = model.encode_batch(batch)
    z_s, z_p, context = model.codes(history_h, batch.history_actions)
    horizon_losses = []
    for hi, _horizon in enumerate(HORIZONS):
        horizon_index = torch.full((context.shape[0],), hi, device=context.device, dtype=torch.long)
        prediction = model.predictor(
            context, batch.future_actions[:, hi], batch.action_masks[:, hi], horizon_index
        )
        horizon_losses.append(F.mse_loss(prediction, target_h[:, hi]))
    prediction_loss = torch.stack(horizon_losses).mean()
    sigreg_loss = sigreg(torch.cat([history_h, target_h], dim=1).transpose(0, 1))
    total = prediction_loss + 0.02 * sigreg_loss
    metrics = {
        "loss_prediction": prediction_loss.detach(),
        "loss_sigreg": sigreg_loss.detach(),
        **{f"loss_h{h}": value.detach() for h, value in zip(HORIZONS, horizon_losses, strict=True)},
    }
    if objective_name == "align":
        half = z_p.shape[0] // 2
        persist, persist_metrics = canonical_vicreg(z_p[:half], z_p[half:])
        total = total + lambda_p * persist
        metrics.update(persist_metrics)
        metrics["loss_persist"] = persist.detach()
    elif objective_name == "rel_infonce":
        contrastive = rel_infonce(z_p, temperature)
        total = total + lambda_nce * contrastive
        metrics["loss_rel_infonce"] = contrastive.detach()
    elif objective_name != "none":
        raise ValueError(f"unknown objective {objective_name}")
    metrics["loss_total"] = total.detach()
    return total, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--family", choices=["relinfonce", "refinement", "fidelity"], required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--history-length", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lambda-nce", type=float, default=1.0)
    p.add_argument("--fidelity-q", type=float)
    p.add_argument("--fidelity-assignment-seed", type=int, default=20260901)
    p.add_argument("--donor-pools", type=Path)
    p.add_argument("--protocol-manifest", type=Path, required=True)
    p.add_argument("--source-bundle-sha256", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)
    if args.family == "relinfonce":
        if args.condition != "Rel-InfoNCE":
            raise ValueError("relinfonce family requires condition Rel-InfoNCE")
        model_variant, objective_name = "B0_split", "rel_infonce"
    elif args.family == "refinement":
        if args.condition not in {"Split", "G1", "G2", "G3", "Random"}:
            raise ValueError(f"invalid refinement condition {args.condition}")
        model_variant = "B0_split" if args.condition == "Split" else "B2"
        objective_name = "none" if args.condition == "Split" else "align"
    else:
        if args.condition not in {"q0", "q05", "q1"}:
            raise ValueError(f"invalid fidelity condition {args.condition}")
        if args.fidelity_q not in {0.0, 0.5, 1.0}:
            raise ValueError("fidelity requires q in {0,.5,1}")
        model_variant, objective_name = "B2", "align"

    device = torch.device(args.device)
    model = PokeJEPA(model_variant, history_length=args.history_length).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, args.steps)
    )
    data = PokeSplit(args.data_root, "train", history_length=args.history_length)
    pools = None
    if args.family == "refinement" and args.condition in {"G1", "G2", "Random"}:
        if args.donor_pools is None:
            raise ValueError("factorized donor pools are required")
        with np.load(args.donor_pools) as payload:
            pools = np.asarray(payload[f"train_{args.condition}"], dtype=np.int64)

    config = {
        "schema_version": "paper-a-revision-training-1.0",
        "family": args.family,
        "condition": args.condition,
        "model_variant": model_variant,
        "variant": model_variant,
        "objective": objective_name,
        "seed": args.seed,
        "steps": args.steps,
        "history_length": args.history_length,
        "batch_systems": 48,
        "batch_interaction_views": 96,
        "batch_relational_pairs": 48,
        "views_per_system": 2,
        "k_pool": 3 if args.family == "refinement" else None,
        "k_used": 1 if args.family == "refinement" else None,
        "temperature": args.temperature if args.family == "relinfonce" else None,
        "lambda_nce": args.lambda_nce if args.family == "relinfonce" else 0.0,
        "lambda_p": 1.0 if objective_name == "align" else 0.0,
        "lambda_x": 0.0,
        "fidelity_q": args.fidelity_q,
        "fidelity_assignment_seed": (
            args.fidelity_assignment_seed if args.family == "fidelity" else None
        ),
        "negative_scope": "single_gpu_batch_local" if args.family == "relinfonce" else None,
        "positives_per_anchor": 1 if args.family == "relinfonce" else None,
        "negatives_per_anchor": 94 if args.family == "relinfonce" else None,
        "projection_head": False if args.family == "relinfonce" else None,
        "sigreg_weight": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "protocol_manifest": str(args.protocol_manifest),
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
        "donor_pools_sha256": sha256_file(args.donor_pools) if args.donor_pools else None,
        "source_bundle_sha256": args.source_bundle_sha256,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "test_read": False,
    }
    atomic_json(args.run_dir / "config.json", config)
    log = args.run_dir / "train.jsonl"
    append_jsonl(log, {"event": "start", "config": config})
    started = time.monotonic()
    correctness_sum = 0
    correctness_count = 0
    for step in range(1, args.steps + 1):
        data_seed = args.seed * 10_000_000 + step
        if args.family == "relinfonce":
            batch = data.paired_batch(48, data_seed)
        elif args.family == "refinement":
            relation = "G3" if args.condition == "Split" else args.condition
            batch = data.relation_paired_batch(48, data_seed, relation, pools)
        else:
            batch, correct_mask = data.fidelity_paired_batch(
                48, data_seed, args.fidelity_assignment_seed, step, float(args.fidelity_q)
            )
            correctness_sum += int(correct_mask.sum())
            correctness_count += int(correct_mask.size)
        if batch.history_current.shape[0] != 96:
            raise AssertionError("every optimizer step must contain exactly 96 interaction views")
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, metrics = objective(
                model, batch, sigreg, objective_name,
                lambda_p=1.0 if objective_name == "align" else 0.0,
                lambda_nce=args.lambda_nce,
                temperature=args.temperature,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step in {3000, args.steps}:
            extra = {}
            if correctness_count:
                extra["fidelity_correct_fraction_running"] = correctness_sum / correctness_count
            append_jsonl(log, {
                "event": "train", "step": step,
                "grad_norm_preclip": gradient,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                **metrics, **extra,
            })
        if step in {3000, args.steps}:
            path = args.run_dir / "checkpoints" / f"step_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "step": step, "config": config}, temporary)
            os.replace(temporary, path)
            append_jsonl(log, {
                "event": "checkpoint", "step": step,
                "path": str(path), "sha256": sha256_file(path),
            })
    append_jsonl(log, {
        "event": "complete", "step": args.steps,
        "elapsed_seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
