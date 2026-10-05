"""Matched fixed-step NOD-Hier / SPRII-Hier training for the FHN closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fhn_minimal.data import FHNHierDataset, one_hot_head
from nod_sprii.canonical_losses import canonical_vicreg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_official_code(path: Path) -> None:
    sys.path.insert(0, str(path))


def create_locs(device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, 129, dtype=torch.float32)[1:]
    xx = x.reshape(128, 1).repeat(1, 128).unsqueeze(0)
    yy = x.reshape(1, 128).repeat(128, 1).unsqueeze(0)
    return torch.cat((xx, yy), dim=0).to(device)


def model_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_four(model: nn.Module, x: torch.Tensor, c: torch.Tensor,
                 indicators: torch.Tensor, locs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_locs = locs.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    prediction = model(x, c, batch_locs, one_hot_head(indicators[:, 0]).to(x.device))
    outputs = [prediction]
    for index in range(1, 4):
        prediction = model.predict(
            prediction, model.latent_vector,
            one_hot_head(indicators[:, index]).to(x.device),
        )
        outputs.append(prediction)
    return torch.stack(outputs, dim=1), model.latent_vector[:, :, 0, 0]


def lambda_at(step: int, maximum: float, warmup: int, stop: int | None) -> float:
    if stop is not None and step >= stop:
        return 0.0
    if maximum == 0.0:
        return 0.0
    if warmup <= 0:
        return maximum
    return maximum * min(1.0, step / warmup)


def training_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                  batch, criterion: nn.Module, locs: torch.Tensor,
                  align_weight: float) -> dict[str, float]:
    x, c1, c2, target, indicators, _ = batch
    device = locs.device
    x = x.to(device, non_blocking=True)
    c1 = c1.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    indicators = indicators.to(device, non_blocking=True)
    prediction, z1 = predict_four(model, x, c1, indicators, locs)
    base = criterion(prediction, target)
    align = base.new_zeros(())
    metrics: dict[str, float] = {}
    if align_weight > 0.0:
        z2 = model.conditioning_encoder(c2.to(device, non_blocking=True))
        align, align_metrics = canonical_vicreg(z1, z2)
        metrics.update({key: float(value) for key, value in align_metrics.items()})
    total = base + align_weight * align
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    metrics.update({
        "base_loss": float(base.detach()),
        "align_loss": float(align.detach()),
        "total_loss": float(total.detach()),
        "lambda_align": align_weight,
    })
    return metrics


def parse_ids(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--official-code", required=True, type=Path,
                        help="released DR2D directory containing ngs/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", choices=("nod", "sprii"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-ids", default="50-89")
    parser.add_argument("--max-systems", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lambda-align", type=float, default=0.003)
    parser.add_argument("--align-warmup-steps", type=int, default=2000)
    parser.add_argument("--align-stop-step", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    # Four independent GPU runs otherwise inherit every host core and heavily
    # oversubscribe small CPU-side symmetry transforms.
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    add_official_code(args.official_code)
    from ngs.neuralnetworks import NGS_metaNet_Hier

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_start = time.time()
    dataset = FHNHierDataset(
        args.data_dir, parse_ids(args.initial_ids), "train", args.max_systems
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=loader_generator,
    )
    model = NGS_metaNet_Hier(
        Decoder_Lift_dim=64, Condition_NumLayer=3,
        Latent_dim=2, Condition_Hidden_dim=128,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    learning_rate = 0.5 / math.sqrt(parameter_count)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    criterion = nn.MSELoss()
    locs = create_locs(device)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.run_name}.jsonl"
    meta = {
        "event": "start", "args": vars(args), "parameter_count": parameter_count,
        "learning_rate": learning_rate, "dataset_size": len(dataset),
        "dataset_load_seconds": time.time() - dataset_start,
        "initial_checksum": model_checksum(model), "time": time.time(),
    }
    meta["args"] = {key: str(value) if isinstance(value, Path) else value
                    for key, value in meta["args"].items()}
    with log_path.open("a") as handle:
        handle.write(json.dumps(meta, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)

    step = start_step
    iterator = iter(loader)
    window: list[dict[str, float]] = []
    wall_start = time.time()
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        step += 1
        weight = 0.0 if args.method == "nod" else lambda_at(
            step, args.lambda_align, args.align_warmup_steps, args.align_stop_step
        )
        metrics = training_step(model, optimizer, batch, criterion, locs, weight)
        scheduler.step()
        window.append(metrics)
        if step % args.log_every == 0 or step == args.max_steps:
            aggregate = {key: float(np.mean([row[key] for row in window if key in row]))
                         for key in sorted({key for row in window for key in row})}
            aggregate.update({
                "event": "progress", "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_second": (step - start_step) / max(time.time() - wall_start, 1e-9),
                "time": time.time(),
            })
            with log_path.open("a") as handle:
                handle.write(json.dumps(aggregate, sort_keys=True) + "\n")
            print(json.dumps(aggregate, sort_keys=True), flush=True)
            window.clear()
        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint_path = args.output_dir / f"{args.run_name}_step{step:07d}.pth"
            atomic_save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step, "args": meta["args"],
                "model_checksum": model_checksum(model),
            }, checkpoint_path)

    final = {
        "event": "complete", "step": step, "elapsed_seconds": time.time() - wall_start,
        "final_checksum": model_checksum(model), "time": time.time(),
    }
    with log_path.open("a") as handle:
        handle.write(json.dumps(final, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
