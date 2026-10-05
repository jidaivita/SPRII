"""Frozen Burgers evaluation for the external NOD comparison.

This module opens final data only after a checkpoint has been selected. It uses
released split/grouping and metric logic through the copied compatibility module;
no training or checkpoint selection happens here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from train_nod_clean import (
    MODEL_DEFAULTS,
    NGS_INR,
    PREDICTION_HORIZON,
    move_batch_to_device,
    resolve_device,
    run_epoch,
)
from ngs.utils import BurgersPairedDataset, _BurgersGroupedEval


def load_model(checkpoint: Path, device: torch.device) -> tuple[NGS_INR, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    kwargs = dict(cfg.get("model_defaults", MODEL_DEFAULTS))
    kwargs["context_dim"] = int(cfg.get("context_dim_override", cfg.get("context_dim", 1)))
    model = NGS_INR(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        **({"prefetch_factor": 4} if num_workers > 0 else {}),
    )


def evaluate(checkpoint: Path, train_root: Path, truth_root: Path, device_name: str,
             batch_size: int, num_workers: int, deterministic_pairing: bool) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, cfg = load_model(checkpoint, device)
    train_ds = BurgersPairedDataset(
        data_root=str(train_root), split="test", prediction_horizon=PREDICTION_HORIZON,
        cache_mode="cond_init", seed=0,
    )
    ood_ds = _BurgersGroupedEval(
        group="ood", eval_root=str(truth_root), prediction_horizon=PREDICTION_HORIZON,
        deterministic_pairing=deterministic_pairing, seed=0,
    )
    inviscid_ds = _BurgersGroupedEval(
        group="ood_inviscid", eval_root=str(truth_root), prediction_horizon=PREDICTION_HORIZON,
        deterministic_pairing=deterministic_pairing, seed=0,
    )
    train_loader = make_loader(train_ds, batch_size, num_workers)
    ood_loader = make_loader(ood_ds, batch_size, num_workers)
    inviscid_loader = make_loader(inviscid_ds, batch_size, num_workers)
    n_t = int(getattr(train_ds, "n_t", 1001))
    criterion = torch.nn.MSELoss()
    common = dict(model=model, criterion=criterion, device=device, optimizer=None,
                  clip_grad=0.0, t_norm_denom=float(max(1, n_t - 1)),
                  num_query_points=None, show_progress=False)
    out = {
        "checkpoint": str(checkpoint.resolve()),
        "config_arch_version": cfg.get("arch_version"),
        "device": str(device),
        "manifest": "available_released_manifest",
        "groups": {},
    }
    for name, loader in (("id_test", train_loader), ("ood_viscous", ood_loader),
                         ("ood_inviscid", inviscid_loader)):
        metrics = run_epoch(loader=loader, split_name=name, **common)
        summary = loader.dataset.describe() if hasattr(loader.dataset, "describe") else {}
        out["groups"][name] = {"metrics": metrics, "dataset": summary}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--train-root", required=True, type=Path)
    ap.add_argument("--truth-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--deterministic-pairing", action="store_true")
    args = ap.parse_args()
    result = evaluate(args.checkpoint, args.train_root, args.truth_root, args.device,
                      args.batch_size, args.num_workers, args.deterministic_pairing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
