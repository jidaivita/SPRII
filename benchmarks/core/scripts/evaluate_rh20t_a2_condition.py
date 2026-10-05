#!/usr/bin/env python3
"""Evaluate one frozen A2 condition on the shared validation Q-star bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from persistent_jepa.rh20t_data import RH20TSplit
from persistent_jepa.rh20t_model import HORIZONS, RH20TJEPA, RH20TModelConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batches(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def stack(windows: list[dict], name: str, device: torch.device) -> torch.Tensor | None:
    if windows[0][name] is None:
        return None
    return torch.from_numpy(np.stack([row[name] for row in windows]).astype(np.float32)).to(device)


def tensors(split: RH20TSplit, rows: list[dict], device: torch.device, *, donor: str | None = None) -> dict:
    windows = []
    for row in rows:
        if donor is None:
            windows.append(split._window(row["query_episode_id"], int(row["query_anchor"])))
        else:
            windows.append(split._window(
                row["donor_episode_ids"][donor], int(row["donor_anchor"]),
                include_targets=False,
            ))
    names = ["history_image", "history_lowdim", "history_actions"]
    if donor is None:
        names += ["future_actions", "action_masks", "target_force", "target_tcp_xyz"]
    result = {name: stack(windows, name, device) for name in names}
    for name in names:
        if name != "history_image" and result[name] is None:
            raise RuntimeError(f"missing tensor {name}")
    return result


def load_model(
    path: Path, condition: str, seed: int, expected_step: int, device: torch.device
) -> RH20TJEPA:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("step") != expected_step:
        raise ValueError(f"checkpoint is not fixed expected step {expected_step}: {path}")
    config = payload.get("config", {})
    if config.get("condition") != condition or int(config.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint identity mismatch: {path}")
    if config.get("test_read") is not False:
        raise ValueError(f"checkpoint test guard invalid: {path}")
    model = RH20TJEPA(RH20TModelConfig(condition)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def errors(
    model: RH20TJEPA, context: torch.Tensor, query: dict, tcp_std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    h4_index = HORIZONS.index(4)
    h16_index = HORIZONS.index(16)
    h4 = torch.full((context.shape[0],), h4_index, dtype=torch.long, device=context.device)
    h16 = torch.full((context.shape[0],), h16_index, dtype=torch.long, device=context.device)
    out4 = model.predictor(context, query["future_actions"][:, h4_index], query["action_masks"][:, h4_index], h4)
    out16 = model.predictor(context, query["future_actions"][:, h16_index], query["action_masks"][:, h16_index], h16)
    force = (out4["force"] - query["target_force"][:, h4_index]).square().mean(dim=-1)
    tcp = 100.0 * torch.linalg.vector_norm(
        (out16["tcp_xyz"] - query["target_tcp_xyz"][:, h16_index]) * tcp_std, dim=-1
    )
    return force, tcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-label", choices=("B3", "Mono-QD"), required=True)
    parser.add_argument("--model-condition", choices=("B3-Indep", "Mono-QD-Indep"), required=True)
    parser.add_argument("--checkpoint-template", required=True, help="format string containing {seed}")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pairing-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--expected-step", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    expected = {"B3": "B3-Indep", "Mono-QD": "Mono-QD-Indep"}
    if expected[args.condition_label] != args.model_condition:
        raise ValueError("condition label/model condition mismatch")
    bank = json.loads(args.bank.read_text())
    if bank.get("status") != "FROZEN_A2_VALIDATION_BANK_NO_MODEL_OUTPUT":
        raise RuntimeError("unfrozen A2 validation bank")
    if bank.get("test_read") is not False or bank.get("split") != "validation":
        raise RuntimeError("A2 evaluator is validation-only")
    device = torch.device(args.device)
    split = RH20TSplit(
        args.cache, args.split_manifest, args.pairing_manifest, args.normalization,
        "validation", args.model_condition,
    )
    norm = json.loads(args.normalization.read_text())
    tcp_std = torch.tensor(norm["tcp"]["std"][:3], dtype=torch.float32, device=device)
    primary_rows, secondary_rows, checkpoint_hashes = [], [], {}
    categories = tuple(bank["categories"])

    for seed in args.seeds:
        checkpoint = Path(args.checkpoint_template.format(seed=seed))
        model = load_model(checkpoint, args.model_condition, seed, args.expected_step, device)
        checkpoint_hashes[str(seed)] = sha256(checkpoint)
        with torch.inference_mode():
            for task_id in sorted(bank["tasks"]):
                for batch_rows in batches(bank["tasks"][task_id], args.batch_size):
                    query = tensors(split, batch_rows, device)
                    query_h = model.observation(query["history_image"], query["history_lowdim"])
                    z_s = None
                    if not model.cfg.monolithic_qd:
                        z_s, _z_p, _context = model.codes(query_h, query["history_actions"])
                        if z_s is None:
                            raise RuntimeError("B3 model lacks transient branch")
                    for category in categories:
                        donor = tensors(split, batch_rows, device, donor=category)
                        donor_h = model.observation(donor["history_image"], donor["history_lowdim"])
                        if model.cfg.monolithic_qd:
                            if model.joint_context is None:
                                raise RuntimeError("Mono-QD model lacks joint context")
                            context = model.joint_context(
                                query_h, query["history_actions"], donor_h, donor["history_actions"]
                            )
                        else:
                            if model.persistent is None:
                                raise RuntimeError("B3 model lacks persistent branch")
                            z_p = model.persistent(donor_h, donor["history_actions"])
                            context = torch.cat([z_s, z_p], dim=-1)
                        force, tcp = errors(model, context, query, tcp_std)
                        for index, row in enumerate(batch_rows):
                            base = {
                                "condition": args.condition_label,
                                "seed": seed,
                                "task_id": task_id,
                                "query_id": row["query_id"],
                                "donor_category": category,
                            }
                            primary_rows.append({**base, "error": float(force[index].cpu())})
                            secondary_rows.append({**base, "error": float(tcp[index].cpu())})

    payload = {
        "schema_version": "1.1b",
        "status": "FROZEN_A2_CONDITION_VALIDATION_RECORDS",
        "split": "validation",
        "test_read": False,
        "condition": args.condition_label,
        "model_condition": args.model_condition,
        "checkpoint_step": args.expected_step,
        "seeds": args.seeds,
        "bank_sha256": sha256(args.bank),
        "checkpoint_sha256": checkpoint_hashes,
        "primary_endpoint": "h4_normalized_force_torque_mse",
        "secondary_endpoint": "h16_tcp_xyz_error_cm",
        "rows": primary_rows,
        "secondary_rows": secondary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
