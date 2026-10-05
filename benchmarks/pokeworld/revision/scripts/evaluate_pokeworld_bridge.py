#!/usr/bin/env python3
"""Frozen acquisition, functional, and donor evaluator for the R0 B3 bridge."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from persistent_jepa.evaluation import RIDGE_GRID, Ridge, mse, r2, system_mean
from persistent_jepa.poke_evaluation import (
    equal_system_components,
    extract_bridge,
    fixed_eval_arrays,
    normalized_targets,
    state_normalizers,
)
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file
from persistent_jepa.test_seal import require_test_unsealed
from persistent_jepa.torch_data import HORIZONS


PAIRING_SEED = 20260823
SPLIT_OFFSET = {"train": 0, "val": 100_000, "test": 200_000}
RIDGE_SELECTION_SEED = 20260903


def train_selected_ridge(train_x, train_y, system_index, fixed_alpha=None):
    """Select alpha only within training systems, then refit all train rows."""
    if fixed_alpha is not None:
        return Ridge(float(fixed_alpha)).fit(train_x, train_y), float(fixed_alpha)
    systems = np.unique(system_index)
    order = np.random.default_rng(RIDGE_SELECTION_SEED).permutation(systems)
    cut = max(1, int(0.8 * order.size))
    fit_systems, selection_systems = order[:cut], order[cut:]
    if not selection_systems.size:
        raise ValueError("ridge selection needs at least one held-out train system")
    fit = np.isin(system_index, fit_systems)
    selection = np.isin(system_index, selection_systems)
    scored = []
    for alpha in RIDGE_GRID:
        candidate = Ridge(alpha).fit(train_x[fit], train_y[fit])
        scored.append((mse(train_y[selection], candidate.predict(train_x[selection])), alpha))
    alpha = min(scored, key=lambda value: (value[0], value[1]))[1]
    return Ridge(alpha).fit(train_x, train_y), float(alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--probe-fit-root", type=Path,
        help="correctly grouped train root used for probe/decoder fitting; defaults to data-root",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--common-anchor-min",
        type=int,
        help="restrict train/eval windows to anchors >= this value (40 for the 24/32/40 ladder)",
    )
    return parser.parse_args()


def _variant(config: dict) -> str:
    value = config.get("variant", "B0")
    return "B0" if value == "vision_only_multi_horizon_JEPA" else value


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    selection = None
    marker = None
    if args.split == "test":
        if args.selection_json is None:
            raise ValueError("formal test requires --selection-json")
        selection = require_test_unsealed(args.selection_json)
        marker = args.selection_json.parent / f".formal_pokeworld_test_done_{args.selection_json.stem}"
        if marker.exists():
            raise RuntimeError("PokeWorld formal test was already read")
    elif args.selection_json is not None:
        raise ValueError("selection JSON is reserved for formal test")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    variant = _variant(config)
    history_length = int(config.get("history_length", 24))
    if int(checkpoint["step"]) not in {3000, 20000}:
        raise ValueError("only registered 3k/20k checkpoints may be evaluated")
    checkpoint_hash = sha256_file(args.checkpoint)
    if selection is not None:
        if checkpoint_hash != selection["checkpoint_sha256"]:
            raise RuntimeError("checkpoint differs from immutable selection")
        if variant != selection["variant"]:
            raise RuntimeError("variant differs from immutable selection")

    set_deterministic(PAIRING_SEED)
    device = torch.device(args.device)
    model = PokeJEPA(variant, history_length=history_length).to(device)
    model.load_state_dict(checkpoint["model"])
    probe_fit_root = args.probe_fit_root or args.data_root
    train = PokeSplit(probe_fit_root, "train", history_length=history_length)
    evaluation = PokeSplit(args.data_root, args.split, history_length=history_length)
    train_arrays = fixed_eval_arrays(
        train,
        PAIRING_SEED + SPLIT_OFFSET["train"],
        anchor_min=args.common_anchor_min,
    )
    eval_arrays = fixed_eval_arrays(
        evaluation,
        PAIRING_SEED + SPLIT_OFFSET[args.split],
        anchor_min=args.common_anchor_min,
    )
    train_extract = extract_bridge(model, train_arrays, device, args.batch_size)
    eval_extract = extract_bridge(model, eval_arrays, device, args.batch_size)
    normalizers = state_normalizers(train)
    train_targets = normalized_targets(train_extract["target_states"], normalizers)
    eval_targets = normalized_targets(eval_extract["target_states"], normalizers)

    decoder_ridge: dict[str, dict[str, float]] = {"object": {}, "full": {}}
    self_functional: dict[str, dict] = {"object": {}, "full": {}}
    h16_decoders = {}
    for target_name in ("object", "full"):
        for hi, horizon in enumerate(HORIZONS):
            fixed = None
            if selection is not None:
                fixed = float(selection["decoder_ridge"][target_name][f"h{horizon}"])
            decoder, alpha = train_selected_ridge(
                train_extract["true_h"][:, hi], train_targets[target_name][:, hi],
                train_arrays.system_index, fixed_alpha=fixed,
            )
            decoder_ridge[target_name][f"h{horizon}"] = alpha
            prediction = decoder.predict(eval_extract["self_prediction_h"][:, hi])
            self_functional[target_name][f"h{horizon}"] = equal_system_components(
                eval_targets[target_name][:, hi], prediction, eval_arrays.system_index
            )
            if horizon == 16:
                h16_decoders[target_name] = decoder

    donor = None
    if "correct_prediction_h16" in eval_extract:
        correct, shuffled = {}, {}
        for target_name in ("object", "full"):
            decoder = h16_decoders[target_name]
            correct[target_name] = equal_system_components(
                eval_targets[target_name][:, 2],
                decoder.predict(eval_extract["correct_prediction_h16"]),
                eval_arrays.system_index,
            )
            shuffled[target_name] = equal_system_components(
                eval_targets[target_name][:, 2],
                decoder.predict(eval_extract["shuffled_prediction_h16"]),
                eval_arrays.system_index,
            )
        targeted = {}
        object_decoder = h16_decoders["object"]
        correct_object_prediction = object_decoder.predict(eval_extract["correct_prediction_h16"])
        for label, item in eval_extract["targeted_prediction_h16"].items():
            rows = item["target_rows"]
            diagnostic = dict(eval_arrays.targeted_diagnostics[label])
            if rows.size:
                prediction = object_decoder.predict(item["prediction"])
                metrics = equal_system_components(
                    eval_targets["object"][rows, 2],
                    prediction,
                    eval_arrays.system_index[rows],
                )
                correct_metrics = equal_system_components(
                    eval_targets["object"][rows, 2],
                    correct_object_prediction[rows],
                    eval_arrays.system_index[rows],
                )
                diagnostic["object_h16"] = metrics
                diagnostic["state_mse_degradation_vs_correct"] = (
                    metrics["state"] - correct_metrics["state"]
                )
            targeted[label] = diagnostic
        donor = {
            "correct_h16": correct,
            "random_shuffled_h16": shuffled,
            "object_state_mse_gap": (
                shuffled["object"]["state"] - correct["object"]["state"]
            ),
            "gamma_targeted": targeted,
        }

    train_rep = system_mean(
        train_extract["representation"], train_arrays.system_index, train.states.shape[0]
    )
    eval_rep = system_mean(
        eval_extract["representation"], eval_arrays.system_index, evaluation.states.shape[0]
    )
    probes = {}
    for name, train_y, eval_y in (
        ("drag", np.asarray(train.gamma), np.asarray(evaluation.gamma)),
        ("mass", np.asarray(train.mass), np.asarray(evaluation.mass)),
        ("stiffness", np.asarray(train.stiffness), np.asarray(evaluation.stiffness)),
    ):
        fixed = None
        if selection is not None:
            fixed = float(selection["probe_ridge"][name])
        probe, alpha = train_selected_ridge(
            train_rep, train_y, np.arange(train_rep.shape[0]), fixed_alpha=fixed
        )
        probes[name] = {"ridge_alpha": alpha, "system_r2": r2(eval_y, probe.predict(eval_rep))}

    report = {
        "schema_version": "r0-bridge-1.0",
        "track": "R0-PokeWorld-B3-bridge",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(checkpoint["step"]),
        "variant": variant,
        "training_config": config,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "training_dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "probe_fit_dataset_manifest_sha256": sha256_file(probe_fit_root / "manifest.json"),
        "probe_fit_root": str(probe_fit_root),
        "pairing_seed": PAIRING_SEED,
        "ridge_selection": {
            "unit": "system", "fit_fraction": 0.8, "seed": RIDGE_SELECTION_SEED,
            "validation_selects_alpha": False, "alpha_grid": list(RIDGE_GRID),
        },
        "windows_per_system": 8,
        "history_length": history_length,
        "evaluation_anchor_min": args.common_anchor_min,
        "evaluation_anchor_max": 47,
        "probe_tensor": "B0_context_128" if variant == "B0" else "persistent_z_p_64",
        "probes": probes,
        "decoder_ridge": decoder_ridge,
        "self_functional": self_functional,
        "donor": donor,
        "aggregation": "mean_windows_within_system_then_equal_mean_systems",
        "test_read": args.split == "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    if marker is not None:
        marker.write_text(
            json.dumps(
                {
                    "selection_sha256": sha256_file(args.selection_json),
                    "report_sha256": sha256_file(args.output),
                    "created_at": report["created_at"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        marker.chmod(0o444)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
