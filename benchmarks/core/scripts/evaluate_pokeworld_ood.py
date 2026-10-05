#!/usr/bin/env python3
"""Frozen support-selected evaluation for PokeWorld unseen-physics runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from persistent_jepa.evaluation import r2, select_ridge, system_mean
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
from persistent_jepa.torch_data import HORIZONS


PAIRING_SEED = 20260828
SPLIT_OFFSET = {"train": 0, "val": 100_000, "ood": 200_000}


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(y), rankdata(prediction))[0, 1])


def regression_metrics(y: np.ndarray, prediction: np.ndarray, gamma_range: float) -> dict:
    residual = np.asarray(prediction) - np.asarray(y)
    return {
        "r2": r2(y, prediction),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "nrmse_by_evaluation_gamma_range": float(np.sqrt(np.square(residual).mean()) / gamma_range),
        "spearman": spearman(y, prediction),
    }


def per_row_state_mse(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.square(np.asarray(target) - np.asarray(prediction)).mean(axis=1)


def mismatch_quartiles(
    mismatch: np.ndarray,
    shuffled_error: np.ndarray,
    correct_error: np.ndarray,
) -> list[dict]:
    """Quartiles are deliberately defined on shuffled donor pairs only."""
    edges = np.quantile(mismatch, [0.0, 0.25, 0.5, 0.75, 1.0])
    output = []
    for index in range(4):
        selected = (mismatch >= edges[index]) & (
            mismatch <= edges[index + 1] if index == 3 else mismatch < edges[index + 1]
        )
        output.append({
            "quartile": f"Q{index + 1}",
            "absolute_gamma_mismatch_range": [float(mismatch[selected].min()), float(mismatch[selected].max())],
            "mean_absolute_gamma_mismatch": float(mismatch[selected].mean()),
            "median_absolute_gamma_mismatch": float(np.median(mismatch[selected])),
            "pairs": int(selected.sum()),
            "mean_shuffled_object_h16_state_mse": float(shuffled_error[selected].mean()),
            "mean_degradation_vs_correct": float((shuffled_error[selected] - correct_error[selected]).mean()),
        })
    return output


def _variant(config: dict) -> str:
    value = config.get("variant", "B0")
    return "B0" if value == "vision_only_multi_horizon_JEPA" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 20_000:
        raise ValueError("OOD evaluation is registered only for the frozen 20k checkpoint")
    config = checkpoint["config"]
    variant = _variant(config)
    history_length = int(config.get("history_length", 24))
    data_manifest = json.loads((args.data_root / "manifest.json").read_text())
    if "ood" not in data_manifest["splits"]:
        raise ValueError("dataset lacks a sealed OOD split")
    ood_gamma_range = float(sum(
        high - low for low, high in data_manifest["metadata"]["ood_gamma_intervals"]
    ))

    set_deterministic(PAIRING_SEED)
    device = torch.device(args.device)
    model = PokeJEPA(variant, history_length=history_length).to(device)
    model.load_state_dict(checkpoint["model"])
    splits = {
        name: PokeSplit(args.data_root, name, history_length=history_length)
        for name in ("train", "val", "ood")
    }
    arrays = {
        name: fixed_eval_arrays(
            split, PAIRING_SEED + SPLIT_OFFSET[name], include_targeted=False
        )
        for name, split in splits.items()
    }
    extracted = {
        name: extract_bridge(model, arrays[name], device, args.batch_size)
        for name in ("train", "val", "ood")
    }
    normalizers = state_normalizers(splits["train"])
    targets = {
        name: normalized_targets(extracted[name]["target_states"], normalizers)
        for name in extracted
    }

    decoder_ridge: dict[str, dict[str, float]] = {"object": {}, "full": {}}
    functional = {"val_support": {"object": {}, "full": {}}, "ood": {"object": {}, "full": {}}}
    h16_decoders = {}
    for target_name in ("object", "full"):
        for hi, horizon in enumerate(HORIZONS):
            decoder, alpha, _ = select_ridge(
                extracted["train"]["true_h"][:, hi], targets["train"][target_name][:, hi],
                extracted["val"]["true_h"][:, hi], targets["val"][target_name][:, hi],
            )
            decoder_ridge[target_name][f"h{horizon}"] = alpha
            for split_name, report_name in (("val", "val_support"), ("ood", "ood")):
                prediction = decoder.predict(extracted[split_name]["self_prediction_h"][:, hi])
                functional[report_name][target_name][f"h{horizon}"] = equal_system_components(
                    targets[split_name][target_name][:, hi], prediction, arrays[split_name].system_index
                )
            if horizon == 16:
                h16_decoders[target_name] = decoder

    representations = {
        name: system_mean(
            extracted[name]["representation"], arrays[name].system_index, splits[name].states.shape[0]
        )
        for name in extracted
    }
    train_gamma = np.asarray(splits["train"].gamma)
    val_gamma = np.asarray(splits["val"].gamma)
    ood_gamma = np.asarray(splits["ood"].gamma)
    probe, probe_alpha, _ = select_ridge(
        representations["train"], train_gamma, representations["val"], val_gamma
    )
    gamma_probe = {
        "ridge_alpha_selected_on_support_validation": probe_alpha,
        "val_support": regression_metrics(
            val_gamma, probe.predict(representations["val"]), float(val_gamma.max() - val_gamma.min())
        ),
        "ood": regression_metrics(
            ood_gamma, probe.predict(representations["ood"]), ood_gamma_range
        ),
    }

    donor = None
    if "correct_prediction_h16" in extracted["ood"]:
        correct_metrics, shuffled_metrics = {}, {}
        for target_name in ("object", "full"):
            decoder = h16_decoders[target_name]
            correct_prediction = decoder.predict(extracted["ood"]["correct_prediction_h16"])
            shuffled_prediction = decoder.predict(extracted["ood"]["shuffled_prediction_h16"])
            correct_metrics[target_name] = equal_system_components(
                targets["ood"][target_name][:, 2], correct_prediction, arrays["ood"].system_index
            )
            shuffled_metrics[target_name] = equal_system_components(
                targets["ood"][target_name][:, 2], shuffled_prediction, arrays["ood"].system_index
            )
        object_decoder = h16_decoders["object"]
        correct_prediction = object_decoder.predict(extracted["ood"]["correct_prediction_h16"])
        shuffled_prediction = object_decoder.predict(extracted["ood"]["shuffled_prediction_h16"])
        correct_error = per_row_state_mse(targets["ood"]["object"][:, 2], correct_prediction)
        shuffled_error = per_row_state_mse(targets["ood"]["object"][:, 2], shuffled_prediction)
        donor_system = arrays["ood"].system_index[arrays["ood"].shuffled_row_index]
        mismatch = np.abs(ood_gamma[arrays["ood"].system_index] - ood_gamma[donor_system])
        if not np.all(mismatch > 0):
            raise RuntimeError("shuffled donor pairs must come from different OOD systems")
        donor = {
            "correct_definition": "same OOD system, independent rollout; absolute gamma mismatch = 0",
            "shuffled_definition": "different OOD system, fixed derangement",
            "correct_h16": correct_metrics,
            "shuffled_h16": shuffled_metrics,
            "object_state_mse_gap": shuffled_metrics["object"]["state"] - correct_metrics["object"]["state"],
            "shuffled_only_absolute_gamma_mismatch_quartiles": mismatch_quartiles(
                mismatch, shuffled_error, correct_error
            ),
        }

    report = {
        "schema_version": "pokeworld-ood-evaluation-1.0",
        "protocol": data_manifest["metadata"]["protocol"],
        "variant": variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "selection_boundary": "all checkpoint, ridge, and decoder choices use support train/validation only",
        "probe_tensor": "B0_context_128" if variant == "B0" else "persistent_z_p_64",
        "gamma_probe": gamma_probe,
        "decoder_ridge_selected_on_support_validation": decoder_ridge,
        "functional": functional,
        "donor": donor,
        "aggregation": "mean windows within system then equal mean across systems",
        "ood_read": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
