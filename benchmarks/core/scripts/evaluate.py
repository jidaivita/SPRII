#!/usr/bin/env python3
"""Frozen validation/test evaluator with decoder and donor controls."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from persistent_jepa.evaluation import (
    HORIZONS,
    _windows_for_split,
    extract,
    mlp_probe,
    r2,
    select_ridge,
    state_normalizer,
    system_mean,
)
from persistent_jepa.model import ModelConfig, PersistentJEPA
from persistent_jepa.runtime import atomic_json, jsonable, set_deterministic, sha256_file
from persistent_jepa.test_seal import require_test_unsealed
from persistent_jepa.torch_data import SplitArrays


PAIRING_SEED = 20260819
SPLIT_SEED_OFFSET = {"train": 0, "val": 100_000, "test": 200_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--probe-fit-root", type=Path,
        help="frozen original train bank; defaults to data-root",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def component_metrics(
    truth: np.ndarray, prediction: np.ndarray, system_index: np.ndarray, systems: int
) -> dict[str, float]:
    squared = np.square(prediction - truth)
    by_system = system_mean(squared, system_index, systems)
    return {
        "state": float(by_system.mean(axis=0).mean()),
        "position": float(by_system.mean(axis=0)[:2].mean()),
        "velocity": float(by_system.mean(axis=0)[2:].mean()),
    }


def main() -> None:
    args = parse_args()
    selection = None
    marker = None
    if args.split == "test":
        if args.selection_json is None:
            raise ValueError("test evaluation requires --selection-json")
        selection = require_test_unsealed(args.selection_json)
        marker = args.selection_json.parent / f".formal_test_evaluation_done_{args.selection_json.stem}"
        if marker.exists():
            raise RuntimeError("formal test evaluation already completed; second read is forbidden")
        if sha256_file(args.checkpoint) != selection["checkpoint_sha256"]:
            raise RuntimeError("checkpoint does not match immutable selection.json")
    elif args.selection_json is not None:
        raise ValueError("--selection-json is reserved for the formal test evaluation")

    set_deterministic(PAIRING_SEED)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) not in {3000, 20000}:
        raise RuntimeError("only registered 3k/20k checkpoints may be evaluated")
    train_config = checkpoint["config"]
    variant = train_config["variant"]
    if selection is not None and variant != selection["variant"]:
        raise RuntimeError("checkpoint variant differs from selection.json")
    model = PersistentJEPA(ModelConfig(**train_config["model"])).to(device)
    model.load_state_dict(checkpoint["model"])

    probe_fit_root = args.probe_fit_root or args.data_root
    train = SplitArrays(probe_fit_root, "train")
    evaluation = SplitArrays(args.data_root, args.split)
    train_arrays = _windows_for_split(train, PAIRING_SEED + SPLIT_SEED_OFFSET["train"])
    eval_arrays = _windows_for_split(
        evaluation, PAIRING_SEED + SPLIT_SEED_OFFSET[args.split]
    )
    train_extract = extract(model, train_arrays, device, args.batch_size)
    eval_extract = extract(model, eval_arrays, device, args.batch_size)
    state_mean, state_std = state_normalizer(train)
    train_target = (train_extract["target_states"] - state_mean) / state_std
    eval_target = (eval_extract["target_states"] - state_mean) / state_std

    decoder_alphas = {}
    decoder_true_embedding_mse = {}
    self_metrics = {}
    correct_metrics = None
    shuffled_metrics = None
    for hi, horizon in enumerate(HORIZONS):
        fixed_alpha = None
        if selection is not None:
            fixed_alpha = float(selection["decoder_ridge"][f"h{horizon}"])
        decoder, alpha, true_mse = select_ridge(
            train_extract["true_h"][:, hi],
            train_target[:, hi],
            eval_extract["true_h"][:, hi],
            eval_target[:, hi],
            fixed_alpha=fixed_alpha,
        )
        decoder_alphas[f"h{horizon}"] = alpha
        decoder_true_embedding_mse[f"h{horizon}"] = true_mse
        self_state = decoder.predict(eval_extract["self_prediction_h"][:, hi])
        self_metrics[f"h{horizon}"] = component_metrics(
            eval_target[:, hi], self_state, eval_arrays.system_index, evaluation.states.shape[0]
        )
        if horizon == 16 and "correct_prediction_h16" in eval_extract:
            correct_state = decoder.predict(eval_extract["correct_prediction_h16"])
            shuffled_state = decoder.predict(eval_extract["shuffled_prediction_h16"])
            correct_metrics = component_metrics(
                eval_target[:, hi], correct_state, eval_arrays.system_index, evaluation.states.shape[0]
            )
            shuffled_metrics = component_metrics(
                eval_target[:, hi], shuffled_state, eval_arrays.system_index, evaluation.states.shape[0]
            )

    train_system_rep = system_mean(
        train_extract["representation"], train_arrays.system_index, train.states.shape[0]
    )
    eval_system_rep = system_mean(
        eval_extract["representation"], eval_arrays.system_index, evaluation.states.shape[0]
    )
    fixed_probe_alpha = float(selection["probe_ridge"]) if selection is not None else None
    probe, probe_alpha, _ = select_ridge(
        train_system_rep,
        np.asarray(train.gamma),
        eval_system_rep,
        np.asarray(evaluation.gamma),
        fixed_alpha=fixed_probe_alpha,
    )
    ridge_prediction = probe.predict(eval_system_rep)
    mlp_prediction = mlp_probe(
        train_system_rep, np.asarray(train.gamma), eval_system_rep, seed=PAIRING_SEED
    )

    donor = None
    if correct_metrics is not None and shuffled_metrics is not None:
        correct_h = eval_extract["correct_prediction_h16"]
        shuffled_h = eval_extract["shuffled_prediction_h16"]
        # Embedding-space degradation is used only for the mismatch-shape diagnostic.
        degradation = np.square(shuffled_h - eval_extract["true_h"][:, 2]).mean(1) - np.square(
            correct_h - eval_extract["true_h"][:, 2]
        ).mean(1)
        gamma_per_row = np.asarray(evaluation.gamma)[eval_arrays.system_index]
        shuffled_gamma = gamma_per_row[eval_arrays.shuffled_row_index]
        mismatch = np.abs(gamma_per_row - shuffled_gamma)
        correlation = float(np.corrcoef(mismatch, degradation)[0, 1])
        donor = {
            "correct_h16": correct_metrics,
            "shuffled_h16": shuffled_metrics,
            "state_mse_gap_shuffled_minus_correct": shuffled_metrics["state"] - correct_metrics["state"],
            "gamma_mismatch_vs_embedding_error_degradation_pearson": correlation,
        }

    report = {
        "schema_version": "1.0",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "variant": variant,
        "training_config": train_config,
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "probe_fit_dataset_manifest_sha256": sha256_file(probe_fit_root / "manifest.json"),
        "probe_fit_root": str(probe_fit_root),
        "pairing_seed": PAIRING_SEED,
        "windows_per_system": 8,
        "decoder_ridge": decoder_alphas,
        "decoder_true_embedding_mse": decoder_true_embedding_mse,
        "self_functional": self_metrics,
        "donor": donor,
        "drag_probe": {
            "ridge_alpha": probe_alpha,
            "ridge_system_r2": r2(np.asarray(evaluation.gamma), ridge_prediction),
            "mlp_system_r2": r2(np.asarray(evaluation.gamma), mlp_prediction),
        },
        "aggregation": "mean_windows_within_system_then_equal_mean_systems",
        "test_read": args.split == "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    if marker is not None:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"report": str(args.output), "sha256": sha256_file(args.output)}) + "\n")
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
