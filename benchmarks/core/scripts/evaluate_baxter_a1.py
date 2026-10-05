#!/usr/bin/env python3
"""Evaluate registered A1-Baxter persistent-organization endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from persistent_jepa.baxter_data import BaxterSplit, CONFIGS, MEASURED_HARDNESS
from persistent_jepa.baxter_model import BaxterJEPA, BaxterModelConfig
from persistent_jepa.runtime import atomic_json, sha256_file


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray, classes: tuple[int, ...]) -> float:
    return float(np.mean([np.mean(prediction[target == value] == value) for value in classes]))


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=0, solver="lbfgs"),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x)


def encode(model: BaxterJEPA, data: BaxterSplit, device: torch.device) -> dict[str, np.ndarray]:
    rows = list(data.iter_records())
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            part = rows[start : start + 128]
            value = torch.from_numpy(np.stack([item[1][:40] for item in part])).to(device)
            history_h = model.observation(value)
            embeddings.append(model.persistent(history_h).float().cpu().numpy())
    return {
        "z": np.concatenate(embeddings),
        "shape": np.asarray([0 if item[0].shape == "cube" else 1 for item in rows]),
        "hardness": np.asarray([item[0].hardness_level for item in rows]),
        "config": np.asarray([item[0].config_id for item in rows]),
    }


def rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def partial_spearman(target: np.ndarray, distance: np.ndarray, control: np.ndarray) -> float:
    y, x, c = rankdata(target), rankdata(distance), rankdata(control)
    design = np.column_stack([np.ones(len(c)), c])
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    denominator = np.linalg.norm(y_residual) * np.linalg.norm(x_residual)
    return float(np.dot(y_residual, x_residual) / denominator) if denominator > 0 else float("nan")


def geometry(encoded: dict[str, np.ndarray]) -> dict:
    centroids = {config: encoded["z"][encoded["config"] == config].mean(axis=0) for config in CONFIGS}
    distances, hardness_difference, shape_mismatch = [], [], []
    for left_index, left in enumerate(CONFIGS):
        for right in CONFIGS[left_index + 1 :]:
            distances.append(float(np.linalg.norm(centroids[left] - centroids[right])))
            hardness_difference.append(abs(MEASURED_HARDNESS[left] - MEASURED_HARDNESS[right]))
            shape_mismatch.append(float(left.split("_h")[0] != right.split("_h")[0]))
    distance = np.asarray(distances)
    hardness = np.asarray(hardness_difference)
    shape = np.asarray(shape_mismatch)
    return {
        "hardness_partial_spearman": partial_spearman(hardness, distance, shape),
        "shape_partial_spearman": partial_spearman(shape, distance, hardness),
        "configuration_centroids": {key: value.tolist() for key, value in centroids.items()},
        "pairwise_distances": distances,
        "physical_configurations": 6,
        "pairwise_distances_are_not_iid": True,
    }


def endpoints(train: dict[str, np.ndarray], evaluation: dict[str, np.ndarray]) -> dict:
    hardness_scores = []
    for train_shape, test_shape in ((0, 1), (1, 0)):
        train_mask = train["shape"] == train_shape
        test_mask = evaluation["shape"] == test_shape
        prediction = fit_predict(train["z"][train_mask], train["hardness"][train_mask], evaluation["z"][test_mask])
        hardness_scores.append(
            balanced_accuracy(evaluation["hardness"][test_mask], prediction, (0, 1, 2))
        )
    shape_scores = []
    for held_out_hardness in (0, 1, 2):
        train_mask = train["hardness"] != held_out_hardness
        test_mask = evaluation["hardness"] == held_out_hardness
        prediction = fit_predict(train["z"][train_mask], train["shape"][train_mask], evaluation["z"][test_mask])
        shape_scores.append(balanced_accuracy(evaluation["shape"][test_mask], prediction, (0, 1)))
    return {
        "g_h_cross_shape_macro_accuracy": float(np.mean(hardness_scores)),
        "g_h_directional_scores": hardness_scores,
        "g_s_leave_one_hardness_out_balanced_accuracy": float(np.mean(shape_scores)),
        "g_s_fold_scores": shape_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirmation-access-receipt", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if args.split == "confirmation":
        if args.confirmation_access_receipt is None:
            raise ValueError("confirmation evaluation requires the frozen access receipt")
        receipt = json.loads(args.confirmation_access_receipt.read_text())
        if receipt.get("status") != "A1_CONFIRMATION_ACCESS_AUTHORIZED":
            raise ValueError("invalid confirmation access receipt")
    device = torch.device(args.device)
    model = BaxterJEPA(BaxterModelConfig(config["condition"])).to(device)
    model.load_state_dict(payload["model"])
    train_data = BaxterSplit(args.data_root, args.manifest, args.normalization, "train", config["condition"])
    evaluation_data = BaxterSplit(args.data_root, args.manifest, args.normalization, args.split, config["condition"])
    train_encoded = encode(model, train_data, device)
    evaluation_encoded = encode(model, evaluation_data, device)
    result = {
        "schema_version": "paper-a-a1-baxter-evaluation-v1.0",
        "status": "FROZEN_A1_RELATION_EVALUATION",
        "condition": config["condition"],
        "seed": config["seed"],
        "split": args.split,
        "primary": endpoints(train_encoded, evaluation_encoded),
        "secondary_geometry": geometry(evaluation_encoded),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "train_records": len(train_encoded["z"]),
        "evaluation_records": len(evaluation_encoded["z"]),
        "inference_boundary": "held_out_grasps_from_six_known_configurations",
        "confirmation_accessed": args.split == "confirmation",
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
