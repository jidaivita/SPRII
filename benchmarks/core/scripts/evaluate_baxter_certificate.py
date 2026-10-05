#!/usr/bin/env python3
"""Evaluate one frozen A1-Baxter input-accessibility certificate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persistent_jepa.baxter_data import BaxterSplit
from persistent_jepa.baxter_model import BaxterCertificate
from persistent_jepa.runtime import atomic_json, sha256_file


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray, classes: tuple[int, ...]) -> float:
    return float(np.mean([np.mean(prediction[target == value] == value) for value in classes]))


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
        receipt = __import__("json").loads(args.confirmation_access_receipt.read_text())
        if receipt.get("status") != "A1_CONFIRMATION_ACCESS_AUTHORIZED":
            raise ValueError("invalid confirmation access receipt")
    input_length = int(config["input_length"])
    device = torch.device(args.device)
    model = BaxterCertificate(input_length).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    data = BaxterSplit(args.data_root, args.manifest, args.normalization, args.split, "Random")
    hardness_target, hardness_prediction, shape_target, shape_prediction = [], [], [], []
    rows = list(data.iter_records())
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            part = rows[start : start + 128]
            value = torch.from_numpy(np.stack([item[1][:input_length] for item in part])).to(device)
            hardness_logits, shape_logits = model(value)
            hardness_target.extend(item[0].hardness_level for item in part)
            shape_target.extend(0 if item[0].shape == "cube" else 1 for item in part)
            hardness_prediction.extend(hardness_logits.argmax(dim=-1).cpu().tolist())
            shape_prediction.extend(shape_logits.argmax(dim=-1).cpu().tolist())
    hardness_target_array = np.asarray(hardness_target)
    shape_target_array = np.asarray(shape_target)
    result = {
        "schema_version": "paper-a-a1-baxter-certificate-eval-v1.0",
        "status": "FROZEN_CERTIFICATE_EVALUATION",
        "variant": config["variant"],
        "seed": config["seed"],
        "split": args.split,
        "hardness_macro_accuracy": balanced_accuracy(
            hardness_target_array, np.asarray(hardness_prediction), (0, 1, 2)
        ),
        "shape_balanced_accuracy": balanced_accuracy(
            shape_target_array, np.asarray(shape_prediction), (0, 1)
        ),
        "records": len(rows),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "confirmation_accessed": args.split == "confirmation",
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
