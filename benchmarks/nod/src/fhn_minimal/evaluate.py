"""Rollout and two-factor accessibility evaluation for FHN NOD/SPRII checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fhn_minimal.data import CONDITION_FRAMES, SYSTEMS, TRAIN_SYSTEMS, load_trajectory, one_hot_head
from fhn_minimal.train import create_locs, set_seed


def load_model(model, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = OrderedDict(
        (name.replace("module.", "").replace("_orig_mod.", ""), value)
        for name, value in checkpoint["model_state_dict"].items()
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return checkpoint


def head_sequence(horizon: int) -> list[int]:
    remaining = horizon
    counts = {}
    for step in (9, 3, 1):
        counts[step], remaining = divmod(remaining, step)
    mapping = {1: 0, 3: 1, 9: 2}
    return [mapping[step] for step in (1, 3, 9) for _ in range(counts[step])]


@torch.no_grad()
def encode(model, trajectories: list[torch.Tensor], device: torch.device,
           batch_size: int = 8) -> np.ndarray:
    rows = []
    for start in range(0, len(trajectories), batch_size):
        histories = torch.stack([
            x[list(CONDITION_FRAMES)] for x in trajectories[start:start + batch_size]
        ]).to(device)
        rows.append(model.conditioning_encoder(histories).cpu().numpy())
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def rollout(model, initial: torch.Tensor, condition: torch.Tensor, frame: int,
            horizon: int, locs: torch.Tensor, device: torch.device) -> tuple[float, float]:
    x = initial[frame].unsqueeze(0).to(device)
    c = condition[list(CONDITION_FRAMES)].unsqueeze(0).to(device)
    sequence = head_sequence(horizon)
    locations = locs.unsqueeze(0)
    head = torch.tensor([sequence[0]], device=device)
    prediction = model(x, c, locations, one_hot_head(head).to(device))
    for index in sequence[1:]:
        head = torch.tensor([index], device=device)
        prediction = model.predict(prediction, model.latent_vector, one_hot_head(head).to(device))
    target = initial[frame + horizon].unsqueeze(0).to(device)
    mse = float(torch.mean((prediction - target).square()))
    l2 = float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target))
    return mse, l2


def ridge_probe(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray,
                test_y: np.ndarray, alpha: float = 1e-3) -> dict:
    z_mean, z_std = train_z.mean(0), train_z.std(0)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    z_std = np.maximum(z_std, 1e-8)
    y_std = np.maximum(y_std, 1e-8)
    x_train = (train_z - z_mean) / z_std
    t_train = (train_y - y_mean) / y_std
    x_design = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=1)
    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ t_train)
    x_test = np.concatenate(((test_z - z_mean) / z_std, np.ones((len(test_z), 1))), axis=1)
    prediction = (x_test @ weights) * y_std + y_mean
    residual = ((test_y - prediction) ** 2).sum(0)
    total = ((test_y - test_y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    rho = [float(spearmanr(test_y[:, i], prediction[:, i]).statistic) for i in range(2)]
    rmse = np.sqrt(np.mean((test_y - prediction) ** 2, axis=0))
    return {
        "r2_k": float(r2[0]), "r2_beta": float(r2[1]),
        "spearman_k": rho[0], "spearman_beta": rho[1],
        "rmse_k": float(rmse[0]), "rmse_beta": float(rmse[1]),
        "n": int(len(test_y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--official-code", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-id", type=int, default=36)
    parser.add_argument("--target-ids", default="5,15,24,45")
    parser.add_argument("--splits", default="train,ood-intra,ood-extra")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--probe-target-ids", default="5,15,24,36,45")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.official_code))
    from ngs.neuralnetworks import NGS_metaNet_Hier

    set_seed(args.seed)
    device = torch.device(args.device)
    model = NGS_metaNet_Hier(64, 3, 2, 128).to(device)
    checkpoint = load_model(model, args.checkpoint, device)
    locs = create_locs(device)
    target_ids = [int(x) for x in args.target_ids.split(",")]
    selected_splits = [x for x in args.splits.split(",") if x]
    unknown_splits = sorted(set(selected_splits) - set(SYSTEMS))
    if unknown_splits:
        raise ValueError(f"unknown splits: {unknown_splits}")

    cases = []
    for split in selected_splits:
        systems = SYSTEMS[split]
        for system_index, system in enumerate(systems):
            condition = load_trajectory(args.data_dir, system, args.condition_id)
            for initial_id in target_ids:
                initial = load_trajectory(args.data_dir, system, initial_id)
                for frame in (12, 42, 72, 92):
                    for horizon in (1, 5, 50):
                        if frame + horizon > 100:
                            continue
                        mse, l2 = rollout(model, initial, condition, frame, horizon, locs, device)
                        cases.append({
                            "split": split, "system_index": system_index,
                            "k": system[0], "beta": system[1], "initial_id": initial_id,
                            "condition_id": args.condition_id, "frame": frame,
                            "horizon": horizon, "mse": mse, "l2_relative": l2,
                        })

    aggregate = []
    for split in selected_splits:
        for horizon in (1, 5, 50):
            selected = [row for row in cases if row["split"] == split and row["horizon"] == horizon]
            for metric in ("mse", "l2_relative"):
                values = np.asarray([row[metric] for row in selected])
                aggregate.append({
                    "split": split, "horizon": horizon, "metric": metric,
                    "mean": float(values.mean()), "std": float(values.std()),
                    "n": int(len(values)),
                })

    probes = {}
    geometry = {}
    if not args.skip_probe:
        train_trajectories, train_parameters = [], []
        for system in TRAIN_SYSTEMS:
            for initial_id in range(50, 90):
                train_trajectories.append(load_trajectory(args.data_dir, system, initial_id))
                train_parameters.append(system)
        train_z = encode(model, train_trajectories, device)
        train_y = np.asarray(train_parameters, dtype=np.float64)
        standardized = (train_z - train_z.mean(0)) / np.maximum(train_z.std(0), 1e-8)
        groups = standardized.reshape(len(TRAIN_SYSTEMS), 40, -1)
        means = groups.mean(1)
        within = float(((groups - means[:, None]) ** 2).sum(-1).mean())
        between = float(((means - means.mean(0)) ** 2).sum(-1).mean())
        geometry = {"statistics_source": "training trajectories only", "within": within,
                    "between": between, "between_within": between / max(within, 1e-12)}
        for split in selected_splits:
            trajectories, parameters = [], []
            for system in SYSTEMS[split]:
                for initial_id in [int(x) for x in args.probe_target_ids.split(',')]:
                    trajectories.append(load_trajectory(args.data_dir, system, initial_id))
                    parameters.append(system)
            probes[split] = ridge_probe(
                train_z, train_y, encode(model, trajectories, device),
                np.asarray(parameters, dtype=np.float64),
            )

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", checkpoint.get("epoch", -1))),
        "evaluation_protocol": {
            "target_ids": target_ids, "condition_id": args.condition_id,
            "frames": [12, 42, 72, 92], "horizons": [1, 5, 50],
            "splits": selected_splits,
            "independent_conditioning": args.condition_id not in target_ids,
            "probe_target_ids": [int(x) for x in args.probe_target_ids.split(',')],
        },
        "aggregate": aggregate, "probe": probes, "geometry": geometry, "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    print(json.dumps({"aggregate": aggregate, "probe": probes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
