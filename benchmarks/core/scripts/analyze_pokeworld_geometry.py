#!/usr/bin/env python3
"""Frozen multi-seed representation-geometry analysis for PokeWorld."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
import torch

from persistent_jepa.evaluation import RIDGE_GRID, Ridge, r2
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file


ANALYSIS_SEED = 20260830


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--analysis-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--smoke-systems", type=int)
    return p.parse_args()


@torch.inference_mode()
def extract_codes(
    model: PokeJEPA,
    data: PokeSplit,
    systems_index: np.ndarray,
    rollout_ids: np.ndarray,
    anchors: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    systems, rollouts = systems_index.size, rollout_ids.size
    s = np.repeat(systems_index, rollouts * anchors.size)
    r = np.tile(np.repeat(rollout_ids, anchors.size), systems)
    a = np.tile(anchors, systems * rollouts)
    z_p, z_s = [], []
    model.eval()
    for start in range(0, s.size, batch_size):
        stop = min(start + batch_size, s.size)
        batch = data._from_indices(s[start:stop], r[start:stop], a[start:stop]).to(device)
        history_h, _ = model.encode_batch(batch)
        transient, persistent, _ = model.codes(history_h, batch.history_actions)
        if persistent is None or transient is None:
            raise ValueError("geometry requires split persistent/transient branches")
        z_p.append(persistent.float().cpu().numpy())
        z_s.append(transient.float().cpu().numpy())
    shape_p = (systems, rollouts, anchors.size, -1)
    return {
        "z_p_window": np.concatenate(z_p).reshape(shape_p),
        "z_s_window": np.concatenate(z_s).reshape(shape_p),
    }


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(0, keepdims=True)
    eigen = np.linalg.eigvalsh(centered.T @ centered / max(1, centered.shape[0] - 1))
    eigen = np.clip(eigen, 0.0, None)
    probability = eigen / max(float(eigen.sum()), 1e-12)
    entropy = -np.sum(probability[probability > 0] * np.log(probability[probability > 0]))
    return float(np.exp(entropy))


def standardize(train_rollout: np.ndarray, val_rollout: np.ndarray) -> tuple[np.ndarray, dict]:
    flat = train_rollout.reshape(-1, train_rollout.shape[-1]).astype(np.float64)
    mean = flat.mean(0)
    std = flat.std(0).clip(1e-6)
    return (val_rollout - mean) / std, {
        "train_mean_norm": float(np.linalg.norm(mean)),
        "train_std_min": float(std.min()),
        "train_std_mean": float(std.mean()),
    }


def partial_spearman(distance: np.ndarray, gamma: np.ndarray, mass: np.ndarray, stiffness: np.ndarray) -> float:
    y = rankdata(distance)
    x = rankdata(gamma)
    controls = np.column_stack([rankdata(mass), rankdata(stiffness)])
    design = np.column_stack([np.ones(controls.shape[0]), controls])
    yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    xr = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return float(np.corrcoef(xr, yr)[0, 1])


def fit_gamma_direction(train_centroid: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(ANALYSIS_SEED)
    order = rng.permutation(train_centroid.shape[0])
    cut = int(0.8 * order.size)
    fit, hold = order[:cut], order[cut:]
    scored = []
    for alpha in RIDGE_GRID:
        model = Ridge(alpha).fit(train_centroid[fit], gamma[fit])
        error = np.mean(np.square(model.predict(train_centroid[hold]) - gamma[hold]))
        scored.append((float(error), float(alpha)))
    alpha = min(scored)[1]
    model = Ridge(alpha).fit(train_centroid, gamma)
    direction = np.asarray(model.weight[:, 0], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-12)
    return direction, alpha


def branch_metrics(rollout: np.ndarray, gamma: np.ndarray, mass: np.ndarray, stiffness: np.ndarray,
                   direction: np.ndarray) -> dict:
    centroid = rollout.mean(1)
    systems, rollouts = rollout.shape[:2]
    within = []
    for i in range(systems):
        within.extend(np.square(pdist(rollout[i], metric="euclidean")))
    within_value = float(np.mean(within))
    between_distance = pdist(centroid, metric="euclidean")
    between_value = float(np.mean(np.square(between_distance)))
    dg = pdist(gamma[:, None], metric="cityblock")
    dm = pdist(np.log(mass)[:, None], metric="cityblock")
    dk = pdist(np.log(stiffness)[:, None], metric="cityblock")
    rho = float(spearmanr(between_distance, dg).statistic)
    partial = partial_spearman(between_distance, dg, dm, dk)
    covariance = np.cov(centroid, rowvar=False)
    concentration = float(np.var(centroid @ direction, ddof=1) / max(np.trace(covariance), 1e-12))
    return {
        "within_system_dispersion": within_value,
        "between_system_centroid_separation": between_value,
        "fisher_ratio": between_value / max(within_value, 1e-12),
        "gamma_distance_spearman": rho,
        "gamma_distance_partial_spearman": partial,
        "gamma_direction_concentration": concentration,
    }


def bootstrap_metrics(rollout: np.ndarray, gamma: np.ndarray, mass: np.ndarray, stiffness: np.ndarray,
                      direction: np.ndarray, draws: int) -> dict:
    rng = np.random.default_rng(ANALYSIS_SEED + 17)
    names = list(branch_metrics(rollout, gamma, mass, stiffness, direction))
    values = {name: [] for name in names}
    for _ in range(draws):
        index = rng.integers(0, rollout.shape[0], size=rollout.shape[0])
        result = branch_metrics(rollout[index], gamma[index], mass[index], stiffness[index], direction)
        for name, value in result.items():
            values[name].append(value)
    return {name: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for name, v in values.items()}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    set_deterministic(ANALYSIS_SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bank = json.loads(args.bank_manifest.read_text())
    if bank["schema_version"] != "pokeworld-analysis-bank-1.0":
        raise ValueError(f"unsupported bank schema {bank['schema_version']}")
    anchors = np.asarray(bank["geometry"]["anchors"], dtype=np.int64)
    rollout_ids = np.asarray(bank["geometry"]["rollout_ids"], dtype=np.int64)
    config = checkpoint["config"]
    variant = config["variant"]
    if variant not in {"B2", "B3", "Bx"}:
        raise ValueError(f"unsupported variant {variant}")
    device = torch.device(args.device)
    model = PokeJEPA(variant, history_length=int(config.get("history_length", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    train = PokeSplit(args.analysis_root, "train", history_length=24)
    val = PokeSplit(args.analysis_root, "val", history_length=24)
    manifest_hash = sha256_file(args.analysis_root / "manifest.json")
    if manifest_hash != bank["dataset_manifest_sha256"]:
        raise ValueError("analysis dataset differs from frozen bank manifest")
    train_systems = np.asarray(bank["geometry"]["train_system_indices"], dtype=np.int64)
    val_systems = np.asarray(bank["geometry"]["val_system_indices"], dtype=np.int64)
    smoke = args.smoke_systems is not None
    if smoke:
        if args.smoke_systems < 8:
            raise ValueError("smoke-systems must be at least 8")
        train_systems = train_systems[: args.smoke_systems]
        val_systems = val_systems[: args.smoke_systems]
    train_codes = extract_codes(
        model, train, train_systems, rollout_ids, anchors, device, args.batch_size
    )
    val_codes = extract_codes(
        model, val, val_systems, rollout_ids, anchors, device, args.batch_size
    )
    branches = {}
    for branch in ("z_p", "z_s"):
        train_window = train_codes[f"{branch}_window"]
        val_window = val_codes[f"{branch}_window"]
        train_rollout_raw = train_window.mean(2)
        val_rollout_raw = val_window.mean(2)
        val_rollout, normalization = standardize(train_rollout_raw, val_rollout_raw)
        train_flat = train_rollout_raw.reshape(-1, train_rollout_raw.shape[-1]).astype(np.float64)
        mean = train_flat.mean(0); std = train_flat.std(0).clip(1e-6)
        train_rollout = (train_rollout_raw - mean) / std
        train_centroid = train_rollout.mean(1)
        train_gamma = np.asarray(train.gamma)[train_systems]
        val_gamma = np.asarray(val.gamma)[val_systems]
        val_mass = np.asarray(val.mass)[val_systems]
        val_stiffness = np.asarray(val.stiffness)[val_systems]
        direction, alpha = fit_gamma_direction(train_centroid, train_gamma)
        point = branch_metrics(val_rollout, val_gamma, val_mass, val_stiffness, direction)
        ci = bootstrap_metrics(
            val_rollout, val_gamma, val_mass, val_stiffness, direction, args.bootstrap
        )
        val_centroid = val_rollout.mean(1)
        gamma_probe = Ridge(alpha).fit(train_centroid, train_gamma)
        raw = val_window.reshape(-1, val_window.shape[-1]).astype(np.float64)
        branches[branch] = {
            **point, "bootstrap_95_ci": ci, "gamma_probe_alpha": alpha,
            "gamma_probe_r2": r2(val_gamma, gamma_probe.predict(val_centroid)),
            "raw_window_norm_mean": float(np.linalg.norm(raw, axis=1).mean()),
            "raw_window_per_dim_std_mean": float(raw.std(0).mean()),
            "raw_window_effective_rank": effective_rank(raw),
            "normalization": normalization,
        }
    report = {
        "schema_version": "pokeworld-geometry-1.0", "condition": args.condition,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]), "training_config": config,
        "analysis_root": str(args.analysis_root), "analysis_manifest_sha256": manifest_hash,
        "bank_manifest": str(args.bank_manifest),
        "bank_manifest_sha256": sha256_file(args.bank_manifest),
        "analysis_split": "train normalization and val metrics", "anchors": anchors.tolist(),
        "rollout_aggregation": "mean fixed anchors then mean rollout codes for system centroid",
        "normalization": "per-checkpoint branch-wise train rollout mean/std",
        "bootstrap_unit": "system resample then reconstruct pairs", "bootstrap_draws": args.bootstrap,
        "rollout_ids": rollout_ids.tolist(),
        "train_system_count": int(train_systems.size),
        "val_system_count": int(val_systems.size),
        "smoke": smoke,
        "branches": branches, "test_read": False, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
