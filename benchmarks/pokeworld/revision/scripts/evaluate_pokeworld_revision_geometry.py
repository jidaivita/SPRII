#!/usr/bin/env python3
"""Common-system geometry and branch diagnostics for all revision conditions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
import torch

from persistent_jepa.evaluation import RIDGE_GRID, Ridge, r2
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file


EVAL_SEED = 20260903
ANCHORS = np.asarray([24, 32, 40, 47], dtype=np.int64)
ROLLOUTS = np.asarray([0, 1, 2, 3], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--train-systems", type=int, default=200)
    p.add_argument("--validation-systems", type=int, default=200)
    p.add_argument("--smoke-systems", type=int)
    return p.parse_args()


@torch.inference_mode()
def extract(model, data, systems_index, device, batch_size):
    systems = systems_index.size
    s = np.repeat(systems_index, ROLLOUTS.size * ANCHORS.size)
    r = np.tile(np.repeat(ROLLOUTS, ANCHORS.size), systems)
    a = np.tile(ANCHORS, systems * ROLLOUTS.size)
    zp, zs = [], []
    model.eval()
    for start in range(0, s.size, batch_size):
        stop = min(start + batch_size, s.size)
        batch = data._from_indices(s[start:stop], r[start:stop], a[start:stop]).to(device)
        history_h, _ = model.encode_batch(batch)
        transient, persistent, _ = model.codes(history_h, batch.history_actions)
        if persistent is None or transient is None:
            raise ValueError("revision evaluator requires split branches")
        zp.append(persistent.float().cpu().numpy())
        zs.append(transient.float().cpu().numpy())
    shape = (systems, ROLLOUTS.size, ANCHORS.size, -1)
    target_state = np.asarray(data.states)[s, r, a]
    target = {
        "position": target_state[:, 4:6].reshape(systems, ROLLOUTS.size, ANCHORS.size, 2),
        "velocity": target_state[:, 6:8].reshape(systems, ROLLOUTS.size, ANCHORS.size, 2),
    }
    if data.contact is not None:
        # Final observed transition ending at history endpoint t.
        contact = np.asarray(data.contact)[s, r, a - 1]
        target["contact"] = contact.reshape(systems, ROLLOUTS.size, ANCHORS.size)
    return {
        "z_p": np.concatenate(zp).reshape(shape),
        "z_s": np.concatenate(zs).reshape(shape),
        "target": target,
    }


def residual_rank(x: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    ranked = rankdata(x)
    design = np.column_stack([np.ones(x.size)] + [rankdata(value) for value in controls])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def partial_spearman(distance, factor, controls):
    yr = residual_rank(distance, controls)
    xr = residual_rank(factor, controls)
    denom = np.linalg.norm(yr) * np.linalg.norm(xr)
    return float(np.dot(yr, xr) / max(float(denom), 1e-12))


def geometry(rollout, mass, gamma, stiffness):
    centroid = rollout.mean(1)
    within = np.concatenate([np.square(pdist(value)) for value in rollout])
    between = pdist(centroid)
    factors = {
        "mass": pdist(np.log(mass)[:, None], metric="cityblock"),
        "drag": pdist(gamma[:, None], metric="cityblock"),
        "stiffness": pdist(np.log(stiffness)[:, None], metric="cityblock"),
    }
    partial = {}
    for name, value in factors.items():
        controls = [other for key, other in factors.items() if key != name]
        partial[name] = partial_spearman(between, value, controls)
    between_sq = float(np.mean(np.square(between)))
    within_sq = float(np.mean(within))
    return {
        "within_system_dispersion": within_sq,
        "between_system_centroid_separation": between_sq,
        "between_within_ratio": between_sq / max(within_sq, 1e-12),
        "partial_geometry": partial,
    }


def choose_alpha(x, y):
    rng = np.random.default_rng(EVAL_SEED)
    order = rng.permutation(x.shape[0])
    cut = max(1, int(0.8 * order.size))
    fit, hold = order[:cut], order[cut:]
    scored = []
    for alpha in RIDGE_GRID:
        model = Ridge(alpha).fit(x[fit], y[fit])
        pred = model.predict(x[hold])
        scored.append((float(np.mean(np.square(pred - y[hold]))), float(alpha)))
    return min(scored)[1]


def probe(train_x, train_y, val_x, val_y):
    alpha = choose_alpha(train_x, train_y)
    model = Ridge(alpha).fit(train_x, train_y)
    return {"alpha": alpha, "r2": r2(val_y, model.predict(val_x))}


def auroc(y, score):
    y = np.asarray(y, dtype=bool)
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(np.asarray(score, dtype=np.float64))
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))


def contact_probe(train_x, train_y, val_x, val_y):
    alpha = choose_alpha(train_x, train_y.astype(np.float64))
    model = Ridge(alpha).fit(train_x, train_y.astype(np.float64))
    return {"alpha": alpha, "auroc": auroc(val_y, model.predict(val_x))}


def standardize(train_rollout, val_rollout):
    flat = train_rollout.reshape(-1, train_rollout.shape[-1]).astype(np.float64)
    mean, std = flat.mean(0), flat.std(0).clip(1e-6)
    return (train_rollout - mean) / std, (val_rollout - mean) / std


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    set_deterministic(EVAL_SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    variant = config.get("model_variant", config.get("variant"))
    model = PokeJEPA(variant, history_length=int(config.get("history_length", 24))).to(args.device)
    model.load_state_dict(checkpoint["model"])
    train, val = PokeSplit(args.data_root, "train"), PokeSplit(args.data_root, "val")
    system_rng = np.random.default_rng(EVAL_SEED)
    train_index = np.sort(
        system_rng.permutation(train.states.shape[0])[: args.train_systems]
    )
    val_index = np.sort(
        system_rng.permutation(val.states.shape[0])[: args.validation_systems]
    )
    if args.smoke_systems is not None:
        if args.smoke_systems < 8:
            raise ValueError("smoke-systems must be at least 8")
        train_index = train_index[: args.smoke_systems]
        val_index = val_index[: args.smoke_systems]
    train_data = extract(model, train, train_index, torch.device(args.device), args.batch_size)
    val_data = extract(model, val, val_index, torch.device(args.device), args.batch_size)
    factors_train = {
        "mass": np.asarray(train.mass)[train_index],
        "drag": np.asarray(train.gamma)[train_index],
        "stiffness": np.asarray(train.stiffness)[train_index],
    }
    factors_val = {
        "mass": np.asarray(val.mass)[val_index],
        "drag": np.asarray(val.gamma)[val_index],
        "stiffness": np.asarray(val.stiffness)[val_index],
    }
    branches = {}
    bootstrap_rng = np.random.default_rng(EVAL_SEED + 17)
    bootstrap_indices = bootstrap_rng.integers(
        0, val_index.size, size=(args.bootstrap, val_index.size)
    )
    for branch in ("z_p", "z_s"):
        train_rollout_raw = train_data[branch].mean(2)
        val_rollout_raw = val_data[branch].mean(2)
        train_rollout, val_rollout = standardize(train_rollout_raw, val_rollout_raw)
        point = geometry(
            val_rollout, factors_val["mass"], factors_val["drag"], factors_val["stiffness"]
        )
        bootstrap = {"ratio": [], "mass": [], "drag": [], "stiffness": []}
        for index in bootstrap_indices:
            result = geometry(
                val_rollout[index], factors_val["mass"][index],
                factors_val["drag"][index], factors_val["stiffness"][index]
            )
            bootstrap["ratio"].append(result["between_within_ratio"])
            for factor in ("mass", "drag", "stiffness"):
                bootstrap[factor].append(result["partial_geometry"][factor])
        train_centroid, val_centroid = train_rollout.mean(1), val_rollout.mean(1)
        factor_probes = {
            name: probe(train_centroid, factors_train[name], val_centroid, factors_val[name])
            for name in factors_train
        }
        train_window = train_data[branch].reshape(-1, train_data[branch].shape[-1])
        val_window = val_data[branch].reshape(-1, val_data[branch].shape[-1])
        transient = {
            "position": probe(
                train_window, train_data["target"]["position"].reshape(-1, 2),
                val_window, val_data["target"]["position"].reshape(-1, 2)
            ),
            "velocity": probe(
                train_window, train_data["target"]["velocity"].reshape(-1, 2),
                val_window, val_data["target"]["velocity"].reshape(-1, 2)
            ),
        }
        if "contact" in train_data["target"]:
            transient["contact"] = contact_probe(
                train_window, train_data["target"]["contact"].reshape(-1),
                val_window, val_data["target"]["contact"].reshape(-1)
            )
        branches[branch] = {
            **point,
            "factor_accessibility": factor_probes,
            "transient_accessibility": transient,
            "bootstrap_values": bootstrap,
        }
    report = {
        "schema_version": "paper-a-revision-geometry-1.0",
        "condition": args.condition,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "training_config": config,
        "data_root": str(args.data_root),
        "dataset_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "analysis_split": "train normalization/probe fit; validation metrics",
        "evaluation_population": "common held-out systems, not training donor pairs",
        "training_relation_pairs_used_for_geometry": False,
        "aggregation": "window -> rollout mean -> system centroid",
        "anchors": ANCHORS.tolist(),
        "rollouts": ROLLOUTS.tolist(),
        "bootstrap_unit": "paired system resample then reconstruct all pairs",
        "bootstrap_seed": EVAL_SEED + 17,
        "bootstrap_draws": args.bootstrap,
        "bootstrap_indices_sha256": sha256(bootstrap_indices.tobytes()).hexdigest(),
        "transient_target_time": "final observed frame of 24-step history",
        "contact_target": "final observed transition ending at the history endpoint",
        "train_system_count": int(train_index.size),
        "validation_system_count": int(val_index.size),
        "branches": branches,
        "test_read": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
