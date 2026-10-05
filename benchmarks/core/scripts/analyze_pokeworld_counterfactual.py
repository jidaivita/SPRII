#!/usr/bin/env python3
"""Gamma-only donor intervention for frozen PokeWorld checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from persistent_jepa.evaluation import RIDGE_GRID, Ridge
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.poke_torch import PokeBatch, PokeSplit
from persistent_jepa.runtime import atomic_json, set_deterministic, sha256_file


ANALYSIS_SEED = 20260831
GAMMA_GRID = np.arange(0.5, 4.0001, 0.25, dtype=np.float64)
HORIZONS = (4, 16)
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-systems", type=int)
    return parser.parse_args()


def _reflect(position: np.ndarray, velocity: np.ndarray, limit: float, restitution: float) -> None:
    for axis in (0, 1):
        high = position[:, axis] > limit
        low = position[:, axis] < -limit
        position[high, axis] = limit
        position[low, axis] = -limit
        velocity[high & (velocity[:, axis] > 0), axis] *= -restitution
        velocity[low & (velocity[:, axis] < 0), axis] *= -restitution


def replay(
    initial_state: np.ndarray,
    actions: np.ndarray,
    mass: np.ndarray,
    gamma: np.ndarray,
    stiffness: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Replay exact PokeWorld dynamics; only sensor noise is intentionally absent."""
    initial = np.asarray(initial_state, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    count, steps = action_array.shape[:2]
    if initial.shape != (count, 8):
        raise ValueError("initial_state must have shape [N,8]")
    m = np.broadcast_to(np.asarray(mass, dtype=np.float64), (count,))
    g = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (count,))
    k = np.broadcast_to(np.asarray(stiffness, dtype=np.float64), (count,))
    finger_position = initial[:, 0:2].copy()
    finger_velocity = initial[:, 2:4].copy()
    object_position = initial[:, 4:6].copy()
    object_velocity = initial[:, 6:8].copy()
    states = np.empty((count, steps + 1, 8), dtype=np.float64)
    states[:, 0] = initial
    substeps = int(cfg["substeps"])
    sub_dt = float(cfg["dt"]) / substeps
    for step in range(steps):
        action = action_array[:, step]
        for _ in range(substeps):
            separation = object_position - finger_position
            distance = np.linalg.norm(separation, axis=1).clip(1e-8)
            normal = separation / distance[:, None]
            overlap = float(cfg["finger_radius"]) + float(cfg["object_radius"]) - distance
            active = overlap > 0
            relative_normal_velocity = np.sum(
                (object_velocity - finger_velocity) * normal, axis=1
            )
            local_stiffness = 1.5 * k * np.sqrt(np.maximum(overlap, 0.0))
            effective_mass = m / (m + float(cfg["finger_mass"]))
            damping = 2.0 * float(cfg["damping_ratio"]) * np.sqrt(
                local_stiffness * effective_mass
            )
            magnitude = np.maximum(
                0.0,
                k * np.maximum(overlap, 0.0) ** 1.5 - damping * relative_normal_velocity,
            )
            magnitude *= active
            object_contact_force = magnitude[:, None] * normal
            finger_force = float(cfg["force_max"]) * action - object_contact_force
            object_force = object_contact_force - (g * m)[:, None] * object_velocity
            finger_velocity += finger_force / float(cfg["finger_mass"]) * sub_dt
            object_velocity += object_force / m[:, None] * sub_dt
            finger_position += finger_velocity * sub_dt
            object_position += object_velocity * sub_dt
            _reflect(
                finger_position,
                finger_velocity,
                float(cfg["arena_half_extent"]) - float(cfg["finger_radius"]),
                float(cfg["wall_restitution"]),
            )
            _reflect(
                object_position,
                object_velocity,
                float(cfg["arena_half_extent"]) - float(cfg["object_radius"]),
                float(cfg["wall_restitution"]),
            )
        states[:, step + 1] = np.concatenate(
            [finger_position, finger_velocity, object_position, object_velocity], axis=1
        )
    return states.astype(np.float32)


def rows_to_indices(rows: list[dict]) -> tuple[np.ndarray, ...]:
    keys = ("system_index", "query_rollout", "query_anchor", "donor_rollout", "donor_anchor")
    return tuple(np.asarray([row[key] for row in rows], dtype=np.int64) for key in keys)


def verify_replay(data: PokeSplit, cfg: dict, rows: list[dict]) -> dict:
    system, query_rollout, _, _, _ = rows_to_indices(rows[: min(16, len(rows))])
    predicted = replay(
        np.asarray(data.states[system, query_rollout, 0]),
        np.asarray(data.actions[system, query_rollout]),
        np.asarray(data.mass)[system],
        np.asarray(data.gamma)[system],
        np.asarray(data.stiffness)[system],
        cfg,
    )
    reference = np.asarray(data.states[system, query_rollout])
    absolute = np.abs(predicted - reference)
    report = {
        "rows": int(system.size),
        "max_absolute_state_error": float(absolute.max()),
        "mean_absolute_state_error": float(absolute.mean()),
    }
    if report["max_absolute_state_error"] > 2e-3:
        raise RuntimeError(f"counterfactual replay does not match stored dynamics: {report}")
    return report


def subset_rows(rows: list[dict], systems: int | None) -> list[dict]:
    if systems is None:
        return rows
    if systems < 8:
        raise ValueError("smoke-systems must be at least 8")
    return [row for row in rows if int(row["system_index"]) < systems]


def slice_batch(batch: PokeBatch, start: int, stop: int, device: torch.device) -> PokeBatch:
    return PokeBatch(**{key: value[start:stop].to(device) for key, value in batch.__dict__.items()})


@torch.inference_mode()
def target_embeddings(
    model: PokeJEPA, data: PokeSplit, rows: list[dict], device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    system, query_rollout, query_anchor, _, _ = rows_to_indices(rows)
    batch = data._from_indices(system, query_rollout, query_anchor)
    embeddings = {4: [], 16: []}
    targets = {4: [], 16: []}
    for start in range(0, system.size, batch_size):
        stop = min(start + batch_size, system.size)
        item = slice_batch(batch, start, stop, device)
        _, target_h = model.encode_batch(item)
        embeddings[4].append(target_h[:, 1].float().cpu().numpy())
        embeddings[16].append(target_h[:, 2].float().cpu().numpy())
        targets[4].append(item.target_current[:, 1, 4:8].float().cpu().numpy())
        targets[16].append(item.target_current[:, 2, 4:8].float().cpu().numpy())
    return (
        np.stack([np.concatenate(embeddings[h]) for h in HORIZONS], axis=1),
        np.stack([np.concatenate(targets[h]) for h in HORIZONS], axis=1),
        system,
    )


def fit_decoders(
    embedding: np.ndarray,
    targets: np.ndarray,
    system_index: np.ndarray,
) -> tuple[dict[int, Ridge], dict[int, float]]:
    systems = np.unique(system_index)
    rng = np.random.default_rng(ANALYSIS_SEED)
    shuffled = rng.permutation(systems)
    cut = max(1, int(0.8 * shuffled.size))
    fit_systems, hold_systems = shuffled[:cut], shuffled[cut:]
    fit_rows = np.isin(system_index, fit_systems)
    hold_rows = np.isin(system_index, hold_systems)
    output, alphas = {}, {}
    for hi, horizon in enumerate(HORIZONS):
        scored = []
        for alpha in RIDGE_GRID:
            candidate = Ridge(alpha).fit(embedding[fit_rows, hi], targets[fit_rows, hi])
            prediction = candidate.predict(embedding[hold_rows, hi])
            scored.append((float(np.mean(np.square(prediction - targets[hold_rows, hi]))), alpha))
        alpha = float(min(scored)[1])
        output[horizon] = Ridge(alpha).fit(embedding[:, hi], targets[:, hi])
        alphas[horizon] = alpha
    return output, alphas


def state_stats(train: PokeSplit) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(train.states)[..., 4:8].reshape(-1, 4).astype(np.float64)
    return state.mean(0), state.std(0).clip(1e-8)


def rms_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(a - b), axis=-1))


def system_equal_mean(values: np.ndarray, systems: np.ndarray) -> float:
    return float(np.mean([np.mean(values[systems == system]) for system in np.unique(systems)]))


def summarize_horizon(
    gamma_eff: np.ndarray,
    preference: np.ndarray,
    systems: np.ndarray,
    sensitivity: np.ndarray,
    threshold: float,
) -> dict:
    def summarize(mask: np.ndarray) -> dict:
        correlations, slopes, preference_per_query = [], [], []
        constant_effective_gamma = []
        for row in np.flatnonzero(mask):
            is_constant = bool(np.ptp(gamma_eff[row]) == 0.0)
            constant_effective_gamma.append(is_constant)
            # A constant response carries zero steering. scipy correctly calls
            # its rank correlation undefined; for an aggregate steering score
            # we register it as zero and report the fraction explicitly.
            correlations.append(
                0.0 if is_constant else float(spearmanr(GAMMA_GRID, gamma_eff[row]).statistic)
            )
            slopes.append(float(np.polyfit(GAMMA_GRID, gamma_eff[row], deg=1)[0]))
            preference_per_query.append(float(np.mean(preference[row])))
        selected_systems = systems[mask]
        return {
            "queries": int(mask.sum()),
            "systems": int(np.unique(selected_systems).size),
            "mean_per_query_spearman": system_equal_mean(np.asarray(correlations), selected_systems),
            "mean_per_query_slope": system_equal_mean(np.asarray(slopes), selected_systems),
            "mean_preference": system_equal_mean(np.asarray(preference_per_query), selected_systems),
            "constant_effective_gamma_fraction": float(np.mean(constant_effective_gamma)),
            "global_spearman": float(
                spearmanr(
                    np.tile(GAMMA_GRID, mask.sum()), gamma_eff[mask].reshape(-1)
                ).statistic
            ),
        }

    return {
        "all": summarize(np.ones(systems.size, dtype=bool)),
        "high_sensitivity": summarize(sensitivity >= threshold),
    }


@torch.inference_mode()
def evaluate_counterfactual(
    model: PokeJEPA,
    data: PokeSplit,
    rows: list[dict],
    cfg: dict,
    decoders: dict[int, Ridge],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    system, query_rollout, query_anchor, donor_rollout, donor_anchor = rows_to_indices(rows)
    query_batch = data._from_indices(system, query_rollout, query_anchor)
    all_eff = {h: [] for h in HORIZONS}
    all_pref = {h: [] for h in HORIZONS}
    all_sensitivity = {h: [] for h in HORIZONS}
    prediction_gamma_spread = {h: [] for h in HORIZONS}
    decoded_gamma_spread = {h: [] for h in HORIZONS}
    donor_code_spread = []
    for start in range(0, system.size, batch_size):
        stop = min(start + batch_size, system.size)
        n = stop - start
        sys = system[start:stop]
        q_roll = query_rollout[start:stop]
        q_anchor = query_anchor[start:stop]
        d_roll = donor_rollout[start:stop]
        d_anchor = donor_anchor[start:stop]
        item = slice_batch(query_batch, start, stop, device)
        history_h, _ = model.encode_batch(item)
        z_s, _, _ = model.codes(history_h, item.history_actions)
        if z_s is None or model.persistent is None:
            raise ValueError("counterfactual steering requires split z_s/z_p model")

        repeat = GAMMA_GRID.size
        donor_initial = np.repeat(np.asarray(data.states[sys, d_roll, 0]), repeat, axis=0)
        donor_actions = np.repeat(np.asarray(data.actions[sys, d_roll]), repeat, axis=0)
        donor_mass = np.repeat(np.asarray(data.mass)[sys], repeat)
        donor_stiffness = np.repeat(np.asarray(data.stiffness)[sys], repeat)
        donor_gamma = np.tile(GAMMA_GRID, n)
        donor_states = replay(
            donor_initial, donor_actions[:, :48], donor_mass, donor_gamma, donor_stiffness, cfg
        )
        expanded_anchor = np.repeat(d_anchor, repeat)
        current_index = expanded_anchor[:, None] + np.arange(-23, 1)[None, :]
        previous_index = current_index - 1
        rows_index = np.arange(n * repeat)[:, None]
        current = torch.from_numpy(donor_states[rows_index, current_index]).to(device)
        previous = torch.from_numpy(donor_states[rows_index, previous_index]).to(device)
        images = model.renderer(current, previous)
        donor_h = model.observation(images)
        history_action_index = expanded_anchor[:, None] + np.arange(-23, 0)[None, :]
        donor_history_actions = torch.from_numpy(
            donor_actions[rows_index, history_action_index]
        ).to(device)
        donor_z = model.persistent(donor_h, donor_history_actions)
        donor_code_spread.append(
            donor_z.reshape(n, repeat, -1).std(dim=1).norm(dim=1).float().cpu().numpy()
        )

        q_initial = np.asarray(data.states[sys, q_roll, q_anchor])
        q_actions = np.stack(
            [np.asarray(data.actions[s, r, a : a + 16]) for s, r, a in zip(sys, q_roll, q_anchor, strict=True)]
        )
        q_mass = np.asarray(data.mass)[sys]
        q_stiffness = np.asarray(data.stiffness)[sys]
        q_gamma = np.asarray(data.gamma)[sys]
        grid_future = replay(
            np.repeat(q_initial, repeat, axis=0),
            np.repeat(q_actions, repeat, axis=0),
            np.repeat(q_mass, repeat),
            np.tile(GAMMA_GRID, n),
            np.repeat(q_stiffness, repeat),
            cfg,
        ).reshape(n, repeat, 17, 8)
        true_future = replay(q_initial, q_actions, q_mass, q_gamma, q_stiffness, cfg)

        for horizon, hi in ((4, 1), (16, 2)):
            context = torch.cat([z_s.repeat_interleave(repeat, dim=0), donor_z], dim=-1)
            future_actions = item.future_actions[:, hi].repeat_interleave(repeat, dim=0)
            action_masks = item.action_masks[:, hi].repeat_interleave(repeat, dim=0)
            horizon_index = torch.full((n * repeat,), hi, dtype=torch.long, device=device)
            prediction_h = model.predictor(context, future_actions, action_masks, horizon_index)
            decoded = decoders[horizon].predict(prediction_h.float().cpu().numpy())
            decoded = decoded.reshape(n, repeat, 4)
            prediction_gamma_spread[horizon].append(
                prediction_h.reshape(n, repeat, -1).std(dim=1).norm(dim=1).float().cpu().numpy()
            )
            decoded_gamma_spread[horizon].append(
                np.linalg.norm(decoded.std(axis=1), axis=1)
            )
            simulated = (grid_future[:, :, horizon, 4:8] - mean) / std
            query_truth = (true_future[:, horizon, 4:8] - mean) / std
            distances = rms_distance(decoded[:, :, None, :], simulated[:, None, :, :])
            effective = GAMMA_GRID[np.argmin(distances, axis=2)]
            d_query = rms_distance(decoded, query_truth[:, None, :])
            d_donor = rms_distance(decoded, simulated)
            preference = (d_query - d_donor) / (d_query + d_donor + EPS)
            pairwise = rms_distance(simulated[:, :, None, :], simulated[:, None, :, :])
            sensitivity = pairwise.max(axis=(1, 2))
            all_eff[horizon].append(effective)
            all_pref[horizon].append(preference)
            all_sensitivity[horizon].append(sensitivity)
    effective = {h: np.concatenate(all_eff[h]) for h in HORIZONS}
    preference = {h: np.concatenate(all_pref[h]) for h in HORIZONS}
    sensitivity = {h: np.concatenate(all_sensitivity[h]) for h in HORIZONS}
    threshold = float(np.median(sensitivity[16]))
    return {
        "gamma_grid": GAMMA_GRID.tolist(),
        "h16_sensitivity_threshold": threshold,
        "donor_code_gamma_spread_mean": float(np.concatenate(donor_code_spread).mean()),
        "horizons": {
            f"h{h}": summarize_horizon(effective[h], preference[h], system, sensitivity[16], threshold)
            for h in HORIZONS
        },
        "diagnostics": {
            **{f"h{h}_sensitivity_mean": float(sensitivity[h].mean()) for h in HORIZONS},
            **{
                f"h{h}_prediction_embedding_gamma_spread_mean": float(
                    np.concatenate(prediction_gamma_spread[h]).mean()
                )
                for h in HORIZONS
            },
            **{
                f"h{h}_decoded_state_gamma_spread_mean": float(
                    np.concatenate(decoded_gamma_spread[h]).mean()
                )
                for h in HORIZONS
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    set_deterministic(ANALYSIS_SEED)
    bank = json.loads(args.bank_manifest.read_text())
    manifest_hash = sha256_file(args.data_root / "manifest.json")
    if bank["schema_version"] != "pokeworld-analysis-bank-1.0":
        raise ValueError("unsupported bank schema")
    if bank["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("dataset differs from frozen analysis bank")
    dataset_manifest = json.loads((args.data_root / "manifest.json").read_text())
    cfg = dataset_manifest["config"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    variant = config["variant"]
    if variant not in {"B2", "B3"}:
        raise ValueError("counterfactual batch is frozen to B2/B3 conditions")
    device = torch.device(args.device)
    model = PokeJEPA(variant, history_length=int(config.get("history_length", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    train = PokeSplit(args.data_root, "train", history_length=24)
    val = PokeSplit(args.data_root, "val", history_length=24)
    train_rows = subset_rows(bank["counterfactual"]["train_rows"], args.smoke_systems)
    val_rows = subset_rows(bank["counterfactual"]["val_rows"], args.smoke_systems)
    replay_check = verify_replay(val, cfg, val_rows)
    train_h, train_target, train_system = target_embeddings(
        model, train, train_rows, device, args.batch_size
    )
    mean, std = state_stats(train)
    normalized_target = (train_target - mean) / std
    decoders, alphas = fit_decoders(train_h, normalized_target, train_system)
    result = evaluate_counterfactual(
        model, val, val_rows, cfg, decoders, mean, std, device, args.batch_size
    )
    report = {
        "schema_version": "pokeworld-counterfactual-1.0",
        "condition": args.condition,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "training_config": config,
        "data_root": str(args.data_root),
        "dataset_manifest_sha256": manifest_hash,
        "bank_manifest": str(args.bank_manifest),
        "bank_manifest_sha256": sha256_file(args.bank_manifest),
        "decoder_alpha": {f"h{h}": alpha for h, alpha in alphas.items()},
        "decoder_selection": "deterministic 80/20 train-system split; refit all train rows",
        "distance": "RMS in train-standardized object [x,y,vx,vy] state",
        "high_sensitivity": "median h16 simulator spread; model outputs not used",
        "replay_check": replay_check,
        "smoke": args.smoke_systems is not None,
        "train_system_count": len({row["system_index"] for row in train_rows}),
        "val_system_count": len({row["system_index"] for row in val_rows}),
        "test_read": False,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
