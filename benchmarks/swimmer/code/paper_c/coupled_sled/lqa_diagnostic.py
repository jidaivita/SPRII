"""Outcome-blind Learner--Query Alignment (LQA) score generation.

This module may read physical/reference inputs and frozen learner artifacts, but
its generator intentionally has no learner-outcome argument.  Outcome joining
belongs to a separate analysis stage after the score artifact is frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from .development import PARAMETER_NAMES, _features
from .formal_data import arrays_from_sample_manifest, load_system_pool
from .learner import PersistentJEPA
from .learner_data import _action_landmarks
from .manifests import load_spec
from .p0_coupled_adapter import _balanced_block_prefix_indices
from .posterior import lognormal_quadrature, posterior_from_observation
from .waveforms import history_probe_bank, query_bank


STATUS = "P3R_LQA_SCORE_GENERATION_COMPLETE_OUTCOME_BLIND"
LEVELS = (1024, 4096, 16384)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    if values.ndim != 1 or not np.all(np.isfinite(values)) or total <= 0:
        raise ValueError("invalid weights")
    return values / total


def weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = normalize_weights(weights)
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(w):
        raise ValueError("feature/weight axis mismatch")
    return x - np.sum(w[:, None] * x, axis=0, keepdims=True)


def weighted_linear_cka(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted linear CKA plus energy/effective-rank diagnostics."""

    w = normalize_weights(weights)
    x = weighted_center(left, w)
    y = weighted_center(right, w)
    cross = x.T @ (w[:, None] * y)
    xx = x.T @ (w[:, None] * x)
    yy = y.T @ (w[:, None] * y)
    cross_energy = float(np.sum(cross * cross))
    x_energy = float(np.sum(xx * xx))
    y_energy = float(np.sum(yy * yy))
    denominator = float(np.sqrt(x_energy * y_energy))
    if denominator <= 100.0 * np.finfo(np.float64).eps or not np.isfinite(denominator):
        score = float("nan")
    else:
        score = float(np.clip(cross_energy / denominator, 0.0, 1.0))

    def effective_rank(covariance: np.ndarray) -> float:
        trace = float(np.trace(covariance))
        squared = float(np.sum(covariance * covariance))
        return float(trace * trace / squared) if squared > 0 else 0.0

    return {
        "score": score,
        "left_energy": x_energy,
        "right_energy": y_energy,
        "left_effective_rank": effective_rank(xx),
        "right_effective_rank": effective_rank(yy),
    }


def permuted_weighted_linear_cka(left: np.ndarray, right: np.ndarray, weights: np.ndarray, permutations: np.ndarray, device: torch.device, batch_size: int = 8) -> np.ndarray:
    """Weighted CKA after fixed query-particle permutations."""

    w_np = normalize_weights(weights)
    x_np = weighted_center(left, w_np)
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = torch.from_numpy(np.asarray(right, dtype=np.float32)).to(device)
    w = torch.from_numpy(w_np.astype(np.float32)).to(device)
    xx = x.T @ (w[:, None] * x)
    x_energy = torch.sum(xx * xx)
    results = []
    with torch.no_grad():
        for start in range(0, len(permutations), batch_size):
            indices = torch.from_numpy(permutations[start : start + batch_size].astype(np.int64, copy=False)).to(device)
            yp = y[indices]
            yc = yp - torch.einsum("n,knd->kd", w, yp)[:, None, :]
            cross = torch.einsum("nd,n,kne->kde", x, w, yc)
            yy = torch.einsum("knd,n,kne->kde", yc, w, yc)
            numerator = torch.sum(cross * cross, dim=(1, 2))
            denominator = torch.sqrt(x_energy * torch.sum(yy * yy, dim=(1, 2)))
            results.append((numerator / denominator).cpu().numpy())
    return np.concatenate(results).astype(np.float64)


def selector_advantage(delta_alignment: float, high_loss: float, low_loss: float) -> float:
    """Positive iff the alignment-selected candidate has lower loss."""

    delta_gain = float(low_loss - high_loss)
    return float(np.sign(delta_alignment) * delta_gain)


def reconstruct_anchor_nuisance(group_seed: int, gain_mu: float, gain_sigma: float, sensor_std: float, feature_dim: int) -> tuple[float, np.ndarray]:
    """Replay the frozen sample-manifest RNG through the anchor draw."""

    rng = np.random.default_rng(int(group_seed))
    anchor_gain = float(rng.lognormal(gain_mu, gain_sigma))
    # arrays_from_sample_manifest draws both segment gains before either noise.
    _ = float(rng.lognormal(gain_mu, gain_sigma))
    anchor_noise = rng.normal(0.0, sensor_std, feature_dim)
    return anchor_gain, anchor_noise


def select_context_shard(contexts: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    positions = np.arange(len(contexts))
    return contexts.iloc[positions % shard_count == shard_index].copy()


def _encode_in_batches(module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            outputs.append(module(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _aggregate_encoded_in_batches(model, encoded: np.ndarray, mask: np.ndarray,
                                  device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            encoded_batch = torch.from_numpy(
                encoded[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            mask_batch = torch.from_numpy(
                mask[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            outputs.append(model.aggregate_encoded(encoded_batch, mask_batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _load_model(base_path: Path, formal_path: Path, checkpoint: Path, history_dim: int, query_dim: int, target_dim: int, device: torch.device):
    base = load_spec(base_path)
    formal = load_spec(formal_path)
    cfg = copy.deepcopy(base["learner_development"])
    cfg.update(formal["learner"])
    model = PersistentJEPA(history_dim, query_dim, target_dim, cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg


def _context_seed_table(evaluation_manifest: Path) -> pd.DataFrame:
    table = pd.read_csv(evaluation_manifest)
    keys = ["system_index", "realization", "anchor_index", "query_index"]
    grouped = table.groupby(keys, as_index=False).agg(
        group_seed=("group_seed", "first"),
        anchor_probe=("anchor_probe", "first"),
        query=("query", "first"),
        seeds=("group_seed", "nunique"),
        anchors=("anchor_probe", "nunique"),
        queries=("query", "nunique"),
    )
    if (grouped[["seeds", "anchors", "queries"]] != 1).any().any():
        raise RuntimeError("evaluation manifest context semantics are not unique")
    return grouped.drop(columns=["seeds", "anchors", "queries"])


def generate_scores(args: argparse.Namespace) -> dict:
    base = load_spec(args.base_spec)
    v3 = json.loads(args.v3_config.read_text())
    cfg = base["development_v0_1"]
    norms = {key: value for key, value in np.load(args.normalization, allow_pickle=False).items()}
    system_ids, systems = load_system_pool(args.system_pool)
    pairs = pd.read_csv(args.pair_manifest)
    required = {"system_index", "realization", "history_index", "query_index", "candidate_high_a", "candidate_low_a", "candidate_pair"}
    if missing := required.difference(pairs.columns):
        raise RuntimeError(f"pair manifest lacks {sorted(missing)}")
    contexts = _context_seed_table(args.evaluation_manifest)
    pairs = pairs.merge(
        contexts,
        left_on=["system_index", "realization", "history_index", "query_index"],
        right_on=["system_index", "realization", "anchor_index", "query_index"],
        how="left",
        validate="many_to_one",
    )
    if pairs.group_seed.isna().any():
        raise RuntimeError("pair/evaluation context join failed")
    context_keys = ["system_index", "realization", "history_index", "query_index"]
    unique_contexts = pairs[context_keys + ["group_seed", "anchor_probe", "query"]].drop_duplicates(context_keys)
    unique_contexts = unique_contexts.sort_values(context_keys).reset_index(drop=True)
    if args.max_contexts is not None:
        unique_contexts = unique_contexts.iloc[: args.max_contexts].copy()
    total_selected_contexts = len(unique_contexts)
    unique_contexts = select_context_shard(unique_contexts, args.shard_index, args.shard_count)
    pairs = pairs.merge(unique_contexts[context_keys], on=context_keys, how="inner", validate="many_to_one")

    particle_payload = json.loads(args.particle_manifest.read_text())
    particle_rows = particle_payload["particles"]
    full_indices = _balanced_block_prefix_indices(particle_rows, max(args.levels))
    particles_all = np.asarray([[row[name] for name in PARAMETER_NAMES] for row in particle_rows], dtype=np.float64)
    particles = particles_all[full_indices]
    original_to_position = {int(original): position for position, original in enumerate(full_indices)}
    level_positions = {
        level: np.asarray([original_to_position[int(index)] for index in _balanced_block_prefix_indices(particle_rows, level)], dtype=int)
        for level in args.levels
    }
    permutation_indices = None
    if args.permutations:
        seed = int.from_bytes(hashlib.sha256(b"p3r-lqa-particle-permutation-v1").digest()[:8], "little")
        permutation_rng = np.random.default_rng(seed)
        permutation_indices = np.stack([permutation_rng.permutation(len(particles)) for _ in range(args.permutations)])

    dt = base["dynamics"]["reference_dt_s"]
    histories = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    queries = query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"]))
    history_ids, query_ids = sorted(histories), sorted(queries)
    if len(history_ids) != 6 or len(query_ids) != 6:
        raise RuntimeError("frozen six-by-six bank changed")
    particle_history = _features(particles, histories, cfg["history_landmarks"], cfg["observation_semantics"])
    system_history = _features(systems, histories, cfg["history_landmarks"], cfg["observation_semantics"])
    particle_query = _features(particles, queries, cfg["query_landmarks"], "both_positions")
    history_actions = {name: _action_landmarks(histories[name], cfg["history_landmarks"]) for name in history_ids}
    query_actions = {name: _action_landmarks(queries[name], cfg["query_landmarks"]) for name in query_ids}
    gain_nodes, gain_weights = lognormal_quadrature(
        base["actuator_gain"]["mean"], base["actuator_gain"]["cv"], base["actuator_gain"]["quadrature_points"]
    )
    gain_log_variance = np.log1p(base["actuator_gain"]["cv"] ** 2)
    gain_sigma = float(np.sqrt(gain_log_variance))
    gain_mu = float(-0.5 * gain_log_variance)
    sensor_std = float(cfg["sensor_std_m"])

    # Independent parity path through the canonical evaluator implementation.
    # This detects seed-order, waveform-axis, and system-index drift before any
    # LQA score is emitted.
    parity_count = min(int(args.parity_contexts), len(unique_contexts))
    parity_keys = unique_contexts.iloc[:parity_count][context_keys]
    evaluation_table = pd.read_csv(args.evaluation_manifest)
    parity_rows = evaluation_table.merge(
        parity_keys,
        left_on=["system_index", "realization", "anchor_index", "query_index"],
        right_on=["system_index", "realization", "history_index", "query_index"],
        how="inner",
    )
    parity_rows = parity_rows[parity_rows.condition == "anchor"].drop_duplicates(context_keys)
    if len(parity_rows) != parity_count:
        raise RuntimeError("could not construct one canonical anchor row per parity context")
    with tempfile.TemporaryDirectory(prefix="p3r-lqa-parity-") as temporary:
        parity_manifest = Path(temporary) / "anchor_rows.csv"
        parity_rows.drop(columns=["history_index"]).to_csv(parity_manifest, index=False)
        parity_arrays = arrays_from_sample_manifest(args.base_spec, args.system_pool, parity_manifest)
    expected_anchor = parity_arrays.history[:, 0]
    reconstructed_anchor = []
    for row in parity_rows.itertuples(index=False):
        gain, noise = reconstruct_anchor_nuisance(
            int(row.group_seed), gain_mu, gain_sigma, sensor_std,
            cfg["history_landmarks"] * 2,
        )
        name = str(row.anchor_probe)
        response = gain * system_history[name][int(row.system_index)] + noise
        reconstructed_anchor.append(np.concatenate((response, history_actions[name])))
    reconstructed_anchor = np.asarray(reconstructed_anchor, dtype=np.float32)
    if not np.array_equal(reconstructed_anchor, expected_anchor):
        difference = float(np.max(np.abs(reconstructed_anchor - expected_anchor)))
        raise RuntimeError(f"canonical anchor replay parity failed: max_abs={difference}")

    device = torch.device(args.device)
    history_dim = cfg["history_landmarks"] * 4
    query_dim = cfg["query_landmarks"] * 2
    target_dim = query_dim
    model, learner_cfg = _load_model(args.base_spec, args.formal_spec, args.checkpoint, history_dim, query_dim, target_dim, device)
    batch_size = int(args.batch_size or learner_cfg["batch_size"])

    # Candidate segment embeddings are independent of H and Q.
    candidate_embeddings: dict[int, np.ndarray] = {}
    for candidate_index, candidate in enumerate(history_ids):
        by_gain = []
        for gain in gain_nodes:
            action = np.broadcast_to(history_actions[candidate], (len(particles), len(history_actions[candidate])))
            segment = np.concatenate((gain * particle_history[candidate], action), axis=1)
            segment = (segment - norms["history_mean"]) / norms["history_std"]
            by_gain.append(_encode_in_batches(model.segment_encoder, segment, device, batch_size))
        candidate_embeddings[candidate_index] = np.stack(by_gain, axis=0)

    query_embeddings: dict[int, np.ndarray] = {}
    target_latents: dict[int, np.ndarray] = {}
    for query_index, query in enumerate(query_ids):
        query_value = (query_actions[query] - norms["query_mean"]) / norms["query_std"]
        query_embeddings[query_index] = _encode_in_batches(model.query_encoder, query_value[None, :], device, 1)[0]
        average = None
        for gain, weight in zip(gain_nodes, gain_weights):
            target = (gain * particle_query[query] - norms["target_mean"]) / norms["target_std"]
            encoded = _encode_in_batches(model.target_encoder, target, device, batch_size)
            average = weight * encoded if average is None else average + weight * encoded
        target_latents[query_index] = average

    def predicted_query_latent(anchor_embedding: np.ndarray, candidate_index: int, query_index: int) -> np.ndarray:
        result = None
        query_embedding = query_embeddings[query_index]
        candidate_by_gain = candidate_embeddings[candidate_index]
        for gain_position, weight in enumerate(gain_weights):
            pieces = []
            for start in range(0, len(particles), batch_size):
                stop = min(start + batch_size, len(particles))
                count = stop - start
                anchor = np.broadcast_to(anchor_embedding, (count, len(anchor_embedding)))
                mask = np.ones((count, 2), dtype=np.float32)
                encoded = np.stack((anchor, candidate_by_gain[gain_position, start:stop]), axis=1)
                z = _aggregate_encoded_in_batches(model, encoded, mask, device, batch_size)
                query = np.broadcast_to(query_embedding, (count, len(query_embedding)))
                pieces.append(_encode_in_batches(model.latent_predictor, np.concatenate((z, query), axis=1), device, batch_size))
            predicted = np.concatenate(pieces, axis=0)
            result = weight * predicted if result is None else result + weight * predicted
        return result

    rows = []
    permutation_rows = []
    for context in unique_contexts.itertuples(index=False):
        system_index = int(context.system_index)
        history_index = int(context.history_index)
        query_index = int(context.query_index)
        anchor_name = history_ids[history_index]
        if context.anchor_probe != anchor_name or context.query != query_ids[query_index]:
            raise RuntimeError("integer/string waveform axes disagree")
        anchor_gain, anchor_noise = reconstruct_anchor_nuisance(
            int(context.group_seed), gain_mu, gain_sigma, sensor_std,
            cfg["history_landmarks"] * 2,
        )
        anchor_response = anchor_gain * system_history[anchor_name][system_index] + anchor_noise
        posterior = posterior_from_observation(
            anchor_response, particle_history[anchor_name], sensor_std, gain_nodes, gain_weights
        ).weights
        anchor_segment = np.concatenate((anchor_response, history_actions[anchor_name]))
        anchor_segment = (anchor_segment - norms["history_mean"]) / norms["history_std"]
        anchor_embedding = _encode_in_batches(model.segment_encoder, anchor_segment[None, :], device, 1)[0]
        context_pairs = pairs[
            (pairs.system_index == system_index)
            & (pairs.realization == int(context.realization))
            & (pairs.history_index == history_index)
            & (pairs.query_index == query_index)
        ]
        candidate_indices = sorted(set(context_pairs.candidate_high_a.astype(int)).union(context_pairs.candidate_low_a.astype(int)))
        for candidate_index in candidate_indices:
            predicted = predicted_query_latent(anchor_embedding, candidate_index, query_index)
            target = target_latents[query_index]
            raw_candidate = particle_history[history_ids[candidate_index]]
            raw_query = particle_query[query_ids[query_index]]
            for level in args.levels:
                positions = level_positions[level]
                weights = normalize_weights(posterior[positions])
                lqa = weighted_linear_cka(predicted[positions], target[positions], weights)
                raw = weighted_linear_cka(raw_candidate[positions], raw_query[positions], weights)
                rows.append({
                    "system_index": system_index,
                    "realization": int(context.realization),
                    "history_index": history_index,
                    "query_index": query_index,
                    "candidate_index": candidate_index,
                    "particle_level": level,
                    "lqa": lqa["score"],
                    "predicted_query_latent_energy": lqa["left_energy"],
                    "target_latent_energy": lqa["right_energy"],
                    "predicted_query_latent_effective_rank": lqa["left_effective_rank"],
                    "target_latent_effective_rank": lqa["right_effective_rank"],
                    "raw_response_cka": raw["score"],
                    "posterior_ess_full": float(1.0 / np.sum(posterior ** 2)),
                })
            if permutation_indices is not None:
                null_scores = permuted_weighted_linear_cka(
                    predicted, target, posterior, permutation_indices, device,
                    batch_size=args.permutation_batch_size,
                )
                for permutation_index, value in enumerate(null_scores):
                    permutation_rows.append({
                        "system_index": system_index,
                        "realization": int(context.realization),
                        "history_index": history_index,
                        "query_index": query_index,
                        "candidate_index": candidate_index,
                        "permutation_index": permutation_index,
                        "lqa_permuted": float(value),
                    })

    score = pd.DataFrame(rows).sort_values(context_keys + ["candidate_index", "particle_level"]).reset_index(drop=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    score_path = args.output_root / "lqa_scores_outcome_blind.csv"
    score.to_csv(score_path, index=False, float_format="%.17g")
    permutation_path = None
    if permutation_rows:
        permutation_score = pd.DataFrame(permutation_rows).sort_values(
            context_keys + ["candidate_index", "permutation_index"]
        ).reset_index(drop=True)
        permutation_path = args.output_root / "lqa_particle_permutation_scores_outcome_blind.csv"
        permutation_score.to_csv(permutation_path, index=False, float_format="%.17g")
    receipt = {
        "status": STATUS,
        "learner_outcomes_accessed": False,
        "canonical_anchor_parity": {"passed": True, "contexts": parity_count, "comparison": "array_exact"},
        "contexts": int(len(unique_contexts)),
        "total_contexts_before_sharding": int(total_selected_contexts),
        "shard": {"index": int(args.shard_index), "count": int(args.shard_count)},
        "score_rows": int(len(score)),
        "particle_levels": list(args.levels),
        "primary_particle_level": int(max(args.levels)),
        "device": str(device),
        "hashes": {
            name: sha256(getattr(args, name))
            for name in ("base_spec", "formal_spec", "v3_config", "particle_manifest", "normalization", "system_pool", "pair_manifest", "evaluation_manifest", "checkpoint")
        },
        "score_sha256": sha256(score_path),
    }
    if permutation_path is not None:
        receipt["particle_permutations"] = int(args.permutations)
        receipt["permutation_score_sha256"] = sha256(permutation_path)
    (args.output_root / "lqa_score_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("base_spec", type=Path)
    result.add_argument("formal_spec", type=Path)
    result.add_argument("v3_config", type=Path)
    result.add_argument("particle_manifest", type=Path)
    result.add_argument("normalization", type=Path)
    result.add_argument("system_pool", type=Path)
    result.add_argument("pair_manifest", type=Path)
    result.add_argument("evaluation_manifest", type=Path)
    result.add_argument("checkpoint", type=Path)
    result.add_argument("output_root", type=Path)
    result.add_argument("--device", default="cpu")
    result.add_argument("--batch-size", type=int)
    result.add_argument("--max-contexts", type=int)
    result.add_argument("--parity-contexts", type=int, default=8)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--shard-count", type=int, default=1)
    result.add_argument("--permutations", type=int, default=0)
    result.add_argument("--permutation-batch-size", type=int, default=8)
    result.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    return result


def main() -> None:
    args = parser().parse_args()
    if tuple(args.levels) != tuple(sorted(set(args.levels))) or max(args.levels) != 16384:
        raise ValueError("particle levels must be unique ascending and end at 16384")
    generate_scores(args)


if __name__ == "__main__":
    main()
