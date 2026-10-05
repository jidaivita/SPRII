"""Outcome-blind launch checks for Articulated LQA confirmation.

This module deliberately stops before learner-loss evaluation.  It constructs
fresh physical contexts, two independent posterior-predictive Bayes reference
streams, the frozen local accessibility score, and the frozen LQA/Raw-CKA
scores.  A later module may join a hash-frozen pair manifest to learner losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from paper_c.coupled_sled.learner import CANONICAL_ARCHITECTURE_TAG, build_persistent_jepa
from paper_c.coupled_sled.lqa_diagnostic import weighted_linear_cka
from paper_c.coupled_sled.p0_coupled_adapter import _reference_value_paths_all_queries_moment_nested
from paper_c.coupled_sled.p0_reference import (
    modal_geometry,
    posterior_averaged_gaussian_information,
    query_utility_curvature,
    weighted_covariance,
    whitened_operators,
)
from paper_c.coupled_sled.posterior import posterior_from_observation

from .model import SwimmerModel, prior_log_scale
from .s0_screen import _landmarks, _sample_log_scales
from .waveforms import banks


AXES = ("system", "realization", "history", "candidate", "query")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def system_pool(count: int, seed: int, prior: dict) -> np.ndarray:
    if count <= 0:
        raise ValueError("system count must be positive")
    sampler = qmc.Sobol(5, scramble=True, seed=int(seed))
    exponent = int(math.ceil(math.log2(count)))
    unit = sampler.random_base2(exponent)[:count]
    lower = np.asarray([prior["mass_scale"][0]] * 3 + [prior["damping_scale"][0]] * 2)
    upper = np.asarray([prior["mass_scale"][1]] * 3 + [prior["damping_scale"][1]] * 2)
    return np.log(lower + unit * (upper - lower))


def particle_pool(count: int, seed: int, prior: dict) -> np.ndarray:
    if count <= 0 or count & (count - 1):
        raise ValueError("particle count must be a positive power of two")
    return system_pool(count, seed, prior)


def fixed_contexts(system_count: int, realizations: int, histories: int, count: int, salt: str) -> pd.DataFrame:
    """Choose at most one deterministic nuisance/history context per system.

    The physical system is the independent statistical unit.  Selecting a
    global hash prefix over all context rows can repeat a system and silently
    make a nominal ``count`` look larger than its independent-system count.
    We therefore choose the lowest-hash context within each system first, then
    hash-order the resulting systems.
    """
    rows = []
    for system in range(system_count):
        choices = []
        for realization in range(realizations):
            for history in range(histories):
                token = f"{salt}|{system}|{realization}|{history}".encode()
                choices.append((hashlib.sha256(token).hexdigest(), realization, history))
        _, realization, history = min(choices)
        system_token = hashlib.sha256(f"{salt}|system|{system}".encode()).hexdigest()
        rows.append((system_token, system, realization, history))
    rows.sort()
    selected = rows[: min(count, len(rows))]
    return pd.DataFrame([(s, r, h) for _, s, r, h in selected], columns=["system_index", "realization", "history_index"])


def _response_bank(model: SwimmerModel, particles: np.ndarray, initial, bank: dict, landmarks: np.ndarray) -> np.ndarray:
    return np.asarray([
        [model.rollout(theta, initial, actions, landmarks) for actions in bank.values()]
        for theta in particles
    ], dtype=np.float64)


def _landmark_actions(actions: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    indices = np.clip(np.asarray(landmarks, dtype=int) - 1, 0, len(actions) - 1)
    return np.asarray(actions)[indices].reshape(-1)


def _response_jacobian_bank(
    model: SwimmerModel,
    particles: np.ndarray,
    initial,
    bank: dict,
    landmarks: np.ndarray,
    parameter_scale: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((len(particles), len(bank), len(landmarks) * 8), dtype=np.float64)
    jacobians = np.empty(means.shape + (5,), dtype=np.float64)
    for p, theta in enumerate(particles):
        for b, actions in enumerate(bank.values()):
            means[p, b] = model.rollout(theta, initial, actions, landmarks)
            for parameter in range(5):
                offset = np.zeros(5, dtype=float)
                offset[parameter] = step * parameter_scale[parameter]
                plus = model.rollout(theta + offset, initial, actions, landmarks)
                minus = model.rollout(theta - offset, initial, actions, landmarks)
                jacobians[p, b, :, parameter] = (plus - minus) / (2.0 * step)
    return means, jacobians


def _context_nuisance(base: dict, model: SwimmerModel, theta: np.ndarray, history: dict, query: dict, landmarks: np.ndarray, seed: int, system: int, realization: int):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(system), int(realization)]))
    initial_h = model.sample_initial_state(rng, base["transient_initial_state"])
    initial_e = model.sample_initial_state(rng, base["transient_initial_state"])
    initial_q = model.sample_initial_state(rng, base["transient_initial_state"])
    true_h = np.asarray([model.rollout(theta, initial_h, actions, landmarks) for actions in history.values()])
    true_e = np.asarray([model.rollout(theta, initial_e, actions, landmarks) for actions in history.values()])
    true_q = np.asarray([model.rollout(theta, initial_q, actions, landmarks) for actions in query.values()])
    sigma = float(base["observation"]["sensor_std"])
    noise_h = rng.normal(0.0, sigma, true_h.shape)
    noise_e = rng.normal(0.0, sigma, true_e.shape)
    noise_q = rng.normal(0.0, sigma, true_q.shape)
    return initial_h, initial_e, initial_q, true_h + noise_h, true_e + noise_e, true_q + noise_q


def _load_jepa(root: Path, cfg: dict, device: torch.device):
    s2 = json.loads((root / cfg["s2_config"]).read_text())
    model_root = root / cfg["s2_models"]
    receipt = json.loads((model_root / "s2_training_receipt.json").read_text())
    if receipt["status"] != "S2_LEARNER_SUFFICIENCY_GO":
        raise RuntimeError("frozen Articulated S2-R learner is not verified GO")
    model = build_persistent_jepa(
        48, 16, 32, s2, expected_architecture_tag=CANONICAL_ARCHITECTURE_TAG,
    ).to(device)
    model.load_state_dict(torch.load(model_root / "persistent_jepa_frozen.pt", map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    norms = {key: value for key, value in np.load(model_root / "train_only_normalization.npz", allow_pickle=False).items()}
    return model, norms, receipt


def _torch_batches(module, values: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            result.append(module(batch).cpu().numpy())
    return np.concatenate(result)


def _aggregate_encoded_batches(model, encoded: np.ndarray, mask: np.ndarray,
                               device: torch.device, batch_size: int = 1024) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            segment_batch = torch.from_numpy(
                encoded[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            mask_batch = torch.from_numpy(
                mask[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            result.append(model.aggregate_encoded(segment_batch, mask_batch).cpu().numpy())
    return np.concatenate(result)


def lqa_bank(model, norms, posterior, anchor_observation, initial_h, initial_e, initial_q, candidate_means, query_means, history, query, landmarks, device):
    history_actions = np.asarray([_landmark_actions(actions, landmarks) for actions in history.values()])
    query_actions = np.asarray([_landmark_actions(actions, landmarks) for actions in query.values()])
    init_h = np.concatenate([initial_h.qpos[2:], initial_h.qvel])
    init_e = np.concatenate([initial_e.qpos[2:], initial_e.qvel])
    init_q = np.concatenate([initial_q.qpos[2:], initial_q.qvel])
    anchor_index = int(anchor_observation[0])
    anchor_value = np.concatenate([init_h, anchor_observation[1], history_actions[anchor_index]])
    anchor_value = (anchor_value - norms["history_mean"]) / norms["history_std"]
    anchor_embedding = _torch_batches(model.segment_encoder, anchor_value[None], device)[0]
    particles = candidate_means.shape[0]
    candidate_segments = np.concatenate([
        np.broadcast_to(init_e, (particles, 6, 8)),
        candidate_means,
        np.broadcast_to(history_actions[None], (particles, 6, 8)),
    ], axis=2)
    candidate_segments = (candidate_segments - norms["history_mean"]) / norms["history_std"]
    candidate_embedding = _torch_batches(model.segment_encoder, candidate_segments.reshape(-1, 48), device).reshape(particles, 6, -1)
    query_input = np.concatenate([np.broadcast_to(init_q, (6, 8)), query_actions], axis=1)
    query_input = (query_input - norms["query_mean"]) / norms["query_std"]
    query_embedding = _torch_batches(model.query_encoder, query_input, device)
    target = (query_means - norms["target_mean"]) / norms["target_std"]
    target_latent = _torch_batches(model.target_encoder, target.reshape(-1, 32), device).reshape(particles, 6, -1)
    scores = np.empty((6, 6), dtype=float)
    raw = np.empty_like(scores)
    for candidate in range(6):
        anchor = np.broadcast_to(anchor_embedding, (particles, len(anchor_embedding)))
        encoded = np.stack((anchor, candidate_embedding[:, candidate]), axis=1)
        mask = np.ones((particles, 2), dtype=np.float32)
        z = _aggregate_encoded_batches(model, encoded, mask, device)
        for q in range(6):
            qembed = np.broadcast_to(query_embedding[q], (particles, query_embedding.shape[1]))
            predicted = _torch_batches(model.latent_predictor, np.concatenate((z, qembed), axis=1), device)
            scores[candidate, q] = weighted_linear_cka(predicted, target_latent[:, q], posterior)["score"]
            raw[candidate, q] = weighted_linear_cka(candidate_means[:, candidate], query_means[:, q], posterior)["score"]
    return scores, raw


def accessibility_bank(particles, posterior, candidate_means, candidate_jacobians, query_jacobians, query_scale, sensor_std, parameter_scale):
    standardized_parameters = particles / parameter_scale[None]
    covariance = weighted_covariance(standardized_parameters, posterior)
    candidate_information = [
        posterior_averaged_gaussian_information(candidate_means[:, e], candidate_jacobians[:, e], posterior, sensor_std, 0.0)
        for e in range(6)
    ]
    scores = np.empty((6, 6), dtype=float)
    local = np.empty_like(scores)
    for q in range(6):
        curvature = query_utility_curvature(query_jacobians[:, q], posterior, query_scale)
        for e in range(6):
            j, c = whitened_operators(covariance, candidate_information[e], curvature)
            geometry = modal_geometry(j, c)
            scores[e, q] = geometry.accessibility
            local[e, q] = geometry.local_value
    return scores, local


def reference_stream_from_response_banks(
    base,
    true_anchor,
    hmeans,
    emeans,
    qmeans,
    outcome_levels,
    outcome_seed,
    query_scale,
):
    """Evaluate one reference stream from already materialized response banks.

    ``qmeans`` intentionally retains the historical query-first layout
    ``[query, particle, feature]``.  Keeping the numerical kernel separate from
    response-bank generation lets callers reuse the exact same banks for
    posterior-predictive means without changing any floating-point operation
    in the reference-value calculation.
    """

    hmeans = np.asarray(hmeans, dtype=np.float64)
    emeans = np.asarray(emeans, dtype=np.float64)
    qmeans = np.asarray(qmeans, dtype=np.float64)
    if hmeans.ndim != 2 or emeans.ndim != 3 or qmeans.ndim != 3:
        raise ValueError("reference response banks have invalid ranks")
    if emeans.shape[0] != hmeans.shape[0] or qmeans.shape[1] != hmeans.shape[0]:
        raise ValueError("reference response banks disagree on particle population")
    if emeans.shape[1] != 6 or qmeans.shape[0] != 6:
        raise ValueError("reference response banks must contain six candidates and queries")
    query_scale = np.broadcast_to(np.asarray(query_scale, dtype=float), (qmeans.shape[0], qmeans.shape[2]))
    posterior = posterior_from_observation(true_anchor, hmeans, base["observation"]["sensor_std"], np.array([1.0]), np.array([1.0])).weights
    values = np.empty((len(outcome_levels), 6, 6), dtype=float)
    parity = 0.0
    u_q = None
    for candidate in range(6):
        first, second, risk = _reference_value_paths_all_queries_moment_nested(
            posterior, emeans[:, candidate], qmeans, query_scale,
            base["observation"]["sensor_std"], np.array([1.0]), np.array([1.0]),
            # Candidate-common randomized-QMC outcomes are required so a
            # pair-difference uncertainty estimate retains the covariance.
            0.0, 0.0, tuple(outcome_levels), outcome_seed,
        )
        values[:, candidate] = first
        parity = max(parity, float(np.max(np.abs(first - second))))
        u_q = risk
    return values, posterior, float(1.0 / np.sum(posterior * posterior)), parity, u_q


def reference_stream(base, model, history, query, landmarks, true_anchor, initial_h, initial_e, initial_q, history_index, particle_count, particle_seed, outcome_levels, outcome_seed, query_scale):
    particles = particle_pool(particle_count, particle_seed, base["persistent_prior"])
    hmeans = _response_bank(model, particles, initial_h, history, landmarks)[:, history_index]
    emeans = _response_bank(model, particles, initial_e, history, landmarks)
    qmeans = _response_bank(model, particles, initial_q, query, landmarks).transpose(1, 0, 2)
    return reference_stream_from_response_banks(
        base, true_anchor, hmeans, emeans, qmeans, outcome_levels, outcome_seed, query_scale,
    )


def interface_parity(root: Path, config_path: Path, output_path: Path) -> dict:
    """Array-exact comparison against the verified Articulated evaluator."""

    from . import s3_evaluate as legacy

    root, config_path, output_path = Path(root), Path(config_path), Path(output_path)
    cfg = json.loads(config_path.read_text())
    base = json.loads((root / cfg["base_config"]).read_text())
    s2 = json.loads((root / cfg["s2_config"]).read_text())
    s0 = json.loads((root / cfg["s0_receipt"]).read_text())
    evaluation = json.loads((root / "configs/swimmer_s3r_evaluation_v1.json").read_text())
    seed = int(evaluation["validation"]["seed"])
    theta = legacy._systems(evaluation["validation"]["systems"], seed, base["persistent_prior"])[0]
    legacy._init_worker(base, s2, float(s0["chosen_horizon_s"]))
    baseline, candidate, baseline_meta, candidate_meta, _ = legacy._worker((seed, 0, theta, 1))

    model = SwimmerModel(base["model"])
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    ih, ie, iq, observed_h, observed_e, observed_q = _context_nuisance(
        base, model, theta, history, query, landmarks, seed, 0, 0,
    )
    history_actions = np.asarray([_landmark_actions(value, landmarks) for value in history.values()])
    query_actions = np.asarray([_landmark_actions(value, landmarks) for value in query.values()])
    init_h = np.concatenate([ih.qpos[2:], ih.qvel])
    init_e = np.concatenate([ie.qpos[2:], ie.qvel])
    init_q = np.concatenate([iq.qpos[2:], iq.qvel])
    checks = []
    for h in range(6):
        for qindex in range(6):
            b = np.flatnonzero(np.all(baseline_meta == np.asarray([0, 0, h, qindex]), axis=1))
            if len(b) != 1:
                raise RuntimeError("legacy baseline metadata is not unique")
            expected_history = np.stack((np.concatenate([init_h, observed_h[h], history_actions[h]]), np.zeros(48)))
            expected_query = np.concatenate([init_q, query_actions[qindex]])
            checks.extend([
                np.array_equal(baseline.history[b[0]], expected_history.astype(baseline.history.dtype)),
                np.array_equal(baseline.query_action[b[0]], expected_query.astype(baseline.query_action.dtype)),
                np.array_equal(baseline.target[b[0]], observed_q[qindex].astype(baseline.target.dtype)),
            ])
            for e in range(6):
                c = np.flatnonzero(np.all(candidate_meta == np.asarray([0, 0, h, e, qindex]), axis=1))
                if len(c) != 1:
                    raise RuntimeError("legacy candidate metadata is not unique")
                expected_candidate = np.stack((expected_history[0], np.concatenate([init_e, observed_e[e], history_actions[e]])))
                checks.extend([
                    np.array_equal(candidate.history[c[0]], expected_candidate.astype(candidate.history.dtype)),
                    np.array_equal(candidate.query_action[c[0]], expected_query.astype(candidate.query_action.dtype)),
                    np.array_equal(candidate.target[c[0]], observed_q[qindex].astype(candidate.target.dtype)),
                ])
    passed = bool(all(checks))
    receipt = {
        "status": "ARTICULATED_LQA_INTERFACE_PARITY_VERIFIED" if passed else "ARTICULATED_LQA_INTERFACE_PARITY_FAILED",
        "passed": passed,
        "array_exact_checks": len(checks),
        "legacy_seed": seed,
        "legacy_system_index": 0,
        "semantic_fields": ["history_segments", "history_mask", "query_initial_state_and_action", "query_target"],
        "source_hashes": {
            "config": sha256(config_path),
            "legacy_evaluator": sha256(Path(legacy.__file__)),
            "implementation": sha256(Path(__file__)),
        },
        "learner_outcomes_read": False,
        "sealed_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("Articulated LQA interface parity failed")
    return receipt


def freshness_audit(root: Path, config_path: Path, output_path: Path) -> dict:
    """Prove development/formal Sobol pools do not reuse known systems."""

    root, config_path, output_path = Path(root), Path(config_path), Path(output_path)
    cfg = json.loads(config_path.read_text())
    base = json.loads((root / cfg["base_config"]).read_text())
    known_parts = []
    known_sources = []
    for relative in (
        "runs/formal/swimmer_s2r_learner_v1/data/train_systems.npy",
        "runs/formal/swimmer_s2r_learner_v1/data/select_systems.npy",
    ):
        path = root / relative
        known_parts.append(np.load(path, allow_pickle=False)); known_sources.append(path)
    for relative, field in (
        ("runs/formal/swimmer_s3r_v1/discovery_replay/discovery_row_level_replay.npz", "systems"),
        ("runs/formal/swimmer_s3r_v1/validation/validation_row_level_replay.npz", "systems"),
    ):
        path = root / relative
        with np.load(path, allow_pickle=False) as archive:
            known_parts.append(archive[field].copy())
        known_sources.append(path)
    # S1 used deterministic per-system NumPy streams rather than a saved pool.
    s1 = []
    for index in range(int(base["s1"]["systems"])):
        rng = np.random.default_rng(np.random.SeedSequence([base["s1"]["seed"], index, 0]))
        s1.append(_sample_log_scales(rng, base["persistent_prior"]))
    known_parts.append(np.asarray(s1)); known_sources.append(root / "runs/formal/swimmer_s1r_physical_v1_3/s1_receipt.json")
    known = np.concatenate(known_parts)
    development = system_pool(cfg["development"]["systems"], cfg["development"]["system_seed"], base["persistent_prior"])
    formal = system_pool(cfg["formal"]["pool_max"], cfg["formal"]["system_seed"], base["persistent_prior"])
    as_set = lambda value: {tuple(row) for row in np.round(np.asarray(value), 12)}
    known_set, development_set, formal_set = as_set(known), as_set(development), as_set(formal)
    overlap = {
        "development_vs_known": len(development_set & known_set),
        "formal_vs_known": len(formal_set & known_set),
        "development_vs_formal": len(development_set & formal_set),
    }
    passed = all(value == 0 for value in overlap.values()) and len(development_set) == len(development) and len(formal_set) == len(formal)
    receipt = {
        "status": "ARTICULATED_LQA_FRESHNESS_VERIFIED" if passed else "ARTICULATED_LQA_FRESHNESS_FAILED",
        "passed": passed,
        "known_systems": len(known),
        "development_systems": len(development),
        "formal_pool_max": len(formal),
        "overlap": overlap,
        "source_hashes": {str(path.relative_to(root)): sha256(path) for path in known_sources},
        "config_sha256": sha256(config_path),
        "learner_outcomes_read": False,
        "sealed_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("fresh-system overlap audit failed")
    return receipt


def benchmark(root: Path, config_path: Path, output_root: Path, contexts: int, particle_max: int, geometry_particles: int, scrambles_per_stream: int, device_name: str, shard_index: int = 0, shard_count: int = 1) -> dict:
    root, config_path, output_root = Path(root), Path(config_path), Path(output_root)
    cfg = json.loads(config_path.read_text())
    base = json.loads((root / cfg["base_config"]).read_text())
    s0_path = root / cfg["s0_receipt"]
    s0 = json.loads(s0_path.read_text())
    if s0["status"] != "S0_GO" or s0["config_sha256"] != sha256(root / cfg["base_config"]):
        raise RuntimeError("Articulated S0/base hash binding failed")
    if any(cfg["outcome_blindness"].values()):
        raise RuntimeError("launch check must remain outcome blind and isolated")
    device = torch.device(device_name)
    learner, norms, training_receipt = _load_jepa(root, cfg, device)
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    systems = system_pool(cfg["development"]["systems"], cfg["development"]["system_seed"], base["persistent_prior"])
    selected_all = fixed_contexts(len(systems), cfg["development"]["nuisance_realizations"], 6, contexts, "articulated-lqa-benchmark-v1")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    selected = selected_all.iloc[np.arange(len(selected_all)) % shard_count == shard_index].copy()
    levels = tuple(level for level in cfg["reference"]["particle_levels"] if level <= particle_max)
    if not levels or levels[-1] != particle_max:
        raise ValueError("particle_max must be one of the frozen particle levels")
    outcome_levels = tuple(level for level in cfg["reference"]["outcome_levels"] if level <= particle_max)
    parameter_scale = prior_log_scale(base["persistent_prior"])
    rows = []
    start_time = time.perf_counter()
    model = SwimmerModel(base["model"])
    for position, context in enumerate(selected.itertuples(index=False)):
        theta = systems[int(context.system_index)]
        ih, ie, iq, observed_h, _, _ = _context_nuisance(
            base, model, theta, history, query, landmarks,
            cfg["development"]["context_seed"], int(context.system_index), int(context.realization),
        )
        history_index = int(context.history_index)
        true_anchor = observed_h[history_index]
        stream_values = {}
        stream_ess = {}
        stream_parity = {}
        for stream_name, seeds in (("ref_a", cfg["reference"]["ref_a_scramble_seeds"]), ("ref_b", cfg["reference"]["ref_b_scramble_seeds"])):
            values = []
            esses = []
            parities = []
            for scramble, seed in enumerate(seeds[:scrambles_per_stream]):
                value, _, ess, parity, _ = reference_stream(
                    base, model, history, query, landmarks, true_anchor, ih, ie, iq,
                    history_index, particle_max, int(seed), outcome_levels,
                    int(seed) + 1000003 * position, norms["target_std"],
                )
                values.append(value); esses.append(ess); parities.append(parity)
            stream_values[stream_name] = np.asarray(values)
            stream_ess[stream_name] = esses
            stream_parity[stream_name] = parities

        geometry = particle_pool(geometry_particles, cfg["reference"]["geometry_scramble_seed"], base["persistent_prior"])
        hmean = _response_bank(model, geometry, ih, history, landmarks)[:, history_index]
        posterior = posterior_from_observation(true_anchor, hmean, base["observation"]["sensor_std"], np.array([1.0]), np.array([1.0])).weights
        candidate_means, candidate_jac = _response_jacobian_bank(
            model, geometry, ie, history, landmarks, parameter_scale,
            cfg["reference"]["finite_difference_log_step"],
        )
        query_means_raw, query_jac_raw = _response_jacobian_bank(
            model, geometry, iq, query, landmarks, parameter_scale,
            cfg["reference"]["finite_difference_log_step"],
        )
        accessibility, local = accessibility_bank(
            geometry, posterior, candidate_means, candidate_jac, query_jac_raw,
            norms["target_std"], base["observation"]["sensor_std"], parameter_scale,
        )
        lqa, raw = lqa_bank(
            learner, norms, posterior, (history_index, true_anchor), ih, ie, iq,
            candidate_means, query_means_raw, history, query, landmarks, device,
        )
        for candidate in range(6):
            for qindex in range(6):
                row = {
                    "system_index": int(context.system_index),
                    "realization": int(context.realization),
                    "history_index": history_index,
                    "candidate_index": candidate,
                    "query_index": qindex,
                    "accessibility": float(accessibility[candidate, qindex]),
                    "local_value": float(local[candidate, qindex]),
                    "lqa": float(lqa[candidate, qindex]),
                    "raw_cka": float(raw[candidate, qindex]),
                    "geometry_posterior_ess": float(1.0 / np.sum(posterior * posterior)),
                }
                for stream in ("ref_a", "ref_b"):
                    values = stream_values[stream]
                    row[f"{stream}_vb"] = float(values[:, -1, candidate, qindex].mean())
                    row[f"{stream}_vb_se"] = float(values[:, -1, candidate, qindex].std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
                    row[f"{stream}_posterior_ess"] = float(np.mean(stream_ess[stream]))
                    row[f"{stream}_dual_path_max_abs"] = float(np.max(stream_parity[stream]))
                    for level_position, level in enumerate(outcome_levels):
                        row[f"{stream}_vb_n{level}"] = float(values[:, level_position, candidate, qindex].mean())
                    for scramble in range(len(values)):
                        row[f"{stream}_vb_scramble{scramble}"] = float(values[scramble, -1, candidate, qindex])
                rows.append(row)
    elapsed = time.perf_counter() - start_time
    table = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "launch_benchmark_rows.csv.gz"
    table.to_csv(rows_path, index=False, compression="gzip")
    opposing = 0
    matched_direction_checks = 0
    for _, cell in table.groupby(["system_index", "realization", "history_index", "query_index"]):
        for first in range(6):
            for second in range(first + 1, 6):
                a = float(cell.loc[cell.candidate_index == first, "accessibility"].iloc[0] - cell.loc[cell.candidate_index == second, "accessibility"].iloc[0])
                l = float(cell.loc[cell.candidate_index == first, "lqa"].iloc[0] - cell.loc[cell.candidate_index == second, "lqa"].iloc[0])
                if np.sign(a) != 0 and np.sign(l) != 0:
                    matched_direction_checks += 1
                    opposing += int(np.sign(a) != np.sign(l))
    maximum_parity = float(table[["ref_a_dual_path_max_abs", "ref_b_dual_path_max_abs"]].max().max())
    convergence = {}
    for stream in ("ref_a", "ref_b"):
        if len(outcome_levels) > 1:
            high = table[f"{stream}_vb_n{outcome_levels[-1]}"].to_numpy()
            low = table[f"{stream}_vb_n{outcome_levels[-2]}"].to_numpy()
            convergence[stream] = {"median_abs_last_step": float(np.median(np.abs(high - low))), "p95_abs_last_step": float(np.quantile(np.abs(high - low), .95))}
    receipt = {
        "status": "ARTICULATED_LQA_LAUNCH_BENCHMARK_COMPLETE_OUTCOME_BLIND",
        "assay_validity": "PENDING_FULL_DEVELOPMENT_FEASIBILITY",
        "contexts": int(len(selected)),
        "total_contexts_before_sharding": int(len(selected_all)),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "rows": int(len(table)),
        "particle_max": particle_max,
        "geometry_particles": geometry_particles,
        "scrambles_per_stream": scrambles_per_stream,
        "elapsed_seconds": elapsed,
        "seconds_per_context": elapsed / max(len(selected), 1),
        "dual_path_max_abs": maximum_parity,
        "posterior_ess": {
            "ref_a_min": float(table.ref_a_posterior_ess.min()),
            "ref_b_min": float(table.ref_b_posterior_ess.min()),
            "geometry_min": float(table.geometry_posterior_ess.min()),
        },
        "reference_last_step_convergence": convergence,
        "opposing_orientation_fraction_before_vb_matching": opposing / max(matched_direction_checks, 1),
        "opposing_orientation_checks": matched_direction_checks,
        "canonical_axes": list(AXES),
        "learner_outcomes_read": False,
        "raw_cka_used_for_pair_selection": False,
        "device": device_name,
        "source_hashes": {
            "protocol": sha256(root / cfg["protocol"]),
            "config": sha256(config_path),
            "base_config": sha256(root / cfg["base_config"]),
            "s0_receipt": sha256(s0_path),
            "jepa": training_receipt["checkpoint_hashes"]["jepa"],
            "normalization": training_receipt["checkpoint_hashes"]["normalization"],
            "rows": sha256(rows_path),
            "implementation": sha256(Path(__file__)),
        },
        "sealed_accessed": False,
        "protected_scope_1_touched": False,
        "protected_scope_2_touched": False,
    }
    (output_root / "launch_benchmark_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def merge_benchmarks(shard_root: Path, output_root: Path) -> dict:
    shard_root, output_root = Path(shard_root), Path(output_root)
    receipts = sorted(shard_root.glob("shard_*/launch_benchmark_receipt.json"))
    if not receipts:
        raise RuntimeError("no launch benchmark shards found")
    payloads = [json.loads(path.read_text()) for path in receipts]
    shard_count = int(payloads[0]["shard_count"])
    indices = sorted(int(item["shard_index"]) for item in payloads)
    if indices != list(range(shard_count)):
        raise RuntimeError(f"incomplete shard set: {indices} of {shard_count}")
    invariants = ("config", "base_config", "s0_receipt", "jepa", "normalization", "protocol")
    for key in invariants:
        if len({item["source_hashes"][key] for item in payloads}) != 1:
            raise RuntimeError(f"shard source hash mismatch for {key}")
    tables = [pd.read_csv(path.parent / "launch_benchmark_rows.csv.gz") for path in receipts]
    table = pd.concat(tables, ignore_index=True)
    semantic = ["system_index", "realization", "history_index", "candidate_index", "query_index"]
    if table.duplicated(semantic).any():
        raise RuntimeError("duplicate semantic cells across shards")
    expected_contexts = int(payloads[0]["total_contexts_before_sharding"])
    contexts = table[["system_index", "realization", "history_index"]].drop_duplicates()
    if len(contexts) != expected_contexts or len(table) != expected_contexts * 36:
        raise RuntimeError("merged benchmark does not contain the complete named-axis grid")
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "launch_benchmark_rows.csv.gz"
    table.sort_values(semantic).to_csv(rows_path, index=False, compression="gzip")
    receipt = {
        "status": "ARTICULATED_LQA_LAUNCH_BENCHMARK_MERGED_OUTCOME_BLIND",
        "contexts": expected_contexts,
        "rows": len(table),
        "shards": shard_count,
        "dual_path_max_abs": float(table[["ref_a_dual_path_max_abs", "ref_b_dual_path_max_abs"]].max().max()),
        "posterior_ess": {
            "ref_a_min": float(table.ref_a_posterior_ess.min()),
            "ref_b_min": float(table.ref_b_posterior_ess.min()),
            "geometry_min": float(table.geometry_posterior_ess.min()),
        },
        "source_hashes": {key: payloads[0]["source_hashes"][key] for key in invariants},
        "rows_sha256": sha256(rows_path),
        "shard_receipt_sha256": [sha256(path) for path in receipts],
        "learner_outcomes_read": False,
        "sealed_accessed": False,
    }
    (output_root / "launch_benchmark_merged_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("benchmark")
    run.add_argument("root", type=Path)
    run.add_argument("config", type=Path)
    run.add_argument("output_root", type=Path)
    run.add_argument("--contexts", type=int, default=2)
    run.add_argument("--particle-max", type=int, default=256)
    run.add_argument("--geometry-particles", type=int, default=16)
    run.add_argument("--scrambles-per-stream", type=int, default=1)
    run.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    parity = sub.add_parser("interface-parity")
    parity.add_argument("root", type=Path)
    parity.add_argument("config", type=Path)
    parity.add_argument("output", type=Path)
    merge = sub.add_parser("merge-benchmarks")
    merge.add_argument("shard_root", type=Path)
    merge.add_argument("output_root", type=Path)
    fresh = sub.add_parser("freshness-audit")
    fresh.add_argument("root", type=Path)
    fresh.add_argument("config", type=Path)
    fresh.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "interface-parity":
        result = interface_parity(args.root, args.config, args.output)
    elif args.command == "merge-benchmarks":
        result = merge_benchmarks(args.shard_root, args.output_root)
    elif args.command == "freshness-audit":
        result = freshness_audit(args.root, args.config, args.output)
    else:
        result = benchmark(args.root, args.config, args.output_root, args.contexts, args.particle_max, args.geometry_particles, args.scrambles_per_stream, args.device, args.shard_index, args.shard_count)
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
