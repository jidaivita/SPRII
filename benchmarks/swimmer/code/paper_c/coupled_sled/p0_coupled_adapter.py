"""Real Coupled-Sled adapter for a bounded, CPU-only P0 development audit.

The adapter reads only development systems, the posterior particle mother pool,
and train-only normalization.  It never reads discovery, validation, test, or
sealed artifacts and never invokes a learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .development import PARAMETER_NAMES, _features, _load_values
from .fisher import response_jacobian
from .manifests import load_spec
from .p0_reference import (
    P0_CANONICAL_AXES,
    local_fidelity,
    modal_geometry,
    normalize_weights,
    posterior_averaged_gaussian_information,
    query_utility_curvature,
    sha256_file,
    weighted_covariance,
    whitened_operators,
    write_freeze_receipt,
)
from .posterior import (
    lognormal_quadrature,
    posterior_batch_from_observations,
    posterior_batch_from_observations_quadratic,
    posterior_from_observation,
)
from .waveforms import history_probe_bank, query_bank


def _hash_order(values: list[str], namespace: str) -> list[int]:
    return sorted(range(len(values)), key=lambda i: hashlib.sha256(f"{namespace}|{values[i]}".encode()).digest())


def _balanced_block_prefix_indices(particle_rows: list[dict], count: int) -> np.ndarray:
    """Select an equal nested Sobol prefix from every frozen seed block.

    The registry stores blocks contiguously.  A raw global prefix would change
    the mixture of scramble blocks as N grows and is therefore not a valid
    convergence sequence for the intended equal-block numerical estimator.
    """

    blocks = sorted({int(row["sobol_block"]) for row in particle_rows})
    if not blocks or count % len(blocks):
        raise ValueError("particle level must divide equally across Sobol blocks")
    per_block = count // len(blocks)
    if per_block <= 0 or per_block & (per_block - 1):
        raise ValueError("per-block Sobol prefix must be a positive power of two")
    selected: list[int] = []
    for block in blocks:
        members = [index for index, row in enumerate(particle_rows) if int(row["sobol_block"]) == block]
        if per_block > len(members):
            raise ValueError("particle level exceeds a frozen Sobol block")
        selected.extend(members[:per_block])
    return np.asarray(selected, dtype=int)


def _weighted_risk(values: np.ndarray, weights: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> float:
    error = ((values - prediction[None, :]) / scale[None, :]) ** 2
    return float(np.sum(normalize_weights(weights)[:, None] * error) / values.shape[1])


def _draw_predictive_observations(
    posterior: np.ndarray,
    candidate_means: np.ndarray,
    sensor_std: float,
    gain_mu: float,
    gain_sigma: float,
    outcomes: int,
    seed: int,
) -> np.ndarray:
    posterior = normalize_weights(posterior)
    if outcomes <= 0 or outcomes & (outcomes - 1):
        raise ValueError("outcomes must be a power of two for nested Sobol QMC")
    dimension = 2 + candidate_means.shape[1]
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed % (2**32)).random_base2(int(np.log2(outcomes)))
    unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    sampled_particle = np.searchsorted(np.cumsum(posterior), unit[:, 0], side="right")
    sampled_particle = np.minimum(sampled_particle, len(posterior) - 1)
    gains = np.exp(gain_mu + gain_sigma * ndtri(unit[:, 1:2]))
    return gains * candidate_means[sampled_particle] + sensor_std * ndtri(unit[:, 2:])


def _reference_value_paths(
    posterior: np.ndarray,
    candidate_means: np.ndarray,
    query_values: np.ndarray,
    query_scale: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
    gain_mu: float,
    gain_sigma: float,
    outcomes: int,
    seed: int,
    observations_override: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Independent Monte-Carlo aggregations of the same acquisition utility."""

    posterior = normalize_weights(posterior)
    observations = (
        _draw_predictive_observations(
            posterior, candidate_means, sensor_std, gain_mu, gain_sigma, outcomes, seed
        )
        if observations_override is None
        else np.asarray(observations_override, dtype=float)
    )
    if observations.shape != (outcomes, candidate_means.shape[1]):
        raise ValueError("override observations must be [outcome,candidate_feature]")
    before_prediction = np.sum(posterior[:, None] * query_values, axis=0)
    before_risk = _weighted_risk(query_values, posterior, before_prediction, query_scale)

    # Path 1: posterior-predictive Bayes risk reduction under the same finite
    # empirical outcome measure used by Path 2.  The pre-acquisition term is
    # Rao--Blackwellized through p(theta | y) for each sampled y; in expectation
    # this equals the exact prior risk, while at finite outcomes it removes an
    # irrelevant Monte-Carlo discrepancy from the parity check.
    sampled_before_risk = 0.0
    after_risk = 0.0
    # Path 2: system-conditional loss reduction, aggregated in the opposite
    # order.  Rao--Blackwellize theta conditional on every sampled outcome
    # rather than using only the single theta draw that generated that outcome.
    # This is an independent algebraic path to the same finite-outcome
    # estimator and makes parity a semantic check instead of a noisy comparison
    # between two Monte-Carlo estimators.
    before_loss_by_particle = np.mean(
        ((query_values - before_prediction[None, :]) / query_scale[None, :]) ** 2,
        axis=1,
    )
    conditional_sum = 0.0
    # Bound memory independently of the numerical outcome budget.
    for start in range(0, outcomes, 32):
        after_weights, _ = posterior_batch_from_observations(
            observations[start:start + 32], candidate_means, sensor_std, gain_nodes, gain_weights
        )
        # posterior_batch_from_observations assumes a uniform prior.  Apply the
        # H posterior as a prior correction.
        after_weights *= posterior[None, :] * len(posterior)
        after_weights /= after_weights.sum(axis=1, keepdims=True)
        for row in after_weights:
            sampled_before_risk += float(np.sum(row * before_loss_by_particle))
            prediction = np.sum(row[:, None] * query_values, axis=0)
            after_risk += _weighted_risk(query_values, row, prediction, query_scale)
            after_loss_by_particle = np.mean(
                ((query_values - prediction[None, :]) / query_scale[None, :]) ** 2,
                axis=1,
            )
            conditional_sum += float(np.sum(row * (before_loss_by_particle - after_loss_by_particle)))
    path_predictive = (sampled_before_risk - after_risk) / outcomes
    path_conditional = conditional_sum / outcomes
    return path_predictive, path_conditional, before_risk


def _reference_value_paths_all_queries(
    posterior: np.ndarray,
    candidate_means: np.ndarray,
    query_values: np.ndarray,
    query_scale: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
    gain_mu: float,
    gain_sigma: float,
    outcomes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized-Q equivalent of ``_reference_value_paths``.

    Candidate outcome posterior updates are computed exactly once and reused
    for every query. This is the required P3 execution path.
    """

    posterior = normalize_weights(posterior)
    queries = np.asarray(query_values, dtype=float)
    scales = np.asarray(query_scale, dtype=float)
    if queries.ndim != 3 or scales.shape != (queries.shape[0], queries.shape[2]):
        raise ValueError("queries/scales must be [Q,particle,feature] and [Q,feature]")
    if outcomes <= 0 or outcomes & (outcomes - 1):
        raise ValueError("outcomes must be a power of two for nested Sobol QMC")
    dimension = 2 + candidate_means.shape[1]
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed % (2**32)).random_base2(int(np.log2(outcomes)))
    unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    sampled_particle = np.searchsorted(np.cumsum(posterior), unit[:, 0], side="right")
    sampled_particle = np.minimum(sampled_particle, len(posterior) - 1)
    gains = np.exp(gain_mu + gain_sigma * ndtri(unit[:, 1:2]))
    observations = gains * candidate_means[sampled_particle]
    observations += sensor_std * ndtri(unit[:, 2:])
    before_prediction = np.einsum("p,qpf->qf", posterior, queries)
    before_by_particle = np.mean(
        ((queries - before_prediction[:, None, :]) / scales[:, None, :]) ** 2,
        axis=2,
    )
    u_q = np.einsum("p,qp->q", posterior, before_by_particle)
    path_one_sum = np.zeros(len(queries), dtype=float)
    path_two_sum = np.zeros(len(queries), dtype=float)
    for start in range(0, outcomes, 32):
        after_weights, _ = posterior_batch_from_observations(
            observations[start:start + 32], candidate_means, sensor_std, gain_nodes, gain_weights
        )
        after_weights *= posterior[None, :] * len(posterior)
        after_weights /= after_weights.sum(axis=1, keepdims=True)
        after_prediction = np.einsum("op,qpf->oqf", after_weights, queries)
        after_by_outcome_particle = np.mean(
            ((queries[None] - after_prediction[:, :, None, :]) / scales[None, :, None, :]) ** 2,
            axis=3,
        )
        for q in range(len(queries)):
            predictive_before = np.einsum("op,p->o", after_weights, before_by_particle[q])
            predictive_after = np.einsum("op,op->o", after_weights, after_by_outcome_particle[:, q])
            path_one_sum[q] += float(np.sum(predictive_before - predictive_after))
            path_two_sum[q] += float(np.sum(np.einsum(
                "op,op->o", after_weights,
                before_by_particle[q][None, :] - after_by_outcome_particle[:, q],
            )))
    path_one = path_one_sum / outcomes
    path_two = path_two_sum / outcomes
    return path_one, path_two, u_q


def _reference_value_paths_all_queries_moment(
    posterior: np.ndarray,
    candidate_means: np.ndarray,
    query_values: np.ndarray,
    query_scale: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
    gain_mu: float,
    gain_sigma: float,
    outcomes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moment-form equivalent of :func:`_reference_value_paths_all_queries`.

    The old implementation materializes posterior squared errors with shape
    ``[outcome, query, particle, feature]``.  Under standardized squared loss,
    the posterior predictive risk depends only on the weighted first and
    second moments of the query response.  This implementation therefore keeps
    only ``[outcome, query, feature]`` moments while preserving the exact same
    QMC observations, posterior updates, finite-outcome measure, and dual-path
    semantics.

    It is intentionally a separate function until the formal runner is
    explicitly switched after parity validation; already-running shards must
    never mix kernels.
    """

    posterior = normalize_weights(posterior)
    candidates = np.asarray(candidate_means, dtype=float)
    queries = np.asarray(query_values, dtype=float)
    scales = np.asarray(query_scale, dtype=float)
    if candidates.ndim != 2 or len(candidates) != len(posterior):
        raise ValueError("candidate means must be [particle,candidate_feature]")
    if queries.ndim != 3 or queries.shape[1] != len(posterior) or scales.shape != (queries.shape[0], queries.shape[2]):
        raise ValueError("queries/scales must be [Q,particle,feature] and [Q,feature]")
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("query scales must be positive and finite")
    if outcomes <= 0 or outcomes & (outcomes - 1):
        raise ValueError("outcomes must be a power of two for nested Sobol QMC")

    dimension = 2 + candidates.shape[1]
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed % (2**32)).random_base2(int(np.log2(outcomes)))
    unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    sampled_particle = np.searchsorted(np.cumsum(posterior), unit[:, 0], side="right")
    sampled_particle = np.minimum(sampled_particle, len(posterior) - 1)
    gains = np.exp(gain_mu + gain_sigma * ndtri(unit[:, 1:2]))
    observations = gains * candidates[sampled_particle]
    observations += sensor_std * ndtri(unit[:, 2:])

    # Work in standardized query coordinates so posterior predictive MSE is
    # E[X^2] - E[X]^2, averaged over query features.
    standardized = queries / scales[:, None, :]
    standardized_squared = standardized * standardized
    prior_mean = np.einsum("p,qpf->qf", posterior, standardized)
    before_by_particle = np.mean((standardized - prior_mean[:, None, :]) ** 2, axis=2)
    u_q = np.einsum("p,qp->q", posterior, before_by_particle)

    path_one_sum = np.zeros(len(queries), dtype=float)
    path_two_sum = np.zeros(len(queries), dtype=float)
    for start in range(0, outcomes, 32):
        after_weights, _ = posterior_batch_from_observations_quadratic(
            observations[start:start + 32], candidates, sensor_std, gain_nodes, gain_weights
        )
        # The likelihood helper assumes uniform particles; restore p(theta|H).
        after_weights *= posterior[None, :] * len(posterior)
        after_weights /= after_weights.sum(axis=1, keepdims=True)

        first_moment = np.einsum("op,qpf->oqf", after_weights, standardized)
        second_moment = np.einsum("op,qpf->oqf", after_weights, standardized_squared)
        posterior_risk = np.mean(second_moment - first_moment * first_moment, axis=2)

        # Path 1: posterior-predictive before risk minus posterior risk after E.
        predictive_before = np.einsum("op,qp->oq", after_weights, before_by_particle)
        path_one_sum += np.sum(predictive_before - posterior_risk, axis=0)

        # Path 2: system-conditional loss reduction, integrated in the opposite
        # algebraic order.  This deliberately does not reuse predictive_before.
        expected_before = np.mean(
            second_moment
            - 2.0 * prior_mean[None, :, :] * first_moment
            + prior_mean[None, :, :] * prior_mean[None, :, :],
            axis=2,
        )
        expected_after = np.mean(
            second_moment - 2.0 * first_moment * first_moment + first_moment * first_moment,
            axis=2,
        )
        path_two_sum += np.sum(expected_before - expected_after, axis=0)

    return path_one_sum / outcomes, path_two_sum / outcomes, u_q


def _reference_value_paths_all_queries_moment_nested(
    posterior: np.ndarray,
    candidate_means: np.ndarray,
    query_values: np.ndarray,
    query_scale: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
    gain_mu: float,
    gain_sigma: float,
    outcome_levels: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-scramble nested-prefix reference estimates at every frozen level.

    The maximum Owen-scrambled Sobol sequence is drawn once.  Cumulative
    moment-form utility sums are snapshotted at the requested nested power-of-
    two prefixes, so lower levels are exact prefixes rather than independent
    redraws.  Returns two dual paths with shape ``[level, query]`` and the
    pre-acquisition query risk ``[query]``.
    """

    levels = tuple(int(value) for value in outcome_levels)
    if not levels or levels != tuple(sorted(set(levels))):
        raise ValueError("nested outcome levels must be unique and increasing")
    if any(value <= 0 or value & (value - 1) for value in levels):
        raise ValueError("every nested outcome level must be a positive power of two")
    posterior = normalize_weights(posterior)
    candidates = np.asarray(candidate_means, dtype=float)
    queries = np.asarray(query_values, dtype=float)
    scales = np.asarray(query_scale, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] != len(posterior):
        raise ValueError("candidate means must be [particle,candidate_feature]")
    if queries.ndim != 3 or queries.shape[1] != len(posterior) or scales.shape != (queries.shape[0], queries.shape[2]):
        raise ValueError("queries/scales must be [Q,particle,feature] and [Q,feature]")
    maximum = levels[-1]
    dimension = 2 + candidates.shape[1]
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed % (2**32)).random_base2(int(np.log2(maximum)))
    unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    sampled_particle = np.searchsorted(np.cumsum(posterior), unit[:, 0], side="right")
    sampled_particle = np.minimum(sampled_particle, len(posterior) - 1)
    gains = np.exp(gain_mu + gain_sigma * ndtri(unit[:, 1:2]))
    observations = gains * candidates[sampled_particle]
    observations += sensor_std * ndtri(unit[:, 2:])

    standardized = queries / scales[:, None, :]
    standardized_squared = standardized * standardized
    prior_mean = np.einsum("p,qpf->qf", posterior, standardized)
    before_by_particle = np.mean((standardized - prior_mean[:, None, :]) ** 2, axis=2)
    u_q = np.einsum("p,qp->q", posterior, before_by_particle)
    path_one_sum = np.zeros(len(queries), dtype=float)
    path_two_sum = np.zeros(len(queries), dtype=float)
    path_one_levels = np.empty((len(levels), len(queries)), dtype=float)
    path_two_levels = np.empty_like(path_one_levels)
    level_position = 0
    for start in range(0, maximum, 32):
        stop = min(start + 32, maximum)
        after_weights, _ = posterior_batch_from_observations_quadratic(
            observations[start:stop], candidates, sensor_std, gain_nodes, gain_weights
        )
        after_weights *= posterior[None, :] * len(posterior)
        after_weights /= after_weights.sum(axis=1, keepdims=True)
        first_moment = np.einsum("op,qpf->oqf", after_weights, standardized)
        second_moment = np.einsum("op,qpf->oqf", after_weights, standardized_squared)
        posterior_risk = np.mean(second_moment - first_moment * first_moment, axis=2)
        predictive_before = np.einsum("op,qp->oq", after_weights, before_by_particle)
        path_one_sum += np.sum(predictive_before - posterior_risk, axis=0)
        expected_before = np.mean(
            second_moment - 2.0 * prior_mean[None] * first_moment + prior_mean[None] ** 2,
            axis=2,
        )
        expected_after = np.mean(second_moment - first_moment * first_moment, axis=2)
        path_two_sum += np.sum(expected_before - expected_after, axis=0)
        while level_position < len(levels) and stop == levels[level_position]:
            level = levels[level_position]
            path_one_levels[level_position] = path_one_sum / level
            path_two_levels[level_position] = path_two_sum / level
            level_position += 1
    if level_position != len(levels):
        raise RuntimeError("nested reference kernel failed to snapshot every level")
    return path_one_levels, path_two_levels, u_q


def run_bounded_p0_audit(
    spec_path: Path,
    manifest_root: Path,
    normalization_path: Path,
    output_root: Path,
    system_count: int = 2,
    history_count: int = 2,
    particle_levels: tuple[int, ...] = (256, 512, 1024),
    outcomes: int = 8,
    seed: int = 58101,
    candidate_common_random_numbers: bool = False,
) -> dict:
    spec = load_spec(spec_path)
    if not spec["access"]["development"] or any(spec["access"][key] for key in ("discovery", "design_validation", "sealed")):
        raise RuntimeError("P0 adapter requires a development-only base spec")
    if not particle_levels or tuple(sorted(set(particle_levels))) != particle_levels:
        raise ValueError("particle levels must be unique and increasing")
    systems_payload = json.loads((manifest_root / "development_v0_1.json").read_text())
    system_ids = [row["system_id"] for row in systems_payload["systems"]]
    systems_all = _load_values(manifest_root / "development_v0_1.json", "systems")
    selected = _hash_order(system_ids, "coupled-p0-development-v1")[:system_count]
    systems = systems_all[selected]
    selected_ids = [system_ids[index] for index in selected]
    particle_payload = json.loads((manifest_root / "posterior_particles_v0_1.json").read_text())
    particle_rows = particle_payload["particles"]
    particles_all = _load_values(manifest_root / "posterior_particles_v0_1.json", "particles")
    if particle_levels[-1] > len(particles_all):
        raise ValueError("requested particle level exceeds fixed mother pool")
    level_indices = [_balanced_block_prefix_indices(particle_rows, count) for count in particle_levels]
    particles = particles_all[level_indices[-1]]
    parameter_scale = particles_all.std(axis=0, ddof=1)
    parameter_center = particles_all.mean(axis=0)
    z_particles = (particles - parameter_center) / parameter_scale

    cfg = spec["development_v0_1"]
    dt = spec["dynamics"]["reference_dt_s"]
    history_bank = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    query_map = query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"]))
    history_ids_all = sorted(history_bank)
    history_indices = _hash_order(history_ids_all, "coupled-p0-history-v1")[:history_count]
    anchor_ids = [history_ids_all[index] for index in history_indices]
    candidate_ids = history_ids_all
    query_ids = sorted(query_map)

    particle_history = _features(particles, history_bank, cfg["history_landmarks"], cfg["observation_semantics"])
    system_history = _features(systems, history_bank, cfg["history_landmarks"], cfg["observation_semantics"])
    particle_query = _features(particles, query_map, cfg["query_landmarks"], "both_positions")
    norms = np.load(normalization_path, allow_pickle=False)
    query_scale = np.asarray(norms["target_std"], dtype=float)

    candidate_jacobians = {}
    candidate_means = {}
    for candidate in candidate_ids:
        mean, jacobian = response_jacobian(
            particles, history_bank[candidate], cfg["history_landmarks"], cfg["observation_semantics"], parameter_scale
        )
        candidate_means[candidate] = mean
        candidate_jacobians[candidate] = jacobian
    query_jacobians = {}
    for query in query_ids:
        _, jacobian = response_jacobian(
            particles, query_map[query], cfg["query_landmarks"], "both_positions", parameter_scale
        )
        query_jacobians[query] = jacobian

    gain_nodes, gain_weights = lognormal_quadrature(
        spec["actuator_gain"]["mean"], spec["actuator_gain"]["cv"], spec["actuator_gain"]["quadrature_points"]
    )
    log_variance = np.log1p(spec["actuator_gain"]["cv"] ** 2)
    gain_sigma = float(np.sqrt(log_variance))
    gain_mu = float(-0.5 * log_variance)
    sensor_std = float(cfg["sensor_std_m"])
    gain_variance = float(spec["actuator_gain"]["cv"] ** 2)
    # Freeze one highest-level posterior-predictive outcome set per
    # (system,H,E).  Every lower particle estimator is evaluated on these same
    # observations, so particle convergence is not contaminated by drawing a
    # different empirical outcome distribution at each N.
    common_candidate_observations: dict[tuple[int, int, str], np.ndarray] = {}
    for s in range(system_count):
        for h, anchor in enumerate(anchor_ids):
            anchor_seed = int.from_bytes(hashlib.sha256(
                f"{seed}|H|{selected_ids[s]}|{anchor}".encode()
            ).digest()[:8], "little")
            anchor_rng = np.random.default_rng(anchor_seed)
            anchor_observation = anchor_rng.lognormal(gain_mu, gain_sigma) * system_history[anchor][s]
            anchor_observation += anchor_rng.normal(0.0, sensor_std, size=anchor_observation.shape)
            max_posterior = posterior_from_observation(
                anchor_observation, particle_history[anchor], sensor_std, gain_nodes, gain_weights
            ).weights
            for candidate in candidate_ids:
                seed_key = (
                    f"{seed}|E-CRN|{selected_ids[s]}|{anchor}"
                    if candidate_common_random_numbers
                    else f"{seed}|E|{selected_ids[s]}|{anchor}|{candidate}"
                )
                outcome_seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
                common_candidate_observations[(s, h, candidate)] = _draw_predictive_observations(
                    max_posterior, candidate_means[candidate], sensor_std,
                    gain_mu, gain_sigma, outcomes, outcome_seed,
                )
    shape = (system_count, 1, history_count, len(candidate_ids), len(query_ids))
    level_reference = np.empty((len(particle_levels),) + shape, dtype=float)
    level_parity = np.empty_like(level_reference)
    primary_local = np.empty(shape, dtype=float)
    primary_accessibility = np.empty(shape, dtype=float)
    mode_shape = shape + (len(PARAMETER_NAMES),)
    eigenvalues = np.empty(mode_shape, dtype=float)
    query_weights = np.empty(mode_shape, dtype=float)
    contributions = np.empty(mode_shape, dtype=float)
    u_q = np.empty((system_count, 1, history_count, len(query_ids)), dtype=float)

    for level_index, particle_count in enumerate(particle_levels):
        # Each level is an equal prefix from every frozen scramble block.
        # Re-index into the full highest-level arrays to preserve this legal
        # nested estimator rather than taking a block-imbalanced global slice.
        global_indices = level_indices[level_index]
        highest_lookup = {int(value): index for index, value in enumerate(level_indices[-1])}
        local_indices = np.asarray([highest_lookup[int(value)] for value in global_indices], dtype=int)
        theta = particles[local_indices]
        for s in range(system_count):
            for h, anchor in enumerate(anchor_ids):
                anchor_seed = int.from_bytes(hashlib.sha256(f"{seed}|H|{selected_ids[s]}|{anchor}".encode()).digest()[:8], "little")
                rng = np.random.default_rng(anchor_seed)
                observation = rng.lognormal(gain_mu, gain_sigma) * system_history[anchor][s]
                observation += rng.normal(0.0, sensor_std, size=observation.shape)
                posterior = posterior_from_observation(
                    observation, particle_history[anchor][local_indices], sensor_std, gain_nodes, gain_weights
                ).weights
                if particle_count == particle_levels[-1]:
                    covariance = weighted_covariance(z_particles[local_indices], posterior)
                    candidate_information = {
                        candidate: posterior_averaged_gaussian_information(
                            candidate_means[candidate][local_indices], candidate_jacobians[candidate][local_indices],
                            posterior, sensor_std, gain_variance,
                        ) for candidate in candidate_ids
                    }
                for q, query in enumerate(query_ids):
                    scale = query_scale
                    if particle_count == particle_levels[-1]:
                        curvature = query_utility_curvature(
                            query_jacobians[query][local_indices], posterior, scale
                        )
                    for e, candidate in enumerate(candidate_ids):
                        # The random stream is deliberately independent of the
                        # particle level.  Fixed uniforms/noise therefore give
                        # a legal common-random-number convergence comparison
                        # across nested prefixes.
                        outcome_seed = int.from_bytes(hashlib.sha256(
                            f"{seed}|E|{selected_ids[s]}|{anchor}|{candidate}|{query}".encode()
                        ).digest()[:8], "little")
                        path_one, path_two, residual = _reference_value_paths(
                            posterior, particle_history[candidate][local_indices], particle_query[query][local_indices],
                            scale, sensor_std, gain_nodes, gain_weights, gain_mu, gain_sigma, outcomes, outcome_seed,
                            common_candidate_observations[(s, h, candidate)],
                        )
                        level_reference[level_index, s, 0, h, e, q] = path_one
                        level_parity[level_index, s, 0, h, e, q] = path_one - path_two
                        if particle_count == particle_levels[-1]:
                            j, c = whitened_operators(covariance, candidate_information[candidate], curvature)
                            modes = modal_geometry(j, c)
                            primary_local[s, 0, h, e, q] = modes.local_value
                            primary_accessibility[s, 0, h, e, q] = modes.accessibility
                            eigenvalues[s, 0, h, e, q] = modes.eigenvalues
                            query_weights[s, 0, h, e, q] = modes.query_weights
                            contributions[s, 0, h, e, q] = modes.mode_contributions
                            u_q[s, 0, h, q] = residual

    reference = level_reference[-1]
    fidelity_arrays = {
        "fidelity_normalized_max_value_error": np.empty(shape),
        "fidelity_normalized_range_error": np.empty(shape),
        "fidelity_normalized_margin_error": np.empty(shape),
        "fidelity_winner_agreement": np.empty(shape, dtype=bool),
        "fidelity_rank_correlation": np.empty(shape),
    }
    numerical_floor = max(float(np.quantile(np.abs(level_reference[-1] - level_reference[-2]), 0.95)), 1e-12)
    for s in range(system_count):
        for h in range(history_count):
            for q in range(len(query_ids)):
                fidelity = local_fidelity(primary_local[s, 0, h, :, q], reference[s, 0, h, :, q], u_q[s, 0, h, q], numerical_floor)
                for name, value in (
                    ("fidelity_normalized_max_value_error", fidelity.normalized_max_value_error),
                    ("fidelity_normalized_range_error", fidelity.normalized_range_error),
                    ("fidelity_normalized_margin_error", fidelity.normalized_margin_error),
                    ("fidelity_winner_agreement", fidelity.winner_agreement),
                    ("fidelity_rank_correlation", fidelity.rank_correlation),
                ):
                    fidelity_arrays[name][s, 0, h, :, q] = value

    artifact = {
        "reference_value": reference,
        "reference_value_by_particle_level": level_reference,
        "dual_path_difference_by_particle_level": level_parity,
        "local_value": primary_local,
        "accessibility": primary_accessibility,
        "u_q": np.broadcast_to(u_q[:, :, :, None, :], shape).copy(),
        "mode_eigenvalues": eigenvalues,
        "mode_query_weights": query_weights,
        "mode_contributions": contributions,
        **fidelity_arrays,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / "p0_canonical_srheq.npz"
    np.savez_compressed(tensor_path, **artifact)
    axis_payload = {
        "canonical_axes": list(P0_CANONICAL_AXES),
        "mode_axis": "parameter_mode",
        "particle_level_axis": "particle_level",
        "particle_levels": list(particle_levels),
        "particle_estimator_construction": "equal_nested_prefix_per_frozen_sobol_scramble_block_with_per_level_weight_normalization",
        "particle_prefix_legality": "manifest particles are fixed prior-integration particles; every level takes the same power-of-two prefix from each block; no resampling, MCMC trimming, or N-dependent proposal",
        "system_source_indices": selected,
        "system_ids": selected_ids,
        "history_ids": anchor_ids,
        "candidate_ids": candidate_ids,
        "query_ids": query_ids,
        "shape": list(shape),
    }
    axis_path = output_root / "p0_axis_spec.json"
    axis_path.write_text(json.dumps(axis_payload, indent=2, sort_keys=True) + "\n")
    convergence_delta = np.abs(level_reference[1:] - level_reference[:-1])
    summary = {
        "status": "P0_BOUNDED_DEVELOPMENT_AUDIT_COMPLETE_GATES_PENDING",
        "device": "cpu",
        "development_systems": system_count,
        "histories": history_count,
        "candidates": len(candidate_ids),
        "queries": len(query_ids),
        "particle_levels": list(particle_levels),
        "outcomes_per_candidate": outcomes,
        "candidate_common_random_numbers": bool(candidate_common_random_numbers),
        "reference_value_range": [float(reference.min()), float(reference.max())],
        "consecutive_particle_delta_p95": [float(np.quantile(row, 0.95)) for row in convergence_delta],
        "dual_path_absolute_difference_p95": [float(np.quantile(np.abs(row), 0.95)) for row in level_parity],
        "local_reference_spearman_mean": float(np.nanmean(fidelity_arrays["fidelity_rank_correlation"][:, 0, :, 0, :])),
        "local_winner_agreement_rate": float(np.mean(fidelity_arrays["fidelity_winner_agreement"][:, 0, :, 0, :])),
        "local_normalized_max_error_median": float(np.median(fidelity_arrays["fidelity_normalized_max_value_error"][:, 0, :, 0, :])),
        "accessibility_range": [float(np.nanmin(primary_accessibility)), float(np.nanmax(primary_accessibility))],
        "mechanical_gates_passed": False,
        "reason": "bounded development audit establishes real adapter behavior; full convergence/fidelity thresholds remain to be frozen from a higher-accuracy audit",
        "discovery_accessed": False,
        "validation_accessed": False,
        "sealed_accessed": False,
        "learner_accessed": False,
        "mps_accessed": False,
        "cuda_accessed": False,
    }
    summary_path = output_root / "p0_bounded_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt = write_freeze_receipt(
        output_root / "p0_bounded_receipt.json", summary["status"],
        (spec_path, manifest_root / "development_v0_1.json", manifest_root / "posterior_particles_v0_1.json", normalization_path, tensor_path, axis_path, summary_path),
        {"mechanical_gates_passed": False, "bounded_real_adapter": True},
    )
    return {**summary, "receipt_sha256": sha256_file(output_root / "p0_bounded_receipt.json"), "receipt": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded real-data CPU-only Coupled P0 audit")
    parser.add_argument("spec", type=Path)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("normalization", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--systems", type=int, default=2)
    parser.add_argument("--histories", type=int, default=2)
    parser.add_argument("--particle-levels", default="256,512,1024")
    parser.add_argument("--outcomes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=58101)
    parser.add_argument("--candidate-common-random-numbers", action="store_true")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    levels = tuple(int(value) for value in args.particle_levels.split(","))
    result = run_bounded_p0_audit(
        args.spec, args.manifest_root, args.normalization, args.output_root,
        args.systems, args.histories, levels, args.outcomes, args.seed,
        args.candidate_common_random_numbers,
    )
    print(json.dumps({key: result[key] for key in ("status", "mechanical_gates_passed", "receipt_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
