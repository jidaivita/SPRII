"""CPU-only reference primitives for the Coupled Sled theory-validation P0.

This module deliberately separates three objects:

* a particle/reference Bayes value under one standardized query-MSE utility;
* a posterior-averaged *local Gaussian information surrogate* for an experience;
* local curvature of that same query utility.

The modal quantities are diagnostics/predictors.  They are not silently
substituted for the particle/reference value used by prospective matching.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .fisher import gain_aware_fisher


@dataclass(frozen=True)
class NestedEstimatorLevel:
    particle_count: int
    construction: str
    parent_count: int | None = None


@dataclass(frozen=True)
class ModeGeometry:
    eigenvalues: np.ndarray
    query_weights: np.ndarray
    mode_contributions: np.ndarray
    local_value: float
    accessibility: float


@dataclass(frozen=True)
class FidelityResult:
    normalized_max_value_error: float
    normalized_range_error: float
    normalized_margin_error: float
    winner_agreement: bool
    rank_correlation: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cpu_only(device: str) -> None:
    if device.lower() != "cpu":
        raise ValueError("P0 is CPU-only; CUDA and MPS are forbidden")


def validate_nested_estimators(levels: Sequence[NestedEstimatorLevel]) -> None:
    """Validate nesting semantics without assuming an array-prefix estimator.

    ``fixed_pool_prefix`` and ``qmc_nested`` may name a parent level.  MCMC,
    resampled, or proposal-changing estimators must instead be marked
    ``independent_valid`` and cannot claim prefix nesting.
    """

    allowed = {"fixed_pool_prefix", "qmc_nested", "independent_valid"}
    counts: set[int] = set()
    previous = 0
    for level in levels:
        if level.particle_count <= previous:
            raise ValueError("particle counts must be strictly increasing")
        if level.construction not in allowed:
            raise ValueError(f"unknown estimator construction: {level.construction}")
        if level.construction in {"fixed_pool_prefix", "qmc_nested"}:
            if level.parent_count is not None and level.parent_count not in counts:
                raise ValueError("nested estimator parent must be an earlier legal level")
        elif level.parent_count is not None:
            raise ValueError("independent_valid estimators cannot claim a prefix parent")
        counts.add(level.particle_count)
        previous = level.particle_count


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=float)
    if result.ndim != 1 or len(result) == 0 or np.any(result < 0) or not np.all(np.isfinite(result)):
        raise ValueError("weights must be a finite nonnegative vector")
    total = float(result.sum())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    return result / total


def weighted_covariance(parameters: np.ndarray, weights: np.ndarray) -> np.ndarray:
    theta = np.asarray(parameters, dtype=float)
    w = normalize_weights(weights)
    if theta.ndim != 2 or theta.shape[0] != len(w):
        raise ValueError("parameters must be [particle, parameter]")
    centered = theta - np.einsum("n,np->p", w, theta)
    covariance = np.einsum("n,np,nq->pq", w, centered, centered)
    return 0.5 * (covariance + covariance.T)


def psd_square_root(matrix: np.ndarray, relative_floor: float = 1e-12) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    tolerance = relative_floor * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("matrix is not positive semidefinite")
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


def posterior_averaged_gaussian_information(
    means: np.ndarray,
    jacobians: np.ndarray,
    posterior_weights: np.ndarray,
    sensor_std: float,
    gain_variance: float,
) -> np.ndarray:
    """Average per-particle Gaussian Fisher matrices, never Jacobians."""

    mean = np.asarray(means, dtype=float)
    jacobian = np.asarray(jacobians, dtype=float)
    weights = normalize_weights(posterior_weights)
    if mean.shape[0] != len(weights) or jacobian.shape[0] != len(weights):
        raise ValueError("particle axes must agree")
    per_particle = gain_aware_fisher(mean, jacobian, sensor_std, gain_variance)
    averaged = np.einsum("n,nij->ij", weights, per_particle)
    return 0.5 * (averaged + averaged.T)


def query_utility_curvature(
    query_jacobians: np.ndarray,
    posterior_weights: np.ndarray,
    standardized_scale: np.ndarray,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Local Hessian metric of expected standardized query-MSE utility.

    This is intentionally not called query Fisher.  The caller supplies the
    sensitivity of the predictive mean and the exact frozen query scaling.
    Irreducible, parameter-independent future noise contributes a constant to
    both risks and therefore has zero local curvature.
    """

    derivative = np.asarray(query_jacobians, dtype=float)
    weights = normalize_weights(posterior_weights)
    scale = np.asarray(standardized_scale, dtype=float).reshape(-1)
    if derivative.ndim != 3 or derivative.shape[0] != len(weights) or derivative.shape[1] != len(scale):
        raise ValueError("query Jacobians must be [particle, feature, parameter]")
    if np.any(scale <= 0):
        raise ValueError("standardized query scales must be positive")
    if feature_weights is None:
        utility_weight = np.full(len(scale), 1.0 / len(scale)) / scale**2
    else:
        feature = np.asarray(feature_weights, dtype=float).reshape(-1)
        if feature.shape != scale.shape or np.any(feature < 0) or feature.sum() <= 0:
            raise ValueError("invalid feature weights")
        utility_weight = (feature / feature.sum()) / scale**2
    per_particle = np.einsum("nfp,f,nfq->npq", derivative, utility_weight, derivative)
    curvature = np.einsum("n,npq->pq", weights, per_particle)
    return 0.5 * (curvature + curvature.T)


def whitened_operators(
    posterior_covariance: np.ndarray,
    candidate_information: np.ndarray,
    query_curvature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root = psd_square_root(posterior_covariance)
    candidate = root @ np.asarray(candidate_information, dtype=float) @ root
    query = root @ np.asarray(query_curvature, dtype=float) @ root
    return 0.5 * (candidate + candidate.T), 0.5 * (query + query.T)


def modal_geometry(candidate_operator: np.ndarray, query_operator: np.ndarray, value_floor: float = 0.0) -> ModeGeometry:
    """Return the frozen continuous accessibility summary and complete modes."""

    j = np.asarray(candidate_operator, dtype=float)
    c = np.asarray(query_operator, dtype=float)
    if j.shape != c.shape or j.ndim != 2 or j.shape[0] != j.shape[1]:
        raise ValueError("candidate/query operators must be aligned square matrices")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (j + j.T))
    if eigenvalues.min() < -1e-9:
        raise ValueError("candidate operator must be positive semidefinite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    query_weights = np.einsum("pi,pq,qi->i", eigenvectors, c, eigenvectors)
    query_weights = np.maximum(query_weights, 0.0)
    acquisition = eigenvalues / (1.0 + eigenvalues)
    contributions = query_weights * acquisition
    local_value = float(contributions.sum())
    accessibility = float(np.dot(contributions, acquisition) / local_value) if local_value > value_floor else float("nan")
    return ModeGeometry(eigenvalues, query_weights, contributions, local_value, accessibility)


def _weighted_query_risk(query_values: np.ndarray, weights: np.ndarray, scale: np.ndarray) -> float:
    values = np.asarray(query_values, dtype=float)
    w = normalize_weights(weights)
    metric_scale = np.asarray(scale, dtype=float).reshape(-1)
    if values.shape != (len(w), len(metric_scale)):
        raise ValueError("query values must be [particle, feature]")
    prediction = np.einsum("n,nf->f", w, values)
    return float(np.einsum("n,nf->", w, ((values - prediction) / metric_scale) ** 2) / len(metric_scale))


def bayes_value_posterior_predictive(
    theta_weights: np.ndarray,
    query_values: np.ndarray,
    outcome_likelihood: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Path 1: risk before minus posterior-predictive expected risk after E.

    ``outcome_likelihood[o, n]`` is a normalized discrete approximation to
    p(y_o | theta_n, E).  Continuous likelihoods use a frozen quadrature/sample
    construction upstream.
    """

    prior = normalize_weights(theta_weights)
    likelihood = np.asarray(outcome_likelihood, dtype=float)
    if likelihood.ndim != 2 or likelihood.shape[1] != len(prior) or np.any(likelihood < 0):
        raise ValueError("outcome likelihood must be [outcome, particle]")
    column_mass = likelihood.sum(axis=0)
    if not np.allclose(column_mass, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("each particle's outcome likelihood must sum to one")
    before = _weighted_query_risk(query_values, prior, scale)
    outcome_probability = likelihood @ prior
    after = 0.0
    for outcome, probability in enumerate(outcome_probability):
        if probability <= 0:
            continue
        posterior = prior * likelihood[outcome]
        posterior /= posterior.sum()
        after += float(probability) * _weighted_query_risk(query_values, posterior, scale)
    return before - after


def bayes_value_system_conditional(
    theta_weights: np.ndarray,
    query_values: np.ndarray,
    outcome_likelihood: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Path 2: independently aggregate posterior-weighted clairvoyant values."""

    prior = normalize_weights(theta_weights)
    values = np.asarray(query_values, dtype=float)
    likelihood = np.asarray(outcome_likelihood, dtype=float)
    metric_scale = np.asarray(scale, dtype=float).reshape(-1)
    if values.shape != (len(prior), len(metric_scale)) or likelihood.shape[1] != len(prior):
        raise ValueError("particle axes must agree")
    if not np.allclose(likelihood.sum(axis=0), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("each particle's outcome likelihood must sum to one")
    before_prediction = np.sum(prior[:, None] * values, axis=0)
    outcome_probability = np.sum(likelihood * prior[None, :], axis=1)
    posterior_predictions = np.zeros((len(outcome_probability), values.shape[1]))
    for outcome, probability in enumerate(outcome_probability):
        if probability > 0:
            unnormalized = likelihood[outcome] * prior
            posterior_predictions[outcome] = np.sum(unnormalized[:, None] * values, axis=0) / probability
    system_values = np.empty(len(prior), dtype=float)
    for particle in range(len(prior)):
        loss_before = np.mean(((values[particle] - before_prediction) / metric_scale) ** 2)
        loss_after = 0.0
        for outcome in range(likelihood.shape[0]):
            loss_after += likelihood[outcome, particle] * np.mean(
                ((values[particle] - posterior_predictions[outcome]) / metric_scale) ** 2
            )
        system_values[particle] = loss_before - loss_after
    return float(np.sum(prior * system_values))


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("rank vectors must align")
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(float)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def local_fidelity(local_values: np.ndarray, reference_values: np.ndarray, residual_query_risk: float, utility_floor: float) -> FidelityResult:
    local = np.asarray(local_values, dtype=float)
    reference = np.asarray(reference_values, dtype=float)
    if local.shape != reference.shape or local.ndim != 1 or len(local) < 2:
        raise ValueError("candidate values must be aligned vectors")
    denominator = max(float(residual_query_risk), float(utility_floor))
    if denominator <= 0:
        raise ValueError("fidelity denominator must be positive")
    local_sorted = np.sort(local)
    reference_sorted = np.sort(reference)
    return FidelityResult(
        normalized_max_value_error=float(np.max(np.abs(local - reference)) / denominator),
        normalized_range_error=float(abs(np.ptp(local) - np.ptp(reference)) / denominator),
        normalized_margin_error=float(abs((local_sorted[-1] - local_sorted[-2]) - (reference_sorted[-1] - reference_sorted[-2])) / denominator),
        winner_agreement=bool(np.argmax(local) == np.argmax(reference)),
        rank_correlation=rank_correlation(local, reference),
    )


def candidate_is_eligible(local_value: float, residual_query_risk: float, value_floor: float, utility_floor: float) -> bool:
    return bool(local_value > value_floor and residual_query_risk > utility_floor)


def write_freeze_receipt(output: Path, status: str, sources: Iterable[Path], metadata: Mapping[str, object]) -> dict:
    source_hashes = {str(path): sha256_file(path) for path in sources}
    payload = {
        "status": status,
        "device": "cpu",
        "cuda_accessed": False,
        "mps_accessed": False,
        "sealed_accessed": False,
        "prospective_systems_generated": False,
        "source_sha256": source_hashes,
        "metadata": dict(metadata),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def mode_geometry_dict(value: ModeGeometry) -> dict:
    result = asdict(value)
    for key in ("eigenvalues", "query_weights", "mode_contributions"):
        result[key] = result[key].tolist()
    return result


P0_CANONICAL_AXES = ("system", "realization", "history", "candidate", "query")


def _context_parameters(parameters: np.ndarray, s: int, r: int, h: int) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim == 5:
        return values[s, r, h]
    raise ValueError("parameters must be [particle,parameter] or [S,R,H,particle,parameter]")


def _query_scale(scale: np.ndarray, s: int, r: int, h: int, q: int) -> np.ndarray:
    values = np.asarray(scale, dtype=float)
    if values.ndim == 2:
        return values[q]
    if values.ndim == 5:
        return values[s, r, h, q]
    raise ValueError("query_scale must be [Q,feature] or [S,R,H,Q,feature]")


def build_p0_canonical_artifact(
    parameters: np.ndarray,
    posterior_weights: np.ndarray,
    candidate_means: np.ndarray,
    candidate_jacobians: np.ndarray,
    query_jacobians: np.ndarray,
    query_scale: np.ndarray,
    reference_value: np.ndarray,
    residual_query_risk: np.ndarray,
    sensor_std: float,
    gain_variance: float,
    utility_floor: float,
) -> dict[str, np.ndarray]:
    """Build the only canonical P0 tensor consumed by later P2/P3 code.

    Required shapes are explicit and unequal-axis safe:

    * posterior_weights: ``[S,R,H,P]``
    * candidate_means: ``[S,R,H,E,P,F_E]``
    * candidate_jacobians: ``[S,R,H,E,P,F_E,D]``
    * query_jacobians: ``[S,R,H,Q,P,F_Q,D]``
    * reference_value: ``[S,R,H,E,Q]``
    * residual_query_risk: ``[S,R,H,Q]``

    The output keeps full modes.  No row ordering is used to infer semantics.
    """

    weights = np.asarray(posterior_weights, dtype=float)
    means = np.asarray(candidate_means, dtype=float)
    candidate_derivative = np.asarray(candidate_jacobians, dtype=float)
    query_derivative = np.asarray(query_jacobians, dtype=float)
    reference = np.asarray(reference_value, dtype=float)
    u_q = np.asarray(residual_query_risk, dtype=float)
    if weights.ndim != 4:
        raise ValueError("posterior_weights must have axes [S,R,H,P]")
    systems, realizations, histories, particles = weights.shape
    if means.ndim != 6 or means.shape[:3] != weights.shape[:3] or means.shape[4] != particles:
        raise ValueError("candidate_means axes must be [S,R,H,E,P,F_E]")
    candidates = means.shape[3]
    if candidate_derivative.shape[:6] != means.shape:
        raise ValueError("candidate_jacobians must extend candidate_means with parameter axis")
    parameters_count = candidate_derivative.shape[-1]
    if query_derivative.ndim != 7 or query_derivative.shape[:3] != weights.shape[:3] or query_derivative.shape[4] != particles or query_derivative.shape[-1] != parameters_count:
        raise ValueError("query_jacobians axes must be [S,R,H,Q,P,F_Q,D]")
    queries = query_derivative.shape[3]
    expected_cell_shape = (systems, realizations, histories, candidates, queries)
    if reference.shape != expected_cell_shape or u_q.shape != (systems, realizations, histories, queries):
        raise ValueError("reference/u_q named axes do not match canonical shape")

    local = np.empty(expected_cell_shape, dtype=float)
    accessibility = np.empty(expected_cell_shape, dtype=float)
    eigenvalues = np.empty(expected_cell_shape + (parameters_count,), dtype=float)
    query_weights = np.empty_like(eigenvalues)
    contributions = np.empty_like(eigenvalues)
    fidelity_value = np.empty(expected_cell_shape, dtype=float)
    fidelity_range = np.empty(expected_cell_shape, dtype=float)
    fidelity_margin = np.empty(expected_cell_shape, dtype=float)
    fidelity_winner = np.empty(expected_cell_shape, dtype=bool)
    fidelity_rank = np.empty(expected_cell_shape, dtype=float)

    for s in range(systems):
        for r in range(realizations):
            for h in range(histories):
                posterior = normalize_weights(weights[s, r, h])
                theta = _context_parameters(parameters, s, r, h)
                if theta.shape != (particles, parameters_count):
                    raise ValueError("context parameters do not align with particle/parameter axes")
                covariance = weighted_covariance(theta, posterior)
                candidate_information = []
                for e in range(candidates):
                    candidate_information.append(posterior_averaged_gaussian_information(
                        means[s, r, h, e], candidate_derivative[s, r, h, e], posterior,
                        sensor_std, gain_variance,
                    ))
                for q in range(queries):
                    curvature = query_utility_curvature(
                        query_derivative[s, r, h, q], posterior,
                        _query_scale(query_scale, s, r, h, q),
                    )
                    for e in range(candidates):
                        j, c = whitened_operators(covariance, candidate_information[e], curvature)
                        modes = modal_geometry(j, c)
                        local[s, r, h, e, q] = modes.local_value
                        accessibility[s, r, h, e, q] = modes.accessibility
                        eigenvalues[s, r, h, e, q] = modes.eigenvalues
                        query_weights[s, r, h, e, q] = modes.query_weights
                        contributions[s, r, h, e, q] = modes.mode_contributions
                    fidelity = local_fidelity(local[s, r, h, :, q], reference[s, r, h, :, q], u_q[s, r, h, q], utility_floor)
                    fidelity_value[s, r, h, :, q] = fidelity.normalized_max_value_error
                    fidelity_range[s, r, h, :, q] = fidelity.normalized_range_error
                    fidelity_margin[s, r, h, :, q] = fidelity.normalized_margin_error
                    fidelity_winner[s, r, h, :, q] = fidelity.winner_agreement
                    fidelity_rank[s, r, h, :, q] = fidelity.rank_correlation
    return {
        "reference_value": reference,
        "local_value": local,
        "accessibility": accessibility,
        "u_q": np.broadcast_to(u_q[:, :, :, None, :], expected_cell_shape).copy(),
        "fidelity_normalized_max_value_error": fidelity_value,
        "fidelity_normalized_range_error": fidelity_range,
        "fidelity_normalized_margin_error": fidelity_margin,
        "fidelity_winner_agreement": fidelity_winner,
        "fidelity_rank_correlation": fidelity_rank,
        "mode_eigenvalues": eigenvalues,
        "mode_query_weights": query_weights,
        "mode_contributions": contributions,
    }


def write_p0_canonical_artifact(input_path: Path, output_root: Path, config_path: Path) -> dict:
    """Stable file API/CLI: named input NPZ -> canonical tensor/modes/receipt."""

    validate_cpu_only("cpu")
    source = np.load(input_path, allow_pickle=False)
    required = {
        "parameters", "posterior_weights", "candidate_means", "candidate_jacobians",
        "query_jacobians", "query_scale", "reference_value", "residual_query_risk",
        "sensor_std", "gain_variance", "utility_floor",
    }
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"missing P0 input arrays: {sorted(missing)}")
    artifact = build_p0_canonical_artifact(
        source["parameters"], source["posterior_weights"], source["candidate_means"],
        source["candidate_jacobians"], source["query_jacobians"], source["query_scale"],
        source["reference_value"], source["residual_query_risk"],
        float(source["sensor_std"]), float(source["gain_variance"]), float(source["utility_floor"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = output_root / "p0_canonical_srheq.npz"
    np.savez_compressed(tensor_path, **artifact)
    axis_spec = {
        "schema_version": "1.0",
        "canonical_axes": list(P0_CANONICAL_AXES),
        "mode_axis": "parameter_mode",
        "cell_shape": list(artifact["reference_value"].shape),
        "mode_shape": list(artifact["mode_eigenvalues"].shape),
        "fields": sorted(artifact),
    }
    axis_path = output_root / "p0_axis_spec.json"
    axis_path.write_text(json.dumps(axis_spec, indent=2, sort_keys=True) + "\n")
    receipt = write_freeze_receipt(
        output_root / "p0_artifact_receipt.json",
        "P0_ARTIFACT_COMPLETE_UNVERIFIED",
        (input_path, config_path, tensor_path, axis_path),
        {"canonical_axes": list(P0_CANONICAL_AXES), "artifact": tensor_path.name},
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CPU-only Coupled P0 canonical reference/local-geometry artifact")
    parser.add_argument("input", type=Path, help="named-axis P0 input NPZ")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/coupled_theory_validation_p0_v1.json"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    validate_cpu_only(args.device)
    result = write_p0_canonical_artifact(args.input, args.output_root, args.config)
    print(json.dumps({"status": result["status"], "device": result["device"]}, sort_keys=True))


if __name__ == "__main__":
    main()
