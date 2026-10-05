from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class PosteriorResult:
    weights: np.ndarray
    ess: float


def lognormal_quadrature(mean: float, cv: float, points: int) -> Tuple[np.ndarray, np.ndarray]:
    if mean <= 0 or cv < 0 or points < 1:
        raise ValueError("invalid lognormal quadrature specification")
    variance = np.log1p(cv**2)
    sigma = np.sqrt(variance)
    mu = np.log(mean) - 0.5 * variance
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    gains = np.exp(mu + np.sqrt(2.0) * sigma * nodes)
    weights = weights / np.sqrt(np.pi)
    return gains, weights


def posterior_from_observation(
    observation: np.ndarray,
    unit_gain_particle_means: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
    prior_weights: np.ndarray | None = None,
) -> PosteriorResult:
    """Importance posterior with fixed log-gain quadrature."""

    y = np.asarray(observation, dtype=float)
    means = np.asarray(unit_gain_particle_means, dtype=float)
    nodes = np.asarray(gain_nodes, dtype=float)
    quadrature = np.asarray(gain_weights, dtype=float)
    if means.ndim != 2 or y.shape != (means.shape[1],):
        raise ValueError("observation/particle response shape mismatch")
    if sensor_std <= 0 or nodes.ndim != 1 or quadrature.shape != nodes.shape:
        raise ValueError("invalid likelihood scales")
    prior = np.full(len(means), 1.0 / len(means)) if prior_weights is None else np.asarray(prior_weights, dtype=float)
    prior = prior / prior.sum()
    residual = y[None, None, :] - nodes[None, :, None] * means[:, None, :]
    log_component = -0.5 * np.sum((residual / sensor_std) ** 2, axis=2)
    log_component += np.log(quadrature)[None, :]
    log_likelihood = logsumexp(log_component, axis=1)
    log_posterior = np.log(prior) + log_likelihood
    log_posterior -= logsumexp(log_posterior)
    weights = np.exp(log_posterior)
    return PosteriorResult(weights=weights, ess=float(1.0 / np.sum(weights**2)))


def posterior_batch_from_observations(
    observations: np.ndarray,
    unit_gain_particle_means: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized form returning weights and ESS for many observations."""

    y = np.asarray(observations, dtype=float)
    means = np.asarray(unit_gain_particle_means, dtype=float)
    nodes = np.asarray(gain_nodes, dtype=float)
    quadrature = np.asarray(gain_weights, dtype=float)
    if y.ndim != 2 or means.ndim != 2 or y.shape[1] != means.shape[1]:
        raise ValueError("observations and particle means must align on feature")
    residual = y[:, None, None, :] - nodes[None, None, :, None] * means[None, :, None, :]
    log_component = -0.5 * np.sum((residual / sensor_std) ** 2, axis=3)
    log_component += np.log(quadrature)[None, None, :]
    log_likelihood = logsumexp(log_component, axis=2)
    log_likelihood -= logsumexp(log_likelihood, axis=1, keepdims=True)
    weights = np.exp(log_likelihood)
    ess = 1.0 / np.sum(weights**2, axis=1)
    return weights, ess


def posterior_batch_from_observations_quadratic(
    observations: np.ndarray,
    unit_gain_particle_means: np.ndarray,
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact quadratic-expansion batch likelihood with bounded feature memory.

    This is algebraically identical to :func:`posterior_batch_from_observations`
    but never materializes ``[outcome, particle, gain_node, feature]``.  It uses

    ``||y - g m||^2 = ||y||^2 - 2 g <y,m> + g^2 ||m||^2``

    and materializes only ``[outcome, particle, gain_node]``.  The original
    function remains the parity reference until the formal caller is explicitly
    switched.
    """

    y = np.asarray(observations, dtype=float)
    means = np.asarray(unit_gain_particle_means, dtype=float)
    nodes = np.asarray(gain_nodes, dtype=float)
    quadrature = np.asarray(gain_weights, dtype=float)
    if y.ndim != 2 or means.ndim != 2 or y.shape[1] != means.shape[1]:
        raise ValueError("observations and particle means must align on feature")
    if sensor_std <= 0 or nodes.ndim != 1 or quadrature.shape != nodes.shape:
        raise ValueError("invalid likelihood scales")
    if np.any(quadrature <= 0) or not all(np.all(np.isfinite(value)) for value in (y, means, nodes, quadrature)):
        raise ValueError("likelihood inputs must be finite with positive quadrature weights")

    observation_norm = np.einsum("of,of->o", y, y)
    observation_particle_dot = y @ means.T
    particle_norm = np.einsum("pf,pf->p", means, means)
    squared_residual = (
        observation_norm[:, None, None]
        - 2.0 * observation_particle_dot[:, :, None] * nodes[None, None, :]
        + particle_norm[None, :, None] * (nodes * nodes)[None, None, :]
    )
    log_component = -0.5 * squared_residual / (sensor_std * sensor_std)
    log_component += np.log(quadrature)[None, None, :]
    log_likelihood = logsumexp(log_component, axis=2)
    log_likelihood -= logsumexp(log_likelihood, axis=1, keepdims=True)
    weights = np.exp(log_likelihood)
    ess = 1.0 / np.sum(weights**2, axis=1)
    return weights, ess


def combine_independent_posteriors(
    observations: tuple[np.ndarray, ...],
    particle_means: tuple[np.ndarray, ...],
    sensor_std: float,
    gain_nodes: np.ndarray,
    gain_weights: np.ndarray,
) -> PosteriorResult:
    if len(observations) != len(particle_means) or not observations:
        raise ValueError("history segments must be nonempty and aligned")
    log_weights = np.zeros(len(particle_means[0]), dtype=float)
    for observation, means in zip(observations, particle_means):
        segment = posterior_from_observation(
            observation, means, sensor_std, gain_nodes, gain_weights
        )
        log_weights += np.log(np.clip(segment.weights, 1e-300, None))
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    return PosteriorResult(weights=weights, ess=float(1.0 / np.sum(weights**2)))


def predictive_mean(unit_gain_query_particle_means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    query = np.asarray(unit_gain_query_particle_means, dtype=float)
    posterior = np.asarray(weights, dtype=float)
    if query.ndim != 3 or posterior.shape != (query.shape[0],):
        raise ValueError("query means must be [particle, query, feature]")
    return np.einsum("n,nqf->qf", posterior / posterior.sum(), query)


def prior_predictive_scale(
    unit_gain_query_particle_means: np.ndarray,
    gain_second_moment: float,
    sensor_std: float,
) -> np.ndarray:
    query = np.asarray(unit_gain_query_particle_means, dtype=float)
    mean = query.mean(axis=0)
    second = gain_second_moment * np.mean(query**2, axis=0) + sensor_std**2
    return np.sqrt(np.maximum(second - mean**2, 1e-18))


def standardized_empirical_risk(truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if truth.shape != prediction.shape or scale.shape != truth.shape:
        raise ValueError("risk arrays must have identical [query, feature] shapes")
    return float(np.mean(((truth - prediction) / scale) ** 2))


def oracle_floor_fraction(r0: float, re: float, r_oracle: float) -> Tuple[float, float]:
    denominator = r0 - r_oracle
    if denominator <= 0:
        raise ValueError("oracle-reducible risk must be positive")
    return r0 - re, (r0 - re) / denominator
