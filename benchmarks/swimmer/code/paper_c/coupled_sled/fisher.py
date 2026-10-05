from typing import Tuple

import numpy as np

from .batch import batch_position_landmarks, simulate_batch
from .waveforms import Probe


def response_jacobian(
    parameters: np.ndarray,
    probe: Probe,
    landmarks: int,
    semantics: str,
    parameter_scale: np.ndarray,
    step_in_standard_units: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-gain response and central-difference derivative with respect to z."""

    theta = np.asarray(parameters, dtype=float)
    scale = np.asarray(parameter_scale, dtype=float)
    states = simulate_batch(theta, probe)
    sides = [probe.side] * len(theta)
    mean = batch_position_landmarks(states, landmarks, semantics, sides)
    jacobian = np.empty((len(theta), mean.shape[1], theta.shape[1]), dtype=float)
    for parameter in range(theta.shape[1]):
        delta = step_in_standard_units * scale[parameter]
        plus = theta.copy()
        minus = theta.copy()
        plus[:, parameter] += delta
        minus[:, parameter] -= delta
        if np.any(minus[:, parameter] <= 0):
            raise ValueError("finite-difference step crossed positive parameter boundary")
        plus_mean = batch_position_landmarks(simulate_batch(plus, probe), landmarks, semantics, sides)
        minus_mean = batch_position_landmarks(simulate_batch(minus, probe), landmarks, semantics, sides)
        jacobian[:, :, parameter] = (plus_mean - minus_mean) / (2.0 * step_in_standard_units)
    return mean, jacobian


def gain_aware_fisher(mean: np.ndarray, jacobian: np.ndarray, sensor_std: float, gain_variance: float) -> np.ndarray:
    """Gaussian Fisher for sigma^2 I + v_g mu mu^T, vectorized by context."""

    mu = np.asarray(mean, dtype=float)
    derivative = np.asarray(jacobian, dtype=float)
    if mu.ndim != 2 or derivative.shape[:2] != mu.shape:
        raise ValueError("mean/Jacobian shapes must be [context, feature, parameter]")
    sigma2 = sensor_std**2
    norm2 = np.sum(mu**2, axis=1)
    projection = np.einsum("nf,nfp->np", mu, derivative)
    denominator = 1.0 + gain_variance * norm2 / sigma2
    inverse_times_d = derivative / sigma2
    inverse_times_d -= (
        gain_variance
        * mu[:, :, None]
        * projection[:, None, :]
        / (sigma2**2 * denominator[:, None, None])
    )
    mean_fisher = np.einsum("nfa,nfb->nab", derivative, inverse_times_d)
    inverse_mu = mu / (sigma2 + gain_variance * norm2)[:, None]
    h = np.einsum("nfa,nf->na", derivative, inverse_mu)
    c = np.einsum("nf,nf->n", mu, inverse_mu)
    covariance_fisher = gain_variance**2 * (
        c[:, None, None] * mean_fisher + h[:, :, None] * h[:, None, :]
    )
    return mean_fisher + covariance_fisher
