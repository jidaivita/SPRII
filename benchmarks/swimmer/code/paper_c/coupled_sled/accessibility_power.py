"""Variance-only calibration for the prospective accessibility contrast."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import erf, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PowerChoice:
    systems: int
    realizations: int
    projected_reliability: float
    projected_power: float
    standard_error: float


def practical_mde(pipeline_sigma: float, median_crossfit_learner_range: float) -> float:
    """Frozen scale rule; never accepts an accessibility-oriented mean effect."""

    if pipeline_sigma < 0 or median_crossfit_learner_range < 0:
        raise ValueError("MDE inputs must be nonnegative")
    return float(max(5.0 * pipeline_sigma, 0.10 * median_crossfit_learner_range))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("reliability vectors must align")
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def split_half_reliability(contrasts: np.ndarray, realizations: int) -> float:
    """Reliability of system-level pair contrasts from common-random-number rows.

    Input is ``[system, realization, matched_pair]``.  Each half averages its
    nuisance realizations and all eligible pairs within a physical system.
    """

    values = np.asarray(contrasts, dtype=float)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("contrasts must be finite [system, realization, pair]")
    if realizations < 4 or realizations > values.shape[1] or realizations % 2:
        raise ValueError("realizations must be an even value between 4 and available R")
    selected = values[:, :realizations]
    midpoint = realizations // 2
    left = selected[:, :midpoint].mean(axis=(1, 2))
    right = selected[:, midpoint:].mean(axis=(1, 2))
    return _pearson(left, right)


def system_contrast_scale(contrasts: np.ndarray, realizations: int) -> float:
    values = np.asarray(contrasts, dtype=float)
    if values.ndim != 3 or realizations < 1 or realizations > values.shape[1]:
        raise ValueError("invalid contrast tensor or realization count")
    centered = values[:, :realizations] - values[:, :realizations].mean(axis=0, keepdims=True)
    system_means = centered.mean(axis=(1, 2))
    return float(np.std(system_means, ddof=1))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def projected_one_sided_power(effect: float, system_std: float, systems: int, alpha: float = 0.025) -> tuple[float, float]:
    """Normal cluster-mean approximation used only for resource calibration."""

    if effect < 0 or system_std < 0 or systems < 2 or alpha != 0.025:
        raise ValueError("invalid calibration arguments")
    standard_error = system_std / sqrt(systems)
    if standard_error == 0:
        return (1.0 if effect > 0 else 0.0), 0.0
    critical = 1.959963984540054
    power = _normal_cdf(effect / standard_error - critical)
    return float(power), float(standard_error)


def choose_resource_grid(
    contrasts: np.ndarray,
    mde: float,
    system_grid: Iterable[int] = (128, 256, 512),
    realization_grid: Iterable[int] = (4, 8, 16, 32),
    minimum_power: float = 0.90,
    minimum_reliability: float = 0.40,
) -> tuple[PowerChoice | None, list[dict]]:
    """Choose the cheapest predeclared CPU evaluation satisfying both gates."""

    if mde <= 0 or not (0 < minimum_power < 1) or not (-1 <= minimum_reliability <= 1):
        raise ValueError("invalid frozen calibration thresholds")
    values = np.asarray(contrasts, dtype=float)
    rows = []
    candidates = []
    for realizations in realization_grid:
        if realizations > values.shape[1]:
            continue
        reliability = split_half_reliability(values, realizations)
        system_std = system_contrast_scale(values, realizations)
        for systems in system_grid:
            power, se = projected_one_sided_power(mde, system_std, int(systems))
            row = asdict(PowerChoice(int(systems), int(realizations), reliability, power, se))
            row["passes"] = bool(
                np.isfinite(reliability)
                and reliability >= minimum_reliability
                and power >= minimum_power
            )
            rows.append(row)
            if row["passes"]:
                candidates.append((systems * realizations, systems, realizations, PowerChoice(
                    int(systems), int(realizations), reliability, power, se
                )))
    if not candidates:
        return None, rows
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][-1], rows
