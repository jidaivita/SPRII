"""Dependency-free metric primitives and an explicit metric registry."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any, Callable


MetricFunction = Callable[[Any, Any], float]


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metric inputs must be finite")
        return [number]
    if isinstance(value, Mapping):
        result: list[float] = []
        for key in sorted(value):
            result.extend(_numbers(value[key]))
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(_numbers(item))
        return result
    raise TypeError(f"unsupported metric payload: {type(value).__name__}")


def mean_squared_error(prediction: Any, target: Any) -> float:
    predicted = _numbers(prediction)
    expected = _numbers(target)
    if not predicted or len(predicted) != len(expected):
        raise ValueError("prediction and target must have equal non-zero sizes")
    return sum((left - right) ** 2 for left, right in zip(predicted, expected)) / len(expected)


def mean_absolute_error(prediction: Any, target: Any) -> float:
    predicted = _numbers(prediction)
    expected = _numbers(target)
    if not predicted or len(predicted) != len(expected):
        raise ValueError("prediction and target must have equal non-zero sizes")
    return sum(abs(left - right) for left, right in zip(predicted, expected)) / len(expected)


def cartpole_normalized_mse(prediction: Any, target: Any) -> float:
    """Trajectory MSE normalized by frozen train-bank state standard deviations."""

    predicted = _numbers(prediction)
    expected = _numbers(target)
    if not predicted or len(predicted) != len(expected) or len(expected) % 4:
        raise ValueError("CartPole prediction and target must align on four-state rows")
    # Frozen from 512 train systems (seeds 11001..11008), protocol 1.0.
    scales = (0.04475246, 0.18130060, 0.01981396, 0.24247076)
    return sum(
        ((left - right) / scales[index % 4]) ** 2
        for index, (left, right) in enumerate(zip(predicted, expected))
    ) / len(expected)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[cursor]], rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        average = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal vectors with at least two values")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(centered_left, centered_right))
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def representation_distance_correlation(prediction: Any, target: Any) -> float:
    """Spearman correlation between pairwise representation and factor distances."""
    representations = [_numbers(item) for item in prediction]
    factors = [_numbers(item) for item in target]
    if len(representations) != len(factors) or len(representations) < 3:
        raise ValueError("representation metric requires at least three aligned systems")
    if len({len(item) for item in representations}) != 1 or len({len(item) for item in factors}) != 1:
        raise ValueError("representation and factor dimensions must be stable")
    representation_dimensions = len(representations[0])
    representation_means = [
        sum(row[index] for row in representations) / len(representations)
        for index in range(representation_dimensions)
    ]
    representation_scales = []
    for index, mean in enumerate(representation_means):
        variance = sum(
            (row[index] - mean) ** 2 for row in representations
        ) / len(representations)
        representation_scales.append(math.sqrt(variance) or 1.0)
    standardized_representations = [
        [
            (row[index] - representation_means[index])
            / representation_scales[index]
            for index in range(representation_dimensions)
        ]
        for row in representations
    ]
    factor_dimensions = len(factors[0])
    means = [sum(row[index] for row in factors) / len(factors) for index in range(factor_dimensions)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in factors) / len(factors)
        scales.append(math.sqrt(variance) or 1.0)
    representation_distances: list[float] = []
    factor_distances: list[float] = []
    for left in range(len(factors)):
        for right in range(left + 1, len(factors)):
            representation_distances.append(
                math.sqrt(
                    sum(
                        (a - b) ** 2
                        for a, b in zip(
                            standardized_representations[left],
                            standardized_representations[right],
                        )
                    )
                )
            )
            factor_distances.append(
                math.sqrt(
                    sum(
                        ((factors[left][index] - factors[right][index]) / scales[index]) ** 2
                        for index in range(factor_dimensions)
                    )
                )
            )
    return _correlation(_rank(representation_distances), _rank(factor_distances))


def cross_trajectory_retrieval(prediction: Any, target: Any) -> float:
    """Nearest-neighbor retrieval accuracy across independently generated views."""
    representations = [_numbers(item) for item in prediction]
    labels = list(target)
    if len(representations) != len(labels) or len(representations) < 4:
        raise ValueError("retrieval requires at least four aligned representations")
    if len({len(item) for item in representations}) != 1:
        raise ValueError("representation dimension must be stable")
    label_counts = {label: labels.count(label) for label in set(labels)}
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("each retrieval group requires at least two independent views")
    expected_correct = 0.0
    for index, representation in enumerate(representations):
        candidates = []
        for other, candidate in enumerate(representations):
            if other == index:
                continue
            distance = sum((left - right) ** 2 for left, right in zip(representation, candidate))
            candidates.append((distance, other))
        minimum = min(distance for distance, _other in candidates)
        nearest = [
            other
            for distance, other in candidates
            if math.isclose(distance, minimum, rel_tol=1e-12, abs_tol=1e-12)
        ]
        expected_correct += sum(labels[other] == labels[index] for other in nearest) / len(
            nearest
        )
    return expected_correct / len(labels)


class MetricRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, MetricFunction] = {}

    def register(self, name: str, function: MetricFunction) -> None:
        if not name or name in self._functions:
            raise ValueError(f"invalid or duplicate metric: {name}")
        self._functions[name] = function

    def evaluate(self, name: str, prediction: Any, target: Any) -> float:
        try:
            function = self._functions[name]
        except KeyError as error:
            raise KeyError(f"unregistered metric: {name}") from error
        value = float(function(prediction, target))
        if not math.isfinite(value):
            raise ValueError(f"metric {name} produced a non-finite value")
        return value

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions))


def default_metrics() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register("mse", mean_squared_error)
    registry.register("mae", mean_absolute_error)
    registry.register("cartpole_normalized_mse", cartpole_normalized_mse)
    registry.register("representation_distance_correlation", representation_distance_correlation)
    registry.register("cross_trajectory_retrieval", cross_trajectory_retrieval)
    return registry
