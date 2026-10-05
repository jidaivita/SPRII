"""Task-paired A2 specificity difference-in-differences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def task_condition_errors(
    rows: Iterable[Mapping[str, Any]], *, required_seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[tuple[str, str, str], float]:
    """Aggregate queries within seed/task, then average seeds within condition."""
    query_values: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["condition"]), int(row["seed"]), str(row["task_id"]),
            str(row["donor_category"]),
        )
        query_values[key].append(float(row["error"]))
    per_seed = {key: float(np.mean(values)) for key, values in query_values.items()}
    grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for (condition, seed, task, donor), value in per_seed.items():
        grouped[(condition, task, donor)][seed] = value
    result = {}
    expected = set(required_seeds)
    for key, values in grouped.items():
        if set(values) != expected:
            raise ValueError(f"{key} has seeds {sorted(values)}, expected {sorted(expected)}")
        result[key] = float(np.mean([values[seed] for seed in required_seeds]))
    return result


def paired_specificity(
    rows: Iterable[Mapping[str, Any]], *, matched: str = "D_M", hard: str = "D_HP",
    b3: str = "B3", mono: str = "Mono-QD", bootstrap_draws: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    values = task_condition_errors(rows)
    task_sets = []
    for condition in (b3, mono):
        for donor in (matched, hard):
            task_sets.append({task for cond, task, category in values if cond == condition and category == donor})
    if not task_sets or any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("B3/Mono and matched/hard task sets must be identical")
    tasks = sorted(task_sets[0])
    if not tasks:
        raise ValueError("no common held-out tasks")
    b3_gap = np.asarray([values[(b3, task, hard)] - values[(b3, task, matched)] for task in tasks])
    mono_gap = np.asarray([values[(mono, task, hard)] - values[(mono, task, matched)] for task in tasks])
    did = b3_gap - mono_gap
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(tasks), size=(bootstrap_draws, len(tasks)))
    def interval(samples: np.ndarray) -> list[float]:
        means = samples[indices].mean(axis=1)
        return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "task_count": len(tasks),
        "tasks": tasks,
        "s_b3": float(b3_gap.mean()),
        "s_b3_task_paired_bootstrap_95ci": interval(b3_gap),
        "s_mono": float(mono_gap.mean()),
        "s_mono_task_paired_bootstrap_95ci": interval(mono_gap),
        "d_spec": float(did.mean()),
        "d_spec_task_paired_bootstrap_95ci": interval(did),
        "per_task": {
            task: {"s_b3": float(b3_gap[i]), "s_mono": float(mono_gap[i]), "d_spec": float(did[i])}
            for i, task in enumerate(tasks)
        },
        "bootstrap_unit": "held_out_task",
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }
