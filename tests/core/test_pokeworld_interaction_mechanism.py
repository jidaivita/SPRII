from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "core" / "scripts" / "prepare_pokeworld_interaction_mechanism.py"
SPEC = importlib.util.spec_from_file_location("prepare_interaction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collision_free_nested_permutations() -> None:
    permutations, _ = MODULE.collision_free_permutations(64, 8, 123)
    assert permutations.shape == (8, 64)
    for column in permutations:
        assert np.array_equal(np.sort(column), np.arange(64))
    for row in permutations.T:
        assert len(set(row.tolist())) == 8
    assert np.array_equal(permutations[:2], permutations[:4][:2])
    assert np.array_equal(permutations[:4], permutations[:8][:4])


def test_pseudo_payload_reuses_each_column_once_and_invalidates_labels() -> None:
    systems, rollouts = 16, 8
    permutations, _ = MODULE.collision_free_permutations(systems, rollouts, 456)
    source = {
        "states": np.arange(systems * rollouts * 3).reshape(systems, rollouts, 3),
        "actions": np.arange(systems * rollouts * 2).reshape(systems, rollouts, 2),
        "touch": np.arange(systems * rollouts).reshape(systems, rollouts, 1),
        "contact": np.zeros((systems, rollouts, 1), dtype=bool),
        "mass": np.ones(systems), "gamma": np.ones(systems),
        "stiffness": np.ones(systems), "system_ids": np.arange(systems),
    }
    payload = MODULE.pseudo_payload(source, permutations, 4)
    assert np.array_equal(payload["source_system_ids_by_rollout"], permutations[:4].T)
    for column in range(4):
        expected = source["states"][permutations[column], column]
        assert np.array_equal(payload["states"][:, column], expected)
    assert all(np.isnan(payload[key]).all() for key in ("mass", "gamma", "stiffness"))
