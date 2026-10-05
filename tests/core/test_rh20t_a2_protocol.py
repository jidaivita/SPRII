from __future__ import annotations

import pytest

from persistent_jepa.rh20t_did import paired_specificity
from persistent_jepa.rh20t_hard_negative import (
    MetadataPredicate,
    common_query_intersection,
    coverage_report,
    deterministic_donor,
)


def records() -> list[dict]:
    return [
        {"query_id": "q1", "episode_id": "e1", "task": "t1", "robot": "r1", "view": "v1"},
        {"query_id": "q2", "episode_id": "e2", "task": "t1", "robot": "r2", "view": "v2"},
        {"query_id": "q3", "episode_id": "e3", "task": "t2", "robot": "r1", "view": "v1"},
        {"query_id": "q4", "episode_id": "e4", "task": "t2", "robot": "r2", "view": "v2"},
    ]


def predicate(name: str, equal: list[str], different: list[str]) -> MetadataPredicate:
    return MetadataPredicate.from_dict({
        "name": name,
        "equal_fields": equal,
        "different_fields": different,
        "query_id_field": "query_id",
        "episode_id_field": "episode_id",
        "task_field": "task",
    })


def test_metadata_predicates_fail_closed_and_build_common_queries() -> None:
    th = coverage_report(records(), predicate("D_HP", ["task"], ["robot"]))
    eh = coverage_report(records(), predicate("D_HS", ["robot", "view"], ["task"]))
    assert th["covered_task_fraction"] == 1.0
    assert eh["covered_task_fraction"] == 1.0
    assert common_query_intersection(th, eh) == ["q1", "q2", "q3", "q4"]
    damaged = records()
    damaged[1] = {key: value for key, value in damaged[1].items() if key != "robot"}
    damaged_report = coverage_report(damaged, predicate("D_HP", ["task"], ["robot"]))
    assert "e2" not in damaged_report["legal_pools"]["q1"]


def test_unresolved_or_ambiguous_predicate_is_rejected() -> None:
    with pytest.raises(ValueError):
        predicate("D_HP", ["FROZEN_FIELD.robot"], ["task"])
    with pytest.raises(ValueError):
        predicate("D_HP", ["task"], ["task"])


def test_donor_assignment_is_deterministic() -> None:
    donors = ["e9", "e2", "e7"]
    selected = deterministic_donor("q1", "D_HP", donors, 17)
    assert selected == deterministic_donor("q1", "D_HP", reversed(donors), 17)


def test_task_paired_did_aggregates_queries_then_seeds_then_tasks() -> None:
    rows = []
    errors = {
        "B3": {"D_M": 1.0, "D_HP": 1.6},
        "Mono-QD": {"D_M": 1.1, "D_HP": 1.3},
    }
    for condition, donor_values in errors.items():
        for seed in (0, 1, 2):
            for task_index, task in enumerate(("t1", "t2", "t3")):
                for donor, base in donor_values.items():
                    for query_offset in (0.0, 0.2):
                        rows.append({
                            "condition": condition,
                            "seed": seed,
                            "task_id": task,
                            "donor_category": donor,
                            "error": base + task_index * 0.1 + seed * 0.01 + query_offset,
                        })
    result = paired_specificity(rows, bootstrap_draws=200, bootstrap_seed=9)
    assert result["s_b3"] == pytest.approx(0.6)
    assert result["s_mono"] == pytest.approx(0.2)
    assert result["d_spec"] == pytest.approx(0.4)
    assert result["bootstrap_unit"] == "held_out_task"


def test_task_paired_did_rejects_missing_seed() -> None:
    rows = [
        {"condition": condition, "seed": seed, "task_id": "t1", "donor_category": donor, "error": 1.0}
        for condition in ("B3", "Mono-QD")
        for seed in (0, 1)
        for donor in ("D_M", "D_HP")
    ]
    with pytest.raises(ValueError):
        paired_specificity(rows, bootstrap_draws=10)
