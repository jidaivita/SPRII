"""Metadata-only, fail-closed RH20T hard-negative construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np


def field(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    if value is None:
        raise KeyError(path)
    return value


@dataclass(frozen=True)
class MetadataPredicate:
    name: str
    equal_fields: tuple[str, ...]
    different_fields: tuple[str, ...]
    query_id_field: str
    episode_id_field: str
    task_field: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetadataPredicate":
        allowed = {
            "name", "equal_fields", "different_fields", "query_id_field",
            "episode_id_field", "task_field",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown predicate fields: {sorted(unknown)}")
        predicate = cls(
            name=str(payload["name"]),
            equal_fields=tuple(payload["equal_fields"]),
            different_fields=tuple(payload["different_fields"]),
            query_id_field=str(payload["query_id_field"]),
            episode_id_field=str(payload["episode_id_field"]),
            task_field=str(payload["task_field"]),
        )
        predicate.validate()
        return predicate

    def validate(self) -> None:
        if self.name not in {"D_M", "D_HP", "D_HS", "D_R"}:
            raise ValueError(f"unregistered donor category {self.name}")
        if not self.equal_fields and not self.different_fields:
            raise ValueError("predicate must constrain at least one metadata field")
        overlap = set(self.equal_fields) & set(self.different_fields)
        if overlap:
            raise ValueError(f"fields cannot be both equal and different: {sorted(overlap)}")
        required = (self.query_id_field, self.episode_id_field, self.task_field)
        if any(not item or "FROZEN_FIELD" in item for item in required):
            raise ValueError("predicate contains an unresolved identity field")
        if any(not item or "FROZEN_FIELD" in item for item in self.equal_fields + self.different_fields):
            raise ValueError("predicate contains an unresolved comparison field")

    def eligible(self, query: Mapping[str, Any], donor: Mapping[str, Any]) -> bool:
        """Return False on missing metadata; never infer or impute eligibility."""
        try:
            if field(query, self.episode_id_field) == field(donor, self.episode_id_field):
                return False
            return all(field(query, key) == field(donor, key) for key in self.equal_fields) and all(
                field(query, key) != field(donor, key) for key in self.different_fields
            )
        except KeyError:
            return False


def legal_pools(
    records: Iterable[Mapping[str, Any]], predicate: MetadataPredicate
) -> dict[str, list[str]]:
    items = list(records)
    pools: dict[str, list[str]] = {}
    for query in items:
        query_id = str(field(query, predicate.query_id_field))
        donors = [
            str(field(donor, predicate.episode_id_field))
            for donor in items
            if predicate.eligible(query, donor)
        ]
        pools[query_id] = sorted(set(donors))
    return pools


def coverage_report(
    records: Iterable[Mapping[str, Any]], predicate: MetadataPredicate
) -> dict[str, Any]:
    items = list(records)
    pools = legal_pools(items, predicate)
    query_to_task = {
        str(field(item, predicate.query_id_field)): str(field(item, predicate.task_field))
        for item in items
    }
    all_tasks = sorted(set(query_to_task.values()))
    eligible_queries = sorted(query for query, donors in pools.items() if donors)
    covered_tasks = sorted(set(query_to_task[query] for query in eligible_queries))
    sizes = np.asarray([len(pools[query]) for query in eligible_queries], dtype=np.float64)
    return {
        "category": predicate.name,
        "query_count": len(items),
        "eligible_query_count": len(eligible_queries),
        "eligible_query_fraction": len(eligible_queries) / len(items) if items else 0.0,
        "task_count": len(all_tasks),
        "covered_task_count": len(covered_tasks),
        "covered_task_fraction": len(covered_tasks) / len(all_tasks) if all_tasks else 0.0,
        "eligible_queries": eligible_queries,
        "pool_size_median": float(np.median(sizes)) if sizes.size else 0.0,
        "pool_size_mean": float(np.mean(sizes)) if sizes.size else 0.0,
        "pool_size_std": float(np.std(sizes)) if sizes.size else 0.0,
        "pool_size_cv": (
            float(np.std(sizes) / np.mean(sizes)) if sizes.size and np.mean(sizes) > 0 else None
        ),
        "legal_pools": pools,
    }


def common_query_intersection(*reports: Mapping[str, Any]) -> list[str]:
    if not reports:
        raise ValueError("at least one donor-category report is required")
    query_sets = [set(report["eligible_queries"]) for report in reports]
    return sorted(set.intersection(*query_sets))


def deterministic_donor(
    query_id: str, category: str, donors: Iterable[str], evaluator_seed: int
) -> str:
    candidates = sorted(set(donors))
    if not candidates:
        raise ValueError(f"query {query_id} has no legal donor for {category}")
    def score(donor: str) -> bytes:
        payload = f"{evaluator_seed}|{category}|{query_id}|{donor}".encode()
        return hashlib.sha256(payload).digest()
    return min(candidates, key=lambda donor: (score(donor), donor))
