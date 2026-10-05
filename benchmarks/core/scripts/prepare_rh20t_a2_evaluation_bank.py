#!/usr/bin/env python3
"""Freeze a window-level A2 validation bank without accessing model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from persistent_jepa.rh20t_hard_negative import MetadataPredicate, deterministic_donor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_id(row: dict[str, Any]) -> str:
    return f'{row["query_episode_id"]}@q{int(row["query_anchor"])}@d{int(row["donor_anchor"])}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-bank", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    parent = json.loads(args.parent_bank.read_text())
    metadata_payload = json.loads(args.metadata.read_text())
    predicate_payload = json.loads(args.predicates.read_text())
    if parent.get("split") != "validation":
        raise RuntimeError("A2 development bank must be validation-only")
    if metadata_payload.get("split") != "validation":
        raise RuntimeError("A2 metadata must be validation-only")
    if predicate_payload.get("model_output_read") is not False:
        raise RuntimeError("predicate freeze does not certify output blindness")

    records = metadata_payload["records"]
    by_episode = {record["episode_id"]: record for record in records}
    predicates = {
        item["name"]: MetadataPredicate.from_dict(item)
        for item in predicate_payload["predicates"]
    }
    required = ("D_M", "D_HP", "D_HS", "D_R")
    if tuple(predicates) != required:
        raise RuntimeError(f"predicate order must be {required}, got {tuple(predicates)}")

    frozen_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = defaultdict(int)
    pool_sizes: dict[str, list[int]] = defaultdict(list)
    all_rows = 0
    for task_id in sorted(parent["tasks"]):
        for row in parent["tasks"][task_id]:
            all_rows += 1
            q_episode = row["query_episode_id"]
            if q_episode not in by_episode:
                rejected["query_missing_metadata"] += 1
                continue
            query = by_episode[q_episode]
            donor_anchor = int(row["donor_anchor"])
            selected: dict[str, str] = {}
            missing = None
            for category in required:
                predicate = predicates[category]
                candidates = [
                    donor["episode_id"]
                    for donor in records
                    if int(donor["frames"]) > donor_anchor
                    and predicate.eligible(query, donor)
                ]
                pool_sizes[category].append(len(candidates))
                if not candidates:
                    missing = category
                    break
                selected[category] = deterministic_donor(
                    query_id(row), category, candidates, int(predicate_payload["evaluator_seed"])
                )
            if missing is not None:
                rejected[f"no_legal_{missing}"] += 1
                continue
            frozen_rows[task_id].append({
                "query_id": query_id(row),
                "query_episode_id": q_episode,
                "query_anchor": int(row["query_anchor"]),
                "donor_anchor": donor_anchor,
                "donor_episode_ids": selected,
            })

    task_counts = {task: len(rows) for task, rows in frozen_rows.items() if rows}
    payload = {
        "schema_version": "1.1b",
        "status": "FROZEN_A2_VALIDATION_BANK_NO_MODEL_OUTPUT",
        "split": "validation",
        "test_read": False,
        "model_output_read": False,
        "query_unit": "fixed_validation_window",
        "eligibility": "predicate_and_donor_frames_strictly_greater_than_donor_anchor",
        "evaluator_seed": int(predicate_payload["evaluator_seed"]),
        "categories": list(required),
        "tasks": {task: frozen_rows[task] for task in sorted(task_counts)},
        "counts": {
            "parent_rows": all_rows,
            "qstar_rows": sum(task_counts.values()),
            "parent_tasks": len(parent["tasks"]),
            "qstar_tasks": len(task_counts),
            "per_task": task_counts,
            "rejected": dict(sorted(rejected.items())),
            "pool_sizes": {
                category: {
                    "minimum": min(values) if values else 0,
                    "maximum": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0.0,
                }
                for category, values in pool_sizes.items()
            },
        },
        "sources": {
            "parent_validation_bank_sha256": sha256(args.parent_bank),
            "metadata_sha256": sha256(args.metadata),
            "predicate_sha256": sha256(args.predicates),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
