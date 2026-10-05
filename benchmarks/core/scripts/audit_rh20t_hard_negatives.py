#!/usr/bin/env python3
"""Run the registered metadata-only A2 donor feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from persistent_jepa.rh20t_hard_negative import (
    MetadataPredicate,
    common_query_intersection,
    coverage_report,
    deterministic_donor,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    metadata_payload = json.loads(args.metadata.read_text())
    records = metadata_payload["records"]
    predicate_payload = json.loads(args.predicates.read_text())
    if predicate_payload.get("status") != "FROZEN_BEFORE_MODEL_OUTPUT":
        raise RuntimeError("predicate manifest is not frozen")
    predicates = [MetadataPredicate.from_dict(item) for item in predicate_payload["predicates"]]
    if {item.name for item in predicates} != {"D_M", "D_HP", "D_HS", "D_R"}:
        raise ValueError("exactly D_M, D_HP, D_HS, and D_R predicates are required")
    reports = {predicate.name: coverage_report(records, predicate) for predicate in predicates}
    categories = ("D_M", "D_HP", "D_HS", "D_R")
    q_star = common_query_intersection(*(reports[name] for name in categories))
    assignments = {}
    for query_id in q_star:
        assignments[query_id] = {
            name: deterministic_donor(
                query_id, name, reports[name]["legal_pools"][query_id], args.evaluator_seed
            )
            for name in categories
        }
    output = {
        "schema_version": "1.1",
        "audit_mode": "METADATA_ONLY",
        "metadata_sha256": sha256(args.metadata),
        "predicate_manifest_sha256": sha256(args.predicates),
        "evaluator_seed": args.evaluator_seed,
        "reports": reports,
        "q_star": q_star,
        "assignments": assignments,
        "model_output_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
