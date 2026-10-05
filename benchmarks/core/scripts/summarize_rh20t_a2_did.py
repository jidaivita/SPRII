#!/usr/bin/env python3
"""Summarize frozen A2 task-level specificity and paired DiD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from persistent_jepa.rh20t_did import paired_specificity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument(
        "--record-key", choices=("rows", "secondary_rows"), default="rows"
    )
    parser.add_argument(
        "--endpoint",
        choices=(
            "task_macro_h4_normalized_force_torque_mse",
            "h16_tcp_xyz_error_cm",
        ),
        default="task_macro_h4_normalized_force_torque_mse",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = json.loads(args.records.read_text())
    if payload.get("status") != "FROZEN_EVALUATION_RECORDS":
        raise RuntimeError("evaluation records are not frozen")
    expected = {
        "rows": "task_macro_h4_normalized_force_torque_mse",
        "secondary_rows": "h16_tcp_xyz_error_cm",
    }
    if args.endpoint != expected[args.record_key]:
        raise ValueError("record key and endpoint mismatch")
    result = paired_specificity(
        payload[args.record_key],
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    result.update({
        "schema_version": "1.1",
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "endpoint": args.endpoint,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
