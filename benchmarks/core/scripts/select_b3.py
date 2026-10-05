#!/usr/bin/env python3
"""Equal-budget B3 selection and immutable test unsealing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from persistent_jepa.runtime import sha256_file
from persistent_jepa.test_seal import write_immutable_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    return parser.parse_args()


def ranking_tuple(report: dict) -> tuple[float, float, float, float, float]:
    donor = report.get("donor")
    if donor is None:
        raise ValueError("B3 candidate lacks donor evaluation")
    config = report["training_config"]
    return (
        float(donor["correct_h16"]["state"]),
        -float(donor["state_mse_gap_shuffled_minus_correct"]),
        -float(report["drag_probe"]["ridge_system_r2"]),
        float(report["self_functional"]["h16"]["state"]),
        float(config["lambda_p"]) + float(config["lambda_x"]) * 1e-6,
    )


def main() -> None:
    args = parse_args()
    candidates = []
    for path in args.metrics:
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["training_config"]
        if report["split"] != "val" or report["variant"] != "B3":
            raise ValueError(f"not a B3 validation report: {path}")
        if report["checkpoint_step"] != 20000 or int(config["seed"]) != 0:
            raise ValueError(f"equal-budget violation (requires seed0@20k): {path}")
        if Path(report["checkpoint"]).is_file() and sha256_file(Path(report["checkpoint"])) != report["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        candidates.append((ranking_tuple(report), path, report))
    candidates.sort(key=lambda item: item[0])
    _, best_path, best = candidates[0]
    config = best["training_config"]
    payload = {
        "variant": "B3",
        "full_config": config,
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "dataset_manifest_sha256": best["dataset_manifest_sha256"],
        "primary_metric": {
            "name": "val_correct_donor_h16_normalized_state_mse",
            "value": best["donor"]["correct_h16"]["state"],
        },
        "tie_breakers": {
            "correct_vs_shuffled_gap": best["donor"]["state_mse_gap_shuffled_minus_correct"],
            "drag_ridge_probe_r2": best["drag_probe"]["ridge_system_r2"],
            "total_validation_future_state_error_h16": best["self_functional"]["h16"]["state"],
        },
        "decoder_ridge": best["decoder_ridge"],
        "probe_ridge": best["drag_probe"]["ridge_alpha"],
        "pairing_seed": best["pairing_seed"],
        "code_revision": args.code_revision,
        "source_validation_report": str(best_path),
        "source_validation_report_sha256": sha256_file(best_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_immutable_selection(args.output, payload)
    decision = {
        "rule": "seed0@20k only; primary ascending, donor gap descending, ridge R2 descending, self h16 ascending",
        "ranked_candidates": [
            {
                "rank": index + 1,
                "metrics": str(path),
                "lambda_p": report["training_config"]["lambda_p"],
                "lambda_x": report["training_config"]["lambda_x"],
                "ranking_tuple": list(rank),
            }
            for index, (rank, path, report) in enumerate(candidates)
        ],
        "selected": str(best_path),
        "selection_json": str(args.output),
    }
    args.decision_log.parent.mkdir(parents=True, exist_ok=True)
    args.decision_log.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
