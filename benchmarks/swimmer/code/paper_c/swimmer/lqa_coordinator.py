"""Remote-safe coordinator for the frozen Articulated LQA post-reference path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _atomic_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _complete_prefix(reference_root: Path, target: int) -> tuple[int, list[int]]:
    missing = []
    for system in range(target):
        root = reference_root / "systems" / f"system_{system:04d}"
        if not (root / "formal_reference_rows.csv.gz").is_file() or not (root / "formal_reference_receipt.json").is_file():
            missing.append(system)
    return target - len(missing), missing


def coordinate(root: Path, reference_root: Path, output_root: Path, target: int, poll_seconds: int, deadline_hours: float, device: str) -> dict:
    root, reference_root, output_root = Path(root), Path(reference_root), Path(output_root)
    state_path = output_root / "postreference_coordinator_state.json"
    deadline = time.monotonic() + deadline_hours * 3600.0
    while True:
        complete, missing = _complete_prefix(reference_root, target)
        _atomic_state(state_path, {
            "status": "WAITING_FOR_REFERENCE",
            "target_systems": target,
            "complete_systems": complete,
            "first_missing_systems": missing[:24],
            "learner_outcomes_read": False,
            "updated_unix_time": time.time(),
        })
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"reference prefix incomplete at deadline: {complete}/{target}")
        time.sleep(poll_seconds)

    final_root = output_root / f"finalized_{target}"
    subprocess.run([
        sys.executable, "-m", "paper_c.swimmer.lqa_finalize",
        str(root), str(root / "configs/articulated_lqa_prospective_v1.json"),
        str(root / "configs/articulated_lqa_finalization_v1.json"), str(reference_root), str(final_root),
        "--processed-systems", str(target),
    ], cwd=root, check=True)
    final_receipt_path = final_root / "formal_finalization_receipt.json"
    final_receipt = json.loads(final_receipt_path.read_text())
    stopping = final_receipt["stopping_decision"]
    if stopping["action"] != "STOP":
        result = {
            "status": "EXTENSION_REQUIRED_OUTCOME_BLIND",
            "next_systems": stopping["next_systems"],
            "contributing_systems": final_receipt["contributing_systems"],
            "learner_outcomes_read": False,
        }
        _atomic_state(state_path, result)
        return result
    if stopping["scope"] == "LOW_COVERAGE_ASSAY_NO_GO":
        result = {
            "status": "LOW_COVERAGE_ASSAY_NO_GO",
            "contributing_systems": final_receipt["contributing_systems"],
            "learner_outcomes_read": False,
        }
        _atomic_state(state_path, result)
        return result

    manifest_path = final_root / "formal_pair_manifest_frozen.csv.gz"
    learner_root = output_root / f"learner_{target}"
    _atomic_state(state_path, {
        "status": "PAIR_MANIFEST_FROZEN_STARTING_ONE_SHOT_LEARNER_INFERENCE",
        "manifest_sha256": final_receipt["pair_manifest_sha256"],
        "contributing_systems": final_receipt["contributing_systems"],
        "learner_outcomes_read": False,
    })
    subprocess.run([
        sys.executable, "-m", "paper_c.swimmer.lqa_evaluate",
        str(root), str(root / "configs/articulated_lqa_prospective_v1.json"),
        str(root / "configs/articulated_lqa_finalization_v1.json"), str(manifest_path), str(final_receipt_path),
        str(learner_root), "--device", device,
    ], cwd=root, check=True)
    learner_receipt = json.loads((learner_root / "formal_learner_receipt.json").read_text())
    result = {
        "status": "FORMAL_ARTICULATED_LQA_COMPLETE",
        "learner_receipt": str(learner_root / "formal_learner_receipt.json"),
        "direction": learner_receipt["direction"],
        "materiality": learner_receipt["materiality"],
        "claim_scope": learner_receipt["claim_scope"],
        "contributing_systems": learner_receipt["contributing_systems"],
        "learner_outcomes_read": True,
    }
    _atomic_state(state_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target", type=int, default=512)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=24.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(coordinate(args.root, args.reference_root, args.output_root, args.target, args.poll_seconds, args.deadline_hours, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
