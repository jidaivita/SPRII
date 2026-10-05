import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s3_evaluate import _alignment, _selection


ATOL = 1e-7
RTOL = 1e-5
SCORE_NAMES = (
    "conditional_physical_value",
    "standalone_query_value",
    "trajectory_diversity",
    "action_diversity",
    "oracle_reducible_value",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _correct_legacy_array(name, value):
    if name.startswith("score_") or name.endswith("_candidate_loss") or name.endswith("_gain"):
        if value.ndim != 4:
            raise ValueError(f"legacy candidate-grid array {name} must be four dimensional")
        return value.transpose(0, 1, 3, 2)
    return value


def _analyze(values, seed, bootstrap_replicates):
    scores = {name:values[f"score_{name}"] for name in SCORE_NAMES}
    results = {}
    for model in ("raw", "jepa"):
        gain = values[f"{model}_gain"]
        selection, comparisons = _selection(
            {name:scores[name] for name in SCORE_NAMES[:4]},
            gain,
            seed + (0 if model == "raw" else 1000),
            bootstrap_replicates,
        )
        alignment = _alignment(
            scores["conditional_physical_value"],
            gain,
            seed + 2000 + (0 if model == "raw" else 1000),
            bootstrap_replicates,
        )
        results[model] = {
            "selection_regret":selection,
            "comparisons":comparisons,
            "physical_gain_alignment":alignment,
        }
    primary_pass = all(row["ci_low"] > 0 for row in results["jepa"]["comparisons"])
    alignment_pass = results["jepa"]["physical_gain_alignment"]["ci_low"] > 0
    return results, primary_pass, alignment_pass


def correct_legacy(config_path, source_root, output_root):
    config_path, source_root, output_root = map(Path, (config_path, source_root, output_root))
    cfg = json.loads(config_path.read_text())
    split = "discovery"
    source_npz = source_root / f"{split}_diagnostic_system_values.npz"
    source_manifest = source_root / f"{split}_physical_manifest.csv.gz"
    source_receipt = source_root / f"{split}_receipt.json"
    with np.load(source_npz) as archive:
        corrected = {name:_correct_legacy_array(name, archive[name]) for name in archive.files}
    # Legacy gains were not merely stored with swapped axes: they also subtracted
    # each candidate loss from the baseline belonging to the misread query axis.
    # Rebuild gain from the correctly indexed baseline and candidate losses.
    for model in ("raw", "jepa"):
        corrected[f"{model}_gain"] = (
            corrected[f"{model}_baseline_loss"][:, :, None, :]
            - corrected[f"{model}_candidate_loss"]
        )
    expected_shape = (cfg[split]["systems"], 6, 6, 6)
    for name, value in corrected.items():
        if value.ndim == 4 and value.shape != expected_shape:
            raise ValueError(f"unexpected corrected shape for {name}: {value.shape}")
    results, primary_pass, alignment_pass = _analyze(
        corrected, cfg[split]["seed"], cfg["bootstrap_replicates"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    corrected_npz = output_root / "discovery_corrected_system_values.npz"
    np.savez_compressed(corrected_npz, **corrected)
    receipt = {
        "status":"DISCOVERY_CORRECTED_GO" if primary_pass and alignment_pass else "DISCOVERY_CORRECTED_NO_GO",
        "source_status":json.loads(source_receipt.read_text())["status"],
        "source_artifacts_immutable":True,
        "models_retrained":False,
        "correction":"legacy score/candidate-loss arrays were transposed from (system, history, query, candidate) to (system, history, candidate, query); gain was then recomputed from the correctly matched baseline and candidate losses",
        "axis_semantics":["system", "history", "candidate", "query"],
        "results":results,
        "primary_pass":primary_pass,
        "alignment_pass":alignment_pass,
        "source_hashes":{
            "diagnostic_system_values":_sha256(source_npz),
            "physical_manifest":_sha256(source_manifest),
            "receipt":_sha256(source_receipt),
            "config":_sha256(config_path),
        },
        "corrected_values_sha256":_sha256(corrected_npz),
        "sealed_accessed":False,
    }
    (output_root / "discovery_corrected_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _comparison(old, replay, exact):
    difference = np.abs(replay - old)
    passed = np.array_equal(replay, old) if exact else np.allclose(replay, old, atol=ATOL, rtol=RTOL)
    return {
        "passed":bool(passed),
        "exact_required":exact,
        "max_abs":float(difference.max(initial=0.0)),
        "atol":0.0 if exact else ATOL,
        "rtol":0.0 if exact else RTOL,
    }


def compare(corrected_root, legacy_root, replay_root, output_path):
    corrected_root, legacy_root, replay_root, output_path = map(Path, (corrected_root, legacy_root, replay_root, output_path))
    with np.load(corrected_root / "discovery_corrected_system_values.npz") as old_archive, np.load(replay_root / "discovery_diagnostic_system_values.npz") as replay_archive:
        arrays = {}
        for name in old_archive.files:
            exact = name.startswith("score_")
            arrays[name] = _comparison(old_archive[name], replay_archive[name], exact)
    old_manifest = pd.read_csv(legacy_root / "discovery_physical_manifest.csv.gz")
    replay_manifest = pd.read_csv(replay_root / "discovery_physical_manifest.csv.gz")
    metadata_columns = ["system_index", "realization", "anchor_index", "candidate_index", "query_index"]
    metadata_exact = old_manifest[metadata_columns].equals(replay_manifest[metadata_columns])
    physical = {
        name:_comparison(old_manifest[name].to_numpy(), replay_manifest[name].to_numpy(), exact=True)
        for name in SCORE_NAMES
    }
    replay_receipt = json.loads((replay_root / "discovery_receipt.json").read_text())
    passed = metadata_exact and all(item["passed"] for item in arrays.values()) and all(item["passed"] for item in physical.values()) and all(item["allclose"] for item in replay_receipt["prediction_replay_checks"].values())
    receipt = {
        "status":"DETERMINISTIC_REPLAY_VERIFIED" if passed else "DETERMINISTIC_REPLAY_MISMATCH",
        "passed":passed,
        "metadata_exact":metadata_exact,
        "array_comparisons":arrays,
        "physical_manifest_comparisons":physical,
        "prediction_repeat_checks":replay_receipt["prediction_replay_checks"],
        "prediction_history_limitation":"legacy S3-R artifacts did not store row-level predictions; old-to-replay prediction comparison is therefore unavailable. Replay inference was repeated on identical row arrays and checked before aggregation.",
        "tolerance":{"atol":ATOL,"rtol":RTOL},
        "source_hashes":{
            "corrected_receipt":_sha256(corrected_root / "discovery_corrected_receipt.json"),
            "legacy_manifest":_sha256(legacy_root / "discovery_physical_manifest.csv.gz"),
            "replay_receipt":_sha256(replay_root / "discovery_receipt.json"),
            "replay_rows":_sha256(replay_root / "discovery_row_level_replay.npz"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    correction = sub.add_parser("correct")
    correction.add_argument("config")
    correction.add_argument("source_root")
    correction.add_argument("output_root")
    comparison = sub.add_parser("compare")
    comparison.add_argument("corrected_root")
    comparison.add_argument("legacy_root")
    comparison.add_argument("replay_root")
    comparison.add_argument("output_path")
    args = parser.parse_args()
    if args.command == "correct":
        result = correct_legacy(args.config, args.source_root, args.output_root)
    else:
        result = compare(args.corrected_root, args.legacy_root, args.replay_root, args.output_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
