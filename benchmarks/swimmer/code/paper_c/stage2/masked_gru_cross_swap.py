"""No-retrain 2x2 cross-swap analysis for the completed masked-GRU assay.

This is an independent, post-hoc supporting consumer.  It never trains or
selects a model and writes only below its own output namespace.  The completed
masked-GRU formal cache, base checkpoints, adapter checkpoints, shuffle maps,
and sufficient statistics are treated as immutable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.formal_data import load_arrays
from paper_c.stage2 import delta_gated_isolation as e3
from paper_c.stage2 import masked_gru_cross_architecture as base
from paper_c.stage2 import masked_gru_formal as upstream
from paper_c.stage2 import routing_intervention as routing


STATUS_FREEZE = "MASKED_GRU_CROSS_SWAP_IMPLEMENTATION_FROZEN"
STATUS_COMMAND_MATRIX = "MASKED_GRU_CROSS_SWAP_COMMAND_MATRIX_FROZEN"
STATUS_PARITY = "MASKED_GRU_CROSS_SWAP_ONE_TASK_TWO_SHARD_PARITY_PASS"
STATUS_SHARD = "MASKED_GRU_CROSS_SWAP_SHARD_COMPLETE"
STATUS_BASE = "MASKED_GRU_CROSS_SWAP_BASE_COMPLETE"
STATUS_FINAL = "MASKED_GRU_CROSS_SWAP_SUPPORTING_COMPLETE"

CELLS = ("TT", "TS", "ST", "SS")
BASE_SEEDS = (64101, 64103)
OPTIMIZATION_SEEDS = (86101, 86103, 86107)
SHARDS = 2
ROWS_PER_SYSTEM = 36
SYSTEMS = 512
BASE_GPU_OWNERSHIP = {64101: 2, 64103: 3}
REQUIRED_ENVIRONMENT = {
    "PAPER_C_REMOTE_EXECUTION": "1",
    "PAPER_C_REMOTE_HOST": "worker_b",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
STAT_COLUMNS = (
    "system_index", "cell", "seed", "count",
    "sum_loss", "sum_gain", "sum2_loss", "sum2_gain",
)
HISTORICAL_STATISTIC_COLUMNS = ("sum_loss", "sum_gain", "sum2_loss", "sum2_gain")
HISTORICAL_ABSOLUTE_TOLERANCE = 1e-15
MAXIMUM_E_INPUT_PERTURBATION = 2.777777777777778e-17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(
        temporary, index=False, float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    os.replace(temporary, path)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("schema_version") != "1.0":
        raise RuntimeError("cross-swap schema changed")
    if cfg.get("status") != "IMPLEMENTED_DO_NOT_EXECUTE_BEFORE_INDEPENDENT_REVIEW":
        raise RuntimeError("cross-swap config is not in the reviewed pre-execution state")
    if cfg.get("execution_revision") != "v3_dual_numerical_compatibility_gate":
        raise RuntimeError("cross-swap execution revision changed")
    if cfg.get("protocol") != "protocol/PAPER_C_MASKED_GRU_CROSS_SWAP_SUPPORTING_V3_NUMERICAL_COMPATIBILITY_ADDENDUM.md":
        raise RuntimeError("cross-swap V3 execution addendum changed")
    if (
        cfg.get("prior_execution_protocol")
        != "protocol/PAPER_C_MASKED_GRU_CROSS_SWAP_SUPPORTING_V2_EXECUTION_ADDENDUM.md"
        or cfg.get("prior_execution_protocol_sha256")
        != "ee9ada2433a172697597dfd041e57a1b65c18293d0793f2a8652544aa60a6d49"
    ):
        raise RuntimeError("cross-swap V2 execution addendum binding changed")
    if (
        cfg.get("scientific_protocol") != "protocol/PAPER_C_MASKED_GRU_CROSS_SWAP_SUPPORTING_V1.md"
        or cfg.get("scientific_protocol_sha256") != "67ea430eb4a38dad14fc2990dcaa4ce9aa37032b0a4e00424550af215b6f6d21"
    ):
        raise RuntimeError("cross-swap scientific protocol binding changed")
    if cfg.get("output_root") != "runs/supporting/masked_gru_cross_swap_v3":
        raise RuntimeError("cross-swap independent output namespace changed")
    if cfg.get("command_matrix") != "work/MASKED_GRU_CROSS_SWAP_REMOTE_MATRIX_V3.md":
        raise RuntimeError("cross-swap command matrix changed")
    expected_superseded = [
        {
            "revision": "v1",
            "output_root": "runs/supporting/masked_gru_cross_swap_v1",
            "implementation_freeze_path": "runs/supporting/masked_gru_cross_swap_v1/IMPLEMENTATION_FROZEN.json",
            "implementation_freeze_sha256": "66d50e50dd9e0dde7d5eed1601b05ae413f2fb13deba9b355a97323197bb8494",
            "command_matrix_freeze_path": "runs/supporting/masked_gru_cross_swap_v1/COMMAND_MATRIX_FROZEN.json",
            "command_matrix_freeze_sha256": "5dcfbac509e5c8315a864f76ab7aa2edfb5da2f82cd959379cff8bf296c4d54e",
            "failed_parity_jobs": [
                "parity-v1-seed-64101",
                "parity-v1-seed-64103",
            ],
            "scientific_artifacts_created": False,
            "reason": "V1 parity exposed a one-ULP reduction-order mismatch before any scientific artifact was written.",
        },
        {
            "revision": "v2",
            "output_root": "runs/supporting/masked_gru_cross_swap_v2",
            "implementation_freeze_path": "runs/supporting/masked_gru_cross_swap_v2/IMPLEMENTATION_FROZEN.json",
            "implementation_freeze_sha256": "dd97bc84b6bd2e82e97e5b6ae0ffeee29935dbc05eec519d003343adc094befc",
            "command_matrix_freeze_path": "runs/supporting/masked_gru_cross_swap_v2/COMMAND_MATRIX_FROZEN.json",
            "command_matrix_freeze_sha256": "8c2894f61ecaa06881ae2a9c88d631843cc0d862925fc883774821296972941f",
            "failed_parity_jobs": [
                "parity-v2-seed-64101",
                "parity-v2-seed-64103",
            ],
            "scientific_artifacts_created": False,
            "reason": "V2 matched the current upstream evaluator exactly but the historical CSV retained sub-1e-16 runtime/serialization differences.",
        },
    ]
    if cfg.get("superseded_executions") != expected_superseded:
        raise RuntimeError("cross-swap V1/V2 supersession binding changed")
    numerical = cfg.get("numerical_compatibility", {})
    if (
        numerical.get("same_runtime_upstream_array_exact") is not True
        or float(numerical.get("historical_absolute_tolerance", -1.0)) != HISTORICAL_ABSOLUTE_TOLERANCE
        or float(numerical.get("historical_relative_tolerance", -1.0)) != 0.0
        or float(numerical.get("maximum_per_system_e_input_perturbation", -1.0))
        != MAXIMUM_E_INPUT_PERTURBATION
        or tuple(numerical.get("historical_statistic_columns", [])) != HISTORICAL_STATISTIC_COLUMNS
    ):
        raise RuntimeError("cross-swap numerical compatibility gate changed")
    if tuple(cfg["upstream"]["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("cross-swap base-seed schedule changed")
    if tuple(cfg["upstream"]["optimization_seeds"]) != OPTIMIZATION_SEEDS:
        raise RuntimeError("cross-swap optimization-seed schedule changed")
    if cfg["upstream"].get("final_result_status") != "MASKED_GRU_COMPETENT_ARCHITECTURE_CONTRADICTION":
        raise RuntimeError("cross-swap upstream final-result status changed")
    if cfg["upstream"].get("articulated_result_keys") != ["64101", "64103"]:
        raise RuntimeError("cross-swap upstream articulated-result keys changed")
    design = cfg["design"]
    if tuple(design["cells"]) != CELLS or int(design["systems"]) != SYSTEMS:
        raise RuntimeError("cross-swap four-cell population changed")
    if design.get("training_arms") != ["true", "shuffled"] or design.get("input_roles") != ["self", "shuffled"]:
        raise RuntimeError("cross-swap factorial roles changed")
    if int(design["rows_per_system"]) != ROWS_PER_SYSTEM:
        raise RuntimeError("cross-swap row population changed")
    if int(design["formal_shards"]) != SHARDS or design["fixed_merge_order"] != [0, 1]:
        raise RuntimeError("cross-swap shard plan changed")
    if design.get("primary") != "E_input=0.5*((G_TT-G_TS)+(G_ST-G_SS))":
        raise RuntimeError("cross-swap primary estimand changed")
    if design.get("secondary") != "E_interaction=(G_TT-G_TS)-(G_ST-G_SS)":
        raise RuntimeError("cross-swap secondary estimand changed")
    inference = cfg["inference"]
    if int(inference["bootstrap_replicates"]) != 4000 or int(inference["bootstrap_seed"]) != 86211:
        raise RuntimeError("cross-swap inference changed")
    if inference.get("same_resample_matrix_for_all_estimands") is not True:
        raise RuntimeError("cross-swap requires one common bootstrap resample matrix")
    if inference.get("optimization_seeds_are_scientific_samples") is not False:
        raise RuntimeError("optimization seeds cannot become scientific samples")
    if inference.get("base_seeds_are_scientific_samples") is not False:
        raise RuntimeError("base seeds cannot become scientific samples")
    if inference.get("scientific_unit") != "physical_system":
        raise RuntimeError("cross-swap scientific unit changed")
    if inference.get("interval") != "two_sided_percentile_95":
        raise RuntimeError("cross-swap interval changed")
    if inference.get("fixed_panel_summary_is_supporting_only") is not True:
        raise RuntimeError("cross-swap fixed-panel evidence role changed")
    execution = cfg.get("execution", {})
    if execution.get("remote_only") is not True:
        raise RuntimeError("cross-swap execution must remain remote-only")
    if execution.get("runtime_root") != "runs/cross_swap_runtime":
        raise RuntimeError("cross-swap isolated runtime root changed")
    if execution.get("allowed_hosts") != ["worker_b"]:
        raise RuntimeError("cross-swap allowed host changed")
    if execution.get("allowed_gpu_ids") != [2, 3]:
        raise RuntimeError("cross-swap allowed physical GPUs changed")
    if int(execution.get("threads_per_worker", -1)) != 1:
        raise RuntimeError("cross-swap worker thread count changed")
    if execution.get("statistics_dtype") != "float64":
        raise RuntimeError("cross-swap sufficient-statistics dtype changed")
    for flag in (
        "protected_scope_2_untouched", "fresh_packet_mutation_forbidden",
        "upstream_artifact_mutation_forbidden",
    ):
        if execution.get(flag) is not True:
            raise RuntimeError(f"cross-swap execution flag changed: {flag}")
    required_forbidden = {
        "model_training", "checkpoint_selection", "hyperparameter_search",
        "result_dependent_metric_selection", "fresh_packet_mutation",
        "upstream_gru_artifact_mutation",
    }
    if not required_forbidden.issubset(set(cfg.get("forbidden", []))):
        raise RuntimeError("cross-swap forbidden-operation set changed")


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _require_runtime_root(root: Path, cfg: dict) -> None:
    actual = root.resolve()
    expected = Path(cfg["execution"]["runtime_root"]).resolve()
    if actual != expected:
        raise RuntimeError(f"cross-swap must execute from isolated runtime root {expected}; got {actual}")


def _output(root: Path, cfg: dict) -> Path:
    return root / cfg["output_root"]


def _validate_command_matrix(root: Path, cfg: dict) -> dict:
    path = root / cfg["command_matrix"]
    text = path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if "unset CUDA_VISIBLE_DEVICES" not in lines:
        raise RuntimeError("cross-swap command matrix must unset CUDA_VISIBLE_DEVICES")
    if any(
        line.startswith("export CUDA_VISIBLE_DEVICES") or line.startswith("CUDA_VISIBLE_DEVICES=")
        for line in lines
    ):
        raise RuntimeError("cross-swap command matrix must not remap physical GPUs")
    for name, value in REQUIRED_ENVIRONMENT.items():
        if f"export {name}={value}" not in lines:
            raise RuntimeError(f"cross-swap command matrix lacks frozen environment: {name}")
    setup = (
        f"export PAPER_C_ROOT={cfg['execution']['runtime_root']}",
        'export PYTHONPATH="$PAPER_C_ROOT/code"',
        'cd "$PAPER_C_ROOT"',
    )
    if any(lines.count(line) != 1 for line in setup):
        raise RuntimeError("cross-swap command matrix changed the remote root or Python path")
    if 'export M=paper_c.stage2.masked_gru_cross_swap' not in lines:
        raise RuntimeError("cross-swap command matrix invokes the wrong consumer")
    if 'export C="$PAPER_C_ROOT/configs/masked_gru_cross_swap_supporting_v3.json"' not in lines:
        raise RuntimeError("cross-swap command matrix invokes the wrong config")
    review_sequence = (
        "python3 -m pytest -q tests/unit/test_masked_gru_cross_swap.py tests/unit/test_masked_gru_formal.py tests/unit/test_masked_gru_cross_architecture.py",
        'python3 -m $M "$PAPER_C_ROOT" "$C" self-test',
        'python3 -m $M "$PAPER_C_ROOT" "$C" freeze-implementation --authorize-reviewed-code',
    )
    review_positions = []
    for command in review_sequence:
        if lines.count(command) != 1:
            raise RuntimeError(f"cross-swap command matrix changed review/freeze command: {command}")
        review_positions.append(lines.index(command))
    if review_positions != sorted(review_positions):
        raise RuntimeError("cross-swap review/freeze command order changed")
    commands = {
        64101: [
            'python3 -m $M "$PAPER_C_ROOT" "$C" parity 64101 --device cuda:2',
            'python3 -m $M "$PAPER_C_ROOT" "$C" evaluate-shard 64101 --shard-index 0 --device cuda:2',
            'python3 -m $M "$PAPER_C_ROOT" "$C" evaluate-shard 64101 --shard-index 1 --device cuda:2',
            'python3 -m $M "$PAPER_C_ROOT" "$C" merge-base 64101',
        ],
        64103: [
            'python3 -m $M "$PAPER_C_ROOT" "$C" parity 64103 --device cuda:3',
            'python3 -m $M "$PAPER_C_ROOT" "$C" evaluate-shard 64103 --shard-index 0 --device cuda:3',
            'python3 -m $M "$PAPER_C_ROOT" "$C" evaluate-shard 64103 --shard-index 1 --device cuda:3',
            'python3 -m $M "$PAPER_C_ROOT" "$C" merge-base 64103',
        ],
    }
    for base_seed, sequence in commands.items():
        positions = []
        for command in sequence:
            if lines.count(command) != 1:
                raise RuntimeError(f"cross-swap command matrix changed command for base {base_seed}: {command}")
            positions.append(lines.index(command))
        if positions != sorted(positions):
            raise RuntimeError(f"cross-swap command order changed for base {base_seed}")
        if positions[0] <= review_positions[-1]:
            raise RuntimeError("cross-swap formal commands must follow implementation freeze")
    forbidden_device_pairs = (
        "64101 --device cuda:3", "64103 --device cuda:2",
        "64101 --shard-index 0 --device cuda:3", "64101 --shard-index 1 --device cuda:3",
        "64103 --shard-index 0 --device cuda:2", "64103 --shard-index 1 --device cuda:2",
    )
    if any(token in text for token in forbidden_device_pairs):
        raise RuntimeError("cross-swap command matrix violates base/GPU ownership")
    summarize_command = 'python3 -m $M "$PAPER_C_ROOT" "$C" summarize'
    if lines.count(summarize_command) != 1:
        raise RuntimeError("cross-swap command matrix must contain one final summarize")
    if lines.index(summarize_command) <= max(lines.index(sequence[-1]) for sequence in commands.values()):
        raise RuntimeError("cross-swap summarize must follow both base merges")
    return {
        "path": _relative(root, path), "sha256": sha256(path),
        "cuda_visible_devices_unset": True,
        "required_environment": REQUIRED_ENVIRONMENT,
        "base_gpu_ownership": {str(key): value for key, value in BASE_GPU_OWNERSHIP.items()},
        "per_base_sequence": ["parity", "shard_0", "shard_1", "merge"],
        "final_command": "summarize",
    }


def _command_matrix_receipt_path(root: Path, cfg: dict) -> Path:
    return _output(root, cfg) / "COMMAND_MATRIX_FROZEN.json"


def _verify_command_matrix_receipt(root: Path, config_path: Path, cfg: dict) -> dict:
    path = _command_matrix_receipt_path(root, cfg)
    receipt = json.loads(path.read_text())
    current = _validate_command_matrix(root, cfg)
    if receipt.get("status") != STATUS_COMMAND_MATRIX:
        raise RuntimeError("cross-swap command-matrix receipt is invalid")
    for key, expected in current.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"cross-swap command-matrix receipt changed: {key}")
    if receipt.get("config_sha256") != sha256(config_path):
        raise RuntimeError("cross-swap command-matrix receipt has a stale config")
    if receipt.get("protocol_sha256") != sha256(root / cfg["protocol"]):
        raise RuntimeError("cross-swap command-matrix receipt has a stale protocol")
    if receipt.get("scientific_protocol_sha256") != sha256(root / cfg["scientific_protocol"]):
        raise RuntimeError("cross-swap command-matrix receipt has a stale scientific protocol")
    return receipt


def _upstream_config(root: Path, cfg: dict) -> tuple[Path, dict]:
    path = root / cfg["upstream"]["config"]
    if sha256(path) != cfg["upstream"]["config_sha256"]:
        raise RuntimeError("masked-GRU upstream config hash changed")
    upstream_cfg = base.load_cfg(path)
    if tuple(upstream_cfg["environments"]["articulated"]["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("masked-GRU upstream Articulated bases changed")
    if tuple(upstream_cfg["adapter"]["seeds"]) != OPTIMIZATION_SEEDS:
        raise RuntimeError("masked-GRU upstream adapter seeds changed")
    if upstream_cfg["adapter"]["arms"] != ["true", "shuffled"]:
        raise RuntimeError("masked-GRU upstream adapter arms changed")
    return path, upstream_cfg


def _upstream_result(root: Path, cfg: dict) -> dict:
    path = root / cfg["upstream"]["final_result"]
    if sha256(path) != cfg["upstream"]["final_result_sha256"]:
        raise RuntimeError("masked-GRU final result hash changed")
    result = json.loads(path.read_text())
    if result.get("status") != cfg["upstream"]["final_result_status"]:
        raise RuntimeError("masked-GRU completed classification changed")
    if sorted(result["articulated_formal_results"]) != sorted(cfg["upstream"]["articulated_result_keys"]):
        raise RuntimeError("masked-GRU final result has the wrong base learners")
    return result


def _upstream_result_binding(root: Path, cfg: dict) -> dict:
    result = _upstream_result(root, cfg)
    path = root / cfg["upstream"]["final_result"]
    return {
        "path": _relative(root, path), "sha256": sha256(path),
        "status": result["status"],
        "articulated_result_keys": sorted(result["articulated_formal_results"]),
    }


def _existing_statistics(root: Path, cfg: dict, base_seed: int) -> tuple[Path, pd.DataFrame]:
    item = cfg["upstream"]["existing_statistics"][str(int(base_seed))]
    path = root / item["path"]
    if sha256(path) != item["sha256"]:
        raise RuntimeError(f"existing masked-GRU statistics changed for base {base_seed}")
    frame = _read_csv(path)
    required = {
        "system_index", "arm", "seed", "count", "sum_loss", "sum_gain",
        "sum2_loss", "sum2_gain",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("existing masked-GRU statistics lack required columns")
    if frame.system_index.nunique() != SYSTEMS:
        raise RuntimeError("existing masked-GRU statistics have the wrong system count")
    return path, frame


def _source_paths(root: Path, config_path: Path, cfg: dict) -> list[Path]:
    paths = [
        config_path,
        root / cfg["protocol"],
        root / cfg["prior_execution_protocol"],
        root / cfg["scientific_protocol"],
        root / cfg["tests"],
        root / cfg["command_matrix"],
        Path(__file__).resolve(),
        root / cfg["upstream"]["config"],
        Path(upstream.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(e3.__file__).resolve(),
        Path(routing.__file__).resolve(),
        root / "code/paper_c/coupled_sled/formal_data.py",
        root / "code/paper_c/coupled_sled/learner.py",
    ]
    unique = {path.resolve(): path.resolve() for path in paths}
    if any(not path.is_file() for path in unique):
        missing = [str(path) for path in unique if not path.is_file()]
        raise FileNotFoundError(f"cross-swap source dependency missing: {missing}")
    if sha256(root / cfg["scientific_protocol"]) != cfg["scientific_protocol_sha256"]:
        raise RuntimeError("cross-swap scientific protocol hash changed")
    if sha256(root / cfg["prior_execution_protocol"]) != cfg["prior_execution_protocol_sha256"]:
        raise RuntimeError("cross-swap prior execution protocol hash changed")
    return sorted(unique, key=lambda path: str(path))


def _verify_superseded_execution(root: Path, cfg: dict) -> list[dict]:
    results = []
    for item in cfg["superseded_executions"]:
        revision = str(item["revision"])
        result = {
            "revision": revision,
            "output_root": item["output_root"],
            "failed_parity_jobs": list(item["failed_parity_jobs"]),
            "scientific_artifacts_created": item["scientific_artifacts_created"],
            "reason": item["reason"],
        }
        for name in ("implementation_freeze", "command_matrix_freeze"):
            relative = item[f"{name}_path"]
            path = root / relative
            expected = item[f"{name}_sha256"]
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"superseded {revision} {name} marker changed or is missing")
            result[name] = {"path": relative, "sha256": expected}
        prior_root = root / item["output_root"]
        extra = sorted(
            str(path.relative_to(prior_root)) for path in prior_root.rglob("*")
            if path.is_file() and path.name not in {"IMPLEMENTATION_FROZEN.json", "COMMAND_MATRIX_FROZEN.json"}
        )
        if extra:
            raise RuntimeError(
                f"superseded {revision} unexpectedly contains scientific artifacts: {extra}"
            )
        results.append(result)
    return results


def _artifact_manifest(root: Path, cfg: dict, upstream_cfg: dict) -> dict:
    # This verifies the completed one-shot authorization and all adapter/base
    # receipts through the original frozen consumer before recording hashes.
    upstream_config_path = root / cfg["upstream"]["config"]
    authorization = upstream._verify_one_shot_authorization(root, upstream_config_path, upstream_cfg)
    authorization_path = upstream._output(root, upstream_cfg) / "authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"
    manifest: dict[str, object] = {
        "upstream_final_result": _upstream_result_binding(root, cfg),
        "upstream_authorization": {
            "path": _relative(root, authorization_path),
            "sha256": sha256(authorization_path),
            "status": authorization["status"],
        },
        "bases": {},
    }
    for base_seed in BASE_SEEDS:
        checkpoint, competence = upstream._verify_base(
            root, upstream_cfg, "articulated", base_seed, require_competent=True,
        )
        competence_path = upstream._base_root(root, upstream_cfg, "articulated", base_seed) / "BASE_COMPETENCE_RECEIPT.json"
        cache, cache_receipt = upstream._merged_cache(root, upstream_cfg, "articulated", base_seed, "formal")
        del cache
        cache_path = root / cache_receipt["cache"]
        parity_path = upstream._formal_eval_root(root, upstream_cfg, base_seed) / "FORMAL_PARITY.json"
        parity = json.loads(parity_path.read_text())
        if parity.get("status") != "MASKED_GRU_ONE_TASK_TWO_SHARD_PARITY_PASS":
            raise RuntimeError("completed masked-GRU parity receipt is invalid")
        stats_path, _ = _existing_statistics(root, cfg, base_seed)
        adapters = []
        for row in upstream._formal_adapter_table(root, upstream_cfg, base_seed).itertuples(index=False):
            checkpoint_path = root / row.checkpoint
            receipt_path = checkpoint_path.parent / "receipt.json"
            adapters.append({
                "arm": str(row.arm), "seed": int(row.seed),
                "checkpoint": _relative(root, checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "receipt": _relative(root, receipt_path),
                "receipt_sha256": sha256(receipt_path),
            })
        manifest["bases"][str(base_seed)] = {
            "base_checkpoint": _relative(root, checkpoint),
            "base_checkpoint_sha256": sha256(checkpoint),
            "competence_receipt": _relative(root, competence_path),
            "competence_receipt_sha256": sha256(competence_path),
            "competence_status": competence["status"],
            "formal_cache": _relative(root, cache_path),
            "formal_cache_sha256": sha256(cache_path),
            "formal_cache_receipt_sha256": sha256(cache_path.parent / "merged.receipt.json"),
            "upstream_parity": _relative(root, parity_path),
            "upstream_parity_sha256": sha256(parity_path),
            "shuffle_map_hashes": parity["shuffle_map_hashes"],
            "existing_statistics": _relative(root, stats_path),
            "existing_statistics_sha256": sha256(stats_path),
            "adapters": adapters,
        }
    return manifest


def freeze_implementation(root: Path, config_path: Path, authorize: bool) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    if not authorize:
        raise RuntimeError("cross-swap freeze requires --authorize-reviewed-code")
    cfg = _load_config(config_path)
    _require_runtime_root(root, cfg)
    _, upstream_cfg = _upstream_config(root, cfg)
    base._require_remote(root, upstream_cfg)
    _upstream_result(root, cfg)
    target = _output(root, cfg) / "IMPLEMENTATION_FROZEN.json"
    command_receipt_path = _command_matrix_receipt_path(root, cfg)
    if target.exists() or command_receipt_path.exists():
        raise RuntimeError("cross-swap implementation or command matrix is already frozen")
    command_binding = _validate_command_matrix(root, cfg)
    sources = {
        _relative(root, path): sha256(path)
        for path in _source_paths(root, config_path, cfg)
    }
    command_receipt = {
        "schema_version": "1.0", "status": STATUS_COMMAND_MATRIX,
        **command_binding,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(root / cfg["protocol"]),
        "scientific_protocol_sha256": sha256(root / cfg["scientific_protocol"]),
    }
    # Finish every fallible validation before publishing either freeze marker.
    # In particular, upstream artifact binding can fail when a shared source
    # file has been superseded by another isolated experiment.  Publishing the
    # command receipt first would leave a misleading half-frozen namespace.
    artifact_manifest = _artifact_manifest(root, cfg, upstream_cfg)
    superseded_execution = _verify_superseded_execution(root, cfg)
    command_text = json.dumps(command_receipt, indent=2, sort_keys=True) + "\n"
    payload = {
        "schema_version": "1.0", "status": STATUS_FREEZE,
        "created_at_unix": time.time(),
        "evidence_identity": "post_hoc_supporting_existing_formal_systems",
        "model_training": False, "checkpoint_selection": False,
        "formal_outcomes_already_known": True,
        "fresh_packet_read": False, "fresh_packet_mutated": False,
        "command_matrix_receipt": _relative(root, command_receipt_path),
        "command_matrix_receipt_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
        "source_hashes": sources,
        "artifact_manifest": artifact_manifest,
        "superseded_execution": superseded_execution,
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    try:
        _atomic_text(command_receipt_path, command_text)
        _atomic_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except BaseException:
        # Both paths are new by contract.  Remove either marker if the paired
        # publication does not complete, so a retry cannot inherit partial
        # authorization state.
        target.unlink(missing_ok=True)
        command_receipt_path.unlink(missing_ok=True)
        raise
    return payload


def _verify_freeze(root: Path, config_path: Path, cfg: dict, upstream_cfg: dict) -> dict:
    _require_runtime_root(root, cfg)
    path = _output(root, cfg) / "IMPLEMENTATION_FROZEN.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_FREEZE:
        raise RuntimeError("cross-swap implementation freeze is missing or invalid")
    if receipt.get("model_training") is not False or receipt.get("checkpoint_selection") is not False:
        raise RuntimeError("cross-swap freeze violates no-training semantics")
    command_receipt = _verify_command_matrix_receipt(root, config_path, cfg)
    command_path = _command_matrix_receipt_path(root, cfg)
    if receipt.get("command_matrix_receipt") != _relative(root, command_path):
        raise RuntimeError("cross-swap implementation freeze points to the wrong command-matrix receipt")
    if receipt.get("command_matrix_receipt_sha256") != sha256(command_path):
        raise RuntimeError("cross-swap implementation freeze has a stale command-matrix receipt")
    if command_receipt.get("status") != STATUS_COMMAND_MATRIX:
        raise RuntimeError("cross-swap command-matrix freeze is invalid")
    for relative, expected in receipt["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"cross-swap source changed after freeze: {relative}")
    if receipt["artifact_manifest"] != _artifact_manifest(root, cfg, upstream_cfg):
        raise RuntimeError("cross-swap upstream artifact manifest changed after freeze")
    if receipt.get("superseded_execution") != _verify_superseded_execution(root, cfg):
        raise RuntimeError("cross-swap V1/V2 supersession evidence changed after freeze")
    return receipt


def canonical_frame_sha(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["system_index", "cell", "seed"], kind="stable").reset_index(drop=True)
    serialized = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def merge_statistic_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        raise ValueError("at least one cross-swap shard is required")
    if any(tuple(part.columns) != tuple(parts[0].columns) for part in parts):
        raise RuntimeError("cross-swap shard schemas differ")
    merged = pd.concat(parts, ignore_index=True).sort_values(
        ["system_index", "cell", "seed"], kind="stable",
    ).reset_index(drop=True)
    if merged.duplicated(["system_index", "cell", "seed"]).any():
        raise RuntimeError("cross-swap merge duplicates system/cell/seed")
    return merged


def _validate_statistic_frame(frame: pd.DataFrame, expected_systems: np.ndarray | None = None) -> None:
    if tuple(frame.columns) != STAT_COLUMNS:
        raise RuntimeError("cross-swap statistic schema changed")
    if frame.duplicated(["system_index", "cell", "seed"]).any():
        raise RuntimeError("cross-swap statistics duplicate system/cell/seed")
    if set(frame.cell.astype(str)) != set(CELLS):
        raise RuntimeError("cross-swap statistics have the wrong cells")
    if set(frame.seed.astype(int)) != set(OPTIMIZATION_SEEDS):
        raise RuntimeError("cross-swap statistics have the wrong optimization seeds")
    if np.any(frame["count"].to_numpy(np.int64) != ROWS_PER_SYSTEM):
        raise RuntimeError("cross-swap statistic key does not contain 36 rows")
    numeric = frame[["sum_loss", "sum_gain", "sum2_loss", "sum2_gain"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("cross-swap statistics contain non-finite float64 values")
    if expected_systems is not None:
        observed = np.sort(frame.system_index.unique().astype(np.int64))
        expected = np.sort(np.asarray(expected_systems, dtype=np.int64))
        if not np.array_equal(observed, expected):
            raise RuntimeError("cross-swap statistics contain the wrong physical systems")
        if len(frame) != len(expected) * len(CELLS) * len(OPTIMIZATION_SEEDS):
            raise RuntimeError("cross-swap statistics do not contain a complete 4x3 matrix")


def _assert_frames_exact(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    left = left.sort_values(["system_index", "cell", "seed"], kind="stable").reset_index(drop=True)
    right = right.sort_values(["system_index", "cell", "seed"], kind="stable").reset_index(drop=True)
    if tuple(left.columns) != tuple(right.columns) or left.shape != right.shape:
        raise RuntimeError(f"{label} schema/shape mismatch")
    for column in left.columns:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        if not np.array_equal(a, b):
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                distance = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                raise RuntimeError(f"{label} is not array-exact in {column}; max_abs={distance}")
            raise RuntimeError(f"{label} is not array-exact in {column}")


def _existing_tt_ss(root: Path, cfg: dict, base_seed: int, systems: np.ndarray) -> pd.DataFrame:
    _, existing = _existing_statistics(root, cfg, base_seed)
    subset = existing[
        existing.system_index.astype(int).isin(set(map(int, systems)))
        & existing.arm.astype(str).isin(("true", "shuffled"))
    ].copy()
    subset["cell"] = subset.arm.map({"true": "TT", "shuffled": "SS"})
    return subset[list(STAT_COLUMNS)].sort_values(["system_index", "cell", "seed"]).reset_index(drop=True)


def _assert_historical_compatible(left: pd.DataFrame, right: pd.DataFrame,
                                  label: str, tolerance: float) -> dict:
    left = left.sort_values(["system_index", "cell", "seed"], kind="stable").reset_index(drop=True)
    right = right.sort_values(["system_index", "cell", "seed"], kind="stable").reset_index(drop=True)
    if tuple(left.columns) != tuple(right.columns) or left.shape != right.shape:
        raise RuntimeError(f"{label} schema/shape mismatch")
    for column in ("system_index", "cell", "seed", "count"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"{label} has incompatible keys/counts in {column}")
    maxima = {}
    for column in HISTORICAL_STATISTIC_COLUMNS:
        observed = left[column].to_numpy(np.float64)
        historical = right[column].to_numpy(np.float64)
        if not np.isfinite(observed).all() or not np.isfinite(historical).all():
            raise RuntimeError(f"{label} has non-finite values in {column}")
        differences = np.abs(observed - historical)
        maximum = float(np.max(differences))
        if np.any(differences > float(tolerance)):
            raise RuntimeError(
                f"{label} exceeds frozen historical tolerance in {column}; "
                f"max_abs={maximum}; tolerance={float(tolerance)}"
            )
        maxima[column] = {"max_abs": maximum}
    return {
        "status": "TT_SS_HISTORICAL_CSV_COMPATIBLE",
        "rows": int(len(left)),
        "absolute_tolerance": float(tolerance),
        "relative_tolerance": 0.0,
        "columns": maxima,
    }


def _verify_historical_compatibility_report(report: dict, cfg: dict) -> None:
    tolerance = float(cfg["numerical_compatibility"]["historical_absolute_tolerance"])
    if (
        not isinstance(report, dict)
        or report.get("status") != "TT_SS_HISTORICAL_CSV_COMPATIBLE"
        or int(report.get("rows", 0)) <= 0
        or float(report.get("absolute_tolerance", -1.0)) != tolerance
        or float(report.get("relative_tolerance", -1.0)) != 0.0
        or set(report.get("columns", {})) != set(HISTORICAL_STATISTIC_COLUMNS)
    ):
        raise RuntimeError("cross-swap historical compatibility report is invalid")
    for column in HISTORICAL_STATISTIC_COLUMNS:
        maximum = float(report["columns"][column].get("max_abs", np.nan))
        if not np.isfinite(maximum) or maximum < 0.0 or maximum > tolerance:
            raise RuntimeError(f"cross-swap historical compatibility report is invalid in {column}")


def validate_tt_ss_historical_compatible(root: Path, cfg: dict, base_seed: int,
                                         stats: pd.DataFrame) -> dict:
    systems = np.sort(stats.system_index.unique().astype(np.int64))
    observed = stats[stats.cell.isin(("TT", "SS"))][list(STAT_COLUMNS)].copy()
    expected = _existing_tt_ss(root, cfg, base_seed, systems)
    report = _assert_historical_compatible(
        observed,
        expected,
        f"base {base_seed} TT/SS historical positive control",
        float(cfg["numerical_compatibility"]["historical_absolute_tolerance"]),
    )
    _verify_historical_compatibility_report(report, cfg)
    return report


def _aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows[["loss", "gain"]].isna().any().any():
        raise RuntimeError("cross-swap rows contain missing metrics")
    if not np.isfinite(rows[["loss", "gain"]].to_numpy(np.float64)).all():
        raise RuntimeError("cross-swap rows contain non-finite metrics")
    # Reproduce the completed formal evaluator's row-order contract before
    # float64 reduction.  Group membership is the same without this sort, but
    # a different append order can move a sum by one ULP and would defeat the
    # frozen TT/SS positive-control parity check.
    rows = rows.sort_values(
        ["system_index", "row_index", "cell", "seed"], kind="stable",
    ).reset_index(drop=True)
    result = rows.groupby(["system_index", "cell", "seed"], sort=True, as_index=False).agg(
        count=("gain", "size"),
        sum_loss=("loss", "sum"), sum_gain=("gain", "sum"),
        sum2_loss=("loss", lambda value: float(np.sum(value.to_numpy(np.float64) ** 2))),
        sum2_gain=("gain", lambda value: float(np.sum(value.to_numpy(np.float64) ** 2))),
    )
    for column in ("sum_loss", "sum_gain", "sum2_loss", "sum2_gain"):
        result[column] = result[column].astype(np.float64)
    return result[list(STAT_COLUMNS)].sort_values(["system_index", "cell", "seed"]).reset_index(drop=True)


def _whole_system_positions(cache: dict[str, np.ndarray], systems: np.ndarray) -> np.ndarray:
    systems = np.sort(np.asarray(systems, dtype=np.int64))
    positions = np.flatnonzero(np.isin(cache["system_index"], systems))
    if not len(positions):
        raise RuntimeError("cross-swap receiver population is empty")
    observed, counts = np.unique(cache["system_index"][positions], return_counts=True)
    if not np.array_equal(observed, systems) or np.any(counts != ROWS_PER_SYSTEM):
        raise RuntimeError("cross-swap execution must own complete 36-row systems")
    return positions


def _load_adapter(root: Path, checkpoint: Path, device: torch.device) -> e3.DeltaGatedAdapter:
    adapter = e3.DeltaGatedAdapter(64, 64).to(device)
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    adapter.eval()
    return adapter


@torch.no_grad()
def evaluate_positions(root: Path, config_path: Path, base_seed: int,
                       systems: np.ndarray, device_name: str) -> tuple[pd.DataFrame, dict]:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(config_path)
    _, upstream_cfg = _upstream_config(root, cfg)
    _verify_freeze(root, config_path, cfg, upstream_cfg)
    base._require_remote(root, upstream_cfg, device_name=device_name)
    if int(base_seed) not in BASE_SEEDS:
        raise ValueError("cross-swap base seed is outside the frozen schedule")
    cache, cache_receipt = upstream._merged_cache(root, upstream_cfg, "articulated", base_seed, "formal")
    positions = _whole_system_positions(cache, systems)
    arrays = load_arrays(root / upstream_cfg["environments"]["articulated"]["train_arrays"])
    device = torch.device(device_name)
    # Match the completed masked-GRU execution contract.  The seed affects
    # only transient module initialization before frozen states are loaded.
    base._configure_training(
        int(cfg["inference"]["bootstrap_seed"]) + int(base_seed),
        device,
        int(cfg["execution"]["threads_per_worker"]),
    )
    model, _, before = upstream._model(root, upstream_cfg, "articulated", base_seed, arrays, device)
    table = upstream._formal_adapter_table(root, upstream_cfg, base_seed)
    choices = {(str(row.arm), int(row.seed)): root / row.checkpoint for row in table.itertuples(index=False)}
    expected_choices = {(arm, seed) for arm in ("true", "shuffled") for seed in OPTIMIZATION_SEEDS}
    if set(choices) != expected_choices:
        raise RuntimeError("cross-swap adapter checkpoint set changed")
    local = {name: value[positions] for name, value in cache.items()}
    parity_path = upstream._formal_eval_root(root, upstream_cfg, base_seed) / "FORMAL_PARITY.json"
    upstream_parity = json.loads(parity_path.read_text())
    rows: list[dict] = []
    shuffle_hashes: dict[str, str] = {}
    batch_size = int(upstream_cfg["adapter"]["batch_size"])

    for seed in OPTIMIZATION_SEEDS:
        donor = routing.cross_system_cell_permutation(cache, seed)
        donor_hash = hashlib.sha256(donor.astype("<i8").tobytes()).hexdigest()
        if donor_hash != upstream_parity["shuffle_map_hashes"][str(seed)]:
            raise RuntimeError(f"cross-swap donor hash changed for seed {seed}")
        shuffle_hashes[str(seed)] = donor_hash
        deltas = {
            "self": local["delta_persistent"],
            "shuffled": cache["delta_persistent"][donor[positions]],
        }
        adapters = {
            "true": _load_adapter(root, choices[("true", seed)], device),
            "shuffled": _load_adapter(root, choices[("shuffled", seed)], device),
        }
        specifications = (
            ("TT", adapters["true"], deltas["self"]),
            ("TS", adapters["true"], deltas["shuffled"]),
            ("ST", adapters["shuffled"], deltas["self"]),
            ("SS", adapters["shuffled"], deltas["shuffled"]),
        )
        for cell, adapter, delta_input in specifications:
            prediction = np.empty_like(local["prediction_original"], dtype=np.float32)
            for system in np.sort(np.unique(local["system_index"])):
                system_positions = np.flatnonzero(local["system_index"] == system)
                for start in range(0, len(system_positions), batch_size):
                    index = system_positions[start:start + batch_size]
                    delta = torch.from_numpy(delta_input[index]).to(device)
                    query = torch.from_numpy(local["query_embedding"][index]).to(device)
                    latent = torch.from_numpy(local["predicted_latent_full"][index]).to(device)
                    prediction[index] = model.target_decoder(latent + adapter(delta, query)).cpu().numpy()
            metrics = routing._row_metrics(
                prediction, local["prediction_anchor"], local["normalized_target"], None,
            )
            for index in range(len(positions)):
                rows.append({
                    "row_index": int(local["row_index"][index]),
                    "system_index": int(local["system_index"][index]),
                    "cell": cell, "seed": int(seed),
                    "loss": float(metrics["loss"][index]),
                    "gain": float(metrics["gain"][index]),
                })
    if routing.original_module_hashes(model) != before:
        raise RuntimeError("masked-GRU base learner changed during cross-swap forward")
    stats = _aggregate_rows(pd.DataFrame(rows))
    historical_report = validate_tt_ss_historical_compatible(root, cfg, base_seed, stats)
    metadata = {
        "base_seed": int(base_seed), "systems": list(map(int, np.sort(systems))),
        "formal_cache_sha256": cache_receipt["cache_sha256"],
        "upstream_parity_sha256": sha256(parity_path),
        "shuffle_map_hashes": shuffle_hashes,
        "base_module_hashes": before,
        "tt_ss_historical_csv_compatible": historical_report,
    }
    return stats, metadata


def _same_runtime_upstream_tt_ss(root: Path, upstream_cfg: dict, base_seed: int,
                                 cache: dict[str, np.ndarray], systems: np.ndarray,
                                 device_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    positions = _whole_system_positions(cache, systems)
    upstream_stats, shuffle_hashes = upstream._evaluate_cache_positions(
        root, upstream_cfg, base_seed, cache, positions, torch.device(device_name),
    )
    expected = upstream_stats[
        upstream_stats.arm.astype(str).isin(("true", "shuffled"))
    ].copy()
    expected["cell"] = expected.arm.map({"true": "TT", "shuffled": "SS"})
    expected = expected[list(STAT_COLUMNS)].sort_values(
        ["system_index", "cell", "seed"], kind="stable",
    ).reset_index(drop=True)
    return expected, shuffle_hashes


def parity(root: Path, config_path: Path, base_seed: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(config_path)
    _, upstream_cfg = _upstream_config(root, cfg)
    _verify_freeze(root, config_path, cfg, upstream_cfg)
    cache, _ = upstream._merged_cache(root, upstream_cfg, "articulated", base_seed, "formal")
    available = np.sort(np.unique(cache["system_index"]).astype(np.int64))
    chosen = np.asarray([available[available % SHARDS == shard][0] for shard in range(SHARDS)], dtype=np.int64)
    direct, direct_meta = evaluate_positions(root, config_path, base_seed, chosen, device_name)
    pieces = []
    for shard in range(SHARDS):
        piece, piece_meta = evaluate_positions(
            root, config_path, base_seed, chosen[chosen % SHARDS == shard], device_name,
        )
        if piece_meta["shuffle_map_hashes"] != direct_meta["shuffle_map_hashes"]:
            raise RuntimeError("cross-swap parity donor hashes differ")
        pieces.append(piece)
    merged = merge_statistic_parts(pieces)
    _assert_frames_exact(direct, merged, "cross-swap one-task/two-shard parity")
    same_runtime, same_runtime_hashes = _same_runtime_upstream_tt_ss(
        root, upstream_cfg, base_seed, cache, chosen, device_name,
    )
    observed_tt_ss = direct[direct.cell.isin(("TT", "SS"))][list(STAT_COLUMNS)].copy()
    _assert_frames_exact(
        observed_tt_ss,
        same_runtime,
        f"base {base_seed} TT/SS same-runtime upstream recomputation",
    )
    if same_runtime_hashes != direct_meta["shuffle_map_hashes"]:
        raise RuntimeError("cross-swap same-runtime upstream donor hashes differ")
    target = _output(root, cfg) / f"parity/articulated_s{base_seed}/PARITY.json"
    if target.exists():
        raise RuntimeError("cross-swap parity receipt already exists")
    receipt = {
        "schema_version": "1.0", "status": STATUS_PARITY,
        "base_seed": int(base_seed), "systems": chosen.tolist(),
        "cells": list(CELLS), "optimization_seeds": list(OPTIMIZATION_SEEDS),
        "fixed_merge_order": [0, 1], "statistics_dtype": "float64",
        "content_sha256": canonical_frame_sha(direct),
        "shuffle_map_hashes": direct_meta["shuffle_map_hashes"],
        "tt_ss_same_runtime_upstream_exact": True,
        "tt_ss_historical_csv_compatible": direct_meta["tt_ss_historical_csv_compatible"],
        "implementation_freeze_sha256": sha256(_output(root, cfg) / "IMPLEMENTATION_FROZEN.json"),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_parity(root: Path, cfg: dict, base_seed: int) -> tuple[Path, dict]:
    if int(base_seed) not in BASE_SEEDS:
        raise ValueError("cross-swap parity base seed is outside the frozen schedule")
    path = _output(root, cfg) / f"parity/articulated_s{base_seed}/PARITY.json"
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != "1.0" or receipt.get("status") != STATUS_PARITY:
        raise RuntimeError("cross-swap parity receipt is missing or invalid")
    if int(receipt.get("base_seed", -1)) != int(base_seed):
        raise RuntimeError("cross-swap parity receipt has the wrong base seed")
    if receipt.get("cells") != list(CELLS):
        raise RuntimeError("cross-swap parity receipt has the wrong cells")
    if receipt.get("optimization_seeds") != list(OPTIMIZATION_SEEDS):
        raise RuntimeError("cross-swap parity receipt has the wrong optimization seeds")
    if receipt.get("fixed_merge_order") != [0, 1]:
        raise RuntimeError("cross-swap parity receipt has the wrong merge order")
    if receipt.get("statistics_dtype") != "float64":
        raise RuntimeError("cross-swap parity receipt has the wrong statistics dtype")
    content_sha = receipt.get("content_sha256")
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise RuntimeError("cross-swap parity receipt lacks a content hash")
    systems = receipt.get("systems")
    if (
        not isinstance(systems, list) or len(systems) != SHARDS
        or {int(system) % SHARDS for system in systems} != set(range(SHARDS))
    ):
        raise RuntimeError("cross-swap parity receipt has the wrong system panel")
    shuffle_hashes = receipt.get("shuffle_map_hashes")
    if (
        not isinstance(shuffle_hashes, dict)
        or set(shuffle_hashes) != {str(seed) for seed in OPTIMIZATION_SEEDS}
        or any(not isinstance(value, str) or len(value) != 64 for value in shuffle_hashes.values())
    ):
        raise RuntimeError("cross-swap parity receipt has invalid shuffle hashes")
    if receipt.get("tt_ss_same_runtime_upstream_exact") is not True:
        raise RuntimeError("cross-swap parity lacks same-runtime TT/SS exact validation")
    _verify_historical_compatibility_report(receipt.get("tt_ss_historical_csv_compatible"), cfg)
    if receipt.get("implementation_freeze_sha256") != sha256(_output(root, cfg) / "IMPLEMENTATION_FROZEN.json"):
        raise RuntimeError("cross-swap parity receipt has a stale implementation freeze")
    return path, receipt


def _verify_shard_receipt(root: Path, cfg: dict, base_seed: int, shard: int,
                          parity_path: Path, parity_receipt: dict) -> tuple[Path, Path, pd.DataFrame]:
    out = _output(root, cfg) / f"formal/articulated_s{base_seed}"
    receipt_path = out / f"receipt_shard_{shard:02d}_of_{SHARDS:02d}.json"
    expected_stats_path = out / f"stats_shard_{shard:02d}_of_{SHARDS:02d}.csv.gz"
    receipt = json.loads(receipt_path.read_text())
    expected_relative = _relative(root, expected_stats_path)
    identity = {
        "schema_version": "1.0",
        "status": STATUS_SHARD,
        "base_seed": int(base_seed),
        "shard_index": int(shard),
        "shard_count": SHARDS,
        "systems": SYSTEMS // SHARDS,
        "statistics": expected_relative,
    }
    for key, expected in identity.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"cross-swap shard receipt has invalid {key}")
    if receipt.get("statistics_sha256") != sha256(expected_stats_path):
        raise RuntimeError("cross-swap shard statistics hash changed")
    part = _read_csv(expected_stats_path)
    if tuple(part.columns) != STAT_COLUMNS:
        raise RuntimeError("cross-swap shard statistics schema changed")
    if receipt.get("semantic_sha256") != canonical_frame_sha(part):
        raise RuntimeError("cross-swap shard semantic hash changed")
    if receipt.get("parity_sha256") != sha256(parity_path):
        raise RuntimeError("cross-swap shard parity binding changed")
    if receipt.get("shuffle_map_hashes") != parity_receipt["shuffle_map_hashes"]:
        raise RuntimeError("cross-swap shard donor hashes differ from parity")
    historical_report = receipt.get("tt_ss_historical_csv_compatible")
    _verify_historical_compatibility_report(historical_report, cfg)
    expected_rows = (SYSTEMS // SHARDS) * len(CELLS) * len(OPTIMIZATION_SEEDS)
    if len(part) != expected_rows or part.system_index.nunique() != SYSTEMS // SHARDS:
        raise RuntimeError("cross-swap shard statistics have the wrong population")
    if part.duplicated(["system_index", "cell", "seed"]).any():
        raise RuntimeError("cross-swap shard statistics duplicate a key")
    for column in ("system_index", "seed", "count"):
        values = part[column].to_numpy()
        if not np.isfinite(values.astype(np.float64)).all() or not np.array_equal(values, values.astype(np.int64)):
            raise RuntimeError(f"cross-swap shard statistics contain non-integer {column}")
    if np.any(part.system_index.to_numpy() % SHARDS != int(shard)):
        raise RuntimeError("cross-swap shard violates modulo ownership")
    if np.any(part["count"].to_numpy() != ROWS_PER_SYSTEM):
        raise RuntimeError("cross-swap shard keys do not contain exactly 36 rows")
    expected_keys = {(cell, seed) for cell in CELLS for seed in OPTIMIZATION_SEEDS}
    for system, group in part.groupby("system_index", sort=False):
        if set(zip(group.cell.astype(str), group.seed.astype(int))) != expected_keys:
            raise RuntimeError(f"cross-swap shard system {system} lacks the frozen 4x3 keys")
    numeric = part[["sum_loss", "sum_gain", "sum2_loss", "sum2_gain"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("cross-swap shard statistics contain non-finite values")
    expected_historical_report = validate_tt_ss_historical_compatible(root, cfg, base_seed, part)
    if historical_report != expected_historical_report:
        raise RuntimeError("cross-swap shard historical compatibility report changed")
    return receipt_path, expected_stats_path, part


def evaluate_shard(root: Path, config_path: Path, base_seed: int,
                   shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(config_path)
    _, upstream_cfg = _upstream_config(root, cfg)
    _verify_freeze(root, config_path, cfg, upstream_cfg)
    if shard_index not in range(SHARDS):
        raise ValueError("cross-swap uses exactly two shards")
    parity_path, parity_receipt = _verify_parity(root, cfg, base_seed)
    cache, _ = upstream._merged_cache(root, upstream_cfg, "articulated", base_seed, "formal")
    systems = np.sort(np.unique(cache["system_index"]).astype(np.int64))
    owned = systems[systems % SHARDS == shard_index]
    if len(owned) != SYSTEMS // SHARDS:
        raise RuntimeError("cross-swap shard does not own exactly 256 systems")
    stats, metadata = evaluate_positions(root, config_path, base_seed, owned, device_name)
    if metadata["shuffle_map_hashes"] != parity_receipt["shuffle_map_hashes"]:
        raise RuntimeError("cross-swap shard donor hashes differ from parity")
    if len(stats) != len(owned) * len(CELLS) * len(OPTIMIZATION_SEEDS):
        raise RuntimeError("cross-swap shard statistic row count is wrong")
    if np.any(stats["count"].to_numpy(np.int64) != ROWS_PER_SYSTEM):
        raise RuntimeError("cross-swap shard key does not contain exactly 36 rows")
    out = _output(root, cfg) / f"formal/articulated_s{base_seed}"
    stats_path = out / f"stats_shard_{shard_index:02d}_of_{SHARDS:02d}.csv.gz"
    receipt_path = out / f"receipt_shard_{shard_index:02d}_of_{SHARDS:02d}.json"
    if stats_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable cross-swap shard output already exists")
    _atomic_csv_gz(stats_path, stats)
    receipt = {
        "schema_version": "1.0", "status": STATUS_SHARD,
        "base_seed": int(base_seed), "shard_index": int(shard_index),
        "shard_count": SHARDS, "systems": int(len(owned)),
        "statistics": _relative(root, stats_path),
        "statistics_sha256": sha256(stats_path),
        "semantic_sha256": canonical_frame_sha(stats),
        "parity_sha256": sha256(parity_path),
        "shuffle_map_hashes": metadata["shuffle_map_hashes"],
        "tt_ss_historical_csv_compatible": metadata["tt_ss_historical_csv_compatible"],
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def system_estimands(stats: pd.DataFrame) -> pd.DataFrame:
    required_keys = {(cell, seed) for cell in CELLS for seed in OPTIMIZATION_SEEDS}
    rows = []
    for system, group in stats.groupby("system_index", sort=True):
        if set(zip(group.cell.astype(str), group.seed.astype(int))) != required_keys:
            raise RuntimeError(f"cross-swap system {system} lacks a complete 4x3 cell matrix")
        group = group.copy()
        group["mean_gain"] = group.sum_gain.astype(np.float64) / group["count"].astype(np.float64)
        gains = group.groupby("cell", sort=True).mean_gain.mean().to_dict()
        row = {"system_index": int(system), **{f"G_{cell}": float(gains[cell]) for cell in CELLS}}
        row["I_true_train"] = row["G_TT"] - row["G_TS"]
        row["I_shuffle_train"] = row["G_ST"] - row["G_SS"]
        row["W_self"] = row["G_TT"] - row["G_ST"]
        row["W_shuffle"] = row["G_TS"] - row["G_SS"]
        row["E_input"] = 0.5 * (row["I_true_train"] + row["I_shuffle_train"])
        row["E_interaction"] = row["I_true_train"] - row["I_shuffle_train"]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("system_index").reset_index(drop=True)
    if result.system_index.duplicated().any() or not np.isfinite(result.drop(columns="system_index")).all().all():
        raise RuntimeError("cross-swap system estimands are invalid")
    return result


def common_bootstrap_summaries(frame: pd.DataFrame, replicates: int, seed: int) -> dict:
    if frame.system_index.duplicated().any() or not len(frame):
        raise RuntimeError("bootstrap requires one row per physical system")
    columns = [column for column in frame.columns if column != "system_index"]
    values = frame[columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("bootstrap values are non-finite")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(frame), size=(int(replicates), len(frame)))
    samples = values[indices].mean(axis=1)
    result = {}
    for position, column in enumerate(columns):
        result[column] = {
            "estimate": float(values[:, position].mean()),
            "ci_low": float(np.quantile(samples[:, position], 0.025)),
            "ci_high": float(np.quantile(samples[:, position], 0.975)),
            "systems": int(len(frame)), "replicates": int(replicates), "seed": int(seed),
        }
    return result


def merge_base(root: Path, config_path: Path, base_seed: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(config_path)
    _, upstream_cfg = _upstream_config(root, cfg)
    _verify_freeze(root, config_path, cfg, upstream_cfg)
    parity_path, parity_receipt = _verify_parity(root, cfg, base_seed)
    parts = []
    receipts = []
    out = _output(root, cfg) / f"formal/articulated_s{base_seed}"
    for shard in cfg["design"]["fixed_merge_order"]:
        receipt_path, _, part = _verify_shard_receipt(
            root, cfg, base_seed, int(shard), parity_path, parity_receipt,
        )
        parts.append(part)
        receipts.append({"path": _relative(root, receipt_path), "sha256": sha256(receipt_path)})
    stats = merge_statistic_parts(parts)
    if stats.system_index.nunique() != SYSTEMS or len(stats) != SYSTEMS * len(CELLS) * len(OPTIMIZATION_SEEDS):
        raise RuntimeError("cross-swap merged population is incomplete")
    if np.any(stats["count"].to_numpy(np.int64) != ROWS_PER_SYSTEM):
        raise RuntimeError("cross-swap merged keys do not contain 36 rows")
    historical_report = validate_tt_ss_historical_compatible(root, cfg, base_seed, stats)
    system_frame = system_estimands(stats)
    if len(system_frame) != SYSTEMS:
        raise RuntimeError("cross-swap system estimand population is incomplete")
    summaries = common_bootstrap_summaries(
        system_frame, cfg["inference"]["bootstrap_replicates"], cfg["inference"]["bootstrap_seed"],
    )
    stats_path = out / "merged_sufficient_statistics.csv.gz"
    systems_path = out / "system_estimands.csv.gz"
    result_path = out / "FINAL_RESULT.json"
    if any(path.exists() for path in (stats_path, systems_path, result_path)):
        raise RuntimeError("immutable cross-swap base result already exists")
    _atomic_csv_gz(stats_path, stats)
    _atomic_csv_gz(systems_path, system_frame)
    result = {
        "schema_version": "1.0", "status": STATUS_BASE,
        "evidence_identity": "post_hoc_supporting_existing_formal_systems",
        "base_seed": int(base_seed), "systems": SYSTEMS,
        "scientific_unit": "physical_system",
        "optimization_seeds_are_scientific_samples": False,
        "binary_success_classification": None,
        "primary_diagnostic": {"name": "E_input", **summaries["E_input"]},
        "key_secondary": {"name": "E_interaction", **summaries["E_interaction"]},
        "descriptive": {
            name: summaries[name]
            for name in ("I_true_train", "I_shuffle_train", "W_self", "W_shuffle", "G_TT", "G_TS", "G_ST", "G_SS")
        },
        "same_bootstrap_resamples_for_all_estimands": True,
        "bootstrap_seed": int(cfg["inference"]["bootstrap_seed"]),
        "bootstrap_replicates": int(cfg["inference"]["bootstrap_replicates"]),
        "fixed_merge_order": [0, 1], "statistics_dtype": "float64",
        "tt_ss_same_runtime_upstream_exact": True,
        "tt_ss_historical_csv_compatible": historical_report,
        "parity_sha256": sha256(parity_path), "parity_status": parity_receipt["status"],
        "shard_receipts": receipts,
        "merged_statistics": _relative(root, stats_path),
        "merged_statistics_sha256": sha256(stats_path),
        "merged_statistics_semantic_sha256": canonical_frame_sha(stats),
        "system_estimands": _relative(root, systems_path),
        "system_estimands_sha256": sha256(systems_path),
        "upstream_classification_unchanged": "MASKED_GRU_COMPETENT_ARCHITECTURE_CONTRADICTION",
        "model_training": False, "checkpoint_selection": False,
    }
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(config_path)
    _, upstream_cfg = _upstream_config(root, cfg)
    _verify_freeze(root, config_path, cfg, upstream_cfg)
    base_results, frames = {}, []
    for base_seed in BASE_SEEDS:
        out = _output(root, cfg) / f"formal/articulated_s{base_seed}"
        result_path = out / "FINAL_RESULT.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != STATUS_BASE:
            raise RuntimeError("cross-swap base result is missing or invalid")
        systems_path = root / result["system_estimands"]
        if result.get("system_estimands_sha256") != sha256(systems_path):
            raise RuntimeError("cross-swap system estimands changed")
        frame = _read_csv(systems_path).sort_values("system_index").reset_index(drop=True)
        frames.append(frame)
        base_results[str(base_seed)] = {
            "path": _relative(root, result_path), "sha256": sha256(result_path),
            "primary_diagnostic": result["primary_diagnostic"],
            "key_secondary": result["key_secondary"],
        }
    if not np.array_equal(frames[0].system_index.to_numpy(), frames[1].system_index.to_numpy()):
        raise RuntimeError("cross-swap base panels have different physical systems")
    numeric_columns = [column for column in frames[0].columns if column != "system_index"]
    fixed_panel = frames[0].copy()
    fixed_panel[numeric_columns] = 0.5 * (
        frames[0][numeric_columns].to_numpy(np.float64)
        + frames[1][numeric_columns].to_numpy(np.float64)
    )
    fixed_summaries = common_bootstrap_summaries(
        fixed_panel, cfg["inference"]["bootstrap_replicates"], cfg["inference"]["bootstrap_seed"],
    )
    target = _output(root, cfg) / "FINAL_SUPPORTING_RESULT.json"
    if target.exists():
        raise RuntimeError("immutable cross-swap supporting summary already exists")
    result = {
        "schema_version": "1.0", "status": STATUS_FINAL,
        "completed_at_unix": time.time(),
        "evidence_identity": "post_hoc_supporting_existing_formal_systems",
        "base_results": base_results,
        "fixed_panel_summary": {
            "scope": "descriptive_average_of_exactly_two_frozen_base_learners",
            "base_seeds_are_scientific_samples": False,
            "binary_success_classification": None,
            "primary_diagnostic": {"name": "E_input", **fixed_summaries["E_input"]},
            "key_secondary": {"name": "E_interaction", **fixed_summaries["E_interaction"]},
        },
        "registered_hierarchy": ["E_input", "E_interaction", "descriptive_simple_effects_and_cell_means"],
        "same_bootstrap_resamples_for_all_estimands": True,
        "model_training": False, "checkpoint_selection": False,
        "upstream_classification_unchanged": "MASKED_GRU_COMPETENT_ARCHITECTURE_CONTRADICTION",
        "claim_boundary": (
            "Same-weight input-role diagnosis on two existing masked-GRU learners and the existing formal systems only; "
            "not prospective, not an optimal-adapter test, and not architecture-population evidence."
        ),
    }
    _atomic_text(target, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test() -> dict:
    rows = []
    for system in range(4):
        gains = {"TT": 4.0 + system, "TS": 2.0 + system, "ST": 3.0 + system, "SS": 2.0 + system}
        for cell, gain in gains.items():
            for seed in OPTIMIZATION_SEEDS:
                rows.append({
                    "system_index": system, "cell": cell, "seed": seed,
                    "count": ROWS_PER_SYSTEM, "sum_loss": 0.0,
                    "sum_gain": gain * ROWS_PER_SYSTEM,
                    "sum2_loss": 0.0, "sum2_gain": gain * gain * ROWS_PER_SYSTEM,
                })
    frame = pd.DataFrame(rows)[list(STAT_COLUMNS)]
    estimands = system_estimands(frame)
    if not np.allclose(estimands.E_input, 1.5) or not np.allclose(estimands.E_interaction, 1.0):
        raise RuntimeError("cross-swap factorial estimand self-test failed")
    first = common_bootstrap_summaries(estimands, 100, 86211)
    second = common_bootstrap_summaries(estimands, 100, 86211)
    if first != second:
        raise RuntimeError("cross-swap common bootstrap is not deterministic")
    return {
        "status": "MASKED_GRU_CROSS_SWAP_STATIC_PASS",
        "cells": list(CELLS), "base_seeds": list(BASE_SEEDS),
        "optimization_seeds": list(OPTIMIZATION_SEEDS),
        "primary": "E_input", "secondary": "E_interaction",
        "model_training": False, "formal_outcome_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-implementation")
    freeze.add_argument("--authorize-reviewed-code", action="store_true")
    parity_parser = sub.add_parser("parity")
    parity_parser.add_argument("base_seed", type=int)
    parity_parser.add_argument("--device", required=True)
    shard = sub.add_parser("evaluate-shard")
    shard.add_argument("base_seed", type=int)
    shard.add_argument("--shard-index", required=True, type=int)
    shard.add_argument("--device", required=True)
    merge = sub.add_parser("merge-base")
    merge.add_argument("base_seed", type=int)
    sub.add_parser("summarize")
    sub.add_parser("self-test")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze-implementation":
        result = freeze_implementation(args.root, args.config, args.authorize_reviewed_code)
    elif args.command == "parity":
        result = parity(args.root, args.config, args.base_seed, args.device)
    elif args.command == "evaluate-shard":
        result = evaluate_shard(args.root, args.config, args.base_seed, args.shard_index, args.device)
    elif args.command == "merge-base":
        result = merge_base(args.root, args.config, args.base_seed)
    elif args.command == "summarize":
        result = summarize(args.root, args.config)
    else:
        result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
