"""Final Paper-C same-checkpoint prospective confirmation.

The state machine is deliberately narrow:
authorize -> freeze population and exact donor -> materialize physical rows ->
extract frozen features -> prediction-only parity -> evaluate -> merge once.
No command in this module trains or selects a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
from datetime import datetime
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from paper_c.extension import prospective_models
from paper_c.extension.prospective_packet import ROW_KEY


CANONICAL_CONFIG = "configs/d1a_same_checkpoint_prospective_v1.json"
CANONICAL_PROTOCOL = "protocol/PAPER_C_D1A_SAME_CHECKPOINT_PROSPECTIVE_V1.md"
CANONICAL_OUTPUT = "runs/diagnostics/transport_alignment_repair_v1/d1a_prospective_block_v2"
IMPLEMENTATION = "code/paper_c/extension/d1a_same_checkpoint_prospective.py"
TEST = "tests/unit/test_d1a_same_checkpoint_prospective.py"
SYSTEMS = 512
ROWS_PER_SYSTEM = 36
ROWS = SYSTEMS * ROWS_PER_SYSTEM
SHARDS = 2
CANONICAL_SEEDS = (86101, 86103, 86107)
GRU_BASE_SEEDS = (64101, 64103)
GRU_SEEDS = (86101, 86103, 86107)
PARITY_TOLERANCE = 2e-6
PARITY_REPAIR = "authorization/ENGINEERING_REPAIR_PARITY_FLOAT32.json"
PARITY_AMENDMENT = "protocol/PAPER_C_D1A_PARITY_FLOAT32_ENGINEERING_AMENDMENT_V1.md"
PARITY_REPAIR_TEST = "tests/unit/test_d1a_parity_float32_repair.py"

STATUS_AUTH = "PAPER_C_D1A_DUAL_ARCH_BRANCH_AUTHORIZED"
STATUS_POP = "PAPER_C_D1A_DUAL_ARCH_POPULATION_AND_DONOR_FROZEN"
STATUS_PHYSICAL = "PAPER_C_D1A_PHYSICAL_SHARD_COMPLETE"
STATUS_PHYSICAL_MERGED = "PAPER_C_D1A_PHYSICAL_POPULATION_COMPLETE"
STATUS_FEATURE = "PAPER_C_D1A_FEATURE_SHARD_COMPLETE"
STATUS_FEATURE_MERGED = "PAPER_C_D1A_FEATURES_COMPLETE"
STATUS_PARITY = "PAPER_C_D1A_ONE_TASK_TWO_SHARD_PARITY_PASS"
STATUS_EVAL = "PAPER_C_D1A_EVALUATION_SHARD_COMPLETE"
STATUS_RESULT = "PAPER_C_D1A_FINAL_PROSPECTIVE_CONFIRMATION_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _config(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    expected = (root / CANONICAL_CONFIG).resolve()
    if config_path != expected:
        raise RuntimeError(f"only the canonical config path is permitted: {expected}")
    cfg = json.loads(config_path.read_text())
    if cfg.get("status") != "D1A_DUAL_ARCHITECTURE_PROSPECTIVE_FROZEN":
        raise RuntimeError("final D1A config status is not frozen")
    if cfg.get("canonical_config_path") != CANONICAL_CONFIG or cfg.get("output_root") != CANONICAL_OUTPUT:
        raise RuntimeError("canonical config/output identity changed")
    design = cfg.get("fresh_design", {})
    exact = {"systems": 512, "system_seed": 89101, "context_seed": 89103,
             "queries": 6, "candidates": 6, "candidate_rows": ROWS}
    if any(int(design.get(key, -1)) != value for key, value in exact.items()):
        raise RuntimeError("frozen fresh design changed")
    execution = cfg.get("execution", {})
    required_flags = {
        "physical_shards": 2, "feature_shards": 2, "evaluation_shards": 2,
        "maximum_new_blocks": 1, "parity_systems": 8,
    }
    if any(int(execution.get(key, -1)) != value for key, value in required_flags.items()):
        raise RuntimeError("execution population or sharding changed")
    if execution.get("fixed_merge_order") != [0, 1] or any(
        execution.get(key) is not False for key in ("model_training", "checkpoint_selection", "reference_geometry")
    ):
        raise RuntimeError("final run must forbid training, selection, and reference geometry")
    if tuple(cfg["canonical_learner"]["optimization_seeds"]) != CANONICAL_SEEDS:
        raise RuntimeError("canonical checkpoint population changed")
    if tuple(cfg["masked_gru"]["base_seeds"]) != GRU_BASE_SEEDS or tuple(cfg["masked_gru"]["optimization_seeds"]) != GRU_SEEDS:
        raise RuntimeError("Masked-GRU checkpoint population changed")
    if cfg["donor"].get("must_be_materialized_before_any_target_or_learner_forward") is not True:
        raise RuntimeError("premature-forward barrier was removed")
    return cfg


def _output(root: Path, cfg: Mapping[str, object]) -> Path:
    path = (root / str(cfg["output_root"])).resolve()
    if path != (root / CANONICAL_OUTPUT).resolve():
        raise RuntimeError("alternate output root is forbidden")
    return path


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _identity(theta: np.ndarray, decimals: int) -> str:
    token = "|".join(f"{value:.{decimals}f}" for value in np.asarray(theta, dtype=np.float64))
    return hashlib.sha256(token.encode()).hexdigest()


def _reference_config(root: Path, cfg: Mapping[str, object]) -> dict:
    path = root / str(cfg["reference_config_template"])
    reference = copy.deepcopy(json.loads(path.read_text()))
    reference["formal"]["system_seed"] = int(cfg["fresh_design"]["system_seed"])
    reference["formal"]["context_seed"] = int(cfg["fresh_design"]["context_seed"])
    reference["formal"]["pool_max"] = SYSTEMS
    reference["formal"]["target_systems"] = SYSTEMS
    reference["formal"]["contexts_per_system"] = 1
    return reference


def _branch_files(root: Path, cfg: dict) -> list[Path]:
    return [root / value for key, value in cfg["branch_authorization"].items() if key != "expected"]


def _source_files(root: Path, cfg: dict, gru_cfg: dict) -> list[Path]:
    reference = json.loads((root / cfg["reference_config_template"]).read_text())
    relative = [
        CANONICAL_CONFIG, CANONICAL_PROTOCOL, IMPLEMENTATION, TEST,
        cfg["reference_config_template"], reference["base_config"], reference["s0_receipt"],
        reference["s2_config"], "runs/formal/swimmer_s2r_learner_v1/data/s2_data_receipt.json",
        "runs/formal/swimmer_s2r_learner_v1/models/s2_training_receipt.json",
        "code/paper_c/swimmer/lqa_prospective.py", "code/paper_c/swimmer/lqa_formal.py",
        "code/paper_c/swimmer/lqa_evaluate.py", "code/paper_c/swimmer/model.py",
        "code/paper_c/swimmer/waveforms.py", "code/paper_c/swimmer/s2_data.py",
        "code/paper_c/swimmer/s2_train.py",
        "code/paper_c/extension/prospective_models.py", "code/paper_c/extension/prospective_packet.py",
        "code/paper_c/extension/prospective_packet_runner.py",
        "code/paper_c/stage2/fresh_articulated_prospective.py",
        "code/paper_c/stage2/delta_gated_isolation.py",
        "code/paper_c/stage2/routing_intervention.py",
        "code/paper_c/stage2/masked_gru_cross_architecture.py",
        "code/paper_c/stage2/masked_gru_formal.py",
        "code/paper_c/stage2/masked_gru_cross_swap.py",
        cfg["masked_gru"]["upstream_config"],
        "configs/transport_alignment_d1_v1.json",
        "protocol/PAPER_C_TRANSPORT_INTERVENTION_ALIGNMENT_REPAIR_V1.md",
        "code/paper_c/coupled_sled/learner.py",
        "code/paper_c/coupled_sled/learner_data.py",
        "code/paper_c/coupled_sled/formal_data.py",
    ]
    relative.extend([cfg["canonical_learner"]["checkpoint"], cfg["canonical_learner"]["normalization"]])
    relative.extend(item["path"] for item in cfg["canonical_learner"]["true_checkpoints"])
    relative.extend([cfg["forbidden_system_pools"]["train"], cfg["forbidden_system_pools"]["select"],
                     cfg["forbidden_system_pools"]["e1_fresh_manifest"], cfg["forbidden_system_pools"]["joint_fresh_systems"]])
    relative.extend(gru_cfg["environments"]["articulated"][key] for key in ("train_arrays", "select_arrays"))
    return sorted({(root / item).resolve() for item in relative} | {path.resolve() for path in _branch_files(root, cfg)})


def _verify_result_receipt(result_path: Path, receipt_path: Path, expected: dict) -> tuple[dict, dict]:
    result, receipt = json.loads(result_path.read_text()), json.loads(receipt_path.read_text())
    observed = sha256(result_path); recorded = receipt.get("results_sha256") or receipt.get("result_sha256")
    if not isinstance(recorded, str) or recorded != observed or observed != expected.get("result_sha256"):
        raise RuntimeError(f"result receipt/hash is absent, stale, or not the frozen branch artifact: {result_path}")
    if receipt.get("status") != expected.get("receipt_status") or receipt.get("schema_version") != "1.0":
        raise RuntimeError(f"result receipt status/schema differs from the frozen branch: {receipt_path}")
    if (result.get("status") != expected.get("result_status") or result.get("schema_version") != "1.0"
            or result.get("evidence_identity") != expected.get("evidence_identity")):
        raise RuntimeError(f"result status/evidence identity differs from the frozen branch: {result_path}")
    implementation = receipt.get("implementation_sha256")
    if not isinstance(implementation, str) or len(implementation) != 64:
        raise RuntimeError("upstream receipt lacks a producer implementation binding")
    protocol_binding = receipt.get("protocol_sha256") or receipt.get("protocol_bindings")
    if not protocol_binding:
        raise RuntimeError("upstream receipt lacks a producer protocol binding")
    return result, receipt


def _masked_gru_artifacts(root: Path, cfg: dict, gru_cfg: dict) -> tuple[list[dict], list[Path]]:
    from paper_c.stage2 import masked_gru_formal as gru

    records, paths = [], []
    barrier = root / gru_cfg["output_root"] / "formal_pipeline/selection/ALL_12_ADAPTERS_AUTHENTICATED.json"
    formal_barrier = root / gru_cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"
    one_shot = root / gru_cfg["output_root"] / "formal_pipeline/authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"
    for path in (barrier, formal_barrier, one_shot):
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    for base_seed in GRU_BASE_SEEDS:
        checkpoint, competence = gru._verify_base(root, gru_cfg, "articulated", base_seed)
        competence_path = checkpoint.parent / "BASE_COMPETENCE_RECEIPT.json"
        normalization = checkpoint.parent / "train_only_normalization.npz"
        paths.extend([checkpoint, competence_path, normalization])
        table = gru._formal_adapter_table(root, gru_cfg, base_seed)
        true = table[table.arm.astype(str) == "true"].copy()
        if tuple(true.seed.astype(int)) != GRU_SEEDS:
            raise RuntimeError(f"Masked-GRU true checkpoint schedule differs for {base_seed}")
        for row in true.itertuples(index=False):
            checkpoint_path = root / str(row.checkpoint)
            if sha256(checkpoint_path) != str(row.checkpoint_sha256):
                raise RuntimeError("Masked-GRU authenticated adapter hash changed")
            paths.append(checkpoint_path)
            records.append({
                "base_seed": int(base_seed), "optimization_seed": int(row.seed),
                "path": _relative(root, checkpoint_path), "sha256": sha256(checkpoint_path),
                "base_checkpoint": _relative(root, checkpoint), "base_checkpoint_sha256": sha256(checkpoint),
                "base_normalization_sha256": sha256(normalization),
                "competence_status": competence["status"],
            })
    return records, paths


def authorize_and_freeze_design(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    output = _output(root, cfg)
    if output.exists():
        raise FileExistsError("final prospective output root already exists")
    if not cfg.get("calendar_stop") or not cfg["execution"].get("run_identity"):
        raise RuntimeError("Set a new timezone-aware calendar_stop and execution.run_identity before freezing a new run")
    stop = datetime.fromisoformat(str(cfg["calendar_stop"])); now = datetime.now(stop.tzinfo)
    observed_run = os.environ.get("SPRII_RUN_ID")
    _validate_runtime_authorization(observed_run, cfg["execution"]["run_identity"], now, stop)
    branch = cfg["branch_authorization"]
    expected = branch["expected"]
    old, _ = _verify_result_receipt(root / branch["d1a_old_result"], root / branch["d1a_old_receipt"], expected["d1a_old"])
    fresh, _ = _verify_result_receipt(root / branch["d1a_fresh_result"], root / branch["d1a_fresh_receipt"], expected["d1a_fresh"])
    d1b, _ = _verify_result_receipt(root / branch["d1b_fresh_result"], root / branch["d1b_fresh_receipt"], expected["d1b_fresh"])
    old_pass = float(old["summary"]["E_input"]["ci_low"]) > 0
    fresh_pass = float(fresh["summary"]["E_input"]["ci_low"]) > 0
    d1b_a = float(d1b["contrasts"]["C_interaction"]["ci_low"]) > 0
    d1b_b = float(d1b["contrasts"]["C_interaction_specific"]["ci_low"]) > 0
    if not (old_pass and fresh_pass) or (d1b_a and d1b_b):
        raise RuntimeError("frozen D1A-only authorization branch is not satisfied")

    canonical = cfg["canonical_learner"]
    if sha256(root / canonical["checkpoint"]) != canonical["checkpoint_sha256"]:
        raise RuntimeError("canonical base checkpoint changed")
    if sha256(root / canonical["normalization"]) != canonical["normalization_sha256"]:
        raise RuntimeError("canonical normalization changed")
    for item in canonical["true_checkpoints"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError("canonical adapter checkpoint changed")
    gru_cfg = json.loads((root / cfg["masked_gru"]["upstream_config"]).read_text())
    gru_records, gru_paths = _masked_gru_artifacts(root, cfg, gru_cfg)
    sources = _source_files(root, cfg, gru_cfg) + gru_paths
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_hashes = {_relative(root, path): sha256(path) for path in sorted(set(sources))}
    receipt = {
        "schema_version": "2.0", "status": STATUS_AUTH, "host": socket.gethostname(),
        "logical_host": cfg["execution"]["host"], "run_identity": observed_run,
        "config": CANONICAL_CONFIG, "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(root / CANONICAL_PROTOCOL),
        "branch_gate": {"d1a_old_lcb_positive": old_pass, "d1a_fresh_lcb_positive": fresh_pass,
                        "d1b_interaction_lcb_positive": d1b_a,
                        "d1b_interaction_specific_lcb_positive": d1b_b,
                        "selected_branch": "D1A_ONLY_EXISTING_CHECKPOINTS_NO_TRAINING"},
        "canonical_checkpoints": canonical["true_checkpoints"],
        "masked_gru_checkpoints": gru_records,
        "source_hashes": source_hashes,
        "calendar_stop": cfg["calendar_stop"], "authorized_before_calendar_stop": True,
        "population_generated": False, "donor_map_generated": False,
        "simulator_target_generated": False, "learner_forward_run": False,
        "model_training": False, "checkpoint_selection": False,
    }
    _atomic_json(output / "authorization/BRANCH_AUTHORIZATION.json", receipt)
    return receipt


def _validate_runtime_authorization(observed_run: str | None, expected_run: str | int,
                                    now: datetime, stop: datetime) -> None:
    if not expected_run or observed_run != str(expected_run):
        raise RuntimeError("final prospective run must match the frozen run identity")
    if now.tzinfo is None or stop.tzinfo is None:
        raise RuntimeError("calendar authorization requires timezone-aware timestamps")
    if now > stop:
        raise RuntimeError("frozen calendar stop has passed; no final block may be authorized")


def _verify_authorization(root: Path, config_path: Path, cfg: dict) -> dict:
    path = _output(root, cfg) / "authorization/BRANCH_AUTHORIZATION.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_AUTH or receipt.get("config_sha256") != sha256(config_path):
        raise RuntimeError("branch authorization is absent or stale")
    if receipt.get("branch_gate", {}).get("selected_branch") != "D1A_ONLY_EXISTING_CHECKPOINTS_NO_TRAINING":
        raise RuntimeError("wrong intervention branch")
    for relative, expected in receipt["source_hashes"].items():
        observed = sha256(root / relative)
        if observed == expected:
            continue
        if relative not in (IMPLEMENTATION, TEST):
            raise RuntimeError(f"frozen source or artifact changed: {relative}")
        _verify_parity_engineering_repair(root, cfg, relative, expected, observed)
    return receipt


def _verify_parity_engineering_repair(root: Path, cfg: dict, relative: str,
                                      old_hash: str, new_hash: str) -> dict:
    path = _output(root, cfg) / PARITY_REPAIR
    repair = json.loads(path.read_text())
    expected_maxima = {
        "canonical": 9.5367431640625e-7,
        "masked_gru_s64101": 9.5367431640625e-7,
        "masked_gru_s64103": 9.5367431640625e-7,
    }
    if (repair.get("status") != "PAPER_C_D1A_PARITY_FLOAT32_ENGINEERING_REPAIR_AUTHORIZED"
            or repair.get("source_repairs", {}).get(relative) != {"old_sha256": old_hash, "new_sha256": new_hash}
            or repair.get("amendment") != {"path": PARITY_AMENDMENT, "sha256": sha256(root / PARITY_AMENDMENT)}
            or repair.get("repair_test") != {"path": PARITY_REPAIR_TEST, "sha256": sha256(root / PARITY_REPAIR_TEST)}
            or repair.get("scientific_definition_changed") is not False
            or repair.get("target_values_used") is not False
            or repair.get("scientific_contrast_computed") is not False
            or float(repair.get("old_tolerance", -1)) != 1e-10
            or float(repair.get("new_tolerance", -1)) != PARITY_TOLERANCE
            or repair.get("observed_maximum_absolute_prediction_differences") != expected_maxima
            or repair.get("independent_audit") != {"scientific": "GO", "engineering": "GO"}):
        raise RuntimeError("parity engineering repair is absent, stale, or changes science")
    return repair


def _row_manifest(contexts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in contexts.itertuples(index=False):
        for query in range(6):
            for candidate in range(6):
                rows.append({"system_index": int(context.system_index), "realization": int(context.realization),
                             "history_index": int(context.history_index), "query_index": query,
                             "candidate_index": candidate})
    result = prospective_models.validate_rows(pd.DataFrame(rows)).sort_values(list(ROW_KEY)).reset_index(drop=True)
    if len(result) != ROWS or result.system_index.nunique() != SYSTEMS:
        raise RuntimeError("outcome-blind row manifest has wrong coverage")
    return result


def _validate_donor(rows: pd.DataFrame, donor: np.ndarray) -> None:
    rows = prospective_models.validate_rows(rows)
    donor = np.asarray(donor, dtype=np.int64)
    if donor.shape != (len(rows),) or np.any(donor < 0) or np.any(donor >= len(rows)):
        raise RuntimeError("donor map has wrong shape or out-of-range positions")
    if len(np.unique(donor)) != len(rows):
        raise RuntimeError("donor map is not bijective")
    if np.any(rows.system_index.to_numpy()[donor] == rows.system_index.to_numpy()):
        raise RuntimeError("donor map retains a recipient system")
    for name in ("history_index", "query_index", "candidate_index"):
        if not np.array_equal(rows[name].to_numpy()[donor], rows[name].to_numpy()):
            raise RuntimeError(f"donor map changes frozen cell field {name}")


def _validate_exact_shard_systems(system_ids: Sequence[int], shard_index: int) -> None:
    observed = sorted(map(int, system_ids))
    expected = list(range(int(shard_index), SYSTEMS, SHARDS))
    if observed != expected:
        raise RuntimeError("shard has missing, duplicate, or wrong-owner systems")


def _validate_exact_system_union(parts: Sequence[Sequence[int]]) -> None:
    seen: set[int] = set()
    for part in parts:
        current = set(map(int, part))
        if len(current) != len(list(part)):
            raise RuntimeError("system list contains duplicates")
        if seen & current:
            raise RuntimeError("system shards overlap")
        seen |= current
    if seen != set(range(SYSTEMS)):
        raise RuntimeError("system shards omit or add systems")


def _theta_from_manifest(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    theta_columns = sorted([name for name in table if name.startswith("theta_")], key=lambda value: int(value.split("_")[1]))
    if not theta_columns:
        raise RuntimeError(f"historical manifest has no theta columns: {path}")
    return table.drop_duplicates("system_index").sort_values("system_index")[theta_columns].to_numpy(np.float64)


def materialize_population_and_donor(root: Path, config_path: Path) -> dict:
    from paper_c.swimmer.lqa_formal import formal_context_table
    from paper_c.swimmer.lqa_prospective import system_pool

    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    authorization = _verify_authorization(root, config_path, cfg); output = _output(root, cfg)
    reference = _reference_config(root, cfg)
    base = json.loads((root / reference["base_config"]).read_text())
    design = cfg["fresh_design"]
    systems = system_pool(SYSTEMS, int(design["system_seed"]), base["persistent_prior"])
    contexts = formal_context_table(reference)[["system_index", "realization", "history_index"]].copy()
    if len(contexts) != SYSTEMS or contexts.system_index.nunique() != SYSTEMS:
        raise RuntimeError("one-context-per-system freeze failed")
    rows = _row_manifest(contexts)
    donor = prospective_models.coherent_cell_derangement(rows, int(cfg["donor"]["salt"]))
    _validate_donor(rows, donor)

    decimals = int(design["identity_round_decimals"])
    new_ids = {_identity(theta, decimals) for theta in systems}
    if len(new_ids) != SYSTEMS:
        raise RuntimeError("new block contains duplicate physical systems")
    pools = cfg["forbidden_system_pools"]
    historical: dict[str, np.ndarray] = {
        "train": np.load(root / pools["train"]),
        "select": np.load(root / pools["select"]),
        "e1_fresh": _theta_from_manifest(root / pools["e1_fresh_manifest"]),
        "joint_fresh": np.load(root / pools["joint_fresh_systems"]),
        "original_formal": system_pool(int(pools["original_formal_regenerated"]["systems"]),
                                       int(pools["original_formal_regenerated"]["seed"]),
                                       base["persistent_prior"]),
    }
    disjoint = {}
    for name, values in historical.items():
        identities = {_identity(theta, decimals) for theta in np.asarray(values)}
        overlap = new_ids & identities
        if overlap:
            raise RuntimeError(f"new block overlaps historical pool {name}")
        disjoint[name] = {"systems": len(identities), "intersection": 0,
                          "identity_set_sha256": hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest()}

    frozen = output / "frozen"
    _atomic_npz(frozen / "systems.npz", systems=systems.astype(np.float64))
    _atomic_csv(frozen / "contexts.csv.gz", contexts)
    _atomic_csv(frozen / "row_manifest.csv.gz", rows)
    _atomic_npz(frozen / "donor_map.npz", donor_position=donor.astype(np.int64))
    _atomic_json(frozen / "REFERENCE_CONFIG.json", reference)
    receipt = {
        "schema_version": "2.0", "status": STATUS_POP, "systems": SYSTEMS, "rows": ROWS,
        "system_seed": 89101, "context_seed": 89103, "donor_salt": int(cfg["donor"]["salt"]),
        "donor_materialized_before_target_or_forward": True,
        "disjointness": disjoint,
        "hashes": {name: sha256(frozen / name) for name in
                   ("systems.npz", "contexts.csv.gz", "row_manifest.csv.gz", "donor_map.npz", "REFERENCE_CONFIG.json")},
        "authorization_sha256": sha256(output / "authorization/BRANCH_AUTHORIZATION.json"),
        "simulator_target_generated": False, "learner_forward_run": False,
    }
    _atomic_json(frozen / "POPULATION_AND_DONOR_FROZEN.json", receipt)
    return receipt


def _verify_population(root: Path, config_path: Path, cfg: dict) -> dict:
    authorization = _verify_authorization(root, config_path, cfg)
    output = _output(root, cfg); frozen = output / "frozen"
    receipt_path = frozen / "POPULATION_AND_DONOR_FROZEN.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != STATUS_POP or receipt.get("donor_materialized_before_target_or_forward") is not True:
        raise RuntimeError("population/donor barrier is invalid")
    if receipt.get("authorization_sha256") != sha256(output / "authorization/BRANCH_AUTHORIZATION.json"):
        raise RuntimeError("population is not bound to authorization")
    for name, expected in receipt["hashes"].items():
        if sha256(frozen / name) != expected:
            raise RuntimeError(f"frozen population artifact changed: {name}")
    if authorization.get("model_training") is not False:
        raise RuntimeError("authorization permits training unexpectedly")
    return receipt


def _arrays_dict(arrays) -> dict[str, np.ndarray]:
    return {field.name: np.asarray(getattr(arrays, field.name)) for field in fields(arrays)}


def _subset_arrays(arrays, positions: np.ndarray):
    return type(arrays)(**{field.name: np.asarray(getattr(arrays, field.name))[positions] for field in fields(arrays)})


def _save_arrays_atomic(path: Path, arrays) -> None:
    _atomic_npz(path, **_arrays_dict(arrays))


def _load_arrays(path: Path):
    from paper_c.coupled_sled.learner_data import LearnerArrays
    values = _load_npz(path)
    return LearnerArrays(**{field.name: values[field.name] for field in fields(LearnerArrays)})


def _reorder_candidate(arrays, rows: pd.DataFrame, expected: pd.DataFrame):
    observed = rows.copy(); observed["_position"] = np.arange(len(observed), dtype=np.int64)
    joined = expected.merge(observed[[*ROW_KEY, "_position"]], on=list(ROW_KEY), how="left", validate="one_to_one", sort=False)
    if joined._position.isna().any() or len(joined) != len(expected):
        raise RuntimeError("simulator rows do not match the frozen outcome-blind manifest")
    positions = joined._position.to_numpy(np.int64)
    return _subset_arrays(arrays, positions), expected.reset_index(drop=True)


def materialize_physical_shard(root: Path, config_path: Path, shard_index: int) -> dict:
    from paper_c.stage2.fresh_articulated_prospective import _all_candidate_manifest
    from paper_c.swimmer.lqa_evaluate import build_unique_learner_rows

    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    population = _verify_population(root, config_path, cfg)
    if shard_index not in range(SHARDS):
        raise ValueError("physical shard index must be 0 or 1")
    output = _output(root, cfg); frozen = output / "frozen"
    contexts = pd.read_csv(frozen / "contexts.csv.gz")
    contexts = contexts[contexts.system_index.astype(int) % SHARDS == shard_index].copy()
    expected = pd.read_csv(frozen / "row_manifest.csv.gz")
    expected = expected[expected.system_index.astype(int) % SHARDS == shard_index].copy().reset_index(drop=True)
    pairs = _all_candidate_manifest(contexts)
    reference = json.loads((frozen / "REFERENCE_CONFIG.json").read_text())
    baseline_arrays, baseline_rows, candidate_arrays, candidate_rows = build_unique_learner_rows(root, reference, pairs)
    candidate_arrays, candidate_rows = _reorder_candidate(candidate_arrays, candidate_rows, expected)
    if len(baseline_rows) != 256 * 6 or len(candidate_rows) != 256 * ROWS_PER_SYSTEM:
        raise RuntimeError("physical shard coverage is wrong")
    baseline_rows = baseline_rows.copy().reset_index(drop=True)
    baseline_rows.insert(0, "baseline_row_index", np.arange(len(baseline_rows), dtype=np.int64))
    shard = output / f"physical/shard_{shard_index}_of_2"
    _save_arrays_atomic(shard / "baseline_arrays.npz", baseline_arrays)
    _save_arrays_atomic(shard / "candidate_arrays.npz", candidate_arrays)
    _atomic_csv(shard / "baseline_rows.csv.gz", baseline_rows)
    _atomic_csv(shard / "candidate_rows.csv.gz", candidate_rows)
    receipt = {
        "schema_version": "2.0", "status": STATUS_PHYSICAL,
        "shard_index": shard_index, "shard_count": SHARDS, "systems": 256, "candidate_rows": 9216,
        "system_ids": sorted(map(int, candidate_rows.system_index.unique())),
        "hashes": {name: sha256(shard / name) for name in
                   ("baseline_arrays.npz", "candidate_arrays.npz", "baseline_rows.csv.gz", "candidate_rows.csv.gz")},
        "raw_candidate_target_content_sha256": hashlib.sha256(
            np.asarray(candidate_arrays.target, dtype="<f4").tobytes()).hexdigest(),
        "population_receipt_sha256": sha256(frozen / "POPULATION_AND_DONOR_FROZEN.json"),
        "implementation_sha256": sha256(root / IMPLEMENTATION),
    }
    _atomic_json(shard / "SHARD_RECEIPT.json", receipt)
    return receipt


def _verify_physical_shard(root: Path, cfg: dict, shard_index: int) -> tuple[Path, dict]:
    output = _output(root, cfg); shard = output / f"physical/shard_{shard_index}_of_2"
    receipt = json.loads((shard / "SHARD_RECEIPT.json").read_text())
    if (receipt.get("status") != STATUS_PHYSICAL or int(receipt.get("shard_index", -1)) != shard_index
            or int(receipt.get("shard_count", -1)) != SHARDS or int(receipt.get("systems", -1)) != 256
            or int(receipt.get("candidate_rows", -1)) != 9216):
        raise RuntimeError("physical shard receipt has wrong identity or coverage")
    _validate_exact_shard_systems(receipt.get("system_ids", []), shard_index)
    output = _output(root, cfg)
    if (receipt.get("population_receipt_sha256") != sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json")
            or receipt.get("implementation_sha256") != sha256(root / IMPLEMENTATION)):
        raise RuntimeError("physical shard upstream binding differs")
    for name, expected in receipt["hashes"].items():
        if sha256(shard / name) != expected:
            raise RuntimeError("physical shard hash mismatch")
    return shard, receipt


def merge_physical(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    population = _verify_population(root, config_path, cfg); output = _output(root, cfg)
    all_rows, receipts, seen = [], [], set()
    for shard_index in range(SHARDS):
        shard, receipt = _verify_physical_shard(root, cfg, shard_index)
        rows = pd.read_csv(shard / "candidate_rows.csv.gz")
        keys = set(map(tuple, rows[list(ROW_KEY)].to_numpy(np.int64)))
        if seen & keys:
            raise RuntimeError("physical shards overlap")
        seen |= keys; all_rows.append(rows[list(ROW_KEY)]); receipts.append(receipt)
    merged = pd.concat(all_rows, ignore_index=True).sort_values(list(ROW_KEY)).reset_index(drop=True)
    frozen_rows = pd.read_csv(output / "frozen/row_manifest.csv.gz").sort_values(list(ROW_KEY)).reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, frozen_rows, check_dtype=False, check_exact=True)
    _validate_exact_system_union([receipt["system_ids"] for receipt in receipts])
    receipt = {
        "schema_version": "2.0", "status": STATUS_PHYSICAL_MERGED,
        "systems": SYSTEMS, "rows": ROWS, "fixed_merge_order": [0, 1],
        "shard_receipt_hashes": [sha256(output / f"physical/shard_{i}_of_2/SHARD_RECEIPT.json") for i in range(SHARDS)],
        "population_receipt_sha256": sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json"),
        "row_manifest_sha256": population["hashes"]["row_manifest.csv.gz"],
    }
    _atomic_json(output / "physical/PHYSICAL_MERGE_RECEIPT.json", receipt)
    return receipt


def _verify_physical_merge(root: Path, config_path: Path, cfg: dict) -> dict:
    _verify_population(root, config_path, cfg); output = _output(root, cfg)
    path = output / "physical/PHYSICAL_MERGE_RECEIPT.json"; receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_PHYSICAL_MERGED or receipt.get("fixed_merge_order") != [0, 1]:
        raise RuntimeError("physical merge receipt invalid")
    for index, expected in enumerate(receipt["shard_receipt_hashes"]):
        if sha256(output / f"physical/shard_{index}_of_2/SHARD_RECEIPT.json") != expected:
            raise RuntimeError("physical merge source changed")
    return receipt


def _baseline_positions(baseline_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> np.ndarray:
    key = ["system_index", "realization", "history_index", "query_index"]
    lookup = baseline_rows.set_index(key).baseline_row_index
    index = pd.MultiIndex.from_frame(candidate_rows[key])
    positions = lookup.reindex(index).to_numpy()
    if pd.isna(positions).any():
        raise RuntimeError("candidate rows lack a baseline")
    return positions.astype(np.int64)


def _canonical_features(root: Path, reference: dict, baseline_arrays, candidate_arrays,
                        baseline_positions: np.ndarray, device_name: str, cfg: dict) -> tuple[dict, dict]:
    import torch
    from paper_c.extension.prospective_packet_runner import _feature_forward
    from paper_c.swimmer.lqa_prospective import _load_jepa

    device = torch.device(device_name); torch.set_num_threads(1)
    model, norms, training = _load_jepa(root, reference, device)
    expected = cfg["canonical_learner"]
    if training["checkpoint_hashes"]["jepa"] != expected["checkpoint_sha256"]:
        raise RuntimeError("loaded canonical learner differs from frozen checkpoint")
    baseline = _feature_forward(model, norms, baseline_arrays, device)
    candidate = _feature_forward(model, norms, candidate_arrays, device)
    values = {
        "prediction_anchor": baseline["prediction"][baseline_positions].astype(np.float32),
        "query_embedding": candidate["query"].astype(np.float32),
        "predicted_latent_full": candidate["predicted"].astype(np.float32),
        "delta_persistent": (candidate["persistent"] - baseline["persistent"][baseline_positions]).astype(np.float32),
        "normalized_target": ((candidate_arrays.target - norms["target_mean"]) / norms["target_std"]).astype(np.float32),
    }
    return values, {"base_checkpoint_sha256": training["checkpoint_hashes"]["jepa"],
                    "normalization_sha256": sha256(root / expected["normalization"])}


def _gru_features(root: Path, gru_cfg: dict, base_seed: int, baseline_arrays, candidate_arrays,
                  baseline_positions: np.ndarray, device_name: str) -> tuple[dict, dict]:
    import torch
    from paper_c.stage2 import masked_gru_formal as gru
    from paper_c.stage2 import routing_intervention as routing

    device = torch.device(device_name); torch.set_num_threads(1)
    model, norms, before = gru._model(root, gru_cfg, "articulated", base_seed, candidate_arrays, device)

    @torch.no_grad()
    def forward(arrays):
        history, mask, query, target = gru._normalized(arrays, norms)
        ht, mt, qt = torch.from_numpy(history).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(query).to(device)
        encoded = model.segment_encoder(ht) * mt[:, :, None]
        persistent = model.aggregate_encoded(encoded, mt)
        qembed = model.query_encoder(qt)
        predicted = model.latent_predictor(torch.cat((persistent, qembed), dim=1))
        prediction = model.target_decoder(predicted)
        return {"persistent": persistent.cpu().numpy().astype(np.float32),
                "query": qembed.cpu().numpy().astype(np.float32),
                "predicted": predicted.cpu().numpy().astype(np.float32),
                "prediction": prediction.cpu().numpy().astype(np.float32),
                "target": target.astype(np.float32)}

    baseline, candidate = forward(baseline_arrays), forward(candidate_arrays)
    if routing.original_module_hashes(model) != before:
        raise RuntimeError("Masked-GRU base learner changed during feature extraction")
    checkpoint, _ = gru._verify_base(root, gru_cfg, "articulated", base_seed)
    values = {
        "prediction_anchor": baseline["prediction"][baseline_positions],
        "query_embedding": candidate["query"], "predicted_latent_full": candidate["predicted"],
        "delta_persistent": candidate["persistent"] - baseline["persistent"][baseline_positions],
        "normalized_target": candidate["target"],
    }
    return values, {"base_checkpoint_sha256": sha256(checkpoint), "original_module_hashes": before}


def _architecture_key(architecture: str, base_seed: int | None = None) -> str:
    if architecture == "canonical":
        if base_seed is not None:
            raise ValueError("canonical architecture has no base seed argument")
        return "canonical"
    if architecture == "masked_gru" and base_seed in GRU_BASE_SEEDS:
        return f"masked_gru_s{int(base_seed)}"
    raise ValueError("unknown architecture/base seed")


def prepare_feature_shard(root: Path, config_path: Path, architecture: str, base_seed: int | None,
                          shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    physical_merge = _verify_physical_merge(root, config_path, cfg)
    if shard_index not in range(SHARDS):
        raise ValueError("feature shard index must be 0 or 1")
    key = _architecture_key(architecture, base_seed); output = _output(root, cfg)
    physical, physical_receipt = _verify_physical_shard(root, cfg, shard_index)
    baseline_arrays, candidate_arrays = _load_arrays(physical / "baseline_arrays.npz"), _load_arrays(physical / "candidate_arrays.npz")
    baseline_rows, candidate_rows = pd.read_csv(physical / "baseline_rows.csv.gz"), pd.read_csv(physical / "candidate_rows.csv.gz")
    positions = _baseline_positions(baseline_rows, candidate_rows)
    reference = json.loads((output / "frozen/REFERENCE_CONFIG.json").read_text())
    if architecture == "canonical":
        values, identity = _canonical_features(root, reference, baseline_arrays, candidate_arrays, positions, device_name, cfg)
    else:
        gru_cfg = json.loads((root / cfg["masked_gru"]["upstream_config"]).read_text())
        values, identity = _gru_features(root, gru_cfg, int(base_seed), baseline_arrays, candidate_arrays, positions, device_name)
    values = {**{name: candidate_rows[name].to_numpy(np.int64) for name in ROW_KEY}, **values}
    if len(values["system_index"]) != 9216 or not all(np.isfinite(value).all() for value in values.values()):
        raise RuntimeError("feature shard is incomplete or non-finite")
    shard = output / f"features/{key}/shard_{shard_index}_of_2"
    _atomic_npz(shard / "features.npz", **values)
    receipt = {
        "schema_version": "2.0", "status": STATUS_FEATURE, "architecture_key": key,
        "shard_index": shard_index, "shard_count": SHARDS, "systems": 256, "rows": 9216,
        "system_ids": sorted(map(int, np.unique(values["system_index"]))),
        "features_sha256": sha256(shard / "features.npz"), "model_identity": identity,
        "physical_receipt_sha256": sha256(physical / "SHARD_RECEIPT.json"),
        "physical_merge_receipt_sha256": sha256(output / "physical/PHYSICAL_MERGE_RECEIPT.json"),
        "population_receipt_sha256": physical_receipt["population_receipt_sha256"],
        "implementation_sha256": sha256(root / IMPLEMENTATION),
    }
    _atomic_json(shard / "SHARD_RECEIPT.json", receipt)
    return receipt


def _verify_feature_shard(root: Path, cfg: dict, key: str, shard_index: int) -> tuple[dict, dict]:
    output = _output(root, cfg); shard = output / f"features/{key}/shard_{shard_index}_of_2"
    receipt = json.loads((shard / "SHARD_RECEIPT.json").read_text())
    if (receipt.get("status") != STATUS_FEATURE or receipt.get("architecture_key") != key
            or int(receipt.get("shard_index", -1)) != shard_index or receipt.get("system_ids") != list(range(shard_index, SYSTEMS, SHARDS))
            or int(receipt.get("rows", -1)) != 9216 or receipt.get("features_sha256") != sha256(shard / "features.npz")):
        raise RuntimeError("feature shard identity/coverage/hash mismatch")
    physical = output / f"physical/shard_{shard_index}_of_2/SHARD_RECEIPT.json"
    if (receipt.get("physical_receipt_sha256") != sha256(physical)
            or receipt.get("physical_merge_receipt_sha256") != sha256(output / "physical/PHYSICAL_MERGE_RECEIPT.json")
            or receipt.get("population_receipt_sha256") != sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json")
            or receipt.get("implementation_sha256") != sha256(root / IMPLEMENTATION)):
        raise RuntimeError("feature shard upstream binding differs")
    _validate_feature_model_identity(root, cfg, key, receipt.get("model_identity"))
    return _load_npz(shard / "features.npz"), receipt


def _validate_feature_model_identity(root: Path, cfg: dict, key: str, identity: object) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError("feature model identity is absent")
    if key == "canonical":
        expected = cfg["canonical_learner"]
        if (identity.get("base_checkpoint_sha256") != expected["checkpoint_sha256"]
                or identity.get("normalization_sha256") != expected["normalization_sha256"]):
            raise RuntimeError("canonical feature model identity differs")
        return
    if not key.startswith("masked_gru_s"):
        raise RuntimeError("unknown feature architecture key")
    base_seed = int(key.rsplit("s", 1)[1])
    authorization = json.loads((_output(root, cfg) / "authorization/BRANCH_AUTHORIZATION.json").read_text())
    records = [item for item in authorization["masked_gru_checkpoints"] if int(item["base_seed"]) == base_seed]
    expected_hashes = {item["base_checkpoint_sha256"] for item in records}
    module_hashes = identity.get("original_module_hashes")
    if len(expected_hashes) != 1 or identity.get("base_checkpoint_sha256") not in expected_hashes:
        raise RuntimeError("Masked-GRU feature base checkpoint identity differs")
    if not isinstance(module_hashes, dict) or not module_hashes:
        raise RuntimeError("Masked-GRU feature module identity is absent")


def merge_features(root: Path, config_path: Path, architecture: str, base_seed: int | None) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    _verify_physical_merge(root, config_path, cfg); key = _architecture_key(architecture, base_seed); output = _output(root, cfg)
    parts, receipts, systems = [], [], set()
    for shard_index in range(SHARDS):
        values, receipt = _verify_feature_shard(root, cfg, key, shard_index)
        observed = set(map(int, np.unique(values["system_index"])))
        if systems & observed:
            raise RuntimeError("feature shards overlap systems")
        systems |= observed; parts.append(values); receipts.append(receipt)
    _validate_exact_system_union([receipt["system_ids"] for receipt in receipts])
    _require_same_model_identity([receipt["model_identity"] for receipt in receipts])
    merged = {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}
    rows = pd.DataFrame({name: merged[name] for name in ROW_KEY})
    order = rows.sort_values(list(ROW_KEY)).index.to_numpy(np.int64)
    merged = {name: value[order] for name, value in merged.items()}
    frozen = pd.read_csv(output / "frozen/row_manifest.csv.gz")
    pd.testing.assert_frame_equal(pd.DataFrame({name: merged[name] for name in ROW_KEY}), frozen, check_dtype=False, check_exact=True)
    target_hash = hashlib.sha256(np.asarray(merged["normalized_target"], dtype="<f4").tobytes()).hexdigest()
    destination = output / f"features/{key}/merged"
    _atomic_npz(destination / "features.npz", **merged)
    receipt = {
        "schema_version": "2.0", "status": STATUS_FEATURE_MERGED, "architecture_key": key,
        "systems": SYSTEMS, "rows": ROWS, "fixed_merge_order": [0, 1],
        "features_sha256": sha256(destination / "features.npz"), "normalized_target_content_sha256": target_hash,
        "shard_receipt_hashes": [sha256(output / f"features/{key}/shard_{i}_of_2/SHARD_RECEIPT.json") for i in range(SHARDS)],
        "model_identity": receipts[0]["model_identity"],
        "donor_sha256": sha256(output / "frozen/donor_map.npz"),
    }
    _atomic_json(destination / "MERGE_RECEIPT.json", receipt)
    return receipt


def _require_same_model_identity(identities: Sequence[object]) -> None:
    if len(identities) != SHARDS or identities[0] != identities[1]:
        raise RuntimeError("feature shards were produced by mixed model identities")


def _merged_features(root: Path, config_path: Path, cfg: dict, key: str) -> tuple[dict, dict]:
    _verify_physical_merge(root, config_path, cfg); output = _output(root, cfg)
    path = output / f"features/{key}/merged/features.npz"
    receipt_path = output / f"features/{key}/merged/MERGE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != STATUS_FEATURE_MERGED or receipt.get("architecture_key") != key
            or receipt.get("features_sha256") != sha256(path)
            or receipt.get("donor_sha256") != sha256(output / "frozen/donor_map.npz")):
        raise RuntimeError("merged feature cache is invalid")
    values = _load_npz(path)
    if len(values["system_index"]) != ROWS or set(map(int, np.unique(values["system_index"]))) != set(range(SYSTEMS)):
        raise RuntimeError("merged feature cache coverage changed")
    return values, receipt


def _checkpoint_records(root: Path, cfg: dict, architecture: str, base_seed: int | None) -> list[dict]:
    if architecture == "canonical":
        records = cfg["canonical_learner"]["true_checkpoints"]
    else:
        authorization = json.loads((_output(root, cfg) / "authorization/BRANCH_AUTHORIZATION.json").read_text())
        records = [item for item in authorization["masked_gru_checkpoints"] if int(item["base_seed"]) == int(base_seed)]
    if tuple(int(item["optimization_seed"]) for item in records) != CANONICAL_SEEDS:
        raise RuntimeError("frozen checkpoint records changed")
    for item in records:
        if sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError("frozen adapter checkpoint hash changed")
    return records


def _load_base_and_adapter(root: Path, cfg: dict, architecture: str, base_seed: int | None,
                           checkpoint: dict, device):
    import torch
    from paper_c.stage2.delta_gated_isolation import DeltaGatedAdapter
    if architecture == "canonical":
        from paper_c.swimmer.lqa_prospective import _load_jepa
        reference = json.loads((_output(root, cfg) / "frozen/REFERENCE_CONFIG.json").read_text())
        model, _, training = _load_jepa(root, reference, device)
        if training["checkpoint_hashes"]["jepa"] != cfg["canonical_learner"]["checkpoint_sha256"]:
            raise RuntimeError("canonical base checkpoint changed during evaluation")
    else:
        from paper_c.stage2 import masked_gru_formal as gru
        gru_cfg = json.loads((root / cfg["masked_gru"]["upstream_config"]).read_text())
        arrays = _load_arrays(_output(root, cfg) / "physical/shard_0_of_2/candidate_arrays.npz")
        model, _, _ = gru._model(root, gru_cfg, "articulated", int(base_seed), arrays, device)
    adapter = DeltaGatedAdapter(64, 64).to(device)
    adapter.load_state_dict(torch.load(root / checkpoint["path"], map_location=device, weights_only=True)); adapter.eval(); model.eval()
    return model, adapter


def _predict(root: Path, cfg: dict, architecture: str, base_seed: int | None, features: dict,
             positions: np.ndarray, donor: np.ndarray, device_name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import torch
    device = torch.device(device_name); positions = np.asarray(positions, dtype=np.int64)
    result = {}; batch_size = int(cfg["execution"]["batch_size"])
    with torch.no_grad():
        for checkpoint in _checkpoint_records(root, cfg, architecture, base_seed):
            model, adapter = _load_base_and_adapter(root, cfg, architecture, base_seed, checkpoint, device)
            self_parts, donor_parts = [], []
            for start in range(0, len(positions), batch_size):
                local = positions[start:start + batch_size]
                query = torch.from_numpy(features["query_embedding"][local]).to(device)
                latent = torch.from_numpy(features["predicted_latent_full"][local]).to(device)
                self_delta = torch.from_numpy(features["delta_persistent"][local]).to(device)
                donor_delta = torch.from_numpy(features["delta_persistent"][donor[local]]).to(device)
                self_parts.append(model.target_decoder(latent + adapter(self_delta, query)).cpu().numpy())
                donor_parts.append(model.target_decoder(latent + adapter(donor_delta, query)).cpu().numpy())
            result[int(checkpoint["optimization_seed"])] = (np.concatenate(self_parts), np.concatenate(donor_parts))
    return result


def parity(root: Path, config_path: Path, architecture: str, base_seed: int | None, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    _verify_authorization(root, config_path, cfg); key = _architecture_key(architecture, base_seed)
    features, merge = _merged_features(root, config_path, cfg, key); output = _output(root, cfg)
    donor = _load_npz(output / "frozen/donor_map.npz")["donor_position"].astype(np.int64)
    systems = np.arange(int(cfg["execution"]["parity_systems"]), dtype=np.int64)
    positions = np.flatnonzero(np.isin(features["system_index"], systems))
    direct = _predict(root, cfg, architecture, base_seed, features, positions, donor, device_name)
    split_positions = [positions[features["system_index"][positions] % SHARDS == shard] for shard in range(SHARDS)]
    pieces = [_predict(root, cfg, architecture, base_seed, features, local, donor, device_name) for local in split_positions]
    order = np.concatenate(split_positions); restore = np.argsort(order, kind="stable")
    maximum = 0.0
    for seed in CANONICAL_SEEDS:
        for branch in (0, 1):
            reconstructed = np.concatenate([piece[seed][branch] for piece in pieces], axis=0)[restore]
            difference = float(np.max(np.abs(direct[seed][branch].astype(np.float64) - reconstructed.astype(np.float64))))
            maximum = max(maximum, difference)
            if difference > PARITY_TOLERANCE:
                raise RuntimeError("one-task/two-shard actual-checkpoint prediction parity failed")
    target = output / f"parity/{key}/PARITY.json"
    receipt = {
        "schema_version": "2.0", "status": STATUS_PARITY, "architecture_key": key,
        "systems": list(map(int, systems)), "rows": int(len(positions)), "fixed_merge_order": [0, 1],
        "maximum_absolute_prediction_difference": maximum, "tolerance": PARITY_TOLERANCE,
        "features_sha256": merge["features_sha256"], "donor_sha256": sha256(output / "frozen/donor_map.npz"),
        "checkpoint_hashes": {str(item["optimization_seed"]): item["sha256"] for item in _checkpoint_records(root, cfg, architecture, base_seed)},
        "target_values_used": False, "scientific_contrast_computed": False,
    }
    _atomic_json(target, receipt)
    return receipt


def _verify_parity(root: Path, cfg: dict, key: str, merge: dict) -> dict:
    output = _output(root, cfg); path = output / f"parity/{key}/PARITY.json"; receipt = json.loads(path.read_text())
    expected_checkpoints = (_expected_checkpoint_hashes(root, cfg, "canonical", None) if key == "canonical"
                            else _expected_checkpoint_hashes(root, cfg, "masked_gru", int(key.rsplit("s", 1)[1])))
    _validate_parity_semantics(receipt, key, merge["features_sha256"],
                               sha256(output / "frozen/donor_map.npz"), expected_checkpoints)
    return receipt


def _validate_parity_semantics(receipt: Mapping[str, object], key: str, features_sha256: str,
                               donor_sha256: str, checkpoint_hashes: Mapping[str, str]) -> None:
    if (receipt.get("status") != STATUS_PARITY or receipt.get("architecture_key") != key
            or receipt.get("features_sha256") != features_sha256
            or float(receipt.get("maximum_absolute_prediction_difference", 1.0)) > PARITY_TOLERANCE
            or float(receipt.get("tolerance", -1.0)) != PARITY_TOLERANCE
            or receipt.get("systems") != list(range(8)) or int(receipt.get("rows", -1)) != 8 * ROWS_PER_SYSTEM
            or receipt.get("fixed_merge_order") != [0, 1]
            or receipt.get("donor_sha256") != donor_sha256
            or receipt.get("checkpoint_hashes") != dict(checkpoint_hashes)
            or receipt.get("target_values_used") is not False
            or receipt.get("scientific_contrast_computed") is not False):
        raise RuntimeError("actual-checkpoint sharding parity is missing or stale")


def _expected_checkpoint_hashes(root: Path, cfg: dict, architecture: str,
                                base_seed: int | None) -> dict[str, str]:
    return {str(item["optimization_seed"]): item["sha256"]
            for item in _checkpoint_records(root, cfg, architecture, base_seed)}


def _verify_evaluation_receipt(root: Path, config_path: Path, cfg: dict, architecture: str,
                               base_seed: int | None, shard_index: int, receipt: dict,
                               statistics_path: Path, merge: dict, parity_receipt: dict) -> None:
    key = _architecture_key(architecture, base_seed); output = _output(root, cfg)
    expected = {
        "status": STATUS_EVAL, "architecture_key": key, "shard_index": shard_index,
        "shard_count": SHARDS, "systems": 256,
        "authorization_sha256": sha256(output / "authorization/BRANCH_AUTHORIZATION.json"),
        "population_sha256": sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json"),
        "donor_sha256": sha256(output / "frozen/donor_map.npz"),
        "features_sha256": merge["features_sha256"],
        "parity_sha256": sha256(output / f"parity/{key}/PARITY.json"),
        "implementation_sha256": sha256(root / IMPLEMENTATION),
        "checkpoint_hashes": _expected_checkpoint_hashes(root, cfg, architecture, base_seed),
        "branch_authorization_status": STATUS_AUTH,
    }
    _require_exact_bindings(receipt, expected, "evaluation shard")
    _validate_exact_shard_systems(receipt.get("system_ids", []), shard_index)
    if receipt.get("statistics_sha256") != sha256(statistics_path):
        raise RuntimeError("evaluation statistics hash differs")
    if parity_receipt.get("checkpoint_hashes") != expected["checkpoint_hashes"]:
        raise RuntimeError("parity checkpoint binding differs from evaluation")


def _require_exact_bindings(observed: Mapping[str, object], expected: Mapping[str, object], label: str) -> None:
    for name, value in expected.items():
        if observed.get(name) != value:
            raise RuntimeError(f"{label} binding differs: {name}")


def evaluate_shard(root: Path, config_path: Path, architecture: str, base_seed: int | None,
                   shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    if shard_index not in range(SHARDS):
        raise ValueError("evaluation shard index must be 0 or 1")
    authorization = _verify_authorization(root, config_path, cfg); key = _architecture_key(architecture, base_seed)
    features, merge = _merged_features(root, config_path, cfg, key); parity_receipt = _verify_parity(root, cfg, key, merge)
    output = _output(root, cfg); donor = _load_npz(output / "frozen/donor_map.npz")["donor_position"].astype(np.int64)
    positions = np.flatnonzero(features["system_index"].astype(np.int64) % SHARDS == shard_index)
    predictions = _predict(root, cfg, architecture, base_seed, features, positions, donor, device_name)
    target, anchor = features["normalized_target"][positions], features["prediction_anchor"][positions]
    baseline_loss = np.mean((target.astype(np.float64) - anchor.astype(np.float64)) ** 2, axis=1)
    blocks = []
    row_values = pd.DataFrame({name: features[name][positions].astype(np.int64) for name in ROW_KEY})
    for seed in CANONICAL_SEEDS:
        prediction_self, prediction_donor = predictions[seed]
        self_loss = np.mean((target.astype(np.float64) - prediction_self.astype(np.float64)) ** 2, axis=1)
        donor_loss = np.mean((target.astype(np.float64) - prediction_donor.astype(np.float64)) ** 2, axis=1)
        block = row_values.copy(); block["optimization_seed"] = seed
        block["count"] = 1; block["sum_gain_self"] = baseline_loss - self_loss
        block["sum_gain_donor"] = baseline_loss - donor_loss; block["sum_specificity"] = donor_loss - self_loss
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True)
    if len(rows) != 256 * ROWS_PER_SYSTEM * 3 or not np.allclose(rows.sum_specificity,
            rows.sum_gain_self - rows.sum_gain_donor, rtol=0.0, atol=1e-12):
        raise RuntimeError("evaluation shard coverage or contrast identity failed")
    stats = rows.groupby(["system_index", "optimization_seed"], sort=True, as_index=False).agg(
        count=("count", "sum"), sum_gain_self=("sum_gain_self", "sum"),
        sum_gain_donor=("sum_gain_donor", "sum"), sum_specificity=("sum_specificity", "sum"))
    for name in ("sum_gain_self", "sum_gain_donor", "sum_specificity"):
        stats[name] = stats[name].astype(np.float64)
    if not (stats["count"] == ROWS_PER_SYSTEM).all() or set(stats.system_index.astype(int)) != set(range(shard_index, SYSTEMS, SHARDS)):
        raise RuntimeError("evaluation sufficient statistics have missing/wrong-shard systems")
    shard = output / f"evaluation/{key}/shard_{shard_index}_of_2"
    _atomic_csv(shard / "sufficient_statistics.csv.gz", stats)
    receipt = {
        "schema_version": "2.0", "status": STATUS_EVAL, "architecture_key": key,
        "shard_index": shard_index, "shard_count": SHARDS, "systems": 256, "rows": int(len(rows)),
        "system_ids": sorted(map(int, stats.system_index.unique())),
        "statistics_sha256": sha256(shard / "sufficient_statistics.csv.gz"),
        "authorization_sha256": sha256(output / "authorization/BRANCH_AUTHORIZATION.json"),
        "population_sha256": sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json"),
        "donor_sha256": sha256(output / "frozen/donor_map.npz"), "features_sha256": merge["features_sha256"],
        "parity_sha256": sha256(output / f"parity/{key}/PARITY.json"),
        "implementation_sha256": sha256(root / IMPLEMENTATION),
        "checkpoint_hashes": parity_receipt["checkpoint_hashes"],
        "branch_authorization_status": authorization["status"],
    }
    _atomic_json(shard / "SHARD_RECEIPT.json", receipt)
    return receipt


def _bootstrap(values: np.ndarray, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (SYSTEMS,) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires exactly 512 finite system values")
    generator = np.random.default_rng(seed); means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        draws = generator.integers(0, SYSTEMS, size=(stop - start, SYSTEMS))
        means[start:stop] = values[draws].mean(axis=1)
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)), "systems": SYSTEMS,
            "replicates": replicates, "seed": seed, "system_positive_fraction": float(np.mean(values > 0))}


def merge_architecture_result(root: Path, config_path: Path, architecture: str, base_seed: int | None) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    _verify_authorization(root, config_path, cfg); key = _architecture_key(architecture, base_seed); output = _output(root, cfg)
    features, merge = _merged_features(root, config_path, cfg, key)
    del features
    parity_receipt = _verify_parity(root, cfg, key, merge)
    pieces, systems, shard_receipt_hashes = [], set(), []
    for shard_index in range(SHARDS):
        shard = output / f"evaluation/{key}/shard_{shard_index}_of_2"
        receipt = json.loads((shard / "SHARD_RECEIPT.json").read_text())
        path = shard / "sufficient_statistics.csv.gz"
        _verify_evaluation_receipt(root, config_path, cfg, architecture, base_seed, shard_index,
                                   receipt, path, merge, parity_receipt)
        table = pd.read_csv(path); observed = set(map(int, table.system_index.unique()))
        if systems & observed:
            raise RuntimeError("evaluation shards overlap")
        systems |= observed; pieces.append(table); shard_receipt_hashes.append(sha256(shard / "SHARD_RECEIPT.json"))
    _validate_exact_system_union([list(map(int, piece.system_index.unique())) for piece in pieces])
    stats = pd.concat(pieces, ignore_index=True).sort_values(["system_index", "optimization_seed"]).reset_index(drop=True)
    if len(stats) != SYSTEMS * 3 or stats.duplicated(["system_index", "optimization_seed"]).any() or not (stats["count"] == 36).all():
        raise RuntimeError("merged sufficient statistics have wrong coverage")
    stats["specificity"] = stats.sum_specificity.astype(np.float64) / stats["count"].astype(np.float64)
    system = stats.groupby("system_index", sort=True, as_index=False).specificity.mean()
    if list(system.system_index.astype(int)) != list(range(SYSTEMS)):
        raise RuntimeError("system aggregation order/coverage changed")
    if architecture == "canonical":
        replicates, seed = int(cfg["primary"]["bootstrap_replicates"]), int(cfg["primary"]["bootstrap_seed"])
    else:
        replicates, seed = int(cfg["secondary"]["bootstrap_replicates"]), int(cfg["secondary"]["bootstrap_seed"]) + GRU_BASE_SEEDS.index(int(base_seed))
    interval = _bootstrap(system.specificity.to_numpy(np.float64), replicates, seed)
    result = {
        "schema_version": "2.0", "status": "SUPPORTED" if interval["ci_low"] > 0 else "NOT_SUPPORTED",
        "architecture_key": key, "evidence_identity": "prospective_frozen_same_checkpoint_input_intervention",
        "primary_for_paper": architecture == "canonical", "systems": SYSTEMS,
        "optimization_seeds_are_scientific_samples": False,
        "same_checkpoint_self_minus_donor": interval,
        "fixed_merge_order": [0, 1], "model_training": False, "checkpoint_selection": False,
    }
    destination = output / f"results/{key}"
    _atomic_csv(destination / "merged_sufficient_statistics.csv.gz", stats)
    _atomic_csv(destination / "system_statistics.csv", system)
    result["output_hashes"] = {name: sha256(destination / name) for name in
                               ("merged_sufficient_statistics.csv.gz", "system_statistics.csv")}
    _atomic_json(destination / "RESULT.json", result)
    _atomic_json(destination / "RESULT_RECEIPT.json", {
        "schema_version": "2.0", "status": "PAPER_C_D1A_ARCHITECTURE_RESULT_RECEIPT_COMPLETE",
        "architecture_key": key, "result_sha256": sha256(destination / "RESULT.json"),
        "evaluation_shard_receipt_hashes": shard_receipt_hashes,
        "authorization_sha256": sha256(output / "authorization/BRANCH_AUTHORIZATION.json"),
        "implementation_sha256": sha256(root / IMPLEMENTATION),
    })
    return result


def _result_interval_equal(observed: dict, expected: dict, tolerance: float = 1e-15) -> bool:
    exact = ("systems", "replicates", "seed")
    numeric = ("estimate", "ci_low", "ci_high", "system_positive_fraction")
    return (all(observed.get(name) == expected.get(name) for name in exact)
            and all(abs(float(observed.get(name, np.nan)) - float(expected.get(name, np.nan))) <= tolerance
                    for name in numeric))


def _conjunctive_secondary_supported(statuses: Sequence[str]) -> bool:
    if len(statuses) != len(GRU_BASE_SEEDS):
        raise ValueError("cross-architecture conjunction requires both frozen GRU bases")
    return all(status == "SUPPORTED" for status in statuses)


def _validate_result_semantics(result: Mapping[str, object], key: str, expected_primary: bool) -> None:
    if (result.get("architecture_key") != key or result.get("primary_for_paper") is not expected_primary
            or result.get("evidence_identity") != "prospective_frozen_same_checkpoint_input_intervention"
            or result.get("fixed_merge_order") != [0, 1] or result.get("systems") != SYSTEMS
            or result.get("model_training") is not False or result.get("checkpoint_selection") is not False):
        raise RuntimeError("architecture result semantic identity differs")


def _verify_architecture_result(root: Path, cfg: dict, key: str, expected_primary: bool,
                                replicates: int, seed: int) -> tuple[dict, np.ndarray]:
    output = _output(root, cfg); destination = output / f"results/{key}"
    result_path, receipt_path = destination / "RESULT.json", destination / "RESULT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "PAPER_C_D1A_ARCHITECTURE_RESULT_RECEIPT_COMPLETE"
            or receipt.get("architecture_key") != key or receipt.get("result_sha256") != sha256(result_path)
            or receipt.get("authorization_sha256") != sha256(output / "authorization/BRANCH_AUTHORIZATION.json")
            or receipt.get("implementation_sha256") != sha256(root / IMPLEMENTATION)):
        raise RuntimeError("architecture result receipt is missing or stale")
    _verify_evaluation_receipt_hash_list(output, key, receipt.get("evaluation_shard_receipt_hashes"))
    result = json.loads(result_path.read_text())
    _validate_result_semantics(result, key, expected_primary)
    for name, expected_hash in result.get("output_hashes", {}).items():
        if name not in ("merged_sufficient_statistics.csv.gz", "system_statistics.csv") or sha256(destination / name) != expected_hash:
            raise RuntimeError("architecture result output hash differs")
    if set(result.get("output_hashes", {})) != {"merged_sufficient_statistics.csv.gz", "system_statistics.csv"}:
        raise RuntimeError("architecture result output manifest is incomplete")
    system = pd.read_csv(destination / "system_statistics.csv")
    if (list(system.columns) != ["system_index", "specificity"] or len(system) != SYSTEMS
            or list(system.system_index.astype(int)) != list(range(SYSTEMS)) or not np.isfinite(system.specificity).all()):
        raise RuntimeError("architecture system table schema/coverage differs")
    values = system.specificity.to_numpy(np.float64)
    recomputed = _bootstrap(values, replicates, seed)
    expected_status = "SUPPORTED" if recomputed["ci_low"] > 0 else "NOT_SUPPORTED"
    if result.get("status") != expected_status or not _result_interval_equal(
            result.get("same_checkpoint_self_minus_donor", {}), recomputed):
        raise RuntimeError("architecture result status/interval does not recompute")
    return result, values


def _verify_evaluation_receipt_hash_list(output: Path, key: str, recorded: object) -> None:
    expected = [sha256(output / f"evaluation/{key}/shard_{index}_of_2/SHARD_RECEIPT.json")
                for index in range(SHARDS)]
    if recorded != expected:
        raise RuntimeError("architecture result no longer binds the current evaluation shards")


def finalize_packet(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = _config(root, config_path)
    _verify_authorization(root, config_path, cfg); output = _output(root, cfg)
    canonical, _ = _verify_architecture_result(
        root, cfg, "canonical", True, int(cfg["primary"]["bootstrap_replicates"]), int(cfg["primary"]["bootstrap_seed"]),
    )
    gru, panel_values = {}, []
    for offset, base_seed in enumerate(GRU_BASE_SEEDS):
        item, values = _verify_architecture_result(
            root, cfg, f"masked_gru_s{base_seed}", False,
            int(cfg["secondary"]["bootstrap_replicates"]), int(cfg["secondary"]["bootstrap_seed"]) + offset,
        )
        gru[str(base_seed)] = item; panel_values.append(values)
    canonical_supported = canonical["status"] == "SUPPORTED"
    gru_supported = _conjunctive_secondary_supported([gru[str(seed)]["status"] for seed in GRU_BASE_SEEDS])
    panel = _bootstrap(np.mean(panel_values, axis=0), int(cfg["secondary"]["bootstrap_replicates"]),
                       int(cfg["secondary"]["bootstrap_seed"]) + 10)
    result = {
        "schema_version": "2.0", "status": STATUS_RESULT,
        "evidence_identity": "prospective_canonical_primary_with_predeclared_cross_architecture_secondary",
        "outcome_bin": ("CANONICAL_AND_CROSS_ARCHITECTURE_SUPPORTED" if canonical_supported and gru_supported
                        else "CANONICAL_SUPPORTED_SECONDARY_MIXED" if canonical_supported
                        else "CANONICAL_PRIMARY_NOT_SUPPORTED"),
        "canonical_primary": canonical,
        "masked_gru_key_secondary": {"per_base": gru, "conjunctive_supported": gru_supported,
                                     "descriptive_fixed_panel_average": panel,
                                     "pooling_cannot_rescue_failed_base": True},
        "scientific_verdict": {
            "canonical_same_checkpoint_input_effect_prospectively_supported": canonical_supported,
            "cross_architecture_confirmation_supported": gru_supported,
        },
        "new_system_blocks": 1, "maximum_new_blocks": 1, "intervention_line_closed": True,
        "claim_boundary": (
            "Corrected same-checkpoint input intervention; does not retroactively relabel the "
            "original diagonal endpoint and does not identify anchor-conditioned transport."
        ),
    }
    destination = output / "results/final"; _atomic_json(destination / "RESULTS.json", result)
    receipt = {
        "schema_version": "2.0", "status": "PAPER_C_D1A_FINAL_RESULTS_RECEIPT_COMPLETE",
        "results_sha256": sha256(destination / "RESULTS.json"),
        "authorization_sha256": sha256(output / "authorization/BRANCH_AUTHORIZATION.json"),
        "population_sha256": sha256(output / "frozen/POPULATION_AND_DONOR_FROZEN.json"),
        "implementation_sha256": sha256(root / IMPLEMENTATION), "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(root / CANONICAL_PROTOCOL),
        "architecture_result_receipt_hashes": {
            key: sha256(output / f"results/{key}/RESULT_RECEIPT.json")
            for key in ("canonical", *[f"masked_gru_s{seed}" for seed in GRU_BASE_SEEDS])
        },
    }
    _atomic_json(destination / "RESULTS_RECEIPT.json", receipt)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("authorize-and-freeze-design"); commands.add_parser("materialize-population-and-donor")
    physical = commands.add_parser("physical-shard"); physical.add_argument("--shard-index", type=int, required=True)
    commands.add_parser("merge-physical")
    for name in ("feature-shard", "merge-features", "parity", "evaluate-shard", "merge-architecture-result"):
        sub = commands.add_parser(name); sub.add_argument("--architecture", choices=("canonical", "masked_gru"), required=True)
        sub.add_argument("--base-seed", type=int)
        if name in ("feature-shard", "evaluate-shard"):
            sub.add_argument("--shard-index", type=int, required=True)
        if name in ("feature-shard", "parity", "evaluate-shard"):
            sub.add_argument("--device", required=True)
    commands.add_parser("finalize-packet")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv); root, config = args.root.resolve(), args.config.resolve()
    if args.command == "authorize-and-freeze-design": result = authorize_and_freeze_design(root, config)
    elif args.command == "materialize-population-and-donor": result = materialize_population_and_donor(root, config)
    elif args.command == "physical-shard": result = materialize_physical_shard(root, config, args.shard_index)
    elif args.command == "merge-physical": result = merge_physical(root, config)
    elif args.command == "feature-shard": result = prepare_feature_shard(root, config, args.architecture, args.base_seed, args.shard_index, args.device)
    elif args.command == "merge-features": result = merge_features(root, config, args.architecture, args.base_seed)
    elif args.command == "parity": result = parity(root, config, args.architecture, args.base_seed, args.device)
    elif args.command == "evaluate-shard": result = evaluate_shard(root, config, args.architecture, args.base_seed, args.shard_index, args.device)
    elif args.command == "merge-architecture-result": result = merge_architecture_result(root, config, args.architecture, args.base_seed)
    elif args.command == "finalize-packet": result = finalize_packet(root, config)
    else: raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
