"""Outcome-blind LQA/Raw-CKA supporting geometry for the fresh packet.

This module is intentionally independent of the prospective outcome runner.
It reads only the frozen fresh physical systems and contexts, the canonical
simulator/reference configuration, and the frozen learner assets required by
``lqa_bank``.  It never opens the learner-target artifact and never computes a
decoded prediction, loss, gain, Bayes correction, or learner outcome.

Two shards are fixed by ``system_index % 2``.  Each shard and the fixed-order
merge are immutable, hash-receipted artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from paper_c.extension import prospective_packet as packet_kernel
from paper_c.extension.prospective_packet import ROW_KEY, row_population_sha256, sha256_file


SCHEMA_VERSION = "1.0"
SHARD_COUNT = 2
SYSTEMS = 512
QUERIES = 6
CANDIDATES = 6
ROWS_PER_SYSTEM = QUERIES * CANDIDATES
GEOMETRY_PARTICLES = 512
GEOMETRY_SEED = 70311
SCORE_FIELDS = ROW_KEY + ("lqa", "raw_cka")
SHARD_STATUS = "PROSPECTIVE_GEOMETRY_SUPPORTING_SHARD_COMPLETE_OUTCOME_BLIND"
MERGE_STATUS = "PROSPECTIVE_GEOMETRY_SUPPORTING_MERGED_OUTCOME_BLIND"


def _root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp_noreplace(temporary: Path, destination: Path) -> None:
    """Publish a fully written file without replacing an existing artifact."""

    temporary, destination = Path(temporary), Path(destination)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise RuntimeError(f"immutable artifact already exists: {destination}") from error
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    if temporary.exists():
        raise RuntimeError(f"temporary artifact already exists: {temporary}")
    try:
        np.savez_compressed(temporary, **values)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_temp_noreplace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary artifact already exists: {temporary}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        _publish_temp_noreplace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _canonical_order(values: Mapping[str, np.ndarray]) -> np.ndarray:
    columns = [np.asarray(values[field], dtype=np.int64) for field in ROW_KEY]
    return np.lexsort(tuple(columns[index] for index in reversed(range(len(columns)))))


def _sort_scores(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = _canonical_order(values)
    return {name: np.asarray(values[name])[order] for name in SCORE_FIELDS}


def _validate_scores(
    values: Mapping[str, np.ndarray], expected_system_ids: Sequence[int], shard_index: int | None,
) -> dict:
    if set(values) != set(SCORE_FIELDS):
        raise ValueError(
            f"geometry score schema mismatch: missing={sorted(set(SCORE_FIELDS) - set(values))}, "
            f"extra={sorted(set(values) - set(SCORE_FIELDS))}"
        )
    expected_system_ids = np.asarray(expected_system_ids, dtype=np.int64)
    rows = len(expected_system_ids) * ROWS_PER_SYSTEM
    integer_values = {}
    for field in ROW_KEY:
        array = np.asarray(values[field])
        if array.shape != (rows,) or array.dtype.kind not in "iu":
            raise ValueError(f"{field} must be a one-dimensional integer array with {rows} rows")
        integer_values[field] = array.astype(np.int64, copy=False)
    for field in ("lqa", "raw_cka"):
        array = np.asarray(values[field])
        if array.shape != (rows,) or array.dtype.kind not in "fc" or not np.isfinite(array).all():
            raise ValueError(f"{field} must contain exactly {rows} finite numeric values")
    keys = np.column_stack([integer_values[field] for field in ROW_KEY])
    if len(np.unique(keys, axis=0)) != rows:
        raise ValueError("geometry score row keys are duplicated")
    systems = np.unique(integer_values["system_index"])
    if not np.array_equal(systems, expected_system_ids):
        raise ValueError("geometry score systems differ from the frozen expected population")
    if shard_index is not None and np.any(systems % SHARD_COUNT != int(shard_index)):
        raise ValueError("geometry score shard violates system_index modulo ownership")
    for system_index in expected_system_ids.tolist():
        mask = integer_values["system_index"] == system_index
        cell = keys[mask]
        if len(cell) != ROWS_PER_SYSTEM:
            raise ValueError(f"system {system_index} does not contain 36 score rows")
        if len(np.unique(cell[:, 1:3], axis=0)) != 1:
            raise ValueError(f"system {system_index} does not have one frozen context")
        observed = set(map(tuple, cell[:, 3:5].tolist()))
        expected = {(query, candidate) for query in range(QUERIES) for candidate in range(CANDIDATES)}
        if observed != expected:
            raise ValueError(f"system {system_index} has incomplete query/candidate coverage")
    return {
        "rows": rows,
        "systems": len(systems),
        "row_population_sha256": row_population_sha256(values),
    }


def _source_paths(root: Path, config_path: Path, reference: Mapping[str, object]) -> dict[str, Path]:
    model_root = _root_path(root, reference["s2_models"])
    return {
        "packet_config": config_path,
        "fresh_reference_config": _root_path(
            root,
            _json(config_path)["output_root"],
        ) / "frozen" / "FRESH_REFERENCE_CONFIG.json",
        "base_config": _root_path(root, reference["base_config"]),
        "s0_receipt": _root_path(root, reference["s0_receipt"]),
        "s2_config": _root_path(root, reference["s2_config"]),
        "checkpoint": model_root / "persistent_jepa_frozen.pt",
        "normalization": model_root / "train_only_normalization.npz",
        "training_receipt": model_root / "s2_training_receipt.json",
        "geometry_kernel": Path(__file__).resolve(),
        "packet_key_kernel": Path(packet_kernel.__file__).resolve(),
    }


def _load_contract(root: Path, config_path: Path) -> dict:
    """Verify all frozen outcome-blind inputs without touching a target artifact."""

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _json(config_path)
    if cfg.get("status") != "IMPLEMENTATION_AUTHORIZED_OUTCOME_LOCKED":
        raise RuntimeError("prospective packet configuration is not outcome-locked")
    design = cfg.get("fresh_design", {})
    if {
        "systems": int(design.get("systems", -1)),
        "system_seed": int(design.get("system_seed", -1)),
        "context_seed": int(design.get("context_seed", -1)),
        "queries": int(design.get("queries", -1)),
        "candidates": int(design.get("candidates", -1)),
    } != {"systems": 512, "system_seed": 78101, "context_seed": 78103, "queries": 6, "candidates": 6}:
        raise RuntimeError("fresh geometry stage requires the frozen 512 x 6 x 6 design")
    identity_round_decimals = int(design.get("identity_round_decimals", -1))
    if identity_round_decimals != 12:
        raise RuntimeError("fresh geometry stage requires the frozen 12-decimal system identity rule")
    if int(cfg.get("reference", {}).get("particles", -1)) != GEOMETRY_PARTICLES:
        raise RuntimeError("packet reference particles are not frozen to 512")
    execution = cfg.get("execution", {})
    if int(execution.get("feature_shards", -1)) != SHARD_COUNT or execution.get("system_assignment") != "system_index_mod_2":
        raise RuntimeError("fresh geometry stage requires the frozen two-shard ownership")

    output = _root_path(root, cfg["output_root"])
    frozen = output / "frozen"
    systems_path = frozen / "fresh_systems.npy"
    manifest_path = frozen / "fresh_system_context_manifest.csv.gz"
    reference_path = frozen / "FRESH_REFERENCE_CONFIG.json"
    population_path = frozen / "FRESH_POPULATION_FROZEN.json"
    for path in (systems_path, manifest_path, reference_path, population_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen fresh geometry input is missing: {path}")
    population = _json(population_path)
    if (
        population.get("status") != "PAPER_C_PROSPECTIVE_FRESH_POPULATION_FROZEN"
        or int(population.get("systems", -1)) != SYSTEMS
        or population.get("learner_outcome_read") is not False
        or population.get("systems_sha256") != sha256_file(systems_path)
        or population.get("manifest_sha256") != sha256_file(manifest_path)
        or population.get("reference_config_sha256") != sha256_file(reference_path)
    ):
        raise RuntimeError("fresh population receipt is stale or not outcome blind")

    reference = _json(reference_path)
    formal = reference.get("formal", {})
    geometry = reference.get("reference", {})
    if (
        int(formal.get("system_seed", -1)) != 78101
        or int(formal.get("context_seed", -1)) != 78103
        or int(formal.get("pool_max", -1)) != SYSTEMS
        or int(formal.get("target_systems", -1)) != SYSTEMS
        or int(formal.get("contexts_per_system", -1)) != 1
        or int(geometry.get("geometry_particles", -1)) != GEOMETRY_PARTICLES
        or int(geometry.get("geometry_scramble_seed", -1)) != GEOMETRY_SEED
    ):
        raise RuntimeError("fresh reference configuration differs from frozen geometry semantics")
    if any(bool(value) for value in reference.get("outcome_blindness", {}).values()):
        raise RuntimeError("fresh reference configuration is not outcome blind")

    systems = np.load(systems_path, allow_pickle=False)
    if systems.shape != (SYSTEMS, 5) or not np.isfinite(systems).all():
        raise RuntimeError("frozen fresh physical-system array is invalid")
    manifest = pd.read_csv(manifest_path).sort_values("system_index").reset_index(drop=True)
    required_columns = {"system_index", "realization", "history_index", *(f"theta_{index}" for index in range(5))}
    if not required_columns.issubset(manifest.columns) or len(manifest) != SYSTEMS:
        raise RuntimeError("fresh system/context manifest has an invalid schema or size")
    if not np.array_equal(manifest.system_index.to_numpy(np.int64), np.arange(SYSTEMS, dtype=np.int64)):
        raise RuntimeError("fresh manifest systems must be unique and complete")
    if not manifest.realization.between(0, 3).all() or not manifest.history_index.between(0, 5).all():
        raise RuntimeError("fresh manifest contains an invalid frozen context")
    manifest_theta = manifest[[f"theta_{index}" for index in range(5)]].to_numpy(np.float64)
    if not np.array_equal(
        np.round(manifest_theta, identity_round_decimals),
        np.round(systems.astype(np.float64, copy=False), identity_round_decimals),
    ):
        raise RuntimeError("fresh manifest physical parameters differ from fresh_systems.npy")

    paths = _source_paths(root, config_path, reference)
    from paper_c.coupled_sled import learner as learner_kernel
    from paper_c.coupled_sled import posterior as posterior_kernel
    from paper_c.swimmer import lqa_prospective as lqa_kernel
    from paper_c.swimmer import model as simulator_kernel
    from paper_c.swimmer import waveforms as waveform_kernel

    paths.update({
        "learner_kernel": Path(learner_kernel.__file__).resolve(),
        "posterior_kernel": Path(posterior_kernel.__file__).resolve(),
        "lqa_kernel": Path(lqa_kernel.__file__).resolve(),
        "simulator_kernel": Path(simulator_kernel.__file__).resolve(),
        "waveform_kernel": Path(waveform_kernel.__file__).resolve(),
        "fresh_systems": systems_path,
        "fresh_manifest": manifest_path,
        "population_receipt": population_path,
    })
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"geometry source binding is missing: {missing}")
    base = _json(paths["base_config"])
    s0 = _json(paths["s0_receipt"])
    training = _json(paths["training_receipt"])
    if (
        base.get("status") != "SWIMMER_S0_CPU_DEVELOPMENT"
        or base.get("model", {}).get("integrator") != "RK4"
        or s0.get("status") != "S0_GO"
        or s0.get("config_sha256") != sha256_file(paths["base_config"])
    ):
        raise RuntimeError("canonical Articulated simulator/version binding failed")
    if (
        training.get("status") != "S2_LEARNER_SUFFICIENCY_GO"
        or training.get("config_sha256") != sha256_file(paths["s2_config"])
        or training.get("checkpoint_hashes", {}).get("jepa") != sha256_file(paths["checkpoint"])
        or training.get("checkpoint_hashes", {}).get("normalization") != sha256_file(paths["normalization"])
    ):
        raise RuntimeError("frozen learner/checkpoint/normalization binding failed")
    source_hashes = {name: sha256_file(path) for name, path in sorted(paths.items())}
    return {
        "root": root,
        "config_path": config_path,
        "config": cfg,
        "output": output,
        "reference": reference,
        "base": base,
        "s0": s0,
        "systems": systems.astype(np.float64, copy=False),
        "manifest": manifest,
        "source_hashes": source_hashes,
        "checkpoint_sha256": source_hashes["checkpoint"],
        "normalization_sha256": source_hashes["normalization"],
        "simulator_contract": {
            "class": "paper_c.swimmer.model.SwimmerModel",
            "implementation_sha256": source_hashes["simulator_kernel"],
            "base_config_sha256": source_hashes["base_config"],
            "integrator": base["model"]["integrator"],
            "timestep_s": float(base["model"]["timestep_s"]),
        },
    }


def _geometry_context(
    base: Mapping[str, object], simulator, theta: np.ndarray, history: Mapping[str, np.ndarray],
    landmarks: np.ndarray, system_index: int, realization: int,
):
    """Reconstruct only the nuisance states and observed history anchor.

    In contrast to the general assay helper, this function does not materialize
    candidate observations or the fresh query observation/learner target.
    """

    rng = np.random.default_rng(np.random.SeedSequence([78103, int(system_index), int(realization)]))
    initial_h = simulator.sample_initial_state(rng, base["transient_initial_state"])
    initial_e = simulator.sample_initial_state(rng, base["transient_initial_state"])
    initial_q = simulator.sample_initial_state(rng, base["transient_initial_state"])
    true_h = np.asarray([
        simulator.rollout(theta, initial_h, actions, landmarks) for actions in history.values()
    ], dtype=np.float64)
    observed_h = true_h + rng.normal(0.0, float(base["observation"]["sensor_std"]), true_h.shape)
    return initial_h, initial_e, initial_q, observed_h


def _load_runtime(contract: Mapping[str, object], device_name: str) -> dict:
    import torch
    from paper_c.swimmer.lqa_prospective import (
        SwimmerModel,
        _landmarks,
        _load_jepa,
        banks,
        particle_pool,
    )

    device = torch.device(device_name)
    torch.set_num_threads(1)
    learner, norms, _ = _load_jepa(contract["root"], contract["reference"], device)
    history, query = banks(
        float(contract["s0"]["chosen_horizon_s"]),
        float(contract["base"]["model"]["timestep_s"]),
    )
    landmarks = _landmarks(
        len(next(iter(history.values()))), int(contract["base"]["observation"]["landmark_count"])
    )
    geometry = particle_pool(
        GEOMETRY_PARTICLES, GEOMETRY_SEED, contract["base"]["persistent_prior"]
    )
    return {
        "device": device,
        "learner": learner,
        "norms": norms,
        "history": history,
        "query": query,
        "landmarks": landmarks,
        "geometry": geometry,
        "simulator": SwimmerModel(contract["base"]["model"]),
    }


def _compute_system_scores(contract: Mapping[str, object], runtime: Mapping[str, object], system_index: int):
    from paper_c.coupled_sled.posterior import posterior_from_observation
    from paper_c.swimmer.lqa_prospective import _response_bank, lqa_bank

    context = contract["manifest"].iloc[int(system_index)]
    realization, history_index = int(context.realization), int(context.history_index)
    initial_h, initial_e, initial_q, observed_h = _geometry_context(
        contract["base"], runtime["simulator"], contract["systems"][system_index],
        runtime["history"], runtime["landmarks"], system_index, realization,
    )
    true_anchor = observed_h[history_index]
    history_means = _response_bank(
        runtime["simulator"], runtime["geometry"], initial_h, runtime["history"], runtime["landmarks"]
    )[:, history_index]
    posterior = posterior_from_observation(
        true_anchor,
        history_means,
        float(contract["base"]["observation"]["sensor_std"]),
        np.asarray([1.0]),
        np.asarray([1.0]),
    ).weights
    candidate_means = _response_bank(
        runtime["simulator"], runtime["geometry"], initial_e, runtime["history"], runtime["landmarks"]
    )
    query_means = _response_bank(
        runtime["simulator"], runtime["geometry"], initial_q, runtime["query"], runtime["landmarks"]
    )
    lqa, raw_cka = lqa_bank(
        runtime["learner"], runtime["norms"], posterior, (history_index, true_anchor),
        initial_h, initial_e, initial_q, candidate_means, query_means,
        runtime["history"], runtime["query"], runtime["landmarks"], runtime["device"],
    )
    return realization, history_index, np.asarray(lqa, dtype=np.float64), np.asarray(raw_cka, dtype=np.float64)


def prepare_shard(
    root: Path,
    config_path: Path,
    shard_index: int,
    device_name: str = "cpu",
    *,
    compute_system: Callable[[Mapping[str, object], Mapping[str, object], int], tuple] | None = None,
) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    if int(shard_index) not in (0, 1):
        raise ValueError("geometry shard index must be zero or one")
    contract = _load_contract(root, config_path)
    shard_index = int(shard_index)
    system_ids = np.arange(shard_index, SYSTEMS, SHARD_COUNT, dtype=np.int64)
    shard_root = contract["output"] / "geometry_supporting" / f"shard_{shard_index}_of_2"
    score_path = shard_root / "geometry_scores.npz"
    receipt_path = shard_root / "SHARD_RECEIPT.json"
    if score_path.exists() or receipt_path.exists():
        raise RuntimeError(f"immutable geometry shard is already occupied: {shard_root}")
    runtime = {} if compute_system is not None else _load_runtime(contract, device_name)
    compute = compute_system or _compute_system_scores
    records = {field: [] for field in SCORE_FIELDS}
    for system_index in system_ids.tolist():
        realization, history_index, lqa, raw_cka = compute(contract, runtime, system_index)
        if lqa.shape != (CANDIDATES, QUERIES) or raw_cka.shape != (CANDIDATES, QUERIES):
            raise ValueError("lqa_bank must return two finite 6 x 6 candidate-by-query matrices")
        if not np.isfinite(lqa).all() or not np.isfinite(raw_cka).all():
            raise ValueError("lqa_bank returned a non-finite score")
        frozen = contract["manifest"].iloc[system_index]
        if int(realization) != int(frozen.realization) or int(history_index) != int(frozen.history_index):
            raise RuntimeError("computed geometry context differs from the frozen manifest")
        for query_index in range(QUERIES):
            for candidate_index in range(CANDIDATES):
                records["system_index"].append(system_index)
                records["realization"].append(int(realization))
                records["history_index"].append(int(history_index))
                records["query_index"].append(query_index)
                records["candidate_index"].append(candidate_index)
                records["lqa"].append(float(lqa[candidate_index, query_index]))
                records["raw_cka"].append(float(raw_cka[candidate_index, query_index]))
    values = {
        **{field: np.asarray(records[field], dtype=np.int64) for field in ROW_KEY},
        "lqa": np.asarray(records["lqa"], dtype=np.float64),
        "raw_cka": np.asarray(records["raw_cka"], dtype=np.float64),
    }
    values = _sort_scores(values)
    coverage = _validate_scores(values, system_ids, shard_index)
    _atomic_npz(score_path, values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": SHARD_STATUS,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "system_assignment": f"system_index_mod_2_equals_{shard_index}",
        "systems": coverage["systems"],
        "rows": coverage["rows"],
        "system_ids_sha256": hashlib.sha256(system_ids.astype("<i8", copy=False).tobytes()).hexdigest(),
        "row_population_sha256": coverage["row_population_sha256"],
        "scores_path": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "normalization_sha256": contract["normalization_sha256"],
        "source_hashes": contract["source_hashes"],
        "simulator_contract": contract["simulator_contract"],
        "geometry_particles": GEOMETRY_PARTICLES,
        "geometry_scramble_seed": GEOMETRY_SEED,
        "fixed_row_schema": list(SCORE_FIELDS),
        "learner_loaded": compute_system is None,
        "target_artifact_read": False,
        "learner_prediction_decoded": False,
        "learner_loss_read": False,
        "learner_gain_read": False,
        "learner_outcome_read": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _verify_shard(contract: Mapping[str, object], shard_index: int) -> tuple[dict, dict[str, np.ndarray], Path]:
    shard_root = contract["output"] / "geometry_supporting" / f"shard_{shard_index}_of_2"
    score_path, receipt_path = shard_root / "geometry_scores.npz", shard_root / "SHARD_RECEIPT.json"
    if not score_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"geometry shard {shard_index} is incomplete")
    receipt = _json(receipt_path)
    expected_system_ids = np.arange(shard_index, SYSTEMS, SHARD_COUNT, dtype=np.int64)
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": SHARD_STATUS,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "system_assignment": f"system_index_mod_2_equals_{shard_index}",
        "systems": len(expected_system_ids),
        "rows": len(expected_system_ids) * ROWS_PER_SYSTEM,
        "config_sha256": sha256_file(contract["config_path"]),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "normalization_sha256": contract["normalization_sha256"],
        "geometry_particles": GEOMETRY_PARTICLES,
        "geometry_scramble_seed": GEOMETRY_SEED,
        "target_artifact_read": False,
        "learner_prediction_decoded": False,
        "learner_loss_read": False,
        "learner_gain_read": False,
        "learner_outcome_read": False,
    }
    if any(receipt.get(name) != value for name, value in required.items()):
        raise RuntimeError(f"geometry shard {shard_index} receipt is stale or inconsistent")
    if receipt.get("source_hashes") != contract["source_hashes"]:
        raise RuntimeError(f"geometry shard {shard_index} source hashes differ from the current frozen inputs")
    if receipt.get("scores_sha256") != sha256_file(score_path):
        raise RuntimeError(f"geometry shard {shard_index} score hash mismatch")
    values = _load_npz(score_path)
    coverage = _validate_scores(values, expected_system_ids, shard_index)
    if receipt.get("row_population_sha256") != coverage["row_population_sha256"]:
        raise RuntimeError(f"geometry shard {shard_index} row population hash mismatch")
    return receipt, values, receipt_path


def merge(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    contract = _load_contract(root, config_path)
    merge_root = contract["output"] / "geometry_supporting" / "merged"
    score_path, receipt_path = merge_root / "geometry_scores.npz", merge_root / "MERGE_RECEIPT.json"
    if score_path.exists() or receipt_path.exists():
        raise RuntimeError(f"immutable geometry merge is already occupied: {merge_root}")
    shard_records = []
    shard_values = []
    for shard_index in (0, 1):
        receipt, values, shard_receipt_path = _verify_shard(contract, shard_index)
        shard_values.append(values)
        shard_records.append({
            "shard_index": shard_index,
            "receipt_sha256": sha256_file(shard_receipt_path),
            "scores_sha256": receipt["scores_sha256"],
            "row_population_sha256": receipt["row_population_sha256"],
        })
    values = {
        field: np.concatenate([shard[field] for shard in shard_values], axis=0)
        for field in SCORE_FIELDS
    }
    values = _sort_scores(values)
    coverage = _validate_scores(values, np.arange(SYSTEMS, dtype=np.int64), None)
    _atomic_npz(score_path, values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": MERGE_STATUS,
        "fixed_merge_order": [0, 1],
        "canonical_row_sort_after_fixed_merge": True,
        "systems": coverage["systems"],
        "rows": coverage["rows"],
        "queries_per_system": QUERIES,
        "candidates_per_query": CANDIDATES,
        "row_population_sha256": coverage["row_population_sha256"],
        "scores_path": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "normalization_sha256": contract["normalization_sha256"],
        "source_hashes": contract["source_hashes"],
        "simulator_contract": contract["simulator_contract"],
        "geometry_particles": GEOMETRY_PARTICLES,
        "geometry_scramble_seed": GEOMETRY_SEED,
        "shards": shard_records,
        "fixed_row_schema": list(SCORE_FIELDS),
        "target_artifact_read": False,
        "learner_prediction_decoded": False,
        "learner_loss_read": False,
        "learner_gain_read": False,
        "learner_outcome_read": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def load_merged_scores(
    root: Path, config_path: Path,
) -> tuple[dict[str, np.ndarray], dict, Path]:
    """Load and fully verify the immutable merged geometry artifact."""

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    contract = _load_contract(root, config_path)
    merge_root = contract["output"] / "geometry_supporting" / "merged"
    score_path = merge_root / "geometry_scores.npz"
    receipt_path = merge_root / "MERGE_RECEIPT.json"
    if not score_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("merged outcome-blind geometry artifact is incomplete")
    receipt = _json(receipt_path)
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": MERGE_STATUS,
        "fixed_merge_order": [0, 1],
        "canonical_row_sort_after_fixed_merge": True,
        "systems": SYSTEMS,
        "rows": SYSTEMS * ROWS_PER_SYSTEM,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "normalization_sha256": contract["normalization_sha256"],
        "geometry_particles": GEOMETRY_PARTICLES,
        "geometry_scramble_seed": GEOMETRY_SEED,
        "target_artifact_read": False,
        "learner_prediction_decoded": False,
        "learner_loss_read": False,
        "learner_gain_read": False,
        "learner_outcome_read": False,
    }
    if any(receipt.get(name) != value for name, value in required.items()):
        raise RuntimeError("merged geometry receipt has stale or invalid semantics")
    if receipt.get("source_hashes") != contract["source_hashes"]:
        raise RuntimeError("merged geometry source hashes differ from current frozen inputs")
    if Path(str(receipt.get("scores_path", ""))).resolve() != score_path.resolve():
        raise RuntimeError("merged geometry receipt points to another score artifact")
    if receipt.get("scores_sha256") != sha256_file(score_path):
        raise RuntimeError("merged geometry score hash mismatch")
    values = _load_npz(score_path)
    coverage = _validate_scores(values, np.arange(SYSTEMS, dtype=np.int64), None)
    if receipt.get("row_population_sha256") != coverage["row_population_sha256"]:
        raise RuntimeError("merged geometry row population hash mismatch")
    return values, receipt, receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("prepare-shard")
    shard.add_argument("--shard-index", type=int, required=True, choices=(0, 1))
    shard.add_argument("--device", default="cpu")
    subparsers.add_parser("merge")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    if args.command == "prepare-shard":
        value = prepare_shard(args.root, config_path, args.shard_index, args.device)
    else:
        value = merge(args.root, config_path)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
