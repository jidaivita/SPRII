"""Pre-outcome integration runner for the Paper-C prospective packet.

The CLI intentionally has no outcome-release command.  It may generate
learner-blind physical targets and frozen learner predictions, but it never
joins them.  The only module allowed to perform that join is the separately
invoked ``materialize_outcome_ledger`` function after human authorization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from paper_c.extension import prospective_models
from paper_c.extension import prospective_packet
from paper_c.extension import prospective_results
from paper_c.extension.prospective_packet import (
    CONTEXT_KEY,
    REQUIRED_AUTHORIZATION_ARTIFACTS,
    REQUIRED_CONSUMERS,
    ROW_KEY,
    build_unbiased_pair_manifest,
    freeze_outcome_authorization,
    merge_preoutcome_shards,
    merge_reference_shards,
    row_population_sha256,
    sha256_file,
    validate_arm_predictions,
    validate_preoutcome_cache,
    validate_reference_vectors,
    validate_target_rows,
    write_arm_prediction_cache,
    write_preoutcome_cache,
    write_reference_vectors,
)


STATUS_DESIGN = "PAPER_C_PROSPECTIVE_PACKET_DESIGN_FROZEN"
STATUS_POPULATION = "PAPER_C_PROSPECTIVE_FRESH_POPULATION_FROZEN"
STATUS_OLD_INPUTS = "PAPER_C_PROSPECTIVE_OLD_INPUTS_COMPLETE"
STATUS_REFERENCE = "PAPER_C_PROSPECTIVE_REFERENCE_MERGED_LEARNER_BLIND"
STATUS_FEATURES = "PAPER_C_PROSPECTIVE_FEATURES_MERGED_PREOUTCOME"
STATUS_SCORED = "PAPER_C_PROSPECTIVE_SCORES_AND_PAIRS_FROZEN"


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_config(root: Path, config_path: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("status") != "IMPLEMENTATION_AUTHORIZED_OUTCOME_LOCKED":
        raise RuntimeError("prospective packet config is not frozen")
    design = cfg.get("fresh_design", {})
    required_design = {
        "systems": 512,
        "system_seed": 78101,
        "context_seed": 78103,
        "identity_round_decimals": 12,
    }
    if any(int(design.get(name, -1)) != value for name, value in required_design.items()):
        raise RuntimeError("fresh design differs from the frozen 512-system block")
    execution = cfg.get("execution", {})
    if int(execution.get("reference_shards", -1)) != 2 or int(execution.get("feature_shards", -1)) != 2:
        raise RuntimeError("prospective packet is frozen to two reference and feature shards")
    if not cfg.get("output_root") or not cfg.get("reference_config_template"):
        raise ValueError("config must name output_root and reference_config_template")
    return cfg


def _output(root: Path, cfg: Mapping[str, object]) -> Path:
    return _root_path(root, cfg["output_root"])


def _design_receipt(root: Path, cfg: Mapping[str, object]) -> Path:
    return _output(root, cfg) / "frozen" / "DESIGN_FROZEN.json"


def _old_fit_receipt(root: Path, cfg: Mapping[str, object]) -> Path:
    return _output(root, cfg) / "old_fit" / "OLD_FIT_FROZEN.json"


def _reference_config(root: Path, cfg: Mapping[str, object]) -> dict:
    reference = copy.deepcopy(json.loads(_root_path(root, cfg["reference_config_template"]).read_text()))
    design = cfg["fresh_design"]
    reference["formal"]["system_seed"] = int(design["system_seed"])
    reference["formal"]["context_seed"] = int(design["context_seed"])
    reference["formal"]["pool_max"] = int(design["systems"])
    reference["formal"]["target_systems"] = int(design["systems"])
    reference["formal"]["contexts_per_system"] = 1
    return reference


def freeze_design(root: Path, config_path: Path) -> dict:
    """Freeze paths and hashes without generating a fresh physical system."""

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path)
    bindings = [config_path, _root_path(root, cfg["reference_config_template"])]
    bindings.extend(_root_path(root, path) for path in cfg.get("source_bindings", []))
    missing = [str(path) for path in bindings if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"design source bindings are missing: {missing}")
    receipt = {
        "schema_version": "1.0",
        "status": STATUS_DESIGN,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "fresh_design": cfg["fresh_design"],
        "source_hashes": {str(path): sha256_file(path) for path in bindings},
        "fresh_systems_generated": False,
        "fresh_target_read": False,
        "learner_outcome_read": False,
    }
    _atomic_json(_design_receipt(root, cfg), receipt)
    return receipt


def _verify_design(root: Path, config_path: Path, cfg: Mapping[str, object]) -> dict:
    path = _design_receipt(root, cfg)
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_DESIGN or receipt.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("design freeze is missing or stale")
    for raw_path, expected in receipt["source_hashes"].items():
        if sha256_file(Path(raw_path)) != expected:
            raise RuntimeError(f"design source is stale: {raw_path}")
    return receipt


def _identity(theta: np.ndarray, decimals: int) -> str:
    token = "|".join(f"{value:.{decimals}f}" for value in np.asarray(theta, dtype=np.float64))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _ensure_fresh_population(root: Path, config_path: Path, cfg: Mapping[str, object]) -> dict:
    """Materialize fresh systems only after the old-only models are frozen."""

    from paper_c.swimmer.lqa_formal import formal_context_table
    from paper_c.swimmer.lqa_prospective import system_pool

    _verify_design(root, config_path, cfg)
    old_fit = json.loads(_old_fit_receipt(root, cfg).read_text())
    if old_fit.get("status") != "PAPER_C_PROSPECTIVE_OLD_FIT_FROZEN" or old_fit.get("fresh_outcome_read") is not False:
        raise RuntimeError("fresh-system generation requires a valid old-model freeze")
    output = _output(root, cfg)
    receipt_path = output / "frozen" / "FRESH_POPULATION_FROZEN.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != STATUS_POPULATION:
            raise RuntimeError("fresh population receipt has an invalid status")
        return receipt
    reference = _reference_config(root, cfg)
    base = json.loads(_root_path(root, reference["base_config"]).read_text())
    design = cfg["fresh_design"]
    systems = system_pool(int(design["systems"]), int(design["system_seed"]), base["persistent_prior"])
    decimals = int(design["identity_round_decimals"])
    identities = [_identity(row, decimals) for row in systems]
    if len(set(identities)) != len(identities):
        raise RuntimeError("fresh system pool contains duplicate physical identities")
    known = set()
    required_pools = {"train", "select", "original_formal", "e1_fresh"}
    pools = cfg.get("disjoint_system_pools", [])
    if {item.get("name") for item in pools} != required_pools:
        raise ValueError(f"disjoint system pools must be exactly {sorted(required_pools)}")
    pool_receipts = []
    for item in pools:
        values = system_pool(int(item["systems"]), int(item["seed"]), base["persistent_prior"])
        pool_ids = {_identity(row, decimals) for row in values}
        overlap = set(identities) & pool_ids
        if overlap:
            raise RuntimeError(f"fresh systems overlap {item['name']}: {len(overlap)}")
        known |= pool_ids
        pool_receipts.append({"name": item["name"], "systems": int(item["systems"]), "seed": int(item["seed"]),
                              "identity_set_sha256": hashlib.sha256("\n".join(sorted(pool_ids)).encode()).hexdigest()})
    contexts = formal_context_table(reference)
    manifest = contexts.copy()
    for index in range(systems.shape[1]):
        manifest[f"theta_{index}"] = systems[:, index]
    manifest["physical_identity_sha256"] = identities
    systems_path = output / "frozen" / "fresh_systems.npy"
    manifest_path = output / "frozen" / "fresh_system_context_manifest.csv.gz"
    reference_path = output / "frozen" / "FRESH_REFERENCE_CONFIG.json"
    systems_path.parent.mkdir(parents=True, exist_ok=True)
    if systems_path.exists() or manifest_path.exists() or reference_path.exists():
        raise RuntimeError("partial fresh population artifacts already exist")
    temporary = systems_path.with_name(systems_path.name + f".tmp.{os.getpid()}.npy")
    np.save(temporary, systems.astype(np.float64)); os.replace(temporary, systems_path)
    _atomic_csv(manifest_path, manifest)
    _atomic_json(reference_path, reference)
    receipt = {
        "schema_version": "1.0", "status": STATUS_POPULATION,
        "systems": 512, "candidate_rows": 18432, "baseline_rows": 3072,
        "system_seed": 78101, "context_seed": 78103, "identity_round_decimals": decimals,
        "systems_sha256": sha256_file(systems_path), "manifest_sha256": sha256_file(manifest_path),
        "reference_config_sha256": sha256_file(reference_path), "known_pools": pool_receipts,
        "intersection_count": 0, "old_fit_receipt_sha256": sha256_file(_old_fit_receipt(root, cfg)),
        "learner_outcome_read": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _rows_frame(values: Mapping[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({field: np.asarray(values[field], dtype=np.int64) for field in ROW_KEY})


def _join_array(
    canonical_rows: pd.DataFrame, source_rows: pd.DataFrame, value: np.ndarray, name: str,
) -> np.ndarray:
    source = source_rows[list(ROW_KEY)].copy()
    source["_position"] = np.arange(len(source), dtype=np.int64)
    joined = canonical_rows[list(ROW_KEY)].merge(source, on=list(ROW_KEY), validate="one_to_one", how="left")
    if joined._position.isna().any() or len(joined) != len(canonical_rows):
        raise ValueError(f"{name} does not join the old canonical row population")
    return np.asarray(value)[joined._position.to_numpy(np.int64)]


def build_old_inputs(root: Path, config_path: Path) -> dict:
    """Join the already-read old outcomes and extract the missing old z_p(H)."""

    import torch
    from paper_c.coupled_sled.formal_data import load_arrays
    from paper_c.stage2.routing_intervention import _load_environment_model

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path); _verify_design(root, config_path, cfg)
    sources = cfg["old_fit_sources"]
    layer_path = _root_path(root, sources["layer_deltas"])
    routing_path = _root_path(root, sources["routing_cache"])
    formal_arrays_path = _root_path(root, sources["formal_arrays"])
    formal_rows_path = _root_path(root, sources["formal_row_manifest"])
    alignment_paths = [_root_path(root, path) for path in sources["alignment_shards"]]
    for path in [layer_path, routing_path, formal_arrays_path, formal_rows_path, *alignment_paths]:
        if not path.is_file(): raise FileNotFoundError(path)
    layer = _load_npz(layer_path)
    canonical = _rows_frame(layer).sort_values(list(ROW_KEY)).reset_index(drop=True)
    if len(canonical) != 18432 or canonical.system_index.nunique() != 512:
        raise RuntimeError("old layer population is not the frozen 512x36 population")
    align_parts = [_load_npz(path) for path in alignment_paths]
    alignment = {name: np.concatenate([part[name] for part in align_parts]) for name in align_parts[0]}
    align_rows = _rows_frame(alignment)
    routing = _load_npz(routing_path)
    history_values = routing["history_index"] if "history_index" in routing else routing["anchor_index"]
    candidate_values = routing["candidate_index"] if "candidate_index" in routing else routing["candidate_identity"].astype(np.int64)
    routing_rows = pd.DataFrame({
        "system_index": routing["system_index"].astype(np.int64),
        "history_index": history_values.astype(np.int64),
        "query_index": routing["query_index"].astype(np.int64),
        "candidate_index": candidate_values.astype(np.int64),
    })
    routing_rows = routing_rows.merge(canonical[["system_index", "realization", "history_index"]].drop_duplicates(),
                                      on=["system_index", "history_index"], validate="many_to_one")
    routing_rows = routing_rows[list(ROW_KEY)]
    realized_gain_raw = np.mean(
        (routing["normalized_target"] - routing["prediction_anchor"]) ** 2
        - (routing["normalized_target"] - routing["prediction_original"]) ** 2,
        axis=1,
    )

    arrays = load_arrays(formal_arrays_path)
    formal_rows = pd.read_csv(formal_rows_path)
    if len(formal_rows) != len(arrays.history):
        raise ValueError("old formal row manifest and arrays differ")
    baseline_mask = formal_rows.candidate_index.to_numpy(np.int64) == -1
    with np.load(_root_path(root, sources["normalization"]), allow_pickle=False) as loaded:
        history_mean = loaded["history_mean"].copy(); history_std = loaded["history_std"].copy()
    history = ((arrays.history[baseline_mask] - history_mean) / history_std).astype(np.float32)
    mask = arrays.history_mask[baseline_mask].astype(np.float32)
    history *= mask[:, :, None]
    device = torch.device("cpu"); torch.set_num_threads(1)
    model, _, before = _load_environment_model(root, cfg, "articulated", arrays, device)
    with torch.no_grad():
        z_anchor = model.persistent(torch.from_numpy(history), torch.from_numpy(mask)).cpu().numpy().astype(np.float32)
    baseline_rows = formal_rows.loc[baseline_mask, list(CONTEXT_KEY)].copy()
    baseline_rows["_position"] = np.arange(len(baseline_rows), dtype=np.int64)
    anchor_join = canonical.merge(baseline_rows, on=list(CONTEXT_KEY), validate="many_to_one")
    if len(anchor_join) != len(canonical): raise RuntimeError("old z_p(H) join is incomplete")
    z_canonical = z_anchor[anchor_join._position.to_numpy(np.int64)]
    output = _output(root, cfg) / "old_fit" / "inputs"
    rows_path = output / "old_rows.csv.gz"; arrays_path = output / "old_arrays.npz"
    _atomic_csv(rows_path, canonical)
    values = {
        "r_b_ref_a": _join_array(canonical, align_rows, alignment["r_b_ref_a"], "r_b_ref_a"),
        "r_b_ref_b": _join_array(canonical, align_rows, alignment["r_b_ref_b"], "r_b_ref_b"),
        "z_p_anchor": z_canonical,
        "delta_segment": _join_array(canonical, _rows_frame(layer), layer["delta_segment"], "delta_segment"),
        "delta_persistent": _join_array(canonical, _rows_frame(layer), layer["delta_persistent"], "delta_persistent"),
        "delta_predicted_query": _join_array(canonical, _rows_frame(layer), layer["delta_predicted"], "delta_predicted"),
        "realized_gain": _join_array(canonical, routing_rows, realized_gain_raw, "realized_gain"),
    }
    _atomic_npz(arrays_path, values)
    receipt = {
        "schema_version": "1.0", "status": STATUS_OLD_INPUTS, "rows": 18432, "systems": 512,
        "rows_path": str(rows_path), "rows_sha256": sha256_file(rows_path),
        "arrays_path": str(arrays_path), "arrays_sha256": sha256_file(arrays_path),
        "source_hashes": {str(path): sha256_file(path) for path in [layer_path, routing_path, formal_arrays_path, formal_rows_path, *alignment_paths]},
        "base_model_hashes": before, "fresh_data_read": False,
    }
    _atomic_json(output / "OLD_INPUTS_RECEIPT.json", receipt)
    return receipt


def fit_and_freeze_old_models(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path); _verify_design(root, config_path, cfg)
    output = _output(root, cfg) / "old_fit"
    inputs = json.loads((output / "inputs" / "OLD_INPUTS_RECEIPT.json").read_text())
    manifest = prospective_models.old_fit(
        Path(inputs["rows_path"]), Path(inputs["arrays_path"]), output / "models",
        fold_salt=79301, old_shuffle_salt=79303,
    )
    return prospective_models.freeze_old_fit(
        output / "models" / "OLD_FIT_MANIFEST.json", _old_fit_receipt(root, cfg),
        [config_path, Path(inputs["rows_path"]), Path(inputs["arrays_path"])],
    )


def _reference_provenance(root: Path, config_path: Path, reference: Mapping[str, object]) -> dict:
    return {
        "geometry_particles": 512,
        "particle_seed": int(reference["reference"]["geometry_scramble_seed"]),
        "ref_a_scramble_seeds": list(map(int, reference["reference"]["ref_a_scramble_seeds"])),
        "ref_b_scramble_seeds": list(map(int, reference["reference"]["ref_b_scramble_seeds"])),
        "source_hashes": {"runner_config": sha256_file(config_path),
                          "reference_config": sha256_file(_output(root, _load_config(root, config_path)) / "frozen" / "FRESH_REFERENCE_CONFIG.json")},
    }


def _candidate_mu_from_response_banks(
    true_anchor: np.ndarray,
    observed_e: np.ndarray,
    hmeans: np.ndarray,
    emeans: np.ndarray,
    qmeans: np.ndarray,
    sensor_std: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    """Compute candidate posterior means from the stream's existing banks."""

    from paper_c.coupled_sled.posterior import combine_independent_posteriors

    candidate_mu = np.empty((6, 6, 32), dtype=np.float64)
    for candidate in range(6):
        posterior = combine_independent_posteriors(
            (true_anchor, observed_e[candidate]),
            (hmeans, emeans[:, candidate]),
            float(sensor_std),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )
        for query_index in range(6):
            raw_mu = np.einsum("n,nf->f", posterior.weights, qmeans[:, query_index])
            candidate_mu[candidate, query_index] = (raw_mu - target_mean) / target_std
    return candidate_mu


def _reference_stream_with_candidate_mu(
    base: Mapping[str, object],
    simulator,
    particles: np.ndarray,
    history: Mapping[object, np.ndarray],
    query: Mapping[object, np.ndarray],
    landmarks: np.ndarray,
    true_anchor: np.ndarray,
    observed_e: np.ndarray,
    initial_h,
    initial_e,
    initial_q,
    history_index: int,
    outcome_levels: Sequence[int],
    outcome_seed: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse one response-bank materialization for V_B and candidate mu."""

    from paper_c.swimmer.lqa_prospective import _response_bank, reference_stream_from_response_banks

    hmeans = _response_bank(simulator, particles, initial_h, history, landmarks)[:, history_index]
    emeans = _response_bank(simulator, particles, initial_e, history, landmarks)
    qmeans = _response_bank(simulator, particles, initial_q, query, landmarks)
    values, _, _, _, _ = reference_stream_from_response_banks(
        base,
        true_anchor,
        hmeans,
        emeans,
        qmeans.transpose(1, 0, 2),
        outcome_levels,
        outcome_seed,
        target_std,
    )
    candidate_mu = _candidate_mu_from_response_banks(
        true_anchor,
        observed_e,
        hmeans,
        emeans,
        qmeans,
        float(base["observation"]["sensor_std"]),
        target_mean,
        target_std,
    )
    return values, candidate_mu


def _reference_worker_system_ids(worker_index: int, worker_count: int, expected_systems: int = 512) -> np.ndarray:
    """Return the frozen ``system_index % worker_count`` partition.

    An even worker count guarantees every worker lies wholly within one of the
    frozen parity-owned top-level shards.  Empty workers are rejected so a
    mistyped launch cannot appear successful.
    """

    worker_index, worker_count = int(worker_index), int(worker_count)
    if worker_count < 2 or worker_count % 2:
        raise ValueError("reference worker count must be positive, even, and at least two")
    if worker_count > expected_systems or not 0 <= worker_index < worker_count:
        raise ValueError("reference worker index/count is outside the frozen population")
    systems = np.arange(worker_index, expected_systems, worker_count, dtype=np.int64)
    if not len(systems) or not np.all(systems % worker_count == worker_index):
        raise RuntimeError("reference worker assignment is empty or inconsistent")
    if not np.all(systems % 2 == worker_index % 2):
        raise RuntimeError("reference worker crosses a frozen parity owner")
    return systems


def _execution_supersession(
    execution_id: str | None,
    supersedes_execution_ids: Sequence[str],
    assignment: str,
) -> dict:
    supersedes = [str(value).strip() for value in supersedes_execution_ids]
    if any(not value for value in supersedes) or len(set(supersedes)) != len(supersedes):
        raise ValueError("superseded execution ids must be unique non-empty strings")
    normalized_id = None if execution_id is None else str(execution_id).strip()
    if execution_id is not None and not normalized_id:
        raise ValueError("execution id must be non-empty when supplied")
    if normalized_id is not None and normalized_id in supersedes:
        raise ValueError("an execution cannot supersede itself")
    return {
        "execution_id": normalized_id,
        "supersedes_execution_ids": supersedes,
        "supersession_reason": "deterministic_engineering_acceleration" if supersedes else None,
        "system_assignment": assignment,
        "scientific_inputs_changed": False,
        "scientific_seeds_changed": False,
        "particle_or_endpoint_definition_changed": False,
        "learner_outcome_read": False,
    }


def _prepare_reference_partition(
    root: Path,
    config_path: Path,
    system_ids: np.ndarray,
    shard_root: Path,
    receipt_identity: Mapping[str, object],
) -> dict:
    """Compute one deterministic learner-blind reference partition."""

    from paper_c.coupled_sled.posterior import posterior_from_observation
    from paper_c.swimmer.lqa_prospective import (
        SwimmerModel, _context_nuisance, _landmarks, _response_bank, _response_jacobian_bank,
        accessibility_bank, banks, particle_pool, prior_log_scale,
    )

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path); _ensure_fresh_population(root, config_path, cfg)
    system_ids = np.asarray(system_ids, dtype=np.int64)
    if system_ids.ndim != 1 or not len(system_ids) or len(np.unique(system_ids)) != len(system_ids):
        raise ValueError("reference partition must name unique frozen systems")
    if np.any(system_ids < 0) or np.any(system_ids >= 512):
        raise ValueError("reference partition contains a system outside the frozen population")
    output = _output(root, cfg)
    reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text())
    base = json.loads(_root_path(root, reference["base_config"]).read_text())
    s0 = json.loads(_root_path(root, reference["s0_receipt"]).read_text())
    systems = np.load(output / "frozen" / "fresh_systems.npy", allow_pickle=False)
    contexts = pd.read_csv(output / "frozen" / "fresh_system_context_manifest.csv.gz")
    model_root = _root_path(root, reference["s2_models"])
    with np.load(model_root / "train_only_normalization.npz", allow_pickle=False) as loaded:
        norms = {name: loaded[name].copy() for name in loaded.files}
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    simulator = SwimmerModel(base["model"])
    records = []; mu_a = []; mu_b = []; local_rows = []; scramble_rows = {f"{stream}_vb_scramble{k}": [] for stream in ("ref_a", "ref_b") for k in range(4)}
    target_records = []; target_values = []
    for system_index in system_ids.tolist():
        context = contexts.iloc[system_index]; realization = int(context.realization); history_index = int(context.history_index)
        ih, ie, iq, observed_h, observed_e, observed_q = _context_nuisance(
            base, simulator, systems[system_index], history, query, landmarks, 78103, system_index, realization,
        )
        true_anchor = observed_h[history_index]
        for query_index in range(6):
            target_records.append((system_index, realization, history_index, query_index))
            target_values.append((observed_q[query_index] - norms["target_mean"]) / norms["target_std"])
        stream_value = {}; stream_mu = {}
        for stream, seeds in (("ref_a", reference["reference"]["ref_a_scramble_seeds"]),
                              ("ref_b", reference["reference"]["ref_b_scramble_seeds"])):
            values = []; means = []
            for seed in map(int, seeds):
                particles = particle_pool(max(reference["reference"]["particle_levels"]), seed, base["persistent_prior"])
                value, candidate_mu = _reference_stream_with_candidate_mu(
                    base, simulator, particles, history, query, landmarks, true_anchor, observed_e,
                    ih, ie, iq, history_index,
                    tuple(reference["reference"]["outcome_levels"]), seed + 1_000_003 * system_index,
                    norms["target_mean"], norms["target_std"],
                )
                values.append(value[-1])
                means.append(candidate_mu)
            stream_value[stream] = np.asarray(values)
            stream_mu[stream] = np.mean(means, axis=0)
        geometry = particle_pool(512, int(reference["reference"]["geometry_scramble_seed"]), base["persistent_prior"])
        hmean = _response_bank(simulator, geometry, ih, history, landmarks)[:, history_index]
        posterior = posterior_from_observation(true_anchor, hmean, float(base["observation"]["sensor_std"]), np.asarray([1.0]), np.asarray([1.0])).weights
        scale = prior_log_scale(base["persistent_prior"])
        candidate_means, candidate_jac = _response_jacobian_bank(simulator, geometry, ie, history, landmarks, scale, float(reference["reference"]["finite_difference_log_step"]))
        _, query_jac = _response_jacobian_bank(simulator, geometry, iq, query, landmarks, scale, float(reference["reference"]["finite_difference_log_step"]))
        _, local = accessibility_bank(geometry, posterior, candidate_means, candidate_jac, query_jac, norms["target_std"], float(base["observation"]["sensor_std"]), scale)
        for query_index in range(6):
            for candidate in range(6):
                records.append((system_index, realization, history_index, query_index, candidate))
                mu_a.append(stream_mu["ref_a"][candidate, query_index]); mu_b.append(stream_mu["ref_b"][candidate, query_index]); local_rows.append(local[candidate, query_index])
                for stream in ("ref_a", "ref_b"):
                    for k in range(4): scramble_rows[f"{stream}_vb_scramble{k}"].append(stream_value[stream][k, candidate, query_index])
    matrix = np.asarray(records, dtype=np.int64)
    reference_values = {field: matrix[:, i] for i, field in enumerate(ROW_KEY)}
    reference_values.update({"mu_ref_a": np.asarray(mu_a, dtype=np.float32), "mu_ref_b": np.asarray(mu_b, dtype=np.float32), "local_value": np.asarray(local_rows, dtype=np.float64), **{name: np.asarray(value, dtype=np.float64) for name, value in scramble_rows.items()}})
    shard_root = Path(shard_root); shard_root.mkdir(parents=True, exist_ok=True)
    ref_path = shard_root / "fresh_reference_vectors.npz"
    write_reference_vectors(ref_path, reference_values, _reference_provenance(root, config_path, reference))
    target_matrix = np.asarray(target_records, dtype=np.int64)
    target_payload = {field: target_matrix[:, i] for i, field in enumerate(CONTEXT_KEY)}
    target_payload["normalized_target"] = np.asarray(target_values, dtype=np.float32)
    target_keys = np.column_stack([target_payload[field] for field in CONTEXT_KEY])
    shard_systems = np.unique(target_payload["system_index"])
    if (
        not np.array_equal(shard_systems, np.sort(system_ids))
        or len(np.unique(target_keys, axis=0)) != len(system_ids) * 6
        or target_payload["normalized_target"].shape != (len(system_ids) * 6, 32)
        or not np.isfinite(target_payload["normalized_target"]).all()
    ):
        raise RuntimeError("learner-blind target shard has invalid coverage")
    target_path = shard_root / "fresh_targets_learner_blind.npz"; _atomic_npz(target_path, target_payload)
    from paper_c.swimmer import lqa_prospective as reference_kernel

    receipt = {"schema_version": "1.0", **dict(receipt_identity), "systems": len(system_ids),
               "system_ids_sha256": hashlib.sha256(system_ids.astype("<i8", copy=False).tobytes()).hexdigest(),
               "config_sha256": sha256_file(config_path),
               "implementation_sha256": sha256_file(Path(__file__)),
               "reference_kernel_sha256": sha256_file(Path(reference_kernel.__file__)),
               "reference_sha256": sha256_file(ref_path), "target_sha256": sha256_file(target_path),
               "learner_loaded": False, "learner_outcome_read": False}
    _atomic_json(shard_root / "SHARD_RECEIPT.json", receipt)
    return receipt


def prepare_reference_shard(root: Path, config_path: Path, shard_index: int) -> dict:
    """Compute one original parity-owned ref-A/ref-B shard."""

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path)
    if shard_index not in (0, 1):
        raise ValueError("reference shard index must be zero or one")
    system_ids = np.arange(shard_index, 512, 2, dtype=np.int64)
    return _prepare_reference_partition(
        root,
        config_path,
        system_ids,
        _output(root, cfg) / "reference" / f"shard_{shard_index}_of_2",
        {
            "status": "PROSPECTIVE_REFERENCE_SHARD_COMPLETE_LEARNER_BLIND",
            "shard_index": shard_index,
            "execution": _execution_supersession(None, (), "system_index_mod_2"),
        },
    )


def prepare_reference_worker(
    root: Path,
    config_path: Path,
    worker_index: int,
    worker_count: int,
    execution_id: str | None = None,
    supersedes_execution_ids: Sequence[str] = (),
) -> dict:
    """Compute one accelerated worker with ``system_index % worker_count`` ownership."""

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path)
    system_ids = _reference_worker_system_ids(worker_index, worker_count)
    owner = int(worker_index) % 2
    worker_root = _output(root, cfg) / "reference" / "workers" / f"worker_{worker_index:02d}_of_{worker_count:02d}"
    aborted_path = _output(root, cfg) / "frozen" / "REFERENCE_ABORTED_EXECUTION_RECEIPT.json"
    if not aborted_path.is_file():
        raise FileNotFoundError("aborted slow-reference receipt must be frozen before accelerated workers")
    aborted = json.loads(aborted_path.read_text())
    if (
        aborted.get("status") != "PROSPECTIVE_REFERENCE_EXECUTION_ABORTED_OUTCOME_BLIND"
        or aborted.get("old_jobs_stopped") is not True
        or aborted.get("learner_outcome_read") is not False
        or list(supersedes_execution_ids) != aborted.get("aborted_job_handles")
    ):
        raise RuntimeError("accelerated worker is outside the frozen supersession chain")
    return _prepare_reference_partition(
        root,
        config_path,
        system_ids,
        worker_root,
        {
            "status": "PROSPECTIVE_REFERENCE_WORKER_COMPLETE_LEARNER_BLIND",
            "worker_index": int(worker_index),
            "worker_count": int(worker_count),
            "top_level_shard_index": owner,
            "execution": _execution_supersession(
                execution_id,
                supersedes_execution_ids,
                f"system_index_mod_{int(worker_count)}_equals_{int(worker_index)}",
            ),
        },
    )


def _merge_target_shards(paths: Sequence[Path], output_path: Path) -> dict:
    parts = [_load_npz(path) for path in paths]
    for index, part in enumerate(parts):
        systems = np.unique(part["system_index"])
        if any(int(system) % 2 != index for system in systems): raise ValueError("target shard violates modulo ownership")
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = np.lexsort(tuple(merged[field] for field in reversed(CONTEXT_KEY)))
    merged = {name: value[order] for name, value in merged.items()}
    validate_target_rows(merged, 512); _atomic_npz(output_path, merged)
    receipt = {"schema_version": "1.0", "status": "PROSPECTIVE_TARGETS_MERGED_LEARNER_BLIND", "rows": 3072,
               "fixed_merge_order": [0, 1], "shard_sha256": [sha256_file(path) for path in paths], "artifact_sha256": sha256_file(output_path)}
    _atomic_json(Path(str(output_path) + ".receipt.json"), receipt); return receipt


def _validate_target_subset(values: Mapping[str, np.ndarray], expected_system_ids: Sequence[int]) -> dict:
    expected = np.asarray(expected_system_ids, dtype=np.int64)
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if set(arrays) != {*CONTEXT_KEY, "normalized_target"}:
        raise ValueError("target worker has an invalid array schema")
    keys = np.column_stack([arrays[field].astype(np.int64, copy=False) for field in CONTEXT_KEY])
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("target worker context keys are duplicated")
    systems = np.unique(keys[:, 0])
    if not np.array_equal(systems, np.sort(expected)) or len(keys) != len(expected) * 6:
        raise ValueError("target worker has incomplete or foreign system coverage")
    for system in systems:
        cell = keys[keys[:, 0] == system]
        if len(np.unique(cell[:, 1:3], axis=0)) != 1 or set(cell[:, 3]) != set(range(6)):
            raise ValueError(f"target worker has incomplete context for system {int(system)}")
    target = arrays["normalized_target"]
    if target.shape != (len(keys), 32) or not np.isfinite(target).all():
        raise ValueError("target worker values must be finite 32-vectors")
    return {"systems": len(systems), "rows": len(keys)}


def _sorted_by_key(values: Mapping[str, np.ndarray], key: Sequence[str]) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(value) for name, value in values.items()}
    order = np.lexsort(tuple(arrays[field] for field in reversed(tuple(key))))
    return {name: value[order] for name, value in arrays.items()}


def merge_reference_workers(root: Path, config_path: Path, worker_count: int) -> dict:
    """Merge deterministic workers into the two original parity shards.

    This command is deliberately separate from ``merge-reference``.  It first
    reconstructs the exact frozen shard-0/shard-1 ownership and only then lets
    the existing fixed ``[0, 1]`` merger operate.
    """

    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = _load_config(root, config_path)
    worker_count = int(worker_count)
    _reference_worker_system_ids(0, worker_count)
    output = _output(root, cfg)
    aborted_path = output / "frozen" / "REFERENCE_ABORTED_EXECUTION_RECEIPT.json"
    if not aborted_path.is_file():
        raise FileNotFoundError("aborted slow-reference receipt must be frozen before worker merge")
    aborted = json.loads(aborted_path.read_text())
    superseded_handles = aborted.get("aborted_job_handles")
    if (
        aborted.get("status") != "PROSPECTIVE_REFERENCE_EXECUTION_ABORTED_OUTCOME_BLIND"
        or not isinstance(superseded_handles, list)
        or len(superseded_handles) != 3
        or aborted.get("old_jobs_stopped") is not True
        or aborted.get("learner_outcome_read") is not False
    ):
        raise RuntimeError("aborted slow-reference receipt is not valid for worker merge")
    reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text())
    expected_impl = sha256_file(Path(__file__))
    from paper_c.swimmer import lqa_prospective as reference_kernel
    expected_kernel = sha256_file(Path(reference_kernel.__file__))
    top_receipts = []
    for owner in (0, 1):
        expected_owner_systems = np.arange(owner, 512, 2, dtype=np.int64)
        reference_parts = []
        target_parts = []
        worker_records = []
        worker_indices = list(range(owner, worker_count, 2))
        for worker_index in worker_indices:
            expected_systems = _reference_worker_system_ids(worker_index, worker_count)
            worker_root = output / "reference" / "workers" / f"worker_{worker_index:02d}_of_{worker_count:02d}"
            receipt_path = worker_root / "SHARD_RECEIPT.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"reference worker receipt is missing: {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            required = {
                "status": "PROSPECTIVE_REFERENCE_WORKER_COMPLETE_LEARNER_BLIND",
                "worker_index": worker_index,
                "worker_count": worker_count,
                "top_level_shard_index": owner,
                "implementation_sha256": expected_impl,
                "reference_kernel_sha256": expected_kernel,
                "config_sha256": sha256_file(config_path),
                "learner_outcome_read": False,
            }
            if any(receipt.get(name) != value for name, value in required.items()):
                raise RuntimeError(f"reference worker {worker_index} receipt is stale or inconsistent")
            expected_execution = {
                "execution_id": f"paper-c-fresh-reference-v2-worker-{worker_index:02d}-of-24",
                "supersedes_execution_ids": superseded_handles,
                "supersession_reason": "deterministic_engineering_acceleration",
                "system_assignment": f"system_index_mod_24_equals_{worker_index}",
                "scientific_inputs_changed": False,
                "scientific_seeds_changed": False,
                "particle_or_endpoint_definition_changed": False,
                "learner_outcome_read": False,
            }
            if receipt.get("execution") != expected_execution:
                raise RuntimeError(f"reference worker {worker_index} is outside the frozen supersession chain")
            ref_path = worker_root / "fresh_reference_vectors.npz"
            target_path = worker_root / "fresh_targets_learner_blind.npz"
            if receipt.get("reference_sha256") != sha256_file(ref_path) or receipt.get("target_sha256") != sha256_file(target_path):
                raise RuntimeError(f"reference worker {worker_index} artifact hash mismatch")
            ref_part = _load_npz(ref_path)
            target_part = _load_npz(target_path)
            validate_reference_vectors(ref_part, expected_system_ids=expected_systems)
            _validate_target_subset(target_part, expected_systems)
            reference_parts.append(ref_part)
            target_parts.append(target_part)
            worker_records.append({
                "worker_index": worker_index,
                "worker_receipt_sha256": sha256_file(receipt_path),
                "reference_sha256": receipt["reference_sha256"],
                "target_sha256": receipt["target_sha256"],
                "execution": receipt.get("execution"),
            })
        if any(set(part) != set(reference_parts[0]) for part in reference_parts[1:]):
            raise ValueError("reference workers have different array schemas")
        merged_reference = _sorted_by_key(
            {name: np.concatenate([part[name] for part in reference_parts]) for name in reference_parts[0]},
            ROW_KEY,
        )
        merged_target = _sorted_by_key(
            {name: np.concatenate([part[name] for part in target_parts]) for name in target_parts[0]},
            CONTEXT_KEY,
        )
        validate_reference_vectors(merged_reference, expected_system_ids=expected_owner_systems)
        _validate_target_subset(merged_target, expected_owner_systems)
        shard_root = output / "reference" / f"shard_{owner}_of_2"
        ref_output = shard_root / "fresh_reference_vectors.npz"
        target_output = shard_root / "fresh_targets_learner_blind.npz"
        receipt_output = shard_root / "SHARD_RECEIPT.json"
        occupied = [path for path in (ref_output, Path(str(ref_output) + ".receipt.json"), target_output, receipt_output) if path.exists()]
        if occupied:
            raise RuntimeError(f"top-level reference shard is immutable or partially occupied: {occupied}")
        shard_root.mkdir(parents=True, exist_ok=True)
        write_reference_vectors(ref_output, merged_reference, _reference_provenance(root, config_path, reference))
        _atomic_npz(target_output, merged_target)
        top_receipt = {
            "schema_version": "1.0",
            "status": "PROSPECTIVE_REFERENCE_SHARD_COMPLETE_LEARNER_BLIND",
            "shard_index": owner,
            "systems": 256,
            "worker_count": worker_count,
            "fixed_worker_order": worker_indices,
            "system_merge_order": list(map(int, expected_owner_systems)),
            "reference_sha256": sha256_file(ref_output),
            "target_sha256": sha256_file(target_output),
            "implementation_sha256": expected_impl,
            "reference_kernel_sha256": expected_kernel,
            "config_sha256": sha256_file(config_path),
            "worker_records": worker_records,
            "execution": {
                "mode": "deterministic_worker_supersession",
                "worker_assignment": f"system_index_mod_{worker_count}",
                "reconstructed_top_level_assignment": "system_index_mod_2",
                "scientific_inputs_changed": False,
                "scientific_seeds_changed": False,
                "particle_or_endpoint_definition_changed": False,
                "learner_outcome_read": False,
            },
            "learner_loaded": False,
            "learner_outcome_read": False,
        }
        _atomic_json(receipt_output, top_receipt)
        top_receipts.append({"shard_index": owner, "receipt_sha256": sha256_file(receipt_output)})
    merge_receipt = {
        "schema_version": "1.0",
        "status": "PROSPECTIVE_REFERENCE_WORKERS_RECONSTRUCTED_TWO_SHARDS",
        "worker_count": worker_count,
        "fixed_top_level_order": [0, 1],
        "top_level_receipts": top_receipts,
        "scientific_identity_unchanged": True,
        "learner_outcome_read": False,
    }
    merge_path = output / "reference" / "workers" / f"WORKERS_{worker_count:02d}_MERGED_TO_TWO_SHARDS.json"
    _atomic_json(merge_path, merge_receipt)
    return merge_receipt


def merge_reference(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path)
    output = _output(root, cfg); reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text())
    refs = [output / "reference" / f"shard_{i}_of_2" / "fresh_reference_vectors.npz" for i in (0, 1)]
    targets = [output / "reference" / f"shard_{i}_of_2" / "fresh_targets_learner_blind.npz" for i in (0, 1)]
    merged_ref = output / "reference" / "merged" / "fresh_reference_vectors.npz"
    ref_receipt = merge_reference_shards(refs, merged_ref, _reference_provenance(root, config_path, reference), 512)
    merged_target = output / "reference" / "merged" / "fresh_targets_learner_blind.npz"
    target_receipt = _merge_target_shards(targets, merged_target)
    receipt = {"schema_version": "1.0", "status": STATUS_REFERENCE, "reference_sha256": sha256_file(merged_ref), "target_sha256": sha256_file(merged_target),
               "reference_receipt_sha256": sha256_file(Path(str(merged_ref) + ".receipt.json")), "target_receipt_sha256": sha256_file(Path(str(merged_target) + ".receipt.json")), "learner_outcome_read": False}
    _atomic_json(output / "reference" / "merged" / "MERGE_RECEIPT.json", receipt); return receipt


def _feature_forward(model, norms: Mapping[str, np.ndarray], arrays, device, batch_size: int = 1024) -> dict[str, np.ndarray]:
    import torch
    history = ((arrays.history - norms["history_mean"]) / norms["history_std"]).astype(np.float32)
    mask = arrays.history_mask.astype(np.float32); history *= mask[:, :, None]
    query = ((arrays.query_action - norms["query_mean"]) / norms["query_std"]).astype(np.float32)
    result = {name: [] for name in ("segment", "persistent", "query", "predicted", "prediction")}
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            h = torch.from_numpy(history[start:start + batch_size]).to(device); m = torch.from_numpy(mask[start:start + batch_size]).to(device); q = torch.from_numpy(query[start:start + batch_size]).to(device)
            encoded = model.segment_encoder(h) * m[:, :, None]; persistent = model.aggregate_encoded(encoded, m); qembed = model.query_encoder(q)
            predicted = model.latent_predictor(torch.cat((persistent, qembed), dim=1)); prediction = model.target_decoder(predicted)
            for name, value in (("segment", encoded[:, 1]), ("persistent", persistent), ("query", qembed), ("predicted", predicted), ("prediction", prediction)):
                result[name].append(value.cpu().numpy())
    return {name: np.concatenate(value).astype(np.float32) for name, value in result.items()}


def prepare_feature_shard(root: Path, config_path: Path, shard_index: int, device_name: str = "cpu") -> dict:
    import torch
    from paper_c.stage2.fresh_articulated_prospective import _all_candidate_manifest
    from paper_c.swimmer.lqa_evaluate import build_unique_learner_rows
    from paper_c.swimmer.lqa_prospective import _load_jepa

    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path); _ensure_fresh_population(root, config_path, cfg)
    if shard_index not in (0, 1): raise ValueError("feature shard index must be zero or one")
    output = _output(root, cfg); reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text())
    manifest = pd.read_csv(output / "frozen" / "fresh_system_context_manifest.csv.gz")
    contexts = manifest[manifest.system_index % 2 == shard_index][["system_index", "realization", "history_index"]]
    pairs = _all_candidate_manifest(contexts); baseline_arrays, baseline_table, candidate_arrays, candidate_table = build_unique_learner_rows(root, reference, pairs)
    device = torch.device(device_name); torch.set_num_threads(1); model, norms, training = _load_jepa(root, reference, device)
    baseline = _feature_forward(model, norms, baseline_arrays, device); candidate = _feature_forward(model, norms, candidate_arrays, device)
    lookup = {(int(row.system_index), int(row.realization), int(row.history_index), int(row.query_index)): i for i, row in enumerate(baseline_table.itertuples(index=False))}
    bp = np.asarray([lookup[(int(row.system_index), int(row.realization), int(row.history_index), int(row.query_index))] for row in candidate_table.itertuples(index=False)], dtype=np.int64)
    values = {field: candidate_table[field].to_numpy(np.int64) for field in ROW_KEY}
    values.update({"prediction_anchor": baseline["prediction"][bp], "prediction_candidate": candidate["prediction"], "z_p_anchor": baseline["persistent"][bp],
                   "delta_segment": candidate["segment"], "delta_persistent": candidate["persistent"] - baseline["persistent"][bp],
                   "delta_predicted_query": candidate["predicted"] - baseline["predicted"][bp], "query_embedding": candidate["query"], "predicted_latent_full": candidate["predicted"]})
    shard_root = output / "features" / f"shard_{shard_index}_of_2"; path = shard_root / "preoutcome_cache.npz"
    provenance = {"base_checkpoint_sha256": training["checkpoint_hashes"]["jepa"], "implementation_sha256": sha256_file(Path(__file__)),
                  "source_hashes": {"design": sha256_file(_design_receipt(root, cfg)), "population": sha256_file(output / "frozen" / "FRESH_POPULATION_FROZEN.json")}}
    receipt = write_preoutcome_cache(path, values, provenance)
    _atomic_json(shard_root / "SHARD_RECEIPT.json", {"schema_version": "1.0", "status": "PROSPECTIVE_FEATURE_SHARD_COMPLETE_PREOUTCOME", "shard_index": shard_index,
                 "systems": 256, "cache_sha256": sha256_file(path), "target_persisted": False, "learner_outcome_read": False})
    return receipt


def _build_e3_predictions(root: Path, cfg: Mapping[str, object], cache_path: Path, output_path: Path, device_name: str) -> dict:
    import torch
    from paper_c.stage2.delta_gated_isolation import DeltaGatedAdapter
    from paper_c.swimmer.lqa_prospective import _load_jepa

    cache = _load_npz(cache_path); rows = _rows_frame(cache); donor = prospective_models.coherent_cell_derangement(rows, 79511)
    output = _output(root, cfg); reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text())
    device = torch.device(device_name); model, _, _ = _load_jepa(root, reference, device); model.eval()
    table_path = _root_path(root, cfg["e3"]["checkpoint_table"]); table = pd.read_csv(table_path)
    table = table[table.arm.isin(["true", "shuffled"]) & table.seed.isin([86101, 86103, 86107])]
    if set(zip(table.arm.astype(str), table.seed.astype(int))) != {(arm, seed) for arm in ("true", "shuffled") for seed in (86101, 86103, 86107)}:
        raise RuntimeError("E3 checkpoint table is not the frozen six-checkpoint product")
    records = {field: [] for field in ROW_KEY}; arms = []; seeds = []; predictions = []
    with torch.no_grad():
        for row in table.sort_values(["arm", "seed"]).itertuples(index=False):
            checkpoint_path = _root_path(root, row.checkpoint)
            if sha256_file(checkpoint_path) != str(row.checkpoint_sha256):
                raise RuntimeError(f"E3 checkpoint hash mismatch: {row.checkpoint}")
            adapter = DeltaGatedAdapter().to(device); adapter.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)); adapter.eval()
            delta = cache["delta_persistent"] if row.arm == "true" else cache["delta_persistent"][donor]
            chunks = []
            for start in range(0, len(delta), 1024):
                d = torch.from_numpy(delta[start:start + 1024]).to(device); q = torch.from_numpy(cache["query_embedding"][start:start + 1024]).to(device); z = torch.from_numpy(cache["predicted_latent_full"][start:start + 1024]).to(device)
                chunks.append(model.target_decoder(z + adapter(d, q)).cpu().numpy())
            for field in ROW_KEY: records[field].append(cache[field])
            arms.append(np.full(len(delta), str(row.arm), dtype="U16")); seeds.append(np.full(len(delta), int(row.seed), dtype=np.int64)); predictions.append(np.concatenate(chunks).astype(np.float32))
    values = {field: np.concatenate(records[field]) for field in ROW_KEY}; values.update({"arm": np.concatenate(arms), "optimization_seed": np.concatenate(seeds), "prediction": np.concatenate(predictions)})
    map_path = output_path.parent / "fresh_e3_shuffle_map.npz"; _atomic_npz(map_path, {"donor_position": donor.astype(np.int64)})
    provenance = {"checkpoint_table_sha256": sha256_file(table_path), "implementation_sha256": sha256_file(Path(__file__)), "shuffle_map_sha256": sha256_file(map_path), "source_hashes": {"base_cache": sha256_file(cache_path)}}
    return write_arm_prediction_cache(output_path, values, cache, provenance)


def merge_features(root: Path, config_path: Path, device_name: str = "cpu") -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path); output = _output(root, cfg)
    shards = [output / "features" / f"shard_{i}_of_2" / "preoutcome_cache.npz" for i in (0, 1)]
    reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG.json").read_text()); training = json.loads((_root_path(root, reference["s2_models"]) / "s2_training_receipt.json").read_text())
    provenance = {"base_checkpoint_sha256": training["checkpoint_hashes"]["jepa"], "implementation_sha256": sha256_file(Path(__file__)), "source_hashes": {"config": sha256_file(config_path)}}
    merged = output / "features" / "merged" / "preoutcome_cache.npz"; base_receipt = merge_preoutcome_shards(shards, merged, provenance, 512)
    arm_path = output / "features" / "merged" / "e3_prediction_cache.npz"; arm_receipt = _build_e3_predictions(root, cfg, merged, arm_path, device_name)
    receipt = {"schema_version": "1.0", "status": STATUS_FEATURES, "base_cache_sha256": sha256_file(merged), "e3_cache_sha256": sha256_file(arm_path),
               "base_receipt_sha256": sha256_file(Path(str(merged) + ".receipt.json")), "e3_receipt_sha256": sha256_file(Path(str(arm_path) + ".receipt.json")), "learner_outcome_read": False}
    _atomic_json(output / "features" / "merged" / "MERGE_RECEIPT.json", receipt); return receipt


def score_and_build_pairs(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path); output = _output(root, cfg)
    reference_path = output / "reference" / "merged" / "fresh_reference_vectors.npz"; feature_path = output / "features" / "merged" / "preoutcome_cache.npz"
    reference = _load_npz(reference_path); features = _load_npz(feature_path); validate_reference_vectors(reference, 512); validate_preoutcome_cache(features, 512)
    if row_population_sha256(reference) != row_population_sha256(features): raise RuntimeError("reference and feature row populations differ")
    scalar = pd.DataFrame({field: reference[field] for field in ROW_KEY}); scalar["local_value"] = reference["local_value"]
    for stream in ("ref_a", "ref_b"):
        for k in range(4): scalar[f"{stream}_vb_scramble{k}"] = reference[f"{stream}_vb_scramble{k}"]
    _, pairs = build_unbiased_pair_manifest(scalar, float(cfg["pairing"]["sesoi"]), salt=79101)
    pair_dir = output / "selector"; pair_path = pair_dir / "selector_pair_manifest.csv.gz"; _atomic_csv(pair_path, pairs)
    old_manifest = json.loads((_output(root, cfg) / "old_fit" / "models" / "OLD_FIT_MANIFEST.json").read_text())
    bundles = {item["logical_name"]: prospective_models.load_bundle(Path(item["path"])) for item in old_manifest["formal_models"]}
    rows = _rows_frame(features); r_b = {stream: reference[f"mu_{stream}"] - features["prediction_anchor"] for stream in ("ref_a", "ref_b")}
    probe_predictions = {}
    for stream in ("ref_b", "ref_a"):
        models = {name: bundles[f"probe_{stream}_{name}"] for name in ("family", "anchor", "shuffle", "true")}
        predicted, _ = prospective_models.predict_probe_models(models, rows, features["z_p_anchor"], features["delta_persistent"], 79307)
        for name, value in predicted.items(): probe_predictions[f"probe_{stream}_{name}"] = value
    realization_models = {name: bundles[f"realization_{name}"] for name in ("family", "upstream", "full")}
    scores, donor = prospective_models.predict_realization_models(realization_models, rows, features["delta_segment"], features["delta_persistent"], features["delta_predicted_query"], 79401)
    score_values = {field: features[field] for field in ROW_KEY}; score_values.update(r_b); score_values.update(probe_predictions); score_values.update({f"score_{name}": value for name, value in scores.items()}); score_values["permutation_donor_position"] = donor.astype(np.int64)
    score_path = pair_dir / "fresh_preoutcome_scores.npz"; _atomic_npz(score_path, score_values)
    orientations = {}
    for name in ("full", "family", "upstream", "permutation"):
        table = prospective_models.orient_pairs(rows, scores[name], pairs, name, 1e-12, 79103); path = pair_dir / f"orientation_{name}.csv.gz"; _atomic_csv(path, table); orientations[name] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {"schema_version": "1.0", "status": STATUS_SCORED, "pairs": len(pairs), "contributing_systems": int(pairs.system_index.nunique()),
               "pair_manifest": str(pair_path), "pair_manifest_sha256": sha256_file(pair_path), "scores_sha256": sha256_file(score_path), "orientations": orientations,
               "fresh_target_read": False, "learner_outcome_read": False}
    _atomic_json(pair_dir / "SCORES_AND_PAIRS_FROZEN.json", receipt); return receipt


def _write_manifest(path: Path, status: str, values: Mapping[str, object]) -> None:
    _atomic_json(path, {"schema_version": "1.0", "status": status, **values})


def _freeze_consumer_implementations(output: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Bind each authorized consumer to the exact evaluator and model-helper bytes."""

    results_path = Path(prospective_results.__file__).resolve()
    models_path = Path(prospective_models.__file__).resolve()
    hashes: dict[str, str] = {}
    manifests: dict[str, dict[str, str]] = {}
    for logical_name in sorted(REQUIRED_CONSUMERS):
        path = Path(output) / "authorization" / f"consumer_implementation_{logical_name}.json"
        _write_manifest(path, "PROSPECTIVE_CONSUMER_IMPLEMENTATION_FROZEN", {
            "logical_name": logical_name,
            "prospective_results_path": str(results_path),
            "prospective_results_sha256": sha256_file(results_path),
            "prospective_models_path": str(models_path),
            "prospective_models_sha256": sha256_file(models_path),
            "prospective_packet_path": str(Path(prospective_packet.__file__).resolve()),
            "prospective_packet_sha256": sha256_file(Path(prospective_packet.__file__).resolve()),
        })
        hashes[logical_name] = sha256_file(path)
        manifests[logical_name] = {"path": str(path.resolve()), "sha256": hashes[logical_name]}
    return hashes, manifests


def _preflight_execution_bindings(root: Path, output: Path, config_path: Path) -> list[Path]:
    """Validate the engineering supersession chain before any authorization write."""

    clarification_paths = [
        root / "protocol" / "PAPER_C_PROSPECTIVE_PREOUTCOME_CLARIFICATION_V1.md",
        root / "protocol" / "PAPER_C_PROSPECTIVE_REFERENCE_EXECUTION_ACCELERATION_V1.md",
    ]
    aborted_path = output / "frozen" / "REFERENCE_ABORTED_EXECUTION_RECEIPT.json"
    supersession_path = output / "frozen" / "REFERENCE_EXECUTION_SUPERSESSION_V1.json"
    for path in [*clarification_paths, aborted_path, supersession_path]:
        if not path.is_file():
            raise FileNotFoundError(f"required pre-outcome execution binding is missing: {path}")

    aborted = json.loads(aborted_path.read_text())
    if aborted.get("schema_version") != "1.0" or aborted.get("status") != "PROSPECTIVE_REFERENCE_EXECUTION_ABORTED_OUTCOME_BLIND":
        raise RuntimeError("aborted reference receipt has invalid schema or status")
    required_aborted = {
        "config_sha256": sha256_file(config_path),
        "valid_partial_reference_artifact": False,
        "learner_outcome_read": False,
        "authorization_existed": False,
        "outcome_ledger_existed": False,
        "release_marker_existed": False,
        "old_jobs_stopped": True,
    }
    for field, expected in required_aborted.items():
        if aborted.get(field) != expected:
            raise RuntimeError(f"aborted reference receipt field {field} differs from {expected!r}")
    if not isinstance(aborted.get("aborted_job_handles"), list) or len(aborted["aborted_job_handles"]) != 3:
        raise RuntimeError("aborted receipt must bind two slow workers and their coordinator")
    legacy_kernel_path = Path(str(aborted.get("legacy_reference_kernel_path", ""))).resolve()
    if (
        not legacy_kernel_path.is_file()
        or aborted.get("legacy_reference_kernel_sha256") != sha256_file(legacy_kernel_path)
    ):
        raise RuntimeError("aborted receipt does not bind the archived legacy reference kernel")

    supersession = json.loads(supersession_path.read_text())
    if supersession.get("schema_version") != "1.0" or supersession.get("status") != "PROSPECTIVE_REFERENCE_EXECUTION_SUPERSESSION_FROZEN":
        raise RuntimeError("reference execution supersession has invalid schema or status")
    parity_script = root / "code" / "paper_c" / "extension" / "reference_acceleration_parity.py"
    reference_kernel = root / "code" / "paper_c" / "swimmer" / "lqa_prospective.py"
    required_supersession = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(root / str(_load_config(root, config_path)["protocol"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "reference_kernel_sha256": sha256_file(reference_kernel),
        "parity_script_sha256": sha256_file(parity_script),
        "legacy_reference_kernel_sha256": aborted.get("legacy_reference_kernel_sha256"),
        "acceleration_protocol_sha256": sha256_file(clarification_paths[1]),
        "preoutcome_clarification_sha256": sha256_file(clarification_paths[0]),
        "aborted_execution_receipt_sha256": sha256_file(aborted_path),
        "scientific_inputs_changed": False,
        "scientific_seeds_changed": False,
        "particle_or_endpoint_definition_changed": False,
        "learner_outcome_read": False,
        "worker_count": 24,
        "reconstructed_top_level_shards": 2,
        "fixed_final_merge_order": [0, 1],
    }
    for field, expected in required_supersession.items():
        if supersession.get(field) != expected:
            raise RuntimeError(f"reference execution supersession field {field} differs from {expected!r}")
    if supersession.get("supersedes_execution_ids") != aborted["aborted_job_handles"]:
        raise RuntimeError("reference execution supersession does not name the aborted executions")
    expected_workers = [f"paper-c-fresh-reference-v2-worker-{index:02d}-of-24" for index in range(24)]
    if supersession.get("worker_execution_ids") != expected_workers:
        raise RuntimeError("reference execution supersession does not bind the exact worker identities")

    parity_paths: list[Path] = []
    owners = set()
    systems_seen = set()
    reference_config_path = output / "frozen" / "FRESH_REFERENCE_CONFIG.json"
    reference_config = json.loads(reference_config_path.read_text())
    expected_comparisons = {
        (stream, int(seed))
        for stream, seeds in (
            ("ref_a", reference_config["reference"]["ref_a_scramble_seeds"]),
            ("ref_b", reference_config["reference"]["ref_b_scramble_seeds"]),
        )
        for seed in seeds
    }
    parity_records = supersession.get("production_parity_receipts")
    if not isinstance(parity_records, list) or len(parity_records) != 2:
        raise RuntimeError("reference execution supersession must bind two production parity receipts")
    for record in parity_records:
        path = Path(record.get("path", "")).resolve()
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise RuntimeError("production parity receipt is missing or stale")
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "PROSPECTIVE_REFERENCE_ACCELERATION_PRODUCTION_PARITY_PASS"
            or payload.get("array_exact") is not True
            or payload.get("learner_outcome_read") is not False
            or payload.get("runner_sha256") != required_supersession["runner_sha256"]
            or payload.get("current_reference_kernel_sha256") != required_supersession["reference_kernel_sha256"]
            or payload.get("implementation_sha256") != required_supersession["parity_script_sha256"]
            or payload.get("config_sha256") != required_supersession["config_sha256"]
            or payload.get("reference_config_sha256") != sha256_file(reference_config_path)
            or payload.get("legacy_reference_kernel_sha256") != aborted.get("legacy_reference_kernel_sha256")
        ):
            raise RuntimeError("production parity receipt has invalid semantics or code bindings")
        comparisons = payload.get("comparisons")
        if (
            not isinstance(comparisons, list)
            or len(comparisons) != 8
            or {(row.get("stream"), int(row.get("seed", -1))) for row in comparisons} != expected_comparisons
            or any(row.get("array_exact") is not True for row in comparisons)
        ):
            raise RuntimeError("production parity receipt does not contain the exact eight frozen comparisons")
        owners.add(int(payload.get("parity_owner", -1)))
        systems_seen.add(int(payload.get("system_index", -1)))
        parity_paths.append(path)
    if owners != {0, 1} or systems_seen != {0, 1}:
        raise RuntimeError("production parity receipts do not cover frozen systems and owners zero and one")

    worker_merge_path = output / "reference" / "workers" / "WORKERS_24_MERGED_TO_TWO_SHARDS.json"
    if not worker_merge_path.is_file() or supersession.get("worker_merge_receipt_sha256") != sha256_file(worker_merge_path):
        raise RuntimeError("24-worker merge receipt is missing or stale")
    worker_merge = json.loads(worker_merge_path.read_text())
    if worker_merge.get("status") != "PROSPECTIVE_REFERENCE_WORKERS_RECONSTRUCTED_TWO_SHARDS" or worker_merge.get("worker_count") != 24:
        raise RuntimeError("24-worker merge receipt has invalid semantics")
    top_paths: list[Path] = []
    expected_top = worker_merge.get("top_level_receipts")
    if not isinstance(expected_top, list) or len(expected_top) != 2:
        raise RuntimeError("worker merge receipt does not bind two top-level receipts")
    for owner, record in enumerate(expected_top):
        receipt_path = output / "reference" / f"shard_{owner}_of_2" / "SHARD_RECEIPT.json"
        if record.get("shard_index") != owner or record.get("receipt_sha256") != sha256_file(receipt_path):
            raise RuntimeError("top-level reference receipt is missing or stale")
        receipt = json.loads(receipt_path.read_text())
        reference_path = receipt_path.parent / "fresh_reference_vectors.npz"
        target_path = receipt_path.parent / "fresh_targets_learner_blind.npz"
        if (
            receipt.get("status") != "PROSPECTIVE_REFERENCE_SHARD_COMPLETE_LEARNER_BLIND"
            or receipt.get("systems") != 256
            or receipt.get("worker_count") != 24
            or receipt.get("reference_sha256") != sha256_file(reference_path)
            or receipt.get("target_sha256") != sha256_file(target_path)
            or receipt.get("implementation_sha256") != required_supersession["runner_sha256"]
            or receipt.get("reference_kernel_sha256") != required_supersession["reference_kernel_sha256"]
            or receipt.get("learner_outcome_read") is not False
        ):
            raise RuntimeError("top-level reconstructed reference shard has invalid semantics")
        records = receipt.get("worker_records")
        expected_indices = list(range(owner, 24, 2))
        if not isinstance(records, list) or [record.get("worker_index") for record in records] != expected_indices:
            raise RuntimeError("top-level reference receipt does not bind its exact worker population")
        for record in records:
            index = int(record["worker_index"])
            execution = record.get("execution")
            if (
                not isinstance(execution, Mapping)
                or execution.get("execution_id") != expected_workers[index]
                or execution.get("supersedes_execution_ids") != aborted["aborted_job_handles"]
                or execution.get("learner_outcome_read") is not False
            ):
                raise RuntimeError("top-level worker execution chain differs from the frozen supersession")
        top_paths.extend([receipt_path, reference_path, target_path])

    final_merge_path = output / "reference" / "merged" / "MERGE_RECEIPT.json"
    final_reference = output / "reference" / "merged" / "fresh_reference_vectors.npz"
    final_target = output / "reference" / "merged" / "fresh_targets_learner_blind.npz"
    if not all(path.is_file() for path in (final_merge_path, final_reference, final_target)):
        raise RuntimeError("final reference merge is incomplete")
    final_merge = json.loads(final_merge_path.read_text())
    if (
        final_merge.get("status") != STATUS_REFERENCE
        or final_merge.get("reference_sha256") != sha256_file(final_reference)
        or final_merge.get("target_sha256") != sha256_file(final_target)
        or final_merge.get("learner_outcome_read") is not False
    ):
        raise RuntimeError("final reference merge receipt has invalid semantics")

    return [
        *clarification_paths,
        aborted_path,
        supersession_path,
        parity_script,
        reference_kernel,
        *parity_paths,
        worker_merge_path,
        *top_paths,
        final_merge_path,
        final_reference,
        final_target,
    ]


def _preflight_authorization_inputs(root: Path, output: Path, config_path: Path, cfg: Mapping[str, object]) -> None:
    """Validate all immutable authorization inputs before creating the first manifest."""

    ledger = output / "outcome" / "all_row_ledger.npz"
    release = Path(str(ledger) + ".release.json")
    if ledger.exists() or release.exists():
        raise RuntimeError("an outcome release already exists or has started")
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        Path(prospective_results.__file__).resolve(),
        Path(prospective_models.__file__).resolve(),
        Path(prospective_packet.__file__).resolve(),
    ]
    source_paths.extend(_root_path(root, path) for path in cfg.get("source_bindings", []))
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"authorization source binding is missing: {missing_sources}")

    reference_path = output / "reference" / "merged" / "fresh_reference_vectors.npz"
    target_path = output / "reference" / "merged" / "fresh_targets_learner_blind.npz"
    reference_receipt = output / "reference" / "merged" / "MERGE_RECEIPT.json"
    base_cache = output / "features" / "merged" / "preoutcome_cache.npz"
    arm_cache = output / "features" / "merged" / "e3_prediction_cache.npz"
    feature_receipt = output / "features" / "merged" / "MERGE_RECEIPT.json"
    score_receipt = output / "selector" / "SCORES_AND_PAIRS_FROZEN.json"
    pair_manifest = output / "selector" / "selector_pair_manifest.csv.gz"
    old_fit = _old_fit_receipt(root, cfg)
    shuffle_map = output / "features" / "merged" / "fresh_e3_shuffle_map.npz"
    required = [
        reference_path,
        target_path,
        reference_receipt,
        base_cache,
        arm_cache,
        feature_receipt,
        score_receipt,
        pair_manifest,
        old_fit,
        shuffle_map,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"authorization artifact is missing: {missing}")
    receipts = {
        reference_receipt: STATUS_REFERENCE,
        feature_receipt: STATUS_FEATURES,
        score_receipt: STATUS_SCORED,
        old_fit: "PAPER_C_PROSPECTIVE_OLD_FIT_FROZEN",
    }
    for path, status_name in receipts.items():
        if json.loads(path.read_text()).get("status") != status_name:
            raise RuntimeError(f"authorization input has invalid status: {path}")

    feature_payload = json.loads(feature_receipt.read_text())
    base_artifact_receipt = Path(str(base_cache) + ".receipt.json")
    arm_artifact_receipt = Path(str(arm_cache) + ".receipt.json")
    if (
        feature_payload.get("base_cache_sha256") != sha256_file(base_cache)
        or feature_payload.get("e3_cache_sha256") != sha256_file(arm_cache)
        or feature_payload.get("base_receipt_sha256") != sha256_file(base_artifact_receipt)
        or feature_payload.get("e3_receipt_sha256") != sha256_file(arm_artifact_receipt)
        or feature_payload.get("learner_outcome_read") is not False
    ):
        raise RuntimeError("feature merge receipt is stale or inconsistent")

    score_payload = json.loads(score_receipt.read_text())
    score_values = output / "selector" / "fresh_preoutcome_scores.npz"
    if (
        score_payload.get("pair_manifest_sha256") != sha256_file(pair_manifest)
        or score_payload.get("scores_sha256") != sha256_file(score_values)
        or score_payload.get("fresh_target_read") is not False
        or score_payload.get("learner_outcome_read") is not False
    ):
        raise RuntimeError("score-and-pair receipt is stale or inconsistent")
    orientations = score_payload.get("orientations")
    if not isinstance(orientations, Mapping) or set(orientations) != {"full", "family", "upstream", "permutation"}:
        raise RuntimeError("score-and-pair receipt has an invalid orientation population")
    for name, record in orientations.items():
        orientation_path = Path(str(record.get("path", "")))
        if not orientation_path.is_absolute():
            orientation_path = root / orientation_path
        if not orientation_path.is_file() or record.get("sha256") != sha256_file(orientation_path):
            raise RuntimeError(f"frozen orientation is missing or stale: {name}")

    reference_values = _load_npz(reference_path)
    validate_reference_vectors(reference_values, 512)
    target_values = _load_npz(target_path)
    validate_target_rows(target_values, 512)
    base_values = _load_npz(base_cache)
    validate_preoutcome_cache(base_values, 512)
    arm_values = _load_npz(arm_cache)
    validate_arm_predictions(arm_values, base_values)
    if row_population_sha256(reference_values) != row_population_sha256(base_values):
        raise RuntimeError("reference and base feature row populations differ before authorization")

    fixed_e3 = {
        _root_path(root, cfg["e3"]["checkpoint_table"]): cfg["e3"]["checkpoint_table_sha256"],
        _root_path(root, cfg["e3"]["authenticated_receipt"]): cfg["e3"]["authenticated_receipt_sha256"],
        _root_path(root, cfg["e3"]["freeze"]): cfg["e3"]["freeze_sha256"],
    }
    for path, expected_hash in fixed_e3.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"frozen E3 binding is missing or stale: {path}")
    checkpoint_rows = pd.read_csv(next(iter(fixed_e3)))
    checkpoint_rows = checkpoint_rows[
        checkpoint_rows.arm.isin(cfg["e3"]["arms"])
        & checkpoint_rows.seed.isin(cfg["e3"]["optimization_seeds"])
    ].sort_values(["arm", "seed"])
    expected_rows = {(arm, seed) for arm in cfg["e3"]["arms"] for seed in cfg["e3"]["optimization_seeds"]}
    if set(zip(checkpoint_rows.arm.astype(str), checkpoint_rows.seed.astype(int))) != expected_rows:
        raise RuntimeError("frozen E3 table is not the exact arm x optimization-seed product")
    for row in checkpoint_rows.itertuples(index=False):
        checkpoint = _root_path(root, row.checkpoint)
        if not checkpoint.is_file() or sha256_file(checkpoint) != str(row.checkpoint_sha256):
            raise RuntimeError(f"frozen E3 checkpoint is missing or stale: {checkpoint}")


def build_authorization(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path); output = _output(root, cfg)
    authorization_root = output / "authorization"
    if authorization_root.exists() and any(authorization_root.iterdir()):
        raise RuntimeError("authorization directory is already occupied before transactional preflight")
    _preflight_authorization_inputs(root, output, config_path, cfg)
    execution_sources = _preflight_execution_bindings(root, output, config_path)
    endpoint = output / "authorization" / "endpoint_and_aggregation_manifest.json"
    _write_manifest(endpoint, "PROSPECTIVE_ENDPOINTS_FROZEN", cfg["endpoints"])
    results_path = Path(prospective_results.__file__).resolve()
    models_path = Path(prospective_models.__file__).resolve()
    consumer_implementation_hashes, consumer_implementation_manifests = _freeze_consumer_implementations(output)
    source_manifest = output / "authorization" / "source_and_input_hashes.json"
    source_paths = [
        config_path,
        Path(__file__),
        results_path,
        models_path,
        Path(prospective_packet.__file__).resolve(),
        *execution_sources,
    ] + [_root_path(root, path) for path in cfg.get("source_bindings", [])]
    _write_manifest(source_manifest, "PROSPECTIVE_SOURCE_INPUTS_FROZEN", {
        "source_hashes": {str(path): sha256_file(path) for path in source_paths},
        "consumer_implementation_manifests": consumer_implementation_manifests,
    })
    e3_manifest = output / "authorization" / "e3_checkpoint_and_shuffle_manifest.json"
    checkpoint_table = _root_path(root, cfg["e3"]["checkpoint_table"]); shuffle_map = output / "features" / "merged" / "fresh_e3_shuffle_map.npz"
    authenticated_receipt = _root_path(root, cfg["e3"]["authenticated_receipt"])
    e3_freeze = _root_path(root, cfg["e3"]["freeze"])
    expected_e3_hashes = {
        checkpoint_table: cfg["e3"]["checkpoint_table_sha256"],
        authenticated_receipt: cfg["e3"]["authenticated_receipt_sha256"],
        e3_freeze: cfg["e3"]["freeze_sha256"],
    }
    for path, expected in expected_e3_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen E3 binding is stale: {path}")
    checkpoint_rows = pd.read_csv(checkpoint_table)
    checkpoint_rows = checkpoint_rows[
        checkpoint_rows.arm.isin(cfg["e3"]["arms"])
        & checkpoint_rows.seed.isin(cfg["e3"]["optimization_seeds"])
    ].sort_values(["arm", "seed"])
    checkpoint_hashes = {}
    for row in checkpoint_rows.itertuples(index=False):
        path = _root_path(root, row.checkpoint)
        if sha256_file(path) != str(row.checkpoint_sha256):
            raise RuntimeError(f"frozen E3 checkpoint is stale: {path}")
        checkpoint_hashes[str(path.resolve())] = str(row.checkpoint_sha256)
    _write_manifest(e3_manifest, "PROSPECTIVE_E3_INPUTS_FROZEN", {
        "checkpoint_table": str(checkpoint_table.resolve()),
        "checkpoint_table_sha256": sha256_file(checkpoint_table),
        "authenticated_receipt": str(authenticated_receipt.resolve()),
        "authenticated_receipt_sha256": sha256_file(authenticated_receipt),
        "e3_freeze": str(e3_freeze.resolve()),
        "e3_freeze_sha256": sha256_file(e3_freeze),
        "checkpoint_hashes": checkpoint_hashes,
        "shuffle_map_sha256": sha256_file(shuffle_map),
    })
    score_receipt = output / "selector" / "SCORES_AND_PAIRS_FROZEN.json"; old_fit = _old_fit_receipt(root, cfg); ref_merge = output / "reference" / "merged" / "MERGE_RECEIPT.json"
    artifacts = {
        "system_context_manifest": output / "frozen" / "fresh_system_context_manifest.csv.gz",
        "disjointness_receipt": output / "frozen" / "FRESH_POPULATION_FROZEN.json",
        "reference_receipts": ref_merge,
        "fresh_reference_vectors": output / "reference" / "merged" / "fresh_reference_vectors.npz",
        "frozen_base_feature_cache": output / "features" / "merged" / "preoutcome_cache.npz",
        "selector_pair_manifest": output / "selector" / "selector_pair_manifest.csv.gz",
        "selector_orientation_models_and_maps": score_receipt,
        "e3_checkpoint_and_shuffle_manifest": e3_manifest,
        "e3_prediction_cache": output / "features" / "merged" / "e3_prediction_cache.npz",
        "probe_models_and_standardizers": old_fit,
        "endpoint_and_aggregation_manifest": endpoint,
        "source_and_input_hashes": source_manifest,
        "fresh_target": output / "reference" / "merged" / "fresh_targets_learner_blind.npz",
    }
    if set(artifacts) != REQUIRED_AUTHORIZATION_ARTIFACTS: raise AssertionError("authorization artifact schema drift")
    config_sha = sha256_file(config_path)
    base = _load_npz(artifacts["frozen_base_feature_cache"]); row_sha = row_population_sha256(base); aggregation_sha = sha256_file(endpoint)
    consumers = [
        {"logical_name": "primary_e3", "config_sha256": config_sha, "checkpoint_or_model_sha256": sha256_file(e3_manifest), "implementation_sha256": consumer_implementation_hashes["primary_e3"], "row_population_sha256": row_sha, "aggregation_sha256": aggregation_sha},
        {"logical_name": "secondary_a_decodability", "config_sha256": config_sha, "checkpoint_or_model_sha256": sha256_file(old_fit), "implementation_sha256": consumer_implementation_hashes["secondary_a_decodability"], "row_population_sha256": row_sha, "aggregation_sha256": aggregation_sha},
        {"logical_name": "secondary_b_realization", "config_sha256": config_sha, "checkpoint_or_model_sha256": sha256_file(old_fit), "implementation_sha256": consumer_implementation_hashes["secondary_b_realization"], "row_population_sha256": sha256_file(artifacts["selector_pair_manifest"]), "aggregation_sha256": aggregation_sha},
    ]
    return freeze_outcome_authorization(output / "authorization" / "OUTCOME_AUTHORIZATION_BUNDLE.json", consumers, artifacts, output / "outcome" / "all_row_ledger.npz")


def status(root: Path, config_path: Path) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve(); cfg = _load_config(root, config_path); output = _output(root, cfg)
    paths = {
        "design": _design_receipt(root, cfg), "old_inputs": output / "old_fit" / "inputs" / "OLD_INPUTS_RECEIPT.json",
        "old_models": _old_fit_receipt(root, cfg), "population": output / "frozen" / "FRESH_POPULATION_FROZEN.json",
        "reference": output / "reference" / "merged" / "MERGE_RECEIPT.json", "features": output / "features" / "merged" / "MERGE_RECEIPT.json",
        "scores_pairs": output / "selector" / "SCORES_AND_PAIRS_FROZEN.json", "authorization": output / "authorization" / "OUTCOME_AUTHORIZATION_BUNDLE.json",
        "release_marker": Path(str(output / "outcome" / "all_row_ledger.npz") + ".release.json"), "outcome_ledger": output / "outcome" / "all_row_ledger.npz",
    }
    return {"schema_version": "1.0", "status": "PREOUTCOME_READY" if paths["authorization"].is_file() and not paths["release_marker"].exists() else "INCOMPLETE_OR_RELEASED",
            "artifacts": {name: {"exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None} for name, path in paths.items()},
            "runner_exposes_outcome_release": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-design"); sub.add_parser("build-old-inputs"); sub.add_parser("fit-freeze-old-models")
    reference = sub.add_parser("prepare-reference-shard"); reference.add_argument("--shard-index", type=int, required=True)
    worker = sub.add_parser("prepare-reference-worker")
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--execution-id")
    worker.add_argument("--supersedes-execution-id", action="append", default=[])
    worker_merge = sub.add_parser("merge-reference-workers")
    worker_merge.add_argument("--worker-count", type=int, required=True)
    sub.add_parser("merge-reference")
    feature = sub.add_parser("prepare-feature-shard"); feature.add_argument("--shard-index", type=int, required=True); feature.add_argument("--device", default="cpu")
    feature_merge = sub.add_parser("merge-features"); feature_merge.add_argument("--device", default="cpu")
    sub.add_parser("score-and-build-pairs"); sub.add_parser("build-authorization"); sub.add_parser("status")
    return parser


def main() -> None:
    args = _parser().parse_args()
    dispatch = {
        "freeze-design": lambda: freeze_design(args.root, args.config),
        "build-old-inputs": lambda: build_old_inputs(args.root, args.config),
        "fit-freeze-old-models": lambda: fit_and_freeze_old_models(args.root, args.config),
        "prepare-reference-shard": lambda: prepare_reference_shard(args.root, args.config, args.shard_index),
        "prepare-reference-worker": lambda: prepare_reference_worker(
            args.root, args.config, args.worker_index, args.worker_count,
            args.execution_id, args.supersedes_execution_id,
        ),
        "merge-reference-workers": lambda: merge_reference_workers(args.root, args.config, args.worker_count),
        "merge-reference": lambda: merge_reference(args.root, args.config),
        "prepare-feature-shard": lambda: prepare_feature_shard(args.root, args.config, args.shard_index, args.device),
        "merge-features": lambda: merge_features(args.root, args.config, args.device),
        "score-and-build-pairs": lambda: score_and_build_pairs(args.root, args.config),
        "build-authorization": lambda: build_authorization(args.root, args.config),
        "status": lambda: status(args.root, args.config),
    }
    print(json.dumps(dispatch[args.command](), sort_keys=True))


if __name__ == "__main__":
    main()
