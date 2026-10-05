"""Fresh-system prospective confirmation of the frozen Articulated routing intervention."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.learner import _normalized
from paper_c.coupled_sled.posterior import combine_independent_posteriors
from paper_c.stage1.bayes_alignment import initial_state_from_features
from paper_c.stage2.routing_intervention import (
    RoutingAdapter,
    _aggregate_formal_rows,
    _paired_bootstrap,
    _row_metrics,
    cross_system_cell_permutation,
    original_module_hashes,
)
from paper_c.swimmer.lqa_evaluate import build_unique_learner_rows
from paper_c.swimmer.lqa_formal import formal_context_table
from paper_c.swimmer.lqa_prospective import (
    SwimmerModel,
    _landmarks,
    _load_jepa,
    _response_bank,
    banks,
    particle_pool,
    system_pool,
)


STATUS_FREEZE = "FRESH_ARTICULATED_PROSPECTIVE_FROZEN"
STATUS_PREP = "FRESH_ARTICULATED_PREPARATION_SHARD_COMPLETE"
STATUS_EVAL = "FRESH_ARTICULATED_EVALUATION_SHARD_COMPLETE"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def _load_config(root: Path, config_path: Path) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("fresh prospective computation is remote-Linux only")
    cfg = json.loads(config_path.read_text())
    if cfg.get("status") != "FROZEN_BEFORE_FRESH_SYSTEM_GENERATION":
        raise RuntimeError("fresh prospective config is not frozen")
    return cfg


def _reference_config(root: Path, cfg: dict) -> dict:
    reference = copy.deepcopy(json.loads((root / cfg["reference_config_template"]).read_text()))
    design = cfg["fresh_design"]
    reference["protocol"] = cfg["protocol"]
    reference["formal"]["system_seed"] = int(design["system_seed"])
    reference["formal"]["context_seed"] = int(design["context_seed"])
    reference["formal"]["pool_max"] = int(design["systems"])
    reference["formal"]["target_systems"] = int(design["systems"])
    reference["formal"]["contexts_per_system"] = 1
    return reference


def _identity(theta: np.ndarray, decimals: int) -> str:
    token = "|".join(f"{value:.{decimals}f}" for value in np.asarray(theta, dtype=np.float64))
    return hashlib.sha256(token.encode()).hexdigest()


def _verify_v1_selection(root: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    receipt_path = root / cfg["v1_selection_receipt"]
    table_path = root / cfg["v1_selection_table"]
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "DOWNSTREAM_ROUTING_SELECT_CHOICES_FROZEN"
        or receipt.get("formal_outcomes_used_for_selection") is not False
        or receipt.get("table_sha256") != sha256(table_path)
    ):
        raise RuntimeError("V1 Articulated selection is invalid")
    table = pd.read_csv(table_path)
    if len(table) != 9 or set(table.environment) != {"articulated"}:
        raise RuntimeError("V1 selection must contain exactly nine Articulated checkpoints")
    expected = {(arm, int(seed)) for arm in cfg["arms"] for seed in cfg["seeds"]}
    if set(zip(table.arm.astype(str), table.seed.astype(int))) != expected:
        raise RuntimeError("V1 selection arm/seed population differs from the fresh protocol")
    for row in table.itertuples(index=False):
        if sha256(root / row.checkpoint) != row.checkpoint_sha256:
            raise RuntimeError(f"selected adapter hash mismatch: {row.checkpoint}")
    return table, receipt


def freeze(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    output = root / cfg["output_root"]
    freeze_path = output / "FREEZE_RECEIPT.json"
    if freeze_path.exists():
        raise RuntimeError("fresh prospective freeze already exists; refusing overwrite")
    selection, selection_receipt = _verify_v1_selection(root, cfg)
    reference = _reference_config(root, cfg)
    base_path = root / reference["base_config"]
    base = json.loads(base_path.read_text())
    design = cfg["fresh_design"]
    count = int(design["systems"])
    fresh = system_pool(count, int(design["system_seed"]), base["persistent_prior"])
    old = system_pool(int(design["old_formal_system_count"]), int(design["old_formal_system_seed"]), base["persistent_prior"])
    decimals = int(design["identity_round_decimals"])
    fresh_ids = [_identity(row, decimals) for row in fresh]
    old_ids = [_identity(row, decimals) for row in old]
    overlap = sorted(set(fresh_ids) & set(old_ids))
    if overlap or len(set(fresh_ids)) != count:
        raise RuntimeError("fresh physical systems are duplicated or overlap the old formal systems")
    contexts = formal_context_table(reference).sort_values("system_index").reset_index(drop=True)
    if len(contexts) != count or contexts.system_index.nunique() != count:
        raise RuntimeError("fresh context table is not one-context-per-system")
    systems_path = output / "frozen" / "fresh_systems.npy"
    manifest_path = output / "frozen" / "fresh_system_manifest.csv.gz"
    reference_path = output / "frozen" / "FRESH_REFERENCE_CONFIG_FROZEN.json"
    _atomic_npy(systems_path, fresh.astype(np.float64))
    manifest = contexts.copy()
    for index in range(fresh.shape[1]):
        manifest[f"theta_{index}"] = fresh[:, index]
    manifest["physical_identity_sha256"] = fresh_ids
    _atomic_csv(manifest_path, manifest)
    _atomic_text(reference_path, json.dumps(reference, indent=2, sort_keys=True) + "\n")
    disjoint_path = output / "SYSTEM_DISJOINTNESS_RECEIPT.json"
    disjoint = {
        "schema_version": "1.0",
        "status": "FRESH_SYSTEMS_DISJOINT_FROM_OLD_FORMAL",
        "fresh_systems": count,
        "old_formal_systems": int(design["old_formal_system_count"]),
        "intersection_count": 0,
        "fresh_unique_count": len(set(fresh_ids)),
        "identity_round_decimals": decimals,
        "fresh_system_seed": int(design["system_seed"]),
        "old_formal_system_seed": int(design["old_formal_system_seed"]),
        "systems_sha256": sha256(systems_path),
        "manifest_sha256": sha256(manifest_path),
        "old_identity_set_sha256": hashlib.sha256("\n".join(sorted(old_ids)).encode()).hexdigest(),
        "fresh_identity_set_sha256": hashlib.sha256("\n".join(sorted(fresh_ids)).encode()).hexdigest(),
    }
    _atomic_text(disjoint_path, json.dumps(disjoint, indent=2, sort_keys=True) + "\n")
    source_paths = [
        config_path,
        root / cfg["protocol"],
        root / cfg["implementation"],
        root / cfg["v1_intervention_config"],
        root / cfg["v1_selection_receipt"],
        root / cfg["v1_selection_table"],
        root / cfg["learner_checkpoint"],
        root / cfg["normalization"],
        base_path,
        root / reference["s0_receipt"],
        reference_path,
        systems_path,
        manifest_path,
        disjoint_path,
    ]
    checkpoint_identities = [
        {
            "arm": str(row.arm),
            "seed": int(row.seed),
            "checkpoint": str(row.checkpoint),
            "checkpoint_sha256": str(row.checkpoint_sha256),
        }
        for row in selection.itertuples(index=False)
    ]
    receipt = {
        "schema_version": "1.0",
        "status": STATUS_FREEZE,
        "frozen_at_unix": time.time(),
        "host": socket.gethostname(),
        "fresh_system_outcomes_read": False,
        "adapter_training_run": False,
        "adapter_or_hyperparameter_reselection": False,
        "selected_checkpoint_identities": checkpoint_identities,
        "v1_selection_status": selection_receipt["status"],
        "v1_selection_table_sha256": selection_receipt["table_sha256"],
        "disjointness_receipt": str(disjoint_path.relative_to(root)),
        "disjointness_receipt_sha256": sha256(disjoint_path),
        "source_hashes": {str(path.relative_to(root)): sha256(path) for path in source_paths},
        "resource_limit": {"maximum_gpus": 2, "allowed_gpu_ids": [0, 1]},
    }
    _atomic_text(freeze_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {**receipt, "freeze_receipt_sha256": sha256(freeze_path)}


def _verify_freeze(root: Path, cfg: dict) -> dict:
    output = root / cfg["output_root"]
    path = output / "FREEZE_RECEIPT.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_FREEZE or receipt.get("fresh_system_outcomes_read") is not False:
        raise RuntimeError("fresh prospective freeze is invalid")
    implementation = cfg["implementation"]
    for relative, expected in receipt["source_hashes"].items():
        observed = sha256(root / relative)
        if observed == expected:
            continue
        if relative != implementation:
            raise RuntimeError(f"frozen source changed: {relative}")
        cursor = expected
        repairs = sorted(output.glob("ENGINEERING_REPAIR_*.json"))
        for repair_path in repairs:
            repair = json.loads(repair_path.read_text())
            if (
                repair.get("old_implementation_sha256") != cursor
                or repair.get("scientific_definition_changed") is not False
            ):
                raise RuntimeError("implementation repair chain is invalid")
            cursor = repair["new_implementation_sha256"]
        if cursor != observed:
            raise RuntimeError("implementation changed without a complete engineering repair chain")
    disjoint_path = root / receipt["disjointness_receipt"]
    disjoint = json.loads(disjoint_path.read_text())
    if disjoint.get("intersection_count") != 0 or sha256(disjoint_path) != receipt["disjointness_receipt_sha256"]:
        raise RuntimeError("fresh-system disjointness receipt is missing or stale")
    _verify_v1_selection(root, cfg)
    return receipt


def _all_candidate_manifest(contexts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in contexts.itertuples(index=False):
        for query_index in range(6):
            for first, second in ((0, 1), (2, 3), (4, 5)):
                rows.append({
                    "system_index": int(context.system_index),
                    "realization": int(context.realization),
                    "history_index": int(context.history_index),
                    "query_index": query_index,
                    "lqa_selected_candidate": first,
                    "other_candidate": second,
                })
    return pd.DataFrame(rows)


@torch.no_grad()
def _features(model, norms: dict, arrays, device: torch.device) -> dict[str, np.ndarray]:
    history, mask, query, target = _normalized(arrays, norms)
    outputs = {name: [] for name in ("persistent", "query", "predicted", "prediction")}
    batch_size = 1024
    for start in range(0, len(history), batch_size):
        h = torch.from_numpy(history[start:start + batch_size]).to(device)
        m = torch.from_numpy(mask[start:start + batch_size]).to(device)
        q = torch.from_numpy(query[start:start + batch_size]).to(device)
        encoded = model.segment_encoder(h) * m[:, :, None]
        persistent = model.aggregator(torch.cat((encoded.flatten(1), m), dim=1))
        qembed = model.query_encoder(q)
        predicted = model.latent_predictor(torch.cat((persistent, qembed), dim=1))
        prediction = model.target_decoder(predicted)
        outputs["persistent"].append(persistent.cpu().numpy())
        outputs["query"].append(qembed.cpu().numpy())
        outputs["predicted"].append(predicted.cpu().numpy())
        outputs["prediction"].append(prediction.cpu().numpy())
    result = {name: np.concatenate(values).astype(np.float32) for name, values in outputs.items()}
    result["target"] = target.astype(np.float32)
    return result


def _ref_b_corrections(root: Path, reference: dict, cfg: dict, table: pd.DataFrame, arrays,
                       baseline_table: pd.DataFrame, baseline_arrays, baseline_prediction: np.ndarray,
                       norms: dict) -> np.ndarray:
    base = json.loads((root / reference["base_config"]).read_text())
    s0 = json.loads((root / reference["s0_receipt"]).read_text())
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    simulator = SwimmerModel(base["model"])
    particles = particle_pool(int(cfg["reference"]["particles"]), int(cfg["reference"]["particle_seed"]), base["persistent_prior"])
    result = np.empty((len(table), 32), dtype=np.float64)
    baseline_lookup = {
        (int(row.system_index), int(row.query_index)): position
        for position, row in enumerate(baseline_table.itertuples(index=False))
    }
    for system_index, positions_raw in table.groupby("system_index", sort=True).indices.items():
        positions = np.asarray(positions_raw, dtype=int)
        first = positions[0]
        history_index = int(table.iloc[first].history_index)
        baseline_zero = baseline_lookup[(int(system_index), 0)]
        ih = initial_state_from_features(baseline_arrays.history[baseline_zero, 0, :8])
        anchor_obs = baseline_arrays.history[baseline_zero, 0, 8:40]
        candidate_zero = positions[table.iloc[positions].candidate_index.to_numpy(int) == 0][0]
        ie = initial_state_from_features(arrays.history[int(candidate_zero), 1, :8])
        iq = initial_state_from_features(baseline_arrays.query_action[baseline_zero, :8])
        hmeans = _response_bank(simulator, particles, ih, history, landmarks)[:, history_index]
        emeans = _response_bank(simulator, particles, ie, history, landmarks)
        qmeans = _response_bank(simulator, particles, iq, query, landmarks)
        for candidate in range(6):
            subset = positions[table.iloc[positions].candidate_index.to_numpy(int) == candidate]
            candidate_obs = arrays.history[int(subset[0]), 1, 8:40]
            posterior = combine_independent_posteriors(
                (anchor_obs, candidate_obs), (hmeans, emeans[:, candidate]),
                float(base["observation"]["sensor_std"]), np.asarray([1.0]), np.asarray([1.0]),
            )
            for row_position in subset:
                query_index = int(table.iloc[row_position].query_index)
                mu = np.einsum("n,nf->f", posterior.weights, qmeans[:, query_index])
                normalized_mu = (mu - norms["target_mean"]) / norms["target_std"]
                result[row_position] = normalized_mu - baseline_prediction[baseline_lookup[(int(system_index), query_index)]]
    return result.astype(np.float32)


def prepare_shard(root: Path, config_path: Path, shard_index: int, shard_count: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    frozen = _verify_freeze(root, cfg)
    if shard_count != int(cfg["execution"]["preparation_shards"]):
        raise RuntimeError("preparation shard count differs from the frozen protocol")
    output = root / cfg["output_root"]
    manifest = pd.read_csv(output / "frozen" / "fresh_system_manifest.csv.gz")
    contexts = manifest[manifest.system_index.to_numpy(int) % shard_count == shard_index][
        ["system_index", "realization", "history_index"]
    ].copy()
    if contexts.empty:
        raise RuntimeError("empty preparation shard")
    reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG_FROZEN.json").read_text())
    pair_manifest = _all_candidate_manifest(contexts)
    baseline_arrays, baseline_table, candidate_arrays, candidate_table = build_unique_learner_rows(root, reference, pair_manifest)
    device = torch.device("cpu")
    torch.set_num_threads(1)
    learner, norms, _ = _load_jepa(root, reference, device)
    before = original_module_hashes(learner)
    baseline = _features(learner, norms, baseline_arrays, device)
    candidate = _features(learner, norms, candidate_arrays, device)
    baseline_lookup = {
        (int(row.system_index), int(row.realization), int(row.history_index), int(row.query_index)): position
        for position, row in enumerate(baseline_table.itertuples(index=False))
    }
    baseline_position = np.asarray([
        baseline_lookup[(int(row.system_index), int(row.realization), int(row.history_index), int(row.query_index))]
        for row in candidate_table.itertuples(index=False)
    ], dtype=np.int64)
    correction = _ref_b_corrections(
        root, reference, cfg, candidate_table, candidate_arrays,
        baseline_table, baseline_arrays, baseline["prediction"], norms,
    )
    row_index = (
        candidate_table.system_index.to_numpy(np.int64) * 36
        + candidate_table.candidate_index.to_numpy(np.int64) * 6
        + candidate_table.query_index.to_numpy(np.int64)
    )
    values = {
        "row_index": row_index,
        "system_index": candidate_table.system_index.to_numpy(np.int64),
        "anchor_index": candidate_table.history_index.to_numpy(np.int64),
        "query_index": candidate_table.query_index.to_numpy(np.int64),
        "candidate_identity": candidate_table.candidate_index.astype(str).to_numpy(),
        "candidate_index": candidate_table.candidate_index.to_numpy(np.int64),
        "delta_persistent": candidate["persistent"] - baseline["persistent"][baseline_position],
        "query_embedding": candidate["query"],
        "predicted_latent_full": candidate["predicted"],
        "prediction_original": candidate["prediction"],
        "prediction_anchor": baseline["prediction"][baseline_position],
        "normalized_target": candidate["target"],
        "bayes_correction": correction,
    }
    if len(np.unique(row_index)) != len(row_index) or not all(np.isfinite(value).all() for value in values.values() if np.issubdtype(value.dtype, np.number)):
        raise RuntimeError("preparation shard contains duplicate or nonfinite rows")
    after = original_module_hashes(learner)
    if before != after:
        raise RuntimeError("frozen learner changed during fresh preparation")
    shard_root = output / "preparation" / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    cache_path = shard_root / "cache.npz"
    receipt_path = shard_root / "receipt.json"
    if cache_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable preparation shard already exists")
    _atomic_npz(cache_path, **values)
    receipt = {
        "status": STATUS_PREP,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "systems": int(contexts.system_index.nunique()),
        "rows": len(row_index),
        "cache": str(cache_path.relative_to(root)),
        "cache_sha256": sha256(cache_path),
        "freeze_receipt_sha256": sha256(output / "FREEZE_RECEIPT.json"),
        "original_module_hashes_before": before,
        "original_module_hashes_after": after,
        "host": socket.gethostname(),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name].copy() for name in payload.files}
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
    # Engineering repair 001: the first preparation pass serialized the
    # integer candidate label through a pandas object-string column.  Permit
    # pickle only for that one local, hash-verified shard field, immediately
    # canonicalize it to NumPy Unicode, and reject any other object array.
    with np.load(path, allow_pickle=True) as payload:
        result = {name: payload[name].copy() for name in payload.files}
    object_names = {name for name, value in result.items() if value.dtype == object}
    if object_names != {"candidate_identity"}:
        raise RuntimeError(f"unexpected object arrays in cache: {sorted(object_names)}")
    result["candidate_identity"] = result["candidate_identity"].astype("U")
    return result


def merge_preparation(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    _verify_freeze(root, cfg)
    output = root / cfg["output_root"]
    shard_count = int(cfg["execution"]["preparation_shards"])
    parts, hashes, model_hashes = [], [], None
    for index in range(shard_count):
        shard_root = output / "preparation" / f"shard_{index:02d}_of_{shard_count:02d}"
        receipt = json.loads((shard_root / "receipt.json").read_text())
        cache_path = root / receipt["cache"]
        if receipt.get("status") != STATUS_PREP or receipt.get("cache_sha256") != sha256(cache_path):
            raise RuntimeError(f"invalid preparation shard {index}")
        if model_hashes is None:
            model_hashes = receipt["original_module_hashes_after"]
        elif model_hashes != receipt["original_module_hashes_after"]:
            raise RuntimeError("learner module hashes differ across preparation shards")
        parts.append(_load_npz(cache_path)); hashes.append(receipt["cache_sha256"])
    names = list(parts[0])
    if any(list(part) != names for part in parts):
        raise RuntimeError("preparation shard schemas differ")
    merged = {name: np.concatenate([part[name] for part in parts]) for name in names}
    order = np.argsort(merged["row_index"], kind="stable")
    merged = {name: value[order] for name, value in merged.items()}
    if len(merged["row_index"]) != 512 * 36 or len(np.unique(merged["row_index"])) != 512 * 36:
        raise RuntimeError("merged fresh cache is incomplete or duplicated")
    cache_path = output / "cache" / "merged.npz"
    receipt_path = output / "cache" / "MERGED_CACHE_RECEIPT.json"
    _atomic_npz(cache_path, **merged)
    receipt = {
        "status": "FRESH_ARTICULATED_MERGED_CACHE_COMPLETE",
        "systems": int(len(np.unique(merged["system_index"]))),
        "rows": int(len(merged["row_index"])),
        "cache": str(cache_path.relative_to(root)),
        "cache_sha256": sha256(cache_path),
        "preparation_shard_hashes": hashes,
        "original_module_hashes": model_hashes,
        "fixed_merge_order": list(range(shard_count)),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    maps = []
    for seed in cfg["seeds"]:
        donor = cross_system_cell_permutation(merged, int(seed))
        path = output / "shuffle_maps" / f"seed_{int(seed)}.npz"
        _atomic_npz(path, donor_position=donor.astype(np.int64), receiver_row_index=merged["row_index"], donor_row_index=merged["row_index"][donor])
        maps.append({"seed": int(seed), "path": str(path.relative_to(root)), "sha256": sha256(path)})
    shuffle_receipt = {
        "status": "FRESH_ARTICULATED_SHUFFLE_MAPS_FROZEN",
        "cell": cfg["shuffled_control"]["cell"],
        "rows": int(len(merged["row_index"])),
        "cache_sha256": receipt["cache_sha256"],
        "maps": maps,
    }
    _atomic_text(output / "shuffle_maps" / "SHUFFLE_MAPS_FROZEN.json", json.dumps(shuffle_receipt, indent=2, sort_keys=True) + "\n")
    return {**receipt, "shuffle_maps": maps}


def _selection_and_maps(root: Path, cfg: dict, cache: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[str, str]]:
    selection, _ = _verify_v1_selection(root, cfg)
    output = root / cfg["output_root"]
    receipt_path = output / "shuffle_maps" / "SHUFFLE_MAPS_FROZEN.json"
    receipt = json.loads(receipt_path.read_text())
    maps, hashes = {}, {}
    for item in receipt["maps"]:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError("fresh shuffle map hash mismatch")
        payload = _load_npz(path)
        if not np.array_equal(payload["receiver_row_index"], cache["row_index"]):
            raise RuntimeError("fresh shuffle receiver population differs")
        maps[int(item["seed"])] = payload["donor_position"].astype(np.int64)
        hashes[str(int(item["seed"]))] = item["sha256"]
    return selection, maps, hashes


@torch.no_grad()
def _evaluate_positions(root: Path, cfg: dict, cache: dict[str, np.ndarray], positions: np.ndarray,
                        device: torch.device) -> tuple[pd.DataFrame, dict, dict, dict[str, str]]:
    output = root / cfg["output_root"]
    reference = json.loads((output / "frozen" / "FRESH_REFERENCE_CONFIG_FROZEN.json").read_text())
    learner, _, _ = _load_jepa(root, reference, device)
    before = original_module_hashes(learner)
    selection, maps, map_hashes = _selection_and_maps(root, cfg, cache)
    local = {name: value[positions] for name, value in cache.items()}
    rows = []

    def append(arm: str, seed: int, prediction: np.ndarray) -> None:
        metrics = _row_metrics(prediction, local["prediction_anchor"], local["normalized_target"], local["bayes_correction"])
        for offset in range(len(positions)):
            row = {
                "row_index": int(local["row_index"][offset]),
                "system_index": int(local["system_index"][offset]),
                "anchor_index": int(local["anchor_index"][offset]),
                "query_index": int(local["query_index"][offset]),
                "candidate_identity": str(local["candidate_identity"][offset]),
                "arm": arm,
                "seed": seed,
            }
            row.update({name: float(value[offset]) for name, value in metrics.items()})
            rows.append(row)

    append("original", -1, local["prediction_original"])
    for choice in selection.itertuples(index=False):
        adapter = RoutingAdapter(128, 64, 64).to(device)
        adapter.load_state_dict(torch.load(root / choice.checkpoint, map_location=device, weights_only=True))
        adapter.eval()
        if choice.arm == "true":
            delta = local["delta_persistent"]
        elif choice.arm == "query_only":
            delta = np.zeros_like(local["delta_persistent"])
        elif choice.arm == "shuffled":
            delta = cache["delta_persistent"][maps[int(choice.seed)][positions]]
        else:
            raise RuntimeError(f"unknown frozen arm {choice.arm}")
        prediction = np.empty_like(local["prediction_original"])
        for system_index in np.sort(np.unique(local["system_index"])):
            system_positions = np.flatnonzero(local["system_index"] == system_index)
            d = torch.from_numpy(delta[system_positions]).to(device)
            q = torch.from_numpy(local["query_embedding"][system_positions]).to(device)
            z = torch.from_numpy(local["predicted_latent_full"][system_positions]).to(device)
            prediction[system_positions] = learner.target_decoder(z + adapter(d, q)).cpu().numpy()
        append(str(choice.arm), int(choice.seed), prediction)
    after = original_module_hashes(learner)
    if before != after:
        raise RuntimeError("frozen learner changed during fresh evaluation")
    table = pd.DataFrame(rows).sort_values(["system_index", "row_index", "arm", "seed"]).reset_index(drop=True)
    return _aggregate_formal_rows(table), before, after, map_hashes


def evaluate_shard(root: Path, config_path: Path, shard_index: int, shard_count: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    _verify_freeze(root, cfg)
    if shard_count != int(cfg["execution"]["evaluation_shards"]):
        raise RuntimeError("evaluation shard count differs from the frozen protocol")
    output = root / cfg["output_root"]
    cache_receipt = json.loads((output / "cache" / "MERGED_CACHE_RECEIPT.json").read_text())
    cache_path = root / cache_receipt["cache"]
    if cache_receipt.get("cache_sha256") != sha256(cache_path):
        raise RuntimeError("fresh merged cache receipt is stale")
    cache = _load_npz(cache_path)
    positions = np.flatnonzero(cache["system_index"] % shard_count == shard_index)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.45, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    stats, before, after, maps = _evaluate_positions(root, cfg, cache, positions, device)
    eval_root = output / "evaluation"
    stats_path = eval_root / f"stats_shard_{shard_index:02d}_of_{shard_count:02d}.csv.gz"
    receipt_path = eval_root / f"receipt_shard_{shard_index:02d}_of_{shard_count:02d}.json"
    if stats_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable evaluation shard already exists")
    _atomic_csv(stats_path, stats)
    receipt = {
        "status": STATUS_EVAL,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "systems": int(stats.system_index.nunique()),
        "statistics": str(stats_path.relative_to(root)),
        "statistics_sha256": sha256(stats_path),
        "statistics_dtype": "float64",
        "original_module_hashes_before": before,
        "original_module_hashes_after": after,
        "shuffle_map_hashes": maps,
        "device": device_name,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def parity(root: Path, config_path: Path, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    _verify_freeze(root, cfg)
    output = root / cfg["output_root"]
    cache_receipt = json.loads((output / "cache" / "MERGED_CACHE_RECEIPT.json").read_text())
    cache = _load_npz(root / cache_receipt["cache"])
    systems = np.asarray([0, 1], dtype=np.int64)
    positions = np.flatnonzero(np.isin(cache["system_index"], systems))
    fresh, _, _, maps = _evaluate_positions(root, cfg, cache, positions, torch.device(device_name))
    frozen_parts = []
    for shard in range(2):
        frozen_parts.append(pd.read_csv(output / "evaluation" / f"stats_shard_{shard:02d}_of_02.csv.gz"))
    frozen = pd.concat(frozen_parts, ignore_index=True)
    frozen = frozen[frozen.system_index.isin(systems)].sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    fresh = fresh.sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    if list(fresh.columns) != list(frozen.columns):
        raise RuntimeError("one-task/two-shard parity schemas differ")
    for name in fresh:
        left, right = fresh[name].to_numpy(), frozen[name].to_numpy()
        # Shard statistics are persisted through decimal CSV.  Their fresh
        # one-task recomputation is bit-identical before serialization, while
        # CSV parsing may move the last binary64 ulp.  Use a fixed numerical
        # serialization tolerance and keep exact equality for discrete axes.
        equal = np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True) if np.issubdtype(left.dtype, np.inexact) else np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"one-task/two-shard parity differs for {name}")
    receipt = {
        "status": "FRESH_ONE_TASK_VS_TWO_SHARD_PARITY_PASS",
        "systems": systems.tolist(),
        "statistics_dtype": "float64",
        "fixed_merge_order": [0, 1],
        "shuffle_map_hashes": maps,
    }
    path = output / "evaluation" / "PARITY_RECEIPT.json"
    _atomic_text(path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def merge_evaluation(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _load_config(root, config_path)
    _verify_freeze(root, cfg)
    output = root / cfg["output_root"]
    parity_receipt = json.loads((output / "evaluation" / "PARITY_RECEIPT.json").read_text())
    if parity_receipt.get("status") != "FRESH_ONE_TASK_VS_TWO_SHARD_PARITY_PASS":
        raise RuntimeError("fresh one-task/two-shard parity did not pass")
    tables, receipts = [], []
    for shard in range(2):
        receipt = json.loads((output / "evaluation" / f"receipt_shard_{shard:02d}_of_02.json").read_text())
        path = root / receipt["statistics"]
        if receipt.get("status") != STATUS_EVAL or receipt.get("statistics_sha256") != sha256(path):
            raise RuntimeError(f"invalid evaluation shard {shard}")
        tables.append(pd.read_csv(path)); receipts.append(receipt)
    stats = pd.concat(tables, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    if stats.duplicated(["system_index", "arm", "seed"]).any():
        raise RuntimeError("fresh evaluation duplicates system-arm-seed cells")
    expected_systems = set(range(512))
    combinations = {("original", -1)} | {(arm, int(seed)) for arm in cfg["arms"] for seed in cfg["seeds"]}
    if set(zip(stats.arm.astype(str), stats.seed.astype(int))) != combinations:
        raise RuntimeError("fresh evaluation arm/seed population is incomplete")
    for arm, seed in combinations:
        if set(stats[(stats.arm == arm) & (stats.seed == seed)].system_index.astype(int)) != expected_systems:
            raise RuntimeError(f"fresh evaluation system coverage differs for {arm}/{seed}")
    mean_gain = stats.assign(value=stats.sum_gain.astype(np.float64) / stats["count"].astype(np.float64))
    original = mean_gain[mean_gain.arm == "original"].set_index("system_index").value.sort_index()
    arm_means = {arm: mean_gain[mean_gain.arm == arm].groupby("system_index", sort=True).value.mean() for arm in cfg["arms"]}
    contrasts = {
        "route": arm_means["true"].to_numpy() - original.to_numpy(),
        "shuffle": arm_means["true"].to_numpy() - arm_means["shuffled"].to_numpy(),
        "query": arm_means["true"].to_numpy() - arm_means["query_only"].to_numpy(),
    }
    estimates = {}
    for offset, (name, values) in enumerate(contrasts.items()):
        point, low, high = _paired_bootstrap(values, int(cfg["analysis"]["bootstrap_replicates"]), int(cfg["analysis"]["bootstrap_seed"]) + offset)
        estimates[name] = {"estimate": point, "ci_low": low, "ci_high": high}
    mediator = {}
    for offset, metric in enumerate(("a", "b", "rho", "cos_theta", "v_l_conditional")):
        values = stats.assign(value=stats[f"sum_{metric}"].astype(np.float64) / stats["count"].astype(np.float64))
        baseline = values[values.arm == "original"].set_index("system_index").value.sort_index()
        arms = {arm: values[values.arm == arm].groupby("system_index", sort=True).value.mean() for arm in cfg["arms"]}
        point, low, high = _paired_bootstrap(arms["true"].to_numpy() - baseline.to_numpy(), 4000, 86211 + offset)
        mediator[metric] = {
            "system_equal_arm_means": {"original": float(baseline.mean()), **{arm: float(value.mean()) for arm, value in arms.items()}},
            "true_minus_original": {"estimate": point, "ci_low": low, "ci_high": high},
        }
    routing = estimates["route"]["ci_low"] > 0
    shuffle = estimates["shuffle"]["ci_low"] > 0
    query = estimates["query"]["ci_low"] > 0
    result = {
        "schema_version": "1.0",
        "status": "FRESH_PROSPECTIVE_ROUTING_RESCUE" if routing else "FRESH_PROSPECTIVE_NO_ROUTING_RESCUE",
        "evidence_identity": cfg["evidence_identity"],
        "environment": "articulated",
        "scientific_unit": "physical_system",
        "systems": 512,
        "contrasts": estimates,
        "routing_rescue": bool(routing),
        "persistent_specificity": {"true_gt_shuffled": bool(shuffle), "true_gt_query_only": bool(query), "both": bool(shuffle and query)},
        "mediators_and_supporting_coordinates": mediator,
        "one_task_vs_two_shard_parity": True,
        "fixed_merge_order": [0, 1],
        "freeze_receipt_sha256": sha256(output / "FREEZE_RECEIPT.json"),
        "disjointness_receipt_sha256": sha256(output / "SYSTEM_DISJOINTNESS_RECEIPT.json"),
        "selected_checkpoint_identities": json.loads((output / "FREEZE_RECEIPT.json").read_text())["selected_checkpoint_identities"],
        "original_module_hashes": receipts[0]["original_module_hashes_after"],
        "gpu_peak_bytes_by_shard": [int(receipt["gpu_peak_bytes"]) for receipt in receipts],
    }
    stats_path = output / "evaluation" / "merged_sufficient_statistics.csv.gz"
    result_path = output / "evaluation" / "FINAL_RESULT.json"
    _atomic_csv(stats_path, stats)
    result["statistics"] = str(stats_path.relative_to(root))
    result["statistics_sha256"] = sha256(stats_path)
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    prep = sub.add_parser("prepare-shard")
    prep.add_argument("--shard-index", type=int, required=True)
    prep.add_argument("--shard-count", type=int, required=True)
    sub.add_parser("merge-preparation")
    evaluate = sub.add_parser("evaluate-shard")
    evaluate.add_argument("--shard-index", type=int, required=True)
    evaluate.add_argument("--shard-count", type=int, required=True)
    evaluate.add_argument("--device", required=True)
    parity_parser = sub.add_parser("parity")
    parity_parser.add_argument("--device", required=True)
    sub.add_parser("merge-evaluation")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.root, args.config)
    elif args.command == "prepare-shard":
        result = prepare_shard(args.root, args.config, args.shard_index, args.shard_count)
    elif args.command == "merge-preparation":
        result = merge_preparation(args.root, args.config)
    elif args.command == "evaluate-shard":
        result = evaluate_shard(args.root, args.config, args.shard_index, args.shard_count, args.device)
    elif args.command == "parity":
        result = parity(args.root, args.config, args.device)
    elif args.command == "merge-evaluation":
        result = merge_evaluation(args.root, args.config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
