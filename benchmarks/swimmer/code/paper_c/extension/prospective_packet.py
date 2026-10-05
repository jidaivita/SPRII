"""Fail-closed engineering core for the Paper-C prospective evidence packet.

This module deliberately contains no probe fitting and no formal learner
evaluation.  It defines the immutable, learner-outcome-blind objects that the
remote executor must produce before the single outcome release:

* a whitelist-only matched-V_B pair manifest;
* reference-vector and pre-outcome feature-cache schemas;
* deterministic row-population hashes and two-shard merges; and
* an authorization bundle plus a one-shot target/prediction ledger join.

All public writers refuse to overwrite an existing artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t


SCHEMA_VERSION = "1.0"
ROW_KEY = (
    "system_index",
    "realization",
    "history_index",
    "query_index",
    "candidate_index",
)
CONTEXT_KEY = ROW_KEY[:-1]
REFERENCE_SCRAMBLES = tuple(
    [f"ref_a_vb_scramble{index}" for index in range(4)]
    + [f"ref_b_vb_scramble{index}" for index in range(4)]
)
REFERENCE_ARRAYS = ROW_KEY + (
    "mu_ref_a",
    "mu_ref_b",
    "local_value",
) + REFERENCE_SCRAMBLES
PREOUTCOME_ARRAYS = ROW_KEY + (
    "prediction_anchor",
    "prediction_candidate",
    "z_p_anchor",
    "delta_segment",
    "delta_persistent",
    "delta_predicted_query",
    "query_embedding",
    "predicted_latent_full",
)
ARM_ARRAYS = ROW_KEY + ("arm", "optimization_seed", "prediction")
CONSUMER_FIELDS = {
    "logical_name",
    "config_sha256",
    "checkpoint_or_model_sha256",
    "implementation_sha256",
    "row_population_sha256",
    "aggregation_sha256",
}
REQUIRED_CONSUMERS = {
    "primary_e3",
    "secondary_a_decodability",
    "secondary_b_realization",
}
REQUIRED_AUTHORIZATION_ARTIFACTS = {
    "system_context_manifest",
    "disjointness_receipt",
    "reference_receipts",
    "fresh_reference_vectors",
    "frozen_base_feature_cache",
    "selector_pair_manifest",
    "selector_orientation_models_and_maps",
    "e3_checkpoint_and_shuffle_manifest",
    "e3_prediction_cache",
    "probe_models_and_standardizers",
    "endpoint_and_aggregation_manifest",
    "source_and_input_hashes",
    "fresh_target",
}
OUTCOME_TOKENS = (
    "target",
    "loss",
    "gain",
    "outcome",
    "squared_error",
    "sse",
    "bayes_correction",
    "realized_utility",
)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without interpreting the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _as_arrays(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in values.items()}


def _require_exact_keys(values: Mapping[str, np.ndarray], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected) - set(values))
    extra = sorted(set(values) - set(expected))
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def _require_integral_key(values: Mapping[str, np.ndarray], field: str, rows: int) -> np.ndarray:
    array = np.asarray(values[field])
    if array.shape != (rows,) or array.dtype.kind not in "iu":
        raise ValueError(f"{field} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _key_matrix(values: Mapping[str, np.ndarray], fields: Sequence[str] = ROW_KEY) -> np.ndarray:
    if not fields:
        raise ValueError("row key cannot be empty")
    rows = len(np.asarray(values[fields[0]]))
    return np.column_stack([_require_integral_key(values, field, rows) for field in fields])


def _canonical_order(values: Mapping[str, np.ndarray], fields: Sequence[str] = ROW_KEY) -> np.ndarray:
    keys = _key_matrix(values, fields)
    return np.lexsort(tuple(keys[:, index] for index in reversed(range(keys.shape[1]))))


def _sorted_arrays(values: Mapping[str, np.ndarray], fields: Sequence[str] = ROW_KEY) -> dict[str, np.ndarray]:
    arrays = _as_arrays(values)
    order = _canonical_order(arrays, fields)
    return {name: array[order] for name, array in arrays.items()}


def row_population_sha256(values: Mapping[str, np.ndarray], fields: Sequence[str] = ROW_KEY) -> str:
    """Hash a row population independently of input enumeration order."""

    keys = _key_matrix(values, fields)
    order = np.lexsort(tuple(keys[:, index] for index in reversed(range(keys.shape[1]))))
    payload = [list(map(int, row)) for row in keys[order]]
    return sha256_json(payload)


def _candidate_system_ids(values: Mapping[str, np.ndarray]) -> np.ndarray:
    systems = np.asarray(values["system_index"], dtype=np.int64)
    return np.unique(systems)


def validate_candidate_coverage(
    values: Mapping[str, np.ndarray],
    expected_systems: int | None = None,
    expected_system_ids: Sequence[int] | None = None,
) -> dict:
    """Validate one context per system and a complete 6x6 query/candidate grid."""

    rows = len(np.asarray(values[ROW_KEY[0]]))
    keys = _key_matrix(values)
    if len(np.unique(keys, axis=0)) != rows:
        raise ValueError("candidate row keys are duplicated")
    systems = np.unique(keys[:, 0])
    if expected_system_ids is not None and not np.array_equal(systems, np.asarray(expected_system_ids, dtype=np.int64)):
        raise ValueError("candidate system identities differ from the frozen shard population")
    if expected_systems is not None and not np.array_equal(systems, np.arange(expected_systems, dtype=np.int64)):
        raise ValueError("candidate systems must be the complete zero-based frozen population")
    if rows != len(systems) * 36:
        raise ValueError("candidate population must contain exactly 36 rows per system")
    for system in systems:
        cell = keys[keys[:, 0] == system]
        if len(np.unique(cell[:, 1:3], axis=0)) != 1:
            raise ValueError(f"system {system} does not have exactly one realization/history context")
        observed = set(map(tuple, cell[:, 3:5].tolist()))
        required = {(query, candidate) for query in range(6) for candidate in range(6)}
        if observed != required:
            raise ValueError(f"system {system} has incomplete query/candidate coverage")
    return {
        "rows": rows,
        "systems": len(systems),
        "system_ids": systems.tolist(),
        "row_population_sha256": row_population_sha256(values),
    }


def shard_owner(system_index: int, shard_count: int = 2) -> int:
    if shard_count != 2:
        raise ValueError("prospective packet V1 is frozen to exactly two shards")
    return int(system_index) % shard_count


def _paired_bound(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if difference.shape != (4,) or not np.isfinite(difference).all():
        raise ValueError("each reference stream requires four finite paired scrambles")
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / 2.0)
    bound = abs(mean) + float(t.ppf(0.975, df=3)) * se
    return mean, se, bound


def _pair_digest(key: tuple[int, ...], low: int, high: int, salt: int) -> str:
    payload = [*map(int, key), int(low), int(high), int(salt)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _forbidden_outcome_columns(columns: Sequence[str]) -> list[str]:
    forbidden = []
    for column in columns:
        normalized = str(column).lower()
        if any(token in normalized for token in OUTCOME_TOKENS):
            forbidden.append(str(column))
    return sorted(forbidden)


def build_unbiased_pair_manifest(
    rows: pd.DataFrame,
    sesoi: float,
    ref_a_fraction: float = 0.5,
    ref_b_fraction: float = 0.25,
    salt: int = 79101,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build score-independent pairs using only local value and matched V_B.

    Benign diagnostic columns are ignored.  Any column whose name denotes a
    learner target/outcome is rejected before projection, so accidental joins
    cannot silently enter pair membership.
    """

    forbidden = _forbidden_outcome_columns(rows.columns)
    if forbidden:
        raise RuntimeError(f"pair builder received forbidden outcome columns: {forbidden}")
    required = set(ROW_KEY) | {"local_value"} | set(REFERENCE_SCRAMBLES)
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"pair-builder columns are incomplete: {missing}")
    if not np.isfinite(float(sesoi)) or float(sesoi) <= 0:
        raise ValueError("SESOI must be finite and positive")
    if (float(ref_a_fraction), float(ref_b_fraction)) != (0.5, 0.25):
        raise ValueError("prospective packet V1 matching fractions are frozen to 0.5 and 0.25")
    projected = rows[list(ROW_KEY) + ["local_value"] + list(REFERENCE_SCRAMBLES)].copy()
    validate_candidate_coverage({field: projected[field].to_numpy() for field in ROW_KEY})
    if not np.isfinite(projected[["local_value", *REFERENCE_SCRAMBLES]].to_numpy(dtype=float)).all():
        raise ValueError("pair-builder reference values must be finite")

    records: list[dict] = []
    for key, cell in projected.groupby(list(CONTEXT_KEY), sort=True):
        indexed = cell.set_index("candidate_index").sort_index()
        if not indexed.index.is_unique:
            raise ValueError(f"duplicated candidate axis for pair cell {key}")
        if list(indexed.index) != list(range(6)):
            raise ValueError(f"incomplete candidate axis for pair cell {key}")
        for low, high in combinations(range(6), 2):
            local_valid = bool(indexed.at[low, "local_value"] > 0 and indexed.at[high, "local_value"] > 0)
            a_mean, a_se, a_bound = _paired_bound(
                indexed.loc[low, list(REFERENCE_SCRAMBLES[:4])].to_numpy(dtype=float),
                indexed.loc[high, list(REFERENCE_SCRAMBLES[:4])].to_numpy(dtype=float),
            )
            b_mean, b_se, b_bound = _paired_bound(
                indexed.loc[low, list(REFERENCE_SCRAMBLES[4:])].to_numpy(dtype=float),
                indexed.loc[high, list(REFERENCE_SCRAMBLES[4:])].to_numpy(dtype=float),
            )
            proposal = bool(local_valid and a_bound <= ref_a_fraction * sesoi)
            verified = bool(proposal and b_bound <= ref_b_fraction * sesoi)
            records.append({
                **dict(zip(CONTEXT_KEY, map(int, key))),
                "candidate_low": low,
                "candidate_high": high,
                "local_valid": local_valid,
                "ref_a_delta_vb": a_mean,
                "ref_a_delta_vb_se_numerical": a_se,
                "ref_a_bound": a_bound,
                "ref_a_proposal": proposal,
                "ref_b_delta_vb": b_mean,
                "ref_b_delta_vb_se_numerical": b_se,
                "ref_b_bound": b_bound,
                "ref_b_verified": verified,
                "pair_sha256": _pair_digest(tuple(map(int, key)), low, high, salt),
            })
    all_pairs = pd.DataFrame.from_records(records).sort_values(
        [*CONTEXT_KEY, "candidate_low", "candidate_high"]
    ).reset_index(drop=True)
    eligible = all_pairs[all_pairs.ref_b_verified].copy()
    if len(eligible):
        eligible = eligible.sort_values(
            [*CONTEXT_KEY, "pair_sha256", "candidate_low", "candidate_high"]
        )
        selected = eligible.groupby(list(CONTEXT_KEY), sort=True, as_index=False).head(1)
        selected = selected.sort_values(list(CONTEXT_KEY)).reset_index(drop=True)
    else:
        selected = eligible.reset_index(drop=True)
    return all_pairs, selected


def _validate_reference_provenance(provenance: Mapping[str, object]) -> None:
    required = {"geometry_particles", "particle_seed", "ref_a_scramble_seeds", "ref_b_scramble_seeds", "source_hashes"}
    if set(provenance) != required:
        raise ValueError(f"reference provenance keys must be exactly {sorted(required)}")
    if int(provenance["geometry_particles"]) != 512:
        raise ValueError("reference geometry particle count is frozen to 512")
    for stream in ("ref_a_scramble_seeds", "ref_b_scramble_seeds"):
        seeds = list(provenance[stream])
        if len(seeds) != 4 or len(set(map(int, seeds))) != 4:
            raise ValueError(f"{stream} must contain four distinct seeds")
    hashes = provenance["source_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or not all(_is_sha256(value) for value in hashes.values()):
        raise ValueError("reference source hashes must be a non-empty SHA-256 mapping")


def validate_reference_vectors(
    values: Mapping[str, np.ndarray], expected_systems: int | None = None,
    expected_system_ids: Sequence[int] | None = None,
) -> dict:
    arrays = _as_arrays(values)
    _require_exact_keys(arrays, REFERENCE_ARRAYS, "fresh reference vectors")
    coverage = validate_candidate_coverage(arrays, expected_systems, expected_system_ids)
    rows = coverage["rows"]
    for name in ("mu_ref_a", "mu_ref_b"):
        if arrays[name].shape != (rows, 32) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite with shape ({rows}, 32)")
    for name in ("local_value", *REFERENCE_SCRAMBLES):
        if arrays[name].shape != (rows,) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be a finite row vector")
    return coverage


def _artifact_receipt_path(path: Path) -> Path:
    return Path(str(path) + ".receipt.json")


def write_reference_vectors(
    path: Path, values: Mapping[str, np.ndarray], provenance: Mapping[str, object],
    expected_systems: int | None = None,
) -> dict:
    _validate_reference_provenance(provenance)
    coverage = validate_reference_vectors(values, expected_systems)
    sorted_values = _sorted_arrays(values)
    receipt_path = _artifact_receipt_path(path)
    if receipt_path.exists():
        raise RuntimeError(f"immutable receipt already exists: {receipt_path}")
    _atomic_npz(path, sorted_values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "FRESH_REFERENCE_VECTORS_COMPLETE_LEARNER_BLIND",
        **coverage,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in sorted_values.items()},
        "provenance": dict(provenance),
        "artifact_sha256": sha256_file(path),
        "learner_outcomes_read": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def validate_preoutcome_cache(
    values: Mapping[str, np.ndarray], expected_systems: int | None = None,
    expected_system_ids: Sequence[int] | None = None,
) -> dict:
    arrays = _as_arrays(values)
    forbidden = _forbidden_outcome_columns(arrays.keys())
    if forbidden:
        raise RuntimeError(f"pre-outcome cache contains forbidden arrays: {forbidden}")
    _require_exact_keys(arrays, PREOUTCOME_ARRAYS, "pre-outcome cache")
    coverage = validate_candidate_coverage(arrays, expected_systems, expected_system_ids)
    rows = coverage["rows"]
    dimensions = {
        "prediction_anchor": 32,
        "prediction_candidate": 32,
        "z_p_anchor": 64,
        "delta_segment": 128,
        "delta_persistent": 64,
        "delta_predicted_query": 64,
        "query_embedding": 64,
        "predicted_latent_full": 64,
    }
    for name, width in dimensions.items():
        if arrays[name].shape != (rows, width) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite with shape ({rows}, {width})")
    table = pd.DataFrame({field: arrays[field] for field in CONTEXT_KEY})
    for name in ("prediction_anchor", "z_p_anchor", "query_embedding"):
        frame = pd.DataFrame(arrays[name])
        frame[list(CONTEXT_KEY)] = table
        for _, group in frame.groupby(list(CONTEXT_KEY), sort=False):
            payload = group.drop(columns=list(CONTEXT_KEY)).to_numpy()
            if not np.array_equal(payload, np.repeat(payload[:1], len(payload), axis=0)):
                raise ValueError(f"{name} differs across candidates in one context")
    return coverage


def _validate_cache_provenance(provenance: Mapping[str, object]) -> None:
    required = {"base_checkpoint_sha256", "implementation_sha256", "source_hashes"}
    if set(provenance) != required:
        raise ValueError(f"cache provenance keys must be exactly {sorted(required)}")
    if not _is_sha256(provenance["base_checkpoint_sha256"]) or not _is_sha256(provenance["implementation_sha256"]):
        raise ValueError("cache checkpoint and implementation hashes must be SHA-256")
    hashes = provenance["source_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or not all(_is_sha256(value) for value in hashes.values()):
        raise ValueError("cache source hashes must be a non-empty SHA-256 mapping")


def write_preoutcome_cache(
    path: Path, values: Mapping[str, np.ndarray], provenance: Mapping[str, object],
    expected_systems: int | None = None,
) -> dict:
    _validate_cache_provenance(provenance)
    coverage = validate_preoutcome_cache(values, expected_systems)
    sorted_values = _sorted_arrays(values)
    receipt_path = _artifact_receipt_path(path)
    if receipt_path.exists():
        raise RuntimeError(f"immutable receipt already exists: {receipt_path}")
    _atomic_npz(path, sorted_values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PROSPECTIVE_PREOUTCOME_CACHE_COMPLETE",
        **coverage,
        "artifact_sha256": sha256_file(path),
        "provenance": dict(provenance),
        "contains_target_or_outcome": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def validate_arm_predictions(
    values: Mapping[str, np.ndarray], base_cache: Mapping[str, np.ndarray],
    arms: Sequence[str] = ("true", "shuffled"), seeds: Sequence[int] = (86101, 86103, 86107),
) -> dict:
    arrays = _as_arrays(values)
    forbidden = _forbidden_outcome_columns(arrays.keys())
    if forbidden:
        raise RuntimeError(f"arm prediction cache contains forbidden arrays: {forbidden}")
    _require_exact_keys(arrays, ARM_ARRAYS, "arm prediction cache")
    rows = len(arrays["system_index"])
    _key_matrix(arrays)
    if arrays["prediction"].shape != (rows, 32) or not np.isfinite(arrays["prediction"]).all():
        raise ValueError("arm prediction must be finite with 32 output coordinates")
    if arrays["arm"].shape != (rows,) or arrays["optimization_seed"].shape != (rows,):
        raise ValueError("arm and optimization_seed must be row vectors")
    base_keys = {tuple(row) for row in _key_matrix(base_cache)}
    expected = {(key, str(arm), int(seed)) for key in base_keys for arm in arms for seed in seeds}
    observed = {
        (tuple(map(int, key)), str(arm), int(seed))
        for key, arm, seed in zip(_key_matrix(arrays), arrays["arm"], arrays["optimization_seed"])
    }
    if observed != expected or rows != len(expected):
        raise ValueError("arm prediction cache is not the exact base-row x arm x seed product")
    return {"rows": rows, "base_rows": len(base_keys), "arms": list(arms), "seeds": list(map(int, seeds))}


def write_arm_prediction_cache(
    path: Path,
    values: Mapping[str, np.ndarray],
    base_cache: Mapping[str, np.ndarray],
    provenance: Mapping[str, object],
) -> dict:
    required = {"checkpoint_table_sha256", "implementation_sha256", "shuffle_map_sha256", "source_hashes"}
    if set(provenance) != required:
        raise ValueError(f"arm-cache provenance keys must be exactly {sorted(required)}")
    if not all(_is_sha256(provenance[field]) for field in required - {"source_hashes"}):
        raise ValueError("arm-cache manifest, implementation, and shuffle-map hashes must be SHA-256")
    hashes = provenance["source_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or not all(_is_sha256(value) for value in hashes.values()):
        raise ValueError("arm-cache source hashes must be a non-empty SHA-256 mapping")
    meta = validate_arm_predictions(values, base_cache)
    arrays = _as_arrays(values)
    order = np.lexsort((
        arrays["optimization_seed"],
        arrays["arm"].astype(str),
        arrays["candidate_index"],
        arrays["query_index"],
        arrays["history_index"],
        arrays["realization"],
        arrays["system_index"],
    ))
    sorted_values = {name: value[order] for name, value in arrays.items()}
    receipt_path = _artifact_receipt_path(path)
    if receipt_path.exists():
        raise RuntimeError(f"immutable receipt already exists: {receipt_path}")
    _atomic_npz(path, sorted_values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PROSPECTIVE_E3_PREDICTION_CACHE_COMPLETE_PREOUTCOME",
        **meta,
        "base_row_population_sha256": row_population_sha256(base_cache),
        "artifact_sha256": sha256_file(path),
        "provenance": dict(provenance),
        "contains_target_or_outcome": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def validate_target_rows(values: Mapping[str, np.ndarray], expected_systems: int) -> dict:
    arrays = _as_arrays(values)
    _require_exact_keys(arrays, (*CONTEXT_KEY, "normalized_target"), "fresh target rows")
    rows = len(arrays["system_index"])
    keys = _key_matrix(arrays, CONTEXT_KEY)
    if len(np.unique(keys, axis=0)) != rows:
        raise ValueError("fresh target context keys are duplicated")
    systems = np.unique(keys[:, 0])
    if not np.array_equal(systems, np.arange(expected_systems, dtype=np.int64)) or rows != expected_systems * 6:
        raise ValueError("fresh target rows must cover all six queries for every frozen system")
    for system in systems:
        cell = keys[keys[:, 0] == system]
        if len(np.unique(cell[:, 1:3], axis=0)) != 1 or set(cell[:, 3]) != set(range(6)):
            raise ValueError(f"fresh target context is incomplete for system {system}")
    target = arrays["normalized_target"]
    if target.shape != (rows, 32) or not np.isfinite(target).all():
        raise ValueError("normalized_target must be finite with 32 output coordinates")
    return {"rows": rows, "systems": expected_systems, "row_population_sha256": row_population_sha256(arrays, CONTEXT_KEY)}


def _merge_candidate_shards(
    shard_paths: Sequence[Path], output_path: Path, provenance: Mapping[str, object],
    expected_systems: int, validator: Callable[..., dict], writer: Callable[..., dict],
) -> dict:
    if len(shard_paths) != 2:
        raise ValueError("prospective packet merge requires exactly [shard0, shard1]")
    parts: list[dict[str, np.ndarray]] = []
    expected_keys: set[str] | None = None
    shard_hashes: list[str] = []
    for shard_index, path in enumerate(map(Path, shard_paths)):
        with np.load(path, allow_pickle=False) as loaded:
            part = {name: loaded[name] for name in loaded.files}
        if expected_keys is None:
            expected_keys = set(part)
        elif set(part) != expected_keys:
            raise ValueError("two shards have different array schemas")
        systems = _candidate_system_ids(part)
        if not len(systems) or any(shard_owner(int(system)) != shard_index for system in systems):
            raise ValueError(f"shard {shard_index} violates system_index % 2 ownership")
        validator(part, expected_system_ids=systems)
        parts.append(part)
        shard_hashes.append(sha256_file(path))
    merged = {name: np.concatenate([parts[0][name], parts[1][name]], axis=0) for name in sorted(expected_keys or ())}
    validator(merged, expected_systems=expected_systems)
    merged_provenance = dict(provenance)
    merged_provenance["merge"] = {"fixed_order": [0, 1], "shard_sha256": shard_hashes}
    # The public provenance schema intentionally excludes merge metadata.  Bind
    # it as another source hash using a canonical JSON digest.
    source_hashes = dict(merged_provenance.pop("source_hashes"))
    source_hashes["two_shard_merge"] = sha256_json(merged_provenance.pop("merge"))
    merged_provenance["source_hashes"] = source_hashes
    merge_receipt_path = Path(str(output_path) + ".merge.receipt.json")
    if merge_receipt_path.exists():
        raise RuntimeError(f"immutable merge receipt already exists: {merge_receipt_path}")
    receipt = writer(output_path, merged, merged_provenance, expected_systems=expected_systems)
    merge_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PROSPECTIVE_TWO_SHARD_MERGE_COMPLETE",
        "fixed_merge_order": [0, 1],
        "shard_sha256": shard_hashes,
        "merged_artifact_sha256": sha256_file(output_path),
        "merged_artifact_receipt_sha256": sha256_file(_artifact_receipt_path(output_path)),
        "row_population_sha256": receipt["row_population_sha256"],
    }
    _atomic_json(merge_receipt_path, merge_receipt)
    receipt["fixed_merge_order"] = [0, 1]
    receipt["shard_sha256"] = shard_hashes
    receipt["merge_receipt_sha256"] = sha256_file(merge_receipt_path)
    return receipt


def merge_reference_shards(
    shard_paths: Sequence[Path], output_path: Path, provenance: Mapping[str, object], expected_systems: int = 512,
) -> dict:
    return _merge_candidate_shards(
        shard_paths, output_path, provenance, expected_systems, validate_reference_vectors, write_reference_vectors
    )


def merge_preoutcome_shards(
    shard_paths: Sequence[Path], output_path: Path, provenance: Mapping[str, object], expected_systems: int = 512,
) -> dict:
    return _merge_candidate_shards(
        shard_paths, output_path, provenance, expected_systems, validate_preoutcome_cache, write_preoutcome_cache
    )


def merge_arm_prediction_shards(
    shard_paths: Sequence[Path],
    base_cache_path: Path,
    output_path: Path,
    provenance: Mapping[str, object],
) -> dict:
    if len(shard_paths) != 2:
        raise ValueError("prospective packet merge requires exactly [shard0, shard1]")
    with np.load(base_cache_path, allow_pickle=False) as loaded:
        base = {name: loaded[name] for name in loaded.files}
    validate_preoutcome_cache(base)
    parts = []
    shard_hashes = []
    for shard_index, path in enumerate(map(Path, shard_paths)):
        with np.load(path, allow_pickle=False) as loaded:
            part = {name: loaded[name] for name in loaded.files}
        systems = np.unique(part["system_index"])
        if not len(systems) or any(shard_owner(int(system)) != shard_index for system in systems):
            raise ValueError(f"arm shard {shard_index} violates system_index % 2 ownership")
        take = np.isin(base["system_index"], systems)
        base_part = {name: value[take] for name, value in base.items()}
        validate_arm_predictions(part, base_part)
        parts.append(part)
        shard_hashes.append(sha256_file(path))
    merged = {name: np.concatenate([parts[0][name], parts[1][name]], axis=0) for name in ARM_ARRAYS}
    validate_arm_predictions(merged, base)
    merge_binding = sha256_json({"fixed_order": [0, 1], "shard_sha256": shard_hashes})
    bound_provenance = dict(provenance)
    source_hashes = dict(bound_provenance["source_hashes"])
    source_hashes["two_shard_merge"] = merge_binding
    bound_provenance["source_hashes"] = source_hashes
    receipt = write_arm_prediction_cache(output_path, merged, base, bound_provenance)
    merge_receipt_path = Path(str(output_path) + ".merge.receipt.json")
    _atomic_json(merge_receipt_path, {
        "schema_version": SCHEMA_VERSION,
        "status": "PROSPECTIVE_E3_TWO_SHARD_MERGE_COMPLETE",
        "fixed_merge_order": [0, 1],
        "shard_sha256": shard_hashes,
        "base_cache_sha256": sha256_file(base_cache_path),
        "merged_artifact_sha256": sha256_file(output_path),
        "merged_artifact_receipt_sha256": sha256_file(_artifact_receipt_path(output_path)),
    })
    receipt.update({
        "fixed_merge_order": [0, 1],
        "shard_sha256": shard_hashes,
        "merge_receipt_sha256": sha256_file(merge_receipt_path),
    })
    return receipt


def validate_consumer_record(record: Mapping[str, object]) -> None:
    if set(record) != CONSUMER_FIELDS:
        raise ValueError(f"consumer record fields must be exactly {sorted(CONSUMER_FIELDS)}")
    if not isinstance(record["logical_name"], str) or not record["logical_name"]:
        raise ValueError("consumer logical_name must be non-empty")
    for field in CONSUMER_FIELDS - {"logical_name"}:
        if not _is_sha256(record[field]):
            raise ValueError(f"consumer {field} must be a lowercase SHA-256")


def freeze_outcome_authorization(
    bundle_path: Path,
    consumers: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, Path],
    ledger_path: Path,
) -> dict:
    """Freeze a non-circular authorization bundle before target/prediction join."""

    bundle_path, ledger_path = Path(bundle_path), Path(ledger_path)
    release_marker = Path(str(ledger_path) + ".release.json")
    if ledger_path.exists() or release_marker.exists():
        raise RuntimeError("an outcome release already exists or has started")
    if not consumers:
        raise ValueError("at least one authorized consumer is required")
    for record in consumers:
        validate_consumer_record(record)
    names = [str(record["logical_name"]) for record in consumers]
    if len(set(names)) != len(names):
        raise ValueError("consumer logical names are duplicated")
    if set(names) != REQUIRED_CONSUMERS:
        raise ValueError(f"consumer population must be exactly {sorted(REQUIRED_CONSUMERS)}")
    if not artifacts:
        raise ValueError("authorization must bind its immutable input artifacts")
    if set(artifacts) != REQUIRED_AUTHORIZATION_ARTIFACTS:
        missing = sorted(REQUIRED_AUTHORIZATION_ARTIFACTS - set(artifacts))
        extra = sorted(set(artifacts) - REQUIRED_AUTHORIZATION_ARTIFACTS)
        raise ValueError(f"authorization artifact population mismatch: missing={missing}, extra={extra}")
    bound_artifacts = {}
    for logical_name, path in sorted(artifacts.items()):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        bound_artifacts[str(logical_name)] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "status": "PAPER_C_PROSPECTIVE_OUTCOME_AUTHORIZED",
        "consumers": sorted([dict(record) for record in consumers], key=lambda row: row["logical_name"]),
        "artifacts": bound_artifacts,
        "ledger_path": str(ledger_path.resolve()),
        "outcome_released": False,
    }
    _atomic_json(bundle_path, bundle)
    return {**bundle, "authorization_bundle_sha256": sha256_file(bundle_path)}


def _verify_authorization_artifact(bundle: Mapping[str, object], logical_name: str, path: Path) -> None:
    artifact = bundle.get("artifacts", {}).get(logical_name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"authorization does not bind artifact {logical_name}")
    if Path(artifact["path"]).resolve() != Path(path).resolve() or artifact["sha256"] != sha256_file(path):
        raise RuntimeError(f"authorized artifact is missing or stale: {logical_name}")


def _join_targets(cache: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]) -> np.ndarray:
    target_lookup = {
        tuple(map(int, key)): target
        for key, target in zip(_key_matrix(targets, CONTEXT_KEY), targets["normalized_target"])
    }
    try:
        return np.stack([target_lookup[tuple(map(int, key))] for key in _key_matrix(cache, CONTEXT_KEY)])
    except KeyError as error:
        raise ValueError(f"candidate cache has no matching target context: {error}") from error


def materialize_outcome_ledger(
    authorization_path: Path,
    preoutcome_cache_path: Path,
    target_path: Path,
    ledger_path: Path,
    arm_prediction_path: Path,
    expected_systems: int = 512,
) -> dict:
    """Perform the only authorized target/prediction join.

    An exclusive release marker is written before any target array is loaded.
    It is intentionally retained after success or failure, making retries
    impossible without an explicit superseding authorization outside this API.
    """

    authorization_path = Path(authorization_path)
    ledger_path = Path(ledger_path)
    receipt_path = _artifact_receipt_path(ledger_path)
    marker_path = Path(str(ledger_path) + ".release.json")
    if ledger_path.exists() or receipt_path.exists() or marker_path.exists():
        raise RuntimeError("outcome ledger already exists or release has already started")
    bundle = json.loads(authorization_path.read_text())
    if bundle.get("status") != "PAPER_C_PROSPECTIVE_OUTCOME_AUTHORIZED" or bundle.get("outcome_released") is not False:
        raise RuntimeError("outcome authorization is invalid")
    if Path(bundle.get("ledger_path", "")).resolve() != ledger_path.resolve():
        raise RuntimeError("authorization names a different ledger path")
    _verify_authorization_artifact(bundle, "frozen_base_feature_cache", preoutcome_cache_path)
    _verify_authorization_artifact(bundle, "fresh_target", target_path)
    _verify_authorization_artifact(bundle, "e3_prediction_cache", arm_prediction_path)

    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "OUTCOME_RELEASE_STARTED_IMMUTABLE",
        "authorization_bundle_sha256": sha256_file(authorization_path),
        "preoutcome_cache_sha256": sha256_file(preoutcome_cache_path),
        "target_sha256": sha256_file(target_path),
        "arm_prediction_sha256": sha256_file(arm_prediction_path),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    with np.load(preoutcome_cache_path, allow_pickle=False) as loaded:
        cache = {name: loaded[name] for name in loaded.files}
    with np.load(target_path, allow_pickle=False) as loaded:
        targets = {name: loaded[name] for name in loaded.files}
    validate_preoutcome_cache(cache, expected_systems)
    target_meta = validate_target_rows(targets, expected_systems)
    candidate_targets = _join_targets(cache, targets)

    context_keys = _key_matrix(cache, CONTEXT_KEY)
    first_indices = np.unique(context_keys, axis=0, return_index=True)[1]
    first_indices = np.sort(first_indices)
    if len(first_indices) != expected_systems * 6:
        raise RuntimeError("baseline context extraction did not yield 3,072 rows")
    baseline_prediction = cache["prediction_anchor"][first_indices]
    baseline_target = candidate_targets[first_indices]
    base_prediction = np.concatenate([baseline_prediction, cache["prediction_candidate"]], axis=0)
    base_target = np.concatenate([baseline_target, candidate_targets], axis=0)
    baseline_keys = {field: cache[field][first_indices] for field in CONTEXT_KEY}
    base_values: dict[str, np.ndarray] = {
        f"base_{field}": np.concatenate([
            baseline_keys[field], cache[field]
        ]) for field in CONTEXT_KEY
    }
    base_values["base_candidate_index"] = np.concatenate([
        np.full(len(first_indices), -1, dtype=np.int64), cache["candidate_index"].astype(np.int64, copy=False)
    ])
    base_values.update({
        "base_prediction": base_prediction,
        "base_normalized_target": base_target,
        "base_squared_error": (base_target - base_prediction) ** 2,
        "base_sse": np.sum((base_target - base_prediction) ** 2, axis=1, dtype=np.float64),
        "base_z_p_anchor": np.concatenate([cache["z_p_anchor"][first_indices], cache["z_p_anchor"]], axis=0),
        "base_delta_segment": np.concatenate([np.zeros_like(cache["delta_segment"][first_indices]), cache["delta_segment"]], axis=0),
        "base_delta_persistent": np.concatenate([np.zeros_like(cache["delta_persistent"][first_indices]), cache["delta_persistent"]], axis=0),
        "base_delta_predicted_query": np.concatenate([
            np.zeros_like(cache["delta_predicted_query"][first_indices]), cache["delta_predicted_query"]
        ], axis=0),
    })

    ledger_values = base_values
    with np.load(arm_prediction_path, allow_pickle=False) as loaded:
        arm_values = {name: loaded[name] for name in loaded.files}
    arm_meta = validate_arm_predictions(arm_values, cache)
    arm_targets = _join_targets(arm_values, targets)
    ledger_values.update({
        **{f"e3_{field}": arm_values[field] for field in ROW_KEY},
        "e3_arm": arm_values["arm"],
        "e3_optimization_seed": arm_values["optimization_seed"],
        "e3_prediction": arm_values["prediction"],
        "e3_normalized_target": arm_targets,
        "e3_squared_error": (arm_targets - arm_values["prediction"]) ** 2,
        "e3_sse": np.sum((arm_targets - arm_values["prediction"]) ** 2, axis=1, dtype=np.float64),
    })
    arm_rows = int(arm_meta["rows"])
    ledger_values["authorization_bundle_sha256"] = np.asarray(sha256_file(authorization_path))
    _atomic_npz(ledger_path, ledger_values)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PAPER_C_PROSPECTIVE_OUTCOME_LEDGER_COMPLETE",
        "authorization_bundle_sha256": sha256_file(authorization_path),
        "release_marker_sha256": sha256_file(marker_path),
        "ledger_sha256": sha256_file(ledger_path),
        "base_rows": int(len(base_prediction)),
        "baseline_rows": int(len(first_indices)),
        "candidate_rows": int(len(cache["system_index"])),
        "e3_rows": arm_rows,
        "target_rows": int(target_meta["rows"]),
        "fixed_merge_order": [0, 1],
        "outcome_released_once": True,
    }
    _atomic_json(receipt_path, receipt)
    return receipt
