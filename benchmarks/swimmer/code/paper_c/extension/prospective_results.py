"""Authorized, post-release analysis for Paper C's prospective packet.

This module is intentionally a *consumer* of an already materialized outcome
ledger.  It has no target path argument and imports no outcome-release entry
point.  Every result is tied to the authorization bundle, immutable ledger,
frozen pair manifest, reference vectors, and frozen old-fit model handles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from paper_c.extension import prospective_geometry_supporting
from paper_c.extension import prospective_models
from paper_c.extension import prospective_packet
from paper_c.extension.prospective_models import (
    PAIR_KEY,
    ROW_KEY,
    bundle_digest,
    classify_fixed_sequence,
    evaluate_probe_predictions,
    evaluate_score_orientations,
    load_bundle,
    orient_pairs,
    attach_oriented_gain,
    predict_probe_models,
    predict_realization_models,
    sha256_file,
    system_equal_bootstrap,
    validate_rows,
)
from paper_c.extension.prospective_packet import (
    REQUIRED_CONSUMERS,
    validate_preoutcome_cache,
    validate_reference_vectors,
)


SCHEMA_VERSION = "1.0"
E3_ARMS = ("true", "shuffled")
E3_SEEDS = (86101, 86103, 86107)
PROBE_SHUFFLE_SALT = 79307
REALIZATION_PERMUTATION_SALT = 79401
TIE_EPSILON = 1e-12
TIE_SALT = 79103
E3_BOOTSTRAP_SEED = 87201
PROBE_BOOTSTRAP_SEED = 79501
SELECTOR_BOOTSTRAP_SEED = 79201
BOOTSTRAP_REPLICATES = 4000
SELECTOR_HANDLE_STATUS = "PAPER_C_PROSPECTIVE_SELECTOR_COMPOSITE_HANDLE_FROZEN"
SELECTOR_MODEL_NAMES = (
    "realization_family",
    "realization_upstream",
    "realization_full",
)
MODEL_ORIENTATION_NAMES = ("full", "family", "upstream", "permutation")
GEOMETRY_ORIENTATION_NAMES = ("lqa", "raw_cka")
ORIENTATION_NAMES = MODEL_ORIENTATION_NAMES + GEOMETRY_ORIENTATION_NAMES
PROBE_PREDICTION_NAMES = tuple(
    f"probe_{reference}_{name}"
    for reference in ("ref_b", "ref_a")
    for name in ("family", "anchor", "shuffle", "true")
)
REALIZATION_SCORE_NAMES = tuple(f"score_{name}" for name in ORIENTATION_NAMES)
PREOUTCOME_SCORE_ARRAYS = tuple(ROW_KEY) + PROBE_PREDICTION_NAMES + REALIZATION_SCORE_NAMES + (
    "permutation_donor_position",
)


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _bound_path(authorization: Mapping[str, object], logical_name: str, path: Path) -> None:
    record = authorization.get("artifacts", {}).get(logical_name)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"authorization does not bind {logical_name}")
    path = Path(path)
    if Path(str(record.get("path", ""))).resolve() != path.resolve():
        raise RuntimeError(f"{logical_name} path differs from authorization")
    if str(record.get("sha256", "")) != sha256_file(path):
        raise RuntimeError(f"{logical_name} is stale")


def _verify_frozen_consumer_implementations(authorization: Mapping[str, object]) -> None:
    """Re-hash every authorized consumer and its current code at analysis time."""

    artifact = authorization.get("artifacts", {}).get("source_and_input_hashes")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("authorization does not bind source_and_input_hashes")
    source_path = Path(str(artifact.get("path", "")))
    if not source_path.is_file() or artifact.get("sha256") != sha256_file(source_path):
        raise RuntimeError("authorized source-and-input manifest is missing or stale")
    source = _json(source_path)
    source_hashes = source.get("source_hashes")
    child_records = source.get("consumer_implementation_manifests")
    if not isinstance(source_hashes, Mapping) or not isinstance(child_records, Mapping):
        raise RuntimeError("source-and-input manifest misses consumer bindings")

    current_paths = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        str(Path(prospective_models.__file__).resolve()): sha256_file(Path(prospective_models.__file__).resolve()),
        str(Path(prospective_packet.__file__).resolve()): sha256_file(Path(prospective_packet.__file__).resolve()),
    }
    for path, digest in current_paths.items():
        if source_hashes.get(path) != digest:
            raise RuntimeError(f"authorized consumer source is stale: {path}")

    consumers = authorization.get("consumers")
    if not isinstance(consumers, list):
        raise RuntimeError("authorization does not contain consumer records")
    by_name = {str(record.get("logical_name")): record for record in consumers if isinstance(record, Mapping)}
    if set(by_name) != REQUIRED_CONSUMERS or set(child_records) != REQUIRED_CONSUMERS:
        raise RuntimeError("authorized consumer population is incomplete")
    for logical_name in sorted(REQUIRED_CONSUMERS):
        child = child_records[logical_name]
        if not isinstance(child, Mapping):
            raise RuntimeError(f"consumer implementation binding is invalid: {logical_name}")
        path = Path(str(child.get("path", "")))
        if not path.is_file() or child.get("sha256") != sha256_file(path):
            raise RuntimeError(f"consumer implementation manifest is stale: {logical_name}")
        payload = _json(path)
        if (
            payload.get("status") != "PROSPECTIVE_CONSUMER_IMPLEMENTATION_FROZEN"
            or payload.get("logical_name") != logical_name
            or payload.get("prospective_results_path") != str(Path(__file__).resolve())
            or payload.get("prospective_results_sha256") != current_paths[str(Path(__file__).resolve())]
            or payload.get("prospective_models_path") != str(Path(prospective_models.__file__).resolve())
            or payload.get("prospective_models_sha256") != current_paths[str(Path(prospective_models.__file__).resolve())]
            or payload.get("prospective_packet_path") != str(Path(prospective_packet.__file__).resolve())
            or payload.get("prospective_packet_sha256") != current_paths[str(Path(prospective_packet.__file__).resolve())]
            or by_name[logical_name].get("implementation_sha256") != child.get("sha256")
        ):
            raise RuntimeError(f"consumer implementation semantics are stale: {logical_name}")


def verify_authorized_ledger(authorization_path: Path, ledger_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Verify and load a completed ledger without offering a release path."""
    authorization_path, ledger_path = Path(authorization_path), Path(ledger_path)
    authorization = _json(authorization_path)
    if authorization.get("status") != "PAPER_C_PROSPECTIVE_OUTCOME_AUTHORIZED":
        raise RuntimeError("authorization bundle has the wrong status")
    _verify_frozen_consumer_implementations(authorization)
    if Path(str(authorization.get("ledger_path", ""))).resolve() != ledger_path.resolve():
        raise RuntimeError("ledger path differs from authorization")
    receipt_path = Path(str(ledger_path) + ".receipt.json")
    receipt = _json(receipt_path)
    if receipt.get("status") != "PAPER_C_PROSPECTIVE_OUTCOME_LEDGER_COMPLETE":
        raise RuntimeError("outcome ledger is not complete")
    if receipt.get("authorization_bundle_sha256") != sha256_file(authorization_path):
        raise RuntimeError("ledger receipt is bound to another authorization")
    if receipt.get("ledger_sha256") != sha256_file(ledger_path):
        raise RuntimeError("outcome ledger is stale")
    marker_path = Path(str(ledger_path) + ".release.json")
    if not marker_path.is_file():
        raise RuntimeError("outcome release marker is missing")
    marker = _json(marker_path)
    if (
        marker.get("status") != "OUTCOME_RELEASE_STARTED_IMMUTABLE"
        or marker.get("authorization_bundle_sha256") != sha256_file(authorization_path)
        or receipt.get("release_marker_sha256") != sha256_file(marker_path)
    ):
        raise RuntimeError("outcome release marker is missing, stale, or inconsistent")
    marker_artifacts = {
        "preoutcome_cache_sha256": "frozen_base_feature_cache",
        "target_sha256": "fresh_target",
        "arm_prediction_sha256": "e3_prediction_cache",
    }
    for marker_field, logical_name in marker_artifacts.items():
        artifact = authorization.get("artifacts", {}).get(logical_name)
        if not isinstance(artifact, Mapping) or marker.get(marker_field) != artifact.get("sha256"):
            raise RuntimeError(f"outcome release marker is not bound to {logical_name}")
    with np.load(ledger_path, allow_pickle=False) as archive:
        ledger = {name: archive[name] for name in archive.files}
    embedded = str(np.asarray(ledger.get("authorization_bundle_sha256", "")).item())
    if embedded != sha256_file(authorization_path):
        raise RuntimeError("ledger payload is bound to another authorization")
    return authorization, ledger


def _load_model_manifest(handle_path: Path) -> tuple[dict, str]:
    handle_path = Path(handle_path)
    handle = _json(handle_path)
    status = handle.get("status")
    if status == "PAPER_C_PROSPECTIVE_OLD_FIT_FROZEN":
        manifest_path = Path(str(handle.get("old_fit_manifest", "")))
        if handle.get("old_fit_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("old-fit freeze points to a stale manifest")
        manifest = _json(manifest_path)
    elif status == "PAPER_C_PROSPECTIVE_OLD_FIT_COMPLETE":
        manifest, manifest_path = handle, handle_path
    else:
        raise RuntimeError("model handle is neither frozen nor a completed old-fit manifest")
    if manifest.get("status") != "PAPER_C_PROSPECTIVE_OLD_FIT_COMPLETE":
        raise RuntimeError("old-fit manifest has the wrong status")
    return manifest, sha256_file(manifest_path)


def load_frozen_models(handle_path: Path, names: Sequence[str]) -> dict:
    manifest, _ = _load_model_manifest(handle_path)
    records = {str(item["logical_name"]): item for item in manifest.get("formal_models", [])}
    missing = sorted(set(names) - set(records))
    if missing:
        raise RuntimeError(f"frozen model manifest misses {missing}")
    models = {}
    for name in names:
        record = records[name]
        path = Path(str(record["path"]))
        if record.get("file_sha256") != sha256_file(path):
            raise RuntimeError(f"frozen model file is stale: {name}")
        model = load_bundle(path)
        if record.get("semantic_sha256") != bundle_digest(model):
            raise RuntimeError(f"frozen model semantics are stale: {name}")
        models[name] = model
    return models


def _selector_composite_state(
    composite_path: Path,
) -> tuple[Path, Path, dict[str, Path], Path]:
    """Validate and resolve every pre-outcome selector-state artifact.

    The compound handle is the single authorization artifact for selector-side
    state: it binds both the pre-outcome score/orientation maps and the frozen
    realization-model bundle used by the result consumer.
    """

    composite_path = Path(composite_path)
    composite = _json(composite_path)
    required = {
        "schema_version",
        "status",
        "score_and_pair_receipt",
        "score_and_pair_receipt_sha256",
        "preoutcome_scores",
        "preoutcome_scores_sha256",
        "selector_pair_manifest",
        "selector_pair_manifest_sha256",
        "orientation_maps",
        "realization_model_handle",
        "realization_model_handle_sha256",
        "realization_model_names",
        "learner_outcome_read",
    }
    if set(composite) != required:
        raise RuntimeError("selector composite handle has schema drift")
    if (
        composite.get("schema_version") != SCHEMA_VERSION
        or composite.get("status") != SELECTOR_HANDLE_STATUS
        or composite.get("learner_outcome_read") is not False
        or composite.get("realization_model_names") != list(SELECTOR_MODEL_NAMES)
    ):
        raise RuntimeError("selector composite handle has invalid semantics")

    score_path = Path(str(composite["score_and_pair_receipt"])).resolve()
    score_values_path = Path(str(composite["preoutcome_scores"])).resolve()
    pair_path = Path(str(composite["selector_pair_manifest"])).resolve()
    model_handle = Path(str(composite["realization_model_handle"])).resolve()
    if (
        not score_path.is_file()
        or composite.get("score_and_pair_receipt_sha256") != sha256_file(score_path)
        or not score_values_path.is_file()
        or composite.get("preoutcome_scores_sha256") != sha256_file(score_values_path)
        or not pair_path.is_file()
        or composite.get("selector_pair_manifest_sha256") != sha256_file(pair_path)
        or not model_handle.is_file()
        or composite.get("realization_model_handle_sha256") != sha256_file(model_handle)
    ):
        raise RuntimeError("selector composite handle contains a stale bound artifact")
    score = _json(score_path)
    if (
        score.get("status") != "PAPER_C_PROSPECTIVE_SCORES_AND_PAIRS_FROZEN"
        or score.get("pair_manifest_sha256") != sha256_file(pair_path)
        or Path(str(score.get("scores_path", ""))).resolve() != score_values_path
        or score.get("scores_sha256") != sha256_file(score_values_path)
        or score.get("reference_correction_formed") is not False
        or score.get("fresh_target_read") is not False
        or score.get("learner_outcome_read") is not False
    ):
        raise RuntimeError("selector composite handle points to an invalid score receipt")
    geometry_path = Path(str(score.get("geometry_scores_path", ""))).resolve()
    geometry_receipt_path = Path(str(score.get("geometry_receipt_path", ""))).resolve()
    if (
        not geometry_path.is_file()
        or score.get("geometry_scores_sha256") != sha256_file(geometry_path)
        or not geometry_receipt_path.is_file()
        or score.get("geometry_receipt_sha256") != sha256_file(geometry_receipt_path)
    ):
        raise RuntimeError("selector score receipt has a missing or stale geometry binding")
    geometry_receipt = _json(geometry_receipt_path)
    if (
        geometry_receipt.get("status") != prospective_geometry_supporting.MERGE_STATUS
        or Path(str(geometry_receipt.get("scores_path", ""))).resolve() != geometry_path
        or geometry_receipt.get("scores_sha256") != sha256_file(geometry_path)
        or int(geometry_receipt.get("systems", 0)) <= 0
        or int(geometry_receipt.get("rows", 0)) != int(geometry_receipt.get("systems", 0)) * 6 * 6
        or geometry_receipt.get("geometry_particles") != 512
        or geometry_receipt.get("geometry_scramble_seed") != 70311
        or any(geometry_receipt.get(name) is not False for name in (
            "target_artifact_read", "learner_prediction_decoded", "learner_loss_read",
            "learner_gain_read", "learner_outcome_read",
        ))
    ):
        raise RuntimeError("selector score receipt points to invalid geometry semantics")
    orientations = score.get("orientations")
    bound_orientations = composite.get("orientation_maps")
    expected_names = set(ORIENTATION_NAMES)
    if (
        not isinstance(orientations, Mapping)
        or not isinstance(bound_orientations, Mapping)
        or set(orientations) != expected_names
        or set(bound_orientations) != expected_names
    ):
        raise RuntimeError("selector composite handle has an invalid orientation population")
    orientation_paths: dict[str, Path] = {}
    for name in sorted(expected_names):
        score_record = orientations[name]
        bound_record = bound_orientations[name]
        if not isinstance(score_record, Mapping) or not isinstance(bound_record, Mapping):
            raise RuntimeError(f"selector orientation record is invalid: {name}")
        orientation_path = Path(str(bound_record.get("path", ""))).resolve()
        if (
            dict(score_record) != dict(bound_record)
            or not orientation_path.is_file()
            or bound_record.get("sha256") != sha256_file(orientation_path)
        ):
            raise RuntimeError(f"selector orientation is missing or stale: {name}")
        orientation_paths[name] = orientation_path
    return model_handle, score_values_path, orientation_paths, geometry_path


def _selector_model_handle(composite_path: Path) -> Path:
    """Compatibility loader for callers that only need the model handle."""

    model_handle, _, _, _ = _selector_composite_state(composite_path)
    return model_handle


def load_frozen_preoutcome_state(handle_path: Path, rows: pd.DataFrame,
                                 pairs: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Load the scores and orientations that were frozen before outcome release."""

    _, score_path, orientation_paths, geometry_path = _selector_composite_state(handle_path)
    with np.load(score_path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREOUTCOME_SCORE_ARRAYS):
            raise RuntimeError("pre-outcome score artifact has schema drift")
        raw = {name: archive[name] for name in archive.files}
    frozen_rows = validate_rows(pd.DataFrame({name: raw[name] for name in ROW_KEY}))
    if frozen_rows.duplicated(list(ROW_KEY)).any() or len(frozen_rows) != len(rows):
        raise RuntimeError("pre-outcome score artifact has invalid row coverage")
    source_index = pd.MultiIndex.from_frame(frozen_rows[list(ROW_KEY)])
    positions = source_index.get_indexer(pd.MultiIndex.from_frame(rows[list(ROW_KEY)]))
    if np.any(positions < 0) or len(np.unique(positions)) != len(rows):
        raise RuntimeError("pre-outcome scores do not align to the authorized ledger rows")
    scores = {name: np.asarray(raw[name])[positions] for name in raw if name not in ROW_KEY}
    if any(not np.isfinite(np.asarray(value)).all() for value in scores.values()):
        raise RuntimeError("pre-outcome score artifact contains a non-finite value")

    with np.load(geometry_path, allow_pickle=False) as archive:
        if set(archive.files) != set(prospective_geometry_supporting.SCORE_FIELDS):
            raise RuntimeError("bound geometry score artifact has schema drift")
        geometry = {name: archive[name] for name in archive.files}
    geometry_rows = validate_rows(pd.DataFrame({name: geometry[name] for name in ROW_KEY}))
    geometry_index = pd.MultiIndex.from_frame(geometry_rows[list(ROW_KEY)])
    geometry_positions = geometry_index.get_indexer(pd.MultiIndex.from_frame(rows[list(ROW_KEY)]))
    if (
        geometry_rows.duplicated(list(ROW_KEY)).any()
        or len(geometry_rows) != len(rows)
        or np.any(geometry_positions < 0)
        or len(np.unique(geometry_positions)) != len(rows)
    ):
        raise RuntimeError("bound geometry scores do not align to the authorized rows")
    for name in GEOMETRY_ORIENTATION_NAMES:
        expected = np.asarray(geometry[name], dtype=np.float64)[geometry_positions]
        _require_preoutcome_parity(f"score_{name}", expected, scores[f"score_{name}"])

    pair_keys = [*PAIR_KEY, "candidate_low", "candidate_high"]
    expected_pairs = pairs[pair_keys].sort_values(pair_keys).reset_index(drop=True)
    orientation_columns = {
        *pair_keys, "score_name", "score_low", "score_high", "score_difference",
        "orientation", "score_tie", "tie_epsilon", "tie_salt",
    }
    orientations: dict[str, pd.DataFrame] = {}
    for name, path in orientation_paths.items():
        frame = pd.read_csv(path)
        if set(frame.columns) != orientation_columns:
            raise RuntimeError(f"frozen orientation schema drift: {name}")
        for field in pair_keys + ["orientation", "tie_salt"]:
            frame[field] = frame[field].astype(np.int64)
        frame = frame.sort_values(pair_keys).reset_index(drop=True)
        if not frame[pair_keys].equals(expected_pairs) or not (frame.score_name.astype(str) == name).all():
            raise RuntimeError(f"frozen orientation population differs from the pair manifest: {name}")
        if not frame.orientation.isin((-1, 1)).all():
            raise RuntimeError(f"frozen orientation has a non-binary direction: {name}")
        if not np.allclose(frame.tie_epsilon.to_numpy(np.float64), TIE_EPSILON, rtol=0.0, atol=0.0):
            raise RuntimeError(f"frozen orientation tie epsilon drift: {name}")
        if not (frame.tie_salt == TIE_SALT).all():
            raise RuntimeError(f"frozen orientation tie salt drift: {name}")
        orientations[name] = frame
    return scores, orientations


def load_frozen_selector_models(handle_path: Path, names: Sequence[str]) -> dict:
    """Load only the realization models authorized by a composite handle."""

    requested = tuple(map(str, names))
    if set(requested) != set(SELECTOR_MODEL_NAMES) or len(requested) != len(SELECTOR_MODEL_NAMES):
        raise RuntimeError("selector consumer requested a model population outside the frozen composite handle")
    model_handle = _selector_model_handle(handle_path)
    return load_frozen_models(model_handle, requested)


def _load_pairs(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".csv", ".gz"}:
        frame = pd.read_csv(path)
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, list):
            frame = pd.DataFrame(value)
        elif isinstance(value, dict) and isinstance(value.get("pairs"), list):
            frame = pd.DataFrame(value["pairs"])
        else:
            key = next((name for name in ("selected_pairs_path", "pair_manifest_path", "artifact_path")
                        if isinstance(value, dict) and name in value), None)
            if key is None:
                raise ValueError("pair JSON contains neither records nor a bound pair path")
            child = Path(str(value[key]))
            if not child.is_absolute():
                child = path.parent / child
            expected = value.get("selected_pairs_sha256", value.get("pair_manifest_sha256", value.get("artifact_sha256")))
            if expected is not None and expected != sha256_file(child):
                raise RuntimeError("pair JSON points to a stale pair table")
            frame = _load_pairs(child)
    else:
        raise ValueError("pair manifest must be CSV, CSV.GZ, or JSON")
    required = set(PAIR_KEY) | {"candidate_low", "candidate_high"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pair manifest misses {missing}")
    result = frame.copy()
    for name in required:
        result[name] = result[name].astype(np.int64)
    if result.duplicated(list(PAIR_KEY)).any():
        raise ValueError("pair manifest contains more than one pair per system-query cell")
    if not ((result.candidate_low >= 0) & (result.candidate_high < 6)
            & (result.candidate_low < result.candidate_high)).all():
        raise ValueError("pair candidates must satisfy 0 <= low < high < 6")
    return result.sort_values(list(PAIR_KEY)).reset_index(drop=True)


def _base_candidates(ledger: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    key_names = [f"base_{name}" for name in ROW_KEY]
    required = set(key_names) | {
        "base_prediction", "base_normalized_target", "base_squared_error", "base_sse",
        "base_z_p_anchor", "base_delta_segment", "base_delta_persistent", "base_delta_predicted_query",
    }
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"outcome ledger misses base arrays: {missing}")
    candidate_index = np.asarray(ledger["base_candidate_index"], dtype=np.int64)
    take = candidate_index >= 0
    rows = validate_rows(pd.DataFrame({name: np.asarray(ledger[f"base_{name}"])[take] for name in ROW_KEY}))
    values = {
        "prediction": np.asarray(ledger["base_prediction"], dtype=np.float64)[take],
        "target": np.asarray(ledger["base_normalized_target"], dtype=np.float64)[take],
        "sse": np.asarray(ledger["base_sse"], dtype=np.float64)[take],
        "z_anchor": np.asarray(ledger["base_z_p_anchor"], dtype=np.float64)[take],
        "delta_segment": np.asarray(ledger["base_delta_segment"], dtype=np.float64)[take],
        "delta_persistent": np.asarray(ledger["base_delta_persistent"], dtype=np.float64)[take],
        "delta_predicted_query": np.asarray(ledger["base_delta_predicted_query"], dtype=np.float64)[take],
    }
    if rows.duplicated(list(ROW_KEY)).any() or not all(len(value) == len(rows) for value in values.values()):
        raise ValueError("candidate ledger has invalid row coverage")
    squared = np.asarray(ledger["base_squared_error"], dtype=np.float64)[take]
    if squared.ndim != 2 or not np.allclose(values["sse"], squared.sum(axis=1, dtype=np.float64), rtol=1e-12, atol=1e-12):
        raise ValueError("candidate SSE differs from per-coordinate squared error")

    baseline_take = candidate_index == -1
    baseline = pd.DataFrame({name: np.asarray(ledger[f"base_{name}"])[baseline_take]
                             for name in PAIR_KEY})
    baseline["baseline_sse"] = np.asarray(ledger["base_sse"], dtype=np.float64)[baseline_take]
    baseline["anchor_position"] = np.flatnonzero(baseline_take)
    if baseline.duplicated(list(PAIR_KEY)).any():
        raise ValueError("ledger contains duplicate baseline contexts")
    joined = rows.merge(baseline, on=list(PAIR_KEY), how="left", validate="many_to_one", sort=False)
    if joined.baseline_sse.isna().any():
        raise ValueError("candidate rows do not all have a baseline")
    width = values["prediction"].shape[1]
    values["baseline_sse"] = joined.baseline_sse.to_numpy(np.float64)
    values["anchor_prediction"] = np.asarray(ledger["base_prediction"], dtype=np.float64)[
        joined.anchor_position.to_numpy(np.int64)
    ]
    values["realized_gain"] = (values["baseline_sse"] - values["sse"]) / float(width)
    return rows, values


def _bootstrap_table(frame: pd.DataFrame, metric: str, seed: int,
                     replicates: int) -> dict[str, float | int]:
    return system_equal_bootstrap(frame[metric].to_numpy(np.float64), frame.system_index.to_numpy(np.int64),
                                  replicates, seed)


def analyze_e3(ledger: Mapping[str, np.ndarray], rows: pd.DataFrame,
               base: Mapping[str, np.ndarray], replicates: int = BOOTSTRAP_REPLICATES) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = {*(f"e3_{name}" for name in ROW_KEY), "e3_arm", "e3_optimization_seed", "e3_sse"}
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"outcome ledger misses E3 arrays: {missing}")
    e3 = pd.DataFrame({name: np.asarray(ledger[f"e3_{name}"]) for name in ROW_KEY})
    e3["arm"] = np.asarray(ledger["e3_arm"]).astype(str)
    e3["optimization_seed"] = np.asarray(ledger["e3_optimization_seed"], dtype=np.int64)
    e3["arm_mse"] = np.asarray(ledger["e3_sse"], dtype=np.float64) / float(base["prediction"].shape[1])
    expected = {(tuple(map(int, key)), arm, seed)
                for key in rows[list(ROW_KEY)].itertuples(index=False, name=None)
                for arm in E3_ARMS for seed in E3_SEEDS}
    observed = {(tuple(map(int, key)), arm, int(seed))
                for key, arm, seed in zip(e3[list(ROW_KEY)].itertuples(index=False, name=None),
                                          e3.arm, e3.optimization_seed)}
    if observed != expected or len(e3) != len(expected):
        raise ValueError("E3 ledger is not the frozen candidate x arm x seed product")
    original = rows.copy()
    original["baseline_mse"] = base["baseline_sse"] / float(base["prediction"].shape[1])
    original["original_mse"] = base["sse"] / float(base["prediction"].shape[1])
    original["G_original"] = base["realized_gain"]
    e3 = e3.merge(original, on=list(ROW_KEY), how="left", validate="many_to_one")
    e3["gain"] = e3.baseline_mse - e3.arm_mse
    wide = e3.pivot(index=[*ROW_KEY, "optimization_seed"], columns="arm", values="gain").reset_index()
    wide = wide.merge(original[[*ROW_KEY, "G_original"]], on=list(ROW_KEY), validate="many_to_one")
    wide["route"] = wide["true"] - wide.G_original
    wide["specificity"] = wide["true"] - wide["shuffled"]
    system = wide.groupby("system_index", sort=True).agg(
        G_original=("G_original", "mean"), G_true=("true", "mean"),
        G_shuffled=("shuffled", "mean"), route=("route", "mean"),
        specificity=("specificity", "mean"), rows=("route", "size"),
    ).reset_index()
    primary = {
        "route": _bootstrap_table(system, "route", E3_BOOTSTRAP_SEED, replicates),
        "specificity": _bootstrap_table(system, "specificity", E3_BOOTSTRAP_SEED, replicates),
    }
    per_seed = {}
    for seed, group in wide.groupby("optimization_seed", sort=True):
        seed_system = group.groupby("system_index", sort=True)[["route", "specificity"]].mean().reset_index()
        per_seed[str(int(seed))] = {name: float(seed_system[name].mean()) for name in ("route", "specificity")}
    summaries = []
    for name in ("G_original", "G_true", "G_shuffled"):
        summaries.append({"metric": name, **_bootstrap_table(system, name, E3_BOOTSTRAP_SEED + 10, replicates)})
    summaries.extend({"metric": name, **primary[name]} for name in ("route", "specificity"))
    return {"primary": primary, "per_seed_point_estimates": per_seed}, system, pd.DataFrame(summaries)


def _align_reference(rows: pd.DataFrame, reference: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    reference_rows = pd.DataFrame({field: reference[field] for field in ROW_KEY})
    if reference_rows.duplicated(list(ROW_KEY)).any():
        raise ValueError("reference vectors contain duplicate rows")
    target_index = pd.MultiIndex.from_frame(reference_rows[list(ROW_KEY)])
    positions = target_index.get_indexer(pd.MultiIndex.from_frame(rows[list(ROW_KEY)]))
    if np.any(positions < 0):
        raise ValueError("reference vectors do not cover ledger candidate rows")
    return np.asarray(reference[name], dtype=np.float64)[positions]


def _require_preoutcome_parity(name: str, recomputed: np.ndarray, frozen: np.ndarray) -> None:
    recomputed = np.asarray(recomputed, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.float64)
    if recomputed.shape != frozen.shape or not np.allclose(recomputed, frozen, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"pre-outcome frozen state differs from deterministic recomputation: {name}")


def _require_orientation_parity(name: str, recomputed: pd.DataFrame,
                                frozen: pd.DataFrame) -> pd.DataFrame:
    pair_keys = [*PAIR_KEY, "candidate_low", "candidate_high"]
    parity_columns = [*pair_keys, "orientation", "score_tie", "tie_salt"]
    numeric_columns = ["score_low", "score_high", "score_difference", "tie_epsilon"]
    frozen = frozen.sort_values(pair_keys).reset_index(drop=True)
    recomputed = recomputed.sort_values(pair_keys).reset_index(drop=True)
    if any(not np.array_equal(recomputed[column].to_numpy(), frozen[column].to_numpy())
           for column in parity_columns):
        raise RuntimeError(f"frozen orientation differs from deterministic recomputation: {name}")
    if any(not np.allclose(recomputed[column].to_numpy(np.float64),
                           frozen[column].to_numpy(np.float64),
                           rtol=1e-10, atol=1e-12)
           for column in numeric_columns):
        raise RuntimeError(f"frozen orientation score differs from deterministic recomputation: {name}")
    return frozen


def validate_preoutcome_consumer_state(selector_handle: Path, probe_handle: Path,
                                       base_cache_path: Path, pair_path: Path,
                                       expected_systems: int = 512) -> dict[str, object]:
    """Fully validate frozen probe/selector state before irreversible release."""

    with np.load(base_cache_path, allow_pickle=False) as archive:
        base_cache = {name: archive[name] for name in archive.files}
    validate_preoutcome_cache(base_cache, expected_systems)
    rows = validate_rows(pd.DataFrame({name: base_cache[name] for name in ROW_KEY}))
    pairs = _load_pairs(pair_path)
    preoutcome, orientations = load_frozen_preoutcome_state(selector_handle, rows, pairs)

    probe_names = [
        f"probe_{reference}_{name}"
        for reference in ("ref_b", "ref_a")
        for name in ("family", "anchor", "shuffle", "true")
    ]
    probe_models = load_frozen_models(probe_handle, probe_names)
    for reference_name in ("ref_b", "ref_a"):
        model_set = {
            name: probe_models[f"probe_{reference_name}_{name}"]
            for name in ("family", "anchor", "shuffle", "true")
        }
        recomputed, _ = predict_probe_models(
            model_set,
            rows,
            np.asarray(base_cache["z_p_anchor"], dtype=np.float64),
            np.asarray(base_cache["delta_persistent"], dtype=np.float64),
            PROBE_SHUFFLE_SALT,
        )
        for name, value in recomputed.items():
            logical_name = f"probe_{reference_name}_{name}"
            _require_preoutcome_parity(logical_name, value, preoutcome[logical_name])

    selector_models = load_frozen_selector_models(selector_handle, SELECTOR_MODEL_NAMES)
    model_set = {
        name: selector_models[f"realization_{name}"]
        for name in ("family", "upstream", "full")
    }
    recomputed_scores, recomputed_donor = predict_realization_models(
        model_set,
        rows,
        np.asarray(base_cache["delta_segment"], dtype=np.float64),
        np.asarray(base_cache["delta_persistent"], dtype=np.float64),
        np.asarray(base_cache["delta_predicted_query"], dtype=np.float64),
        REALIZATION_PERMUTATION_SALT,
    )
    if not np.array_equal(
        np.asarray(recomputed_donor, dtype=np.int64),
        np.asarray(preoutcome["permutation_donor_position"], dtype=np.int64),
    ):
        raise RuntimeError("pre-outcome permutation donor map differs from deterministic recomputation")
    for name in MODEL_ORIENTATION_NAMES:
        logical_name = f"score_{name}"
        _require_preoutcome_parity(logical_name, recomputed_scores[name], preoutcome[logical_name])
        recomputed_orientation = orient_pairs(
            rows, preoutcome[logical_name], pairs, name, TIE_EPSILON, TIE_SALT,
        )
        _require_orientation_parity(name, recomputed_orientation, orientations[name])
    for name in GEOMETRY_ORIENTATION_NAMES:
        logical_name = f"score_{name}"
        recomputed_orientation = orient_pairs(
            rows, preoutcome[logical_name], pairs, name, TIE_EPSILON, TIE_SALT,
        )
        _require_orientation_parity(name, recomputed_orientation, orientations[name])
    return {
        "status": "PROSPECTIVE_PREOUTCOME_CONSUMER_STATE_VERIFIED",
        "systems": int(rows.system_index.nunique()),
        "rows": int(len(rows)),
        "pairs": int(len(pairs)),
        "probe_predictions": len(PROBE_PREDICTION_NAMES),
        "selector_orientations": len(ORIENTATION_NAMES),
    }


def analyze_probes(rows: pd.DataFrame, base: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
                   frozen: Mapping[str, object], preoutcome: Mapping[str, np.ndarray],
                   replicates: int = BOOTSTRAP_REPLICATES) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    outputs, systems, summaries = {}, [], []
    for reference_name in ("ref_b", "ref_a"):
        model_set = {name: frozen[f"probe_{reference_name}_{name}"]
                     for name in ("family", "anchor", "shuffle", "true")}
        recomputed, _ = predict_probe_models(model_set, rows, base["z_anchor"], base["delta_persistent"],
                                             PROBE_SHUFFLE_SALT)
        predictions = {}
        for name, value in recomputed.items():
            logical_name = f"probe_{reference_name}_{name}"
            if logical_name not in preoutcome:
                raise RuntimeError(f"pre-outcome score artifact misses {logical_name}")
            _require_preoutcome_parity(logical_name, value, preoutcome[logical_name])
            predictions[name] = np.asarray(preoutcome[logical_name], dtype=np.float64)
        r_b = _align_reference(rows, reference, f"mu_{reference_name}") - base["anchor_prediction"]
        aggregate = evaluate_probe_predictions(
            rows, r_b, predictions, replicates, PROBE_BOOTSTRAP_SEED
        )
        risk = {name: np.mean((r_b - prediction) ** 2, axis=1) for name, prediction in predictions.items()}
        row = rows[["system_index", "query_index"]].copy()
        row["delta_anchor"] = risk["anchor"] - risk["true"]
        row["delta_shuffle"] = risk["shuffle"] - risk["true"]
        row["delta_family"] = risk["family"] - risk["true"]
        per_system = row.groupby("system_index", sort=True)[["delta_anchor", "delta_shuffle", "delta_family"]].mean().reset_index()
        per_system.insert(1, "reference", reference_name)
        systems.append(per_system)
        outputs[reference_name] = aggregate
        for metric, result in aggregate.items():
            summaries.append({"reference": reference_name, "query_index": -1, "metric": metric, **result})
        for query, group in row.groupby("query_index", sort=True):
            for metric in ("delta_anchor", "delta_shuffle", "delta_family"):
                result = system_equal_bootstrap(group[metric].to_numpy(), group.system_index.to_numpy(),
                                                replicates, PROBE_BOOTSTRAP_SEED + 200 + int(query))
                summaries.append({"reference": reference_name, "query_index": int(query), "metric": metric, **result})
    return outputs, pd.concat(systems, ignore_index=True), pd.DataFrame(summaries)


def analyze_selector_with_state(rows: pd.DataFrame, base: Mapping[str, np.ndarray], pairs: pd.DataFrame,
                                frozen: Mapping[str, object], preoutcome: Mapping[str, np.ndarray],
                                frozen_orientations: Mapping[str, pd.DataFrame],
                                replicates: int = BOOTSTRAP_REPLICATES,
                                minimum_systems: int = 128) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    supporting_metrics = ("d_upstream", "d_permutation", "d_lqa", "d_raw_cka", "d_hash")
    system_columns = ["system_index", "d_full", "d_family", *supporting_metrics,
                      "d_full_minus_family", "pair_count"]
    if pairs.empty:
        unavailable = {"estimate": None, "ci_low": None, "ci_high": None,
                       "systems": 0, "replicates": 0, "seed": SELECTOR_BOOTSTRAP_SEED}
        result = {
            "d_full": dict(unavailable), "d_full_minus_family": dict(unavailable),
            "tie_fraction_full": {"estimate": None}, "tie_fraction_family": {"estimate": None},
            "supporting": {name: dict(unavailable) for name in supporting_metrics},
            "contributing_systems": 0, "minimum_contributing_systems": int(minimum_systems),
            "underidentified": True,
        }
        summary = pd.DataFrame([{"metric": name, **value} for name, value in (
            ("d_full", result["d_full"]), ("d_full_minus_family", result["d_full_minus_family"]),
            *tuple(result["supporting"].items()),
        )])
        return result, pd.DataFrame(columns=system_columns), summary
    model_set = {name: frozen[f"realization_{name}"] for name in ("family", "upstream", "full")}
    recomputed_scores, recomputed_donor = predict_realization_models(
        model_set, rows, base["delta_segment"], base["delta_persistent"],
        base["delta_predicted_query"], REALIZATION_PERMUTATION_SALT,
    )
    if not np.array_equal(
        np.asarray(recomputed_donor, dtype=np.int64),
        np.asarray(preoutcome.get("permutation_donor_position"), dtype=np.int64),
    ):
        raise RuntimeError("pre-outcome permutation donor map differs from deterministic recomputation")
    scores = {}
    for name in MODEL_ORIENTATION_NAMES:
        logical_name = f"score_{name}"
        if name not in recomputed_scores or logical_name not in preoutcome:
            raise RuntimeError(f"pre-outcome realization score population misses {name}")
        _require_preoutcome_parity(logical_name, recomputed_scores[name], preoutcome[logical_name])
        scores[name] = np.asarray(preoutcome[logical_name], dtype=np.float64)
    for name in GEOMETRY_ORIENTATION_NAMES:
        logical_name = f"score_{name}"
        if logical_name not in preoutcome:
            raise RuntimeError(f"pre-outcome geometry score population misses {name}")
        scores[name] = np.asarray(preoutcome[logical_name], dtype=np.float64)
    scores["hash"] = np.zeros(len(rows), dtype=np.float64)
    attached = {}
    for name in ORIENTATION_NAMES:
        recomputed = orient_pairs(rows, scores[name], pairs, name, TIE_EPSILON, TIE_SALT)
        frozen_orientation = frozen_orientations.get(name)
        if frozen_orientation is None:
            raise RuntimeError(f"frozen orientation population misses {name}")
        frozen_orientation = _require_orientation_parity(name, recomputed, frozen_orientation)
        attached[name] = attach_oriented_gain(rows, base["realized_gain"], frozen_orientation)
    hash_orientation = orient_pairs(rows, scores["hash"], pairs, "hash", TIE_EPSILON, TIE_SALT)
    attached["hash"] = attach_oriented_gain(rows, base["realized_gain"], hash_orientation)
    keys = [*PAIR_KEY, "candidate_low", "candidate_high"]
    pair_table = attached["full"][keys].copy()
    for name, frame in attached.items():
        pair_table = pair_table.merge(frame[keys + ["oriented_gain", "score_tie"]].rename(
            columns={"oriented_gain": f"d_{name}", "score_tie": f"tie_{name}"}),
            on=keys, how="inner", validate="one_to_one")
    pair_table["d_full_minus_family"] = pair_table.d_full - pair_table.d_family
    system = pair_table.groupby("system_index", sort=True).agg(
        d_full=("d_full", "mean"), d_family=("d_family", "mean"),
        d_upstream=("d_upstream", "mean"), d_permutation=("d_permutation", "mean"),
        d_lqa=("d_lqa", "mean"), d_raw_cka=("d_raw_cka", "mean"),
        d_hash=("d_hash", "mean"), d_full_minus_family=("d_full_minus_family", "mean"),
        pair_count=("d_full", "size"),
    ).reset_index()
    contributing = int(system.system_index.nunique())
    if contributing >= 2:
        confirmatory = evaluate_score_orientations(attached["full"], attached["family"], replicates,
                                                   SELECTOR_BOOTSTRAP_SEED)
    else:
        unavailable = {"estimate": float(system.d_full.mean()), "ci_low": None, "ci_high": None,
                       "systems": contributing, "replicates": 0, "seed": SELECTOR_BOOTSTRAP_SEED}
        confirmatory = {
            "d_full": dict(unavailable),
            "d_full_minus_family": {**unavailable, "estimate": float(system.d_full_minus_family.mean())},
            "tie_fraction_full": {"estimate": float(pair_table.tie_full.mean())},
            "tie_fraction_family": {"estimate": float(pair_table.tie_family.mean())},
        }
    supporting = {}
    for index, metric in enumerate(supporting_metrics):
        if contributing >= 2:
            supporting[metric] = _bootstrap_table(system, metric, SELECTOR_BOOTSTRAP_SEED + 10 + index, replicates)
        else:
            supporting[metric] = {"estimate": float(system[metric].mean()), "ci_low": None, "ci_high": None,
                                  "systems": contributing, "replicates": 0,
                                  "seed": SELECTOR_BOOTSTRAP_SEED + 10 + index}
    result = {
        **confirmatory,
        "supporting": supporting,
        "contributing_systems": contributing,
        "minimum_contributing_systems": int(minimum_systems),
        "underidentified": bool(contributing < int(minimum_systems)),
    }
    summary = []
    for metric in ("d_full", "d_full_minus_family"):
        summary.append({"metric": metric, **confirmatory[metric]})
    summary.extend({"metric": metric, **value} for metric, value in supporting.items())
    return result, system, pd.DataFrame(summary)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def analyze_authorized_packet(authorization_path: Path, ledger_path: Path, reference_path: Path,
                              pair_path: Path, probe_handle: Path, selector_handle: Path,
                              output_dir: Path, expected_systems: int = 512,
                              minimum_selector_systems: int = 128,
                              replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Analyze one authorized ledger and atomically emit all frozen results."""
    authorization, ledger = verify_authorized_ledger(authorization_path, ledger_path)
    _bound_path(authorization, "fresh_reference_vectors", reference_path)
    _bound_path(authorization, "selector_pair_manifest", pair_path)
    _bound_path(authorization, "probe_models_and_standardizers", probe_handle)
    _bound_path(authorization, "selector_orientation_models_and_maps", selector_handle)
    rows, base = _base_candidates(ledger)
    systems = np.unique(rows.system_index.to_numpy())
    if len(systems) != int(expected_systems) or not np.array_equal(systems, np.arange(expected_systems)):
        raise ValueError("authorized ledger has the wrong physical-system population")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    validate_reference_vectors(reference, expected_systems)
    pairs = _load_pairs(pair_path)
    if not set(pairs.system_index.astype(int)).issubset(set(map(int, systems))):
        raise ValueError("pair manifest contains systems outside the ledger")

    probe_names = [f"probe_{ref}_{name}" for ref in ("ref_b", "ref_a")
                   for name in ("family", "anchor", "shuffle", "true")]
    selector_names = list(SELECTOR_MODEL_NAMES)
    probe_models = load_frozen_models(probe_handle, probe_names)
    selector_models = load_frozen_selector_models(selector_handle, selector_names)
    preoutcome_scores, frozen_orientations = load_frozen_preoutcome_state(
        selector_handle, rows, pairs,
    )

    e3, e3_system, e3_figure = analyze_e3(ledger, rows, base, replicates)
    probes, probe_system, probe_figure = analyze_probes(
        rows, base, reference, probe_models, preoutcome_scores, replicates,
    )
    selector, selector_system, selector_figure = analyze_selector_with_state(
        rows, base, pairs, selector_models, preoutcome_scores, frozen_orientations,
        replicates, minimum_selector_systems,
    )
    underidentified = {"secondary_b_realization"} if selector["underidentified"] else set()
    fixed = classify_fixed_sequence(e3["primary"], probes["ref_b"], selector,
                                    underidentified=underidentified)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PAPER_C_PROSPECTIVE_RESULTS_COMPLETE",
        "primary_e3": e3,
        "secondary_a_decodability": probes,
        "secondary_b_realization": selector,
        "fixed_sequence": fixed,
        "claim_boundary": {
            "single_joint_fresh_block": True,
            "secondary_after_failure_is_exploratory": True,
            "outcome_release_triggered_by_this_module": False,
        },
    }

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"immutable result directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        files = {
            "e3_system_table.csv": e3_system,
            "probe_system_table.csv": probe_system,
            "selector_system_table.csv": selector_system,
            "figure_e3.csv": e3_figure,
            "figure_probe.csv": probe_figure,
            "figure_selector.csv": selector_figure,
        }
        for name, frame in files.items():
            _write_csv(frame, temporary / name)
        (temporary / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        input_hashes = {
            "authorization": sha256_file(Path(authorization_path)), "ledger": sha256_file(Path(ledger_path)),
            "reference": sha256_file(Path(reference_path)), "pairs": sha256_file(Path(pair_path)),
            "probe_handle": sha256_file(Path(probe_handle)), "selector_handle": sha256_file(Path(selector_handle)),
        }
        output_hashes = {name: sha256_file(temporary / name) for name in [*files, "RESULTS.json"]}
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PAPER_C_PROSPECTIVE_RESULTS_RECEIPT_COMPLETE",
            "authorization_bundle_sha256": input_hashes["authorization"],
            "ledger_sha256": input_hashes["ledger"],
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "systems": int(len(systems)), "selector_systems": int(selector["contributing_systems"]),
            "bootstrap_replicates": int(replicates),
            "outcome_release_triggered": False,
            "fixed_sequence": fixed,
        }
        (temporary / "RESULTS_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary)
        raise
    receipt_path = output_dir / "RESULTS_RECEIPT.json"
    return {**receipt, "receipt_sha256": sha256_file(receipt_path), "output_dir": str(output_dir.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an already authorized Paper-C outcome ledger")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--probe-handle", type=Path, required=True)
    parser.add_argument("--selector-handle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_authorized_packet(args.authorization, args.ledger, args.reference, args.pairs,
                                       args.probe_handle, args.selector_handle, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
