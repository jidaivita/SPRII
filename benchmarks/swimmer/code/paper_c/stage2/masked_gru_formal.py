"""Authorized formal consumer for the frozen masked-GRU replication.

The base-training module deliberately stops at an authorization barrier.  This
module implements the remaining consumer, but every command that touches the
existing formal population is guarded by a second, implementation-bound,
one-shot authorization receipt.  Merely importing or testing this module does
not open or materialize any formal learner outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.linear_model import Ridge

from paper_c.coupled_sled.formal_data import load_arrays
from paper_c.coupled_sled.learner import (
    MASKED_GRU_ARCHITECTURE_TAG,
    _normalized,
    load_tagged_persistent_jepa,
)
from paper_c.stage1 import bayes_alignment as alignment_module
from paper_c.stage1.bayes_alignment import alignment_metrics
from paper_c.stage2 import delta_gated_isolation as e3
from paper_c.stage2 import masked_gru_cross_architecture as base
from paper_c.stage2 import routing_intervention as routing
from paper_c.extension import prospective_models as probe_models


STATUS_PLAN = "MASKED_GRU_FORMAL_IMPLEMENTATION_FROZEN"
STATUS_CACHE_SHARD = "MASKED_GRU_FEATURE_CACHE_SHARD_COMPLETE"
STATUS_CACHE = "MASKED_GRU_FEATURE_CACHE_COMPLETE"
STATUS_ADAPTER = "MASKED_GRU_DELTA_GATED_ADAPTER_COMPLETE"
STATUS_ADAPTER_BARRIER = "MASKED_GRU_12_ADAPTERS_AUTHENTICATED"
STATUS_AUTHORIZATION = "MASKED_GRU_ONE_SHOT_FORMAL_AUTHORIZATION"
STATUS_FORMAL_SHARD = "MASKED_GRU_FORMAL_EVALUATION_SHARD_COMPLETE"
STATUS_STAGE1 = "MASKED_GRU_STAGE1A_STAGE1B_COMPLETE"
STATUS_FINAL = "MASKED_GRU_CROSS_ARCHITECTURE_COMPLETE"

CANONICAL_ROUTING_CONFIG = "configs/downstream_routing_intervention_v1.json"
CANONICAL_STAGE1A_ROOT = "runs/diagnostics/stage1_symptom_localization_v1"
CANONICAL_ROUTING_ROOT = "runs/intervention/downstream_routing_v1"
FORMAL_TEST = "tests/unit/test_masked_gru_formal.py"
FORMAL_JOB_MATRIX = "work/MASKED_GRU_FORMAL_REMOTE_MATRIX_V1.md"
CACHE_SHARDS = 2
FORMAL_SHARDS = 2
CACHE_SORT_KEYS = ("system_index", "realization", "history_index", "query_index", "candidate_index", "row_index")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def merge_cache_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Fixed-order cache merge used by production and synthetic parity tests."""
    if not parts:
        raise ValueError("at least one cache shard is required")
    names = set(parts[0])
    if any(set(part) != names for part in parts):
        raise RuntimeError("masked-GRU cache shard schemas differ")
    merged = {name: np.concatenate([part[name] for part in parts]) for name in names}
    missing = [name for name in CACHE_SORT_KEYS if name not in merged]
    if missing:
        raise RuntimeError(f"masked-GRU cache lacks sorted merge keys: {missing}")
    order = np.lexsort(tuple(merged[name] for name in reversed(CACHE_SORT_KEYS)))
    merged = {name: value[order] for name, value in merged.items()}
    if len(np.unique(merged["row_index"])) != len(merged["row_index"]):
        raise RuntimeError("masked-GRU cache merge duplicates immutable rows")
    key_rows = np.rec.fromarrays([merged[name] for name in CACHE_SORT_KEYS], names=CACHE_SORT_KEYS)
    if len(np.unique(key_rows)) != len(key_rows):
        raise RuntimeError("masked-GRU cache merge duplicates scientific row keys")
    return merged


def merge_statistic_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge sufficient statistics in caller-supplied fixed shard order."""
    if not parts:
        raise ValueError("at least one statistic shard is required")
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["system_index", "arm", "seed"], kind="stable",
    ).reset_index(drop=True)
    if result.duplicated(["system_index", "arm", "seed"]).any():
        raise RuntimeError("masked-GRU formal merge duplicates system/arm/seed")
    return result


def canonical_cache_content_sha(cache: dict[str, np.ndarray]) -> str:
    """Hash arrays without NPZ container metadata such as ZIP timestamps."""
    digest = hashlib.sha256()
    for name in sorted(cache):
        value = np.ascontiguousarray(cache[name])
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_frame_sha(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["system_index", "arm", "seed"], kind="stable").reset_index(drop=True)
    serialized = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stage1_summary_metrics() -> tuple[str, ...]:
    return (
        "delta_l_norm", "r_b_norm", "cos_theta", "rho", "a", "b", "v_l_conditional",
        "delta_l_norm_ref_a", "r_b_norm_ref_a", "cos_theta_ref_a", "rho_ref_a",
        "a_ref_a", "b_ref_a", "v_l_conditional_ref_a",
        "segment_norm", "persistent_norm", "predicted_norm", "prediction_norm",
    )


def _runtime_versions() -> dict:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index, "name": properties.name,
                "total_memory": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    return {
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": cuda_devices,
    }


def _direct_dependency_paths(root: Path, config_path: Path, cfg: dict) -> list[Path]:
    paths = [
        config_path, root / cfg["protocol"], root / cfg["probe_protocol"],
        Path(__file__).resolve(), root / FORMAL_TEST, root / FORMAL_JOB_MATRIX,
        root / CANONICAL_ROUTING_CONFIG,
        Path(base.__file__).resolve(), Path(e3.__file__).resolve(), Path(routing.__file__).resolve(),
        Path(alignment_module.__file__).resolve(),
        Path(probe_models.__file__).resolve(),
        root / "code/paper_c/coupled_sled/learner.py",
        root / "code/paper_c/coupled_sled/formal_data.py",
    ]
    routing_cfg = _routing_cfg(root)
    for environment, item in routing_cfg["environments"].items():
        for split in ("train", "select"):
            paths.extend((root / item[f"{split}_arrays"], root / item[f"{split}_manifest"]))
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("masked-GRU direct dependency list contains duplicates")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"masked-GRU direct dependencies missing: {missing}")
    return resolved


def _assert_no_preexisting_formal_outputs(root: Path, cfg: dict) -> None:
    output = _output(root, cfg)
    forbidden = []
    cache_root = output / "cache"
    if cache_root.exists():
        forbidden.extend(path for path in cache_root.glob("*/formal/**/*") if path.is_file())
    for relative in ("formal_eval", "stage1", "decodability"):
        path = output / relative
        if path.exists():
            forbidden.extend(item for item in path.rglob("*") if item.is_file())
    if (output / "FINAL_RESULT.json").exists():
        forbidden.append(output / "FINAL_RESULT.json")
    for environment, item in cfg["environments"].items():
        for seed in item["base_seeds"]:
            base_root = _base_root(root, cfg, environment, int(seed))
            forbidden.extend(path for path in base_root.glob("formal*") if path.is_file())
            forbidden.extend(
                child for path in base_root.glob("formal*") if path.is_dir()
                for child in path.rglob("*") if child.is_file()
            )
    if forbidden:
        names = [str(path.relative_to(root)) for path in sorted(set(forbidden))]
        raise RuntimeError(f"pre-existing unbound formal result/cache found: {names}")


def _ref_a_bindings(root: Path, environment: str) -> dict:
    config_path = root / "configs/stage1_bayes_alignment_v1.json"
    reference_manifest = root / "runs/diagnostics/stage1_bayes_alignment_v1/reference_manifest_frozen.json"
    cfg = json.loads(config_path.read_text())
    shard_count = int(cfg["execution"][f"{environment}_shards"])
    shards = []
    for shard in range(shard_count):
        directory = root / "runs/diagnostics/stage1_bayes_alignment_v1/shards" / f"{environment}_{shard:02d}_of_{shard_count:02d}"
        vector, receipt = directory / "alignment_vectors.npz", directory / "receipt.json"
        shards.append({
            "shard_index": shard,
            "vector": str(vector.relative_to(root)), "vector_sha256": base.sha256(vector),
            "receipt": str(receipt.relative_to(root)), "receipt_sha256": base.sha256(receipt),
        })
    return {
        "config": str(config_path.relative_to(root)), "config_sha256": base.sha256(config_path),
        "reference_manifest": str(reference_manifest.relative_to(root)),
        "reference_manifest_sha256": base.sha256(reference_manifest),
        "shard_count": shard_count, "shards": shards,
    }


def ordered_row_positions(source_rows: np.ndarray, requested_rows: np.ndarray) -> np.ndarray:
    source = np.asarray(source_rows, dtype=np.int64)
    requested = np.asarray(requested_rows, dtype=np.int64)
    if source.ndim != 1 or requested.ndim != 1:
        raise RuntimeError("row join requires one-dimensional row ids")
    if len(np.unique(source)) != len(source) or len(np.unique(requested)) != len(requested):
        raise RuntimeError("row join contains duplicate row ids")
    lookup = {int(row): position for position, row in enumerate(source)}
    missing = [int(row) for row in requested if int(row) not in lookup]
    if missing:
        raise RuntimeError(f"row join is missing requested rows: {missing[:5]}")
    positions = np.asarray([lookup[int(row)] for row in requested], dtype=np.int64)
    if not np.array_equal(source[positions], requested):
        raise AssertionError("row join did not preserve exact requested order")
    return positions


def ordered_reference_join(manifest: pd.DataFrame, requested_rows: np.ndarray,
                           reference_index: pd.DataFrame,
                           key_names: list[str]) -> pd.DataFrame:
    requested = np.asarray(requested_rows, dtype=np.int64)
    if not isinstance(manifest.index, pd.RangeIndex) or manifest.index.start != 0 or manifest.index.step != 1:
        raise RuntimeError("formal manifest must have an exact zero-based RangeIndex")
    if len(np.unique(requested)) != len(requested) or np.any(requested < 0) or np.any(requested >= len(manifest)):
        raise RuntimeError("requested formal row ids are duplicate or out of bounds")
    candidate = manifest.iloc[requested].copy()
    candidate.insert(0, "row_index", requested)
    if "candidate_index" not in candidate or np.any(candidate.candidate_index.to_numpy(np.int64) < 0):
        raise RuntimeError("requested formal rows are not all candidate rows")
    if candidate.duplicated(key_names).any() or reference_index.duplicated(key_names).any():
        raise RuntimeError("physical-reference join keys are not unique")
    joined = candidate[["row_index", *key_names]].merge(
        reference_index, on=key_names, how="left", validate="one_to_one", sort=False,
    )
    if len(joined) != len(requested) or not np.array_equal(joined.row_index.to_numpy(np.int64), requested):
        raise AssertionError("physical-reference join changed exact requested row order")
    return joined


def _routing_cfg(root: Path) -> dict:
    return json.loads((root / CANONICAL_ROUTING_CONFIG).read_text())


def _output(root: Path, cfg: dict) -> Path:
    return root / cfg["output_root"] / "formal_pipeline"


def _base_root(root: Path, cfg: dict, environment: str, base_seed: int) -> Path:
    return root / cfg["output_root"] / "base_learners" / f"{environment}_s{int(base_seed)}"


def _cache_root(root: Path, cfg: dict, environment: str, base_seed: int, split: str) -> Path:
    return _output(root, cfg) / "cache" / f"{environment}_s{int(base_seed)}" / split


def _adapter_root(root: Path, cfg: dict, base_seed: int, arm: str, seed: int) -> Path:
    return _output(root, cfg) / "adapters" / f"articulated_s{int(base_seed)}" / f"{arm}_s{int(seed)}"


def _formal_eval_root(root: Path, cfg: dict, base_seed: int) -> Path:
    return _output(root, cfg) / "formal_eval" / f"articulated_s{int(base_seed)}"


def expected_adapter_jobs(cfg: dict) -> list[tuple[int, str, int]]:
    return [
        (int(base_seed), arm, int(seed))
        for base_seed in cfg["environments"]["articulated"]["base_seeds"]
        for arm in cfg["adapter"]["arms"]
        for seed in cfg["adapter"]["seeds"]
    ]


def _verify_base(root: Path, cfg: dict, environment: str,
                 base_seed: int, require_competent: bool = True) -> tuple[Path, dict]:
    if environment not in cfg["environments"] or int(base_seed) not in set(
        map(int, cfg["environments"][environment]["base_seeds"])
    ):
        raise RuntimeError("base learner is outside the frozen four-checkpoint schedule")
    base_root = _base_root(root, cfg, environment, base_seed)
    receipt_path = base_root / "BASE_COMPETENCE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    checkpoint = base_root / "masked_gru_frozen.pt"
    allowed_status = {base.STATUS_COMPETENT} if require_competent else {
        base.STATUS_COMPETENT, base.STATUS_UNDERIDENTIFIED,
    }
    if (
        receipt.get("status") not in allowed_status
        or receipt.get("architecture_tag") != MASKED_GRU_ARCHITECTURE_TAG
        or receipt.get("checkpoint_sha256") != base.sha256(checkpoint)
        or receipt.get("normalization_sha256") != base.sha256(base_root / "train_only_normalization.npz")
        or receipt.get("formal_outcomes_read") is not False
    ):
        raise RuntimeError("masked-GRU base checkpoint is not competent and immutable")
    return checkpoint, receipt


def _verify_base_barrier(root: Path, config_path: Path, cfg: dict) -> dict:
    base.verify_freeze(root, config_path, cfg)
    path = root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"
    receipt = json.loads(path.read_text())
    if (
        receipt.get("status") != base.STATUS_BARRIER
        or receipt.get("formal_outcomes_read") is not False
        or receipt.get("articulated_primary_authorized") is not True
    ):
        raise RuntimeError("masked-GRU base formal barrier is not valid")
    for environment, item in cfg["environments"].items():
        for seed in item["base_seeds"]:
            checkpoint, base_receipt = _verify_base(
                root, cfg, environment, int(seed), require_competent=(environment == "articulated"),
            )
            bound = receipt["receipts"][f"{environment}:{int(seed)}"]
            receipt_path = _base_root(root, cfg, environment, int(seed)) / "BASE_COMPETENCE_RECEIPT.json"
            if (
                bound["receipt_sha256"] != base.sha256(receipt_path)
                or bound["checkpoint_sha256"] != base.sha256(checkpoint)
                or bound["status"] != base_receipt["status"]
            ):
                raise RuntimeError("base authorization barrier is stale")
    if any(
        value["status"] != base.STATUS_COMPETENT
        for key, value in receipt["receipts"].items() if key.startswith("articulated:")
    ):
        raise RuntimeError("both Articulated bases must be competent")
    coupled_status = [
        receipt["receipts"][f"coupled:{int(seed)}"]["status"] == base.STATUS_COMPETENT
        for seed in cfg["environments"]["coupled"]["base_seeds"]
    ]
    if receipt.get("coupled_supporting_authorized") != coupled_status:
        raise RuntimeError("Coupled supporting authorization flags are stale")
    return receipt


def competent_base_seeds(root: Path, cfg: dict, environment: str) -> list[int]:
    result = []
    for seed in cfg["environments"][environment]["base_seeds"]:
        _, receipt = _verify_base(root, cfg, environment, int(seed), require_competent=False)
        if receipt["status"] == base.STATUS_COMPETENT:
            result.append(int(seed))
    if environment == "articulated" and len(result) != 2:
        raise RuntimeError("Articulated primary is underidentified")
    return result


def _model(root: Path, cfg: dict, environment: str, base_seed: int,
           arrays, device: torch.device):
    checkpoint, receipt = _verify_base(root, cfg, environment, base_seed)
    model_cfg = base._environment_config(root, cfg, environment, base_seed)
    model = load_tagged_persistent_jepa(
        checkpoint, model_cfg, MASKED_GRU_ARCHITECTURE_TAG, device,
    )
    dimensions = receipt["dimensions"]
    observed = {
        "history_dim": int(arrays.history.shape[-1]),
        "query_dim": int(arrays.query_action.shape[-1]),
        "target_dim": int(arrays.target.shape[-1]),
    }
    if dimensions != observed:
        raise RuntimeError("formal arrays differ from tagged checkpoint dimensions")
    norms = _load_npz(_base_root(root, cfg, environment, base_seed) / "train_only_normalization.npz")
    before = routing.freeze_original(model)
    model.eval()
    return model, norms, before


def _split_inputs(root: Path, environment: str, split: str) -> tuple[Path, Path, str]:
    routing_cfg = _routing_cfg(root)
    item = routing_cfg["environments"][environment]
    if split in ("train", "select"):
        return root / item[f"{split}_arrays"], root / item[f"{split}_manifest"], item["candidate_column"]
    if split != "formal":
        raise ValueError(split)
    receipt_path = root / CANONICAL_ROUTING_ROOT / "formal_inputs" / environment / "FORMAL_INPUTS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "DOWNSTREAM_ROUTING_FORMAL_INPUTS_MATERIALIZED":
        raise RuntimeError("canonical formal input receipt is invalid")
    arrays_path, manifest_path = root / receipt["arrays"], root / receipt["manifest"]
    if base.sha256(arrays_path) != receipt["arrays_sha256"] or base.sha256(manifest_path) != receipt["manifest_sha256"]:
        raise RuntimeError("canonical formal input arrays or manifest changed")
    return arrays_path, manifest_path, item["candidate_column"]


def _active_positions(arrays, environment: str, split: str, shard_index: int,
                      shard_count: int, root: Path) -> np.ndarray:
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid cache shard")
    if split == "formal":
        active = arrays.history_mask[:, 1] == 1
    else:
        active_conditions = _routing_cfg(root)["environments"][environment]["active_condition_indices"]
        active = np.isin(arrays.condition, np.asarray(active_conditions, dtype=np.int64))
        active &= arrays.history_mask[:, 1] == 1
    active &= arrays.system_index % shard_count == shard_index
    return np.flatnonzero(active)


def _physical_reference_means(root: Path, environment: str,
                              row_index: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Recover frozen physical means, never reuse canonical learner corrections.

    Ref-B is recovered from the authenticated routing materialization. Ref-A is
    recovered from the independently computed Stage-1B vectors.  In both cases
    the canonical anchor prediction is used only to invert the already frozen
    correction into the learner-independent physical predictive mean.
    """
    formal_root = root / CANONICAL_ROUTING_ROOT / "formal_inputs" / environment
    input_receipt_path = formal_root / "FORMAL_INPUTS_RECEIPT.json"
    input_receipt = json.loads(input_receipt_path.read_text())
    correction_path = root / input_receipt["correction"]
    if base.sha256(correction_path) != input_receipt["correction_sha256"]:
        raise RuntimeError("frozen ref-B correction artifact changed")
    correction = _load_npz(correction_path)
    canonical_cache_root = root / CANONICAL_ROUTING_ROOT / "cache" / environment / "formal"
    cache_receipt_path = canonical_cache_root / "merged.receipt.json"
    cache_receipt = json.loads(cache_receipt_path.read_text())
    cache_path = root / cache_receipt["merged_cache"]
    if cache_receipt.get("status") != routing.STATUS_CACHE or base.sha256(cache_path) != cache_receipt["merged_sha256"]:
        raise RuntimeError("canonical formal cache provenance is invalid")
    canonical = _load_npz(cache_path)
    correction_values = correction.get("bayes_correction", correction.get("r_b_ref_b"))
    correction_rows = correction.get("row_index", canonical["row_index"])
    if (
        correction_values is None or correction_values.ndim != 2
        or len(correction_values) != len(correction_rows)
        or not np.array_equal(correction_rows, canonical["row_index"])
        or len(np.unique(correction_rows)) != len(correction_rows)
        or len(canonical["prediction_anchor"]) != len(correction_rows)
    ):
        raise RuntimeError("ref-B correction does not align with canonical formal cache")
    positions = ordered_row_positions(canonical["row_index"], row_index)
    canonical_anchor = canonical["prediction_anchor"].astype(np.float64)
    mu_b = canonical_anchor + correction_values.astype(np.float64)
    manifest = pd.read_csv(root / input_receipt["manifest"])
    manifest = routing.canonicalize_formal_manifest(manifest, environment)
    key_names = ["system_index", "realization", "history_index", "candidate_index", "query_index"]
    stage1b_cfg = json.loads((root / "configs/stage1_bayes_alignment_v1.json").read_text())
    shard_count = int(stage1b_cfg["execution"][f"{environment}_shards"])
    ref_a_parts, ref_a_hashes = [], []
    reference_manifest_path = root / "runs/diagnostics/stage1_bayes_alignment_v1/reference_manifest_frozen.json"
    reference_manifest_sha = base.sha256(reference_manifest_path)
    for shard in range(shard_count):
        path = root / "runs/diagnostics/stage1_bayes_alignment_v1/shards" / f"{environment}_{shard:02d}_of_{shard_count:02d}" / "alignment_vectors.npz"
        receipt_path = path.parent / "receipt.json"
        values = _load_npz(path)
        if "r_b_ref_a" not in values or values["r_b_ref_a"].ndim != 2:
            raise RuntimeError("Ref-A Stage1B vector is missing or malformed")
        frame = pd.DataFrame({name: values[name] for name in key_names})
        routing.verify_stage1b_shard_receipt(
            json.loads(receipt_path.read_text()), environment, shard, shard_count,
            base.sha256(path), reference_manifest_sha, len(frame),
        )
        frame["ref_position"] = np.arange(len(frame), dtype=np.int64)
        frame["ref_shard"] = shard
        ref_a_parts.append((frame, values["r_b_ref_a"].astype(np.float64)))
        ref_a_hashes.append(base.sha256(path))
    index = pd.concat([frame for frame, _ in ref_a_parts], ignore_index=True)
    joined = ordered_reference_join(manifest, row_index, index, key_names)
    if joined.ref_position.isna().any():
        raise RuntimeError("ref-A vectors do not cover every GRU formal candidate row")
    if np.any(joined.ref_shard.to_numpy(np.int64) < 0) or np.any(joined.ref_shard.to_numpy(np.int64) >= shard_count):
        raise RuntimeError("Ref-A shard positions are out of bounds")
    ref_a_correction = np.empty_like(mu_b[positions], dtype=np.float64)
    for output_position, row in enumerate(joined.itertuples(index=False)):
        if int(row.ref_position) < 0 or int(row.ref_position) >= len(ref_a_parts[int(row.ref_shard)][1]):
            raise RuntimeError("Ref-A within-shard position is out of bounds")
        ref_a_correction[output_position] = ref_a_parts[int(row.ref_shard)][1][int(row.ref_position)]
    mu_a = canonical_anchor[positions] + ref_a_correction
    if mu_a.shape != mu_b[positions].shape or not np.isfinite(mu_a).all() or not np.isfinite(mu_b[positions]).all():
        raise RuntimeError("physical reference means are non-finite or dimensionally inconsistent")
    provenance = {
        "formal_input_receipt_sha256": base.sha256(input_receipt_path),
        "correction_sha256": base.sha256(correction_path),
        "canonical_cache_sha256_used_only_to_recover_physical_mu": base.sha256(cache_path),
        "ref_a_stage1b_vector_hashes": ref_a_hashes,
    }
    return {"ref_b": mu_b[positions].astype(np.float32), "ref_a": mu_a.astype(np.float32)}, provenance


@torch.no_grad()
def _extract_positions(root: Path, cfg: dict, environment: str, base_seed: int,
                       split: str, arrays, manifest: pd.DataFrame, positions: np.ndarray,
                       candidate_column: str, device: torch.device) -> tuple[dict[str, np.ndarray], dict, dict]:
    model, norms, before = _model(root, cfg, environment, base_seed, arrays, device)
    candidates = routing._candidate_identity(manifest, candidate_column)
    names = (
        "row_index", "system_index", "realization", "anchor_index", "history_index", "query_index", "candidate_index", "candidate_identity",
        "z_p_anchor", "delta_segment", "delta_persistent", "delta_predicted_query",
        "query_embedding", "predicted_latent_full", "prediction_original",
        "prediction_anchor", "normalized_target",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    batch_size = int(cfg["adapter"]["batch_size"])
    for system in np.sort(np.unique(arrays.system_index[positions])):
        local = positions[arrays.system_index[positions] == system]
        for start in range(0, len(local), batch_size):
            idx = local[start:start + batch_size]
            subset = type(arrays)(**{
                field: np.asarray(getattr(arrays, field)[idx]) for field in arrays.__dataclass_fields__
            })
            history, mask, query, target = _normalized(subset, norms)
            anchor_mask = routing.masked_second_segment(mask)
            ht = torch.from_numpy(history).to(device)
            mt = torch.from_numpy(mask).to(device)
            mat = torch.from_numpy(anchor_mask).to(device)
            qt = torch.from_numpy(query).to(device)
            encoded = model.segment_encoder(ht) * mt[:, :, None]
            z_full = model.aggregate_encoded(encoded, mt)
            z_anchor = model.persistent(ht, mat)
            q_embed = model.query_encoder(qt)
            predicted_full = model.latent_predictor(torch.cat((z_full, q_embed), dim=1))
            predicted_anchor = model.latent_predictor(torch.cat((z_anchor, q_embed), dim=1))
            values = {
                "row_index": idx.astype(np.int64),
                "system_index": arrays.system_index[idx].astype(np.int64),
                "realization": (
                    manifest["realization"].to_numpy(np.int64)[idx]
                    if "realization" in manifest else np.zeros(len(idx), dtype=np.int64)
                ),
                "anchor_index": arrays.anchor_index[idx].astype(np.int64),
                "history_index": arrays.anchor_index[idx].astype(np.int64),
                "query_index": arrays.query_index[idx].astype(np.int64),
                "candidate_index": (
                    manifest["candidate_index"].to_numpy(np.int64)[idx]
                    if "candidate_index" in manifest
                    else pd.Categorical(candidates).codes.astype(np.int64)[idx]
                ),
                "candidate_identity": candidates[idx],
                "z_p_anchor": z_anchor.cpu().numpy().astype(np.float32),
                "delta_segment": encoded[:, 1].cpu().numpy().astype(np.float32),
                "delta_persistent": (z_full - z_anchor).cpu().numpy().astype(np.float32),
                "delta_predicted_query": (predicted_full - predicted_anchor).cpu().numpy().astype(np.float32),
                "query_embedding": q_embed.cpu().numpy().astype(np.float32),
                "predicted_latent_full": predicted_full.cpu().numpy().astype(np.float32),
                "prediction_original": model.target_decoder(predicted_full).cpu().numpy().astype(np.float32),
                "prediction_anchor": model.target_decoder(predicted_anchor).cpu().numpy().astype(np.float32),
                "normalized_target": target.astype(np.float32),
            }
            for name, value in values.items():
                parts[name].append(value)
    result = {name: np.concatenate(values) for name, values in parts.items()}
    order = np.argsort(result["row_index"], kind="stable")
    result = {name: values[order] for name, values in result.items()}
    reference = {}
    if split == "formal":
        means, reference = _physical_reference_means(root, environment, result["row_index"])
        result["mu_ref_b"] = means["ref_b"]
        result["mu_ref_a"] = means["ref_a"]
        result["bayes_correction_gru"] = (means["ref_b"] - result["prediction_anchor"]).astype(np.float32)
        result["bayes_correction_gru_ref_a"] = (means["ref_a"] - result["prediction_anchor"]).astype(np.float32)
    after = routing.original_module_hashes(model)
    if before != after:
        raise RuntimeError("masked-GRU base model changed during cache extraction")
    return result, before, reference


def freeze_formal_implementation(root: Path, config_path: Path, authorize_reviewed_code: bool) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    if not authorize_reviewed_code:
        raise RuntimeError("formal implementation freeze requires --authorize-reviewed-code")
    cfg = base.load_cfg(config_path)
    host = base._require_remote(root, cfg)
    base_barrier = _verify_base_barrier(root, config_path, cfg)
    target = _output(root, cfg) / "FORMAL_IMPLEMENTATION_FROZEN.json"
    if target.exists():
        raise RuntimeError("masked-GRU formal implementation is already frozen")
    sources = _direct_dependency_paths(root, config_path, cfg)
    versions = _runtime_versions()
    receipt = {
        "schema_version": "1.0", "status": STATUS_PLAN, "created_at_unix": time.time(),
        "host": host, "source_hashes": {str(path.relative_to(root)): base.sha256(path) for path in sources},
        "runtime_versions": versions,
        "base_barrier_sha256": base.sha256(root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"),
        "base_receipt_count": len(base_barrier["receipts"]),
        "adapter_jobs": [{"base_seed": s, "arm": a, "seed": o} for s, a, o in expected_adapter_jobs(cfg)],
        "formal_outcomes_read": False, "one_shot_authorization_created": False,
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_formal_plan(root: Path, config_path: Path, cfg: dict) -> dict:
    path = _output(root, cfg) / "FORMAL_IMPLEMENTATION_FROZEN.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_PLAN or receipt.get("formal_outcomes_read") is not False:
        raise RuntimeError("masked-GRU formal implementation freeze is invalid")
    if receipt.get("base_barrier_sha256") != base.sha256(root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"):
        raise RuntimeError("masked-GRU formal implementation has a stale base barrier")
    for relative, expected in receipt["source_hashes"].items():
        if base.sha256(root / relative) != expected:
            raise RuntimeError(f"masked-GRU formal source changed: {relative}")
    if receipt.get("runtime_versions") != _runtime_versions():
        raise RuntimeError("masked-GRU package/CUDA runtime changed after implementation freeze")
    _verify_base_barrier(root, config_path, cfg)
    return receipt


def build_cache_shard(root: Path, config_path: Path, environment: str, base_seed: int,
                      split: str, shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = base.load_cfg(config_path)
    _verify_formal_plan(root, config_path, cfg)
    authorization = None
    if split == "formal":
        authorization = _verify_one_shot_authorization(root, config_path, cfg)
    base._require_remote(root, cfg, device_name=device_name)
    if shard_index not in range(CACHE_SHARDS):
        raise ValueError("masked-GRU caches use exactly two system shards")
    arrays_path, manifest_path, candidate_column = _split_inputs(root, environment, split)
    arrays, manifest = load_arrays(arrays_path), pd.read_csv(manifest_path)
    routing._validate_manifest(arrays, routing.canonicalize_formal_manifest(manifest, environment) if split == "formal" else manifest)
    if split == "formal":
        manifest = routing.canonicalize_formal_manifest(manifest, environment)
    positions = _active_positions(arrays, environment, split, shard_index, CACHE_SHARDS, root)
    cache, before, reference = _extract_positions(
        root, cfg, environment, base_seed, split, arrays, manifest, positions,
        candidate_column, torch.device(device_name),
    )
    out = _cache_root(root, cfg, environment, base_seed, split)
    cache_path = out / f"shard_{shard_index:02d}_of_{CACHE_SHARDS:02d}.npz"
    receipt_path = out / f"shard_{shard_index:02d}_of_{CACHE_SHARDS:02d}.receipt.json"
    if cache_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable masked-GRU cache shard already exists")
    _atomic_npz(cache_path, **cache)
    receipt = {
        "schema_version": "1.0", "status": STATUS_CACHE_SHARD,
        "environment": environment, "base_seed": int(base_seed), "split": split,
        "shard_index": int(shard_index), "shard_count": CACHE_SHARDS,
        "systems": int(len(np.unique(cache["system_index"]))), "rows": int(len(cache["row_index"])),
        "cache": str(cache_path.relative_to(root)), "cache_sha256": base.sha256(cache_path),
        "arrays_sha256": base.sha256(arrays_path), "manifest_sha256": base.sha256(manifest_path),
        "original_module_hashes": before, "reference_provenance": reference,
        "formal_outcomes_read": split == "formal",
        "base_authorization_sha256": (
            authorization["base_authorizations"][f"{environment}:{int(base_seed)}"]["sha256"]
            if authorization is not None else None
        ),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def merge_cache(root: Path, config_path: Path, environment: str, base_seed: int, split: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_formal_plan(root, config_path, cfg); base._require_remote(root, cfg)
    if split == "formal":
        _verify_one_shot_authorization(root, config_path, cfg)
    arrays_path, _, _ = _split_inputs(root, environment, split)
    arrays = load_arrays(arrays_path)
    parts, receipts = [], []
    out = _cache_root(root, cfg, environment, base_seed, split)
    for shard in range(CACHE_SHARDS):
        receipt_path = out / f"shard_{shard:02d}_of_{CACHE_SHARDS:02d}.receipt.json"
        receipt = json.loads(receipt_path.read_text()); path = root / receipt["cache"]
        if (
            receipt.get("status") != STATUS_CACHE_SHARD
            or receipt.get("environment") != environment
            or int(receipt.get("base_seed")) != int(base_seed)
            or receipt.get("split") != split
            or int(receipt.get("shard_index")) != shard
            or int(receipt.get("shard_count")) != CACHE_SHARDS
            or receipt.get("cache_sha256") != base.sha256(path)
        ):
            raise RuntimeError("invalid masked-GRU cache shard")
        part = _load_npz(path)
        if np.any(part["system_index"] % CACHE_SHARDS != shard):
            raise RuntimeError("masked-GRU cache shard violates system ownership")
        parts.append(part); receipts.append(receipt)
    merged = merge_cache_parts(parts)
    expected = _active_positions(arrays, environment, split, 0, 1, root)
    if (
        len(merged["row_index"]) != len(expected)
        or set(map(int, merged["row_index"])) != set(map(int, expected))
        or len(np.unique(merged["row_index"])) != len(expected)
    ):
        raise RuntimeError("masked-GRU merged cache has duplicate/missing rows")
    if any(receipt["original_module_hashes"] != receipts[0]["original_module_hashes"] for receipt in receipts):
        raise RuntimeError("masked-GRU module hashes differ across shards")
    if split == "formal":
        authorization = _verify_one_shot_authorization(root, config_path, cfg)
        expected_authorization = authorization["base_authorizations"][f"{environment}:{int(base_seed)}"]["sha256"]
        if any(receipt.get("base_authorization_sha256") != expected_authorization for receipt in receipts):
            raise RuntimeError("formal cache shard is not bound to its base authorization")
    path, receipt_path = out / "merged.npz", out / "merged.receipt.json"
    if path.exists() or receipt_path.exists():
        raise RuntimeError("immutable masked-GRU merged cache already exists")
    _atomic_npz(path, **merged)
    receipt = {
        "schema_version": "1.0", "status": STATUS_CACHE,
        "environment": environment, "base_seed": int(base_seed), "split": split,
        "shards": CACHE_SHARDS, "fixed_merge_order": [0, 1],
        "rows": int(len(expected)), "systems": int(len(np.unique(merged["system_index"]))),
        "cache": str(path.relative_to(root)), "cache_sha256": base.sha256(path),
        "canonical_content_sha256": canonical_cache_content_sha(merged),
        "shard_hashes": [row["cache_sha256"] for row in receipts],
        "original_module_hashes": receipts[0]["original_module_hashes"],
        "base_authorization_sha256": receipts[0].get("base_authorization_sha256"),
        "formal_outcomes_read": split == "formal",
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _merged_cache(root: Path, cfg: dict, environment: str, base_seed: int,
                  split: str) -> tuple[dict[str, np.ndarray], dict]:
    out = _cache_root(root, cfg, environment, base_seed, split)
    receipt = json.loads((out / "merged.receipt.json").read_text())
    path = root / receipt["cache"]
    if receipt.get("status") != STATUS_CACHE or receipt.get("cache_sha256") != base.sha256(path):
        raise RuntimeError("masked-GRU merged cache is invalid")
    cache = _load_npz(path)
    if receipt.get("canonical_content_sha256") != canonical_cache_content_sha(cache):
        raise RuntimeError("masked-GRU merged cache semantic content is invalid")
    return cache, receipt


def _configure_adapter(seed: int, device: torch.device, cfg: dict) -> None:
    base._configure_training(int(seed), device, int(cfg["execution"]["threads_per_worker"]))


def _adapter_input(cache: dict[str, np.ndarray], arm: str, seed: int) -> tuple[np.ndarray, str | None]:
    if arm == "true":
        return cache["delta_persistent"], None
    if arm == "shuffled":
        donor = routing.cross_system_cell_permutation(cache, int(seed))
        digest = __import__("hashlib").sha256(donor.astype("<i8").tobytes()).hexdigest()
        return cache["delta_persistent"][donor], digest
    raise ValueError(arm)


def train_adapter(root: Path, config_path: Path, base_seed: int, arm: str,
                  seed: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_formal_plan(root, config_path, cfg); base._require_remote(root, cfg, device_name=device_name)
    if (int(base_seed), arm, int(seed)) not in set(expected_adapter_jobs(cfg)):
        raise RuntimeError("adapter job is outside the frozen 12-fit schedule")
    train, train_receipt = _merged_cache(root, cfg, "articulated", base_seed, "train")
    select, select_receipt = _merged_cache(root, cfg, "articulated", base_seed, "select")
    if "bayes_correction_gru" in train or "bayes_correction_gru" in select:
        raise RuntimeError("train/select adapter cache contains forbidden formal correction")
    arrays = load_arrays(root / cfg["environments"]["articulated"]["train_arrays"])
    model, _, before = _model(root, cfg, "articulated", base_seed, arrays, torch.device(device_name))
    train_delta, train_shuffle_sha = _adapter_input(train, arm, seed)
    select_delta, select_shuffle_sha = _adapter_input(select, arm, seed)
    device = torch.device(device_name); _configure_adapter(seed, device, cfg)
    adapter = e3.DeltaGatedAdapter(64, 64).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(cfg["adapter"]["learning_rate"]),
        weight_decay=float(cfg["adapter"]["weight_decay"]),
    )
    best_loss, best_epoch, best_state, bad, curve = float("inf"), 0, None, 0, []
    batch_size = int(cfg["adapter"]["batch_size"])
    for epoch in range(1, int(cfg["adapter"]["maximum_epochs"]) + 1):
        adapter.train()
        order = np.random.default_rng(routing._stable_seed("delta-gated-order", arm, seed, epoch)).permutation(len(train_delta))
        total, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            delta = torch.from_numpy(train_delta[idx]).to(device)
            query = torch.from_numpy(train["query_embedding"][idx]).to(device)
            latent = torch.from_numpy(train["predicted_latent_full"][idx]).to(device)
            target = torch.from_numpy(train["normalized_target"][idx]).to(device)
            prediction = model.target_decoder(latent + adapter(delta, query))
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach().cpu()) * len(idx); count += len(idx)
        select_loss = e3._mse(adapter, model.target_decoder, select, select_delta, device, batch_size)
        curve.append({"epoch": epoch, "train_mse": total / count, "select_mse": select_loss})
        if np.isfinite(select_loss) and select_loss < best_loss:
            best_loss, best_epoch = float(select_loss), epoch
            best_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch >= int(cfg["adapter"]["minimum_epochs"]) and bad >= int(cfg["adapter"]["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("masked-GRU adapter produced no finite selected checkpoint")
    if routing.original_module_hashes(model) != before:
        raise RuntimeError("masked-GRU base model changed during adapter training")
    out = _adapter_root(root, cfg, base_seed, arm, seed)
    checkpoint, curve_path, receipt_path = out / "adapter_best.pt", out / "training_curve.csv", out / "receipt.json"
    if any(path.exists() for path in (checkpoint, curve_path, receipt_path)):
        raise RuntimeError("immutable masked-GRU adapter output already exists")
    _atomic_torch(checkpoint, best_state)
    _atomic_text(curve_path, pd.DataFrame(curve).to_csv(index=False))
    receipt = {
        "schema_version": "1.0", "status": STATUS_ADAPTER,
        "base_seed": int(base_seed), "arm": arm, "optimization_seed": int(seed),
        "learning_rate": float(cfg["adapter"]["learning_rate"]), "best_epoch": int(best_epoch),
        "best_select_mse": float(best_loss), "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": base.sha256(checkpoint), "curve_sha256": base.sha256(curve_path),
        "train_cache_sha256": train_receipt["cache_sha256"], "select_cache_sha256": select_receipt["cache_sha256"],
        "train_shuffle_sha256": train_shuffle_sha, "select_shuffle_sha256": select_shuffle_sha,
        "original_module_hashes_before": before,
        "original_module_hashes_after": routing.original_module_hashes(model),
        "formal_outcomes_read": False,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def authenticate_adapters(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_formal_plan(root, config_path, cfg); base._require_remote(root, cfg)
    rows, hashes = [], {}
    for base_seed, arm, seed in expected_adapter_jobs(cfg):
        path = _adapter_root(root, cfg, base_seed, arm, seed) / "receipt.json"
        receipt = json.loads(path.read_text()); checkpoint = root / receipt["checkpoint"]
        if (
            receipt.get("status") != STATUS_ADAPTER
            or receipt.get("formal_outcomes_read") is not False
            or receipt.get("checkpoint_sha256") != base.sha256(checkpoint)
            or float(receipt.get("learning_rate")) != float(cfg["adapter"]["learning_rate"])
        ):
            raise RuntimeError("masked-GRU adapter receipt is invalid")
        rows.append({"base_seed": base_seed, "arm": arm, "seed": seed,
                     "checkpoint": receipt["checkpoint"], "checkpoint_sha256": receipt["checkpoint_sha256"]})
        hashes[str(path.relative_to(root))] = base.sha256(path)
    table = pd.DataFrame(rows).sort_values(["base_seed", "arm", "seed"]).reset_index(drop=True)
    out = _output(root, cfg) / "selection"; table_path, receipt_path = out / "all_12_checkpoints.csv", out / "ALL_12_ADAPTERS_AUTHENTICATED.json"
    if table_path.exists() or receipt_path.exists():
        raise RuntimeError("masked-GRU adapter barrier already exists")
    _atomic_text(table_path, table.to_csv(index=False))
    receipt = {
        "schema_version": "1.0", "status": STATUS_ADAPTER_BARRIER, "jobs": 12,
        "table": str(table_path.relative_to(root)), "table_sha256": base.sha256(table_path),
        "job_receipt_hashes": hashes, "seed_selection": False, "formal_outcomes_read": False,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_adapter_barrier(root: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    path = _output(root, cfg) / "selection/ALL_12_ADAPTERS_AUTHENTICATED.json"
    receipt = json.loads(path.read_text()); table_path = root / receipt["table"]
    if receipt.get("status") != STATUS_ADAPTER_BARRIER or receipt.get("jobs") != 12 or receipt.get("table_sha256") != base.sha256(table_path):
        raise RuntimeError("masked-GRU adapter barrier is invalid")
    table = pd.read_csv(table_path)
    if set(zip(table.base_seed.astype(int), table.arm.astype(str), table.seed.astype(int))) != set(expected_adapter_jobs(cfg)):
        raise RuntimeError("masked-GRU adapter table has wrong schedule")
    for relative, expected in receipt.get("job_receipt_hashes", {}).items():
        if base.sha256(root / relative) != expected:
            raise RuntimeError("masked-GRU adapter job receipt changed")
    if len(receipt.get("job_receipt_hashes", {})) != 12:
        raise RuntimeError("masked-GRU adapter barrier does not bind all 12 job receipts")
    for row in table.itertuples(index=False):
        if base.sha256(root / row.checkpoint) != row.checkpoint_sha256:
            raise RuntimeError("masked-GRU adapter checkpoint changed")
    return table, receipt


def authorize_one_shot(root: Path, config_path: Path, authorize_reviewed_formal_code: bool) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    if not authorize_reviewed_formal_code:
        raise RuntimeError("one-shot authorization requires --authorize-reviewed-formal-code")
    _verify_formal_plan(root, config_path, cfg); host = base._require_remote(root, cfg)
    table, adapters = _verify_adapter_barrier(root, cfg)
    target = _output(root, cfg) / "authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"
    if target.exists():
        raise RuntimeError("one-shot formal authorization already exists")
    _assert_no_preexisting_formal_outputs(root, cfg)
    source_paths = [Path(__file__).resolve(), root / FORMAL_TEST, config_path, root / cfg["protocol"]]
    base_authorizations = {}
    authorized_environments = [
        environment for environment in cfg["environments"]
        if competent_base_seeds(root, cfg, environment)
    ]
    for environment in authorized_environments:
        for base_seed in competent_base_seeds(root, cfg, environment):
            checkpoint, competence = _verify_base(root, cfg, environment, int(base_seed))
            competence_path = _base_root(root, cfg, environment, int(base_seed)) / "BASE_COMPETENCE_RECEIPT.json"
            child_path = _output(root, cfg) / "authorization/bases" / f"{environment}_s{int(base_seed)}.json"
            child = {
                "schema_version": "1.0", "status": "MASKED_GRU_BASE_FORMAL_AUTHORIZED",
                "environment": environment, "base_seed": int(base_seed),
                "checkpoint_sha256": base.sha256(checkpoint),
                "competence_receipt_sha256": base.sha256(competence_path),
                "competence_status": competence["status"], "formal_outcomes_read": False,
            }
            serialized = json.dumps(child, indent=2, sort_keys=True) + "\n"
            if child_path.exists():
                if child_path.read_text() != serialized:
                    raise RuntimeError("pre-existing base formal authorization differs")
            else:
                _atomic_text(child_path, serialized)
            base_authorizations[f"{environment}:{int(base_seed)}"] = {
                "path": str(child_path.relative_to(root)), "sha256": base.sha256(child_path),
            }
    formal_inputs = {}
    for environment in authorized_environments:
        input_receipt = root / CANONICAL_ROUTING_ROOT / "formal_inputs" / environment / "FORMAL_INPUTS_RECEIPT.json"
        canonical_cache_receipt = root / CANONICAL_ROUTING_ROOT / "cache" / environment / "formal/merged.receipt.json"
        input_payload = json.loads(input_receipt.read_text())
        cache_payload = json.loads(canonical_cache_receipt.read_text())
        artifact_bindings = {}
        for logical, path_key, hash_key in (
            ("arrays", "arrays", "arrays_sha256"),
            ("manifest", "manifest", "manifest_sha256"),
            ("correction", "correction", "correction_sha256"),
        ):
            path = root / input_payload[path_key]
            if base.sha256(path) != input_payload[hash_key]:
                raise RuntimeError(f"formal {environment} {logical} changed before authorization")
            artifact_bindings[logical] = {"path": str(path.relative_to(root)), "sha256": base.sha256(path)}
        cache_path = root / cache_payload["merged_cache"]
        if base.sha256(cache_path) != cache_payload["merged_sha256"]:
            raise RuntimeError("canonical cache changed before physical-mu authorization")
        formal_inputs[environment] = {
            "formal_input_receipt": str(input_receipt.relative_to(root)),
            "formal_input_receipt_sha256": base.sha256(input_receipt),
            "canonical_cache_receipt": str(canonical_cache_receipt.relative_to(root)),
            "canonical_cache_receipt_sha256": base.sha256(canonical_cache_receipt),
            "artifacts": artifact_bindings,
            "canonical_cache_used_only_for_physical_mu": {
                "path": str(cache_path.relative_to(root)), "sha256": base.sha256(cache_path),
            },
            "ref_a_stage1b": _ref_a_bindings(root, environment),
        }
    payload = {
        "schema_version": "1.0", "status": STATUS_AUTHORIZATION,
        "created_at_unix": time.time(), "host": host,
        "source_hashes": {str(path.relative_to(root)): base.sha256(path) for path in source_paths},
        "formal_implementation_freeze_sha256": base.sha256(_output(root, cfg) / "FORMAL_IMPLEMENTATION_FROZEN.json"),
        "base_barrier_sha256": base.sha256(root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"),
        "adapter_barrier_sha256": base.sha256(_output(root, cfg) / "selection/ALL_12_ADAPTERS_AUTHENTICATED.json"),
        "adapter_table_sha256": adapters["table_sha256"], "adapter_rows": int(len(table)),
        "base_authorizations": base_authorizations,
        "formal_inputs": formal_inputs, "authorized_consumers": [
            "gru_stage1a_stage1b", "query_relevant_delta_persistent_decodability",
            "articulated_delta_gated_formal",
        ] + (["coupled_transport_descriptive"] if "coupled" in authorized_environments else []),
        "formal_outcomes_read": False, "single_authorization_only": True,
    }
    _atomic_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _verify_one_shot_authorization(root: Path, config_path: Path, cfg: dict) -> dict:
    # Every outcome-bearing consumer revalidates the implementation freeze first,
    # including all direct dependency and runtime bindings.
    _verify_formal_plan(root, config_path, cfg)
    path = _output(root, cfg) / "authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_AUTHORIZATION or receipt.get("single_authorization_only") is not True:
        raise RuntimeError("masked-GRU formal authorization is invalid")
    if receipt.get("formal_implementation_freeze_sha256") != base.sha256(_output(root, cfg) / "FORMAL_IMPLEMENTATION_FROZEN.json"):
        raise RuntimeError("masked-GRU one-shot authorization is stale")
    if receipt.get("adapter_barrier_sha256") != base.sha256(_output(root, cfg) / "selection/ALL_12_ADAPTERS_AUTHENTICATED.json"):
        raise RuntimeError("masked-GRU adapter authorization binding is stale")
    if receipt.get("base_barrier_sha256") != base.sha256(root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"):
        raise RuntimeError("masked-GRU base authorization binding is stale")
    for relative, expected in receipt["source_hashes"].items():
        if base.sha256(root / relative) != expected:
            raise RuntimeError("authorized masked-GRU consumer source changed")
    for environment, item in receipt["formal_inputs"].items():
        for key in ("formal_input_receipt", "canonical_cache_receipt"):
            if base.sha256(root / item[key]) != item[f"{key}_sha256"]:
                raise RuntimeError(f"authorized {environment} formal provenance changed")
        for artifact in list(item["artifacts"].values()) + [item["canonical_cache_used_only_for_physical_mu"]]:
            if base.sha256(root / artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"authorized {environment} formal artifact changed")
        ref_a = item["ref_a_stage1b"]
        for key in ("config", "reference_manifest"):
            if base.sha256(root / ref_a[key]) != ref_a[f"{key}_sha256"]:
                raise RuntimeError(f"authorized {environment} Ref-A {key} changed")
        for shard in ref_a["shards"]:
            if base.sha256(root / shard["vector"]) != shard["vector_sha256"] or base.sha256(root / shard["receipt"]) != shard["receipt_sha256"]:
                raise RuntimeError(f"authorized {environment} Ref-A shard changed")
    expected_bases = {
        f"{environment}:{int(seed)}"
        for environment in cfg["environments"] for seed in competent_base_seeds(root, cfg, environment)
    }
    if set(receipt.get("base_authorizations", {})) != expected_bases:
        raise RuntimeError("one-shot authorization does not bind all four base learners")
    for item in receipt["base_authorizations"].values():
        if base.sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError("per-base outcome authorization changed")
    return receipt


def classify_cross_architecture_replication(successes: list[bool]) -> str:
    """Pure final classifier; scientific success flags are per base learner."""
    if not successes:
        raise ValueError("at least one competent Articulated base learner is required")
    if all(successes):
        return "MASKED_GRU_CROSS_ARCHITECTURE_REPLICATION"
    if any(successes):
        return "MASKED_GRU_MIXED_REPLICATION"
    return "MASKED_GRU_COMPETENT_ARCHITECTURE_CONTRADICTION"


def analyze_stage1(root: Path, config_path: Path, environment: str, base_seed: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_one_shot_authorization(root, config_path, cfg); base._require_remote(root, cfg)
    cache, receipt = _merged_cache(root, cfg, environment, base_seed, "formal")
    delta = cache["prediction_original"].astype(np.float64) - cache["prediction_anchor"].astype(np.float64)
    correction = cache["bayes_correction_gru"].astype(np.float64)
    metrics = alignment_metrics(delta, correction)
    verification = alignment_metrics(delta, cache["bayes_correction_gru_ref_a"].astype(np.float64))
    table = pd.DataFrame({
        "row_index": cache["row_index"], "system_index": cache["system_index"],
        "anchor_index": cache["anchor_index"], "query_index": cache["query_index"],
        "candidate_identity": cache["candidate_identity"], **metrics,
    })
    table["cos_theta_ref_a"] = verification["cos_theta"]
    table["rho_ref_a"] = verification["rho"]
    table["a_ref_a"] = verification["a"]
    table["b_ref_a"] = verification["b"]
    table["delta_l_norm_ref_a"] = verification["delta_l_norm"]
    table["r_b_norm_ref_a"] = verification["r_b_norm"]
    table["v_l_conditional_ref_a"] = verification["v_l_conditional"]
    for name, vector in (
        ("segment_norm", cache["delta_segment"]),
        ("persistent_norm", cache["delta_persistent"]),
        ("predicted_norm", cache["delta_predicted_query"]),
        ("prediction_norm", delta),
    ):
        table[name] = np.linalg.norm(vector.astype(np.float64), axis=1)
    intervals = {}
    summary_metrics = stage1_summary_metrics()
    for offset, metric in enumerate(summary_metrics):
        values = table.groupby("system_index", sort=True)[metric].mean().dropna().to_numpy(np.float64)
        intervals[metric] = base._bootstrap(values, int(cfg["formal"]["bootstrap_replicates"]), int(cfg["formal"]["bootstrap_seed"]) + int(base_seed) + offset)
    out = _output(root, cfg) / "stage1" / f"{environment}_s{int(base_seed)}"
    rows_path, summary_path = out / "alignment_rows.csv.gz", out / "summary.json"
    if rows_path.exists() or summary_path.exists():
        raise RuntimeError("immutable masked-GRU Stage1 output already exists")
    out.mkdir(parents=True, exist_ok=True)
    temporary = rows_path.with_name(rows_path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip"); os.replace(temporary, rows_path)
    result = {
        "schema_version": "1.0", "status": STATUS_STAGE1,
        "environment": environment, "base_seed": int(base_seed),
        "systems": int(table.system_index.nunique()), "rows": int(len(table)),
        "intervals": intervals,
        "reported_stage1_metrics": list(summary_metrics),
        "positive_system_mean_cosine_fraction": float((table.groupby("system_index").cos_theta.mean() > 0).mean()),
        "r_b_definition": "frozen_mu_ref_b_minus_masked_gru_anchor_prediction",
        "canonical_learner_r_b_reused": False, "formal_cache_sha256": receipt["cache_sha256"],
        "rows_sha256": base.sha256(rows_path),
    }
    _atomic_text(summary_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _crossfit_probe_predictions(rows: pd.DataFrame, target: np.ndarray,
                                designs: dict[str, probe_models.DesignMatrix]) -> tuple[dict[str, np.ndarray], dict]:
    """Five-system-fold, query-family OOF predictions with the frozen ridge grid."""
    table = probe_models.validate_rows(rows)
    target = np.asarray(target, dtype=np.float64)
    folds = probe_models.system_hash_folds(table.system_index.to_numpy(), 79301, 5)
    lambdas = tuple(map(float, probe_models.DEFAULT_LAMBDAS))
    predictions = {name: np.full_like(target, np.nan, dtype=np.float64) for name in designs}
    selected: dict[str, dict[str, float]] = {name: {} for name in designs}
    for query in sorted(table.query_index.unique()):
        qpos = np.flatnonzero(table.query_index.to_numpy() == int(query))
        for name, design in designs.items():
            risks = {}
            for alpha in lambdas:
                errors, systems = [], []
                for held_out in range(5):
                    train = qpos[folds[qpos] != held_out]; valid = qpos[folds[qpos] == held_out]
                    if not len(train) or not len(valid):
                        raise RuntimeError("masked-GRU decodability has an empty system fold")
                    mean, scale = probe_models._fit_scaler(design, train)
                    transformed = probe_models._transform(design, mean, scale)
                    reg = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(transformed[train], target[train])
                    pred = reg.predict(transformed[valid])
                    if pred.ndim == 1: pred = pred[:, None]
                    errors.append(np.mean((target[valid] - pred) ** 2, axis=1)); systems.append(table.system_index.to_numpy()[valid])
                risks[alpha] = probe_models._system_equal_risk(np.concatenate(errors), np.concatenate(systems))
            best = min(lambdas, key=lambda alpha: (risks[alpha], -alpha))
            selected[name][str(int(query))] = float(best)
            for held_out in range(5):
                train = qpos[folds[qpos] != held_out]; valid = qpos[folds[qpos] == held_out]
                mean, scale = probe_models._fit_scaler(design, train)
                transformed = probe_models._transform(design, mean, scale)
                reg = Ridge(alpha=best, fit_intercept=True, solver="svd").fit(transformed[train], target[train])
                pred = reg.predict(transformed[valid])
                predictions[name][valid] = pred if pred.ndim == 2 else pred[:, None]
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("masked-GRU decodability left non-finite OOF predictions")
    return predictions, {"fold_salt": 79301, "folds": 5, "lambda_grid": list(lambdas), "selected_lambda": selected}


def analyze_decodability(root: Path, config_path: Path, environment: str, base_seed: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_one_shot_authorization(root, config_path, cfg); base._require_remote(root, cfg)
    cache, receipt = _merged_cache(root, cfg, environment, base_seed, "formal")
    rows = pd.DataFrame({
        "system_index": cache["system_index"].astype(np.int64),
        "realization": cache["realization"].astype(np.int64),
        "history_index": cache["history_index"].astype(np.int64),
        "query_index": cache["query_index"].astype(np.int64),
        "candidate_index": cache["candidate_index"].astype(np.int64),
    })
    donor = probe_models.coherent_cell_derangement(rows, 79303)
    designs = probe_models.probe_designs(
        rows, cache["z_p_anchor"], cache["delta_persistent"], cache["delta_persistent"][donor],
    )
    references, model_details = {}, {}
    for offset, (stream, key) in enumerate((
        ("ref_b", "bayes_correction_gru"), ("ref_a", "bayes_correction_gru_ref_a")
    )):
        predictions, details = _crossfit_probe_predictions(rows, cache[key], designs)
        references[stream] = probe_models.evaluate_probe_predictions(
            rows, cache[key], predictions, 4000, 79501 + offset,
        )
        model_details[stream] = details
    out = _output(root, cfg) / "decodability" / f"{environment}_s{int(base_seed)}"
    donor_path, result_path = out / "coherent_shuffle_map.npz", out / "result.json"
    if donor_path.exists() or result_path.exists():
        raise RuntimeError("immutable masked-GRU decodability result already exists")
    _atomic_npz(donor_path, row_index=cache["row_index"].astype(np.int64), donor_position=donor.astype(np.int64))
    result = {
        "schema_version": "1.0", "status": "MASKED_GRU_QUERY_RELEVANT_DECODABILITY_COMPLETE",
        "environment": environment, "base_seed": int(base_seed), "systems": int(rows.system_index.nunique()),
        "designs": ["family", "anchor", "shuffle", "true"],
        "risk_estimand": "system_equal_five_fold_out_of_fold_prediction",
        "references": references, "model_details": model_details,
        "shuffle_salt": 79303, "shuffle_map": str(donor_path.relative_to(root)),
        "shuffle_map_sha256": base.sha256(donor_path), "formal_cache_sha256": receipt["cache_sha256"],
    }
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _formal_adapter_table(root: Path, cfg: dict, base_seed: int) -> pd.DataFrame:
    table, _ = _verify_adapter_barrier(root, cfg)
    result = table[table.base_seed.astype(int) == int(base_seed)].copy()
    if len(result) != 6:
        raise RuntimeError("masked-GRU base seed does not have exactly six adapters")
    return result.sort_values(["arm", "seed"]).reset_index(drop=True)


@torch.no_grad()
def _evaluate_cache_positions(root: Path, cfg: dict, base_seed: int,
                              cache: dict[str, np.ndarray], positions: np.ndarray,
                              device: torch.device) -> tuple[pd.DataFrame, dict]:
    positions = np.asarray(positions, dtype=np.int64)
    arrays = load_arrays(root / cfg["environments"]["articulated"]["train_arrays"])
    model, _, before = _model(root, cfg, "articulated", base_seed, arrays, device)
    local = {name: values[positions] for name, values in cache.items()}
    rows: list[dict] = []

    def append(arm: str, seed: int, prediction: np.ndarray) -> None:
        metrics = routing._row_metrics(
            prediction, local["prediction_anchor"], local["normalized_target"],
            local["bayes_correction_gru"],
        )
        for index in range(len(positions)):
            row = {"row_index": int(local["row_index"][index]), "system_index": int(local["system_index"][index]),
                   "arm": arm, "seed": int(seed)}
            row.update({name: float(values[index]) for name, values in metrics.items()}); rows.append(row)

    append("original", -1, local["prediction_original"])
    shuffle_hashes = {}
    for choice in _formal_adapter_table(root, cfg, base_seed).itertuples(index=False):
        adapter = e3.DeltaGatedAdapter(64, 64).to(device)
        adapter.load_state_dict(torch.load(root / choice.checkpoint, map_location=device, weights_only=True)); adapter.eval()
        if choice.arm == "true":
            delta = local["delta_persistent"]
        else:
            donor = routing.cross_system_cell_permutation(cache, int(choice.seed))
            delta = cache["delta_persistent"][donor[positions]]
            shuffle_hashes[str(int(choice.seed))] = __import__("hashlib").sha256(donor.astype("<i8").tobytes()).hexdigest()
        prediction = np.empty_like(local["prediction_original"], dtype=np.float32)
        batch_size = int(cfg["adapter"]["batch_size"])
        for system in np.sort(np.unique(local["system_index"])):
            system_positions = np.flatnonzero(local["system_index"] == system)
            for start in range(0, len(system_positions), batch_size):
                idx = system_positions[start:start + batch_size]
                d = torch.from_numpy(delta[idx]).to(device)
                q = torch.from_numpy(local["query_embedding"][idx]).to(device)
                z = torch.from_numpy(local["predicted_latent_full"][idx]).to(device)
                prediction[idx] = model.target_decoder(z + adapter(d, q)).cpu().numpy()
        append(str(choice.arm), int(choice.seed), prediction)
    if routing.original_module_hashes(model) != before:
        raise RuntimeError("masked-GRU base model changed during formal adapter evaluation")
    frame = pd.DataFrame(rows).sort_values(["system_index", "row_index", "arm", "seed"]).reset_index(drop=True)
    return routing._aggregate_formal_rows(frame), shuffle_hashes


def formal_parity(root: Path, config_path: Path, base_seed: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    authorization = _verify_one_shot_authorization(root, config_path, cfg)
    base._require_remote(root, cfg, device_name=device_name)
    cache, receipt = _merged_cache(root, cfg, "articulated", base_seed, "formal")
    systems = []
    available = np.sort(np.unique(cache["system_index"]))
    for shard in range(FORMAL_SHARDS):
        systems.append(int(available[available % FORMAL_SHARDS == shard][0]))
    positions = np.flatnonzero(np.isin(cache["system_index"], systems))
    device = torch.device(device_name)
    direct, hashes = _evaluate_cache_positions(root, cfg, base_seed, cache, positions, device)
    pieces = []
    for shard in range(FORMAL_SHARDS):
        subset = positions[cache["system_index"][positions] % FORMAL_SHARDS == shard]
        part, part_hashes = _evaluate_cache_positions(root, cfg, base_seed, cache, subset, device)
        if part_hashes != hashes:
            raise RuntimeError("masked-GRU parity shuffle maps differ")
        pieces.append(part)
    merged = pd.concat(pieces, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(direct, merged, check_exact=True)
    direct_serialized_sha = canonical_frame_sha(direct)
    merged_serialized_sha = canonical_frame_sha(merged)
    if direct_serialized_sha != merged_serialized_sha:
        raise RuntimeError("masked-GRU parity differs after canonical serialization")
    target = _formal_eval_root(root, cfg, base_seed) / "FORMAL_PARITY.json"
    if target.exists():
        raise RuntimeError("masked-GRU formal parity already exists")
    payload = {
        "schema_version": "1.0", "status": "MASKED_GRU_ONE_TASK_TWO_SHARD_PARITY_PASS",
        "base_seed": int(base_seed), "systems": systems, "fixed_merge_order": [0, 1],
        "statistics_dtype": "float64", "formal_cache_sha256": receipt["cache_sha256"],
        "authorization_sha256": base.sha256(_output(root, cfg) / "authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"),
        "shuffle_map_hashes": hashes,
        "direct_serialized_sha256": direct_serialized_sha,
        "merged_serialized_sha256": merged_serialized_sha,
    }
    _atomic_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_formal_shard(root: Path, config_path: Path, base_seed: int,
                          shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_one_shot_authorization(root, config_path, cfg); base._require_remote(root, cfg, device_name=device_name)
    if shard_index not in range(FORMAL_SHARDS):
        raise ValueError("masked-GRU formal evaluation uses exactly two shards")
    cache, receipt = _merged_cache(root, cfg, "articulated", base_seed, "formal")
    parity_path = _formal_eval_root(root, cfg, base_seed) / "FORMAL_PARITY.json"
    parity = json.loads(parity_path.read_text())
    if parity.get("status") != "MASKED_GRU_ONE_TASK_TWO_SHARD_PARITY_PASS" or parity.get("formal_cache_sha256") != receipt["cache_sha256"]:
        raise RuntimeError("masked-GRU formal parity is missing or stale")
    positions = np.flatnonzero(cache["system_index"] % FORMAL_SHARDS == shard_index)
    stats, hashes = _evaluate_cache_positions(root, cfg, base_seed, cache, positions, torch.device(device_name))
    if hashes != parity["shuffle_map_hashes"]:
        raise RuntimeError("masked-GRU formal shuffle maps differ from parity")
    out = _formal_eval_root(root, cfg, base_seed)
    path = out / f"stats_shard_{shard_index:02d}_of_{FORMAL_SHARDS:02d}.csv.gz"
    receipt_path = out / f"receipt_shard_{shard_index:02d}_of_{FORMAL_SHARDS:02d}.json"
    if path.exists() or receipt_path.exists():
        raise RuntimeError("immutable masked-GRU formal shard already exists")
    out.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    stats.to_csv(temporary, index=False, compression="gzip"); os.replace(temporary, path)
    payload = {
        "schema_version": "1.0", "status": STATUS_FORMAL_SHARD,
        "base_seed": int(base_seed), "shard_index": int(shard_index), "shard_count": FORMAL_SHARDS,
        "systems": int(stats.system_index.nunique()), "statistics": str(path.relative_to(root)),
        "statistics_sha256": base.sha256(path), "formal_cache_sha256": receipt["cache_sha256"],
        "parity_sha256": base.sha256(parity_path), "shuffle_map_hashes": hashes,
    }
    _atomic_text(receipt_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def merge_formal_evaluation(root: Path, config_path: Path, base_seed: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_one_shot_authorization(root, config_path, cfg); base._require_remote(root, cfg)
    out = _formal_eval_root(root, cfg, base_seed); parts = []
    for shard in range(FORMAL_SHARDS):
        receipt_path = out / f"receipt_shard_{shard:02d}_of_{FORMAL_SHARDS:02d}.json"
        receipt = json.loads(receipt_path.read_text()); path = root / receipt["statistics"]
        if receipt.get("status") != STATUS_FORMAL_SHARD or receipt.get("statistics_sha256") != base.sha256(path):
            raise RuntimeError("invalid masked-GRU formal shard")
        table = pd.read_csv(path)
        if np.any(table.system_index.to_numpy(np.int64) % FORMAL_SHARDS != shard):
            raise RuntimeError("masked-GRU formal shard violates ownership")
        parts.append(table)
    stats = merge_statistic_parts(parts)
    systems = sorted(map(int, stats.system_index.unique()))
    if len(systems) != 512:
        raise RuntimeError("masked-GRU Articulated formal merge must contain 512 systems")
    values = stats.assign(mean_gain=stats.sum_gain.astype(np.float64) / stats["count"].astype(np.float64))
    original = values[values.arm == "original"].set_index("system_index").mean_gain.loc[systems]
    true = values[values.arm == "true"].groupby("system_index", sort=True).mean_gain.mean().loc[systems]
    shuffled = values[values.arm == "shuffled"].groupby("system_index", sort=True).mean_gain.mean().loc[systems]
    vectors = {"true_minus_original": true.to_numpy() - original.to_numpy(),
               "true_minus_shuffled": true.to_numpy() - shuffled.to_numpy()}
    contrasts = {
        name: base._bootstrap(vector, int(cfg["formal"]["bootstrap_replicates"]), int(cfg["formal"]["bootstrap_seed"]) + offset)
        for offset, (name, vector) in enumerate(vectors.items())
    }
    per_seed = {}
    for seed in cfg["adapter"]["seeds"]:
        t = values[(values.arm == "true") & (values.seed == int(seed))].set_index("system_index").mean_gain.loc[systems]
        s = values[(values.arm == "shuffled") & (values.seed == int(seed))].set_index("system_index").mean_gain.loc[systems]
        per_seed[str(int(seed))] = {"true_minus_original": float((t - original).mean()),
                                   "true_minus_shuffled": float((t - s).mean())}
    route = contrasts["true_minus_original"]["ci_low"] > 0
    specificity = contrasts["true_minus_shuffled"]["ci_low"] > 0
    result = {
        "schema_version": "1.0", "status": "MASKED_GRU_PER_BASE_ROUTING_RESCUE" if route and specificity else "MASKED_GRU_PER_BASE_ROUTING_NOT_REPLICATED",
        "base_seed": int(base_seed), "systems": len(systems), "scientific_unit": "physical_system",
        "optimization_seeds_are_scientific_samples": False,
        "route_rescue": bool(route), "persistent_specificity_vs_shuffled": bool(specificity),
        "contrasts": contrasts, "per_optimization_seed_point_contrasts": per_seed,
        "fixed_merge_order": [0, 1], "canonical_statistics_content_sha256": canonical_frame_sha(stats),
    }
    stats_path, result_path = out / "merged_sufficient_statistics.csv.gz", out / "FINAL_RESULT.json"
    if stats_path.exists() or result_path.exists():
        raise RuntimeError("immutable masked-GRU formal final result already exists")
    temporary = stats_path.with_name(stats_path.name + f".tmp.{os.getpid()}")
    stats.to_csv(temporary, index=False, compression="gzip"); os.replace(temporary, stats_path)
    result["statistics"] = str(stats_path.relative_to(root)); result["statistics_sha256"] = base.sha256(stats_path)
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve(); cfg = base.load_cfg(config_path)
    _verify_one_shot_authorization(root, config_path, cfg); base._require_remote(root, cfg)
    results, stage1, decodability = {}, {}, {}
    for environment, item in cfg["environments"].items():
        stage1[environment], decodability[environment] = {}, {}
        for seed in competent_base_seeds(root, cfg, environment):
            path = _output(root, cfg) / "stage1" / f"{environment}_s{int(seed)}" / "summary.json"
            stage1[environment][str(int(seed))] = json.loads(path.read_text())
            decoding_path = _output(root, cfg) / "decodability" / f"{environment}_s{int(seed)}" / "result.json"
            decodability[environment][str(int(seed))] = json.loads(decoding_path.read_text())
    for seed in cfg["environments"]["articulated"]["base_seeds"]:
        path = _formal_eval_root(root, cfg, int(seed)) / "FINAL_RESULT.json"
        results[str(int(seed))] = json.loads(path.read_text())
    successes = [row["route_rescue"] and row["persistent_specificity_vs_shuffled"] for row in results.values()]
    status = classify_cross_architecture_replication(successes)
    payload = {
        "schema_version": "1.0", "status": status, "completed_at_unix": time.time(),
        "stage1": stage1, "query_relevant_decodability": decodability,
        "articulated_formal_results": results,
        "articulated_both_base_seeds_replicate": bool(all(successes)),
        "coupled_role": "transport_descriptive_only", "coupled_adapter_fits": 0,
        "coupled_competent_base_seeds": competent_base_seeds(root, cfg, "coupled"),
        "coupled_underidentified_base_seeds": sorted(set(map(int, cfg["environments"]["coupled"]["base_seeds"])) - set(competent_base_seeds(root, cfg, "coupled"))),
        "architecture_search_reopened": False, "formal_outcomes_used_for_selection": False,
    }
    target = _output(root, cfg) / "FINAL_RESULT.json"
    if target.exists():
        raise RuntimeError("immutable masked-GRU cross-architecture summary already exists")
    _atomic_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def self_test(root: Path, config_path: Path) -> dict:
    cfg = base.load_cfg(config_path.resolve())
    jobs = expected_adapter_jobs(cfg)
    if len(jobs) != 12 or len(set(jobs)) != 12:
        raise RuntimeError("masked-GRU formal adapter schedule is not the exact 12-fit product")
    cache = {
        "row_index": np.arange(16, dtype=np.int64),
        "system_index": np.repeat(np.arange(8), 2),
        "anchor_index": np.tile([0, 0], 8),
        "query_index": np.tile([1, 1], 8),
        "candidate_identity": np.tile(["c", "c"], 8),
        "delta_persistent": np.arange(16 * 4, dtype=np.float32).reshape(16, 4),
    }
    shuffled, digest = _adapter_input(cache, "shuffled", 86101)
    if shuffled.shape != cache["delta_persistent"].shape or digest is None:
        raise RuntimeError("masked-GRU coherent shuffle failed")
    delta = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    correction = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    metrics = alignment_metrics(delta, correction)
    if not np.allclose(metrics["a"], [1.0, 0.0]) or not np.allclose(metrics["b"], [0.0, 1.0]):
        raise RuntimeError("masked-GRU synthetic Stage1 alignment failed")
    return {"status": "MASKED_GRU_FORMAL_STATIC_PASS", "adapter_jobs": 12,
            "formal_shards": FORMAL_SHARDS, "cache_shards": CACHE_SHARDS,
            "formal_outcomes_read": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-formal-implementation"); freeze.add_argument("--authorize-reviewed-code", action="store_true")
    cache = sub.add_parser("build-cache-shard"); cache.add_argument("environment", choices=("articulated", "coupled")); cache.add_argument("base_seed", type=int); cache.add_argument("split", choices=("train", "select", "formal")); cache.add_argument("--shard-index", type=int, required=True); cache.add_argument("--device", required=True)
    merge = sub.add_parser("merge-cache"); merge.add_argument("environment", choices=("articulated", "coupled")); merge.add_argument("base_seed", type=int); merge.add_argument("split", choices=("train", "select", "formal"))
    train = sub.add_parser("train-adapter"); train.add_argument("base_seed", type=int); train.add_argument("arm", choices=("true", "shuffled")); train.add_argument("seed", type=int); train.add_argument("--device", required=True)
    sub.add_parser("authenticate-adapters")
    authorize = sub.add_parser("authorize-one-shot"); authorize.add_argument("--authorize-reviewed-formal-code", action="store_true")
    stage1 = sub.add_parser("analyze-stage1"); stage1.add_argument("environment", choices=("articulated", "coupled")); stage1.add_argument("base_seed", type=int)
    decoding = sub.add_parser("analyze-decodability"); decoding.add_argument("environment", choices=("articulated", "coupled")); decoding.add_argument("base_seed", type=int)
    parity = sub.add_parser("formal-parity"); parity.add_argument("base_seed", type=int); parity.add_argument("--device", required=True)
    evaluate = sub.add_parser("evaluate-formal-shard"); evaluate.add_argument("base_seed", type=int); evaluate.add_argument("--shard-index", type=int, required=True); evaluate.add_argument("--device", required=True)
    merge_eval = sub.add_parser("merge-formal-evaluation"); merge_eval.add_argument("base_seed", type=int)
    sub.add_parser("summarize"); sub.add_parser("self-test")
    return parser


def main() -> None:
    args = _parser().parse_args(); root, config = args.root, args.config
    if args.command == "freeze-formal-implementation": result = freeze_formal_implementation(root, config, args.authorize_reviewed_code)
    elif args.command == "build-cache-shard": result = build_cache_shard(root, config, args.environment, args.base_seed, args.split, args.shard_index, args.device)
    elif args.command == "merge-cache": result = merge_cache(root, config, args.environment, args.base_seed, args.split)
    elif args.command == "train-adapter": result = train_adapter(root, config, args.base_seed, args.arm, args.seed, args.device)
    elif args.command == "authenticate-adapters": result = authenticate_adapters(root, config)
    elif args.command == "authorize-one-shot": result = authorize_one_shot(root, config, args.authorize_reviewed_formal_code)
    elif args.command == "analyze-stage1": result = analyze_stage1(root, config, args.environment, args.base_seed)
    elif args.command == "analyze-decodability": result = analyze_decodability(root, config, args.environment, args.base_seed)
    elif args.command == "formal-parity": result = formal_parity(root, config, args.base_seed, args.device)
    elif args.command == "evaluate-formal-shard": result = evaluate_formal_shard(root, config, args.base_seed, args.shard_index, args.device)
    elif args.command == "merge-formal-evaluation": result = merge_formal_evaluation(root, config, args.base_seed)
    elif args.command == "summarize": result = summarize(root, config)
    else: result = self_test(root, config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
