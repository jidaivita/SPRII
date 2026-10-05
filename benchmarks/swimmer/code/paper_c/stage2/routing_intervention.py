"""Frozen-representation downstream-routing intervention for Paper C.

The module is deliberately split into freeze/cache/train/select/evaluate
commands.  In particular, training cannot see a formal cache and formal
evaluation refuses to run until immutable-select checkpoint selection has a
hash-bound receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import socket
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from paper_c.coupled_sled.formal_data import arrays_from_sample_manifest, load_arrays, save_arrays
from paper_c.coupled_sled.learner import PersistentJEPA, _normalized
from paper_c.stage1.symptom_localization import _load_coupled_model
from paper_c.stage1 import symptom_localization as stage1a
from paper_c.swimmer.lqa_prospective import _load_jepa


STATUS_FROZEN = "DOWNSTREAM_ROUTING_INTERVENTION_FROZEN"
STATUS_CACHE = "DOWNSTREAM_ROUTING_FEATURE_CACHE_COMPLETE"
STATUS_JOB = "DOWNSTREAM_ROUTING_TRAINING_JOB_COMPLETE"
STATUS_SELECTION = "DOWNSTREAM_ROUTING_SELECT_CHOICES_FROZEN"
STATUS_EVAL_SHARD = "DOWNSTREAM_ROUTING_FORMAL_EVAL_SHARD_COMPLETE"
ORIGINAL_MODULES = (
    "segment_encoder", "aggregator", "query_encoder", "target_encoder",
    "latent_predictor", "target_decoder",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value)
    os.replace(tmp, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def module_sha256(module: nn.Module) -> str:
    """Hash a module state independently of torch serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def original_module_hashes(model: PersistentJEPA) -> dict[str, str]:
    return {name: module_sha256(getattr(model, name)) for name in ORIGINAL_MODULES}


def freeze_original(model: PersistentJEPA) -> dict[str, str]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("an original model parameter remains trainable")
    return original_module_hashes(model)


class RoutingAdapter(nn.Module):
    """Query-conditioned residual route with an exactly inert initial state."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, delta_persistent: torch.Tensor, query_embedding: torch.Tensor) -> torch.Tensor:
        if delta_persistent.shape != query_embedding.shape:
            raise ValueError("delta-persistent/query-embedding shape mismatch")
        return self.net(torch.cat((delta_persistent, query_embedding), dim=1))


def adapter_dimensions(cfg: dict, environment: str) -> tuple[int, int, int]:
    """Resolve the same width-64 route in each frozen learner's latent space."""
    adapter_cfg = cfg["adapter"]
    if adapter_cfg.get("dimension_rule") != "input=2*latent_dim,output=latent_dim":
        raise RuntimeError("unknown frozen adapter dimension rule")
    latent_dims = adapter_cfg.get("latent_dim_by_environment", {})
    if environment not in latent_dims:
        raise RuntimeError(f"missing frozen adapter latent dimension for {environment}")
    latent_dim = int(latent_dims[environment])
    hidden_dim = int(adapter_cfg["hidden_dim"])
    if latent_dim < 1 or hidden_dim < 1:
        raise RuntimeError("adapter dimensions must be positive")
    return 2 * latent_dim, hidden_dim, latent_dim


def routing_adapter_from_config(cfg: dict, environment: str) -> RoutingAdapter:
    return RoutingAdapter(*adapter_dimensions(cfg, environment))


def validate_adapter_cache_dimensions(cache: dict[str, np.ndarray], cfg: dict, environment: str) -> None:
    _, _, latent_dim = adapter_dimensions(cfg, environment)
    for name in ("delta_persistent", "query_embedding", "predicted_latent_full"):
        value = cache[name]
        if value.ndim != 2 or value.shape[1] != latent_dim:
            raise RuntimeError(
                f"{environment} {name} dimension {value.shape} does not match frozen latent_dim={latent_dim}"
            )


def masked_second_segment(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[1] != 2:
        raise ValueError("expected a [rows,2] history mask")
    if not np.all(mask[:, 1] == 1):
        raise ValueError("same-row counterfactual requires an active second segment")
    result = mask.copy()
    result[:, 1] = 0
    if not np.array_equal(result[:, 0], mask[:, 0]):
        raise AssertionError("anchor mask changed")
    return result


def _config(root: Path, config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    if cfg["remote_execution"]["training_jobs"] != 54:
        raise RuntimeError("frozen job count is not 54")
    return cfg


def require_remote_execution(cfg: dict, root: Path, environment: str | None = None,
                             partition_index: int | None = None) -> dict:
    """Verify OS, actual hostname, host sentinel, and frozen allowlists."""
    remote = cfg["remote_execution"]
    operating_system = platform.system()
    if operating_system == "Darwin":
        raise RuntimeError("remote-only protocol: Darwin/Mac execution is forbidden")
    if remote.get("require_linux", False) and operating_system != "Linux":
        raise RuntimeError(f"remote-only protocol requires Linux, got {operating_system}")
    if os.environ.get("PAPER_C_REMOTE_EXECUTION") != "1":
        raise RuntimeError("remote-only protocol: PAPER_C_REMOTE_EXECUTION=1 is required")
    logical_host = os.environ.get("PAPER_C_REMOTE_HOST")
    if logical_host not in remote["hosts"]:
        raise RuntimeError("PAPER_C_REMOTE_HOST is not in the frozen host allowlist")
    actual_hostname = socket.gethostname()
    sentinel = root / remote["sentinel_relative_directory"] / f"{logical_host}.json"
    payload = json.loads(sentinel.read_text())
    if payload.get("logical_host") != logical_host or payload.get("actual_hostname") != actual_hostname:
        raise RuntimeError("remote host sentinel does not match actual hostname")
    if payload.get("operating_system") != "Linux":
        raise RuntimeError("remote sentinel is not Linux")
    if sorted(payload.get("allowed_gpu_ids", [])) != sorted(remote["allowed_gpu_ids_per_host"]):
        raise RuntimeError("remote sentinel GPU allowlist differs from frozen config")
    allowed_environment = remote["logical_host_environment"][logical_host]
    if environment is not None and environment != allowed_environment:
        raise RuntimeError(f"{logical_host} is not allowed to execute {environment}")
    if partition_index is not None and int(remote["host_partition"][logical_host]) != int(partition_index):
        raise RuntimeError("logical host and queue partition disagree")
    return {
        "logical_host": logical_host, "actual_hostname": actual_hostname,
        "operating_system": operating_system, "sentinel": str(sentinel),
        "sentinel_sha256": sha256(sentinel), "allowed_environment": allowed_environment,
    }


def _load_environment_model(root: Path, cfg: dict, environment: str, arrays, device: torch.device):
    if environment == "coupled":
        model, norms, _ = _load_coupled_model(root, arrays, device)
    elif environment == "articulated":
        reference = json.loads((root / "configs/articulated_lqa_prospective_v1.json").read_text())
        model, norms, _ = _load_jepa(root, reference, device)
    else:
        raise ValueError(environment)
    before = freeze_original(model)
    return model, norms, before


def configure_worker_resources(cfg: dict, device: torch.device) -> dict:
    threads = int(cfg["remote_execution"]["cpu_threads_per_worker"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    gpu_fraction = float(cfg["remote_execution"]["gpu_memory_fraction_per_worker"])
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "torch_threads": threads,
        "environment_threads": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "container_memory_limit_gib": int(cfg["remote_execution"]["container_memory_limit_gib"]),
        "maximum_workers_per_host": int(cfg["remote_execution"]["maximum_workers_per_host"]),
        "gpu_memory_fraction": gpu_fraction if device.type == "cuda" else None,
    }


def peak_resource_usage(device: torch.device) -> dict:
    usage = {"peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if device.type == "cuda":
        usage.update({
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
    return usage


def _active_positions(arrays, active_conditions: Iterable[int], shard_index: int, shard_count: int) -> np.ndarray:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    active = np.isin(arrays.condition, np.asarray(list(active_conditions), dtype=int))
    active &= arrays.history_mask[:, 1] == 1
    active &= arrays.system_index % shard_count == shard_index
    return np.flatnonzero(active)


def require_canonical_shard_count(cfg: dict, shard_count: int) -> None:
    expected = int(cfg["cache"]["system_shards_per_environment"])
    if int(shard_count) != expected:
        raise RuntimeError(
            f"canonical cache/evaluation requires shard_count={expected}; "
            "one-task parity must use the isolated remote correctness namespace"
        )


def host_specific_receipt_path(root: Path, cfg: dict, key: str, logical_host: str) -> Path:
    """Resolve a frozen host-qualified receipt template inside the repository."""
    template = cfg["correctness"][key]
    if "{logical_host}" not in template:
        raise RuntimeError(f"host-specific receipt template {key} lacks {{logical_host}}")
    relative = Path(template.format(logical_host=logical_host))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"host-specific receipt template {key} escapes repository")
    return root / relative


def _candidate_identity(manifest: pd.DataFrame, column: str) -> np.ndarray:
    if column not in manifest:
        raise ValueError(f"sample manifest lacks candidate column {column}")
    return manifest[column].astype(str).to_numpy(dtype="U")


def _validate_manifest(arrays, manifest: pd.DataFrame) -> None:
    if len(manifest) != len(arrays.history):
        raise RuntimeError("sample manifest and immutable arrays differ in length")
    required = {"system_index", "anchor_index", "query_index"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"sample manifest lacks {sorted(required - set(manifest.columns))}")
    for key in required:
        left = np.asarray(getattr(arrays, key), dtype=np.int64)
        right = manifest[key].to_numpy(np.int64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"sample manifest row order differs from arrays for {key}")
    if "sample_index" in manifest and not np.array_equal(manifest.sample_index.to_numpy(np.int64), np.arange(len(manifest))):
        raise RuntimeError("sample_index is not exact immutable row order")


def canonicalize_formal_manifest(manifest: pd.DataFrame, environment: str) -> pd.DataFrame:
    """Expose one canonical anchor axis without changing Stage-1 identity.

    Stage 1A's Articulated manifest calls the anchor bank coordinate
    ``history_index`` and intentionally has no duplicate ``anchor_index``
    column. Coupled records both names. The intervention cache schema uses
    ``anchor_index`` for both environments, so Articulated receives an exact
    alias and Coupled is required to prove the two columns already agree.
    """
    table = manifest.copy()
    if "history_index" not in table:
        raise RuntimeError("formal manifest lacks history_index")
    if environment == "articulated":
        if "anchor_index" in table and not np.array_equal(
            table.anchor_index.to_numpy(np.int64), table.history_index.to_numpy(np.int64)
        ):
            raise RuntimeError("Articulated anchor_index conflicts with history_index")
        table["anchor_index"] = table.history_index.to_numpy(np.int64)
    elif environment == "coupled":
        if "anchor_index" not in table:
            raise RuntimeError("Coupled formal manifest lacks anchor_index")
        if not np.array_equal(table.anchor_index.to_numpy(np.int64), table.history_index.to_numpy(np.int64)):
            raise RuntimeError("Coupled anchor_index/history_index mapping changed")
    else:
        raise ValueError(environment)
    required_keys = set(stage1a.KEYS)
    if not required_keys.issubset(table.columns):
        raise RuntimeError(f"formal manifest lacks Stage-1 keys {sorted(required_keys - set(table.columns))}")
    if table.duplicated(stage1a.KEYS).any():
        raise RuntimeError("canonical formal manifest duplicates Stage-1 row keys")
    return table


def verify_frozen_learner_receipt(root: Path, environment: str, item: dict) -> dict:
    """Authenticate checkpoint and normalization against canonical training."""
    receipt_path = root / item["training_receipt"]
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != item["training_receipt_status"]:
        raise RuntimeError(f"{environment} canonical training receipt has the wrong status")
    hashes = receipt.get(item["training_receipt_hash_field"], {})
    checkpoint = root / item["checkpoint"]
    normalization = root / item["normalization"]
    if hashes.get("jepa") != sha256(checkpoint):
        raise RuntimeError(f"{environment} checkpoint is not the canonically frozen learner")
    if hashes.get("normalization") != sha256(normalization):
        raise RuntimeError(f"{environment} normalization is not canonically frozen")
    return {
        "status": receipt["status"], "receipt_sha256": sha256(receipt_path),
        "checkpoint_sha256": sha256(checkpoint), "normalization_sha256": sha256(normalization),
    }


def verify_stage1b_shard_receipt(
    receipt: dict,
    environment: str,
    shard_index: int,
    shard_count: int,
    vector_sha256: str,
    reference_manifest_sha256: str,
    vector_rows: int,
) -> None:
    """Verify one canonical Stage-1B correction-vector provenance link."""
    expected = {
        "status": "STAGE1B_BAYES_ALIGNMENT_SHARD_COMPLETE",
        "environment": environment,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "alignment_vectors_sha256": vector_sha256,
        "reference_manifest_sha256": reference_manifest_sha256,
        "rows": vector_rows,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise RuntimeError(f"Stage-1B shard provenance mismatch for {name}")


def verify_manifest_source_hashes(root: Path, manifest: dict, label: str) -> dict[str, str]:
    """Fail closed unless every source named by a frozen manifest is current."""
    sources = manifest.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError(f"{label} has no frozen source-hash map")
    verified = {}
    for relative, expected in sorted(sources.items()):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"{label} source is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"{label} source changed: {relative}")
        verified[relative] = observed
    return verified


def freeze(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    execution_host = require_remote_execution(cfg, root)
    logical_host = execution_host["logical_host"]
    target = host_specific_receipt_path(root, cfg, "freeze_receipt_template", logical_host)
    if target.exists():
        raise RuntimeError("intervention already frozen; refusing overwrite")
    static_correctness_path = host_specific_receipt_path(
        root, cfg, "static_receipt_template", logical_host,
    )
    static_correctness = json.loads(static_correctness_path.read_text())
    if static_correctness.get("status") != "REMOTE_STATIC_CORRECTNESS_PASS":
        raise RuntimeError("remote static/unit correctness did not pass")
    static_host = static_correctness.get("execution_host", {})
    if (
        static_host.get("logical_host") != logical_host
        or static_host.get("actual_hostname") != execution_host["actual_hostname"]
    ):
        raise RuntimeError("remote static correctness belongs to a different host")
    for relative, expected in static_correctness.get("source_hashes", {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"remote static correctness is stale for {relative}")
    sources = [
        config_path, root / cfg["protocol"], root / cfg["implementation"],
        root / cfg["queue_implementation"], root / cfg["remote_worker_launcher"],
        root / cfg["remote_host_registration"],
        root / cfg["job_manifest"],
        static_correctness_path,
    ]
    upstream_manifest_paths = {
        "stage1a": root / "runs/diagnostics/stage1_symptom_localization_v1/feature_manifest_frozen.json",
        "stage1b": root / "runs/diagnostics/stage1_bayes_alignment_v1/reference_manifest_frozen.json",
    }
    sources.extend(upstream_manifest_paths.values())
    prerequisite_rows = []
    for item in cfg["prerequisites"]:
        path = root / item["path"]
        payload = json.loads(path.read_text())
        if payload.get("status") not in item["allowed_status"]:
            raise RuntimeError(f"prerequisite not complete: {path}")
        sources.append(path)
        prerequisite_rows.append({"path": item["path"], "status": payload["status"], "sha256": sha256(path)})
    frozen_learners = {}
    for environment, item in cfg["environments"].items():
        for split in ("train", "select"):
            sources.extend((root / item[f"{split}_arrays"], root / item[f"{split}_manifest"]))
        sources.extend((root / item["checkpoint"], root / item["normalization"], root / item["training_receipt"]))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    for environment, item in cfg["environments"].items():
        frozen_learners[environment] = verify_frozen_learner_receipt(root, environment, item)
    upstream_manifests = {}
    expected_upstream_status = {
        "stage1a": "STAGE1_SYMPTOM_FEATURE_MANIFEST_FROZEN",
        "stage1b": "STAGE1B_BAYES_ALIGNMENT_MANIFEST_FROZEN",
    }
    for name, path in upstream_manifest_paths.items():
        manifest = json.loads(path.read_text())
        if manifest.get("status") != expected_upstream_status[name]:
            raise RuntimeError(f"{name} upstream manifest has the wrong status")
        verified_sources = verify_manifest_source_hashes(root, manifest, name)
        upstream_manifests[name] = {
            "path": str(path.relative_to(root)),
            "manifest_sha256": sha256(path),
            "verified_source_count": len(verified_sources),
            "verified_sources_digest": hashlib.sha256(
                _canonical_json(verified_sources).encode()
            ).hexdigest(),
        }
    jobs = pd.read_csv(root / cfg["job_manifest"])
    validate_job_manifest(jobs, cfg)
    receipt = {
        "schema_version": "1.0",
        "status": STATUS_FROZEN,
        "frozen_at_unix": time.time(),
        "source_hashes": {str(path.relative_to(root)): sha256(path) for path in sources},
        "prerequisites": prerequisite_rows,
        "training_jobs": int(len(jobs)),
        "formal_outcomes_read": False,
        "formal_outcomes_used_for_selection": False,
        "local_compute_used": False,
        "protected_scope_2_touched": False,
        "execution_host": execution_host,
        "freeze_receipt": str(target.relative_to(root)),
        "static_correctness_receipt": str(static_correctness_path.relative_to(root)),
        "static_correctness_sha256": sha256(static_correctness_path),
        "frozen_learners": frozen_learners,
        "upstream_manifests": upstream_manifests,
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def record_static_correctness(root: Path, config_path: Path, test_log: Path) -> dict:
    root, config_path, test_log = root.resolve(), config_path.resolve(), test_log.resolve()
    cfg = _config(root, config_path)
    host = require_remote_execution(cfg, root)
    text = test_log.read_text(errors="replace")
    if "passed" not in text or any(marker in text for marker in (" failed", " error", "ERRORS")):
        raise RuntimeError("test log does not show a clean passing remote run")
    target = host_specific_receipt_path(
        root, cfg, "static_receipt_template", host["logical_host"],
    )
    if target.exists():
        raise RuntimeError("immutable static correctness receipt already exists")
    sources = [
        config_path, root / cfg["protocol"], root / cfg["implementation"],
        root / cfg["queue_implementation"], root / "tests/unit/test_downstream_routing_intervention.py",
        root / cfg["job_manifest"], test_log,
    ]
    receipt = {
        "schema_version": "1.0", "status": "REMOTE_STATIC_CORRECTNESS_PASS",
        "execution_host": host, "source_hashes": {str(path.relative_to(root)): sha256(path) for path in sources},
        "test_log": str(test_log.relative_to(root)), "test_log_sha256": sha256(test_log),
        "canonical_result_artifacts_written": False, "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_freeze(root: Path, cfg: dict, host: dict | None = None) -> dict:
    host = host or require_remote_execution(cfg, root)
    path = host_specific_receipt_path(
        root, cfg, "freeze_receipt_template", host["logical_host"],
    )
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_FROZEN or receipt.get("formal_outcomes_read") is not False:
        raise RuntimeError("invalid intervention freeze receipt")
    execution_host = receipt.get("execution_host", {})
    if (
        execution_host.get("logical_host") != host["logical_host"]
        or execution_host.get("actual_hostname") != host["actual_hostname"]
        or receipt.get("freeze_receipt") != str(path.relative_to(root))
    ):
        raise RuntimeError("intervention freeze belongs to a different remote host")
    static_path = host_specific_receipt_path(
        root, cfg, "static_receipt_template", host["logical_host"],
    )
    if (
        receipt.get("static_correctness_receipt") != str(static_path.relative_to(root))
        or receipt.get("static_correctness_sha256") != sha256(static_path)
    ):
        raise RuntimeError("host-local static correctness is not bound by intervention freeze")
    for relative, expected in receipt["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    for name, item in receipt.get("upstream_manifests", {}).items():
        path = root / item["path"]
        if sha256(path) != item["manifest_sha256"]:
            raise RuntimeError(f"frozen upstream manifest changed: {name}")
        verified = verify_manifest_source_hashes(root, json.loads(path.read_text()), name)
        digest = hashlib.sha256(_canonical_json(verified).encode()).hexdigest()
        if len(verified) != int(item["verified_source_count"]) or digest != item["verified_sources_digest"]:
            raise RuntimeError(f"frozen upstream source identity changed: {name}")
    return receipt


def _common_freeze_source_hashes(receipt: dict, cfg: dict) -> dict:
    host_local = {receipt["static_correctness_receipt"]}
    return {path: digest for path, digest in receipt["source_hashes"].items() if path not in host_local}


def _freeze_equivalence_digest(receipt: dict, cfg: dict) -> str:
    identity = {
        "common_source_hashes": _common_freeze_source_hashes(receipt, cfg),
        "prerequisites": receipt["prerequisites"],
        "training_jobs": receipt["training_jobs"],
        "formal_outcomes_read": receipt["formal_outcomes_read"],
        "upstream_manifests": receipt.get("upstream_manifests", {}),
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def write_host_ownership_receipt(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    host = require_remote_execution(cfg, root)
    frozen = verify_freeze(root, cfg, host)
    logical = host["logical_host"]
    freeze_path = host_specific_receipt_path(root, cfg, "freeze_receipt_template", logical)
    target = root / cfg["output_root"] / "host_ownership" / f"{logical}.json"
    if target.exists():
        raise RuntimeError("immutable host ownership receipt already exists")
    receipt = {
        "schema_version": "1.0", "status": "HOST_OWNERSHIP_FROZEN",
        "logical_host": logical, "actual_hostname": host["actual_hostname"],
        "owned_environment": cfg["remote_execution"]["logical_host_environment"][logical],
        "partition_index": int(cfg["remote_execution"]["host_partition"][logical]),
        "freeze_equivalence_digest": _freeze_equivalence_digest(frozen, cfg),
        "common_source_hashes": _common_freeze_source_hashes(frozen, cfg),
        "local_static_correctness_sha256": frozen["static_correctness_sha256"],
        "local_freeze_receipt": str(freeze_path.relative_to(root)),
        "local_freeze_receipt_sha256": sha256(freeze_path),
        "config_sha256": sha256(config_path), "job_manifest_sha256": sha256(root / cfg["job_manifest"]),
        "sentinel_sha256": host["sentinel_sha256"], "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_two_host_equivalence(root: Path, config_path: Path, worker_a_receipt: Path,
                                worker_b_receipt: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    writer = require_remote_execution(cfg, root)
    if writer["logical_host"] != cfg["remote_execution"]["go_writer_logical_host"]:
        raise RuntimeError("only the frozen GO-writer host may create two-host equivalence")
    paths = {"worker_a": worker_a_receipt.resolve(), "worker_b": worker_b_receipt.resolve()}
    receipts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for logical, receipt in receipts.items():
        if receipt.get("status") != "HOST_OWNERSHIP_FROZEN" or receipt.get("logical_host") != logical:
            raise RuntimeError(f"invalid {logical} ownership receipt")
        if receipt.get("owned_environment") != cfg["remote_execution"]["logical_host_environment"][logical]:
            raise RuntimeError(f"{logical} environment ownership differs from config")
        if int(receipt.get("partition_index")) != int(cfg["remote_execution"]["host_partition"][logical]):
            raise RuntimeError(f"{logical} partition differs from config")
        expected_freeze = host_specific_receipt_path(root, cfg, "freeze_receipt_template", logical)
        if (
            receipt.get("local_freeze_receipt") != str(expected_freeze.relative_to(root))
            or receipt.get("local_freeze_receipt_sha256") != sha256(expected_freeze)
        ):
            raise RuntimeError(f"{logical} ownership does not bind its current host-specific freeze")
    if receipts["worker_a"]["actual_hostname"] == receipts["worker_b"]["actual_hostname"]:
        raise RuntimeError("worker_a/worker_b ownership receipts resolve to the same actual hostname")
    equal_fields = ("freeze_equivalence_digest", "common_source_hashes", "config_sha256", "job_manifest_sha256")
    for field in equal_fields:
        if receipts["worker_a"][field] != receipts["worker_b"][field]:
            raise RuntimeError(f"two-host equivalence failed for {field}")
    target = root / cfg["output_root"] / "host_ownership" / "TWO_HOST_EQUIVALENCE_GO.json"
    if target.exists():
        raise RuntimeError("immutable two-host equivalence receipt already exists")
    result = {
        "schema_version": "1.0", "status": "TWO_HOST_EQUIVALENCE_GO",
        "freeze_equivalence_digest": receipts["worker_a"]["freeze_equivalence_digest"],
        "ownership_receipts": {
            logical: {"sha256": sha256(paths[logical]), "actual_hostname": receipts[logical]["actual_hostname"],
                      "owned_environment": receipts[logical]["owned_environment"]}
            for logical in ("worker_a", "worker_b")
        },
        "verified_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def require_two_host_equivalence(root: Path, cfg: dict, host: dict) -> dict:
    path = root / cfg["output_root"] / "host_ownership" / "TWO_HOST_EQUIVALENCE_GO.json"
    receipt = json.loads(path.read_text())
    logical = host["logical_host"]
    ownership = root / cfg["output_root"] / "host_ownership" / f"{logical}.json"
    ownership_payload = json.loads(ownership.read_text())
    if receipt.get("status") != "TWO_HOST_EQUIVALENCE_GO":
        raise RuntimeError("two-host equivalence is not GO")
    if receipt["ownership_receipts"][logical]["sha256"] != sha256(ownership):
        raise RuntimeError("current host ownership receipt is not bound by equivalence GO")
    if receipt["ownership_receipts"][logical]["actual_hostname"] != host["actual_hostname"]:
        raise RuntimeError("equivalence GO actual hostname differs from current host")
    freeze_path = host_specific_receipt_path(root, cfg, "freeze_receipt_template", logical)
    if ownership_payload.get("local_freeze_receipt_sha256") != sha256(freeze_path):
        raise RuntimeError("current host freeze is not bound by ownership/equivalence GO")
    return receipt


def require_post_ownership_gate(root: Path, cfg: dict,
                                environment: str | None = None) -> tuple[dict, dict, dict]:
    """Require this host's freeze and the shared two-host GO after ownership."""
    host = require_remote_execution(cfg, root, environment)
    frozen = verify_freeze(root, cfg, host)
    equivalence = require_two_host_equivalence(root, cfg, host)
    return host, frozen, equivalence


def materialize_formal_inputs(root: Path, config_path: Path, environment: str) -> dict:
    """Reconstruct frozen Stage-1 formal rows and bind ref-B corrections.

    This is deterministic materialization of existing formal artifacts, not a
    new system draw and not learner-outcome access.
    """
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    _, selection_receipt = _verify_selection(root, cfg, environment)
    global_job_auth = require_global_job_authentication(root, cfg, config_path, environment)
    global_job_auth_path = root / cfg["output_root"] / "selection" / "ALL_54_JOBS_AUTHENTICATED_GO.json"
    selection_receipt_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    stage1_root = root / "runs/diagnostics/stage1_symptom_localization_v1"
    stage1b_root = root / "runs/diagnostics/stage1_bayes_alignment_v1"
    table_path = stage1_root / "frozen" / f"{environment}_stage1_rows.csv.gz"
    feature_manifest_path = stage1_root / "feature_manifest_frozen.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if feature_manifest.get("status") != "STAGE1_SYMPTOM_FEATURE_MANIFEST_FROZEN":
        raise RuntimeError("Stage-1A feature manifest is not frozen")
    stage1a_verified_sources = verify_manifest_source_hashes(root, feature_manifest, "Stage-1A feature manifest")
    stage1a_environment = feature_manifest.get("environments", {}).get(environment, {})
    if (
        stage1a_environment.get("row_manifest") != str(table_path.relative_to(root))
        or stage1a_environment.get("row_manifest_sha256") != sha256(table_path)
    ):
        raise RuntimeError("Stage-1A formal row table is not authenticated by its frozen manifest")
    source_table = pd.read_csv(table_path)
    if int(stage1a_environment.get("rows", -1)) != len(source_table):
        raise RuntimeError("Stage-1A frozen row count differs from the row table")
    table = canonicalize_formal_manifest(source_table, environment)
    if environment == "coupled":
        arrays = arrays_from_sample_manifest(
            stage1a._coupled_paths(root)["base_spec"], stage1a._coupled_paths(root)["system_pool"], table_path,
        )
    elif environment == "articulated":
        arrays = stage1a._articulated_arrays(root, table)
    else:
        raise ValueError(environment)
    _validate_manifest(arrays, table)
    if not np.array_equal(arrays.anchor_index.astype(np.int64), table.history_index.to_numpy(np.int64)):
        raise RuntimeError("materialized array anchor axis does not map exactly to frozen history_index")
    formal_root = root / cfg["output_root"] / "formal_inputs" / environment
    arrays_path = formal_root / "formal_arrays.npz"
    manifest_path = formal_root / "formal_row_manifest.csv.gz"
    formal_input_receipt_path = formal_root / "FORMAL_INPUTS_RECEIPT.json"
    correction_path = formal_root / "bayes_correction_ref_b.npz"
    if any(path.exists() for path in (arrays_path, manifest_path, correction_path, formal_input_receipt_path)):
        raise RuntimeError("immutable formal materialization already exists; refusing overwrite")

    stage1b_cfg = json.loads((root / "configs/stage1_bayes_alignment_v1.json").read_text())
    shard_count = int(stage1b_cfg["execution"][f"{environment}_shards"])
    reference_manifest_path = stage1b_root / "reference_manifest_frozen.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    reference_manifest_sha = sha256(reference_manifest_path)
    if reference_manifest.get("status") != "STAGE1B_BAYES_ALIGNMENT_MANIFEST_FROZEN":
        raise RuntimeError("Stage-1B reference manifest is not frozen")
    stage1b_verified_sources = verify_manifest_source_hashes(root, reference_manifest, "Stage-1B reference manifest")
    stage1a_manifest_relative = str(feature_manifest_path.relative_to(root))
    stage1a_table_relative = str(table_path.relative_to(root))
    if (
        reference_manifest.get("source_hashes", {}).get(stage1a_manifest_relative) != sha256(feature_manifest_path)
        or reference_manifest.get("source_hashes", {}).get(stage1a_table_relative) != sha256(table_path)
    ):
        raise RuntimeError("Stage-1B reference manifest does not bind current Stage-1A inputs")
    vector_parts = []
    vector_hashes = []
    vector_receipt_hashes = []
    for shard_index in range(shard_count):
        path = stage1b_root / "shards" / f"{environment}_{shard_index:02d}_of_{shard_count:02d}" / "alignment_vectors.npz"
        shard_receipt_path = path.parent / "receipt.json"
        with np.load(path, allow_pickle=False) as values:
            frame = pd.DataFrame({key: values[key] for key in stage1a.KEYS})
            frame["correction_position"] = np.arange(len(frame), dtype=np.int64)
            corrections = values["r_b_ref_b"].astype(np.float32)
        if len(corrections) != len(frame):
            raise RuntimeError("Stage-1B correction-vector rows differ from key rows")
        vector_sha = sha256(path)
        shard_receipt = json.loads(shard_receipt_path.read_text())
        verify_stage1b_shard_receipt(
            shard_receipt, environment, shard_index, shard_count, vector_sha,
            reference_manifest_sha, len(frame),
        )
        frame["shard_index"] = shard_index
        vector_parts.append((frame, corrections))
        vector_hashes.append(vector_sha)
        vector_receipt_hashes.append(sha256(shard_receipt_path))
    index = pd.concat([frame for frame, _ in vector_parts], ignore_index=True)
    if index.duplicated(stage1a.KEYS).any():
        raise RuntimeError("Stage-1B correction vectors duplicate formal keys")
    candidate = table[table.candidate_index >= 0].copy()
    candidate["row_index"] = candidate.index.to_numpy(np.int64)
    joined = candidate[[*stage1a.KEYS, "row_index"]].merge(index, on=stage1a.KEYS, how="left", validate="one_to_one")
    if joined.correction_position.isna().any() or len(joined) != len(candidate):
        raise RuntimeError("Stage-1B ref-B corrections do not cover every formal candidate row")
    correction_values = np.empty((len(joined), vector_parts[0][1].shape[1]), dtype=np.float32)
    for position, row in enumerate(joined.itertuples(index=False)):
        correction_values[position] = vector_parts[int(row.shard_index)][1][int(row.correction_position)]
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    save_arrays(arrays_path, arrays)
    manifest_tmp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    table.to_csv(manifest_tmp, index=False, compression="gzip"); os.replace(manifest_tmp, manifest_path)
    _atomic_npz(
        correction_path,
        row_index=joined.row_index.to_numpy(np.int64),
        bayes_correction=correction_values,
    )
    receipt = {
        "schema_version": "1.0", "status": "DOWNSTREAM_ROUTING_FORMAL_INPUTS_MATERIALIZED",
        "environment": environment, "rows": int(len(table)), "candidate_rows": int(len(candidate)),
        "arrays": str(arrays_path.relative_to(root)), "arrays_sha256": sha256(arrays_path),
        "manifest": str(manifest_path.relative_to(root)), "manifest_sha256": sha256(manifest_path),
        "correction": str(correction_path.relative_to(root)), "correction_sha256": sha256(correction_path),
        "source_stage1_manifest_sha256": sha256(table_path), "source_stage1b_vector_hashes": vector_hashes,
        "source_stage1a_feature_manifest_sha256": sha256(feature_manifest_path),
        "source_stage1b_reference_manifest_sha256": reference_manifest_sha,
        "source_stage1b_shard_receipt_hashes": vector_receipt_hashes,
        "source_stage1a_verified_sources_digest": hashlib.sha256(
            _canonical_json(stage1a_verified_sources).encode()
        ).hexdigest(),
        "source_stage1b_verified_sources_digest": hashlib.sha256(
            _canonical_json(stage1b_verified_sources).encode()
        ).hexdigest(),
        "anchor_axis_canonicalization": "anchor_index := history_index" if environment == "articulated" else "anchor_index == history_index verified",
        "reference_stream": "ref_b", "formal_learner_outcomes_read": False,
        "selection_frozen_before_materialization": True,
        "selection_receipt_sha256": sha256(selection_receipt_path),
        "selection_table_sha256": selection_receipt["table_sha256"],
        "global_54_job_authentication_sha256": sha256(global_job_auth_path),
    }
    _atomic_text(formal_input_receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def extract_cache(
    root: Path,
    config_path: Path,
    environment: str,
    split: str,
    shard_index: int,
    shard_count: int,
    device_name: str,
    arrays_path: Path | None = None,
    manifest_path: Path | None = None,
    correction_path: Path | None = None,
) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    require_canonical_shard_count(cfg, shard_count)
    item = cfg["environments"][environment]
    selection_receipt_sha256 = None
    formal_inputs_receipt_sha256 = None
    if split in ("train", "select"):
        arrays_path = root / item[f"{split}_arrays"]
        manifest_path = root / item[f"{split}_manifest"]
        if correction_path is not None:
            raise RuntimeError("Bayes correction is forbidden in train/select cache extraction")
    elif split == "formal":
        _verify_selection(root, cfg, environment)
        require_global_job_authentication(root, cfg, config_path, environment)
        selection_receipt_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
        selection_receipt_sha256 = sha256(selection_receipt_path)
        formal_root = root / cfg["output_root"] / "formal_inputs" / environment
        arrays_path = arrays_path or (formal_root / "formal_arrays.npz")
        manifest_path = manifest_path or (formal_root / "formal_row_manifest.csv.gz")
        correction_path = correction_path or (formal_root / "bayes_correction_ref_b.npz")
        arrays_path, manifest_path = arrays_path.resolve(), manifest_path.resolve()
        formal_inputs_receipt_path = formal_root / "FORMAL_INPUTS_RECEIPT.json"
        formal_inputs_receipt = json.loads(formal_inputs_receipt_path.read_text())
        if (
            formal_inputs_receipt.get("status") != "DOWNSTREAM_ROUTING_FORMAL_INPUTS_MATERIALIZED"
            or formal_inputs_receipt.get("environment") != environment
            or formal_inputs_receipt.get("arrays_sha256") != sha256(arrays_path)
            or formal_inputs_receipt.get("manifest_sha256") != sha256(manifest_path)
            or formal_inputs_receipt.get("correction_sha256") != sha256(correction_path)
            or formal_inputs_receipt.get("selection_receipt_sha256") != selection_receipt_sha256
        ):
            raise RuntimeError("formal input materialization receipt is invalid or stale")
        formal_inputs_receipt_sha256 = sha256(formal_inputs_receipt_path)
    else:
        raise ValueError(split)
    arrays = load_arrays(arrays_path)
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(arrays, manifest)
    active_conditions = item["active_condition_indices"]
    if split == "formal":
        # Formal manifests use their own condition labels (for example the
        # Coupled candidate condition is 1 rather than train relation labels
        # 2/3/4). The intervention population is protocol-invariant: every row
        # with an observed second segment.
        active_conditions = np.unique(arrays.condition[arrays.history_mask[:, 1] == 1]).tolist()
    positions = _active_positions(arrays, active_conditions, shard_index, shard_count)
    if not len(positions):
        raise RuntimeError("empty feature-cache shard")
    device = torch.device(device_name)
    resource_limits = configure_worker_resources(cfg, device)
    model, norms, before_hashes = _load_environment_model(root, cfg, environment, arrays, device)
    candidate_all = _candidate_identity(manifest, item["candidate_column"])
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "row_index", "system_index", "anchor_index", "query_index", "candidate_identity",
            "delta_persistent", "query_embedding", "predicted_latent_full",
            "prediction_original", "prediction_anchor", "normalized_target",
        )
    }
    correction = None
    if correction_path is not None:
        with np.load(correction_path, allow_pickle=False) as values:
            if "row_index" not in values or "bayes_correction" not in values:
                raise RuntimeError("formal correction file needs row_index and bayes_correction")
            lookup = {int(row): value for row, value in zip(values["row_index"], values["bayes_correction"])}
        correction = lookup
        parts["bayes_correction"] = []
    if split == "formal" and cfg["formal_evaluation"].get("bayes_correction_required", False) and correction is None:
        raise RuntimeError("formal cache requires an outcome-independent Bayes-correction vector")
    selected_systems = np.sort(np.unique(arrays.system_index[positions]))
    batch_size = int(cfg["optimization"]["batch_size"])
    with torch.no_grad():
        for system_index in selected_systems:
            system_positions = positions[arrays.system_index[positions] == system_index]
            for start in range(0, len(system_positions), batch_size):
                idx = system_positions[start:start + batch_size]
                subset = type(arrays)(
                    history=arrays.history[idx], history_mask=arrays.history_mask[idx],
                    query_action=arrays.query_action[idx], target=arrays.target[idx],
                    condition=arrays.condition[idx], system_index=arrays.system_index[idx],
                    anchor_index=arrays.anchor_index[idx], query_index=arrays.query_index[idx], theta=arrays.theta[idx],
                )
                h, m, q, target = _normalized(subset, norms)
                m_anchor = masked_second_segment(m)
                ht = torch.from_numpy(h).to(device)
                mt = torch.from_numpy(m).to(device)
                mat = torch.from_numpy(m_anchor).to(device)
                qt = torch.from_numpy(q).to(device)
                zp_full = model.persistent(ht, mt)
                zp_anchor = model.persistent(ht, mat)
                qembed = model.query_encoder(qt)
                z_full = model.latent_predictor(torch.cat((zp_full, qembed), dim=1))
                z_anchor = model.latent_predictor(torch.cat((zp_anchor, qembed), dim=1))
                pred_full = model.target_decoder(z_full)
                pred_anchor = model.target_decoder(z_anchor)
                batch_values = {
                    "row_index": idx.astype(np.int64),
                    "system_index": arrays.system_index[idx].astype(np.int64),
                    "anchor_index": arrays.anchor_index[idx].astype(np.int64),
                    "query_index": arrays.query_index[idx].astype(np.int64),
                    "candidate_identity": candidate_all[idx],
                    "delta_persistent": (zp_full - zp_anchor).cpu().numpy().astype(np.float32),
                    "query_embedding": qembed.cpu().numpy().astype(np.float32),
                    "predicted_latent_full": z_full.cpu().numpy().astype(np.float32),
                    "prediction_original": pred_full.cpu().numpy().astype(np.float32),
                    "prediction_anchor": pred_anchor.cpu().numpy().astype(np.float32),
                    "normalized_target": target.astype(np.float32),
                }
                if correction is not None:
                    missing = [int(row) for row in idx if int(row) not in correction]
                    if missing:
                        raise RuntimeError(f"Bayes correction missing formal rows: {missing[:3]}")
                    batch_values["bayes_correction"] = np.asarray([correction[int(row)] for row in idx], dtype=np.float32)
                for name, value in batch_values.items():
                    parts[name].append(value)
    merged = {name: np.concatenate(values) for name, values in parts.items()}
    order = np.argsort(merged["row_index"], kind="stable")
    merged = {name: value[order] for name, value in merged.items()}
    if len(np.unique(merged["row_index"])) != len(merged["row_index"]):
        raise RuntimeError("duplicate immutable row in cache shard")
    after_hashes = original_module_hashes(model)
    if before_hashes != after_hashes:
        raise RuntimeError("frozen original module changed during extraction")
    cache_root = root / cfg["output_root"] / "cache" / environment / split
    cache_path = cache_root / f"shard_{shard_index:02d}_of_{shard_count:02d}.npz"
    receipt_path = cache_path.with_suffix(".receipt.json")
    if cache_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable cache shard already exists; refusing overwrite")
    _atomic_npz(cache_path, **merged)
    receipt = {
        "schema_version": "1.0", "status": STATUS_CACHE,
        "environment": environment, "split": split,
        "shard_index": shard_index, "shard_count": shard_count,
        "system_assignment": "system_index_mod_shard_count",
        "rows": int(len(merged["row_index"])),
        "systems": int(len(np.unique(merged["system_index"]))),
        "arrays_sha256": sha256(arrays_path), "manifest_sha256": sha256(manifest_path),
        "arrays_path": str(arrays_path), "manifest_path": str(manifest_path),
        "correction_sha256": sha256(correction_path) if correction_path else None,
        "cache": str(cache_path.relative_to(root)), "cache_sha256": sha256(cache_path),
        "original_module_hashes_before": before_hashes,
        "original_module_hashes_after": after_hashes,
        "formal_outcomes_used_for_selection": False,
        "selection_receipt_sha256": selection_receipt_sha256,
        "formal_inputs_receipt_sha256": formal_inputs_receipt_sha256,
        "resource_limits": resource_limits, "peak_resource_usage": peak_resource_usage(device),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def run_cache_parity(root: Path, config_path: Path, environment: str, device_name: str) -> dict:
    """Fresh one-task forwards on fixed systems versus canonical 4-shard cache."""
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    device = torch.device(device_name); configure_worker_resources(cfg, device)
    split_results = {}
    for split in ("train", "select"):
        item = cfg["environments"][environment]
        arrays = load_arrays(root / item[f"{split}_arrays"])
        manifest = pd.read_csv(root / item[f"{split}_manifest"])
        _validate_manifest(arrays, manifest)
        active = _active_positions(arrays, item["active_condition_indices"], 0, 1)
        systems = np.sort(np.unique(arrays.system_index[active]))[:2]
        positions = active[np.isin(arrays.system_index[active], systems)]
        model, norms, module_hashes = _load_environment_model(root, cfg, environment, arrays, device)
        candidate = _candidate_identity(manifest, item["candidate_column"])
        direct = {name: [] for name in (
            "row_index", "system_index", "anchor_index", "query_index", "candidate_identity",
            "delta_persistent", "query_embedding", "predicted_latent_full",
            "prediction_original", "prediction_anchor", "normalized_target",
        )}
        with torch.no_grad():
            for system in systems:
                idx = positions[arrays.system_index[positions] == system]
                subset = type(arrays)(
                    history=arrays.history[idx], history_mask=arrays.history_mask[idx], query_action=arrays.query_action[idx],
                    target=arrays.target[idx], condition=arrays.condition[idx], system_index=arrays.system_index[idx],
                    anchor_index=arrays.anchor_index[idx], query_index=arrays.query_index[idx], theta=arrays.theta[idx],
                )
                h, m, q, target = _normalized(subset, norms)
                ma = masked_second_segment(m)
                ht = torch.from_numpy(h).to(device); mt = torch.from_numpy(m).to(device)
                mat = torch.from_numpy(ma).to(device); qt = torch.from_numpy(q).to(device)
                zf = model.persistent(ht, mt); za = model.persistent(ht, mat); qe = model.query_encoder(qt)
                pf = model.latent_predictor(torch.cat((zf, qe), dim=1)); pa = model.latent_predictor(torch.cat((za, qe), dim=1))
                values = {
                    "row_index": idx.astype(np.int64), "system_index": arrays.system_index[idx].astype(np.int64),
                    "anchor_index": arrays.anchor_index[idx].astype(np.int64), "query_index": arrays.query_index[idx].astype(np.int64),
                    "candidate_identity": candidate[idx], "delta_persistent": (zf - za).cpu().numpy().astype(np.float32),
                    "query_embedding": qe.cpu().numpy().astype(np.float32), "predicted_latent_full": pf.cpu().numpy().astype(np.float32),
                    "prediction_original": model.target_decoder(pf).cpu().numpy().astype(np.float32),
                    "prediction_anchor": model.target_decoder(pa).cpu().numpy().astype(np.float32),
                    "normalized_target": target.astype(np.float32),
                }
                for name, value in values.items(): direct[name].append(value)
        direct = {name: np.concatenate(values) for name, values in direct.items()}
        order = np.argsort(direct["row_index"], kind="stable"); direct = {name: values[order] for name, values in direct.items()}
        merged_path, merged_receipt = _cache_receipt(root, cfg, environment, split)
        merged = _load_npz(merged_path); keep = np.isin(merged["system_index"], systems)
        expected = {name: values[keep] for name, values in merged.items() if name in direct}
        if not np.array_equal(direct["row_index"], expected["row_index"]):
            raise RuntimeError("one-task and canonical-sharded cache row IDs differ")
        for name in direct:
            equal = np.array_equal(direct[name], expected[name], equal_nan=True) if np.issubdtype(direct[name].dtype, np.inexact) else np.array_equal(direct[name], expected[name])
            if not equal:
                raise RuntimeError(f"one-task and canonical-sharded cache differ for {split}/{name}")
        if original_module_hashes(model) != module_hashes:
            raise RuntimeError("original module changed during cache parity")
        split_results[split] = {"systems": systems.astype(int).tolist(), "rows": int(len(positions)), "cache_sha256": merged_receipt["merged_sha256"]}
    target = root / cfg["correctness"]["cache_parity_receipts"][environment]
    if target.exists():
        raise RuntimeError("immutable cache parity receipt already exists")
    receipt = {
        "schema_version": "1.0", "status": "REMOTE_ONE_TASK_VS_FOUR_SHARD_CACHE_PARITY_PASS",
        "environment": environment, "canonical_shard_count": 4, "parity_namespace_only": True,
        "splits": split_results, "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def merge_cache(root: Path, config_path: Path, environment: str, split: str, shard_count: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    require_canonical_shard_count(cfg, shard_count)
    cache_root = root / cfg["output_root"] / "cache" / environment / split
    item = cfg["environments"][environment]
    expected_selection_sha = None
    expected_formal_inputs_sha = None
    if split in ("train", "select"):
        expected_arrays_sha = sha256(root / item[f"{split}_arrays"])
        expected_manifest_sha = sha256(root / item[f"{split}_manifest"])
        expected_correction_sha = None
    elif split == "formal":
        formal_root = root / cfg["output_root"] / "formal_inputs" / environment
        formal_input_receipt_path = formal_root / "FORMAL_INPUTS_RECEIPT.json"
        formal_input_receipt = json.loads(formal_input_receipt_path.read_text())
        if (
            formal_input_receipt.get("status") != "DOWNSTREAM_ROUTING_FORMAL_INPUTS_MATERIALIZED"
            or formal_input_receipt.get("environment") != environment
            or formal_input_receipt.get("arrays_sha256") != sha256(formal_root / "formal_arrays.npz")
            or formal_input_receipt.get("manifest_sha256") != sha256(formal_root / "formal_row_manifest.csv.gz")
            or formal_input_receipt.get("correction_sha256") != sha256(formal_root / "bayes_correction_ref_b.npz")
        ):
            raise RuntimeError("formal materialization receipt is invalid")
        expected_arrays_sha = formal_input_receipt["arrays_sha256"]
        expected_manifest_sha = formal_input_receipt["manifest_sha256"]
        expected_correction_sha = formal_input_receipt["correction_sha256"]
        expected_formal_inputs_sha = sha256(formal_input_receipt_path)
        expected_selection_sha = sha256(
            root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
        )
    else:
        raise ValueError(split)
    receipts, parts = [], []
    for shard_index in range(shard_count):
        path = cache_root / f"shard_{shard_index:02d}_of_{shard_count:02d}.npz"
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        if (
            receipt.get("status") != STATUS_CACHE
            or receipt.get("environment") != environment
            or receipt.get("split") != split
            or int(receipt.get("shard_index", -1)) != shard_index
            or int(receipt.get("shard_count", -1)) != shard_count
            or receipt.get("cache_sha256") != sha256(path)
            or receipt.get("arrays_sha256") != expected_arrays_sha
            or receipt.get("manifest_sha256") != expected_manifest_sha
            or receipt.get("correction_sha256") != expected_correction_sha
            or receipt.get("selection_receipt_sha256") != expected_selection_sha
            or receipt.get("formal_inputs_receipt_sha256") != expected_formal_inputs_sha
        ):
            raise RuntimeError(f"invalid cache shard {path}")
        receipts.append(receipt)
        part = _load_npz(path)
        if np.any(part["system_index"] % shard_count != shard_index):
            raise RuntimeError(f"cache shard {shard_index} violates system ownership")
        parts.append(part)
    names = set(parts[0])
    if any(set(part) != names for part in parts):
        raise RuntimeError("cache shard schemas differ")
    merged = {name: np.concatenate([part[name] for part in parts]) for name in names}
    order = np.argsort(merged["row_index"], kind="stable")
    merged = {name: value[order] for name, value in merged.items()}
    rows = merged["row_index"].astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("merged cache repeats immutable rows")
    if split in ("train", "select"):
        arrays = load_arrays(root / item[f"{split}_arrays"])
        expected = _active_positions(arrays, item["active_condition_indices"], 0, 1)
        if not np.array_equal(rows, expected):
            raise RuntimeError("merged cache omits or adds active immutable rows")
    else:
        formal_arrays_path = root / cfg["output_root"] / "formal_inputs" / environment / "formal_arrays.npz"
        formal_arrays = load_arrays(formal_arrays_path)
        expected = np.flatnonzero(formal_arrays.history_mask[:, 1] == 1)
        if not np.array_equal(rows, expected):
            raise RuntimeError("merged formal cache does not exactly equal materialized active-row IDs")
        expected_systems = np.unique(formal_arrays.system_index[expected])
        if not np.array_equal(np.unique(merged["system_index"]), expected_systems):
            raise RuntimeError("merged formal cache has incomplete or extra physical systems")
        if sha256(formal_arrays_path) != expected_arrays_sha:
            raise RuntimeError("formal arrays changed after materialization")
    merged_path = cache_root / "merged.npz"
    merged_receipt_path = cache_root / "merged.receipt.json"
    if merged_path.exists() or merged_receipt_path.exists():
        raise RuntimeError("immutable merged cache already exists; refusing overwrite")
    _atomic_npz(merged_path, **merged)
    selection_receipt_sha256 = None
    formal_inputs_receipt_sha256 = None
    if split == "formal":
        selection_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
        selection_receipt_sha256 = sha256(selection_path)
        if any(row.get("selection_receipt_sha256") != selection_receipt_sha256 for row in receipts):
            raise RuntimeError("formal cache shard was not bound to the current frozen selection")
        formal_inputs_receipt_sha256 = expected_formal_inputs_sha
        if any(row.get("formal_inputs_receipt_sha256") != formal_inputs_receipt_sha256 for row in receipts):
            raise RuntimeError("formal cache shard was not bound to current materialized formal inputs")
    receipt = {
        "schema_version": "1.0", "status": STATUS_CACHE,
        "environment": environment, "split": split, "shards": shard_count,
        "fixed_merge_order": list(range(shard_count)), "rows": int(len(rows)),
        "systems": int(len(np.unique(merged["system_index"]))),
        "expected_active_rows": int(len(rows)),
        "shard_hashes": [row["cache_sha256"] for row in receipts],
        "merged_cache": str(merged_path.relative_to(root)), "merged_sha256": sha256(merged_path),
        "original_module_hashes": receipts[0]["original_module_hashes_after"],
        "selection_receipt_sha256": selection_receipt_sha256,
        "formal_inputs_receipt_sha256": formal_inputs_receipt_sha256,
    }
    if any(row["original_module_hashes_after"] != receipt["original_module_hashes"] for row in receipts):
        raise RuntimeError("original module hashes differ across cache shards")
    _atomic_text(merged_receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**63 - 1)


def cross_system_cell_permutation(cache: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Return donor positions matched on H,Q,candidate and deranged by system."""
    table = pd.DataFrame({
        "position": np.arange(len(cache["system_index"]), dtype=np.int64),
        "system_index": cache["system_index"].astype(np.int64),
        "anchor_index": cache["anchor_index"].astype(np.int64),
        "query_index": cache["query_index"].astype(np.int64),
        "candidate_identity": cache["candidate_identity"].astype(str),
        "row_index": cache["row_index"].astype(np.int64),
    })
    donor = np.full(len(table), -1, dtype=np.int64)
    cell_keys = ["anchor_index", "query_index", "candidate_identity"]
    for cell, group in table.groupby(cell_keys, sort=True):
        by_system = {int(system): values.sort_values("row_index").position.to_numpy(np.int64)
                     for system, values in group.groupby("system_index", sort=True)}
        systems = np.asarray(sorted(by_system), dtype=np.int64)
        if len(systems) < 2:
            raise RuntimeError(f"shuffle cell has fewer than two systems: {cell}")
        counts = {len(value) for value in by_system.values()}
        if len(counts) != 1:
            raise RuntimeError(f"shuffle cell has unequal within-system row counts: {cell}")
        rng = np.random.default_rng(_stable_seed("routing-shuffle-v1", seed, *cell))
        shift = int(rng.integers(1, len(systems)))
        donor_systems = np.roll(systems, shift)
        for receiver_system, donor_system in zip(systems, donor_systems):
            donor[by_system[int(receiver_system)]] = by_system[int(donor_system)]
    if np.any(donor < 0):
        raise RuntimeError("shuffle permutation left unmatched rows")
    if np.any(cache["system_index"][donor] == cache["system_index"]):
        raise RuntimeError("shuffle donor came from the same physical system")
    for name in ("anchor_index", "query_index", "candidate_identity"):
        if not np.array_equal(cache[name][donor], cache[name]):
            raise RuntimeError(f"shuffle changed exact-cell coordinate {name}")
    return donor


def validate_job_manifest(jobs: pd.DataFrame, cfg: dict) -> None:
    columns = ["job_id", "environment", "arm", "seed", "learning_rate"]
    if not set(columns).issubset(jobs.columns) or jobs.job_id.duplicated().any():
        raise RuntimeError("invalid or duplicate job manifest")
    expected = {
        (environment, arm, int(seed), float(rate))
        for environment in cfg["environments"]
        for arm in cfg["arms"]
        for seed in cfg["optimization"]["seeds"]
        for rate in cfg["optimization"]["learning_rates"]
    }
    observed = set(zip(jobs.environment, jobs.arm, jobs.seed.astype(int), jobs.learning_rate.astype(float)))
    if observed != expected or len(jobs) != 54:
        raise RuntimeError("job manifest is not the exact frozen 54-job Cartesian product")


def _cache_receipt(root: Path, cfg: dict, environment: str, split: str) -> tuple[Path, dict]:
    cache_root = root / cfg["output_root"] / "cache" / environment / split
    path = cache_root / "merged.npz"
    receipt = json.loads((cache_root / "merged.receipt.json").read_text())
    if receipt.get("status") != STATUS_CACHE or receipt.get("merged_sha256") != sha256(path):
        raise RuntimeError(f"invalid merged {environment}/{split} cache")
    return path, receipt


def _adapter_inputs(cache: dict[str, np.ndarray], arm: str, seed: int,
                    donor_positions: np.ndarray | None = None) -> np.ndarray:
    if arm == "true":
        return cache["delta_persistent"]
    if arm == "query_only":
        return np.zeros_like(cache["delta_persistent"])
    if arm == "shuffled":
        donor = cross_system_cell_permutation(cache, seed) if donor_positions is None else donor_positions
        return cache["delta_persistent"][donor]
    raise ValueError(arm)


@torch.no_grad()
def _selection_mse(adapter: RoutingAdapter, decoder: nn.Module, cache: dict[str, np.ndarray],
                   delta_input: np.ndarray, device: torch.device, batch_size: int) -> float:
    total, count = 0.0, 0
    adapter.eval()
    for start in range(0, len(delta_input), batch_size):
        stop = min(start + batch_size, len(delta_input))
        delta = torch.from_numpy(delta_input[start:stop]).to(device)
        query = torch.from_numpy(cache["query_embedding"][start:stop]).to(device)
        base = torch.from_numpy(cache["predicted_latent_full"][start:stop]).to(device)
        target = torch.from_numpy(cache["normalized_target"][start:stop]).to(device)
        prediction = decoder(base + adapter(delta, query))
        total += float(torch.sum((prediction - target) ** 2).cpu())
        count += int(target.numel())
    return total / count


def train_job(root: Path, config_path: Path, job_id: str, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    jobs = pd.read_csv(root / cfg["job_manifest"])
    validate_job_manifest(jobs, cfg)
    selected = jobs[jobs.job_id == job_id]
    if len(selected) != 1:
        raise ValueError(f"unknown job id {job_id}")
    job = selected.iloc[0]
    environment, arm = str(job.environment), str(job.arm)
    require_post_ownership_gate(root, cfg, environment)
    seed, learning_rate = int(job.seed), float(job.learning_rate)
    job_root = root / cfg["output_root"] / "jobs" / job_id
    immutable_outputs = (job_root / "adapter_best.pt", job_root / "training_curve.csv", job_root / "receipt.json")
    if any(path.exists() for path in immutable_outputs):
        raise RuntimeError("immutable job output already exists; retry requires queue-authorized archival/supersession")
    train_path, train_receipt = _cache_receipt(root, cfg, environment, "train")
    select_path, select_receipt = _cache_receipt(root, cfg, environment, "select")
    parity_path = root / cfg["correctness"]["cache_parity_receipts"][environment]
    parity = json.loads(parity_path.read_text())
    if parity.get("status") != "REMOTE_ONE_TASK_VS_FOUR_SHARD_CACHE_PARITY_PASS":
        raise RuntimeError("remote cache parity gate did not pass")
    if parity["splits"]["train"]["cache_sha256"] != train_receipt["merged_sha256"] or parity["splits"]["select"]["cache_sha256"] != select_receipt["merged_sha256"]:
        raise RuntimeError("cache parity receipt is stale")
    train_cache, select_cache = _load_npz(train_path), _load_npz(select_path)
    validate_adapter_cache_dimensions(train_cache, cfg, environment)
    validate_adapter_cache_dimensions(select_cache, cfg, environment)
    if "bayes_correction" in train_cache or "bayes_correction" in select_cache:
        raise RuntimeError("train/select cache contains forbidden formal correction data")
    train_delta = _adapter_inputs(train_cache, arm, seed)
    select_delta = _adapter_inputs(select_cache, arm, seed)
    arrays = load_arrays(root / cfg["environments"][environment]["train_arrays"])
    device = torch.device(device_name)
    resource_limits = configure_worker_resources(cfg, device)
    model, _, before_hashes = _load_environment_model(root, cfg, environment, arrays, device)
    if before_hashes != train_receipt["original_module_hashes"] or before_hashes != select_receipt["original_module_hashes"]:
        raise RuntimeError("cache/model original-module hash mismatch")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    adapter = routing_adapter_from_config(cfg, environment).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=cfg["optimization"]["weight_decay"])
    batch_size = int(cfg["optimization"]["batch_size"])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    bad_epochs, curve = 0, []
    for epoch in range(1, int(cfg["optimization"]["maximum_epochs"]) + 1):
        adapter.train()
        order = np.random.default_rng(_stable_seed("routing-train-order", seed, epoch)).permutation(len(train_delta))
        train_sum, train_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            delta = torch.from_numpy(train_delta[idx]).to(device)
            query = torch.from_numpy(train_cache["query_embedding"][idx]).to(device)
            base = torch.from_numpy(train_cache["predicted_latent_full"][idx]).to(device)
            target = torch.from_numpy(train_cache["normalized_target"][idx]).to(device)
            prediction = model.target_decoder(base + adapter(delta, query))
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(idx)
            train_count += len(idx)
        select_mse = _selection_mse(adapter, model.target_decoder, select_cache, select_delta, device, batch_size)
        curve.append({"epoch": epoch, "train_mse": train_sum / train_count, "select_mse": select_mse})
        if select_mse < best_loss:
            best_loss, best_epoch = select_mse, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in adapter.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= int(cfg["optimization"]["minimum_epochs"]) and bad_epochs >= int(cfg["optimization"]["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    adapter.load_state_dict(best_state)
    after_hashes = original_module_hashes(model)
    if before_hashes != after_hashes:
        raise RuntimeError("an original frozen module changed during adapter training")
    checkpoint = job_root / "adapter_best.pt"
    _atomic_torch(checkpoint, adapter.state_dict())
    curve_path = job_root / "training_curve.csv"
    curve_text = pd.DataFrame(curve).to_csv(index=False)
    _atomic_text(curve_path, curve_text)
    receipt = {
        "schema_version": "1.0", "status": STATUS_JOB, "job_id": job_id,
        "environment": environment, "arm": arm, "seed": seed, "learning_rate": learning_rate,
        "best_epoch": best_epoch, "best_select_mse": best_loss,
        "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": sha256(checkpoint),
        "curve": str(curve_path.relative_to(root)), "curve_sha256": sha256(curve_path),
        "train_cache_sha256": train_receipt["merged_sha256"], "select_cache_sha256": select_receipt["merged_sha256"],
        "original_module_hashes_before": before_hashes, "original_module_hashes_after": after_hashes,
        "selection_data": "immutable_select_only", "formal_outcomes_read": False,
        "resource_limits": resource_limits, "peak_resource_usage": peak_resource_usage(device),
        "job_manifest_sha256": sha256(root / cfg["job_manifest"]),
        "cache_parity_receipt_sha256": sha256(parity_path),
    }
    _atomic_text(job_root / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def select_checkpoints(root: Path, config_path: Path, environment_only: str | None = None) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment_only)
    jobs = pd.read_csv(root / cfg["job_manifest"])
    validate_job_manifest(jobs, cfg)
    rows = []
    environments = [environment_only] if environment_only is not None else list(cfg["environments"])
    if any(environment not in cfg["environments"] for environment in environments):
        raise ValueError(environment_only)
    for environment in environments:
        require_remote_execution(cfg, root, environment)
    authenticated_job_hashes = {}
    for environment in environments:
        train_path, train_cache_receipt = _cache_receipt(root, cfg, environment, "train")
        select_path, select_cache_receipt = _cache_receipt(root, cfg, environment, "select")
        environment_jobs = jobs[jobs.environment == environment]
        if len(environment_jobs) != 27:
            raise RuntimeError("environment does not have exactly 27 frozen jobs")
        for manifest_row in environment_jobs.itertuples(index=False):
            receipt_path = root / cfg["output_root"] / "jobs" / manifest_row.job_id / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            expected_tuple = (
                str(manifest_row.job_id), str(manifest_row.environment), str(manifest_row.arm),
                int(manifest_row.seed), float(manifest_row.learning_rate),
            )
            observed_tuple = (
                str(receipt.get("job_id")), str(receipt.get("environment")), str(receipt.get("arm")),
                int(receipt.get("seed")), float(receipt.get("learning_rate")),
            )
            if receipt.get("status") != STATUS_JOB or observed_tuple != expected_tuple:
                raise RuntimeError(f"job receipt tuple differs from manifest: {manifest_row.job_id}")
            if receipt.get("formal_outcomes_read") is not False or receipt.get("selection_data") != "immutable_select_only":
                raise RuntimeError("job receipt violates select-only boundary")
            checkpoint = root / receipt["checkpoint"]; curve = root / receipt["curve"]
            if sha256(checkpoint) != receipt["checkpoint_sha256"] or sha256(curve) != receipt["curve_sha256"]:
                raise RuntimeError(f"job artifact hash changed: {manifest_row.job_id}")
            if receipt.get("train_cache_sha256") != train_cache_receipt["merged_sha256"] or receipt.get("select_cache_sha256") != select_cache_receipt["merged_sha256"]:
                raise RuntimeError(f"job cache binding changed: {manifest_row.job_id}")
            if receipt.get("original_module_hashes_before") != receipt.get("original_module_hashes_after"):
                raise RuntimeError(f"original module mutated in job: {manifest_row.job_id}")
            if receipt.get("original_module_hashes_before") != train_cache_receipt["original_module_hashes"] or receipt.get("original_module_hashes_before") != select_cache_receipt["original_module_hashes"]:
                raise RuntimeError(f"job original-module hash differs from caches: {manifest_row.job_id}")
            if receipt.get("job_manifest_sha256") != sha256(root / cfg["job_manifest"]):
                raise RuntimeError(f"job manifest binding changed: {manifest_row.job_id}")
            done_path = root / cfg["output_root"] / "queue" / "done" / f"{manifest_row.job_id}.json"
            done = json.loads(done_path.read_text())
            if done.get("status") != "DONE" or done.get("job_receipt_sha256") != sha256(receipt_path) or done.get("checkpoint_sha256") != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"queue DONE receipt does not authenticate job: {manifest_row.job_id}")
            authenticated_job_hashes[str(manifest_row.job_id)] = sha256(receipt_path)
        for arm in cfg["arms"]:
            for seed in cfg["optimization"]["seeds"]:
                candidates = []
                for row in jobs[(jobs.environment == environment) & (jobs.arm == arm) & (jobs.seed == seed)].itertuples(index=False):
                    path = root / cfg["output_root"] / "jobs" / row.job_id / "receipt.json"
                    receipt = json.loads(path.read_text())
                    if receipt.get("status") != STATUS_JOB or receipt.get("formal_outcomes_read") is not False:
                        raise RuntimeError(f"invalid training job receipt: {path}")
                    candidates.append((float(receipt["best_select_mse"]), float(receipt["learning_rate"]), receipt, path))
                if len(candidates) != 3:
                    raise RuntimeError("environment-arm-seed does not have exactly three learning rates")
                _, _, chosen, receipt_path = min(candidates, key=lambda item: (item[0], item[1]))
                rows.append({
                    "environment": environment, "arm": arm, "seed": int(seed),
                    "job_id": chosen["job_id"], "learning_rate": chosen["learning_rate"],
                    "epoch": chosen["best_epoch"], "select_mse": chosen["best_select_mse"],
                    "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["checkpoint_sha256"],
                    "job_receipt": str(receipt_path.relative_to(root)), "job_receipt_sha256": sha256(receipt_path),
                })
    table = pd.DataFrame(rows).sort_values(["environment", "arm", "seed"]).reset_index(drop=True)
    expected_choices = 9 * len(environments)
    if len(table) != expected_choices:
        raise RuntimeError(f"selection did not produce {expected_choices} environment-arm-seed choices")
    selection_root = root / cfg["output_root"] / "selection" / (environment_only or "combined")
    table_path = selection_root / "selected_checkpoints.csv"
    if table_path.exists() or (selection_root / "SELECTION_FROZEN.json").exists():
        raise RuntimeError("immutable selection output already exists; refusing overwrite")
    _atomic_text(table_path, table.to_csv(index=False))
    receipt = {
        "schema_version": "1.0", "status": STATUS_SELECTION,
        "selected_at_unix": time.time(), "choices": len(table), "environments": environments,
        "selection_rule": "minimum immutable select MSE; lower learning rate is deterministic exact-tie break",
        "formal_outcomes_read": False, "formal_outcomes_used_for_selection": False,
        "table": str(table_path.relative_to(root)), "table_sha256": sha256(table_path),
        "authenticated_environment_jobs": authenticated_job_hashes,
        "authenticated_job_count": len(authenticated_job_hashes),
        "config_sha256": sha256(config_path), "job_manifest_sha256": sha256(root / cfg["job_manifest"]),
    }
    _atomic_text(selection_root / "SELECTION_FROZEN.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_selection(root: Path, cfg: dict, environment: str) -> tuple[pd.DataFrame, dict]:
    selection_root = root / cfg["output_root"] / "selection" / environment
    receipt = json.loads((selection_root / "SELECTION_FROZEN.json").read_text())
    table_path = selection_root / "selected_checkpoints.csv"
    if receipt.get("status") != STATUS_SELECTION or receipt.get("formal_outcomes_used_for_selection") is not False:
        raise RuntimeError("formal selection receipt is invalid")
    if receipt["table_sha256"] != sha256(table_path):
        raise RuntimeError("selected checkpoint table changed")
    table = pd.read_csv(table_path)
    if set(table.environment) != {environment} or len(table) != 9:
        raise RuntimeError("environment selection receipt has wrong scope")
    for row in table.itertuples(index=False):
        if sha256(root / row.checkpoint) != row.checkpoint_sha256:
            raise RuntimeError(f"selected adapter changed: {row.checkpoint}")
    return table, receipt


def verify_global_job_authentication(root: Path, config_path: Path, worker_a_selection: Path,
                                     worker_b_selection: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    writer, _, _ = require_post_ownership_gate(root, cfg)
    if writer["logical_host"] != cfg["remote_execution"]["go_writer_logical_host"]:
        raise RuntimeError("only the frozen GO-writer host may authenticate all 54 jobs")
    inputs = [worker_a_selection.resolve(), worker_b_selection.resolve()]
    receipts = [json.loads(path.read_text()) for path in inputs]
    if {tuple(row.get("environments", [])) for row in receipts} != {("articulated",), ("coupled",)}:
        raise RuntimeError("global job authentication needs one selection receipt per environment")
    authenticated = {}
    for path, receipt in zip(inputs, receipts):
        if receipt.get("status") != STATUS_SELECTION or receipt.get("authenticated_job_count") != 27:
            raise RuntimeError("selection receipt did not authenticate its 27 host-owned jobs")
        if receipt.get("config_sha256") != sha256(config_path) or receipt.get("job_manifest_sha256") != sha256(root / cfg["job_manifest"]):
            raise RuntimeError("selection receipt config/job-manifest binding differs")
        overlap = set(authenticated) & set(receipt["authenticated_environment_jobs"])
        if overlap:
            raise RuntimeError("selection receipts overlap job ownership")
        authenticated.update(receipt["authenticated_environment_jobs"])
    jobs = pd.read_csv(root / cfg["job_manifest"]); validate_job_manifest(jobs, cfg)
    if set(authenticated) != set(jobs.job_id.astype(str)) or len(authenticated) != 54:
        raise RuntimeError("distributed selection receipts do not authenticate all 54 jobs")
    target = root / cfg["output_root"] / "selection" / "ALL_54_JOBS_AUTHENTICATED_GO.json"
    if target.exists():
        raise RuntimeError("immutable global 54-job authentication already exists")
    selection_receipts = {
        str(receipt["environments"][0]): {
            "path": str(path), "sha256": sha256(path),
        }
        for path, receipt in zip(inputs, receipts)
    }
    receipt = {
        "schema_version": "1.0", "status": "ALL_54_JOBS_AUTHENTICATED_GO",
        "authenticated_job_count": 54, "authenticated_job_receipt_hashes": authenticated,
        "selection_receipts": selection_receipts,
        "config_sha256": sha256(config_path), "job_manifest_sha256": sha256(root / cfg["job_manifest"]),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def require_global_job_authentication(root: Path, cfg: dict, config_path: Path,
                                      environment: str) -> dict:
    path = root / cfg["output_root"] / "selection" / "ALL_54_JOBS_AUTHENTICATED_GO.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "ALL_54_JOBS_AUTHENTICATED_GO" or receipt.get("authenticated_job_count") != 54:
        raise RuntimeError("all 54 jobs have not been globally authenticated")
    if receipt.get("config_sha256") != sha256(config_path):
        raise RuntimeError("global job authentication config binding is stale")
    if receipt.get("job_manifest_sha256") != sha256(root / cfg["job_manifest"]):
        raise RuntimeError("global job authentication is stale")
    local_selection = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    bound = receipt.get("selection_receipts", {}).get(environment)
    if not isinstance(bound, dict) or bound.get("sha256") != sha256(local_selection):
        raise RuntimeError("current environment selection is not bound by global job authentication")
    return receipt


def freeze_formal_shuffle_maps(root: Path, config_path: Path, environment: str) -> dict:
    """Freeze global formal donor maps before any system-sharded evaluation."""
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment); _verify_selection(root, cfg, environment)
    cache_path, cache_receipt = _cache_receipt(root, cfg, environment, "formal")
    cache = _load_npz(cache_path)
    map_root = root / cfg["output_root"] / "formal_shuffle_maps" / environment
    final_receipt = map_root / "FORMAL_SHUFFLE_MAPS_FROZEN.json"
    if final_receipt.exists():
        raise RuntimeError("formal shuffle maps already frozen; refusing overwrite")
    maps = []
    for seed in cfg["optimization"]["seeds"]:
        donor_position = cross_system_cell_permutation(cache, int(seed))
        path = map_root / f"seed_{int(seed)}.npz"
        _atomic_npz(
            path,
            receiver_row_index=cache["row_index"].astype(np.int64),
            donor_row_index=cache["row_index"][donor_position].astype(np.int64),
            donor_position=donor_position.astype(np.int64),
        )
        maps.append({"seed": int(seed), "path": str(path.relative_to(root)), "sha256": sha256(path)})
    selection_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    receipt = {
        "schema_version": "1.0", "status": "FORMAL_GLOBAL_SHUFFLE_MAPS_FROZEN",
        "environment": environment, "rows": int(len(cache["row_index"])),
        "scope": "complete_merged_formal_cache_before_sharding", "maps": maps,
        "formal_cache_sha256": cache_receipt["merged_sha256"],
        "selection_receipt_sha256": sha256(selection_path),
    }
    _atomic_text(final_receipt, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _load_formal_shuffle_map(root: Path, cfg: dict, environment: str, seed: int,
                             cache: dict[str, np.ndarray], cache_receipt: dict) -> tuple[np.ndarray, str]:
    map_root = root / cfg["output_root"] / "formal_shuffle_maps" / environment
    receipt_path = map_root / "FORMAL_SHUFFLE_MAPS_FROZEN.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FORMAL_GLOBAL_SHUFFLE_MAPS_FROZEN" or receipt.get("formal_cache_sha256") != cache_receipt["merged_sha256"]:
        raise RuntimeError("global formal shuffle-map receipt is invalid")
    item = next((row for row in receipt["maps"] if int(row["seed"]) == int(seed)), None)
    if item is None:
        raise RuntimeError(f"missing global formal shuffle map for seed {seed}")
    path = root / item["path"]
    if sha256(path) != item["sha256"]:
        raise RuntimeError("global formal shuffle map changed")
    values = _load_npz(path)
    if not np.array_equal(values["receiver_row_index"], cache["row_index"]):
        raise RuntimeError("global formal shuffle receiver order differs from merged cache")
    donor = values["donor_position"].astype(np.int64)
    if not np.array_equal(cache["row_index"][donor], values["donor_row_index"]):
        raise RuntimeError("global formal donor IDs do not match stored positions")
    return donor, item["sha256"]


def _row_metrics(prediction: np.ndarray, anchor: np.ndarray, target: np.ndarray,
                 correction: np.ndarray | None) -> dict[str, np.ndarray]:
    if prediction.shape != anchor.shape or prediction.shape != target.shape or prediction.ndim != 2:
        raise RuntimeError("formal prediction/anchor/target shapes differ")
    for name, values in (("prediction", prediction), ("anchor", anchor), ("target", target)):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"formal {name} contains a nonfinite value")
    loss = np.mean((prediction - target) ** 2, axis=1).astype(np.float64)
    anchor_loss = np.mean((anchor - target) ** 2, axis=1).astype(np.float64)
    result = {"loss": loss, "gain": anchor_loss - loss}
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(result["gain"])):
        raise RuntimeError("formal loss/gain contains a nonfinite value")
    if correction is not None:
        if correction.shape != prediction.shape:
            raise RuntimeError("Bayes correction shape differs from prediction")
        delta = prediction.astype(np.float64) - anchor.astype(np.float64)
        reference = correction.astype(np.float64)
        if not np.all(np.isfinite(reference)):
            raise RuntimeError("Bayes correction contains a nonfinite value")
        r2 = np.sum(reference * reference, axis=1)
        dot = np.sum(delta * reference, axis=1)
        d2 = np.sum(delta * delta, axis=1)
        if np.any(r2 <= 0) or not np.all(np.isfinite(r2)):
            raise RuntimeError("required Bayes correction has zero or invalid norm")
        a = dot / r2
        orthogonal2 = np.maximum(d2 - (dot * dot) / r2, 0.0)
        b = np.sqrt(orthogonal2) / np.sqrt(r2)
        rho = np.sqrt(d2) / np.sqrt(r2)
        # A zero learner update has no direction. Freeze its supporting cosine
        # coordinate to 0 rather than allowing pandas to skip NaN while later
        # dividing by the total candidate-row count.
        cos = np.divide(dot, np.sqrt(d2 * r2), out=np.zeros(len(delta)), where=d2 > 0)
        result.update({"a": a, "b": b, "rho": rho, "cos_theta": cos, "v_l_conditional": 2.0 * dot - d2})
        if any(not np.all(np.isfinite(values)) for values in result.values()):
            raise RuntimeError("formal transport metric contains a nonfinite value")
    return result


def _aggregate_formal_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical float64 per-system sufficient statistics."""
    metric_names = [
        name for name in ("loss", "gain", "a", "b", "rho", "cos_theta", "v_l_conditional")
        if name in rows
    ]
    if not metric_names or rows[metric_names].isna().any().any():
        raise RuntimeError("formal rows contain missing metric values")
    if not np.all(np.isfinite(rows[metric_names].to_numpy(np.float64))):
        raise RuntimeError("formal rows contain nonfinite metric values")
    aggregate_spec = {f"sum_{name}": (name, "sum") for name in metric_names}
    aggregate_spec.update({
        f"sum2_{name}": (
            name, lambda values: float(np.sum(values.to_numpy(np.float64) ** 2)),
        )
        for name in metric_names
    })
    stats = rows.groupby(["system_index", "arm", "seed"], sort=True, as_index=False).agg(
        count=("gain", "size"), **aggregate_spec,
    )
    for name in stats.columns:
        if name.startswith("sum"):
            stats[name] = stats[name].astype(np.float64)
    return stats.sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)


def _parity_systems_per_shard(system_index: np.ndarray, shard_count: int) -> np.ndarray:
    """Choose one deterministic physical system from every modulo shard."""
    available = np.sort(np.unique(np.asarray(system_index, dtype=np.int64)))
    chosen = []
    for shard_index in range(shard_count):
        candidates = available[available % shard_count == shard_index]
        if not len(candidates):
            raise RuntimeError("formal parity requires at least one system in every modulo shard")
        chosen.append(int(candidates[0]))
    return np.asarray(chosen, dtype=np.int64)


@torch.no_grad()
def _evaluate_formal_positions(
    root: Path,
    cfg: dict,
    environment: str,
    selection: pd.DataFrame,
    full_cache: dict[str, np.ndarray],
    formal_receipt: dict,
    receiver_positions: np.ndarray,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    """Independently forward and aggregate one explicit receiver population.

    Shuffled donors are always resolved against the complete merged cache.
    Calling this function separately for the full parity population and for
    each modulo shard therefore tests receiver partitioning without changing
    donor scope.
    """
    receiver_positions = np.asarray(receiver_positions, dtype=np.int64)
    if not len(receiver_positions):
        raise RuntimeError("formal receiver population is empty")
    if len(np.unique(receiver_positions)) != len(receiver_positions):
        raise RuntimeError("formal receiver positions repeat")
    if np.any(receiver_positions < 0) or np.any(receiver_positions >= len(full_cache["row_index"])):
        raise RuntimeError("formal receiver position is out of range")
    cache = {name: value[receiver_positions] for name, value in full_cache.items()}
    arrays = load_arrays(root / cfg["environments"][environment]["train_arrays"])
    model, _, before_hashes = _load_environment_model(root, cfg, environment, arrays, device)
    if before_hashes != formal_receipt["original_module_hashes"]:
        raise RuntimeError("formal cache/model module hash mismatch")
    correction = cache.get("bayes_correction")
    if cfg["formal_evaluation"].get("bayes_correction_required", False) and correction is None:
        raise RuntimeError("formal evaluation requires Bayes correction for mediator reporting")

    output_rows: list[dict] = []

    def append_metrics(arm: str, seed: int, metrics: dict[str, np.ndarray]) -> None:
        for position in range(len(cache["row_index"])):
            row = {
                "row_index": int(cache["row_index"][position]),
                "system_index": int(cache["system_index"][position]),
                "anchor_index": int(cache["anchor_index"][position]),
                "query_index": int(cache["query_index"][position]),
                "candidate_identity": str(cache["candidate_identity"][position]),
                "arm": arm, "seed": seed,
            }
            row.update({name: float(values[position]) for name, values in metrics.items()})
            output_rows.append(row)

    append_metrics(
        "original", -1,
        _row_metrics(
            cache["prediction_original"], cache["prediction_anchor"],
            cache["normalized_target"], correction,
        ),
    )
    shuffle_map_hashes: dict[str, str] = {}
    batch_size = int(cfg["optimization"]["batch_size"])
    for choice in selection[selection.environment == environment].itertuples(index=False):
        validate_adapter_cache_dimensions(cache, cfg, environment)
        adapter = routing_adapter_from_config(cfg, environment).to(device)
        adapter.load_state_dict(torch.load(root / choice.checkpoint, map_location=device, weights_only=True))
        adapter.eval()
        if str(choice.arm) == "shuffled":
            global_donor, map_hash = _load_formal_shuffle_map(
                root, cfg, environment, int(choice.seed), full_cache, formal_receipt,
            )
            delta_input = full_cache["delta_persistent"][global_donor[receiver_positions]]
            shuffle_map_hashes[str(int(choice.seed))] = map_hash
        elif str(choice.arm) == "query_only":
            delta_input = np.zeros_like(cache["delta_persistent"])
        elif str(choice.arm) == "true":
            delta_input = cache["delta_persistent"]
        else:
            raise ValueError(f"unknown adapter arm {choice.arm}")
        # Keep every batch system-local. The one-task parity call and each
        # modulo-shard call then execute identical matrix shapes per physical
        # system, so exact float comparison is meaningful rather than relying
        # on a numerical tolerance for differently shaped GEMMs.
        prediction = np.empty_like(cache["prediction_original"], dtype=np.float32)
        for system in np.sort(np.unique(cache["system_index"])):
            local = np.flatnonzero(cache["system_index"] == system)
            for start in range(0, len(local), batch_size):
                positions = local[start:start + batch_size]
                delta = torch.from_numpy(delta_input[positions]).to(device)
                query = torch.from_numpy(cache["query_embedding"][positions]).to(device)
                base = torch.from_numpy(cache["predicted_latent_full"][positions]).to(device)
                prediction[positions] = model.target_decoder(base + adapter(delta, query)).cpu().numpy()
        append_metrics(
            str(choice.arm), int(choice.seed),
            _row_metrics(
                prediction, cache["prediction_anchor"], cache["normalized_target"], correction,
            ),
        )
    after_hashes = original_module_hashes(model)
    if before_hashes != after_hashes:
        raise RuntimeError("original model changed during formal evaluation")
    rows = pd.DataFrame(output_rows).sort_values(
        ["system_index", "row_index", "arm", "seed"],
    ).reset_index(drop=True)
    return rows, _aggregate_formal_rows(rows), before_hashes, after_hashes, shuffle_map_hashes


def run_formal_parity(root: Path, config_path: Path, environment: str, device_name: str) -> dict:
    """Independent one-task/four-forward evaluation and global-donor parity gate."""
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment); selection, _ = _verify_selection(root, cfg, environment)
    formal_path, formal_receipt = _cache_receipt(root, cfg, environment, "formal")
    cache = _load_npz(formal_path)
    shard_count = int(cfg["cache"]["system_shards_per_environment"])
    systems = _parity_systems_per_shard(cache["system_index"], shard_count)
    receiver = np.flatnonzero(np.isin(cache["system_index"], systems))
    if not len(receiver):
        raise RuntimeError("formal parity system subset is empty")
    device = torch.device(device_name); configure_worker_resources(cfg, device)
    _, one_task, one_before, one_after, shuffle_hashes = _evaluate_formal_positions(
        root, cfg, environment, selection, cache, formal_receipt, receiver, device,
    )
    shard_parts = []
    shard_module_hashes = []
    shard_shuffle_hashes = []
    shard_rows = []
    for shard_index in range(shard_count):
        shard_receiver = np.flatnonzero(
            np.isin(cache["system_index"], systems) & (cache["system_index"] % shard_count == shard_index),
        )
        rows, stats, before, after, hashes = _evaluate_formal_positions(
            root, cfg, environment, selection, cache, formal_receipt, shard_receiver, device,
        )
        shard_parts.append(stats)
        shard_module_hashes.append({"before": before, "after": after})
        shard_shuffle_hashes.append(hashes)
        shard_rows.append(int(len(rows)))
    fixed_merge = pd.concat(shard_parts, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    if list(one_task.columns) != list(fixed_merge.columns):
        raise RuntimeError("formal parity schemas differ")
    for column in one_task:
        left, right = one_task[column].to_numpy(), fixed_merge[column].to_numpy()
        equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.inexact) else np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"one-task and fixed four-shard formal stats differ for {column}")
    if one_before != one_after or any(row["before"] != row["after"] for row in shard_module_hashes):
        raise RuntimeError("original model changed during independent formal parity forwards")
    if any(hashes != shuffle_hashes for hashes in shard_shuffle_hashes):
        raise RuntimeError("global shuffle-map bindings differ across parity forwards")
    target = root / cfg["correctness"]["formal_parity_receipts"][environment]
    if target.exists():
        raise RuntimeError("immutable formal parity receipt already exists")
    selection_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    receipt = {
        "schema_version": "1.0", "status": "REMOTE_ONE_TASK_VS_FOUR_SHARD_FORMAL_PARITY_PASS",
        "environment": environment, "systems": systems.astype(int).tolist(),
        "system_by_modulo_shard": {str(index): int(system) for index, system in enumerate(systems)},
        "canonical_shard_count": 4, "independent_forward_count": 5,
        "one_task_receiver_rows": int(len(receiver)), "four_shard_rows_with_arms": shard_rows,
        "statistics_dtype": "float64", "fixed_merge_order": [0, 1, 2, 3],
        "formal_cache_sha256": formal_receipt["merged_sha256"], "selection_receipt_sha256": sha256(selection_path),
        "global_shuffle_map_hashes": shuffle_hashes, "parity_namespace_only": True,
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


@torch.no_grad()
def evaluate_shard(root: Path, config_path: Path, environment: str, shard_index: int,
                   shard_count: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    require_canonical_shard_count(cfg, shard_count)
    selection, selection_receipt = _verify_selection(root, cfg, environment)
    formal_path, formal_receipt = _cache_receipt(root, cfg, environment, "formal")
    parity_path = root / cfg["correctness"]["formal_parity_receipts"][environment]
    parity = json.loads(parity_path.read_text())
    selection_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    if (
        parity.get("status") != "REMOTE_ONE_TASK_VS_FOUR_SHARD_FORMAL_PARITY_PASS"
        or parity.get("formal_cache_sha256") != formal_receipt["merged_sha256"]
        or parity.get("selection_receipt_sha256") != sha256(selection_path)
        or int(parity.get("canonical_shard_count", -1)) != shard_count
        or int(parity.get("independent_forward_count", -1)) != 5
    ):
        raise RuntimeError("formal parity gate missing or stale")
    full_cache = _load_npz(formal_path)
    receiver_positions = np.flatnonzero(full_cache["system_index"] % shard_count == shard_index)
    if not len(receiver_positions):
        raise RuntimeError("empty formal-evaluation shard")
    device = torch.device(device_name)
    resource_limits = configure_worker_resources(cfg, device)
    if formal_receipt.get("selection_receipt_sha256") != sha256(selection_path):
        raise RuntimeError("formal cache is not bound to the current frozen selection")
    rows, stats, before_hashes, after_hashes, shuffle_map_hashes = _evaluate_formal_positions(
        root, cfg, environment, selection, full_cache, formal_receipt, receiver_positions, device,
    )
    if shuffle_map_hashes != parity.get("global_shuffle_map_hashes"):
        raise RuntimeError("formal evaluation shuffle maps differ from the parity-gated maps")
    eval_root = root / cfg["output_root"] / "formal_eval" / environment
    row_path = eval_root / f"rows_shard_{shard_index:02d}_of_{shard_count:02d}.csv.gz"
    stats_path = eval_root / f"stats_shard_{shard_index:02d}_of_{shard_count:02d}.csv.gz"
    receipt_path = eval_root / f"receipt_shard_{shard_index:02d}_of_{shard_count:02d}.json"
    if row_path.exists() or stats_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable formal-evaluation shard already exists; refusing overwrite")
    # pandas compression to a temporary path is atomic after replacement.
    row_tmp = row_path.with_name(row_path.name + f".tmp.{os.getpid()}")
    stats_tmp = stats_path.with_name(stats_path.name + f".tmp.{os.getpid()}")
    row_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(row_tmp, index=False, compression="gzip")
    stats.to_csv(stats_tmp, index=False, compression="gzip")
    os.replace(row_tmp, row_path); os.replace(stats_tmp, stats_path)
    receipt = {
        "schema_version": "1.0", "status": STATUS_EVAL_SHARD,
        "environment": environment, "shard_index": shard_index, "shard_count": shard_count,
        "systems": int(rows.system_index.nunique()), "rows_with_arms": int(len(rows)),
        "row_table": str(row_path.relative_to(root)), "row_table_sha256": sha256(row_path),
        "statistics": str(stats_path.relative_to(root)), "statistics_sha256": sha256(stats_path),
        "statistics_dtype": "float64", "selection_receipt_sha256": sha256(root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"),
        "original_module_hashes_before": before_hashes, "original_module_hashes_after": after_hashes,
        "global_formal_shuffle_map_hashes": shuffle_map_hashes,
        "formal_parity_receipt_sha256": sha256(parity_path),
        "resource_limits": resource_limits, "peak_resource_usage": peak_resource_usage(device),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _paired_bootstrap(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        index = rng.integers(0, len(values), size=(count, len(values)))
        boot[start:start + count] = values[index].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def merge_evaluation(root: Path, config_path: Path, environment: str, shard_count: int) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _config(root, config_path)
    require_post_ownership_gate(root, cfg, environment)
    require_canonical_shard_count(cfg, shard_count)
    _verify_selection(root, cfg, environment)
    eval_root = root / cfg["output_root"] / "formal_eval" / environment
    selection_path = root / cfg["output_root"] / "selection" / environment / "SELECTION_FROZEN.json"
    parity_path = root / cfg["correctness"]["formal_parity_receipts"][environment]
    expected_selection_sha = sha256(selection_path)
    expected_parity_sha = sha256(parity_path)
    tables, receipts = [], []
    for shard_index in range(shard_count):
        receipt_path = eval_root / f"receipt_shard_{shard_index:02d}_of_{shard_count:02d}.json"
        receipt = json.loads(receipt_path.read_text())
        path = root / receipt["statistics"]
        if (
            receipt.get("status") != STATUS_EVAL_SHARD
            or receipt.get("environment") != environment
            or int(receipt.get("shard_index", -1)) != shard_index
            or int(receipt.get("shard_count", -1)) != shard_count
            or receipt["statistics_sha256"] != sha256(path)
            or receipt.get("selection_receipt_sha256") != expected_selection_sha
            or receipt.get("formal_parity_receipt_sha256") != expected_parity_sha
        ):
            raise RuntimeError(f"invalid formal shard {shard_index}")
        table = pd.read_csv(path)
        if np.any(table.system_index.to_numpy(np.int64) % shard_count != shard_index):
            raise RuntimeError(f"formal statistics shard {shard_index} violates system ownership")
        tables.append(table)
        receipts.append(receipt)
    stats = pd.concat(tables, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    duplicated = stats.duplicated(["system_index", "arm", "seed"])
    if duplicated.any():
        raise RuntimeError("physical system appears in multiple formal shards")
    if any(row["original_module_hashes_after"] != receipts[0]["original_module_hashes_after"] for row in receipts):
        raise RuntimeError("original module hash differs across formal shards")
    if any(row.get("global_formal_shuffle_map_hashes") != receipts[0].get("global_formal_shuffle_map_hashes") for row in receipts):
        raise RuntimeError("global formal shuffle-map hashes differ across evaluation shards")
    per_seed = stats.assign(mean_gain=stats.sum_gain.astype(np.float64) / stats["count"].astype(np.float64))
    formal_cache, _ = _cache_receipt(root, cfg, environment, "formal")
    formal_values = _load_npz(formal_cache)
    expected_systems = set(map(int, np.unique(formal_values["system_index"])))
    expected_combinations = {("original", -1)} | {
        (arm, int(seed)) for arm in cfg["arms"] for seed in cfg["optimization"]["seeds"]
    }
    observed_combinations = set(zip(per_seed.arm.astype(str), per_seed.seed.astype(int)))
    if observed_combinations != expected_combinations:
        raise RuntimeError("formal evaluation does not contain the exact original/arm/seed combinations")
    for arm, seed in sorted(expected_combinations):
        subset = per_seed[(per_seed.arm == arm) & (per_seed.seed == seed)]
        if set(map(int, subset.system_index)) != expected_systems:
            raise RuntimeError(f"formal evaluation system coverage differs for {arm}/{seed}")
    counts = per_seed.pivot(index="system_index", columns=["arm", "seed"], values="count")
    if counts.isna().any().any() or not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise RuntimeError("formal evaluation row counts differ across arms/seeds within system")
    original = per_seed[per_seed.arm == "original"].set_index("system_index").mean_gain.sort_index()
    arm_means = {arm: per_seed[per_seed.arm == arm].groupby("system_index", sort=True)["mean_gain"].mean()
                 for arm in cfg["arms"]}
    systems = sorted(expected_systems)
    if set(map(int, original.index)) != expected_systems or any(set(map(int, values.index)) != expected_systems for values in arm_means.values()):
        raise RuntimeError("system coverage mismatch; intersection fallback is forbidden")
    original = original.loc[systems]
    arm_means = {arm: values.loc[systems] for arm, values in arm_means.items()}
    contrasts = {
        "route": arm_means["true"].to_numpy() - original.to_numpy(),
        "shuffle": arm_means["true"].to_numpy() - arm_means["shuffled"].to_numpy(),
        "query": arm_means["true"].to_numpy() - arm_means["query_only"].to_numpy(),
    }
    estimates = {}
    for offset, (name, values) in enumerate(contrasts.items()):
        point, low, high = _paired_bootstrap(
            values, int(cfg["formal_evaluation"]["bootstrap_replicates"]),
            int(cfg["formal_evaluation"]["bootstrap_seed"]) + offset,
        )
        estimates[name] = {"estimate": point, "ci_low": low, "ci_high": high}
    routing_rescue = estimates["route"]["ci_low"] > 0
    specificity_shuffle = estimates["shuffle"]["ci_low"] > 0
    specificity_query = estimates["query"]["ci_low"] > 0
    mediator_summary = {}
    for metric in ("a", "b", "rho", "cos_theta", "v_l_conditional"):
        column = f"sum_{metric}"
        if column not in stats:
            continue
        metric_seed = stats.assign(metric_mean=stats[column].astype(np.float64) / stats["count"].astype(np.float64))
        metric_original = metric_seed[metric_seed.arm == "original"].set_index("system_index").metric_mean.loc[systems]
        metric_arms = {
            arm: metric_seed[metric_seed.arm == arm].groupby("system_index", sort=True)["metric_mean"].mean().loc[systems]
            for arm in cfg["arms"]
        }
        point, low, high = _paired_bootstrap(
            metric_arms["true"].to_numpy() - metric_original.to_numpy(),
            int(cfg["formal_evaluation"]["bootstrap_replicates"]),
            int(cfg["formal_evaluation"]["bootstrap_seed"]) + 10 + len(mediator_summary),
        )
        mediator_summary[metric] = {
            "system_equal_arm_means": {
                "original": float(metric_original.mean()),
                **{arm: float(values.mean()) for arm, values in metric_arms.items()},
            },
            "true_minus_original": {"estimate": point, "ci_low": low, "ci_high": high},
            "hard_gate": False if metric == "a" else None,
        }
    result = {
        "schema_version": "1.0",
        "status": "ROUTING_RESCUE" if routing_rescue else "NO_ROUTING_RESCUE",
        "evidence_identity": cfg["evidence_identity"],
        "environment": environment, "scientific_unit": "physical_system",
        "systems": len(systems), "optimization_seeds_are_scientific_samples": False,
        "contrasts": estimates,
        "routing_rescue": bool(routing_rescue),
        "persistent_specificity": {
            "true_gt_shuffled": bool(specificity_shuffle),
            "true_gt_query_only": bool(specificity_query),
            "both": bool(specificity_shuffle and specificity_query),
        },
        "a_is_mediator_not_hard_gate": True,
        "mediators_and_supporting_coordinates": mediator_summary,
        "fixed_merge_order": list(range(shard_count)),
        "original_module_hashes": receipts[0]["original_module_hashes_after"],
    }
    stats_path = eval_root / "merged_sufficient_statistics.csv.gz"
    result_path = eval_root / "FINAL_RESULT.json"
    if stats_path.exists() or result_path.exists():
        raise RuntimeError("immutable final evaluation already exists; refusing overwrite")
    tmp = stats_path.with_name(stats_path.name + f".tmp.{os.getpid()}")
    stats.to_csv(tmp, index=False, compression="gzip"); os.replace(tmp, stats_path)
    result["statistics"] = str(stats_path.relative_to(root)); result["statistics_sha256"] = sha256(stats_path)
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    static = sub.add_parser("record-static-correctness")
    static.add_argument("--test-log", type=Path, required=True)
    sub.add_parser("host-ownership")
    equivalence = sub.add_parser("verify-two-host-equivalence")
    equivalence.add_argument("--worker_a-receipt", type=Path, required=True)
    equivalence.add_argument("--worker_b-receipt", type=Path, required=True)
    global_auth = sub.add_parser("verify-global-job-authentication")
    global_auth.add_argument("--worker_a-selection", type=Path, required=True)
    global_auth.add_argument("--worker_b-selection", type=Path, required=True)
    materialize = sub.add_parser("materialize-formal")
    materialize.add_argument("environment", choices=("articulated", "coupled"))
    shuffle_maps = sub.add_parser("freeze-formal-shuffle-maps")
    shuffle_maps.add_argument("environment", choices=("articulated", "coupled"))
    cache = sub.add_parser("cache")
    cache.add_argument("environment", choices=("articulated", "coupled")); cache.add_argument("split", choices=("train", "select", "formal"))
    cache.add_argument("--shard-index", type=int, required=True); cache.add_argument("--shard-count", type=int, required=True)
    cache.add_argument("--device", required=True); cache.add_argument("--arrays", type=Path); cache.add_argument("--manifest", type=Path)
    cache.add_argument("--correction", type=Path)
    merge = sub.add_parser("merge-cache")
    merge.add_argument("environment", choices=("articulated", "coupled")); merge.add_argument("split", choices=("train", "select", "formal")); merge.add_argument("--shard-count", type=int, required=True)
    cache_parity = sub.add_parser("cache-parity")
    cache_parity.add_argument("environment", choices=("articulated", "coupled")); cache_parity.add_argument("--device", required=True)
    formal_parity = sub.add_parser("formal-parity")
    formal_parity.add_argument("environment", choices=("articulated", "coupled")); formal_parity.add_argument("--device", required=True)
    train = sub.add_parser("train-job"); train.add_argument("job_id"); train.add_argument("--device", required=True)
    select_parser = sub.add_parser("select"); select_parser.add_argument("--environment", choices=("articulated", "coupled"), required=True)
    evaluate = sub.add_parser("evaluate-shard")
    evaluate.add_argument("environment", choices=("articulated", "coupled")); evaluate.add_argument("--shard-index", type=int, required=True)
    evaluate.add_argument("--shard-count", type=int, required=True); evaluate.add_argument("--device", required=True)
    final = sub.add_parser("merge-evaluation"); final.add_argument("environment", choices=("articulated", "coupled")); final.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze": result = freeze(args.root, args.config)
    elif args.command == "record-static-correctness": result = record_static_correctness(args.root, args.config, args.test_log)
    elif args.command == "host-ownership": result = write_host_ownership_receipt(args.root, args.config)
    elif args.command == "verify-two-host-equivalence": result = verify_two_host_equivalence(args.root, args.config, args.worker_a_receipt, args.worker_b_receipt)
    elif args.command == "verify-global-job-authentication": result = verify_global_job_authentication(args.root, args.config, args.worker_a_selection, args.worker_b_selection)
    elif args.command == "materialize-formal": result = materialize_formal_inputs(args.root, args.config, args.environment)
    elif args.command == "freeze-formal-shuffle-maps": result = freeze_formal_shuffle_maps(args.root, args.config, args.environment)
    elif args.command == "cache": result = extract_cache(args.root, args.config, args.environment, args.split, args.shard_index, args.shard_count, args.device, args.arrays, args.manifest, args.correction)
    elif args.command == "merge-cache": result = merge_cache(args.root, args.config, args.environment, args.split, args.shard_count)
    elif args.command == "cache-parity": result = run_cache_parity(args.root, args.config, args.environment, args.device)
    elif args.command == "formal-parity": result = run_formal_parity(args.root, args.config, args.environment, args.device)
    elif args.command == "train-job": result = train_job(args.root, args.config, args.job_id, args.device)
    elif args.command == "select": result = select_checkpoints(args.root, args.config, args.environment)
    elif args.command == "evaluate-shard": result = evaluate_shard(args.root, args.config, args.environment, args.shard_index, args.shard_count, args.device)
    elif args.command == "merge-evaluation": result = merge_evaluation(args.root, args.config, args.environment, args.shard_count)
    else: raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
