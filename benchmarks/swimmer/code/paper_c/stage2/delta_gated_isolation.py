"""Post-hoc Articulated delta-gated routing isolation for Paper C E3.

This module deliberately reuses the immutable feature caches produced by the
completed downstream-routing V1 run.  It writes only to the E3 namespace and
refuses local execution, alternate learning rates, seed selection, or formal
evaluation before all six select-only jobs are authenticated.
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

import numpy as np
import pandas as pd
import torch
from torch import nn

from paper_c.coupled_sled.formal_data import load_arrays
from paper_c.stage2 import routing_intervention as v1


STATUS_STATIC = "ARTICULATED_DELTA_GATED_REMOTE_STATIC_PASS"
STATUS_FROZEN = "ARTICULATED_DELTA_GATED_ISOLATION_FROZEN"
STATUS_JOB = "ARTICULATED_DELTA_GATED_TRAINING_COMPLETE"
STATUS_BARRIER = "ARTICULATED_DELTA_GATED_ALL_JOBS_AUTHENTICATED"
STATUS_EVAL = "ARTICULATED_DELTA_GATED_FORMAL_SHARD_COMPLETE"


class DeltaGatedAdapter(nn.Module):
    """A query-modulated route satisfying A(0,q)=0 for all parameters."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.delta_projection = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.query_gate = nn.Linear(latent_dim, hidden_dim, bias=True)
        self.output_projection = nn.Linear(hidden_dim, latent_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, delta_persistent: torch.Tensor, query_embedding: torch.Tensor) -> torch.Tensor:
        if delta_persistent.ndim != 2 or delta_persistent.shape != query_embedding.shape:
            raise ValueError("delta-persistent/query-embedding shape mismatch")
        delta_features = torch.nn.functional.gelu(self.delta_projection(delta_persistent))
        gate = torch.sigmoid(self.query_gate(query_embedding))
        return self.output_projection(delta_features * gate)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _cfg(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    if cfg.get("environment") != "articulated" or cfg.get("arms") != ["true", "shuffled"]:
        raise RuntimeError("E3 scope or arm order changed")
    opt = cfg["optimization"]
    if opt.get("learning_rate_grid") is not False or float(opt["learning_rate"]) != 1e-3:
        raise RuntimeError("E3 must use the single frozen learning rate 1e-3")
    if list(map(int, opt["seeds"])) != [86101, 86103, 86107]:
        raise RuntimeError("E3 optimization seeds changed")
    arch = cfg["architecture"]
    required = {
        "name": "delta_gated_persistent_only",
        "latent_dim": 64,
        "hidden_dim": 64,
        "delta_projection_bias": False,
        "output_projection_bias": False,
        "structural_invariant": "A(0,q)=0",
    }
    for key, value in required.items():
        if arch.get(key) != value:
            raise RuntimeError(f"frozen E3 architecture changed for {key}")
    return cfg


def _require_remote(root: Path, cfg: dict, device_name: str | None = None) -> dict:
    remote = cfg["remote_execution"]
    if platform.system() == "Darwin":
        raise RuntimeError("remote-only protocol: Mac execution is forbidden")
    if remote.get("require_linux") and platform.system() != "Linux":
        raise RuntimeError("E3 requires Linux")
    if os.environ.get("PAPER_C_REMOTE_EXECUTION") != "1":
        raise RuntimeError("PAPER_C_REMOTE_EXECUTION=1 is required")
    if os.environ.get("PAPER_C_REMOTE_HOST") != remote["logical_host"]:
        raise RuntimeError("E3 is frozen to logical host worker_b")
    sentinel = root / remote["sentinel"]
    payload = json.loads(sentinel.read_text())
    if payload.get("logical_host") != "worker_b" or payload.get("actual_hostname") != socket.gethostname():
        raise RuntimeError("worker_b sentinel does not match the actual host")
    if device_name is not None:
        device = torch.device(device_name)
        if device.type != "cuda" or device.index not in set(map(int, remote["allowed_gpu_ids"])):
            raise RuntimeError("E3 may use only CUDA devices 2 and 3")
    return {
        "logical_host": "worker_b",
        "actual_hostname": socket.gethostname(),
        "sentinel": str(sentinel.relative_to(root)),
        "sentinel_sha256": v1.sha256(sentinel),
    }


def _configure_device(cfg: dict, device: torch.device) -> dict:
    threads = int(cfg["remote_execution"]["cpu_threads_per_worker"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["remote_execution"]["gpu_memory_fraction_per_worker"]), device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
    return {"torch_threads": threads, "device": str(device)}


def _peak(device: torch.device) -> dict:
    result = {"peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if device.type == "cuda":
        result.update({
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
    return result


def validate_job_manifest(table: pd.DataFrame, cfg: dict) -> None:
    required = {"job_id", "environment", "arm", "seed", "learning_rate"}
    if not required.issubset(table.columns) or table.job_id.duplicated().any():
        raise RuntimeError("invalid E3 job manifest")
    expected = {
        ("articulated", arm, int(seed), 1e-3)
        for arm in ("true", "shuffled") for seed in cfg["optimization"]["seeds"]
    }
    observed = set(zip(
        table.environment.astype(str), table.arm.astype(str),
        table.seed.astype(int), table.learning_rate.astype(float),
    ))
    if observed != expected or len(table) != 6:
        raise RuntimeError("E3 job manifest is not the exact six-job product")


def _v1_cfg(root: Path, cfg: dict) -> dict:
    return json.loads((root / cfg["upstream"]["v1_config"]).read_text())


def _v1_cache(root: Path, cfg: dict, split: str) -> tuple[dict[str, np.ndarray], dict]:
    receipt_path = root / cfg["upstream"][f"{split}_cache_receipt"]
    receipt = json.loads(receipt_path.read_text())
    path = root / receipt["merged_cache"]
    if (
        receipt.get("status") != v1.STATUS_CACHE
        or receipt.get("environment") != "articulated"
        or receipt.get("split") != split
        or receipt.get("merged_sha256") != v1.sha256(path)
    ):
        raise RuntimeError(f"invalid immutable V1 {split} cache")
    cache = v1._load_npz(path)
    v1.validate_adapter_cache_dimensions(cache, _v1_cfg(root, cfg), "articulated")
    return cache, receipt


def _lr_provenance(root: Path, cfg: dict) -> dict:
    output = root / "runs/intervention/downstream_routing_v1/jobs"
    values: dict[float, list[float]] = {1e-4: [], 3e-4: [], 1e-3: []}
    receipts = {}
    for seed in cfg["optimization"]["seeds"]:
        for label, rate in (("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3)):
            path = output / f"articulated_true_s{int(seed)}_lr{label}" / "receipt.json"
            row = json.loads(path.read_text())
            if (
                row.get("status") != v1.STATUS_JOB
                or row.get("formal_outcomes_read") is not False
                or row.get("selection_data") != "immutable_select_only"
                or int(row.get("seed")) != int(seed)
                or float(row.get("learning_rate")) != rate
            ):
                raise RuntimeError("invalid V1 select-only LR provenance")
            values[rate].append(float(row["best_select_mse"]))
            receipts[str(path.relative_to(root))] = v1.sha256(path)
    means = {rate: float(np.mean(rows)) for rate, rows in values.items()}
    frozen = {float(key): float(value) for key, value in cfg["optimization"]["v1_mean_select_mse_by_lr"].items()}
    for rate in means:
        if not np.isclose(means[rate], frozen[rate], rtol=0.0, atol=1e-15):
            raise RuntimeError("frozen V1 select-MSE LR summary changed")
    if min(means, key=means.get) != float(cfg["optimization"]["learning_rate"]):
        raise RuntimeError("fixed E3 learning rate is not V1 select-only winner")
    return {"mean_select_mse_by_lr": {str(k): v for k, v in means.items()}, "receipt_hashes": receipts}


def record_static_correctness(root: Path, config_path: Path, test_log: Path) -> dict:
    root, config_path, test_log = root.resolve(), config_path.resolve(), test_log.resolve()
    cfg = _cfg(config_path)
    host = _require_remote(root, cfg)
    target = root / cfg["output_root"] / "dev/correctness/REMOTE_STATIC_CORRECTNESS_RECEIPT_worker_b.json"
    if target.exists():
        raise RuntimeError("E3 static correctness receipt already exists")
    text = test_log.read_text()
    if "failed" in text.lower() or " passed" not in text.lower():
        raise RuntimeError("E3 remote unit tests did not pass")
    sources = [
        config_path, root / cfg["protocol"], root / cfg["implementation"],
        root / cfg["tests"], root / cfg["job_manifest"],
    ]
    receipt = {
        "schema_version": "1.0", "status": STATUS_STATIC,
        "execution_host": host, "test_log": str(test_log.relative_to(root)),
        "test_log_sha256": v1.sha256(test_log),
        "source_hashes": {str(path.relative_to(root)): v1.sha256(path) for path in sources},
        "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def freeze(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _cfg(config_path)
    host = _require_remote(root, cfg)
    output = root / cfg["output_root"]
    target = output / "E3_FROZEN.json"
    if target.exists():
        raise RuntimeError("E3 is already frozen")
    static_path = output / "dev/correctness/REMOTE_STATIC_CORRECTNESS_RECEIPT_worker_b.json"
    static = json.loads(static_path.read_text())
    if static.get("status") != STATUS_STATIC or static.get("execution_host", {}).get("actual_hostname") != host["actual_hostname"]:
        raise RuntimeError("missing current-host E3 static correctness")
    for relative, expected in static["source_hashes"].items():
        if v1.sha256(root / relative) != expected:
            raise RuntimeError(f"E3 static correctness stale for {relative}")
    jobs = pd.read_csv(root / cfg["job_manifest"])
    validate_job_manifest(jobs, cfg)
    upstream_paths = [root / value for key, value in cfg["upstream"].items() if key != "train_arrays"]
    upstream_hashes = {str(path.relative_to(root)): v1.sha256(path) for path in upstream_paths}
    cache_hashes = {}
    original_hashes = None
    for split in ("train", "select", "formal"):
        _, receipt = _v1_cache(root, cfg, split)
        cache_hashes[split] = receipt["merged_sha256"]
        if original_hashes is None:
            original_hashes = receipt["original_module_hashes"]
        elif receipt["original_module_hashes"] != original_hashes:
            raise RuntimeError("V1 cache original-module hashes disagree")
    lr = _lr_provenance(root, cfg)
    source_hashes = dict(static["source_hashes"])
    source_hashes[str(static_path.relative_to(root))] = v1.sha256(static_path)
    receipt = {
        "schema_version": "1.0", "status": STATUS_FROZEN,
        "evidence_identity": cfg["evidence_identity"],
        "formal_results_already_known": True,
        "formal_outcomes_used_for_training_or_epoch_selection": False,
        "execution_host": host, "source_hashes": source_hashes,
        "upstream_receipt_hashes": upstream_hashes,
        "upstream_cache_hashes": cache_hashes,
        "original_module_hashes": original_hashes,
        "lr_provenance": lr,
        "jobs": 6, "arms": ["true", "shuffled"],
        "query_only_structurally_equals_original": True,
        "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_freeze(root: Path, cfg: dict) -> dict:
    path = root / cfg["output_root"] / "E3_FROZEN.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != STATUS_FROZEN or receipt.get("jobs") != 6:
        raise RuntimeError("invalid E3 freeze receipt")
    for relative, expected in receipt["source_hashes"].items():
        if v1.sha256(root / relative) != expected:
            raise RuntimeError(f"frozen E3 source changed: {relative}")
    for relative, expected in receipt["upstream_receipt_hashes"].items():
        if v1.sha256(root / relative) != expected:
            raise RuntimeError(f"frozen E3 upstream receipt changed: {relative}")
    return receipt


def _adapter_inputs(cache: dict[str, np.ndarray], arm: str, seed: int) -> np.ndarray:
    if arm == "true":
        return cache["delta_persistent"]
    if arm == "shuffled":
        donors = v1.cross_system_cell_permutation(cache, seed)
        return cache["delta_persistent"][donors]
    raise ValueError(arm)


@torch.no_grad()
def _mse(adapter: DeltaGatedAdapter, decoder: nn.Module, cache: dict[str, np.ndarray],
         delta_input: np.ndarray, device: torch.device, batch_size: int) -> float:
    adapter.eval()
    total, count = 0.0, 0
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
    cfg = _cfg(config_path); host = _require_remote(root, cfg, device_name); frozen = verify_freeze(root, cfg)
    jobs = pd.read_csv(root / cfg["job_manifest"]); validate_job_manifest(jobs, cfg)
    chosen = jobs[jobs.job_id == job_id]
    if len(chosen) != 1:
        raise ValueError(f"unknown E3 job {job_id}")
    job = chosen.iloc[0]; arm, seed = str(job.arm), int(job.seed)
    if float(job.learning_rate) != float(cfg["optimization"]["learning_rate"]):
        raise RuntimeError("job learning rate differs from frozen single rate")
    job_root = root / cfg["output_root"] / "jobs" / job_id
    outputs = [job_root / name for name in ("adapter_best.pt", "training_curve.csv", "receipt.json")]
    if any(path.exists() for path in outputs):
        raise RuntimeError("immutable E3 job output already exists")
    train_cache, train_receipt = _v1_cache(root, cfg, "train")
    select_cache, select_receipt = _v1_cache(root, cfg, "select")
    if "bayes_correction" in train_cache or "bayes_correction" in select_cache:
        raise RuntimeError("train/select cache contains forbidden Bayes correction")
    delta_train = _adapter_inputs(train_cache, arm, seed)
    delta_select = _adapter_inputs(select_cache, arm, seed)
    device = torch.device(device_name); limits = _configure_device(cfg, device)
    base_cfg = _v1_cfg(root, cfg)
    arrays = load_arrays(root / cfg["upstream"]["train_arrays"])
    model, _, before = v1._load_environment_model(root, base_cfg, "articulated", arrays, device)
    if before != frozen["original_module_hashes"] or before != train_receipt["original_module_hashes"] or before != select_receipt["original_module_hashes"]:
        raise RuntimeError("base model/cache/freeze hash mismatch")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    adapter = DeltaGatedAdapter().to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(cfg["optimization"]["learning_rate"]),
        weight_decay=float(cfg["optimization"]["weight_decay"]),
    )
    batch_size = int(cfg["optimization"]["batch_size"])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    bad_epochs, curve = 0, []
    for epoch in range(1, int(cfg["optimization"]["maximum_epochs"]) + 1):
        adapter.train()
        order = np.random.default_rng(v1._stable_seed("delta-gated-order", arm, seed, epoch)).permutation(len(delta_train))
        train_sum, train_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            delta = torch.from_numpy(delta_train[idx]).to(device)
            query = torch.from_numpy(train_cache["query_embedding"][idx]).to(device)
            base = torch.from_numpy(train_cache["predicted_latent_full"][idx]).to(device)
            target = torch.from_numpy(train_cache["normalized_target"][idx]).to(device)
            prediction = model.target_decoder(base + adapter(delta, query))
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(idx); train_count += len(idx)
        select_mse = _mse(adapter, model.target_decoder, select_cache, delta_select, device, batch_size)
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
        raise RuntimeError("E3 training produced no checkpoint")
    adapter.load_state_dict(best_state)
    query = torch.randn(17, 64, device=device)
    if not torch.equal(adapter(torch.zeros_like(query), query), torch.zeros_like(query)):
        raise RuntimeError("trained E3 adapter violates A(0,q)=0")
    after = v1.original_module_hashes(model)
    if after != before:
        raise RuntimeError("original model changed during E3 training")
    checkpoint = job_root / "adapter_best.pt"; curve_path = job_root / "training_curve.csv"
    _atomic_torch(checkpoint, adapter.state_dict()); _atomic_text(curve_path, pd.DataFrame(curve).to_csv(index=False))
    receipt = {
        "schema_version": "1.0", "status": STATUS_JOB, "job_id": job_id,
        "environment": "articulated", "arm": arm, "seed": seed,
        "learning_rate": float(cfg["optimization"]["learning_rate"]),
        "weight_decay": float(cfg["optimization"]["weight_decay"]),
        "best_epoch": best_epoch, "best_select_mse": best_loss,
        "selection_data": "immutable_select_only", "formal_outcomes_read": False,
        "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": v1.sha256(checkpoint),
        "curve": str(curve_path.relative_to(root)), "curve_sha256": v1.sha256(curve_path),
        "train_cache_sha256": train_receipt["merged_sha256"],
        "select_cache_sha256": select_receipt["merged_sha256"],
        "freeze_receipt_sha256": v1.sha256(root / cfg["output_root"] / "E3_FROZEN.json"),
        "original_module_hashes_before": before, "original_module_hashes_after": after,
        "structural_zero_verified_after_training": True,
        "execution_host": host, "resource_limits": limits, "peak_resource_usage": _peak(device),
    }
    _atomic_text(job_root / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def authenticate_jobs(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _cfg(config_path); _require_remote(root, cfg); verify_freeze(root, cfg)
    jobs = pd.read_csv(root / cfg["job_manifest"]); validate_job_manifest(jobs, cfg)
    target = root / cfg["output_root"] / "selection/ALL_6_JOBS_AUTHENTICATED.json"
    if target.exists():
        raise RuntimeError("E3 job barrier already exists")
    hashes, rows = {}, []
    for job in jobs.itertuples(index=False):
        path = root / cfg["output_root"] / "jobs" / job.job_id / "receipt.json"
        receipt = json.loads(path.read_text())
        expected = (str(job.job_id), str(job.arm), int(job.seed), float(job.learning_rate))
        observed = (receipt.get("job_id"), receipt.get("arm"), int(receipt.get("seed")), float(receipt.get("learning_rate")))
        if receipt.get("status") != STATUS_JOB or observed != expected or receipt.get("formal_outcomes_read") is not False:
            raise RuntimeError(f"invalid E3 job receipt {job.job_id}")
        checkpoint = root / receipt["checkpoint"]
        if v1.sha256(checkpoint) != receipt["checkpoint_sha256"] or receipt["original_module_hashes_before"] != receipt["original_module_hashes_after"]:
            raise RuntimeError(f"E3 job artifact changed {job.job_id}")
        hashes[str(path.relative_to(root))] = v1.sha256(path)
        rows.append({
            "environment": "articulated", "arm": job.arm, "seed": int(job.seed),
            "learning_rate": float(job.learning_rate), "epoch": int(receipt["best_epoch"]),
            "select_mse": float(receipt["best_select_mse"]), "checkpoint": receipt["checkpoint"],
            "checkpoint_sha256": receipt["checkpoint_sha256"], "job_receipt": str(path.relative_to(root)),
            "job_receipt_sha256": v1.sha256(path),
        })
    table = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    table_path = target.parent / "all_checkpoints.csv"
    _atomic_text(table_path, table.to_csv(index=False))
    receipt = {
        "schema_version": "1.0", "status": STATUS_BARRIER,
        "jobs": 6, "checkpoint_selection_across_seeds": False,
        "learning_rate_grid": False, "formal_outcomes_read": False,
        "table": str(table_path.relative_to(root)), "table_sha256": v1.sha256(table_path),
        "job_receipt_hashes": hashes,
        "freeze_receipt_sha256": v1.sha256(root / cfg["output_root"] / "E3_FROZEN.json"),
        "created_at_unix": time.time(),
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_barrier(root: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    path = root / cfg["output_root"] / "selection/ALL_6_JOBS_AUTHENTICATED.json"
    receipt = json.loads(path.read_text()); table_path = root / receipt["table"]
    if receipt.get("status") != STATUS_BARRIER or receipt.get("jobs") != 6 or receipt.get("formal_outcomes_read") is not False:
        raise RuntimeError("E3 training barrier is invalid")
    if receipt["table_sha256"] != v1.sha256(table_path):
        raise RuntimeError("E3 checkpoint table changed")
    table = pd.read_csv(table_path)
    if len(table) != 6 or set(table.arm) != {"true", "shuffled"}:
        raise RuntimeError("E3 checkpoint table has wrong scope")
    for row in table.itertuples(index=False):
        if v1.sha256(root / row.checkpoint) != row.checkpoint_sha256:
            raise RuntimeError("E3 checkpoint changed")
    return table, receipt


def _formal_donor(root: Path, cfg: dict, cache: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, str]:
    receipt_path = root / cfg["upstream"]["formal_shuffle_receipt"]
    receipt = json.loads(receipt_path.read_text())
    _, cache_receipt = _v1_cache(root, cfg, "formal")
    if receipt.get("status") != "FORMAL_GLOBAL_SHUFFLE_MAPS_FROZEN" or receipt.get("formal_cache_sha256") != cache_receipt["merged_sha256"]:
        raise RuntimeError("invalid V1 formal shuffle receipt")
    item = next((row for row in receipt["maps"] if int(row["seed"]) == int(seed)), None)
    if item is None:
        raise RuntimeError("missing formal donor map")
    path = root / item["path"]
    if v1.sha256(path) != item["sha256"]:
        raise RuntimeError("formal donor map changed")
    values = v1._load_npz(path); donor = values["donor_position"].astype(np.int64)
    if not np.array_equal(values["receiver_row_index"], cache["row_index"]):
        raise RuntimeError("formal donor receiver identity changed")
    if np.any(cache["system_index"][donor] == cache["system_index"]):
        raise RuntimeError("formal donor map is not cross-system")
    for key in ("anchor_index", "query_index", "candidate_identity"):
        if not np.array_equal(cache[key][donor], cache[key]):
            raise RuntimeError(f"formal donor map changed {key}")
    return donor, item["sha256"]


@torch.no_grad()
def _evaluate_positions(root: Path, cfg: dict, table: pd.DataFrame,
                        full_cache: dict[str, np.ndarray], positions: np.ndarray,
                        device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    positions = np.asarray(positions, dtype=np.int64)
    cache = {key: value[positions] for key, value in full_cache.items()}
    arrays = load_arrays(root / cfg["upstream"]["train_arrays"])
    model, _, before = v1._load_environment_model(root, _v1_cfg(root, cfg), "articulated", arrays, device)
    _, formal_receipt = _v1_cache(root, cfg, "formal")
    if before != formal_receipt["original_module_hashes"]:
        raise RuntimeError("formal base-model hash differs from cache")
    correction = cache.get("bayes_correction")
    if correction is None:
        raise RuntimeError("E3 formal mediator requires Bayes correction")
    rows: list[dict] = []

    def append(arm: str, seed: int, prediction: np.ndarray) -> None:
        metrics = v1._row_metrics(prediction, cache["prediction_anchor"], cache["normalized_target"], correction)
        for index in range(len(positions)):
            row = {
                "row_index": int(cache["row_index"][index]), "system_index": int(cache["system_index"][index]),
                "anchor_index": int(cache["anchor_index"][index]), "query_index": int(cache["query_index"][index]),
                "candidate_identity": str(cache["candidate_identity"][index]), "arm": arm, "seed": seed,
            }
            row.update({name: float(values[index]) for name, values in metrics.items()}); rows.append(row)

    append("original", -1, cache["prediction_original"])
    map_hashes = {}
    batch_size = int(cfg["optimization"]["batch_size"])
    for choice in table.itertuples(index=False):
        adapter = DeltaGatedAdapter().to(device)
        adapter.load_state_dict(torch.load(root / choice.checkpoint, map_location=device, weights_only=True)); adapter.eval()
        if choice.arm == "true":
            delta_input = cache["delta_persistent"]
        elif choice.arm == "shuffled":
            donor, digest = _formal_donor(root, cfg, full_cache, int(choice.seed))
            delta_input = full_cache["delta_persistent"][donor[positions]]; map_hashes[str(int(choice.seed))] = digest
        else:
            raise RuntimeError("unexpected E3 arm")
        prediction = np.empty_like(cache["prediction_original"], dtype=np.float32)
        for system in np.sort(np.unique(cache["system_index"])):
            local = np.flatnonzero(cache["system_index"] == system)
            for start in range(0, len(local), batch_size):
                idx = local[start:start + batch_size]
                delta = torch.from_numpy(delta_input[idx]).to(device)
                query = torch.from_numpy(cache["query_embedding"][idx]).to(device)
                base = torch.from_numpy(cache["predicted_latent_full"][idx]).to(device)
                prediction[idx] = model.target_decoder(base + adapter(delta, query)).cpu().numpy()
        append(str(choice.arm), int(choice.seed), prediction)
    after = v1.original_module_hashes(model)
    if before != after:
        raise RuntimeError("original model changed during E3 formal evaluation")
    frame = pd.DataFrame(rows).sort_values(["system_index", "row_index", "arm", "seed"]).reset_index(drop=True)
    return frame, v1._aggregate_formal_rows(frame), before, after, map_hashes


def formal_parity(root: Path, config_path: Path, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _cfg(config_path); _require_remote(root, cfg, device_name); verify_freeze(root, cfg)
    table, barrier = _verify_barrier(root, cfg); cache, formal_receipt = _v1_cache(root, cfg, "formal")
    systems = []
    available = np.sort(np.unique(cache["system_index"]))
    for shard in range(2):
        systems.append(int(available[available % 2 == shard][0]))
    receiver = np.flatnonzero(np.isin(cache["system_index"], systems))
    device = torch.device(device_name); _configure_device(cfg, device)
    _, direct, before, after, maps = _evaluate_positions(root, cfg, table, cache, receiver, device)
    parts = []
    for shard in range(2):
        subset = receiver[cache["system_index"][receiver] % 2 == shard]
        _, stats, b, a, m = _evaluate_positions(root, cfg, table, cache, subset, device)
        if b != a or m != maps:
            raise RuntimeError("E3 formal parity module/map mismatch")
        parts.append(stats)
    merged = pd.concat(parts, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(direct, merged, check_exact=True)
    target = root / cfg["output_root"] / "dev/parity/FORMAL_PARITY.json"
    if target.exists():
        raise RuntimeError("E3 formal parity receipt already exists")
    receipt = {
        "schema_version": "1.0", "status": "ARTICULATED_DELTA_GATED_ONE_TASK_TWO_SHARD_PARITY_PASS",
        "systems": systems, "independent_forward_count": 3,
        "statistics_dtype": "float64", "fixed_merge_order": [0, 1],
        "formal_cache_sha256": formal_receipt["merged_sha256"],
        "training_barrier_sha256": v1.sha256(root / cfg["output_root"] / "selection/ALL_6_JOBS_AUTHENTICATED.json"),
        "original_module_hashes_before": before, "original_module_hashes_after": after,
        "formal_shuffle_map_hashes": maps,
    }
    _atomic_text(target, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def evaluate_shard(root: Path, config_path: Path, shard_index: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _cfg(config_path); _require_remote(root, cfg, device_name); verify_freeze(root, cfg)
    if shard_index not in (0, 1):
        raise ValueError("E3 uses exactly two formal shards")
    table, barrier = _verify_barrier(root, cfg); cache, formal_receipt = _v1_cache(root, cfg, "formal")
    parity_path = root / cfg["output_root"] / "dev/parity/FORMAL_PARITY.json"
    parity = json.loads(parity_path.read_text())
    if (
        parity.get("status") != "ARTICULATED_DELTA_GATED_ONE_TASK_TWO_SHARD_PARITY_PASS"
        or parity.get("formal_cache_sha256") != formal_receipt["merged_sha256"]
        or parity.get("training_barrier_sha256") != v1.sha256(root / cfg["output_root"] / "selection/ALL_6_JOBS_AUTHENTICATED.json")
    ):
        raise RuntimeError("E3 formal parity is missing or stale")
    positions = np.flatnonzero(cache["system_index"] % 2 == shard_index)
    device = torch.device(device_name); limits = _configure_device(cfg, device)
    rows, stats, before, after, maps = _evaluate_positions(root, cfg, table, cache, positions, device)
    if maps != parity["formal_shuffle_map_hashes"]:
        raise RuntimeError("E3 formal maps differ from parity gate")
    output = root / cfg["output_root"] / "formal_eval"
    stats_path = output / f"stats_shard_{shard_index:02d}_of_02.csv.gz"
    receipt_path = output / f"receipt_shard_{shard_index:02d}_of_02.json"
    if stats_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable E3 formal shard exists")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stats_path.with_name(stats_path.name + f".tmp.{os.getpid()}")
    stats.to_csv(tmp, index=False, compression="gzip"); os.replace(tmp, stats_path)
    receipt = {
        "schema_version": "1.0", "status": STATUS_EVAL, "shard_index": shard_index,
        "systems": int(rows.system_index.nunique()), "rows_with_arms": int(len(rows)),
        "statistics": str(stats_path.relative_to(root)), "statistics_sha256": v1.sha256(stats_path),
        "statistics_dtype": "float64", "formal_cache_sha256": formal_receipt["merged_sha256"],
        "formal_parity_sha256": v1.sha256(parity_path),
        "original_module_hashes_before": before, "original_module_hashes_after": after,
        "formal_shuffle_map_hashes": maps, "resource_limits": limits, "peak_resource_usage": _peak(device),
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def merge_evaluation(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = _cfg(config_path); _require_remote(root, cfg); verify_freeze(root, cfg); _verify_barrier(root, cfg)
    output = root / cfg["output_root"] / "formal_eval"
    parts, receipts = [], []
    parity_path = root / cfg["output_root"] / "dev/parity/FORMAL_PARITY.json"
    for shard in range(2):
        receipt_path = output / f"receipt_shard_{shard:02d}_of_02.json"
        receipt = json.loads(receipt_path.read_text()); path = root / receipt["statistics"]
        if (
            receipt.get("status") != STATUS_EVAL or int(receipt.get("shard_index")) != shard
            or receipt.get("statistics_sha256") != v1.sha256(path)
            or receipt.get("formal_parity_sha256") != v1.sha256(parity_path)
        ):
            raise RuntimeError("invalid E3 formal shard")
        table = pd.read_csv(path)
        if np.any(table.system_index.to_numpy(np.int64) % 2 != shard):
            raise RuntimeError("E3 formal shard violates ownership")
        parts.append(table); receipts.append(receipt)
    stats = pd.concat(parts, ignore_index=True).sort_values(["system_index", "arm", "seed"]).reset_index(drop=True)
    if stats.duplicated(["system_index", "arm", "seed"]).any():
        raise RuntimeError("E3 formal systems duplicated")
    expected = {("original", -1)} | {(arm, seed) for arm in ("true", "shuffled") for seed in cfg["optimization"]["seeds"]}
    if set(zip(stats.arm.astype(str), stats.seed.astype(int))) != expected:
        raise RuntimeError("E3 formal arm/seed coverage differs")
    per_seed = stats.assign(mean_gain=stats.sum_gain.astype(np.float64) / stats["count"].astype(np.float64))
    systems = sorted(map(int, per_seed.system_index.unique()))
    if len(systems) != int(cfg["systems"]):
        raise RuntimeError("E3 formal system count differs from frozen 512")
    original = per_seed[per_seed.arm == "original"].set_index("system_index").mean_gain.loc[systems]
    true = per_seed[per_seed.arm == "true"].groupby("system_index", sort=True).mean_gain.mean().loc[systems]
    shuffled = per_seed[per_seed.arm == "shuffled"].groupby("system_index", sort=True).mean_gain.mean().loc[systems]
    values = {"route": true.to_numpy() - original.to_numpy(), "shuffle": true.to_numpy() - shuffled.to_numpy()}
    contrasts = {}
    for offset, (name, vector) in enumerate(values.items()):
        point, low, high = v1._paired_bootstrap(
            vector, int(cfg["formal_evaluation"]["bootstrap_replicates"]),
            int(cfg["formal_evaluation"]["bootstrap_seed"]) + offset,
        )
        contrasts[name] = {"estimate": point, "ci_low": low, "ci_high": high}
    per_seed_contrasts = {}
    for seed in cfg["optimization"]["seeds"]:
        t = per_seed[(per_seed.arm == "true") & (per_seed.seed == seed)].set_index("system_index").mean_gain.loc[systems]
        s = per_seed[(per_seed.arm == "shuffled") & (per_seed.seed == seed)].set_index("system_index").mean_gain.loc[systems]
        per_seed_contrasts[str(seed)] = {
            "true_minus_original": float((t - original).mean()),
            "true_minus_shuffled": float((t - s).mean()),
        }
    mediator = {}
    for metric in ("a", "b", "rho", "cos_theta", "v_l_conditional"):
        column = f"sum_{metric}"
        frame = stats.assign(value=stats[column].astype(np.float64) / stats["count"].astype(np.float64))
        base = frame[frame.arm == "original"].set_index("system_index").value.loc[systems]
        arm_values = {
            arm: frame[frame.arm == arm].groupby("system_index", sort=True).value.mean().loc[systems]
            for arm in ("true", "shuffled")
        }
        mediator[metric] = {
            "system_equal_arm_means": {"original": float(base.mean()), **{arm: float(x.mean()) for arm, x in arm_values.items()}},
            "true_minus_original": float((arm_values["true"] - base).mean()),
            "true_minus_shuffled": float((arm_values["true"] - arm_values["shuffled"]).mean()),
            "hard_gate": False if metric == "a" else None,
        }
    route = contrasts["route"]["ci_low"] > 0
    specificity = contrasts["shuffle"]["ci_low"] > 0
    status = "DELTA_GATED_PERSISTENT_SPECIFIC_RESCUE" if route and specificity else ("DELTA_GATED_RESCUE_WITHOUT_SPECIFICITY" if route else "NO_DELTA_GATED_RESCUE")
    result = {
        "schema_version": "1.0", "status": status,
        "evidence_identity": cfg["evidence_identity"], "environment": "articulated",
        "scientific_unit": "physical_system", "systems": len(systems),
        "optimization_seeds_are_scientific_samples": False,
        "single_frozen_architecture": "delta_gated_persistent_only",
        "structural_invariant": "A(0,q)=0",
        "query_only_structurally_equals_original": True,
        "contrasts": contrasts, "per_optimization_seed_contrasts": per_seed_contrasts,
        "route_rescue": bool(route), "persistent_specificity_vs_shuffled": bool(specificity),
        "mediators_and_supporting_coordinates": mediator,
        "fixed_merge_order": [0, 1],
        "original_module_hashes": receipts[0]["original_module_hashes_after"],
    }
    stats_path = output / "merged_sufficient_statistics.csv.gz"; result_path = output / "FINAL_RESULT.json"
    if stats_path.exists() or result_path.exists():
        raise RuntimeError("immutable E3 final result already exists")
    tmp = stats_path.with_name(stats_path.name + f".tmp.{os.getpid()}")
    stats.to_csv(tmp, index=False, compression="gzip"); os.replace(tmp, stats_path)
    result["statistics"] = str(stats_path.relative_to(root)); result["statistics_sha256"] = v1.sha256(stats_path)
    _atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path); parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("record-static-correctness"); static.add_argument("--test-log", type=Path, required=True)
    sub.add_parser("freeze")
    train = sub.add_parser("train-job"); train.add_argument("job_id"); train.add_argument("--device", required=True)
    sub.add_parser("authenticate-jobs")
    parity = sub.add_parser("formal-parity"); parity.add_argument("--device", required=True)
    evaluate = sub.add_parser("evaluate-shard"); evaluate.add_argument("--shard-index", type=int, required=True); evaluate.add_argument("--device", required=True)
    sub.add_parser("merge-evaluation")
    args = parser.parse_args(); root, config = args.root, args.config
    if args.command == "record-static-correctness": result = record_static_correctness(root, config, args.test_log)
    elif args.command == "freeze": result = freeze(root, config)
    elif args.command == "train-job": result = train_job(root, config, args.job_id, args.device)
    elif args.command == "authenticate-jobs": result = authenticate_jobs(root, config)
    elif args.command == "formal-parity": result = formal_parity(root, config, args.device)
    elif args.command == "evaluate-shard": result = evaluate_shard(root, config, args.shard_index, args.device)
    elif args.command == "merge-evaluation": result = merge_evaluation(root, config)
    else: raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
