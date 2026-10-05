"""Bounded masked-GRU cross-architecture replication engineering.

This stage implements only frozen input binding, base training, immutable-select
checkpoint selection, and competence eligibility.  It deliberately exposes no
formal-evaluation command; a separate authorization barrier is required before
later formal consumers can be implemented or invoked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.formal_data import load_arrays
from paper_c.coupled_sled.learner import (
    MASKED_GRU_ARCHITECTURE_TAG,
    MaskedGRUPersistentAggregator,
    _loader,
    _normalized,
    _normalizers,
    _train_jepa,
    build_persistent_jepa,
    tagged_checkpoint_payload,
)
from paper_c.coupled_sled.manifests import load_spec


STATUS_FREEZE = "MASKED_GRU_CROSS_ARCHITECTURE_FROZEN"
STATUS_COMPETENT = "MASKED_GRU_BASE_COMPETENT"
STATUS_UNDERIDENTIFIED = "MASKED_GRU_BASE_UNDERIDENTIFIED"
STATUS_BARRIER = "MASKED_GRU_FORMAL_AUTHORIZATION_BARRIER"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def load_cfg(config_path: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("aggregator_type") != "masked_gru" or cfg.get("architecture_tag") != MASKED_GRU_ARCHITECTURE_TAG:
        raise RuntimeError("masked-GRU architecture identity changed")
    if cfg["select_split"] != {
        "salt": 84501,
        "checkpoint_fraction": 0.5,
        "assignment": "exact_hash_order_half_split",
        "checkpoint_tie_break": "earliest_epoch",
    }:
        raise RuntimeError("masked-GRU select responsibility split changed")
    if cfg["adapter"].get("fits") != 12 or cfg["adapter"].get("arms") != ["true", "shuffled"]:
        raise RuntimeError("masked-GRU adapter schedule changed")
    return cfg


def _job_table(root: Path, cfg: dict) -> pd.DataFrame:
    jobs = pd.read_csv(root / cfg["job_manifest"])
    required = {"job_id", "environment", "base_seed", "logical_host", "gpu_id"}
    if not required.issubset(jobs.columns) or jobs.job_id.duplicated().any() or len(jobs) != 4:
        raise RuntimeError("masked-GRU base job manifest is not the exact four-job schedule")
    expected = {
        (environment, int(seed))
        for environment, item in cfg["environments"].items()
        for seed in item["base_seeds"]
    }
    observed = set(zip(jobs.environment.astype(str), jobs.base_seed.astype(int)))
    if observed != expected:
        raise RuntimeError("masked-GRU environment/base-seed schedule changed")
    if set(jobs.logical_host.astype(str)) != {"worker_b"} or set(jobs.gpu_id.astype(int)) != {2, 3}:
        raise RuntimeError("masked-GRU remote allocation changed")
    return jobs.sort_values(["environment", "base_seed"]).reset_index(drop=True)


def _require_remote(root: Path, cfg: dict, logical_host: str | None = None,
                    device_name: str | None = None) -> dict:
    if platform.system() != "Linux" or os.environ.get("PAPER_C_REMOTE_EXECUTION") != "1":
        raise RuntimeError("masked-GRU result-bearing commands are remote-Linux only")
    actual_logical = os.environ.get("PAPER_C_REMOTE_HOST")
    if actual_logical not in cfg["execution"]["allowed_hosts"]:
        raise RuntimeError("masked-GRU logical host is not allowed")
    if logical_host is not None and actual_logical != logical_host:
        raise RuntimeError("masked-GRU job was dispatched to the wrong logical host")
    sentinel = root / cfg["execution"]["sentinel_template"].format(logical_host=actual_logical)
    payload = json.loads(sentinel.read_text())
    if payload.get("logical_host") != actual_logical or payload.get("actual_hostname") != socket.gethostname():
        raise RuntimeError("masked-GRU remote sentinel does not match the current host")
    if device_name is not None:
        device = torch.device(device_name)
        if device.type != "cuda" or device.index not in set(map(int, cfg["execution"]["allowed_gpu_ids"])):
            raise RuntimeError("masked-GRU job is using an unauthorized GPU")
    return {
        "logical_host": actual_logical,
        "actual_hostname": socket.gethostname(),
        "sentinel": str(sentinel.relative_to(root)),
        "sentinel_sha256": sha256(sentinel),
    }


def _environment_config(root: Path, cfg: dict, environment: str, base_seed: int) -> dict:
    item = cfg["environments"][environment]
    if int(base_seed) not in set(map(int, item["base_seeds"])):
        raise RuntimeError("base seed is outside the frozen masked-GRU schedule")
    if environment == "articulated":
        result = json.loads((root / item["training_config"]).read_text())
        result["learner_seed"] = int(base_seed)
    elif environment == "coupled":
        base = load_spec(root / item["base_spec"])
        formal = load_spec(root / item["formal_config"])
        result = copy.deepcopy(base["learner_development"])
        result.update(formal["learner"])
        result["seed"] = int(base_seed)
    else:
        raise ValueError(environment)
    result["aggregator_type"] = "masked_gru"
    if int(result["batch_size"]) != int(item["base_batch_size"]):
        raise RuntimeError("environment-specific canonical batch size changed")
    return result


def exact_system_half_split(system_index: np.ndarray, environment: str, salt: int) -> tuple[np.ndarray, np.ndarray]:
    systems = np.asarray(sorted(np.unique(system_index).astype(int)), dtype=np.int64)
    if len(systems) < 2 or len(systems) % 2:
        raise RuntimeError("immutable-select systems cannot be split into exact halves")
    ordered = sorted(
        systems.tolist(),
        key=lambda system: hashlib.sha256(f"{salt}|{environment}|{system}".encode()).hexdigest(),
    )
    checkpoint_systems = np.asarray(ordered[: len(ordered) // 2], dtype=np.int64)
    eligibility_systems = np.asarray(ordered[len(ordered) // 2 :], dtype=np.int64)
    if np.intersect1d(checkpoint_systems, eligibility_systems).size:
        raise AssertionError("select responsibility halves overlap")
    return checkpoint_systems, eligibility_systems


def _subset(arrays, positions: np.ndarray):
    return type(arrays)(**{
        field: np.asarray(getattr(arrays, field)[positions])
        for field in arrays.__dataclass_fields__
    })


def _system_positions(arrays, systems: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(arrays.system_index, systems))


def _bootstrap(values: np.ndarray, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("invalid system-level competence values")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "systems": int(len(values)),
    }


@torch.no_grad()
def _eligibility_values(model, arrays, norms: dict, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    history, mask, query, target = _normalized(arrays, norms)
    model_loss, trivial_loss, masked_loss = [], [], []
    for start in range(0, len(history), batch_size):
        stop = start + batch_size
        h = torch.from_numpy(history[start:stop]).to(device)
        m = torch.from_numpy(mask[start:stop]).to(device)
        q = torch.from_numpy(query[start:stop]).to(device)
        y = torch.from_numpy(target[start:stop]).to(device)
        prediction = model(h, m, q)[0]
        masked = m.clone()
        masked[:, 1] = 0
        masked_prediction = model(h, masked, q)[0]
        model_loss.append(torch.mean((prediction - y) ** 2, dim=1).cpu().numpy())
        masked_loss.append(torch.mean((masked_prediction - y) ** 2, dim=1).cpu().numpy())
        trivial_loss.append(torch.mean(y ** 2, dim=1).cpu().numpy())
    model_loss = np.concatenate(model_loss).astype(np.float64)
    masked_loss = np.concatenate(masked_loss).astype(np.float64)
    trivial_loss = np.concatenate(trivial_loss).astype(np.float64)
    systems = np.asarray(sorted(np.unique(arrays.system_index).astype(int)), dtype=np.int64)
    trivial_advantage, history_advantage = [], []
    k2 = arrays.history_mask[:, 1].astype(bool)
    for system in systems:
        rows = arrays.system_index == system
        k2_rows = rows & k2
        if not np.any(k2_rows):
            raise RuntimeError("eligibility system has no observed K2 rows")
        trivial_advantage.append(np.mean(trivial_loss[rows] - model_loss[rows], dtype=np.float64))
        history_advantage.append(np.mean(masked_loss[k2_rows] - model_loss[k2_rows], dtype=np.float64))
    if len(trivial_advantage) != len(systems) or len(history_advantage) != len(systems):
        raise AssertionError("competence statistics are not one value per physical system")
    return np.asarray(trivial_advantage), np.asarray(history_advantage)


def _normalization_equal(left: Path, right_values: dict[str, np.ndarray]) -> bool:
    with np.load(left, allow_pickle=False) as expected:
        return expected.files == list(right_values) and all(
            np.array_equal(expected[name], right_values[name]) for name in expected.files
        )


def _input_paths(root: Path, cfg: dict) -> list[Path]:
    result = [
        root / cfg["protocol"], root / cfg["probe_protocol"], root / cfg["implementation"],
        root / cfg["tests"], root / cfg["job_manifest"],
    ]
    result.extend(root / value for value in cfg["source_files"])
    for environment, item in cfg["environments"].items():
        for key in ("train_arrays", "select_arrays", "normalization"):
            result.append(root / item[key])
        if environment == "articulated":
            result.append(root / item["training_config"])
        else:
            result.extend((root / item["base_spec"], root / item["formal_config"], root / item["training_receipt"]))
    return result


def freeze(root: Path, config_path: Path, authorize_reviewed_spec: bool) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = load_cfg(config_path)
    if not authorize_reviewed_spec:
        raise RuntimeError("freeze requires explicit --authorize-reviewed-spec")
    host = _require_remote(root, cfg)
    jobs = _job_table(root, cfg)
    probe_path = root / cfg["probe_protocol"]
    if sha256(probe_path) != cfg["probe_protocol_sha256"]:
        raise RuntimeError("supporting-probe protocol hash changed")
    paths = _input_paths(root, cfg)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    output = root / cfg["output_root"]
    receipt_path = output / "FREEZE_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError("masked-GRU freeze receipt already exists")
    trivial_hashes = {}
    for environment, item in cfg["environments"].items():
        with np.load(root / item["normalization"], allow_pickle=False) as norms:
            target_dim = int(norms["target_mean"].shape[0])
        path = output / "frozen" / f"{environment}_trivial_predictor.npz"
        _atomic_npz(path, prediction_normalized=np.zeros(target_dim, dtype=np.float32))
        trivial_hashes[str(path.relative_to(root))] = sha256(path)
    receipt = {
        "schema_version": "1.0", "status": STATUS_FREEZE, "frozen_at_unix": time.time(),
        "host": host, "config": cfg, "config_sha256": sha256(config_path),
        "input_hashes": {str(path.relative_to(root)): sha256(path) for path in paths},
        "trivial_predictor_hashes": trivial_hashes,
        "job_rows": jobs.to_dict(orient="records"),
        "formal_outcomes_read": False, "formal_evaluation_implemented": False,
        "protected_scope_2_touched": False,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_freeze(root: Path, config_path: Path, cfg: dict) -> dict:
    receipt_path = root / cfg["output_root"] / "FREEZE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != STATUS_FREEZE or receipt.get("config_sha256") != sha256(config_path):
        raise RuntimeError("masked-GRU freeze receipt is absent or stale")
    if receipt.get("config") != cfg:
        raise RuntimeError("masked-GRU config changed after freeze")
    for relative, expected in {**receipt["input_hashes"], **receipt["trivial_predictor_hashes"]}.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"frozen masked-GRU input changed: {relative}")
    return receipt


def _configure_training(seed: int, device: torch.device, threads: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before launch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _pipeline_root(root: Path, cfg: dict, environment: str, base_seed: int) -> Path:
    return root / cfg["output_root"] / "base_learners" / f"{environment}_s{base_seed}"


def train_base(root: Path, config_path: Path, environment: str, base_seed: int, device_name: str) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = load_cfg(config_path)
    verify_freeze(root, config_path, cfg)
    jobs = _job_table(root, cfg)
    row = jobs[(jobs.environment == environment) & (jobs.base_seed.astype(int) == int(base_seed))]
    if len(row) != 1:
        raise RuntimeError("masked-GRU base job is outside the frozen schedule")
    job = row.iloc[0]
    host = _require_remote(root, cfg, str(job.logical_host), device_name)
    device = torch.device(device_name)
    _configure_training(int(base_seed), device, int(cfg["execution"]["threads_per_worker"]))
    item = cfg["environments"][environment]
    train = load_arrays(root / item["train_arrays"])
    select = load_arrays(root / item["select_arrays"])
    norms = _normalizers(train)
    if not _normalization_equal(root / item["normalization"], norms):
        raise RuntimeError("recomputed normalization differs from the canonical arrays")
    checkpoint_systems, eligibility_systems = exact_system_half_split(
        select.system_index, environment, int(cfg["select_split"]["salt"]),
    )
    checkpoint_rows = _system_positions(select, checkpoint_systems)
    eligibility_rows = _system_positions(select, eligibility_systems)
    checkpoint_select = _subset(select, checkpoint_rows)
    eligibility_select = _subset(select, eligibility_rows)
    model_cfg = _environment_config(root, cfg, environment, int(base_seed))
    dimensions = {
        "history_dim": int(train.history.shape[2]),
        "query_dim": int(train.query_action.shape[1]),
        "target_dim": int(train.target.shape[1]),
    }
    model = build_persistent_jepa(
        dimensions["history_dim"], dimensions["query_dim"], dimensions["target_dim"], model_cfg,
        expected_architecture_tag=MASKED_GRU_ARCHITECTURE_TAG,
    ).to(device)
    aggregator = model.aggregator
    if not isinstance(aggregator, MaskedGRUPersistentAggregator):
        raise RuntimeError("factory did not construct the frozen masked-GRU aggregator")
    if aggregator.hidden_dim != int(item["gru_hidden_dim"]):
        raise RuntimeError("masked-GRU hidden width differs from the frozen integer width")
    train_loader = _loader(train, norms, int(model_cfg["batch_size"]), True)
    checkpoint_loader = _loader(checkpoint_select, norms, int(model_cfg["batch_size"]), False)
    curve, best_select_mse = _train_jepa(model, train_loader, checkpoint_loader, model_cfg, device)
    finite = bool(np.isfinite([entry["selection_mse"] for entry in curve]).all())
    minimum_epoch = len(curve) >= int(model_cfg["minimum_epochs"])
    model.eval()
    trivial_values, history_values = _eligibility_values(
        model, eligibility_select, norms, int(model_cfg["batch_size"]), device,
    )
    replicates = int(cfg["competence"]["bootstrap_replicates"])
    bootstrap_seed = int(cfg["competence"]["bootstrap_seed"])
    trivial = _bootstrap(trivial_values, replicates, bootstrap_seed)
    history = _bootstrap(history_values, replicates, bootstrap_seed)
    competent = finite and minimum_epoch and trivial["ci_low"] > 0 and history["ci_low"] > 0
    out = _pipeline_root(root, cfg, environment, int(base_seed))
    checkpoint = out / "masked_gru_frozen.pt"
    _atomic_torch(checkpoint, tagged_checkpoint_payload(model, dimensions))
    normalization = out / "train_only_normalization.npz"
    _atomic_npz(normalization, **norms)
    curve_path = out / "training_curve.csv"
    out.mkdir(parents=True, exist_ok=True)
    curve_tmp = curve_path.with_name(curve_path.name + f".tmp.{os.getpid()}")
    pd.DataFrame(curve).to_csv(curve_tmp, index=False)
    os.replace(curve_tmp, curve_path)
    receipt = {
        "schema_version": "1.0",
        "status": STATUS_COMPETENT if competent else STATUS_UNDERIDENTIFIED,
        "environment": environment, "base_seed": int(base_seed), "host": host,
        "architecture_tag": model.architecture_tag,
        "dimensions": dimensions,
        "aggregator_parameter_count": aggregator.parameter_count,
        "checkpoint_selection_systems": checkpoint_systems.tolist(),
        "competence_eligibility_systems": eligibility_systems.tolist(),
        "select_halves_overlap": 0,
        "best_epoch": int(np.argmin([entry["selection_mse"] for entry in curve]) + 1),
        "best_checkpoint_select_mse": float(best_select_mse),
        "optimization_finite": finite, "minimum_epoch_reached": minimum_epoch,
        "trivial_risk_reduction": trivial, "same_row_history_use_gain": history,
        "checkpoint_sha256": sha256(checkpoint), "normalization_sha256": sha256(normalization),
        "training_curve_sha256": sha256(curve_path),
        "config_sha256": sha256(config_path),
        "freeze_receipt_sha256": sha256(root / cfg["output_root"] / "FREEZE_RECEIPT.json"),
        "formal_outcomes_read": False, "formal_outcome_authorized": False,
        "protected_scope_2_touched": False,
    }
    _atomic_text(out / "BASE_COMPETENCE_RECEIPT.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def authorize_formal(root: Path, config_path: Path) -> dict:
    """Create a barrier only; this module still has no formal evaluation path."""
    root, config_path = root.resolve(), config_path.resolve()
    cfg = load_cfg(config_path)
    verify_freeze(root, config_path, cfg)
    host = _require_remote(root, cfg)
    jobs = _job_table(root, cfg)
    receipts = {}
    for row in jobs.itertuples(index=False):
        base_root = _pipeline_root(root, cfg, str(row.environment), int(row.base_seed))
        receipt_path = base_root / "BASE_COMPETENCE_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("formal_outcomes_read") is not False or receipt.get("formal_outcome_authorized") is not False:
            raise RuntimeError("base receipt crossed the pre-formal boundary")
        if list(base_root.glob("formal*")):
            raise RuntimeError("pre-existing unbound formal artifact found")
        receipts[f"{row.environment}:{int(row.base_seed)}"] = {
            "status": receipt["status"], "receipt": str(receipt_path.relative_to(root)),
            "receipt_sha256": sha256(receipt_path), "checkpoint_sha256": receipt["checkpoint_sha256"],
        }
    art = [value for key, value in receipts.items() if key.startswith("articulated:")]
    art_ready = len(art) == 2 and all(value["status"] == STATUS_COMPETENT for value in art)
    coupled = [value for key, value in receipts.items() if key.startswith("coupled:")]
    payload = {
        "schema_version": "1.0", "status": STATUS_BARRIER,
        "created_at_unix": time.time(), "host": host,
        "articulated_primary_authorized": art_ready,
        "coupled_supporting_authorized": [value["status"] == STATUS_COMPETENT for value in coupled],
        "receipts": receipts,
        "formal_evaluation_implemented": False,
        "formal_outcomes_read": False,
        "claim_if_articulated_not_ready": "UNDERIDENTIFIED",
    }
    target = root / cfg["output_root"] / "FORMAL_AUTHORIZATION_BARRIER.json"
    if target.exists():
        raise RuntimeError("formal authorization barrier already exists")
    _atomic_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def self_test(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    cfg = load_cfg(config_path)
    jobs = _job_table(root, cfg)
    expected = {"articulated": (128, 160, 64, 149504), "coupled": (64, 80, 32, 37632)}
    counts = {}
    for environment, (embedding, hidden, persistent, count) in expected.items():
        module = MaskedGRUPersistentAggregator(embedding, persistent)
        if module.hidden_dim != hidden or module.parameter_count != count:
            raise RuntimeError("masked-GRU integer width or parameter count changed")
        counts[environment] = module.parameter_count
    systems = np.repeat(np.arange(8), 3)
    left, right = exact_system_half_split(systems, "toy", 84501)
    if len(left) != len(right) or np.intersect1d(left, right).size or set(np.r_[left, right]) != set(range(8)):
        raise RuntimeError("exact system-half split failed")
    return {
        "status": "MASKED_GRU_LOCAL_STATIC_PASS", "jobs": len(jobs),
        "adapter_fits": int(cfg["adapter"]["fits"]), "parameter_counts": counts,
        "formal_evaluation_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--authorize-reviewed-spec", action="store_true")
    train_parser = sub.add_parser("train-base")
    train_parser.add_argument("environment", choices=("articulated", "coupled"))
    train_parser.add_argument("base_seed", type=int)
    train_parser.add_argument("--device", required=True)
    sub.add_parser("authorize-formal")
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.root, args.config, args.authorize_reviewed_spec)
    elif args.command == "train-base":
        result = train_base(args.root, args.config, args.environment, args.base_seed, args.device)
    elif args.command == "authorize-formal":
        result = authorize_formal(args.root, args.config)
    else:
        result = self_test(args.root, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
