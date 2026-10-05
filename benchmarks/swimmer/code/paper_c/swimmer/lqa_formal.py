"""Atomic, outcome-blind formal reference construction for Articulated LQA.

Each worker owns a deterministic modulo subset of physical-system indices.  A
completed system is the atomic resume unit: its named-axis rows are written
first and its hash-bound receipt is written last.  This module never generates
or reads learner losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .lqa_prospective import (
    _context_nuisance,
    _jsonable,
    _landmarks,
    _load_jepa,
    _response_bank,
    _response_jacobian_bank,
    accessibility_bank,
    banks,
    fixed_contexts,
    lqa_bank,
    particle_pool,
    prior_log_scale,
    reference_stream,
    sha256,
    system_pool,
    SwimmerModel,
)


def assigned_system_indices(start: int, end: int, worker_index: int, worker_count: int) -> list[int]:
    if start < 0 or end <= start:
        raise ValueError("formal system range must be non-empty and non-negative")
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("worker index must lie in [0, worker_count)")
    return [index for index in range(start, end) if index % worker_count == worker_index]


def formal_context_table(config: dict) -> pd.DataFrame:
    formal = config["formal"]
    salt = f"articulated-lqa-formal-v1|{int(formal['context_seed'])}"
    table = fixed_contexts(
        int(formal["pool_max"]),
        int(formal["nuisance_candidates_per_system"]),
        6,
        int(formal["pool_max"]),
        salt,
    )
    table = table.sort_values("system_index").reset_index(drop=True)
    if len(table) != int(formal["pool_max"]) or table.system_index.nunique() != len(table):
        raise RuntimeError("formal context table is not one-context-per-system")
    return table


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _completed(system_root: Path, invariant_hashes: dict[str, str]) -> bool:
    receipt_path = system_root / "formal_reference_receipt.json"
    rows_path = system_root / "formal_reference_rows.csv.gz"
    if not receipt_path.is_file() or not rows_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        receipt.get("status") == "ARTICULATED_LQA_FORMAL_SYSTEM_COMPLETE_OUTCOME_BLIND"
        and receipt.get("rows") == 36
        and receipt.get("rows_sha256") == sha256(rows_path)
        and receipt.get("source_hashes") == invariant_hashes
        and receipt.get("learner_outcomes_read") is False
    )


def _compute_system(
    root: Path,
    cfg: dict,
    base: dict,
    learner,
    norms: dict,
    training_receipt: dict,
    model: SwimmerModel,
    history: dict,
    query: dict,
    landmarks: np.ndarray,
    systems: np.ndarray,
    contexts: pd.DataFrame,
    system_index: int,
    device: torch.device,
    invariant_hashes: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    context = contexts.iloc[system_index]
    if int(context.system_index) != system_index:
        raise RuntimeError("formal context/system axis mismatch")
    realization = int(context.realization)
    history_index = int(context.history_index)
    theta = systems[system_index]
    ih, ie, iq, observed_h, _, _ = _context_nuisance(
        base,
        model,
        theta,
        history,
        query,
        landmarks,
        int(cfg["formal"]["context_seed"]),
        system_index,
        realization,
    )
    true_anchor = observed_h[history_index]
    particle_max = int(max(cfg["reference"]["particle_levels"]))
    outcome_levels = tuple(int(value) for value in cfg["reference"]["outcome_levels"])
    stream_values: dict[str, np.ndarray] = {}
    stream_ess: dict[str, list[float]] = {}
    stream_parity: dict[str, list[float]] = {}
    for stream_name, seeds in (
        ("ref_a", cfg["reference"]["ref_a_scramble_seeds"]),
        ("ref_b", cfg["reference"]["ref_b_scramble_seeds"]),
    ):
        values, esses, parities = [], [], []
        for seed in seeds:
            value, _, ess, parity, _ = reference_stream(
                base,
                model,
                history,
                query,
                landmarks,
                true_anchor,
                ih,
                ie,
                iq,
                history_index,
                particle_max,
                int(seed),
                outcome_levels,
                # Outcome QMC is bound to physical system identity, not worker
                # position, so redistributing workers cannot change results.
                int(seed) + 1_000_003 * system_index,
                norms["target_std"],
            )
            values.append(value)
            esses.append(ess)
            parities.append(parity)
        stream_values[stream_name] = np.asarray(values)
        stream_ess[stream_name] = esses
        stream_parity[stream_name] = parities

    geometry_particles = int(cfg["reference"]["geometry_particles"])
    geometry = particle_pool(
        geometry_particles,
        int(cfg["reference"]["geometry_scramble_seed"]),
        base["persistent_prior"],
    )
    hmean = _response_bank(model, geometry, ih, history, landmarks)[:, history_index]
    from paper_c.coupled_sled.posterior import posterior_from_observation

    posterior = posterior_from_observation(
        true_anchor,
        hmean,
        base["observation"]["sensor_std"],
        np.array([1.0]),
        np.array([1.0]),
    ).weights
    parameter_scale = prior_log_scale(base["persistent_prior"])
    candidate_means, candidate_jac = _response_jacobian_bank(
        model,
        geometry,
        ie,
        history,
        landmarks,
        parameter_scale,
        float(cfg["reference"]["finite_difference_log_step"]),
    )
    query_means, query_jac = _response_jacobian_bank(
        model,
        geometry,
        iq,
        query,
        landmarks,
        parameter_scale,
        float(cfg["reference"]["finite_difference_log_step"]),
    )
    accessibility, local = accessibility_bank(
        geometry,
        posterior,
        candidate_means,
        candidate_jac,
        query_jac,
        norms["target_std"],
        base["observation"]["sensor_std"],
        parameter_scale,
    )
    lqa, raw = lqa_bank(
        learner,
        norms,
        posterior,
        (history_index, true_anchor),
        ih,
        ie,
        iq,
        candidate_means,
        query_means,
        history,
        query,
        landmarks,
        device,
    )

    rows = []
    for candidate in range(6):
        for query_index in range(6):
            row = {
                "system_index": system_index,
                "realization": realization,
                "history_index": history_index,
                "candidate_index": candidate,
                "query_index": query_index,
                "accessibility": float(accessibility[candidate, query_index]),
                "local_value": float(local[candidate, query_index]),
                "lqa": float(lqa[candidate, query_index]),
                "raw_cka": float(raw[candidate, query_index]),
                "geometry_posterior_ess": float(1.0 / np.sum(posterior * posterior)),
            }
            for stream in ("ref_a", "ref_b"):
                values = stream_values[stream]
                row[f"{stream}_vb"] = float(values[:, -1, candidate, query_index].mean())
                row[f"{stream}_vb_se"] = float(values[:, -1, candidate, query_index].std(ddof=1) / np.sqrt(len(values)))
                row[f"{stream}_posterior_ess"] = float(np.mean(stream_ess[stream]))
                row[f"{stream}_dual_path_max_abs"] = float(np.max(stream_parity[stream]))
                for level_position, level in enumerate(outcome_levels):
                    row[f"{stream}_vb_n{level}"] = float(values[:, level_position, candidate, query_index].mean())
                for scramble in range(len(values)):
                    row[f"{stream}_vb_scramble{scramble}"] = float(values[scramble, -1, candidate, query_index])
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["candidate_index", "query_index"]).reset_index(drop=True)
    if len(table) != 36 or not np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("formal system did not produce a complete finite candidate-query grid")
    receipt = {
        "status": "ARTICULATED_LQA_FORMAL_SYSTEM_COMPLETE_OUTCOME_BLIND",
        "system_index": system_index,
        "theta_log_scales": theta.tolist(),
        "realization": realization,
        "history_index": history_index,
        "rows": 36,
        "elapsed_seconds": time.perf_counter() - started,
        "particle_max": particle_max,
        "geometry_particles": geometry_particles,
        "ref_a_scrambles": len(cfg["reference"]["ref_a_scramble_seeds"]),
        "ref_b_scrambles": len(cfg["reference"]["ref_b_scramble_seeds"]),
        "dual_path_max_abs": float(table[["ref_a_dual_path_max_abs", "ref_b_dual_path_max_abs"]].max().max()),
        "source_hashes": invariant_hashes,
        "checkpoint_hash": training_receipt["checkpoint_hashes"]["jepa"],
        "learner_outcomes_read": False,
        "raw_cka_used_for_pair_selection": False,
        "sealed_accessed": False,
    }
    return table, receipt


def run_worker(
    root: Path,
    config_path: Path,
    output_root: Path,
    start: int,
    end: int,
    worker_index: int,
    worker_count: int,
    device_name: str = "cpu",
) -> dict:
    root, config_path, output_root = Path(root), Path(config_path), Path(output_root)
    cfg = json.loads(config_path.read_text())
    base_path = root / cfg["base_config"]
    s0_path = root / cfg["s0_receipt"]
    base = json.loads(base_path.read_text())
    s0 = json.loads(s0_path.read_text())
    if s0["status"] != "S0_GO" or s0["config_sha256"] != sha256(base_path):
        raise RuntimeError("Articulated S0/base hash binding failed")
    if end > int(cfg["formal"]["pool_max"]):
        raise ValueError("formal range exceeds frozen pool maximum")
    if any(cfg["outcome_blindness"].values()):
        raise RuntimeError("formal reference construction must remain outcome blind")
    device = torch.device(device_name)
    learner, norms, training_receipt = _load_jepa(root, cfg, device)
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    systems = system_pool(int(cfg["formal"]["pool_max"]), int(cfg["formal"]["system_seed"]), base["persistent_prior"])
    contexts = formal_context_table(cfg)
    model = SwimmerModel(base["model"])
    invariant_hashes = {
        "config": sha256(config_path),
        "protocol": sha256(root / cfg["protocol"]),
        "base_config": sha256(base_path),
        "s0_receipt": sha256(s0_path),
        "jepa": training_receipt["checkpoint_hashes"]["jepa"],
        "normalization": training_receipt["checkpoint_hashes"]["normalization"],
        "implementation": sha256(Path(__file__)),
        "prospective_implementation": sha256(Path(__file__).with_name("lqa_prospective.py")),
    }
    assigned = assigned_system_indices(start, end, worker_index, worker_count)
    completed, skipped = [], []
    worker_started = time.perf_counter()
    for system_index in assigned:
        system_root = output_root / "systems" / f"system_{system_index:04d}"
        if _completed(system_root, invariant_hashes):
            skipped.append(system_index)
            continue
        table, receipt = _compute_system(
            root,
            cfg,
            base,
            learner,
            norms,
            training_receipt,
            model,
            history,
            query,
            landmarks,
            systems,
            contexts,
            system_index,
            device,
            invariant_hashes,
        )
        rows_path = system_root / "formal_reference_rows.csv.gz"
        _atomic_csv(rows_path, table)
        receipt["rows_sha256"] = sha256(rows_path)
        _atomic_text(system_root / "formal_reference_receipt.json", json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
        completed.append(system_index)
        print(json.dumps({"worker": worker_index, "completed_system": system_index, "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    worker_receipt = {
        "status": "ARTICULATED_LQA_FORMAL_WORKER_COMPLETE_OUTCOME_BLIND",
        "host": socket.gethostname(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "range": [start, end],
        "assigned": assigned,
        "completed_now": completed,
        "skipped_verified": skipped,
        "elapsed_seconds": time.perf_counter() - worker_started,
        "source_hashes": invariant_hashes,
        "learner_outcomes_read": False,
    }
    _atomic_text(output_root / "workers" / f"worker_{worker_index:03d}_of_{worker_count:03d}.json", json.dumps(_jsonable(worker_receipt), indent=2, sort_keys=True) + "\n")
    return worker_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=512)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    result = run_worker(
        args.root,
        args.config,
        args.output_root,
        args.start,
        args.end,
        args.worker_index,
        args.worker_count,
        args.device,
    )
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
