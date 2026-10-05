"""Frozen Stage 1B Bayes-correction alignment for Coupled and Articulated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_c.coupled_sled.learner import _normalized
from paper_c.coupled_sled.learner_data import _positions
from paper_c.coupled_sled.manifests import load_spec, sobol_systems
from paper_c.coupled_sled.posterior import combine_independent_posteriors, lognormal_quadrature
from paper_c.coupled_sled.waveforms import history_probe_bank, query_bank
from paper_c.swimmer.lqa_prospective import SwimmerModel, _landmarks, _response_bank, banks, particle_pool
from paper_c.swimmer.model import InitialState
from paper_c.stage1 import symptom_localization as stage1a


KEYS = ["system_index", "realization", "history_index", "candidate_index", "query_index"]
STATUS_FROZEN = "STAGE1B_BAYES_ALIGNMENT_MANIFEST_FROZEN"
STATUS_SHARD = "STAGE1B_BAYES_ALIGNMENT_SHARD_COMPLETE"


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
    table.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def alignment_metrics(delta: np.ndarray, correction: np.ndarray) -> dict[str, np.ndarray]:
    delta = np.asarray(delta, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if delta.shape != correction.shape or delta.ndim != 2:
        raise ValueError("delta and correction must be aligned matrices")
    delta_norm = np.linalg.norm(delta, axis=1)
    correction_norm = np.linalg.norm(correction, axis=1)
    denominator = delta_norm * correction_norm
    cosine = np.divide(
        np.einsum("ij,ij->i", delta, correction), denominator,
        out=np.full(len(delta), np.nan), where=denominator > 0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    rho = np.divide(delta_norm, correction_norm, out=np.full(len(delta), np.nan), where=correction_norm > 0)
    return {
        "delta_l_norm": delta_norm,
        "r_b_norm": correction_norm,
        "cos_theta": cosine,
        "rho": rho,
        "a": rho * cosine,
        "b": rho * np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine)),
        "v_l_conditional": 2.0 * np.einsum("ij,ij->i", correction, delta) - delta_norm * delta_norm,
    }


def initial_state_from_features(values: np.ndarray) -> InitialState:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (8,):
        raise ValueError("Articulated initial-state feature must have eight components")
    qpos = np.zeros(5, dtype=np.float64)
    qpos[2:] = values[:3]
    return InitialState(qpos=qpos, qvel=values[3:].copy())


def _source_paths(root: Path, cfg: dict) -> dict[str, Path]:
    stage1_root = root / cfg["stage1a_root"]
    paths = {
        "config": root / "configs/stage1_bayes_alignment_v1.json",
        "protocol": root / cfg["protocol"],
        "implementation": Path(__file__).resolve(),
        "stage1a_implementation": Path(stage1a.__file__).resolve(),
        "stage1a_freeze": stage1_root / "feature_manifest_frozen.json",
        "coupled_rows": stage1_root / "frozen/coupled_stage1_rows.csv.gz",
        "articulated_rows": stage1_root / "frozen/articulated_stage1_rows.csv.gz",
        "coupled_vectors": stage1_root / "merged/coupled/layer_deltas.npz",
        "articulated_vectors": stage1_root / "merged/articulated/layer_deltas.npz",
        "coupled_reference_config": root / "configs/coupled_theory_validation_p3r_v3.json",
        "articulated_reference_config": root / "configs/articulated_lqa_prospective_v1.json",
    }
    paths.update({f"coupled_{k}": v for k, v in stage1a._coupled_paths(root).items()})
    paths.update({f"articulated_{k}": v for k, v in stage1a._articulated_paths(root).items()})
    return paths


def freeze(root: Path, config_path: Path, output_root: Path) -> dict:
    root, config_path, output_root = Path(root).resolve(), Path(config_path).resolve(), Path(output_root).resolve()
    receipt_path = output_root / "reference_manifest_frozen.json"
    if receipt_path.exists():
        raise RuntimeError("Stage 1B manifest already frozen; refusing overwrite")
    cfg = json.loads(config_path.read_text())
    paths = _source_paths(root, cfg)
    paths["config"] = config_path
    missing = [str(value) for value in paths.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage 1B inputs: {missing}")
    stage1_freeze = json.loads(paths["stage1a_freeze"].read_text())
    if stage1_freeze.get("status") != stage1a.STATUS_FROZEN:
        raise RuntimeError("Stage 1A freeze is not valid")
    for env in ("coupled", "articulated"):
        if sha256(paths[f"{env}_rows"]) != stage1_freeze["environments"][env]["row_manifest_sha256"]:
            raise RuntimeError(f"{env} Stage 1A row-manifest hash mismatch")
        vectors = np.load(paths[f"{env}_vectors"], allow_pickle=False)
        expected = stage1_freeze["environments"][env]["systems"] * 36
        if len(vectors["delta_prediction"]) != expected:
            raise RuntimeError(f"{env} Stage 1A vectors incomplete")
    receipt = {
        "schema_version": "1.0", "status": STATUS_FROZEN, "frozen_at_unix": time.time(),
        "config": cfg, "source_hashes": {str(path.relative_to(root)): sha256(path) for path in paths.values()},
        "stage1a_mutated": False, "models_retrained": False, "transport_regression_run": False,
        "formal_learner_result_files_read": False, "sealed_accessed": False,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_freeze(root: Path, output_root: Path) -> dict:
    receipt = json.loads((output_root / "reference_manifest_frozen.json").read_text())
    if receipt.get("status") != STATUS_FROZEN:
        raise RuntimeError("Stage 1B reference manifest is not frozen")
    for relative, expected in receipt["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"frozen Stage 1B source changed: {relative}")
    return receipt


def _load_stage1_delta(root: Path, cfg: dict, environment: str, candidate_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = root / cfg["stage1a_root"] / "merged" / environment / "layer_deltas.npz"
    payload = np.load(path, allow_pickle=False)
    index = pd.DataFrame({key: payload[key] for key in KEYS})
    index["position"] = np.arange(len(index))
    joined = candidate_table[KEYS].merge(index, on=KEYS, how="left", validate="one_to_one")
    if joined.position.isna().any():
        raise RuntimeError("Stage 1B rows do not join Stage 1A vectors")
    positions = joined.position.to_numpy(int)
    return payload["delta_prediction"][positions].astype(np.float64), payload["observed_target_residual"][positions].astype(np.float64)


def _normal_targets(arrays, norms: dict) -> np.ndarray:
    return _normalized(arrays, norms)[3].astype(np.float64)


def _coupled_reference(root: Path, cfg: dict, table: pd.DataFrame, arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = load_spec(stage1a._coupled_paths(root)["base_spec"])
    design = base["development_v0_1"]
    dt = base["dynamics"]["reference_dt_s"]
    histories = history_probe_bank(design["experience_duration_s"], dt, design["history_energy"])
    queries = query_bank(design["query_duration_s"], dt, design["query_energy"], tuple(design["query_chirp_hz"]))
    history_ids, query_ids = sorted(histories), sorted(queries)
    rcfg = cfg["reference"]["coupled"]
    gain_nodes, gain_weights = lognormal_quadrature(base["actuator_gain"]["mean"], base["actuator_gain"]["cv"], rcfg["gain_quadrature_points"])
    streams = []
    ess_streams = []
    for stream in ("ref_a", "ref_b"):
        particles = sobol_systems(int(rcfg["particles_per_stream"]), int(rcfg[f"{stream}_particle_seed"]), base)
        hmeans = _positions(particles, histories, int(design["history_landmarks"]))
        qmeans_map = _positions(particles, queries, int(design["query_landmarks"]))
        qmeans = np.stack([qmeans_map[name] for name in query_ids], axis=1)
        result = np.empty((len(table), qmeans.shape[2]), dtype=np.float64)
        esses = np.empty(len(table), dtype=np.float64)
        by_system = table.groupby("system_index", sort=True).indices
        for positions in by_system.values():
            positions = np.asarray(positions, dtype=int)
            first = positions[0]
            history_index = int(table.iloc[first].history_index)
            baseline_row = int(np.flatnonzero((table.system_index.to_numpy() == int(table.iloc[first].system_index)) & (table.candidate_index.to_numpy() == -1))[0])
            anchor_obs = arrays.history[baseline_row, 0, :16]
            for candidate in range(6):
                subset = positions[table.iloc[positions].candidate_index.to_numpy(int) == candidate]
                candidate_obs = arrays.history[int(subset[0]), 1, :16]
                posterior = combine_independent_posteriors(
                    (anchor_obs, candidate_obs),
                    (hmeans[history_ids[history_index]], hmeans[history_ids[candidate]]),
                    float(design["sensor_std_m"]), gain_nodes, gain_weights,
                )
                for row_position in subset:
                    query_index = int(table.iloc[row_position].query_index)
                    result[row_position] = np.einsum("n,nf->f", posterior.weights, qmeans[:, query_index])
                    esses[row_position] = posterior.ess
        streams.append(result)
        ess_streams.append(esses)
    return streams[0], streams[1], ess_streams[0], ess_streams[1]


def _articulated_reference(root: Path, cfg: dict, table: pd.DataFrame, arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = json.loads(stage1a._articulated_paths(root)["reference_config"].read_text())
    base = json.loads((root / reference["base_config"]).read_text())
    s0 = json.loads((root / reference["s0_receipt"]).read_text())
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    model = SwimmerModel(base["model"])
    rcfg = cfg["reference"]["articulated"]
    streams, ess_streams = [], []
    for stream in ("ref_a", "ref_b"):
        particles = particle_pool(int(rcfg["particles_per_stream"]), int(rcfg[f"{stream}_particle_seed"]), base["persistent_prior"])
        result = np.empty((len(table), 32), dtype=np.float64)
        esses = np.empty(len(table), dtype=np.float64)
        for _, positions_raw in table.groupby("system_index", sort=True).indices.items():
            positions = np.asarray(positions_raw, dtype=int)
            first = positions[0]
            system_index = int(table.iloc[first].system_index)
            history_index = int(table.iloc[first].history_index)
            baseline_row = int(np.flatnonzero((table.system_index.to_numpy() == system_index) & (table.candidate_index.to_numpy() == -1))[0])
            ih = initial_state_from_features(arrays.history[baseline_row, 0, :8])
            anchor_obs = arrays.history[baseline_row, 0, 8:40]
            candidate_zero = int(positions[table.iloc[positions].candidate_index.to_numpy(int) == 0][0])
            ie = initial_state_from_features(arrays.history[candidate_zero, 1, :8])
            iq = initial_state_from_features(arrays.query_action[baseline_row, :8])
            hmeans = _response_bank(model, particles, ih, history, landmarks)[:, history_index]
            emeans = _response_bank(model, particles, ie, history, landmarks)
            qmeans = _response_bank(model, particles, iq, query, landmarks)
            for candidate in range(6):
                subset = positions[table.iloc[positions].candidate_index.to_numpy(int) == candidate]
                candidate_obs = arrays.history[int(subset[0]), 1, 8:40]
                posterior = combine_independent_posteriors(
                    (anchor_obs, candidate_obs), (hmeans, emeans[:, candidate]),
                    float(base["observation"]["sensor_std"]), np.asarray([1.0]), np.asarray([1.0]),
                )
                for row_position in subset:
                    query_index = int(table.iloc[row_position].query_index)
                    result[row_position] = np.einsum("n,nf->f", posterior.weights, qmeans[:, query_index])
                    esses[row_position] = posterior.ess
        streams.append(result)
        ess_streams.append(esses)
    return streams[0], streams[1], ess_streams[0], ess_streams[1]


def extract(root: Path, output_root: Path, environment: str, shard_index: int, shard_count: int) -> dict:
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    frozen = _verify_freeze(root, output_root)
    cfg = frozen["config"]
    rows_path = root / cfg["stage1a_root"] / "frozen" / f"{environment}_stage1_rows.csv.gz"
    full = pd.read_csv(rows_path)
    table = full.loc[full.system_index.to_numpy(int) % shard_count == shard_index].reset_index(drop=True)
    started = time.perf_counter()
    if environment == "coupled":
        work_manifest = output_root / "work" / f"coupled_rows_{shard_index:02d}_of_{shard_count:02d}.csv.gz"
        _atomic_csv(work_manifest, table)
        arrays = stage1a.arrays_from_sample_manifest(stage1a._coupled_paths(root)["base_spec"], stage1a._coupled_paths(root)["system_pool"], work_manifest)
        _, norms, _ = stage1a._load_coupled_model(root, arrays, __import__("torch").device("cpu"))
    elif environment == "articulated":
        arrays = stage1a._articulated_arrays(root, table)
        reference = json.loads(stage1a._articulated_paths(root)["reference_config"].read_text())
        _, norms, _ = stage1a._load_jepa(root, reference, __import__("torch").device("cpu"))
    else:
        raise ValueError("environment must be coupled or articulated")
    candidate_mask = table.candidate_index.to_numpy(int) >= 0
    candidate_table = table.loc[candidate_mask, KEYS].reset_index(drop=True)
    delta, residual = _load_stage1_delta(root, cfg, environment, candidate_table)
    target = _normal_targets(arrays, norms)[candidate_mask]
    baseline_prediction = target - residual
    if environment == "coupled":
        mu_a, mu_b, ess_a, ess_b = _coupled_reference(root, cfg, table, arrays)
    else:
        mu_a, mu_b, ess_a, ess_b = _articulated_reference(root, cfg, table, arrays)
    mu_a = (mu_a[candidate_mask] - norms["target_mean"]) / norms["target_std"]
    mu_b = (mu_b[candidate_mask] - norms["target_mean"]) / norms["target_std"]
    r_a, r_b = mu_a - baseline_prediction, mu_b - baseline_prediction
    metrics = alignment_metrics(delta, r_b)
    verification = alignment_metrics(delta, r_a)
    ref_difference = np.linalg.norm(mu_a - mu_b, axis=1)
    summary = candidate_table.copy()
    for name, values in metrics.items():
        summary[name] = values
    summary["cos_theta_ref_a"] = verification["cos_theta"]
    summary["rho_ref_a"] = verification["rho"]
    summary["ref_a_b_predictive_disagreement"] = ref_difference
    summary["ref_a_b_disagreement_over_r_b"] = np.divide(ref_difference, metrics["r_b_norm"], out=np.full(len(summary), np.nan), where=metrics["r_b_norm"] > 0)
    summary["ref_a_ess"] = ess_a[candidate_mask]
    summary["ref_b_ess"] = ess_b[candidate_mask]
    shard_root = output_root / "shards" / f"{environment}_{shard_index:02d}_of_{shard_count:02d}"
    table_path, vector_path = shard_root / "alignment_rows.csv.gz", shard_root / "alignment_vectors.npz"
    _atomic_csv(table_path, summary)
    _atomic_npz(vector_path, delta_prediction=delta.astype(np.float32), r_b_ref_a=r_a.astype(np.float32), r_b_ref_b=r_b.astype(np.float32), **{key: summary[key].to_numpy() for key in KEYS})
    receipt = {
        "status": STATUS_SHARD, "environment": environment, "shard_index": shard_index, "shard_count": shard_count,
        "systems": int(summary.system_index.nunique()), "rows": len(summary), "elapsed_seconds": time.perf_counter() - started,
        "reference_manifest_sha256": sha256(output_root / "reference_manifest_frozen.json"),
        "alignment_rows_sha256": sha256(table_path), "alignment_vectors_sha256": sha256(vector_path),
        "ref_a_ess_min": float(np.min(ess_a)), "ref_b_ess_min": float(np.min(ess_b)),
        "models_retrained": False, "stage1a_mutated": False, "transport_regression_run": False,
    }
    _atomic_text(shard_root / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _bootstrap_mean(values: pd.DataFrame, column: str, replicates: int, seed: int) -> dict:
    systems = values.groupby("system_index", sort=True)[column].mean().dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = systems[rng.integers(0, len(systems), size=(replicates, len(systems)))].mean(axis=1)
    return {"mean": float(systems.mean()), "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)), "systems": int(len(systems))}


def merge_analyze(root: Path, output_root: Path, shard_counts: dict[str, int]) -> dict:
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    frozen = _verify_freeze(root, output_root)
    cfg = frozen["config"]
    all_tables = []
    receipts = []
    for environment, shard_count in shard_counts.items():
        pieces = []
        for shard in range(shard_count):
            shard_root = output_root / "shards" / f"{environment}_{shard:02d}_of_{shard_count:02d}"
            receipt = json.loads((shard_root / "receipt.json").read_text())
            if receipt.get("status") != STATUS_SHARD:
                raise RuntimeError("incomplete Stage 1B shard")
            receipts.append(receipt)
            pieces.append(pd.read_csv(shard_root / "alignment_rows.csv.gz"))
        table = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        expected_systems = 704 if environment == "coupled" else 512
        if table.duplicated(KEYS).any() or len(table) != expected_systems * 36 or table.system_index.nunique() != expected_systems:
            raise RuntimeError(f"{environment} Stage 1B merge incomplete or duplicated")
        table.insert(0, "environment", environment)
        all_tables.append(table)
    combined = pd.concat(all_tables, ignore_index=True)
    merged_path = output_root / "merged" / "alignment_rows.csv.gz"
    _atomic_csv(merged_path, combined)
    replicates, seed = int(cfg["analysis"]["bootstrap_replicates"]), int(cfg["analysis"]["bootstrap_seed"])
    intervals = {}
    for environment, env_table in combined.groupby("environment"):
        intervals[environment] = {metric: _bootstrap_mean(env_table, metric, replicates, seed + i) for i, metric in enumerate(("cos_theta", "rho", "a", "b"))}
        intervals[environment]["query_families"] = {
            str(query): {metric: _bootstrap_mean(group, metric, replicates, seed + 100 * int(query) + i) for i, metric in enumerate(("cos_theta", "rho", "a", "b"))}
            for query, group in env_table.groupby("query_index")
        }
    figure_path = output_root / "analysis" / "alignment_by_environment_and_query_family.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex="col")
    colors = {"coupled": "#2878B5", "articulated": "#D95F02"}
    for row, environment in enumerate(("coupled", "articulated")):
        env = intervals[environment]
        for col, metric in enumerate(("cos_theta", "rho", "a", "b")):
            ax = axes[row, col]
            xs = np.arange(6)
            means = np.asarray([env["query_families"][str(q)][metric]["mean"] for q in xs])
            lo = np.asarray([env["query_families"][str(q)][metric]["ci_low"] for q in xs])
            hi = np.asarray([env["query_families"][str(q)][metric]["ci_high"] for q in xs])
            ax.errorbar(xs, means, yerr=np.vstack((means - lo, hi - means)), marker="o", color=colors[environment], capsize=3)
            if metric in ("cos_theta", "a"):
                ax.axhline(0, color="#888888", linewidth=.8)
            ax.set_title(metric if row == 0 else "")
            ax.set_ylabel(environment.capitalize())
            ax.set_xticks(xs, [f"Q{q}" for q in xs])
            ax.grid(alpha=.2)
    fig.suptitle("Stage 1B: Bayes-correction alignment (system-bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    result = {
        "status": "STAGE1B_BAYES_ALIGNMENT_COMPLETE", "systems": {"coupled": 704, "articulated": 512},
        "rows": len(combined), "intervals": intervals,
        "median_ref_a_b_disagreement_over_r_b": {env: float(group.ref_a_b_disagreement_over_r_b.median()) for env, group in combined.groupby("environment")},
        "minimum_ess": {env: {"ref_a": float(group.ref_a_ess.min()), "ref_b": float(group.ref_b_ess.min())} for env, group in combined.groupby("environment")},
        "merged_rows_sha256": sha256(merged_path), "figure_sha256": sha256(figure_path),
        "elapsed_seconds_sum": float(sum(item["elapsed_seconds"] for item in receipts)),
        "models_retrained": False, "transport_regression_run": False,
    }
    _atomic_text(output_root / "analysis" / "summary.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("root", type=Path); p.add_argument("config", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("extract"); p.add_argument("root", type=Path); p.add_argument("output", type=Path); p.add_argument("environment", choices=("coupled", "articulated")); p.add_argument("--shard-index", type=int, required=True); p.add_argument("--shard-count", type=int, required=True)
    p = sub.add_parser("merge-analyze"); p.add_argument("root", type=Path); p.add_argument("output", type=Path); p.add_argument("--coupled-shards", type=int, default=2); p.add_argument("--articulated-shards", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze": result = freeze(args.root, args.config, args.output)
    elif args.command == "extract": result = extract(args.root, args.output, args.environment, args.shard_index, args.shard_count)
    else: result = merge_analyze(args.root, args.output, {"coupled": args.coupled_shards, "articulated": args.articulated_shards})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
