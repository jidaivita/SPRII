import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles

from .model import PARAMETER_NAMES, SwimmerModel, prior_log_scale
from .waveforms import banks


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_log_scales(rng, prior):
    mass = rng.uniform(*prior["mass_scale"], size=3)
    damping = rng.uniform(*prior["damping_scale"], size=2)
    return np.log(np.concatenate([mass, damping]))


def _landmarks(steps: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(max(1, steps // count), steps, count).round().astype(int))


def _response_and_jacobian(model, theta, initial, actions, landmarks, scale, step):
    mean = model.rollout(theta, initial, actions, landmarks)
    jacobian = np.empty((len(mean), len(theta)), dtype=float)
    for parameter in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[parameter] = step * scale[parameter]
        plus = model.rollout(theta + offset, initial, actions, landmarks)
        minus = model.rollout(theta - offset, initial, actions, landmarks)
        jacobian[:, parameter] = (plus - minus) / (2 * step)
    return mean, jacobian


def _effective_rank(metric, trace_fraction, relative_floor):
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[0] <= 0:
        return 0, eigenvalues, eigenvectors
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    trace_rank = int(np.searchsorted(cumulative, trace_fraction) + 1)
    floor_rank = int(np.sum(eigenvalues >= relative_floor * eigenvalues[0]))
    return min(trace_rank, floor_rank), eigenvalues, eigenvectors


def _principal_angle_deg(left, right, rank):
    if rank == 0:
        return 90.0
    return float(np.degrees(subspace_angles(left[:, :rank], right[:, :rank])).max())


def _signal_snr(mean, baseline, probe, sensor_std):
    delta = (mean - baseline).reshape(-1, 8)
    if probe.startswith("j1_"):
        actuated, coupled = delta[:, [1, 6]], delta[:, [2, 7]]
    elif probe.startswith("j2_"):
        actuated, coupled = delta[:, [2, 7]], delta[:, [1, 6]]
    else:
        actuated, coupled = delta[:, [1, 2, 6, 7]], delta[:, [3, 4]]
    return float(np.sqrt(np.mean(actuated ** 2)) / sensor_std), float(np.sqrt(np.mean(coupled ** 2)) / sensor_std)


def _conditional_ratios(history_fisher, query_fisher):
    ratios = []
    values = []
    eye = np.eye(5)
    for anchor in range(6):
        covariance_h = np.linalg.inv(eye + history_fisher[anchor])
        for query in range(6):
            remaining = float(np.trace(query_fisher[query] @ covariance_h))
            candidate_values = []
            for candidate in range(6):
                covariance_pair = np.linalg.inv(eye + history_fisher[anchor] + history_fisher[candidate])
                candidate_values.append(float(np.trace(query_fisher[query] @ (covariance_h - covariance_pair))))
            dynamic_range = max(candidate_values) - min(candidate_values)
            ratios.append(dynamic_range / max(remaining, 1e-12))
            values.append(candidate_values)
    return np.asarray(ratios), np.asarray(values)


def run_s0(root: Path, config_path: Path, output_root: Path):
    config = json.loads(config_path.read_text())
    if any(config["access"].values()):
        raise RuntimeError("S0 must remain isolated from Paper A/B, sealed, NAD, and GPU training")
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["s0"]["seed"])
    model = SwimmerModel(config["model"])
    contexts = []
    for index in range(config["s0"]["contexts"]):
        contexts.append((_sample_log_scales(rng, config["persistent_prior"]), model.sample_initial_state(rng, config["transient_initial_state"])))
    prior_samples = np.vstack([item[0] for item in contexts])
    parameter_scale = prior_log_scale(config["persistent_prior"])
    sensor_std = config["observation"]["sensor_std"]
    horizons = []
    cached = {}
    for duration in config["s0"]["candidate_horizons_s"]:
        history, query = banks(duration, config["model"]["timestep_s"])
        all_names = [("history", key, value) for key, value in history.items()] + [("query", key, value) for key, value in query.items()]
        steps = len(next(iter(history.values())))
        landmarks = _landmarks(steps, config["observation"]["landmark_count"])
        fisher = np.empty((len(contexts), 12, 5, 5), dtype=float)
        snr = []
        stable = True
        for context_index, (theta, initial) in enumerate(contexts):
            zero = np.zeros((steps, 2), dtype=float)
            baseline = model.rollout(theta, initial, zero, landmarks)
            for waveform_index, (_, name, actions) in enumerate(all_names):
                mean, jacobian = _response_and_jacobian(
                    model, theta, initial, actions, landmarks, parameter_scale,
                    config["s0"]["finite_difference_log_step"],
                )
                stable = stable and bool(np.isfinite(mean).all() and np.isfinite(jacobian).all())
                fisher[context_index, waveform_index] = jacobian.T @ jacobian / sensor_std ** 2
                if waveform_index < 6:
                    snr.append(_signal_snr(mean, baseline, name, sensor_std))
        mean_fisher = fisher.mean(axis=0)
        reductions = [float((5 - np.trace(np.linalg.inv(np.eye(5) + item))) / 5) for item in mean_fisher[:6]]
        snr_array = np.asarray(snr).reshape(len(contexts), 6, 2)
        summary = {
            "duration_s": duration,
            "stable": stable,
            "median_actuated_snr": float(np.median(snr_array[:, :, 0])),
            "median_coupled_snr": float(np.median(snr_array[:, :, 1])),
            "single_probe_trace_reduction_min": min(reductions),
            "single_probe_trace_reduction_max": max(reductions),
            "single_probe_trace_reductions": reductions,
        }
        horizons.append(summary)
        cached[duration] = (fisher, history, query)

    gate = config["s0"]
    usable = [row for row in horizons if row["stable"]
              and row["median_actuated_snr"] >= gate["minimum_snr_actuated"]
              and row["median_coupled_snr"] >= gate["minimum_snr_coupled"]
              and row["single_probe_trace_reduction_min"] >= gate["minimum_single_probe_trace_reduction"]
              and row["single_probe_trace_reduction_max"] <= gate["maximum_single_probe_trace_reduction"]]
    chosen = usable[0]["duration_s"] if usable else None
    if chosen is None:
        receipt = {"status": "S0_NO_GO", "horizon_diagnostics": horizons, "reason": "NO_HORIZON_PASSES_TECHNICAL_GATE"}
        (output_root / "s0_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
        return receipt

    fisher, history, query = cached[chosen]
    pooled_per_context = fisher.mean(axis=1)
    pooled = pooled_per_context.mean(axis=0)
    rank, eigenvalues, eigenvectors = _effective_rank(pooled, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"])
    half_a = pooled_per_context[::2].mean(axis=0)
    half_b = pooled_per_context[1::2].mean(axis=0)
    rank_a, _, vectors_a = _effective_rank(half_a, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"])
    rank_b, _, vectors_b = _effective_rank(half_b, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"])
    stable_rank = min(rank, rank_a, rank_b)
    half_angle = _principal_angle_deg(vectors_a, vectors_b, stable_rank)
    radii = np.linalg.norm((prior_samples-prior_samples.mean(axis=0))/parameter_scale, axis=1)
    inner = pooled_per_context[radii <= np.median(radii)].mean(axis=0)
    outer = pooled_per_context[radii > np.median(radii)].mean(axis=0)
    rank_inner, _, vectors_inner = _effective_rank(inner, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"])
    rank_outer, _, vectors_outer = _effective_rank(outer, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"])
    prior_rank = min(rank, rank_inner, rank_outer)
    prior_angle = _principal_angle_deg(vectors_inner, vectors_outer, prior_rank)

    query_subspaces = []
    for query_index, query_name in enumerate(query):
        query_metric = fisher[:, 6 + query_index].mean(axis=0)
        query_rank, query_eigenvalues, query_vectors = _effective_rank(
            query_metric, gate["effective_rank_trace_fraction"], gate["effective_rank_relative_eigenvalue"]
        )
        query_subspaces.append({
            "query": query_name,
            "rank": query_rank,
            "eigenvalues": query_eigenvalues,
            "basis": query_vectors[:, :query_rank],
        })

    history_mean = fisher[:, :6].mean(axis=0)
    query_mean = fisher[:, 6:].mean(axis=0)
    ratios, conditional_values = _conditional_ratios(history_mean, query_mean)
    recommended = float(np.clip(np.quantile(ratios, 0.25), gate["minimum_practical_ratio_floor"], 0.20))
    stable_global = half_angle <= gate["pooled_subspace_max_principal_angle_deg"] and prior_angle <= gate["pooled_subspace_max_principal_angle_deg"]
    status = "S0_GO" if stable_global and stable_rank >= 2 else "S0_NO_GO"

    np.savez_compressed(
        output_root / "s0_physical_metrics.npz", pooled_metric=pooled,
        pooled_eigenvalues=eigenvalues, pooled_basis=eigenvectors,
        parameter_scale=parameter_scale,
        history_fisher=history_mean, query_fisher=query_mean,
        conditional_values=conditional_values, practical_ratios=ratios,
    )
    query_json = [{**row, "basis": _jsonable(row["basis"]), "eigenvalues": _jsonable(row["eigenvalues"])} for row in query_subspaces]
    (output_root / "query_conditioned_subspaces.json").write_text(json.dumps(query_json, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(PARAMETER_NAMES, eigenvalues / max(eigenvalues.sum(), 1e-12), color="#315a8a")
    axes[0].tick_params(axis="x", rotation=35); axes[0].set_ylabel("pooled eigenvalue fraction")
    axes[1].hist(ratios, bins=12, color="#b5523b"); axes[1].axvline(recommended, color="black", ls="--")
    axes[1].set_xlabel("candidate range / oracle-reducible value")
    fig.tight_layout(); fig.savefig(output_root / "s0_identifiability_and_range.png", dpi=180); plt.close(fig)

    receipt = {
        "status": status,
        "chosen_horizon_s": chosen,
        "horizon_diagnostics": horizons,
        "global_effective_rank": rank,
        "stable_effective_rank": min(stable_rank, prior_rank),
        "pooled_eigenvalues": eigenvalues,
        "hash_half_max_principal_angle_deg": half_angle,
        "prior_subrange_max_principal_angle_deg": prior_angle,
        "query_conditioned_ranks": {row["query"]: row["rank"] for row in query_subspaces},
        "query_subspaces_are_allowed_to_rotate": True,
        "s1_practical_ratio_threshold_frozen": recommended,
        "practical_ratio_summary": {
            "min": float(ratios.min()), "q25": float(np.quantile(ratios, .25)),
            "median": float(np.median(ratios)), "max": float(ratios.max()),
        },
        "initial_state_treatment": "OBSERVED_TRANSIENT_NUISANCE_CONDITIONED_AND_PAIRED",
        "config_sha256": _sha256(config_path),
        "gpu_used": False,
        "protected_scope_2_touched": False,
        "sealed_accessed": False,
    }
    (output_root / "s0_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_s0(args.root, args.config, args.output_root)), sort_keys=True))


if __name__ == "__main__":
    main()
