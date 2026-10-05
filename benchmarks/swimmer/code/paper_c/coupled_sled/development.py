import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .batch import batch_position_landmarks, simulate_batch
from .manifests import load_spec
from .posterior import (
    lognormal_quadrature,
    oracle_floor_fraction,
    posterior_batch_from_observations,
    prior_predictive_scale,
)
from .waveforms import history_probe_bank, query_bank


PARAMETER_NAMES = ("m_L", "m_R", "b_L", "b_R", "k_c", "d_c")


def _load_values(path: Path, key: str) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray([[row[name] for name in PARAMETER_NAMES] for row in payload[key]], dtype=float)


def _features(theta: np.ndarray, probes: Dict[str, Any], landmarks: int, semantics: str) -> Dict[str, np.ndarray]:
    result = {}
    for probe_id, probe in probes.items():
        states = simulate_batch(theta, probe)
        sides = [probe.side] * len(theta)
        result[probe_id] = batch_position_landmarks(states, landmarks, semantics, sides)
    return result


def run_fast_single_screen(
    spec_path: Path,
    manifest_root: Path,
    output_root: Path,
    realizations: int = 1,
    chunk_size: int = 16,
) -> Dict[str, Any]:
    """Development-only single-probe screen.

    This intentionally omits pair construction.  It answers the first failure
    question cheaply: does v0.1 already saturate from one local experience?
    """

    spec = load_spec(spec_path)
    if not spec["access"]["development"] or any(
        spec["access"][key] for key in ("discovery", "design_validation", "sealed")
    ):
        raise RuntimeError("fast screen requires development-only access")
    cfg = spec["development_v0_1"]
    dt = spec["dynamics"]["reference_dt_s"]
    systems = _load_values(manifest_root / "development_v0_1.json", "systems")
    particles = _load_values(manifest_root / "posterior_particles_v0_1.json", "particles")
    history = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    queries = query_bank(
        cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"])
    )
    history_particles = _features(particles, history, cfg["history_landmarks"], cfg["observation_semantics"])
    history_truth = _features(systems, history, cfg["history_landmarks"], cfg["observation_semantics"])
    query_particles_map = _features(particles, queries, cfg["query_landmarks"], "both_positions")
    query_truth_map = _features(systems, queries, cfg["query_landmarks"], "both_positions")
    query_ids = list(queries)
    query_particles = np.stack([query_particles_map[key] for key in query_ids], axis=1)
    query_truth = np.stack([query_truth_map[key] for key in query_ids], axis=1)

    gain_nodes, gain_weights = lognormal_quadrature(
        spec["actuator_gain"]["mean"], spec["actuator_gain"]["cv"], spec["actuator_gain"]["quadrature_points"]
    )
    gain_second = float(np.dot(gain_weights, gain_nodes**2))
    sensor_std = cfg["sensor_std_m"]
    scale = prior_predictive_scale(query_particles, gain_second, sensor_std)
    prior_prediction = query_particles.mean(axis=0)

    rng = np.random.default_rng(43001)
    gain_variance = np.log1p(spec["actuator_gain"]["cv"] ** 2)
    gain_sigma = np.sqrt(gain_variance)
    gain_mu = -0.5 * gain_variance
    rows = []
    risk_by_probe: Dict[str, np.ndarray] = {}
    observations_by_probe: Dict[str, list[np.ndarray]] = {key: [] for key in history}
    truth_index = []
    r0_values = []
    oracle_values = []
    actual_queries = []
    for system_index in range(len(systems)):
        for realization in range(realizations):
            truth_index.append(system_index)
            for probe_id in history:
                gain = float(rng.lognormal(gain_mu, gain_sigma))
                noise = rng.normal(0.0, sensor_std, size=history_truth[probe_id].shape[1])
                observations_by_probe[probe_id].append(gain * history_truth[probe_id][system_index] + noise)
            query_gains = rng.lognormal(gain_mu, gain_sigma, size=(len(query_ids), 1))
            query_noise = rng.normal(0.0, sensor_std, size=query_truth[system_index].shape)
            actual = query_gains * query_truth[system_index] + query_noise
            actual_queries.append(actual)
            r0_values.append(float(np.mean(((actual - prior_prediction) / scale) ** 2)))
            oracle_values.append(float(np.mean(((actual - query_truth[system_index]) / scale) ** 2)))
    actual_queries_array = np.asarray(actual_queries)
    r0 = float(np.mean(r0_values))
    r_oracle = float(np.mean(oracle_values))

    for probe_id, particle_means in history_particles.items():
        observations = np.asarray(observations_by_probe[probe_id])
        risks = []
        ess_values = []
        for start in range(0, len(observations), chunk_size):
            stop = min(start + chunk_size, len(observations))
            weights, ess = posterior_batch_from_observations(
                observations[start:stop], particle_means, sensor_std, gain_nodes, gain_weights
            )
            predictions = np.einsum("bn,nqf->bqf", weights, query_particles)
            risks.extend(np.mean(((actual_queries_array[start:stop] - predictions) / scale) ** 2, axis=(1, 2)))
            ess_values.extend(ess)
        re = float(np.mean(risks))
        risk_by_probe[probe_id] = np.asarray(risks, dtype=float).reshape(len(systems), realizations)
        iq_abs, iq_frac = oracle_floor_fraction(r0, re, r_oracle)
        rows.append(
            {
                "probe_id": probe_id,
                "r0": r0,
                "re": re,
                "r_oracle": r_oracle,
                "iq_abs": iq_abs,
                "iq_frac": iq_frac,
                "median_ess": float(np.median(ess_values)),
                "p05_ess": float(np.quantile(ess_values, 0.05)),
                "peak_force": history[probe_id].peak_force,
            }
        )
    r0_system = np.asarray(r0_values, dtype=float).reshape(len(systems), realizations).mean(axis=1)
    oracle_system = np.asarray(oracle_values, dtype=float).reshape(len(systems), realizations).mean(axis=1)
    bootstrap_rng = np.random.default_rng(43091)
    bootstrap_replicates = int(spec["risk"]["cluster_bootstrap_replicates"])
    bootstrap_indices = bootstrap_rng.integers(0, len(systems), size=(bootstrap_replicates, len(systems)))
    system_rows = {"system_index": np.arange(len(systems)), "r0": r0_system, "r_oracle": oracle_system}
    for row in rows:
        probe_id = row["probe_id"]
        re_system = risk_by_probe[probe_id].mean(axis=1)
        system_rows[f"re__{probe_id}"] = re_system
        boot_r0 = r0_system[bootstrap_indices].mean(axis=1)
        boot_re = re_system[bootstrap_indices].mean(axis=1)
        boot_oracle = oracle_system[bootstrap_indices].mean(axis=1)
        boot_iq = (boot_r0 - boot_re) / (boot_r0 - boot_oracle)
        row["iq_frac_ci_low"] = float(np.quantile(boot_iq, 0.025))
        row["iq_frac_ci_high"] = float(np.quantile(boot_iq, 0.975))
    table = pd.DataFrame(rows).sort_values("iq_frac", ascending=False)
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "single_probe_screen.csv", index=False)
    pd.DataFrame(system_rows).to_csv(output_root / "single_probe_system_risks.csv", index=False)
    low = spec["development_targets"]["single_iq_fraction_low"]
    high = spec["development_targets"]["single_iq_fraction_high"]
    nonsaturated = int(((table["iq_frac"] > low) & (table["iq_frac"] < high)).sum())
    receipt = {
        "status": "FAST_DEVELOPMENT_SCREEN_NOT_FORMAL",
        "systems": len(systems),
        "realizations_per_system": realizations,
        "probes": len(history),
        "queries": len(queries),
        "r0": r0,
        "r_oracle": r_oracle,
        "oracle_gap": r0 - r_oracle,
        "nonsaturated_probe_count": nonsaturated,
        "iq_fraction_range": [float(table["iq_frac"].min()), float(table["iq_frac"].max())],
        "cluster_bootstrap_replicates": bootstrap_replicates,
        "iq_fraction_ci_envelope": [
            float(table["iq_frac_ci_low"].min()),
            float(table["iq_frac_ci_high"].max()),
        ],
        "discovery_accessed": False,
        "design_validation_accessed": False,
        "sealed_accessed": False,
    }
    (output_root / "screen_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_fast_single_screen(args.spec, args.manifest_root, args.output_root, args.realizations, args.chunk_size), sort_keys=True))


if __name__ == "__main__":
    main()
