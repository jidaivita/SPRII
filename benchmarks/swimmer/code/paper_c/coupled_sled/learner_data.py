import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .batch import batch_position_landmarks, simulate_batch
from .development import PARAMETER_NAMES, _load_values
from .manifests import load_spec
from .waveforms import Probe, history_probe_bank, query_bank


CONDITIONS = ("query_only", "anchor", "repeated", "redundant", "complementary", "wrong_system")


@dataclass(frozen=True)
class LearnerArrays:
    history: np.ndarray
    history_mask: np.ndarray
    query_action: np.ndarray
    target: np.ndarray
    condition: np.ndarray
    system_index: np.ndarray
    anchor_index: np.ndarray
    query_index: np.ndarray
    theta: np.ndarray


def _action_landmarks(probe: Probe, count: int) -> np.ndarray:
    indices = np.linspace(0, len(probe.actions) - 1, count).round().astype(int)
    return probe.actions[indices].reshape(-1)


def _positions(parameters: np.ndarray, probes: Dict[str, Probe], count: int) -> Dict[str, np.ndarray]:
    result = {}
    for probe_id, probe in probes.items():
        states = simulate_batch(parameters, probe)
        result[probe_id] = batch_position_landmarks(states, count, "both_positions", [probe.side] * len(parameters))
    return result


def _segment(position: np.ndarray, action: np.ndarray, gain: float, noise: np.ndarray) -> np.ndarray:
    return np.concatenate((gain * position + noise, action))


def build_development_arrays(
    spec_path: Path,
    manifest_root: Path,
    construction_path: Path,
) -> Dict[str, LearnerArrays]:
    spec = load_spec(spec_path)
    if any(spec["access"][key] for key in ("discovery", "design_validation", "sealed", "formal_training")):
        raise RuntimeError("learner development requires all formal/reserved access to remain closed")
    cfg = spec["development_v0_1"]
    learner_cfg = spec["learner_development"]
    systems = _load_values(manifest_root / "development_v0_1.json", "systems")
    construction = json.loads(construction_path.read_text())
    pair_by_anchor = {row["anchor"]: row for row in construction["pairs"]}
    dt = spec["dynamics"]["reference_dt_s"]
    history_probes = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    queries = query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"]))
    history_ids = sorted(history_probes)
    query_ids = sorted(queries)
    history_position = _positions(systems, history_probes, cfg["history_landmarks"])
    query_position = _positions(systems, queries, cfg["query_landmarks"])
    history_action = {key: _action_landmarks(value, cfg["history_landmarks"]) for key, value in history_probes.items()}
    query_action = {key: _action_landmarks(value, cfg["query_landmarks"]) for key, value in queries.items()}

    log_variance = np.log1p(spec["actuator_gain"]["cv"] ** 2)
    gain_sigma = np.sqrt(log_variance)
    gain_mu = -0.5 * log_variance
    sensor_std = cfg["sensor_std_m"]
    rng = np.random.default_rng(learner_cfg["seed"])
    permutation = rng.permutation(len(systems))
    counts = learner_cfg["split_system_counts"]
    boundaries = np.cumsum([0, counts["train"], counts["select"], counts["holdout"]])
    split_indices = {
        name: permutation[boundaries[index] : boundaries[index + 1]]
        for index, name in enumerate(("train", "select", "holdout"))
    }
    arrays = {}
    condition_to_index = {name: index for index, name in enumerate(CONDITIONS)}
    history_feature_dim = cfg["history_landmarks"] * 4
    query_feature_dim = cfg["query_landmarks"] * 2

    for split_name, indices in split_indices.items():
        rows_history = []
        rows_mask = []
        rows_query_action = []
        rows_target = []
        rows_condition = []
        rows_system = []
        rows_anchor = []
        rows_query = []
        rows_theta = []
        for system_index in indices:
            wrong_index = int(indices[(np.where(indices == system_index)[0][0] + 1) % len(indices)])
            for realization in range(learner_cfg["realizations_per_system"]):
                for anchor_index, anchor in enumerate(history_ids):
                    pair = pair_by_anchor[anchor]
                    first_gain = float(rng.lognormal(gain_mu, gain_sigma))
                    second_gain = float(rng.lognormal(gain_mu, gain_sigma))
                    first_noise = rng.normal(0.0, sensor_std, cfg["history_landmarks"] * 2)
                    second_noise = rng.normal(0.0, sensor_std, cfg["history_landmarks"] * 2)
                    first = _segment(history_position[anchor][system_index], history_action[anchor], first_gain, first_noise)
                    for query_index, query_id in enumerate(query_ids):
                        future_gain = float(rng.lognormal(gain_mu, gain_sigma))
                        future_noise = rng.normal(0.0, sensor_std, query_feature_dim)
                        target = future_gain * query_position[query_id][system_index] + future_noise
                        for condition in CONDITIONS:
                            values = np.zeros((2, history_feature_dim), dtype=np.float32)
                            mask = np.zeros(2, dtype=np.float32)
                            if condition != "query_only":
                                values[0] = first
                                mask[0] = 1.0
                            if condition in ("repeated", "redundant", "complementary", "wrong_system"):
                                added_id = pair[condition if condition != "wrong_system" else "complementary"]
                                added_system = wrong_index if condition == "wrong_system" else system_index
                                values[1] = _segment(
                                    history_position[added_id][added_system], history_action[added_id], second_gain, second_noise
                                )
                                mask[1] = 1.0
                            rows_history.append(values)
                            rows_mask.append(mask)
                            rows_query_action.append(query_action[query_id])
                            rows_target.append(target)
                            rows_condition.append(condition_to_index[condition])
                            rows_system.append(system_index)
                            rows_anchor.append(anchor_index)
                            rows_query.append(query_index)
                            rows_theta.append(systems[system_index])
        arrays[split_name] = LearnerArrays(
            history=np.asarray(rows_history, dtype=np.float32),
            history_mask=np.asarray(rows_mask, dtype=np.float32),
            query_action=np.asarray(rows_query_action, dtype=np.float32),
            target=np.asarray(rows_target, dtype=np.float32),
            condition=np.asarray(rows_condition, dtype=np.int64),
            system_index=np.asarray(rows_system, dtype=np.int64),
            anchor_index=np.asarray(rows_anchor, dtype=np.int64),
            query_index=np.asarray(rows_query, dtype=np.int64),
            theta=np.asarray(rows_theta, dtype=np.float32),
        )
    return arrays


def condition_names() -> Tuple[str, ...]:
    return CONDITIONS
