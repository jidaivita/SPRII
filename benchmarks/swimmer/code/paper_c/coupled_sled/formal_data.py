import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .batch import batch_position_landmarks, simulate_batch
from .development import PARAMETER_NAMES
from .learner_data import LearnerArrays, _action_landmarks, _positions, _segment
from .manifests import load_spec, sobol_systems
from .waveforms import history_probe_bank, query_bank


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _system_rows(values: np.ndarray, prefix: str):
    rows = []
    for index, vector in enumerate(values):
        row = {"system_id": f"{prefix}_{index:05d}"}
        row.update({name: float(value) for name, value in zip(PARAMETER_NAMES, vector)})
        rows.append(row)
    return rows


def write_formal_system_pools(base_spec_path: Path, formal_spec_path: Path, output_root: Path):
    base = load_spec(base_spec_path)
    formal = load_spec(formal_spec_path)
    output_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, split_cfg in formal["system_pools"].items():
        values = sobol_systems(split_cfg["count"], split_cfg["seed"], base)
        path = output_root / f"{split}.json"
        payload = {
            "schema_version": "1.0",
            "split": split,
            "count": len(values),
            "seed": split_cfg["seed"],
            "systems": _system_rows(values, split),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        files[path.name] = _sha256(path)
    receipt = {
        "status": "FORMAL_SYSTEM_POOLS_CREATED",
        "base_spec_sha256": _sha256(base_spec_path),
        "formal_spec_sha256": _sha256(formal_spec_path),
        "files": files,
        "sealed_generated": False,
    }
    (output_root / "pool_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_system_pool(path: Path) -> Tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text())
    ids = [row["system_id"] for row in payload["systems"]]
    values = np.asarray([[row[name] for name in PARAMETER_NAMES] for row in payload["systems"]], dtype=float)
    return ids, values


def deterministic_derangement(system_ids: list[str], seed: int) -> Tuple[np.ndarray, str]:
    order = sorted(range(len(system_ids)), key=lambda index: hashlib.sha256(system_ids[index].encode()).hexdigest())
    rng = np.random.default_rng(seed)
    permuted = np.asarray(order, dtype=int)
    while True:
        candidate = permuted[rng.permutation(len(permuted))]
        mapping = np.empty(len(permuted), dtype=int)
        mapping[np.asarray(order)] = candidate
        if np.all(mapping != np.arange(len(mapping))):
            break
    digest = hashlib.sha256(mapping.astype("<i8").tobytes()).hexdigest()
    return mapping, digest


def write_sample_manifest(
    split: str,
    system_ids: list[str],
    conditions: Iterable[str],
    anchors: list[str],
    queries: list[str],
    construction: Dict,
    output_path: Path,
    base_seed: int,
    realizations: int,
    wrong_mapping: np.ndarray | None = None,
):
    pair_by_anchor = {row["anchor"]: row for row in construction["pairs"]}
    rows = []
    condition_list = list(conditions)
    for system_index, system_id in enumerate(system_ids):
        for realization in range(realizations):
            for anchor_index, anchor in enumerate(anchors):
                pair = pair_by_anchor[anchor]
                for query_index, query in enumerate(queries):
                    group_key = f"{split}|{system_id}|{realization}|{anchor}|{query}|{base_seed}"
                    group_seed = int.from_bytes(hashlib.sha256(group_key.encode()).digest()[:8], "little") % (2**63 - 1)
                    for condition_index, condition in enumerate(condition_list):
                        if condition in ("query_only", "anchor"):
                            added = "NONE"
                        elif condition == "wrong_system":
                            added = pair["complementary"]
                        else:
                            added = pair[condition]
                        rows.append({
                            "system_index": system_index,
                            "system_id": system_id,
                            "condition_index": condition_index,
                            "condition": condition,
                            "anchor_index": anchor_index,
                            "anchor_probe": anchor,
                            "query_index": query_index,
                            "query": query,
                            "added_probe": added,
                            "realization": realization,
                            "group_seed": group_seed,
                            "wrong_system_index": int(wrong_mapping[system_index]) if condition == "wrong_system" else -1,
                        })
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return {"rows": len(table), "sha256": _sha256(output_path), "condition_counts": table.condition.value_counts().sort_index().to_dict()}


def arrays_from_sample_manifest(
    base_spec_path: Path,
    system_pool_path: Path,
    sample_manifest_path: Path,
) -> LearnerArrays:
    spec = load_spec(base_spec_path)
    cfg = spec["development_v0_1"]
    _, systems = load_system_pool(system_pool_path)
    table = pd.read_csv(sample_manifest_path)
    dt = spec["dynamics"]["reference_dt_s"]
    history_probes = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    queries = query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"]))
    history_position = _positions(systems, history_probes, cfg["history_landmarks"])
    query_position = _positions(systems, queries, cfg["query_landmarks"])
    history_action = {key: _action_landmarks(value, cfg["history_landmarks"]) for key, value in history_probes.items()}
    query_action = {key: _action_landmarks(value, cfg["query_landmarks"]) for key, value in queries.items()}
    log_variance = np.log1p(spec["actuator_gain"]["cv"] ** 2)
    gain_sigma = np.sqrt(log_variance)
    gain_mu = -0.5 * log_variance
    sensor_std = cfg["sensor_std_m"]
    history_feature_dim = cfg["history_landmarks"] * 4
    query_feature_dim = cfg["query_landmarks"] * 2
    history_values = np.zeros((len(table), 2, history_feature_dim), dtype=np.float32)
    masks = np.zeros((len(table), 2), dtype=np.float32)
    query_values = np.empty((len(table), query_feature_dim), dtype=np.float32)
    targets = np.empty((len(table), query_feature_dim), dtype=np.float32)
    theta = np.empty((len(table), len(PARAMETER_NAMES)), dtype=np.float32)
    for row_index, row in enumerate(table.itertuples(index=False)):
        rng = np.random.default_rng(int(row.group_seed))
        first_gain = float(rng.lognormal(gain_mu, gain_sigma))
        second_gain = float(rng.lognormal(gain_mu, gain_sigma))
        first_noise = rng.normal(0.0, sensor_std, cfg["history_landmarks"] * 2)
        second_noise = rng.normal(0.0, sensor_std, cfg["history_landmarks"] * 2)
        future_gain = float(rng.lognormal(gain_mu, gain_sigma))
        future_noise = rng.normal(0.0, sensor_std, query_feature_dim)
        if row.condition != "query_only":
            history_values[row_index, 0] = _segment(history_position[row.anchor_probe][row.system_index], history_action[row.anchor_probe], first_gain, first_noise)
            masks[row_index, 0] = 1.0
        if row.added_probe != "NONE":
            added_system = row.wrong_system_index if row.condition == "wrong_system" else row.system_index
            history_values[row_index, 1] = _segment(history_position[row.added_probe][added_system], history_action[row.added_probe], second_gain, second_noise)
            masks[row_index, 1] = 1.0
        query_values[row_index] = query_action[row.query]
        targets[row_index] = future_gain * query_position[row.query][row.system_index] + future_noise
        theta[row_index] = systems[row.system_index]
    return LearnerArrays(
        history=history_values,
        history_mask=masks,
        query_action=query_values,
        target=targets,
        condition=table.condition_index.to_numpy(np.int64),
        system_index=table.system_index.to_numpy(np.int64),
        anchor_index=table.anchor_index.to_numpy(np.int64),
        query_index=table.query_index.to_numpy(np.int64),
        theta=theta,
    )


def save_arrays(path: Path, arrays: LearnerArrays):
    np.savez_compressed(path, **arrays.__dict__)


def load_arrays(path: Path) -> LearnerArrays:
    values = np.load(path)
    return LearnerArrays(**{key: values[key] for key in LearnerArrays.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_spec", type=Path)
    parser.add_argument("formal_spec", type=Path)
    parser.add_argument("construction", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    receipt = write_formal_system_pools(args.base_spec, args.formal_spec, args.output_root / "systems")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
