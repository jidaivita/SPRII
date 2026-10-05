import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from paper_c.coupled_sled.formal_data import save_arrays
from paper_c.coupled_sled.learner_data import LearnerArrays

from .model import SwimmerModel
from .s0_screen import _landmarks
from .waveforms import banks


_CFG = None
_S2 = None
_MODEL = None
_HISTORY = None
_QUERY = None
_LANDMARKS = None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _systems(count, seed, prior):
    values = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    lower = np.asarray([prior["mass_scale"][0]] * 3 + [prior["damping_scale"][0]] * 2)
    upper = np.asarray([prior["mass_scale"][1]] * 3 + [prior["damping_scale"][1]] * 2)
    return np.log(lower + values * (upper-lower))


def _manifest(split, count, s2):
    rows = []
    for system_index in range(count):
        if s2.get("manifest_mode") == "full_all_pair_condition_balanced":
            for condition_index, condition in enumerate(("K0", "K1", "K2")):
                for anchor in range(6):
                    for candidate in range(6):
                        for query in range(6):
                            realization = (anchor + candidate + query + condition_index + system_index) % s2["realizations_per_system"]
                            rows.append({
                                "sample_index": len(rows), "split": split,
                                "system_index": system_index, "condition_index": condition_index,
                                "condition": condition, "anchor_index": anchor,
                                "candidate_index": candidate, "query_index": query,
                                "history_probe_1": anchor if condition != "K0" else -1,
                                "history_probe_2": candidate if condition == "K2" else -1,
                                "nuisance_realization": realization,
                            })
            continue
        for condition_index, condition in enumerate(("K0", "K1", "K2")):
            for slot in range(12):
                anchor = slot % 6
                query = (anchor + 3 * (slot // 6) + system_index) % 6
                candidate = (2 * anchor + (slot // 6) + system_index) % 6
                realization = (slot + condition_index + system_index) % s2["realizations_per_system"]
                rows.append({
                    "sample_index": len(rows), "split": split,
                    "system_index": system_index, "condition_index": condition_index,
                    "condition": condition, "anchor_index": anchor,
                    "candidate_index": candidate, "query_index": query,
                    "history_probe_1": anchor if condition != "K0" else -1,
                    "history_probe_2": candidate if condition == "K2" else -1,
                    "nuisance_realization": realization,
                })
    return pd.DataFrame(rows)


def _init_worker(base, s2, horizon):
    global _CFG, _S2, _MODEL, _HISTORY, _QUERY, _LANDMARKS
    _CFG, _S2 = base, s2
    _MODEL = SwimmerModel(base["model"])
    _HISTORY, _QUERY = banks(horizon, base["model"]["timestep_s"])
    _LANDMARKS = _landmarks(len(next(iter(_HISTORY.values()))), base["observation"]["landmark_count"])


def _action_landmarks(actions):
    indices = np.clip(_LANDMARKS - 1, 0, len(actions)-1)
    return actions[indices].reshape(-1)


def _system_arrays(task):
    split, system_index, theta, manifest_rows = task
    history_names, query_names = list(_HISTORY), list(_QUERY)
    cache = {}
    for realization in range(_S2["realizations_per_system"]):
        rng = np.random.default_rng(np.random.SeedSequence([_S2["data_seed"], 0 if split == "train" else 1, system_index, realization]))
        initial_h = _MODEL.sample_initial_state(rng, _CFG["transient_initial_state"])
        initial_e = _MODEL.sample_initial_state(rng, _CFG["transient_initial_state"])
        initial_q = _MODEL.sample_initial_state(rng, _CFG["transient_initial_state"])
        init_h = np.concatenate([initial_h.qpos[2:], initial_h.qvel])
        init_e = np.concatenate([initial_e.qpos[2:], initial_e.qvel])
        init_q = np.concatenate([initial_q.qpos[2:], initial_q.qvel])
        first, second, future = [], [], []
        for name in history_names:
            first_mean = _MODEL.rollout(theta, initial_h, _HISTORY[name], _LANDMARKS)
            second_mean = _MODEL.rollout(theta, initial_e, _HISTORY[name], _LANDMARKS)
            first_noise = rng.normal(0, _CFG["observation"]["sensor_std"], first_mean.shape)
            second_noise = rng.normal(0, _CFG["observation"]["sensor_std"], second_mean.shape)
            first.append(np.concatenate([init_h, first_mean + first_noise, _action_landmarks(_HISTORY[name])]))
            second.append(np.concatenate([init_e, second_mean + second_noise, _action_landmarks(_HISTORY[name])]))
        for name in query_names:
            mean = _MODEL.rollout(theta, initial_q, _QUERY[name], _LANDMARKS)
            future.append(mean + rng.normal(0, _CFG["observation"]["sensor_std"], mean.shape))
        cache[realization] = (np.asarray(first), np.asarray(second), init_q, np.asarray(future))
    nrows = len(manifest_rows)
    history = np.zeros((nrows, 2, 48), dtype=np.float32)
    mask = np.zeros((nrows, 2), dtype=np.float32)
    query_action = np.empty((nrows, 16), dtype=np.float32)
    target = np.empty((nrows, 32), dtype=np.float32)
    condition = np.empty(nrows, dtype=np.int64)
    anchor_index = np.empty(nrows, dtype=np.int64)
    query_index = np.empty(nrows, dtype=np.int64)
    for local, row in enumerate(manifest_rows.itertuples(index=False)):
        first, second, init_q, future = cache[row.nuisance_realization]
        if row.condition != "K0":
            history[local, 0] = first[row.anchor_index]; mask[local, 0] = 1
        if row.condition == "K2":
            history[local, 1] = second[row.candidate_index]; mask[local, 1] = 1
        query_action[local] = np.concatenate([init_q, _action_landmarks(_QUERY[query_names[row.query_index]])])
        target[local] = future[row.query_index]
        condition[local] = row.condition_index
        anchor_index[local] = row.anchor_index
        query_index[local] = row.query_index
    return {
        "history": history, "history_mask": mask, "query_action": query_action, "target": target,
        "condition": condition, "system_index": np.full(nrows, system_index, dtype=np.int64),
        "anchor_index": anchor_index, "query_index": query_index,
        "theta": np.repeat(theta[None, :], nrows, axis=0).astype(np.float32),
    }


def run(root, s2_path, s0_root, output_root, workers):
    root, s2_path, s0_root, output_root = map(Path, (root, s2_path, s0_root, output_root))
    s2 = json.loads(s2_path.read_text())
    base_path = root / s2["base_config"]
    base = json.loads(base_path.read_text())
    s1_path = root / s2["s1r_receipt"]
    s1 = json.loads(s1_path.read_text())
    s0 = json.loads((s0_root / "s0_receipt.json").read_text())
    if s1["status"] != "S1_GO" or s1["config_sha256"] != _sha256(base_path):
        raise RuntimeError("S2 requires a hash-bound S1-R GO")
    if any(s2["access"].values()):
        raise RuntimeError("S2 data must not access evaluation, sealed, A/B, or NAD")
    output_root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    all_systems = {}
    for split, count, seed in (("train", s2["train_systems"], s2["system_seed_train"]), ("select", s2["select_systems"], s2["system_seed_select"])):
        systems = _systems(count, seed, base["persistent_prior"])
        all_systems[split] = systems
        system_path = output_root / f"{split}_systems.npy"
        np.save(system_path, systems)
        manifest = _manifest(split, count, s2)
        manifest_path = output_root / f"{split}_sample_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        tasks = [(split, index, systems[index], manifest[manifest.system_index == index]) for index in range(count)]
        context = mp.get_context("spawn")
        with context.Pool(max(1, workers), initializer=_init_worker, initargs=(base, s2, s0["chosen_horizon_s"])) as pool:
            chunks = list(pool.imap(_system_arrays, tasks, chunksize=1))
        combined = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
        arrays = LearnerArrays(**combined)
        arrays_path = output_root / f"{split}_arrays.npz"
        save_arrays(arrays_path, arrays)
        receipts[split] = {
            "systems": count, "rows": len(manifest),
            "manifest_sha256": _sha256(manifest_path), "arrays_sha256": _sha256(arrays_path),
            "condition_counts": manifest.condition.value_counts().sort_index().to_dict(),
        }
    overlap = len(set(map(tuple, np.round(all_systems["train"], 12))) & set(map(tuple, np.round(all_systems["select"], 12))))
    receipt = {
        "status": "S2_IMMUTABLE_TRAIN_SELECT_DATA_COMPLETE", "splits": receipts,
        "exact_train_select_overlap": overlap,
        "raw_and_jepa_manifest_identity": "EXACT_SAME_FILES",
        "history_initial_observation_in_segment": True,
        "query_initial_observation_in_Q": True,
        "condition_labels_exposed_to_model": False,
        "s1r_receipt_sha256": _sha256(s1_path), "base_config_sha256": _sha256(base_path),
        "s2_config_sha256": _sha256(s2_path), "sealed_accessed": False, "protected_scope_2_touched": False,
    }
    (output_root / "s2_data_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("config"); parser.add_argument("s0_root"); parser.add_argument("output_root")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2)//2))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.config, args.s0_root, args.output_root, args.workers), sort_keys=True))


if __name__ == "__main__":
    main()
