"""Paired native AD tokens from official packed histories and explicit episodes.

No episode boundaries are inferred from `dones`: the official recorder may mark
the end of a recorded prefix as done. The episode sidecar records the original
episode identity and its interval in the packed HDF5 stream.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from runners.history_adapter import HistoryStore


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class NativeHistorySampler:
    def __init__(self, h5_path, index_path, task_manifest, episode_index, *,
                 query_len=300, support_count=2, support_len=100, split="train",
                 use_teammate_actions=False, cache_mb=64):
        require(query_len > 0 and support_count >= 2 and support_len > 0, "Invalid history budget")
        require(split in ("train", "validation", "val", "development"), "Offline trainer cannot access final test")
        self.query_len, self.support_count, self.support_len = query_len, support_count, support_len
        self.use_teammate_actions = use_teammate_actions
        self.paths = {"h5": str(Path(h5_path).resolve()), "index": str(Path(index_path).resolve()),
                      "task_manifest": str(Path(task_manifest).resolve()), "episode_index": str(Path(episode_index).resolve())}
        self.store = HistoryStore(h5_path, index_path, cache_size_mb=cache_mb)
        tasks = read_rows(task_manifest)
        self.tasks = {row["task_id"]: row for row in tasks}
        require(len(self.tasks) == len(tasks), "Duplicate task IDs in authoritative task manifest")
        require(all(row["history_id"] == i for i, row in enumerate(self.store._index)),
                "Official adapter requires consecutive history IDs in index order")
        episodes = {}
        for source in read_rows(episode_index):
            row = dict(source)
            hid = int(row["history_id"])
            meta = self.store.get_history_meta(hid)
            require(row["task_id"] == meta["task_id"] and int(row["env_idx"]) == meta["env_idx"],
                    "Episode sidecar identifies another packed learning history")
            start, end = int(row["start"]), int(row["end"])
            require(0 <= start < end <= meta["T"], "Episode interval outside packed history")
            require("episode_id" in row and isinstance(row.get("complete"), bool)
                    and isinstance(row.get("recorded_prefix"), bool), "Episode identity/recording provenance missing")
            require(row["complete"] or row["recorded_prefix"], "Unqualified partial episode segment")
            row.update(start=start, end=end, history_id=hid)
            episodes.setdefault(hid, []).append(row)
        self.episodes, self.identity_histories, self.starts = {}, {}, {}
        self.task_identities = {}
        shapes, action_dims = set(), set()
        for hid, meta in enumerate(self.store._index):
            if meta["split"] != split:
                continue
            task = self.tasks.get(meta["task_id"])
            require(task is not None and task["split"] == split and task["layout_name"] == meta["layout"],
                    "Training history lacks matching authoritative task specification")
            teammate = task.get("teammate")
            require(isinstance(teammate, dict) and teammate.get("kind") and teammate.get("family"),
                    "Fixed teammate specification missing")
            # Random-number stream seeds are not distinct behavioral identities.
            traits = {key: value for key, value in teammate.items() if key not in ("base_seed", "seed")}
            identity = hashlib.sha256(canonical(traits).encode()).hexdigest()
            self.task_identities[meta["task_id"]] = identity
            rows = sorted(episodes.get(hid, []), key=lambda row: row["start"])
            require(rows, f"Explicit episode metadata missing for history {hid}")
            require(len({str(row['episode_id']) for row in rows}) == len(rows),
                    "One real episode appears in multiple sidecar rows; merge its contiguous pieces first")
            require(rows[0]["start"] == 0 and rows[-1]["end"] <= meta["T"]
                    and all(a["end"] == b["start"] for a, b in zip(rows, rows[1:])),
                    "Explicit episodes must partition an initial packed-history prefix")
            if len(rows) <= support_count:
                continue
            earliest = rows[support_count]["start"]
            # A recorder can leave an unannotated incomplete tail. It is not
            # promoted to an episode; paired training uses the annotated prefix.
            latest = rows[-1]["end"] - query_len
            if latest < earliest:
                continue
            self.episodes[hid] = rows
            self.starts[hid] = (earliest, latest)
            self.identity_histories.setdefault(identity, []).append(hid)
            shapes.add(tuple(meta["obs_shape"]))
            action_dims.add(meta["action_dim"])
            if use_teammate_actions:
                require(meta.get("has_teammate_actions") is True, "Requested teammate actions missing")
        require(self.identity_histories, "No eligible history with two old episodes and a full query window")
        require(len(shapes) == len(action_dims) == 1, "Packed observation/action shapes are inconsistent")
        self.obs_shape, self.num_actions = next(iter(shapes)), next(iter(action_dims))
        self.identities = sorted(self.identity_histories)
        self.last_metadata = None

    def close(self):
        self.store.close()

    def fingerprint(self):
        return {name: {"path": path, "sha256": sha256_file(path)} for name, path in self.paths.items()}

    def _window(self, hid, start, length, *, end=None, independent_start=None):
        meta = self.store.get_history_meta(hid)
        stop = min(start + length, meta["T"] if end is None else end)
        valid = stop - start
        require(valid > 0, "Empty token window")
        # Query uses the original AD previous token convention, including across
        # episode resets. Independent support starts have no outside predecessor.
        have_previous = start > (0 if independent_start is None else independent_start)
        read_start = start - int(have_previous)
        data = self.store.load_slice(hid, read_start, valid + int(have_previous),
                                     include_teammate_actions=self.use_teammate_actions)
        offset = int(have_previous)
        result = {"obs": np.zeros((length,) + self.obs_shape, np.float32),
                  "prev_actions": np.zeros(length, np.int32), "prev_rewards": np.zeros(length, np.float32),
                  "target_actions": np.zeros(length, np.int32), "dones": np.zeros(length, bool),
                  "attention_mask": np.arange(length) < valid}
        result["obs"][:valid] = data["obs"][offset:]
        result["target_actions"][:valid] = data["actions"][offset:]
        result["dones"][:valid] = data["dones"][offset:]
        if have_previous:
            result["prev_actions"][:valid] = data["actions"][:valid]
            result["prev_rewards"][:valid] = data["rewards"][:valid]
        else:
            result["prev_actions"][1:valid] = data["actions"][:valid-1]
            result["prev_rewards"][1:valid] = data["rewards"][:valid-1]
        if self.use_teammate_actions:
            require("teammate_actions" in data, "Missing teammate action array")
            result["prev_teammate_actions"] = np.zeros(length, np.int32)
            if have_previous:
                result["prev_teammate_actions"][:valid] = data["teammate_actions"][:valid]
            else:
                result["prev_teammate_actions"][1:valid] = data["teammate_actions"][:valid-1]
        require(np.isfinite(result["obs"]).all() and np.isfinite(result["prev_rewards"]).all(), "Nonfinite data")
        require(((result["target_actions"][:valid] >= 0) & (result["target_actions"][:valid] < self.num_actions)).all(),
                "Ego action label outside native action space")
        return result

    def sample(self, rng, batch_size):
        require(1 <= batch_size <= len(self.identities), "Batch requires distinct fixed-partner identities")
        identities = rng.choice(self.identities, size=batch_size, replace=False)
        query_rows, support_rows, audit = [], [], []
        for identity in identities:
            hid = int(rng.choice(self.identity_histories[identity]))
            start = int(rng.integers(self.starts[hid][0], self.starts[hid][1] + 1))
            episodes = self.episodes[hid]
            current = next(row for row in episodes if row["start"] <= start < row["end"])
            eligible = [row for row in episodes if row["end"] <= current["start"]]
            require(len(eligible) >= self.support_count, "Insufficient earlier independent episodes")
            indices = sorted(rng.choice(len(eligible), size=self.support_count, replace=False))
            supports, records = [], []
            for index in indices:
                episode = eligible[index]
                latest = max(episode["start"], episode["end"] - self.support_len)
                support_start = int(rng.integers(episode["start"], latest + 1))
                support = self._window(hid, support_start, self.support_len, end=episode["end"], independent_start=episode["start"])
                supports.append({key: value for key, value in support.items() if key not in ("target_actions", "dones")})
                records.append({"episode_id": episode["episode_id"], "episode_start": episode["start"],
                                "episode_end": episode["end"], "start": support_start,
                                "valid_tokens": int(support["attention_mask"].sum()),
                                "recorded_prefix": episode["recorded_prefix"]})
            query_rows.append(self._window(hid, start, self.query_len))
            support_rows.append({key: np.stack([row[key] for row in supports]) for key in supports[0]})
            audit.append({"partner_identity": str(identity), "task_id": self.store.get_history_meta(hid)["task_id"],
                          "history_id": hid, "env_idx": self.store.get_history_meta(hid)["env_idx"],
                          "query_start": start, "query_end": start + self.query_len,
                          "query_first_episode_id": current["episode_id"], "support": records})
        query = {key: np.stack([row[key] for row in query_rows]) for key in query_rows[0]}
        support = {key: np.stack([row[key] for row in support_rows]) for key in support_rows[0]}
        result = {"query": query, "support": support,
                  "target_actions": query["target_actions"], "loss_mask": query["attention_mask"],
                  "pair_indices": np.tile(np.array([0, 1], np.int32), (batch_size, 1))}
        self.last_metadata = {"rows": audit, "batch_plan_sha256": hashlib.sha256(canonical(audit).encode()).hexdigest(),
                              "target": "recorded_ego_actions", "episode_boundaries_from": "explicit_sidecar",
                              "total_tokens_per_example": self.query_len + self.support_count * self.support_len}
        return result


def baseline_batch(batch):
    """Original AD architecture receives exactly the same observed tokens.

    Support windows are ordered in time; cross-window predecessor tokens remain
    exactly those given to the extension. Only query positions contribute CE.
    """
    query, support = batch["query"], batch["support"]
    b, k, length = support["prev_actions"].shape
    result = {}
    for key in ("obs", "prev_actions", "prev_rewards", "attention_mask", "prev_teammate_actions"):
        if key in query:
            flat = support[key].reshape((b, k * length) + support[key].shape[3:])
            result[key] = np.concatenate([flat, query[key]], axis=1)
    result["target_actions"] = np.concatenate([np.zeros((b, k * length), np.int32), query["target_actions"]], axis=1)
    result["loss_mask"] = np.concatenate([np.zeros((b, k * length), bool), batch["loss_mask"]], axis=1)
    return result
