#!/usr/bin/env python3
"""New-run adapters for real large batches; no training loop or job launcher.

The original training loop can compute one VICReg loss over the entire physical
batch. Repeated partner identities are allowed, but each row still contains two
different, earlier episodes of that same fixed partner. Exact support-episode
pairs cannot repeat within a batch. Nothing here changes an existing run.

Integration in a NEW source copy requires three explicit changes:
  * instantiate ReplacementHistorySampler from make_replacement_sampler_class();
  * remove the obsolete batch_size <= number_of_partner_identities guard;
  * record distinct_partners_per_batch=False and the sampler policy below.
The old bound train.py is deliberately not monkeypatched or rewritten here.
For original AD, enable_retained_handles(store) changes only HDF5 handle lifetime.

This module deliberately does not implement microbatch VICReg averaging. If a
physical batch of 1024 fits, the unchanged native objective computes joint1024
statistics directly. Two-pass gradient accumulation is a separate OOM fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SAMPLER_POLICY = {
    "identity_sampling": "uniform_with_replacement",
    "distinct_partners_per_batch": False,
    "one_row": "same fixed partner; two distinct earlier support episodes; one later query",
    "duplicate_support_episode_pair_in_batch": "rejected and resampled",
    "query_support_overlap": "forbidden within each row",
    "cross_row_episode_reuse": "allowed except the exact two-episode support pair",
    "batch_statistic_unit": "episode pair, not distinct partner identity or query token",
}


class _RetainedGroup:
    """Keep Dataset objects alive, preserving their existing raw chunk caches."""

    def __init__(self, group):
        self._group = group
        self._datasets = {}

    def __getitem__(self, key):
        if key not in self._datasets:
            self._datasets[key] = self._group[key]
        return self._datasets[key]

    def __contains__(self, key):
        return key in self._group

    def __iter__(self):
        return iter(self._group)

    def __len__(self):
        return len(self._group)

    def __getattr__(self, key):
        return getattr(self._group, key)


class _RetainedFile:
    """Read-only proxy used only by a newly created HistoryStore instance."""

    def __init__(self, h5_file):
        if h5_file.mode != "r":
            raise ValueError("Retained handles require a read-only HDF5 file")
        self._file = h5_file
        self._groups = {}

    def __getitem__(self, key):
        if key not in self._groups:
            self._groups[key] = _RetainedGroup(self._file[key])
        return self._groups[key]

    def __contains__(self, key):
        return key in self._file

    def __iter__(self):
        return iter(self._file)

    def __len__(self):
        return len(self._file)

    def __getattr__(self, key):
        return getattr(self._file, key)

    def close(self):
        self._groups.clear()
        self._file.close()


def enable_retained_handles(store):
    """Optimize reads on this store, without changing arrays, RNG, or sampling."""
    from native_a.mapped_history import enable_mapped_store
    return enable_mapped_store(store)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _support_pair_key(row):
    return _canonical({"partner_identity": row["partner_identity"],
                       "history_id": row["history_id"],
                       "episodes": sorted(str(x["episode_id"]) for x in row["support"])})


def validate_pair_row(row):
    """Validate actual metadata; never infer episode boundaries from done flags."""
    support = row["support"]
    if len(support) != 2 or len({str(x["episode_id"]) for x in support}) != 2:
        raise ValueError("Each positive pair requires two different real episodes")
    if not row.get("partner_identity"):
        raise ValueError("Fixed partner identity is required")
    query_start = int(row["query_start"])
    query_episode_start = int(row["query_first_episode_start"])
    if query_start < query_episode_start or int(row["query_end"]) <= query_start:
        raise ValueError("Invalid query interval")
    for item in support:
        start, end = int(item["episode_start"]), int(item["episode_end"])
        window, valid = int(item["start"]), int(item["valid_tokens"])
        if not (0 <= start <= window < end <= query_episode_start):
            raise ValueError("Support must precede the query's first episode")
        if valid <= 0 or window + valid > end:
            raise ValueError("Support window crosses its real episode boundary")
        if str(item["episode_id"]) == str(row["query_first_episode_id"]):
            raise ValueError("Support and query use the same real episode")
    return _support_pair_key(row)


def make_replacement_sampler_class():
    """Import after the caller places the chosen new-run repo on sys.path."""
    from native_a.sampler import NativeHistorySampler, require

    class ReplacementHistorySampler(NativeHistorySampler):
        def __init__(self, *args, retain_dataset_handles=True, **kwargs):
            super().__init__(*args, **kwargs)
            require(self.support_count == 2, "This declared protocol uses exactly two support episodes")
            if retain_dataset_handles:
                enable_retained_handles(self.store)

        def _one_pair(self, rng, identity):
            hid = int(rng.choice(self.identity_histories[identity]))
            start = int(rng.integers(self.starts[hid][0], self.starts[hid][1] + 1))
            episodes = self.episodes[hid]
            current = next(row for row in episodes if row["start"] <= start < row["end"])
            eligible = [row for row in episodes if row["end"] <= current["start"]]
            require(len(eligible) >= 2, "Insufficient earlier independent episodes")
            indices = sorted(rng.choice(len(eligible), size=2, replace=False))
            supports, records = [], []
            for index in indices:
                episode = eligible[index]
                latest = max(episode["start"], episode["end"] - self.support_len)
                support_start = int(rng.integers(episode["start"], latest + 1))
                support = self._window(hid, support_start, self.support_len,
                                       end=episode["end"], independent_start=episode["start"])
                supports.append({key: value for key, value in support.items()
                                 if key not in ("target_actions", "dones")})
                records.append({"episode_id": episode["episode_id"], "episode_start": episode["start"],
                                "episode_end": episode["end"], "start": support_start,
                                "valid_tokens": int(support["attention_mask"].sum()),
                                "recorded_prefix": episode["recorded_prefix"]})
            row = {"partner_identity": str(identity), "task_id": self.store.get_history_meta(hid)["task_id"],
                   "history_id": hid, "env_idx": self.store.get_history_meta(hid)["env_idx"],
                   "query_start": start, "query_end": start + self.query_len,
                   "query_first_episode_id": current["episode_id"],
                   "query_first_episode_start": current["start"], "support": records}
            validate_pair_row(row)
            return (self._window(hid, start, self.query_len),
                    {key: np.stack([item[key] for item in supports]) for key in supports[0]}, row)

        def sample(self, rng, batch_size):
            require(batch_size >= 1, "Batch size must be positive")
            query_batch, support_batch, audit, seen = None, None, [], set()
            # Identity draws can repeat. Once an identity is drawn for a row,
            # duplicate-pair resampling stays within it, preserving its frequency.
            identities = rng.choice(self.identities, size=batch_size, replace=True)
            duplicate_draws = 0
            for row_index, identity in enumerate(identities):
                for _ in range(1024):
                    query, support, row = self._one_pair(rng, identity)
                    pair_key = validate_pair_row(row)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        if query_batch is None:
                            query_batch = {key: np.empty((batch_size, *value.shape), dtype=value.dtype) for key, value in query.items()}
                            support_batch = {key: np.empty((batch_size, *value.shape), dtype=value.dtype) for key, value in support.items()}
                        for key, value in query.items():
                            query_batch[key][row_index] = value
                        for key, value in support.items():
                            support_batch[key][row_index] = value
                        audit.append(row)
                        break
                    duplicate_draws += 1
                else:
                    raise ValueError("Cannot fill this batch without repeating an exact support episode pair")
            query, support = query_batch, support_batch
            result = {"query": query, "support": support,
                      "target_actions": query["target_actions"], "loss_mask": query["attention_mask"],
                      "pair_indices": np.tile(np.array([0, 1], np.int32), (batch_size, 1))}
            self.last_metadata = {"rows": audit,
                "batch_plan_sha256": hashlib.sha256(_canonical(audit).encode()).hexdigest(),
                "target": "recorded_ego_actions", "episode_boundaries_from": "explicit_sidecar",
                "total_tokens_per_example": self.query_len + 2 * self.support_len,
                "sampler_policy": SAMPLER_POLICY, "duplicate_pair_draws_rejected": duplicate_draws,
                "distinct_actual_partners": len(set(str(x) for x in identities)),
                "effective_batch_rows": batch_size, "distinct_support_episode_pairs": len(seen)}
            return result

    return ReplacementHistorySampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "ADAPTER_PREPARED_NOT_TRAINING_LAUNCHED", "sampler_policy": SAMPLER_POLICY,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "training_loop": "unchanged physical full-batch loop, if measured to fit",
                      "fallback_accumulation": "not implemented; small-microbatch VC averaging is forbidden"},
                     indent=2))


if __name__ == "__main__":
    main()
