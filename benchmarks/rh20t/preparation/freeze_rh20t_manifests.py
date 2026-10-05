#!/usr/bin/env python3
"""Freeze RH20T task split, relation pairing rules, and field definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEED_TEXT = "rh20t-paper-a-task-split-v1-20260819"
HISTORY = 24
MAX_HORIZON = 16


def stable_score(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.eligible_manifest.read_text())
    episodes = source["eligible"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_task = defaultdict(list)
    by_id = {}
    for episode in episodes:
        by_task[episode["task_id"]].append(episode)
        by_id[episode["episode_id"]] = episode
    if any(len(items) < 4 for items in by_task.values()):
        raise RuntimeError("every frozen task must have at least four eligible episodes")

    task_order = sorted(by_task, key=lambda task: stable_score(SEED_TEXT, task))
    if len(task_order) != 124:
        raise RuntimeError(f"expected 124 tasks, found {len(task_order)}")
    split_tasks = {
        "train": sorted(task_order[:74]),
        "validation": sorted(task_order[74:99]),
        "test": sorted(task_order[99:]),
    }
    assert len(split_tasks["train"]) == 74
    assert len(split_tasks["validation"]) == 25
    assert len(split_tasks["test"]) == 25
    assert not (set(split_tasks["train"]) & set(split_tasks["validation"]))
    assert not (set(split_tasks["train"]) & set(split_tasks["test"]))
    assert not (set(split_tasks["validation"]) & set(split_tasks["test"]))

    split_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_eligible_manifest": str(args.eligible_manifest.resolve()),
        "source_eligible_manifest_sha256": digest(args.eligible_manifest),
        "unit": "task_id",
        "algorithm": "sort tasks by SHA256(seed_text|task_id), then 74/25/25",
        "seed_text": SEED_TEXT,
        "splits": {},
    }
    for split, tasks in split_tasks.items():
        ids = sorted(item["episode_id"] for task in tasks for item in by_task[task])
        split_payload["splits"][split] = {
            "task_ids": tasks,
            "task_count": len(tasks),
            "episode_ids": ids,
            "episode_count": len(ids),
        }
    split_path = output / "task_split_manifest.json"
    write_frozen(split_path, split_payload)

    pairing_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_eligible_manifest_sha256": digest(args.eligible_manifest),
        "task_split_manifest_sha256": digest(split_path),
        "history_frames": HISTORY,
        "max_horizon": MAX_HORIZON,
        "common_rules": {
            "query_anchor_min": 2 * HISTORY - 1,
            "donor_end": "query_anchor - history_frames",
            "donor_window": "[donor_end-history_frames+1, donor_end]",
            "query_history": "[query_anchor-history_frames+1, query_anchor]",
            "query_target": "[query_anchor+1, query_anchor+horizon]",
            "sameep_nonoverlap": True,
            "target_causal": True,
            "all_relation_conditions_use_same_query_anchor_range_per_episode": True,
            "future_episode_length_not_used_as_model_input": True,
        },
        "splits": {},
    }
    for split, tasks in split_tasks.items():
        split_episode_ids = split_payload["splits"][split]["episode_ids"]
        split_other = {episode_id: by_id[episode_id] for episode_id in split_episode_ids}
        entries = []
        for query_id in split_episode_ids:
            query = by_id[query_id]
            independent = sorted(
                (item for item in by_task[query["task_id"]] if item["episode_id"] != query_id),
                key=lambda item: item["episode_id"],
            )
            max_independent_frames = max(item["frames"] for item in independent)
            wrong_all = [
                item for item in split_other.values() if item["task_id"] != query["task_id"]
            ]
            max_wrong_frames = max(item["frames"] for item in wrong_all)
            query_anchor_max = min(
                query["frames"] - MAX_HORIZON - 1,
                max_independent_frames + HISTORY - 1,
                max_wrong_frames + HISTORY - 1,
            )
            if query_anchor_max < 2 * HISTORY - 1:
                raise RuntimeError(f"no common strict anchor for {query_id}")

            wrong_pool = [item for item in wrong_all if item["frames"] >= query_anchor_max - HISTORY + 1]
            wrong_pool.sort(
                key=lambda item: stable_score("random-v1", split, query_id, item["episode_id"])
            )
            random_count = min(8, len(independent), len(wrong_pool))
            random_candidates = wrong_pool[:random_count]
            if not random_candidates:
                raise RuntimeError(f"insufficient matched random candidates for {query_id}")
            entries.append({
                "query_episode_id": query_id,
                "task_id": query["task_id"],
                "query_frames": query["frames"],
                "query_anchor_min": 2 * HISTORY - 1,
                "query_anchor_max": query_anchor_max,
                "sameep": {
                    "donor_episode_id": query_id,
                    "donor_end_rule": "query_anchor - 24",
                },
                "independent_candidate_episode_ids": [item["episode_id"] for item in independent],
                "random_candidate_episode_ids": [item["episode_id"] for item in random_candidates],
            })
        pairing_payload["splits"][split] = {
            "query_count": len(entries),
            "entries": entries,
        }
    pairing_path = output / "causal_pairing_manifest.json"
    write_frozen(pairing_path, pairing_payload)

    field_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_eligible_manifest_sha256": digest(args.eligible_manifest),
        "task_split_manifest_sha256": digest(split_path),
        "causal_pairing_manifest_sha256": digest(pairing_path),
        "task_definition": "history-conditioned future FT/TCP prediction with partial causal action",
        "camera": {
            "serial": source.get("camera_serial", "750612070851"),
            "type": "fixed global/top-down RGB camera",
            "canonical_timestamps": True,
            "container_fps_ignored": True,
            "resize": [96, 96],
            "channels": ["grayscale", "signed temporal difference"],
        },
        "history_frames": 24,
        "horizons": [1, 4, 16],
        "lowdim_history": {
            "force_torque": "6D base-frame zeroed FT",
            "tcp": "7D base-frame XYZ plus quaternion",
            "gripper_command": "1D commanded width",
        },
        "targets": {
            "primary": "h4 future 6D base-frame zeroed FT normalized MSE",
            "secondary": "h16 future base-frame TCP XYZ error in centimetres",
            "recorded_auxiliary": ["FT h1", "FT h16", "TCP XYZ h4"],
        },
        "partial_causal_action": {
            "field": "gripper_command[0] width",
            "alignment": "latest command whose issue timestamp <= observation timestamp",
            "future_tcp_or_joint_as_action": False,
            "applied_identically_to_all_conditions": True,
        },
        "alignment": {
            "ft_tcp": "sort timestamps, then component-wise linear interpolation to RGB timestamps",
            "tcp_quaternion": "sign continuity before interpolation, unit normalization after interpolation",
            "gripper": "causal backward as-of; never future interpolation",
        },
        "normalization": {
            "statistics_source": "train tasks only",
            "validation_test_statistics_forbidden": True,
            "same_statistics_for_all_conditions": True,
        },
        "sampling": {
            "task_uniform": True,
            "episode_and_anchor_sampled_after_task": True,
            "same_eligible_universe_for_all_conditions": True,
        },
        "conditions": [
            "B0", "Bx-SameEp", "Bx-Indep", "B3-SameEp",
            "B3-Indep", "B3-Random", "B3-LowDim-Same", "B3-LowDim-Random",
        ],
    }
    field_path = output / "field_action_manifest.json"
    write_frozen(field_path, field_payload)

    summary = {
        "task_split_manifest": str(split_path),
        "task_split_manifest_sha256": digest(split_path),
        "causal_pairing_manifest": str(pairing_path),
        "causal_pairing_manifest_sha256": digest(pairing_path),
        "field_action_manifest": str(field_path),
        "field_action_manifest_sha256": digest(field_path),
        "split_counts": {
            split: {
                "tasks": item["task_count"], "episodes": item["episode_count"]
            } for split, item in split_payload["splits"].items()
        },
    }
    write_frozen(output / "MANIFEST_FREEZE_COMPLETE.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
