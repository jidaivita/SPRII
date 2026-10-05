#!/usr/bin/env python3
"""Create a model-output-free A2 metadata table from frozen RH20T manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EPISODE = re.compile(
    r"^(?P<task>task_\d+)_user_(?P<user>\d+)_scene_(?P<scene>\d+)_cfg_(?P<cfg>\d+)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--inventory-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    eligible_payload = json.loads(args.eligible.read_text())
    eligible_items = {item["episode_id"]: item for item in eligible_payload["eligible"]}
    eligible = set(eligible_items)
    split_payload = json.loads(args.split_manifest.read_text())
    split_ids = set(split_payload["splits"][args.split]["episode_ids"])
    inventory = {}
    for line in args.inventory_jsonl.open():
        item = json.loads(line)
        inventory[item["scene_id"]] = item
    selected = sorted(eligible & split_ids)
    if selected != sorted(split_ids):
        raise RuntimeError("split contains episodes absent from the frozen eligible universe")
    records = []
    for episode_id in selected:
        match = EPISODE.match(episode_id)
        if match is None:
            raise ValueError(f"invalid frozen episode id {episode_id}")
        source = inventory[episode_id]
        metadata = source["metadata"]
        if args.camera_serial not in source["cameras"]:
            raise ValueError(f"frozen camera absent for {episode_id}")
        records.append({
            "episode_id": episode_id,
            "task_id": match.group("task"),
            "user_id": match.group("user"),
            "scene_ordinal": match.group("scene"),
            "cfg_id": match.group("cfg"),
            "camera_serial": args.camera_serial,
            "action_id": str(metadata["action"]),
            "calib_id": str(metadata["calib"]),
            "finish_time_ms": int(metadata["finish_time"]),
            "frames": int(eligible_items[episode_id]["frames"]),
        })
    payload = {
        "schema_version": "1.1a",
        "status": "FROZEN_METADATA_INPUT_NO_MODEL_OUTPUT",
        "split": args.split,
        "records": records,
        "sources": {
            "eligible_sha256": sha256(args.eligible),
            "split_manifest_sha256": sha256(args.split_manifest),
            "inventory_jsonl_sha256": sha256(args.inventory_jsonl),
        },
        "fixed_camera_serial": args.camera_serial,
        "model_output_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
