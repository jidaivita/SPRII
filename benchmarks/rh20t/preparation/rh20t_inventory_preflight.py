#!/usr/bin/env python3
"""Build a provenance-first inventory for an extracted RH20T cfg directory.

This script deliberately does not decide the camera, action channel, coordinate
frame, task split, or model inputs.  It records the filesystem facts needed to
make those decisions in the later sample preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENE_RE = re.compile(
    r"^(task_\d+)_user_(\d+)_scene_(\d+)_cfg_(\d+)$"
)
CAM_RE = re.compile(r"^cam_(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:  # recorded rather than silently discarded
        return {"_read_error": f"{type(exc).__name__}: {exc}"}
    return value if isinstance(value, dict) else {"_non_object": value}


def ffprobe(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {"status": "not_run", "reason": "ffprobe_unavailable"}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=120
        )
        return {"status": "ok", "payload": json.loads(result.stdout)}
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def camera_files(scene: Path) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for child in sorted(scene.iterdir()):
        if not child.is_dir():
            continue
        match = CAM_RE.match(child.name)
        if not match:
            continue
        serial = match.group(1)
        mp4_candidates = (child / "color.mp4", child / "color" / "color.mp4")
        timestamp_candidates = (child / "timestamps.npy", child / "color" / "timestamps.npy")
        mp4 = next((p for p in mp4_candidates if p.is_file()), None)
        timestamps = next((p for p in timestamp_candidates if p.is_file()), None)
        result[serial] = {
            "camera_dir": str(child),
            "color_mp4": str(mp4) if mp4 else None,
            "timestamps_npy": str(timestamps) if timestamps else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--probe-camera-count", type=int, default=3)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SystemExit(f"dataset root is absent: {root}")

    scenes = sorted(
        path for path in root.iterdir() if path.is_dir() and SCENE_RE.match(path.name)
    )
    if not scenes:
        raise SystemExit(f"no RH20T scene directories found under {root}")

    required_transformed = (
        "force_torque.npy",
        "force_torque_base.npy",
        "tcp.npy",
        "tcp_base.npy",
        "joint.npy",
        "gripper.npy",
        "high_freq_data.npy",
    )
    camera_coverage: Counter[str] = Counter()
    camera_video_coverage: Counter[str] = Counter()
    camera_timestamp_coverage: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    metadata_keys: Counter[str] = Counter()
    per_task_scenes: dict[str, list[str]] = defaultdict(list)
    records: list[dict[str, Any]] = []

    for scene in scenes:
        match = SCENE_RE.match(scene.name)
        assert match is not None
        task, user, ordinal, cfg = match.groups()
        task_counts[task] += 1
        per_task_scenes[task].append(scene.name)
        metadata_path = scene / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        if not metadata_path.is_file():
            missing_counts["metadata.json"] += 1
        metadata_keys.update(metadata.keys())

        cameras = camera_files(scene)
        for serial, paths in cameras.items():
            camera_coverage[serial] += 1
            if paths["color_mp4"]:
                camera_video_coverage[serial] += 1
            else:
                missing_counts[f"camera:{serial}:color_mp4"] += 1
            if paths["timestamps_npy"]:
                camera_timestamp_coverage[serial] += 1
            else:
                missing_counts[f"camera:{serial}:timestamps_npy"] += 1

        transformed = scene / "transformed"
        transformed_files: dict[str, str | None] = {}
        for name in required_transformed:
            path = transformed / name
            transformed_files[name] = str(path) if path.is_file() else None
            if not path.is_file():
                missing_counts[f"transformed/{name}"] += 1

        records.append(
            {
                "scene_id": scene.name,
                "task_id": task,
                "user_id": user,
                "scene_ordinal": ordinal,
                "cfg_id": cfg,
                "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
                "metadata": metadata,
                "cameras": cameras,
                "transformed": transformed_files,
            }
        )

    # Deterministic task-stratified sample: round-robin over sorted tasks and scenes.
    selected: list[str] = []
    tasks = sorted(per_task_scenes)
    depth = 0
    while len(selected) < min(args.sample_size, len(scenes)):
        made_progress = False
        for task in tasks:
            values = sorted(per_task_scenes[task])
            if depth < len(values):
                selected.append(values[depth])
                made_progress = True
                if len(selected) >= min(args.sample_size, len(scenes)):
                    break
        if not made_progress:
            break
        depth += 1

    by_id = {record["scene_id"]: record for record in records}
    # Header-only probes on a few highest-coverage candidates.  This is an
    # audit shortlist, not the final fixed-camera decision.
    probe_serials = [
        serial
        for serial, _ in camera_video_coverage.most_common(args.probe_camera_count)
    ]
    video_checks: dict[str, Any] = {}
    for scene_id in selected:
        camera_paths = by_id[scene_id]["cameras"]
        for serial in probe_serials:
            paths = camera_paths.get(serial, {})
            if paths.get("color_mp4"):
                video_checks[f"{scene_id}/{serial}"] = ffprobe(Path(paths["color_mp4"]))

    inventory = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "episode_count": len(scenes),
        "task_count": len(task_counts),
        "episodes_per_task": dict(sorted(task_counts.items())),
        "camera_directory_coverage": dict(camera_coverage.most_common()),
        "camera_video_coverage": dict(camera_video_coverage.most_common()),
        "camera_timestamp_coverage": dict(camera_timestamp_coverage.most_common()),
        "metadata_key_counts": dict(metadata_keys.most_common()),
        "missing_file_counts": dict(missing_counts.most_common()),
        "required_transformed_files": list(required_transformed),
        "sample_selection_rule": "sorted-task round-robin, then sorted scene id",
        "sample_scene_ids": selected,
        "ffprobe_camera_shortlist": probe_serials,
        "decisions_intentionally_not_frozen": [
            "camera_serial",
            "force_coordinate_frame",
            "tcp_coordinate_frame",
            "action_channel",
            "action_temporal_offset",
            "tcp_training_parameterization",
            "task_split",
        ],
    }
    records_path = out / "RH20T_CFG1_EPISODE_INVENTORY.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    inventory["episode_inventory_path"] = str(records_path)
    inventory["episode_inventory_sha256"] = sha256(records_path)

    sample_path = out / "RH20T_CFG1_PREFLIGHT_SAMPLE.json"
    sample_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": inventory["sample_selection_rule"],
        "scene_ids": selected,
        "video_ffprobe": video_checks,
    }
    sample_path.write_text(json.dumps(sample_payload, indent=2) + "\n")
    inventory["sample_manifest_path"] = str(sample_path)
    inventory["sample_manifest_sha256"] = sha256(sample_path)

    inventory_path = out / "RH20T_CFG1_INVENTORY_SUMMARY.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")

    print(json.dumps({
        "inventory": str(inventory_path),
        "episodes": len(scenes),
        "tasks": len(task_counts),
        "sample_size": len(selected),
        "top_camera_video_coverage": camera_video_coverage.most_common(5),
        "missing_file_counts": missing_counts.most_common(10),
    }, indent=2))


if __name__ == "__main__":
    main()
