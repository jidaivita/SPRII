#!/usr/bin/env python3
"""Build the deterministic RH20T cfg1 10 Hz cache and eligibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCENE_RE = re.compile(r"^(task_\d+)_user_\d+_scene_\d+_cfg_0001$")
CAMERA = "750612070851"
HISTORY = 24
MAX_HORIZON = 16
MIN_FRAMES = HISTORY + HISTORY + MAX_HORIZON


def load_object(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_branch(scene: Path, filename: str, value_key: str, rgb_ts: np.ndarray, camera: str = CAMERA) -> np.ndarray:
    payload = load_object(scene / "transformed" / filename)
    branch = payload.get(camera, []) if isinstance(payload, dict) else []
    if not branch:
        raise ValueError(f"missing_camera_branch:{filename}")
    branch = sorted(branch, key=lambda item: int(item["timestamp"]))
    source_t = np.asarray([int(item["timestamp"]) for item in branch], dtype=np.int64)
    source_x = np.asarray([item[value_key] for item in branch], dtype=np.float64)
    if source_x.ndim != 2:
        raise ValueError(f"invalid_shape:{filename}:{source_x.shape}")
    if not np.isfinite(source_x).all():
        raise ValueError(f"nonfinite:{filename}")
    if source_t[0] > rgb_ts[0] or source_t[-1] < rgb_ts[-1]:
        raise ValueError(f"timestamp_range_does_not_cover_rgb:{filename}")
    # Collapse duplicate timestamps deterministically by retaining the last item.
    keep = np.r_[source_t[1:] != source_t[:-1], True]
    source_t = source_t[keep]
    source_x = source_x[keep]
    if filename.startswith("tcp") and source_x.shape[1] >= 7:
        # Resolve quaternion sign ambiguity before linear interpolation, matching
        # the official API's component-wise interpolation as closely as possible.
        for index in range(1, len(source_x)):
            if np.dot(source_x[index - 1, 3:7], source_x[index, 3:7]) < 0:
                source_x[index, 3:7] *= -1
    output = np.stack(
        [np.interp(rgb_ts, source_t, source_x[:, dim]) for dim in range(source_x.shape[1])],
        axis=1,
    )
    if filename.startswith("tcp") and output.shape[1] >= 7:
        norm = np.linalg.norm(output[:, 3:7], axis=1, keepdims=True)
        output[:, 3:7] /= np.maximum(norm, 1e-12)
    return output.astype(np.float32)


def gripper_asof(scene: Path, rgb_ts: np.ndarray, camera: str = CAMERA) -> tuple[np.ndarray, np.ndarray]:
    payload = load_object(scene / "transformed" / "gripper.npy")
    branch = payload.get(camera, {}) if isinstance(payload, dict) else {}
    events: dict[int, float] = {}
    for _, item in sorted(branch.items(), key=lambda pair: int(pair[0])):
        command = item.get("gripper_command") if isinstance(item, dict) else None
        if isinstance(command, (list, tuple, np.ndarray)) and len(command) >= 3:
            events[int(command[2])] = float(command[0])
    if not events:
        raise ValueError("missing_gripper_command_events")
    event_t = np.asarray(sorted(events), dtype=np.int64)
    event_x = np.asarray([events[int(timestamp)] for timestamp in event_t], dtype=np.float32)
    indices = np.searchsorted(event_t, rgb_ts, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("no_causal_gripper_command_before_first_rgb")
    return event_x[indices, None], event_t[indices]


def decode_gray(video_path: Path, expected_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    capture.release()
    if len(frames) != expected_frames:
        raise ValueError(f"video_timestamp_count_mismatch:{len(frames)}:{expected_frames}")
    return np.stack(frames).astype(np.uint8)


def process_one(arguments: tuple[str, str, str, str]) -> dict[str, Any]:
    scene_raw, cache_dir_raw, dataset_root_raw, camera = arguments
    scene = Path(scene_raw)
    cache_dir = Path(cache_dir_raw)
    dataset_root = Path(dataset_root_raw)
    match = SCENE_RE.match(scene.name)
    assert match is not None
    task_id = match.group(1)
    result: dict[str, Any] = {"episode_id": scene.name, "task_id": task_id}
    try:
        metadata = json.loads((scene / "metadata.json").read_text())
        rating = int(metadata.get("rating", -1))
        result["rating"] = rating
        if rating < 2:
            raise ValueError(f"rating_below_2:{rating}")
        camera_root = scene / f"cam_{camera}"
        timestamp_path = next((p for p in (camera_root / "color" / "timestamps.npy", camera_root / "timestamps.npy") if p.is_file()), camera_root / "color" / "timestamps.npy")
        video_path = next((p for p in (camera_root / "color" / "color.mp4", camera_root / "color.mp4") if p.is_file()), camera_root / "color" / "color.mp4")
        for required in (
            timestamp_path,
            video_path,
            scene / "transformed" / "force_torque_base.npy",
            scene / "transformed" / "tcp_base.npy",
            scene / "transformed" / "gripper.npy",
        ):
            if not required.is_file():
                raise ValueError(f"missing:{required.relative_to(dataset_root)}")
        rgb_ts = np.asarray(load_object(timestamp_path), dtype=np.int64).reshape(-1)
        if len(rgb_ts) < MIN_FRAMES:
            raise ValueError(f"too_short:{len(rgb_ts)}")
        if not np.all(np.diff(rgb_ts) > 0):
            raise ValueError("rgb_timestamps_not_strictly_increasing")
        ft = aligned_branch(scene, "force_torque_base.npy", "zeroed", rgb_ts, camera)
        tcp = aligned_branch(scene, "tcp_base.npy", "tcp", rgb_ts, camera)
        if ft.shape != (len(rgb_ts), 6):
            raise ValueError(f"ft_shape:{ft.shape}")
        if tcp.shape != (len(rgb_ts), 7):
            raise ValueError(f"tcp_shape:{tcp.shape}")
        gripper, gripper_issue_ts = gripper_asof(scene, rgb_ts, camera)
        gray = decode_gray(video_path, len(rgb_ts))
        cache_dir.mkdir(parents=True, exist_ok=True)
        final = cache_dir / f"{scene.name}.npz"
        temporary = cache_dir / f".{scene.name}.tmp.npz"
        np.savez_compressed(
            temporary,
            rgb_gray=gray,
            timestamps_ms=rgb_ts,
            ft_base_zeroed=ft,
            tcp_base=tcp,
            gripper_command_width=gripper.astype(np.float32),
            gripper_command_issue_ms=gripper_issue_ts,
        )
        temporary.replace(final)
        result.update({
            "eligible": True,
            "frames": int(len(rgb_ts)),
            "duration_seconds": float((rgb_ts[-1] - rgb_ts[0]) / 1000.0),
            "strict_sameep_query_anchor_min": 2 * HISTORY - 1,
            "query_anchor_max_h16": int(len(rgb_ts) - MAX_HORIZON - 1),
            "cache_path": str(final),
            "cache_bytes": final.stat().st_size,
            "cache_sha256": sha256(final),
        })
    except Exception as exc:
        result.update({"eligible": False, "reason": f"{type(exc).__name__}:{exc}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--camera-serial", default=CAMERA, help="Frozen cfg1 camera by default; changing it creates a different corpus")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_cache = cache_root / "episodes"
    scenes = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and SCENE_RE.match(path.name)
    )
    jobs = [(str(path), str(episode_cache), str(dataset_root), args.camera_serial) for path in scenes]
    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                print(json.dumps({
                    "completed": completed,
                    "total": len(jobs),
                    "eligible_so_far": sum(item.get("eligible", False) for item in results),
                }), flush=True)
    results.sort(key=lambda item: item["episode_id"])
    eligible = [item for item in results if item.get("eligible")]
    excluded = [item for item in results if not item.get("eligible")]
    counts = Counter(item["task_id"] for item in eligible)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "camera_serial": args.camera_serial,
        "history_frames": HISTORY,
        "max_horizon": MAX_HORIZON,
        "minimum_frames_for_strict_sameep": MIN_FRAMES,
        "eligibility_rules": [
            "standard cfg1 robot scene name",
            "metadata rating >= 2",
            "fixed-camera RGB video and timestamps present",
            "base-frame zeroed FT, base-frame TCP, and gripper files present",
            "strictly increasing RGB timestamps",
            "at least 64 frames",
            "FT/TCP finite and covering RGB timestamp range",
            "causal gripper command exists at every RGB timestamp",
            "entire fixed-camera video decodes and frame count equals timestamp count",
        ],
        "alignment": {
            "canonical_grid": "fixed-camera RGB timestamps",
            "ft_tcp": "sort source timestamps then linear interpolation",
            "gripper_command": "latest command with issue timestamp <= observation timestamp",
            "video_container_fps_ignored": True,
        },
        "raw_standard_scene_count": len(scenes),
        "eligible_episode_count": len(eligible),
        "excluded_episode_count": len(excluded),
        "eligible_task_count": len(counts),
        "eligible_episodes_per_task": dict(sorted(counts.items())),
        "eligible": eligible,
        "excluded": excluded,
        "excluded_reason_counts": dict(Counter(item["reason"] for item in excluded).most_common()),
    }
    manifest = output_dir / "eligible_episode_manifest.json"
    temporary = output_dir / ".eligible_episode_manifest.tmp.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest)
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "tasks": len(counts),
        "cache_bytes": sum(item["cache_bytes"] for item in eligible),
        "excluded_reason_counts": payload["excluded_reason_counts"],
    }
    (output_dir / "CACHE_BUILD_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
