#!/usr/bin/env python3
"""Build frozen interaction- and system-diversity PokeWorld datasets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np

from persistent_jepa.pokeworld import PokeConfig, simulate_split


ALGORITHM_VERSION = "maximin-normalized-logm-gamma-logk-v1"
ORDERING_SEED = 20260823
EXTRA_ROLLOUT_SEED = 20260824


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximin_order(payload: dict[str, np.ndarray], cfg: PokeConfig, seed: int) -> np.ndarray:
    points = np.stack(
        [
            (np.log(payload["mass"]) - np.log(cfg.mass_low))
            / (np.log(cfg.mass_high) - np.log(cfg.mass_low)),
            (payload["gamma"] - cfg.gamma_low) / (cfg.gamma_high - cfg.gamma_low),
            (np.log(payload["stiffness"]) - np.log(cfg.stiffness_low))
            / (np.log(cfg.stiffness_high) - np.log(cfg.stiffness_low)),
        ],
        axis=1,
    ).astype(np.float64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(points.shape[0]))
    order = np.empty(points.shape[0], dtype=np.int64)
    order[0] = first
    selected = np.zeros(points.shape[0], dtype=bool)
    selected[first] = True
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    min_distance[first] = -1.0
    for position in range(1, points.shape[0]):
        candidate = int(np.argmax(min_distance))
        order[position] = candidate
        selected[candidate] = True
        distance = np.sum((points - points[candidate]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1.0
    return order


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def shapes(payload: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
    }


def coverage(payload: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    report = {}
    for key in ("mass", "gamma", "stiffness"):
        values = payload[key].astype(np.float64)
        report[key] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return report


def write_derived_root(
    root: Path,
    source_val: Path,
    train: dict[str, np.ndarray],
    metadata: dict[str, object],
    cfg: PokeConfig,
) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {root}")
    temporary = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        np.savez_compressed(temporary / "train.npz", **train)
        try:
            os.link(source_val, temporary / "val.npz")
            val_storage = "hardlink-to-frozen-source"
        except OSError:
            shutil.copy2(source_val, temporary / "val.npz")
            val_storage = "byte-copy-of-frozen-source"
        manifest = {
            "protocol": "pokeworld-scaling-v1",
            "config": cfg.__dict__,
            "metadata": {**metadata, "validation_storage": val_storage},
            "splits": {"train": shapes(train)},
            "source_val_sha256": file_sha256(source_val),
            "test_read": False,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["manifest_sha256"] = sha256(encoded).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads((args.source_root / "manifest.json").read_text())
    cfg = PokeConfig(**source_manifest["config"])
    source_train_path = args.source_root / "train.npz"
    source_val_path = args.source_root / "val.npz"
    train = load_npz(source_train_path)
    if train["states"].shape[:2] != (cfg.train_systems, 4):
        raise ValueError(f"expected frozen R4 train data, got {train['states'].shape[:2]}")
    if not np.array_equal(train["system_ids"], np.arange(cfg.train_systems)):
        raise ValueError("expected contiguous frozen train system IDs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "manifest": file_sha256(args.source_root / "manifest.json"),
        "train": file_sha256(source_train_path),
        "val": file_sha256(source_val_path),
    }
    ordering_rows = maximin_order(train, cfg, ORDERING_SEED)
    ordering_ids = train["system_ids"][ordering_rows]
    system_manifest = {
        "algorithm": ALGORITHM_VERSION,
        "seed": ORDERING_SEED,
        "coordinate_space": ["normalized_log_mass", "normalized_gamma", "normalized_log_stiffness"],
        "source_hashes": source_hashes,
        "ordering_system_ids": ordering_ids.tolist(),
        "subsets": {
            "S10": ordering_ids[:200].tolist(),
            "S25": ordering_ids[:500].tolist(),
            "S50": ordering_ids[:1000].tolist(),
            "S100": train["system_ids"].tolist(),
        },
        "uses_trajectory_or_eval_results": False,
    }
    (args.output_root / "system_diversity_manifest.json").write_text(
        json.dumps(system_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rollout_manifest = {
        "source_hashes": source_hashes,
        "extra_rollout_seed": EXTRA_ROLLOUT_SEED,
        "same_system_parameters": True,
        "same_action_policy": True,
        "validation_rollout_pool": [0, 1, 2, 3],
        "pools": {"R2": [0, 1], "R4": [0, 1, 2, 3], "R8": list(range(8))},
    }
    (args.output_root / "interaction_manifest.json").write_text(
        json.dumps(rollout_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    r2 = {
        key: (value[:, :2].copy() if value.ndim >= 2 and value.shape[0] == cfg.train_systems
              and value.shape[1] == 4 else value.copy())
        for key, value in train.items()
    }
    write_derived_root(
        args.output_root / "r2", source_val_path, r2,
        {"experiment_axis": "interaction_diversity", "train_rollout_ids": [0, 1],
         "source_hashes": source_hashes},
        replace(cfg, rollouts_per_system=2),
    )

    extra_cfg = replace(cfg, rollouts_per_system=4)
    extra = simulate_split(
        cfg.train_systems, 0, np.random.SeedSequence(EXTRA_ROLLOUT_SEED), extra_cfg,
        system_parameters=(train["mass"], train["gamma"], train["stiffness"]),
    )
    r8 = {}
    rollout_keys = {"states", "actions", "touch", "contact"}
    for key, value in train.items():
        r8[key] = np.concatenate([value, extra[key]], axis=1) if key in rollout_keys else value.copy()
    write_derived_root(
        args.output_root / "r8", source_val_path, r8,
        {"experiment_axis": "interaction_diversity", "train_rollout_ids": list(range(8)),
         "original_rollout_ids": [0, 1, 2, 3], "new_rollout_ids": [4, 5, 6, 7],
         "extra_rollout_seed": EXTRA_ROLLOUT_SEED, "source_hashes": source_hashes},
        replace(cfg, rollouts_per_system=8),
    )

    for label, count in (("s10", 200), ("s25", 500), ("s50", 1000)):
        rows = ordering_rows[:count]
        subset = {key: value[rows].copy() for key, value in train.items()}
        write_derived_root(
            args.output_root / label, source_val_path, subset,
            {"experiment_axis": "system_diversity", "train_rollout_ids": [0, 1, 2, 3],
             "selected_system_ids": subset["system_ids"].tolist(), "selection_algorithm": ALGORITHM_VERSION,
             "selection_seed": ORDERING_SEED, "source_hashes": source_hashes,
             "physics_coverage": coverage(subset)},
            replace(cfg, train_systems=count),
        )

    checks = {
        "nested_system_subsets": bool(
            set(ordering_ids[:200]) < set(ordering_ids[:500]) < set(ordering_ids[:1000])
            < set(train["system_ids"])
        ),
        "r2_is_source_prefix": all(np.array_equal(r2[key], train[key][:, :2]) for key in rollout_keys),
        "r8_source_prefix_bitwise": all(np.array_equal(r8[key][:, :4], train[key]) for key in rollout_keys),
        "r8_parameters_exact": all(
            np.array_equal(r8[key], train[key]) for key in ("mass", "gamma", "stiffness", "system_ids")
        ),
        "validation_sha256_unchanged": all(
            file_sha256(args.output_root / label / "val.npz") == source_hashes["val"]
            for label in ("r2", "r8", "s10", "s25", "s50")
        ),
        "test_read": False,
        "source_coverage": coverage(train),
        "subset_coverage": {
            label.upper(): coverage(load_npz(args.output_root / label / "train.npz"))
            for label in ("s10", "s25", "s50")
        },
    }
    if not all(value for key, value in checks.items() if key not in {"test_read", "source_coverage", "subset_coverage"}):
        raise AssertionError(checks)
    (args.output_root / "consistency_check.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
