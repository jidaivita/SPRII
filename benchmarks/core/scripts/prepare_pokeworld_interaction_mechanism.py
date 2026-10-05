#!/usr/bin/env python3
"""Build frozen nested pseudo-system datasets for interaction-mechanism controls."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np

from persistent_jepa.pokeworld import PokeConfig


PSEUDO_SEED = 20260825
ALGORITHM_VERSION = "column-permutation-row-collision-rejection-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_free_permutations(systems: int, columns: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Independently propose each column until every row maps to a new true system."""
    if systems < columns:
        raise ValueError("systems must be at least the number of rollout columns")
    rng = np.random.default_rng(seed)
    permutations = np.empty((columns, systems), dtype=np.int64)
    attempts = []
    for column in range(columns):
        for attempt in range(1, 100_001):
            candidate = rng.permutation(systems)
            if column == 0 or np.all(candidate[None, :] != permutations[:column]):
                permutations[column] = candidate
                attempts.append(attempt)
                break
        else:
            raise RuntimeError(f"failed to find collision-free permutation for column {column}")
    return permutations, attempts


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def pseudo_payload(source: dict[str, np.ndarray], permutations: np.ndarray, rollouts: int) -> dict[str, np.ndarray]:
    systems = source["states"].shape[0]
    if source["states"].shape[1] < rollouts:
        raise ValueError("source does not contain requested rollout pool")
    selected = permutations[:rollouts]
    payload = {}
    for key in ("states", "actions", "touch", "contact"):
        payload[key] = np.stack(
            [source[key][selected[column], column] for column in range(rollouts)], axis=1
        )
    # A pseudo-system has no legitimate single physical label. NaNs make accidental
    # supervision/probe use fail loudly while remaining unused by B2/B3/Bx objectives.
    for key in ("mass", "gamma", "stiffness"):
        payload[key] = np.full(systems, np.nan, dtype=np.float32)
    payload["system_ids"] = np.arange(systems, dtype=np.int64)
    payload["source_system_ids_by_rollout"] = selected.T.copy()
    return payload


def shapes(payload: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
    }


def write_root(
    root: Path,
    payload: dict[str, np.ndarray],
    source_val: Path,
    cfg: PokeConfig,
    metadata: dict[str, object],
) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    temporary = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        np.savez_compressed(temporary / "train.npz", **payload)
        try:
            os.link(source_val, temporary / "val.npz")
            validation_storage = "hardlink-to-frozen-correct-validation"
        except OSError:
            shutil.copy2(source_val, temporary / "val.npz")
            validation_storage = "byte-copy-of-frozen-correct-validation"
        manifest = {
            "protocol": "pokeworld-interaction-mechanism-random-grouping-v1",
            "config": cfg.__dict__,
            "metadata": {**metadata, "validation_storage": validation_storage},
            "splits": {"train": shapes(payload)},
            "source_val_sha256": file_sha256(source_val),
            "pseudo_parameter_semantics": "invalid; NaN; prohibited for supervision/probe/decoder",
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
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("SPRII_WORK_ROOT", ".")))
    args = parser.parse_args()
    project = args.project_root.resolve()
    r8_root = project / "data" / "pokeworld_scaling_v1" / "r8"
    r4_root = project / "data" / "pokeworld_r0_v1"
    output = project / "data" / "pokeworld_interaction_mechanism_v1"
    if output.exists():
        raise FileExistsError(output)
    r8_manifest = json.loads((r8_root / "manifest.json").read_text())
    r4_manifest = json.loads((r4_root / "manifest.json").read_text())
    cfg = PokeConfig(**r4_manifest["config"])
    source = load_npz(r8_root / "train.npz")
    systems = source["states"].shape[0]
    if source["states"].shape[:2] != (2000, 8):
        raise ValueError(f"expected frozen R8 source, got {source['states'].shape[:2]}")
    permutations, attempts = collision_free_permutations(systems, 8, PSEUDO_SEED)
    output.mkdir(parents=True)
    source_hashes = {
        "r8_manifest": file_sha256(r8_root / "manifest.json"),
        "r8_train": file_sha256(r8_root / "train.npz"),
        "r4_manifest": file_sha256(r4_root / "manifest.json"),
        "r4_train": file_sha256(r4_root / "train.npz"),
        "validation": file_sha256(r4_root / "val.npz"),
    }
    manifest = {
        "algorithm": ALGORITHM_VERSION,
        "seed": PSEUDO_SEED,
        "column_generation_attempts": attempts,
        "permutations": permutations.tolist(),
        "source_system_ids_by_rollout": permutations.T.tolist(),
        "source_hashes": source_hashes,
        "pools": {"P2": [0, 1], "P4": [0, 1, 2, 3], "P8": list(range(8))},
        "row_collision_count": int(sum(
            len(set(permutations[:, row].tolist())) != 8 for row in range(systems)
        )),
        "each_column_is_permutation": bool(all(
            np.array_equal(np.sort(permutations[column]), np.arange(systems))
            for column in range(8)
        )),
        "test_read": False,
    }
    if manifest["row_collision_count"] != 0 or not manifest["each_column_is_permutation"]:
        raise AssertionError(manifest)
    (output / "pseudo_system_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for rollouts in (2, 4, 8):
        payload = pseudo_payload(source, permutations, rollouts)
        write_root(
            output / f"random_r{rollouts}", payload, r4_root / "val.npz",
            replace(cfg, rollouts_per_system=rollouts),
            {
                "pseudo_rollouts": rollouts,
                "rollout_ids": list(range(rollouts)),
                "source_dataset_hashes": source_hashes,
                "pseudo_manifest_sha256": file_sha256(output / "pseudo_system_manifest.json"),
                "correct_probe_fit_root": str(
                    r4_root if rollouts == 4 else project / "data" / "pokeworld_scaling_v1" / f"r{rollouts}"
                ),
            },
        )
    checks = {
        "p2_p4_p8_nested": bool(
            np.array_equal(permutations[:2], permutations[:4][:2])
            and np.array_equal(permutations[:4], permutations[:8][:4])
        ),
        "row_collision_count": manifest["row_collision_count"],
        "each_column_is_permutation": manifest["each_column_is_permutation"],
        "same_frozen_r8_hash": source_hashes["r8_train"],
        "r8_prefix_matches_frozen_r4": bool(all(
            np.array_equal(source[key][:, :4], load_npz(r4_root / "train.npz")[key])
            for key in ("states", "actions", "touch", "contact")
        )),
        "pseudo_parameters_all_nan": bool(all(
            np.isnan(load_npz(output / f"random_r{r}" / "train.npz")[key]).all()
            for r in (2, 4, 8) for key in ("mass", "gamma", "stiffness")
        )),
        "validation_hash_unchanged": bool(all(
            file_sha256(output / f"random_r{r}" / "val.npz") == source_hashes["validation"]
            for r in (2, 4, 8)
        )),
        "test_read": False,
    }
    if not all(value for key, value in checks.items() if key not in {"row_collision_count", "same_frozen_r8_hash", "test_read"}):
        raise AssertionError(checks)
    if checks["row_collision_count"] != 0:
        raise AssertionError(checks)
    (output / "consistency_check.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
