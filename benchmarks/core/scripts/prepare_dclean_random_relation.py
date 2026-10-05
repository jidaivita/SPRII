#!/usr/bin/env python3
"""Create the frozen collision-free pseudo-system relation for Paper A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from persistent_jepa.sampling import build_collision_free_pseudo_systems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    train = np.load(args.data_root / "train.npz", mmap_mode="r")
    systems, rollouts = train["states"].shape[:2]
    mapping = build_collision_free_pseudo_systems(systems, rollouts, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output_dir / "dclean_random_b2_r8_relation.npy"
    np.save(mapping_path, mapping)
    manifest = {
        "environment": "D-Clean",
        "relation": "random_system_fixed",
        "algorithm": "seeded_base_permutation_plus_cyclic_column_shifts_v1",
        "seed": args.seed,
        "shape": list(mapping.shape),
        "mapping_file": mapping_path.name,
        "mapping_sha256": sha256(mapping_path),
        "dataset_manifest_sha256": sha256(args.data_root / "manifest.json"),
        "collision_check": bool(all(np.unique(row).size == rollouts for row in mapping)),
        "column_permutation_check": bool(
            all(np.unique(mapping[:, column]).size == systems for column in range(rollouts))
        ),
    }
    manifest_path = args.output_dir / "dclean_random_b2_r8_relation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
