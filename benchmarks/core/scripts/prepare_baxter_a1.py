#!/usr/bin/env python3
"""Audit the public Baxter bank and freeze Paper A A1 manifests before model output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from persistent_jepa.baxter_data import (
    CONFIGS,
    HARDNESS_LEVELS,
    MEASURED_HARDNESS,
    SHAPES,
    find_dataset_root,
    load_peak,
    parse_grasp_id,
)
from persistent_jepa.runtime import atomic_json, sha256_file


SOURCE_SHA256 = "a7d3782b29df55d46313ca0d3a9b1bd37a400fc15bcb4af12fc6072533f2643c"
SPLIT_SEED = 20260822


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.source_archive) != SOURCE_SHA256:
        raise ValueError("Baxter source archive does not match the frozen SHA-256")
    dataset = find_dataset_root(args.extracted_root)
    ids_by_config: dict[str, set[int]] = {}
    paths_by_config: dict[str, dict[int, Path]] = {}
    for shape in SHAPES:
        source_shape = "Cubes" if shape == "cube" else "Cylinders"
        for level in HARDNESS_LEVELS:
            config = f"{shape}_h{level}"
            paths = sorted((dataset / source_shape / "80_Samples" / str(level)).glob("*.csv"))
            keyed = {parse_grasp_id(path): path for path in paths}
            if len(paths) != 170 or len(keyed) != 170:
                raise ValueError(f"{config} does not contain 170 unique grasp files")
            for path in paths:
                load_peak(path)
            ids_by_config[config] = set(keyed)
            paths_by_config[config] = keyed
    common_ids = set.intersection(*ids_by_config.values())
    if common_ids != set(range(1, 171)):
        raise ValueError("the six configurations do not share grasp ids 1..170")
    permutation = np.random.default_rng(SPLIT_SEED).permutation(sorted(common_ids)).tolist()
    split_ids = {
        "train": set(permutation[:110]),
        "validation": set(permutation[110:140]),
        "confirmation": set(permutation[140:]),
    }
    records = []
    train_values = []
    for config in CONFIGS:
        shape, level_text = config.rsplit("_h", 1)
        level = int(level_text)
        for grasp_id in sorted(common_ids):
            split = next(name for name, ids in split_ids.items() if grasp_id in ids)
            path = paths_by_config[config][grasp_id]
            value = load_peak(path)
            if split == "train":
                train_values.append(value)
            records.append({
                "record_id": f"{config}_g{grasp_id:03d}",
                "config_id": config,
                "shape": shape,
                "hardness_level": level,
                "measured_hardness": MEASURED_HARDNESS[config],
                "grasp_id": grasp_id,
                "split": split,
                "relative_path": str(path.relative_to(args.extracted_root)),
                "sha256": sha256_file(path),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "A1_BAXTER_DATA_MANIFEST_v1.0.json"
    normalization_path = args.output_dir / "A1_BAXTER_NORMALIZATION_v1.0.json"
    pairing_path = args.output_dir / "A1_BAXTER_PAIRING_v1.0.json"
    manifest = {
        "schema_version": "paper-a-a1-baxter-v1.0",
        "status": "FROZEN_BEFORE_MODEL_OUTPUT",
        "source_doi": "10.5281/zenodo.18246104",
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_bytes": args.source_archive.stat().st_size,
        "window_samples": 80,
        "tactile_channels": 16,
        "registered_variant": "80_Samples_only",
        "split_seed": SPLIT_SEED,
        "split_counts_per_configuration": {"train": 110, "validation": 30, "confirmation": 30},
        "configurations": list(CONFIGS),
        "measured_hardness": MEASURED_HARDNESS,
        "records": records,
        "confirmation_accessed": False,
    }
    atomic_json(manifest_path, manifest)
    stacked = np.stack(train_values).reshape(-1, 16).astype(np.float64)
    atomic_json(normalization_path, {
        "schema_version": "paper-a-a1-baxter-normalization-v1.0",
        "status": "TRAIN_SPLIT_ONLY",
        "manifest_sha256": sha256_file(manifest_path),
        "channel_mean": stacked.mean(axis=0).tolist(),
        "channel_std": stacked.std(axis=0, ddof=0).clip(min=1e-8).tolist(),
        "train_grasps": len(train_values),
        "train_scalar_observations_per_channel": int(stacked.shape[0]),
    })
    atomic_json(pairing_path, {
        "schema_version": "paper-a-a1-baxter-pairing-v1.0",
        "status": "FROZEN_BEFORE_MODEL_OUTPUT",
        "conditions": {
            "R_H": {"equal": ["hardness_level"], "different": ["shape"]},
            "R_S": {"equal": ["shape"], "different": ["hardness_level"]},
            "Random": {"equal": [], "different": ["config_id"]},
        },
        "batch_pairs": 48,
        "queries_per_configuration": 8,
        "donor_marginals": "exactly_balanced_per_batch",
        "donors_used_per_query": 1,
    })
    atomic_json(args.output_dir / "A1_BAXTER_AUDIT_RECEIPT_v1.0.json", {
        "status": "PASS__READY_FOR_IMPLEMENTATION_SMOKE",
        "source_archive_sha256": SOURCE_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "normalization_sha256": sha256_file(normalization_path),
        "pairing_sha256": sha256_file(pairing_path),
        "registered_physical_configurations": 6,
        "registered_unique_grasps": 1020,
        "excluded_alternate_40_sample_files": 1360,
        "primary_evidence_boundary": "held_out_grasp_interactions_from_known_configurations",
    })


if __name__ == "__main__":
    main()
