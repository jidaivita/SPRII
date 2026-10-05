#!/usr/bin/env python3
"""Generate R0 independent-reproduction corpus and train/val coverage only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from persistent_jepa.certificates import pokeworld_drag_estimates, r2_score_system
from persistent_jepa.pokeworld import PokeConfig, generate_pokeworld, save_pokeworld


def diagnostics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    object_speed = np.linalg.norm(payload["states"][..., 6:8], axis=-1)
    return {
        "contact_transition_fraction": float(payload["contact"].mean()),
        "contact_rollout_fraction": float(payload["contact"].any(axis=-1).mean()),
        "glide_transition_fraction": float((~payload["contact"] & (object_speed[..., :-1] > 1e-3)).mean()),
        "moving_object_fraction": float((object_speed.max(axis=-1) > 0.1).mean()),
        "finite_fraction": float(np.isfinite(payload["states"]).mean()),
        "max_abs_state": float(np.abs(payload["states"]).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = PokeConfig()
    dataset = generate_pokeworld(cfg)
    manifest = save_pokeworld(args.output, dataset, cfg)
    split_reports = {}
    for split in ("train", "val"):
        estimate, certificate_coverage = pokeworld_drag_estimates(
            dataset[split]["states"], dataset[split]["contact"], cfg.dt, cfg.substeps
        )
        split_reports[split] = {
            **diagnostics(dataset[split]),
            **certificate_coverage,
            "analytic_drag_system_r2": r2_score_system(dataset[split]["gamma"], estimate),
        }
    report = {
        "manifest_sha256": manifest["manifest_sha256"],
        **split_reports,
        "test_read": False,
    }
    (args.output / "preflight_train_val.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
