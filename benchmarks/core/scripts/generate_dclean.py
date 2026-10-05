#!/usr/bin/env python3
"""Generate the frozen D-Clean v1 corpus and train/validation preflight report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from persistent_jepa.certificates import r2_score_system, system_gamma_estimates
from persistent_jepa.simulator import DCleanConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = DCleanConfig()
    dataset = generate_dataset(cfg)
    manifest = dataset.save(args.output)
    report = {"manifest_sha256": manifest["manifest_sha256"], "splits": {}}
    for split in ("train", "val"):
        estimate, diagnostics = system_gamma_estimates(
            dataset.states[split], dataset.actions[split], cfg.dt
        )
        report["splits"][split] = {
            **diagnostics,
            "analytic_gamma_r2": r2_score_system(dataset.gamma[split], estimate),
            "zero_force_transition_fraction": float(
                (np.linalg.norm(dataset.actions[split], axis=-1) <= 1e-10).mean()
            ),
            "nonzero_force_transition_fraction": float(
                (np.linalg.norm(dataset.actions[split], axis=-1) > 1e-10).mean()
            ),
        }
    report_path = args.output / "preflight_train_val.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
