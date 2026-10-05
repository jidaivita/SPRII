"""Nested-particle numerical stability check for Articulated A and LQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.posterior import posterior_from_observation

from .lqa_prospective import (
    _context_nuisance,
    _jsonable,
    _load_jepa,
    _response_bank,
    _response_jacobian_bank,
    accessibility_bank,
    fixed_contexts,
    lqa_bank,
    particle_pool,
    sha256,
    system_pool,
)
from .model import SwimmerModel, prior_log_scale
from .s0_screen import _landmarks
from .waveforms import banks


def run(root: Path, config_path: Path, output_root: Path, contexts: int, levels: tuple[int, ...], shard_index: int, shard_count: int):
    root, config_path, output_root = Path(root), Path(config_path), Path(output_root)
    cfg = json.loads(config_path.read_text())
    base = json.loads((root / cfg["base_config"]).read_text())
    s0 = json.loads((root / cfg["s0_receipt"]).read_text())
    device = torch.device("cpu")
    learner, norms, training = _load_jepa(root, cfg, device)
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history.values()))), base["observation"]["landmark_count"])
    systems = system_pool(cfg["development"]["systems"], cfg["development"]["system_seed"], base["persistent_prior"])
    selected_all = fixed_contexts(len(systems), cfg["development"]["nuisance_realizations"], 6, contexts, "articulated-lqa-benchmark-v1")
    selected = selected_all.iloc[np.arange(len(selected_all)) % shard_count == shard_index]
    maximum = max(levels)
    particles = particle_pool(maximum, cfg["reference"]["geometry_scramble_seed"], base["persistent_prior"])
    parameter_scale = prior_log_scale(base["persistent_prior"])
    model = SwimmerModel(base["model"])
    rows = []
    for context in selected.itertuples(index=False):
        theta = systems[int(context.system_index)]
        ih, ie, iq, observed_h, _, _ = _context_nuisance(
            base, model, theta, history, query, landmarks,
            cfg["development"]["context_seed"], int(context.system_index), int(context.realization),
        )
        hindex = int(context.history_index)
        anchor_bank = _response_bank(model, particles, ih, history, landmarks)[:, hindex]
        candidate_mean, candidate_jac = _response_jacobian_bank(
            model, particles, ie, history, landmarks, parameter_scale,
            cfg["reference"]["finite_difference_log_step"],
        )
        query_mean, query_jac = _response_jacobian_bank(
            model, particles, iq, query, landmarks, parameter_scale,
            cfg["reference"]["finite_difference_log_step"],
        )
        for level in levels:
            posterior = posterior_from_observation(
                observed_h[hindex], anchor_bank[:level], base["observation"]["sensor_std"],
                np.array([1.0]), np.array([1.0]),
            ).weights
            a, local = accessibility_bank(
                particles[:level], posterior, candidate_mean[:level], candidate_jac[:level],
                query_jac[:level], norms["target_std"], base["observation"]["sensor_std"], parameter_scale,
            )
            lqa, raw = lqa_bank(
                learner, norms, posterior, (hindex, observed_h[hindex]), ih, ie, iq,
                candidate_mean[:level], query_mean[:level], history, query, landmarks, device,
            )
            for candidate in range(6):
                for qindex in range(6):
                    rows.append({
                        "system_index": int(context.system_index), "realization": int(context.realization),
                        "history_index": hindex, "candidate_index": candidate, "query_index": qindex,
                        "particle_level": level, "accessibility": float(a[candidate, qindex]),
                        "local_value": float(local[candidate, qindex]), "lqa": float(lqa[candidate, qindex]),
                        "raw_cka": float(raw[candidate, qindex]), "posterior_ess": float(1.0 / np.sum(posterior**2)),
                    })
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "geometry_stability_rows.csv.gz"
    pd.DataFrame(rows).to_csv(rows_path, index=False, compression="gzip")
    receipt = {
        "status": "ARTICULATED_LQA_GEOMETRY_STABILITY_SHARD_COMPLETE_OUTCOME_BLIND",
        "contexts": len(selected), "total_contexts": len(selected_all),
        "levels": list(levels), "shard_index": shard_index, "shard_count": shard_count,
        "rows_sha256": sha256(rows_path), "implementation_sha256": sha256(Path(__file__)),
        "source_lqa_implementation_sha256": sha256(root / "code/paper_c/swimmer/lqa_prospective.py"),
        "config_sha256": sha256(config_path), "jepa_sha256": training["checkpoint_hashes"]["jepa"],
        "learner_outcomes_read": False, "sealed_accessed": False,
    }
    (output_root / "geometry_stability_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path); p.add_argument("config", type=Path); p.add_argument("output_root", type=Path)
    p.add_argument("--contexts", type=int, default=12); p.add_argument("--levels", type=int, nargs="+", default=(32, 64, 128))
    p.add_argument("--shard-index", type=int, default=0); p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    print(json.dumps(_jsonable(run(a.root, a.config, a.output_root, a.contexts, tuple(a.levels), a.shard_index, a.shard_count)), sort_keys=True))


if __name__ == "__main__":
    main()
