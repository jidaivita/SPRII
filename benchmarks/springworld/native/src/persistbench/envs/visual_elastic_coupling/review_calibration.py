"""Reproducible bounded re-analysis of an existing development bank.

No source receipt is overwritten, no training or sealed data is read. Paired
image-resolution comparisons reuse exact MuJoCo states. True states enter only
renderer replays and evaluator diagnostics, never the visible estimator.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .schema import Config
from .calibration import track, track_interior, identify
from .identification import identify_visible, reference_rollout
from .rendering import render_states


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new attempt directory; never overwrite calibration")
    args.output.mkdir(parents=True)
    began = time.monotonic()
    tracking, fits, references, sources = [], [], [], []
    with (args.output/"progress.jsonl").open("w") as log:
        def emit(row):
            log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
            print(json.dumps(row), flush=True)
        for path in sorted((args.source/"episodes").glob("*/visible.npz")):
            visible = np.load(path, allow_pickle=False)
            private = np.load(path.with_name("private.npz"), allow_pickle=False)
            metadata = json.loads(path.with_name("private.json").read_text())
            if metadata.get("split") != "development":
                raise ValueError("development-only calibration")
            for file in (path, path.with_name("private.npz"), path.with_name("private.json")):
                sources.append(dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            cfg = Config(**metadata["config"])
            theta = [metadata["theta"][key] for key in ("m", "gamma", "k")]
            state = private["state"]
            reference = reference_rollout(theta, state[0], visible["actions"], cfg)
            fine = reference_rollout(theta, state[0], visible["actions"], cfg, rtol=1e-10)
            references.append(dict(episode=path.parent.name,
                max_error_vs_mujoco=float(np.max(np.abs(reference-state))),
                max_error_vs_tighter_solver=float(np.max(np.abs(reference-fine)))))
            for resolution in (64, 128):
                cfg = replace(cfg, resolution=resolution)
                images = visible["images"] if resolution == metadata["config"]["resolution"] else render_states(state, cfg)
                row = dict(episode=path.parent.name, kind=metadata["kind"], theta=theta,
                           resolution=resolution, field_width=cfg.field_width)
                tracking.append(dict(**row, interior_rmse_m=float(np.sqrt(np.mean((track_interior(images, cfg)-state[:, :4])**2))),
                                     coverage_rmse_m=float(np.sqrt(np.mean((track(images, cfg)-state[:, :4])**2)))))
                if metadata["kind"] not in ("forced", "free"):
                    continue
                for budget in (24, 48, len(images)):
                    old = identify(images[:budget], visible["actions"][:budget-1], cfg)
                    start = time.monotonic()
                    try:
                        fitted = identify_visible(images[:budget], visible["actions"][:budget-1], cfg, starts=args.starts)
                        recovered = [fitted["m"], fitted["gamma"], fitted["k"], fitted["k_over_m"]]
                        true_values = theta+[theta[2]/theta[0]]
                        errors = [None if a is None else abs(a-b)/b for a, b in zip(recovered, true_values)]
                        outcome = dict(**row, frames=budget, duration_s=(budget-1)*cfg.control_dt,
                                       fit=fitted, relative_errors_m_gamma_k_ratio=errors,
                                       legacy_integrated_fit=old, seconds=time.monotonic()-start)
                    except (ValueError, FloatingPointError) as exc:
                        outcome = dict(**row, frames=budget, error=str(exc), legacy_integrated_fit=old,
                                       seconds=time.monotonic()-start)
                    fits.append(outcome)
                    emit(dict(episode=row["episode"], resolution=resolution, frames=budget,
                              status=outcome.get("fit", {}).get("status", "FAILED"),
                              relative_errors=outcome.get("relative_errors_m_gamma_k_ratio")))
    report = dict(schema="vec.calibration-review.v1.1", scope="development_reanalysis_only",
                  sources=sources, tracking=tracking, trajectory_fits=fits, solver_checks=references,
                  elapsed_seconds=time.monotonic()-began, training_run=False, sealed_read=False,
                  qualification="NOT_YET_QUALIFIED: small reused bank; need independent systems, coverage and conditional prediction")
    write_json(args.output/"report.json", report)
    source_files = [Path(__file__).with_name(x) for x in ("calibration.py", "identification.py", "review_calibration.py", "rendering.py")]
    write_json(args.output/"source_manifest.json", [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files])
    print(json.dumps(dict(status="COMPLETE", output=str(args.output), seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
