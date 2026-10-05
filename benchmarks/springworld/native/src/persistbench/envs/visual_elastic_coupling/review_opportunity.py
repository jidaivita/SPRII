"""Independent-donor prediction diagnostic on the prior cold-query examples.

These four systems are a reused development diagnostic, not a representative
evaluation population. The Sobol reference approximates an explicitly declared
continuous uniform prior. Because the cold query is theta-independent, no
theta posterior update is required here; that simplification is invalid for
general moving queries and is checked rather than silently assumed.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .schema import Config
from .calibration import track, component_errors
from .identification import reference_rollout
from .rendering import render_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh attempt directory")
    args.output.mkdir(parents=True)
    fits = json.loads(args.fit_report.read_text())["trajectory_fits"]
    queries = []
    for path in sorted((args.source/"queries").glob("*/visible.npz")):
        v = dict(np.load(path, allow_pickle=False))
        s = np.load(path.with_name("private.npz"), allow_pickle=False)["state"]
        meta = json.loads(path.with_name("private.json").read_text())
        if meta["split"] != "development":
            raise ValueError("development-only")
        # This profile declares a cold rest state. Validate before using it as a
        # public prior, including the previous observed zero-action interval.
        if not np.array_equal(v["actions"][0], [0, 0]) or np.max(np.abs(s[:2, 4:])) > 1e-12:
            raise ValueError("query is not a cold rest profile")
        queries.append((v, s, meta))
    if not all(np.array_equal(queries[0][0]["images"][:2], v["images"][:2]) and
               np.array_equal(queries[0][0]["actions"], v["actions"]) for v, _, _ in queries):
        raise ValueError("cold parameter-independent query equivalence not established")
    theta_samples = qmc.scale(qmc.Sobol(3, scramble=True, seed=20260909).random_base2(10),
                              [.5, .25, 4.], [2., 1.5, 25.])
    rows, convergence, contrasts = [], [], []
    for res in (64, 128):
        cfg = replace(Config(), resolution=res)
        for index, (v, s, meta) in enumerate(queries):
            query_images = v["images"][:2] if res == 64 else render_states(s[:2], cfg)
            visual_state = np.r_[track(query_images, cfg)[-1], np.zeros(4)]
            theta = [meta["theta"][k] for k in ("m", "gamma", "k")]
            actions = v["actions"][1:17]
            samples = np.array([reference_rollout(p, visual_state, actions, cfg)[-1]-visual_state
                                for p in theta_samples])
            null = samples.mean(axis=0)
            for n in (128, 256, 512):
                convergence.append(dict(query=index, resolution=res, points=n,
                    delta_to_1024=component_errors(samples[:n].mean(axis=0), null)))
            truth = s[17]-s[1]
            common = dict(query=index, resolution=res, horizon=16,
                          query_frames=2, query_past_actions=0,
                          query_sha256=hashlib.sha256(query_images.tobytes()+actions.tobytes()).hexdigest(),
                          target_sha256=hashlib.sha256(truth.tobytes()).hexdigest())
            predictions = {"query_only_uniform_prior_mean": null,
                           "same_visual_true_theta": reference_rollout(theta, visual_state, actions, cfg)[-1]-visual_state,
                           "privileged_state_theta": reference_rollout(theta, s[1], actions, cfg)[-1]-s[1]}
            contrast = np.mean((samples-null)**2, axis=0)
            contrasts.append(dict(query=index, resolution=res,
                conditional_theta_variance_position_m2=float(contrast[:4].mean()),
                conditional_theta_variance_velocity_m2_s2=float(contrast[4:].mean()),
                interpretation="variance under declared continuous theta prior at plug-in visual state; not an exact visual Bayes risk"))
            for budget in (24, 48, 97):
                for label, donor_index in (("matched", index), ("wrong", (index+1) % len(queries))):
                    donor_id = str(910003+100*donor_index)
                    match = [f for f in fits if f["episode"] == donor_id and f["resolution"] == res and f["frames"] == budget]
                    if len(match) != 1:
                        raise ValueError("missing or duplicate fit")
                    fitted = match[0]["fit"]
                    if any(fitted[k] is None for k in ("m", "gamma", "k")):
                        rows.append(dict(**common, condition=label, frames=budget, status="FIT_FAILED"))
                        continue
                    p = [fitted[k] for k in ("m", "gamma", "k")]
                    predictions[f"{label}_{budget}"] = reference_rollout(p, visual_state, actions, cfg)[-1]-visual_state
            for label, prediction in predictions.items():
                rows.append(dict(**common, condition=label, prediction=prediction.tolist(),
                                 error=component_errors(prediction, truth), status="EXECUTED"))
            print(json.dumps(dict(query=index, resolution=res, completed=True)), flush=True)
    result = dict(schema="vec.cold-opportunity-review.v1.1", scope="four reused development systems only",
                  query_equivalence_checked=True, query_known_rest_prior=True, results=rows,
                  continuous_prior="uniform m in [.5,2], gamma in [.25,1.5], k in [4,25], independent",
                  prior_approximation="1024 scrambled Sobol samples; common random numbers; plug-in visual initial positions",
                  convergence=convergence, conditional_variances=contrasts,
                  fit_report_sha256=hashlib.sha256(args.fit_report.read_bytes()).hexdigest(),
                  sealed_read=False, learned_q_only_run=False,
                  qualification="PRELIMINARY_ONLY; full-support calibration and learned baselines remain required")
    (args.output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
