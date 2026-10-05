import argparse
import json
from pathlib import Path

import numpy as np

from .model import SwimmerModel, prior_log_scale
from .s0_screen import _landmarks, _response_and_jacobian, _sample_log_scales
from .waveforms import banks, development_alternatives


def _fisher(model, theta, initial, actions, landmarks, scale, config):
    _, jacobian = _response_and_jacobian(
        model, theta, initial, actions, landmarks, scale,
        config["s0"]["finite_difference_log_step"],
    )
    return jacobian.T @ jacobian / config["observation"]["sensor_std"] ** 2


def run(config_path, s0_root, output_path, contexts=64):
    config = json.loads(Path(config_path).read_text())
    s0 = json.loads((Path(s0_root) / "s0_receipt.json").read_text())
    duration = s0["chosen_horizon_s"]
    history, query = banks(duration, config["model"]["timestep_s"])
    alternatives = development_alternatives(duration, config["model"]["timestep_s"])
    steps = len(next(iter(history.values())))
    landmarks = _landmarks(steps, config["observation"]["landmark_count"])
    model = SwimmerModel(config["model"])
    scale = prior_log_scale(config["persistent_prior"])
    v1_history = dict(history)
    v1_history.pop("quadrature")
    v1_history["anti_phase"] = alternatives["anti_phase"]
    banks_to_test = {"v1_anti_phase": v1_history}
    for name, waveform in alternatives.items():
        if name == "anti_phase":
            continue
        candidate = dict(v1_history)
        candidate.pop("anti_phase")
        candidate[name] = waveform
        banks_to_test[f"replace_with_{name}"] = dict(sorted(candidate.items()))
    accumulators = {name: [] for name in banks_to_test}
    rng = np.random.default_rng(60311)
    for _ in range(contexts):
        theta = _sample_log_scales(rng, config["persistent_prior"])
        initial = model.sample_initial_state(rng, config["transient_initial_state"])
        all_history = {**history, **alternatives}
        fishers = {name: _fisher(model, theta, initial, action, landmarks, scale, config) for name, action in all_history.items()}
        query_fisher = {name: _fisher(model, theta, initial, action, landmarks, scale, config) for name, action in query.items()}
        for bank_name, bank in banks_to_test.items():
            names = list(bank)
            conditional = np.empty((6, 6, 6))
            standalone = np.empty((6, 6, 6))
            eye = np.eye(5)
            for h, hname in enumerate(names):
                cov_h = np.linalg.inv(eye + fishers[hname])
                for e, ename in enumerate(names):
                    cov_e = np.linalg.inv(eye + fishers[ename])
                    cov_he = np.linalg.inv(eye + fishers[hname] + fishers[ename])
                    for q, qname in enumerate(query):
                        conditional[h, e, q] = np.trace(query_fisher[qname] @ (cov_h-cov_he))
                        standalone[h, e, q] = np.trace(query_fisher[qname] @ (eye-cov_e))
            accumulators[bank_name].append((conditional, standalone, names))
    rows = []
    for bank_name, values in accumulators.items():
        conditional = np.mean([item[0] for item in values], axis=0)
        standalone = np.mean([item[1] for item in values], axis=0)
        names = values[0][2]
        conditional_choice = conditional.argmax(axis=1)
        standalone_choice = standalone.argmax(axis=1)
        disagreement = int(np.sum(conditional_choice != standalone_choice))
        counts = np.bincount(standalone_choice.ravel(), minlength=6)
        best = conditional.max(axis=1)
        chosen = np.take_along_axis(conditional, standalone_choice[:, None, :], axis=1).squeeze(1)
        rows.append({
            "bank": bank_name,
            "candidate_names": names,
            "conditional_standalone_choice_disagreement_of_36": disagreement,
            "standalone_max_winner_count_of_36": int(counts.max()),
            "standalone_winner": names[int(counts.argmax())],
            "standalone_regret": float(np.mean(best-chosen)),
        })
    rows.sort(key=lambda row: (-row["conditional_standalone_choice_disagreement_of_36"], row["standalone_max_winner_count_of_36"], -row["standalone_regret"]))
    receipt = {
        "status": "BOUNDED_PRE_LEARNER_PROBE_DIAGNOSIS_COMPLETE",
        "selection_rule": "maximize choice disagreement; tie-break lower global-winner count then larger standalone regret",
        "selected_bank": rows[0]["bank"],
        "diagnostics": rows,
        "contexts": contexts,
        "learner_used": False,
        "sealed_accessed": False,
    }
    Path(output_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("s0_root")
    parser.add_argument("output")
    parser.add_argument("--contexts", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.s0_root, args.output, args.contexts), sort_keys=True))


if __name__ == "__main__":
    main()
