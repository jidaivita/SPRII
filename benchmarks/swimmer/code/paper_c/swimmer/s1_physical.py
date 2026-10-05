import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import SwimmerModel, prior_log_scale
from .s0_screen import _landmarks, _response_and_jacobian, _sample_log_scales
from .waveforms import banks


_WORKER_CONFIG = None
_WORKER_MODEL = None
_WORKER_SCALE = None
_WORKER_HORIZON = None


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _init_worker(config, horizon):
    global _WORKER_CONFIG, _WORKER_MODEL, _WORKER_SCALE, _WORKER_HORIZON
    _WORKER_CONFIG = config
    _WORKER_MODEL = SwimmerModel(config["model"])
    _WORKER_SCALE = prior_log_scale(config["persistent_prior"])
    _WORKER_HORIZON = horizon


def _fisher_bank(theta, initial, bank, landmarks):
    sensor_std = _WORKER_CONFIG["observation"]["sensor_std"]
    fishers, means = [], []
    for actions in bank.values():
        mean, jacobian = _response_and_jacobian(
            _WORKER_MODEL, theta, initial, actions, landmarks, _WORKER_SCALE,
            _WORKER_CONFIG["s0"]["finite_difference_log_step"],
        )
        means.append(mean)
        fishers.append(jacobian.T @ jacobian / sensor_std ** 2)
    return np.asarray(fishers), np.asarray(means)


def _system_rows(system_index):
    config = _WORKER_CONFIG
    seed = config["s1"]["seed"]
    system_rng = np.random.default_rng(np.random.SeedSequence([seed, system_index, 0]))
    theta = _sample_log_scales(system_rng, config["persistent_prior"])
    history, query = banks(_WORKER_HORIZON, config["model"]["timestep_s"])
    history_names, query_names = list(history), list(query)
    steps = len(next(iter(history.values())))
    landmarks = _landmarks(steps, config["observation"]["landmark_count"])
    action_flat = np.asarray([value.ravel() for value in history.values()])
    rows = []
    for realization in range(config["s1"]["nuisance_realizations"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, system_index, realization, 1]))
        initial_h = _WORKER_MODEL.sample_initial_state(rng, config["transient_initial_state"])
        initial_e = _WORKER_MODEL.sample_initial_state(rng, config["transient_initial_state"])
        initial_q = _WORKER_MODEL.sample_initial_state(rng, config["transient_initial_state"])
        fisher_h, _ = _fisher_bank(theta, initial_h, history, landmarks)
        fisher_e, mean_e = _fisher_bank(theta, initial_e, history, landmarks)
        fisher_q, _ = _fisher_bank(theta, initial_q, query, landmarks)
        initial_obs_e = np.concatenate([initial_e.qpos[2:], initial_e.qvel])
        response_e = mean_e.reshape(6, -1, 8) - initial_obs_e[None, None, :]
        eye = np.eye(5)
        for anchor in range(6):
            covariance_h = np.linalg.inv(eye + fisher_h[anchor])
            for query_index in range(6):
                remaining = float(np.trace(fisher_q[query_index] @ covariance_h))
                for candidate in range(6):
                    covariance_e = np.linalg.inv(eye + fisher_e[candidate])
                    covariance_pair = np.linalg.inv(eye + fisher_h[anchor] + fisher_e[candidate])
                    conditional = float(np.trace(fisher_q[query_index] @ (covariance_h - covariance_pair)))
                    standalone = float(np.trace(fisher_q[query_index] @ (eye - covariance_e)))
                    rows.append({
                        "system_index": system_index,
                        "realization": realization,
                        "anchor_index": anchor,
                        "anchor_probe": history_names[anchor],
                        "candidate_index": candidate,
                        "candidate_probe": history_names[candidate],
                        "query_index": query_index,
                        "query": query_names[query_index],
                        "conditional_physical_value": conditional,
                        "standalone_query_value": standalone,
                        "trajectory_diversity": float(np.mean((response_e[anchor] - response_e[candidate]) ** 2)),
                        "action_diversity": float(np.mean((action_flat[anchor] - action_flat[candidate]) ** 2)),
                        "oracle_reducible_value": remaining,
                        "posterior_condition_number": float(np.linalg.cond(covariance_pair)),
                    })
    return rows


def _bootstrap_cell_means(values, seed, replicates):
    rng = np.random.default_rng(seed)
    results = []
    remaining = replicates
    while remaining:
        batch = min(50, remaining)
        sampled = rng.integers(0, values.shape[0], size=(batch, values.shape[0]))
        results.append(values[sampled].mean(axis=1))
        remaining -= batch
    return np.concatenate(results, axis=0)


def _resolved_reversals(system_values, seed, replicates):
    aggregate = system_values.mean(axis=0)
    bootstrap = _bootstrap_cell_means(system_values, seed, replicates)
    signs = np.zeros((6, 6, 6, 6), dtype=np.int8)
    for anchor in range(6):
        for query in range(6):
            for first in range(6):
                for second in range(first + 1, 6):
                    difference = aggregate[anchor, first, query] - aggregate[anchor, second, query]
                    bootstrap_difference = bootstrap[:, anchor, first, query] - bootstrap[:, anchor, second, query]
                    low = float(np.quantile(bootstrap_difference, .025))
                    high = float(np.quantile(bootstrap_difference, .975))
                    if difference > 0 and low > 0:
                        signs[anchor, first, second, query] = 1
                    elif difference < 0 and high < 0:
                        signs[anchor, first, second, query] = -1
    query_reversals = 0
    for anchor in range(6):
        for first in range(6):
            for second in range(first + 1, 6):
                sequence = signs[anchor, first, second]
                query_reversals += sum(sequence[a] * sequence[b] == -1 for a in range(6) for b in range(a + 1, 6))
    history_reversals = 0
    for query in range(6):
        for first in range(6):
            for second in range(first + 1, 6):
                sequence = signs[:, first, second, query]
                history_reversals += sum(sequence[a] * sequence[b] == -1 for a in range(6) for b in range(a + 1, 6))
    return query_reversals, history_reversals


def _selector_analysis(system_arrays, seed, replicates):
    physical = system_arrays["conditional_physical_value"]
    proxies = {key: system_arrays[key] for key in (
        "conditional_physical_value", "standalone_query_value", "trajectory_diversity", "action_diversity"
    )}
    nsystems = physical.shape[0]
    folds = np.arange(nsystems) % 2
    regret_rows = {key: np.zeros(nsystems) for key in proxies}
    choices = {key: np.zeros((2, 6, 6), dtype=int) for key in proxies}
    for fold in (0, 1):
        source, target = folds != fold, folds == fold
        for name, proxy in proxies.items():
            choice = proxy[source].mean(axis=0).argmax(axis=1)
            choices[name][fold] = choice
            selected = np.take_along_axis(physical[target], choice[None, :, None, :], axis=2).squeeze(2)
            best = physical[target].max(axis=2)
            regret_rows[name][target] = (best - selected).mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    summary = []
    for name, values in regret_rows.items():
        sampled = values[rng.integers(0, nsystems, size=(replicates, nsystems))].mean(axis=1)
        summary.append({"selector": name, "mean_regret": float(values.mean()),
                        "ci_low": float(np.quantile(sampled, .025)), "ci_high": float(np.quantile(sampled, .975))})
    differences = []
    base = regret_rows["conditional_physical_value"]
    for name in ("standalone_query_value", "trajectory_diversity", "action_diversity"):
        values = regret_rows[name] - base
        sampled = values[rng.integers(0, nsystems, size=(replicates, nsystems))].mean(axis=1)
        differences.append({"comparison": f"{name}_minus_conditional", "mean": float(values.mean()),
                            "ci_low": float(np.quantile(sampled, .025)), "ci_high": float(np.quantile(sampled, .975))})
    disagreement = int(np.sum(choices["conditional_physical_value"] != choices["standalone_query_value"]))
    return pd.DataFrame(summary), pd.DataFrame(differences), disagreement


def run_s1(root, config_path, s0_root, output_root, workers):
    root, config_path, s0_root, output_root = map(Path, (root, config_path, s0_root, output_root))
    config = json.loads(config_path.read_text())
    s0 = json.loads((s0_root / "s0_receipt.json").read_text())
    if s0["status"] != "S0_GO" or s0["config_sha256"] != _sha256(config_path):
        raise RuntimeError("S1 requires a GO S0 receipt bound to the exact config")
    if any(config["access"].values()):
        raise RuntimeError("S1 must remain isolated from Paper A/B, sealed, NAD, and GPU training")
    output_root.mkdir(parents=True, exist_ok=True)
    horizon = s0["chosen_horizon_s"]
    context = mp.get_context("spawn")
    workers = max(1, min(workers, config["s1"]["systems"]))
    with context.Pool(workers, initializer=_init_worker, initargs=(config, horizon)) as pool:
        chunks = pool.imap(_system_rows, range(config["s1"]["systems"]), chunksize=1)
        rows = [row for chunk in chunks for row in chunk]
    cells = pd.DataFrame(rows)
    cells.to_csv(output_root / "physical_value_rows.csv.gz", index=False, compression="gzip")
    metrics = ["conditional_physical_value", "standalone_query_value", "trajectory_diversity", "action_diversity", "oracle_reducible_value"]
    system_cells = cells.groupby(["system_index", "anchor_index", "candidate_index", "query_index"], as_index=False)[metrics].mean()
    arrays = {}
    for metric in metrics:
        arrays[metric] = system_cells.pivot_table(index="system_index", columns=["anchor_index", "candidate_index", "query_index"], values=metric).to_numpy().reshape(-1, 6, 6, 6)
    replicates = config["s1"]["bootstrap_replicates"]
    query_reversals, history_reversals = _resolved_reversals(
        arrays["conditional_physical_value"], config["s1"]["seed"] + 10, replicates
    )
    selectors, comparisons, choice_disagreement = _selector_analysis(arrays, config["s1"]["seed"] + 20, replicates)
    selectors.to_csv(output_root / "selector_regret.csv", index=False)
    comparisons.to_csv(output_root / "selector_regret_comparisons.csv", index=False)

    physical = arrays["conditional_physical_value"]
    remaining = arrays["oracle_reducible_value"][:, :, 0, :]
    aggregate = physical.mean(axis=0)
    best = aggregate.argmax(axis=1)
    worst = aggregate.argmin(axis=1)
    ratios = np.empty((physical.shape[0], 6, 6))
    for anchor in range(6):
        for query in range(6):
            ratios[:, anchor, query] = (
                physical[:, anchor, best[anchor, query], query] - physical[:, anchor, worst[anchor, query], query]
            ) / np.maximum(remaining[:, anchor, query], 1e-12)
    boot_ratio = _bootstrap_cell_means(ratios, config["s1"]["seed"] + 30, replicates)
    ratio_lower = np.quantile(boot_ratio, .025, axis=0)
    ratio_threshold = s0["s1_practical_ratio_threshold_frozen"]
    practical_contexts = int(np.sum(ratio_lower >= ratio_threshold))
    max_condition = float(cells.posterior_condition_number.max())
    finite = bool(np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all())
    comparison_go = bool((comparisons.ci_low > 0).all())
    status = "S1_GO" if (query_reversals >= config["s1"]["minimum_query_reversals"]
                           and history_reversals >= config["s1"]["minimum_history_reversals"]
                           and practical_contexts >= config["s1"]["minimum_practical_contexts"]
                           and choice_disagreement >= config["s1"]["minimum_conditional_standalone_choice_disagreement"]
                           and comparison_go and finite and max_condition < config["s1"]["maximum_posterior_condition_number"]) else "S1_NO_GO"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(selectors.selector, selectors.mean_regret, color=["#315a8a", "#777777", "#b5523b", "#c49a3a"])
    axes[0].tick_params(axis="x", rotation=30); axes[0].set_ylabel("cross-fit physical selection regret")
    axes[1].hist(ratios.mean(axis=0).ravel(), bins=12, color="#315a8a")
    axes[1].axvline(ratio_threshold, color="black", ls="--"); axes[1].set_xlabel("practical value-range ratio")
    fig.tight_layout(); fig.savefig(output_root / "s1_physical_conditionality.png", dpi=180); plt.close(fig)

    receipt = {
        "status": status,
        "systems": config["s1"]["systems"],
        "nuisance_realizations": config["s1"]["nuisance_realizations"],
        "rows": len(cells),
        "query_conditioned_resolved_reversals": query_reversals,
        "history_conditioned_resolved_reversals": history_reversals,
        "conditional_vs_standalone_choice_disagreements_across_two_folds": choice_disagreement,
        "practical_contexts_passing": practical_contexts,
        "practical_contexts_total": 36,
        "frozen_practical_ratio_threshold": ratio_threshold,
        "selector_regret": selectors.to_dict(orient="records"),
        "selector_comparisons": comparisons.to_dict(orient="records"),
        "maximum_posterior_condition_number": max_condition,
        "all_numeric_finite": finite,
        "physical_score_semantics": "ORACLE_LOCAL_FISHER_CONDITIONED_ON_EXACT_OBSERVED_INITIAL_STATES",
        "candidate_initial_state_pairing": "COMMON_RANDOM_NUMBERS_WITHIN_SYSTEM_REALIZATION",
        "initial_state_diversity_used_as_proxy": False,
        "s0_receipt_sha256": _sha256(s0_root / "s0_receipt.json"),
        "config_sha256": _sha256(config_path),
        "gpu_used": False,
        "protected_scope_2_touched": False,
        "sealed_accessed": False,
    }
    (output_root / "s1_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("config")
    parser.add_argument("s0_root")
    parser.add_argument("output_root")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_s1(args.root, args.config, args.s0_root, args.output_root, args.workers)), sort_keys=True))


if __name__ == "__main__":
    main()
