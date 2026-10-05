"""One-shot frozen-learner evaluation for the Articulated LQA assay.

The pair manifest must already be frozen outcome-blind.  Each unique learner
row is inferred once, pair contrasts are averaged within physical system, and
physical systems receive equal weight in the primary fixed-design analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.learner import _predictions_and_z
from paper_c.coupled_sled.learner_data import LearnerArrays

from .lqa_formal import formal_context_table
from .lqa_prospective import (
    _context_nuisance,
    _landmark_actions,
    _landmarks,
    _load_jepa,
    banks,
    system_pool,
    SwimmerModel,
)


PAIR_KEY = ["system_index", "realization", "history_index", "query_index", "candidate_low_id", "candidate_high_id"]
BASE_KEY = ["system_index", "realization", "history_index", "query_index"]
CANDIDATE_KEY = BASE_KEY[:3] + ["candidate_index", "query_index"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _arrays(history, mask, query, target, metadata, theta, candidate: bool) -> LearnerArrays:
    metadata = np.asarray(metadata, dtype=np.int64)
    return LearnerArrays(
        np.asarray(history, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
        np.asarray(query, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        np.full(len(metadata), 2 if candidate else 1, dtype=np.int64),
        metadata[:, 0],
        metadata[:, 2],
        metadata[:, 4] if candidate else metadata[:, 3],
        np.asarray(theta, dtype=np.float32),
    )


def build_unique_learner_rows(root: Path, reference_config: dict, manifest: pd.DataFrame) -> tuple[LearnerArrays, pd.DataFrame, LearnerArrays, pd.DataFrame]:
    """Reconstruct exact frozen contexts and deduplicate learner calls."""

    base = json.loads((root / reference_config["base_config"]).read_text())
    s0 = json.loads((root / reference_config["s0_receipt"]).read_text())
    model = SwimmerModel(base["model"])
    history_bank, query_bank = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    landmarks = _landmarks(len(next(iter(history_bank.values()))), base["observation"]["landmark_count"])
    history_actions = np.asarray([_landmark_actions(value, landmarks) for value in history_bank.values()])
    query_actions = np.asarray([_landmark_actions(value, landmarks) for value in query_bank.values()])
    systems = system_pool(int(reference_config["formal"]["pool_max"]), int(reference_config["formal"]["system_seed"]), base["persistent_prior"])
    contexts = formal_context_table(reference_config).set_index("system_index")

    needed_candidates = set()
    for row in manifest.itertuples(index=False):
        needed_candidates.add((int(row.system_index), int(row.realization), int(row.history_index), int(row.lqa_selected_candidate), int(row.query_index)))
        needed_candidates.add((int(row.system_index), int(row.realization), int(row.history_index), int(row.other_candidate), int(row.query_index)))
    needed_baselines = sorted({(s, r, h, q) for s, r, h, _, q in needed_candidates})

    by_context: dict[tuple[int, int, int], tuple] = {}
    for system_index, realization, history_index, _ in needed_baselines:
        frozen = contexts.loc[system_index]
        if int(frozen.realization) != realization or int(frozen.history_index) != history_index:
            raise RuntimeError("manifest context does not match the frozen formal context table")
        context_key = (system_index, realization, history_index)
        if context_key not in by_context:
            by_context[context_key] = _context_nuisance(
                base, model, systems[system_index], history_bank, query_bank, landmarks,
                int(reference_config["formal"]["context_seed"]), system_index, realization,
            )

    baseline_history, baseline_mask, baseline_query, baseline_target, baseline_meta, baseline_theta = [], [], [], [], [], []
    candidate_history, candidate_mask, candidate_query, candidate_target, candidate_meta, candidate_theta = [], [], [], [], [], []
    for system_index, realization, history_index, query_index in needed_baselines:
        ih, ie, iq, observed_h, observed_e, observed_q = by_context[(system_index, realization, history_index)]
        init_h = np.concatenate([ih.qpos[2:], ih.qvel])
        init_q = np.concatenate([iq.qpos[2:], iq.qvel])
        anchor = np.concatenate([init_h, observed_h[history_index], history_actions[history_index]])
        qinput = np.concatenate([init_q, query_actions[query_index]])
        baseline_history.append(np.stack([anchor, np.zeros(48)]))
        baseline_mask.append([1.0, 0.0])
        baseline_query.append(qinput)
        baseline_target.append(observed_q[query_index])
        baseline_meta.append((system_index, realization, history_index, query_index))
        baseline_theta.append(systems[system_index])

    for system_index, realization, history_index, candidate_index, query_index in sorted(needed_candidates):
        ih, ie, iq, observed_h, observed_e, observed_q = by_context[(system_index, realization, history_index)]
        init_h = np.concatenate([ih.qpos[2:], ih.qvel])
        init_e = np.concatenate([ie.qpos[2:], ie.qvel])
        init_q = np.concatenate([iq.qpos[2:], iq.qvel])
        anchor = np.concatenate([init_h, observed_h[history_index], history_actions[history_index]])
        candidate = np.concatenate([init_e, observed_e[candidate_index], history_actions[candidate_index]])
        qinput = np.concatenate([init_q, query_actions[query_index]])
        candidate_history.append(np.stack([anchor, candidate]))
        candidate_mask.append([1.0, 1.0])
        candidate_query.append(qinput)
        candidate_target.append(observed_q[query_index])
        candidate_meta.append((system_index, realization, history_index, candidate_index, query_index))
        candidate_theta.append(systems[system_index])

    baseline_arrays = _arrays(baseline_history, baseline_mask, baseline_query, baseline_target, baseline_meta, baseline_theta, False)
    candidate_arrays = _arrays(candidate_history, candidate_mask, candidate_query, candidate_target, candidate_meta, candidate_theta, True)
    baseline_index = pd.DataFrame(baseline_meta, columns=BASE_KEY)
    candidate_index = pd.DataFrame(candidate_meta, columns=CANDIDATE_KEY)

    baseline_lookup = {tuple(row): i for i, row in enumerate(baseline_meta)}
    for i, row in enumerate(candidate_meta):
        b = baseline_lookup[(row[0], row[1], row[2], row[4])]
        if not np.array_equal(candidate_arrays.history[i, 0], baseline_arrays.history[b, 0]):
            raise RuntimeError("candidate and baseline anchors are not array-exact")
        if not np.array_equal(candidate_arrays.query_action[i], baseline_arrays.query_action[b]):
            raise RuntimeError("candidate and baseline query inputs are not array-exact")
        if not np.array_equal(candidate_arrays.target[i], baseline_arrays.target[b]):
            raise RuntimeError("candidate and baseline query targets are not array-exact")
    return baseline_arrays, baseline_index, candidate_arrays, candidate_index


def _cluster_interval(system_values: np.ndarray, seed: int, replicates: int) -> dict:
    values = np.asarray(system_values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("system-level interval requires at least two finite systems")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "systems": int(len(values)),
        "system_positive_fraction": float(np.mean(values > 0)),
    }


def _system_interval_from_pairs(pairs: pd.DataFrame, value: str, seed: int, replicates: int) -> dict:
    systems = pairs.groupby("system_index", sort=True)[value].mean()
    return _cluster_interval(systems.to_numpy(float), seed, replicates)


def _similarity_adjusted_interval(pairs: pd.DataFrame, seed: int, replicates: int) -> dict:
    """System-equal WLS estimate at zero action-query similarity.

    A separate intercept is fit for every unordered candidate family.  Those
    family intercepts are then averaged using the frozen assay's system-equal
    family weights.  Unlike centering the covariate at its observed mean, this
    genuinely asks whether the LQA-oriented contrast remains positive after
    removing the realized action-query-similarity advantage.
    """

    systems = np.sort(pairs.system_index.unique())

    def estimate(table: pd.DataFrame) -> tuple[float, bool]:
        cluster = "_bootstrap_system" if "_bootstrap_system" in table else "system_index"
        weights = 1.0 / table.groupby(cluster)[cluster].transform("size").to_numpy(float)
        weights = weights / weights.sum()
        similarity = table.delta_action_query_similarity_lqa_orientation.to_numpy(float)
        family = table.candidate_low_id.astype(str) + "_" + table.candidate_high_id.astype(str)
        dummy_frame = pd.get_dummies(family, dtype=float)
        dummy = dummy_frame.to_numpy()
        design = np.column_stack([dummy, similarity])
        root_weights = np.sqrt(weights)
        weighted_design = design * root_weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design,
            table.delta_gain_lqa_orientation.to_numpy(float) * root_weights,
            rcond=None,
        )
        if rank != design.shape[1]:
            return float("nan"), False
        family_weights = np.asarray([
            float(weights[(family == name).to_numpy()].sum()) for name in dummy_frame.columns
        ])
        family_weights /= family_weights.sum()
        return float(np.dot(family_weights, coefficients[: len(dummy_frame.columns)])), True

    point, identified = estimate(pairs)
    rng = np.random.default_rng(seed)
    boot = []
    grouped = {int(system): group for system, group in pairs.groupby("system_index", sort=True)}
    for _ in range(replicates):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        blocks = []
        for draw, system in enumerate(sampled):
            block = grouped[int(system)].copy()
            block["_bootstrap_system"] = draw
            blocks.append(block)
        value, valid = estimate(pd.concat(blocks, ignore_index=True))
        if valid:
            boot.append(value)
    identified_fraction = len(boot) / replicates
    if not identified or identified_fraction < 0.90:
        return {"mean": point, "ci_low": None, "ci_high": None, "identified": False, "bootstrap_identified_fraction": identified_fraction}
    return {
        "mean": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "identified": True,
        "bootstrap_identified_fraction": identified_fraction,
    }


def _verify_manifest(manifest_path: Path, receipt_path: Path, finalization_config_path: Path) -> tuple[pd.DataFrame, dict]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "ARTICULATED_LQA_PAIR_MANIFEST_FROZEN_OUTCOME_BLIND":
        raise RuntimeError("formal pair manifest is not frozen outcome-blind")
    if receipt.get("pair_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("formal pair manifest hash mismatch")
    if receipt.get("source_hashes", {}).get("finalization_config") != sha256(finalization_config_path):
        raise RuntimeError("formal finalization config hash mismatch")
    if receipt.get("learner_outcomes_read") is not False or receipt.get("sealed_accessed") is not False:
        raise RuntimeError("formal manifest receipt violated access boundaries")
    manifest = pd.read_csv(manifest_path)
    required = set(PAIR_KEY + ["lqa_selected_candidate", "other_candidate", "ref_b_delta_vb", "delta_action_query_similarity_lqa_orientation"])
    if not required.issubset(manifest.columns) or manifest.duplicated(PAIR_KEY).any() or len(manifest) == 0:
        raise RuntimeError("formal pair manifest has invalid pair semantics")
    if np.any(manifest.lqa_selected_candidate.to_numpy(int) == manifest.other_candidate.to_numpy(int)):
        raise RuntimeError("formal pair contains identical candidate orientations")
    return manifest, receipt


def evaluate(root: Path, reference_config_path: Path, finalization_config_path: Path, manifest_path: Path, manifest_receipt_path: Path, output_root: Path, device_name: str) -> dict:
    root, reference_config_path, finalization_config_path, manifest_path, manifest_receipt_path, output_root = map(
        Path, (root, reference_config_path, finalization_config_path, manifest_path, manifest_receipt_path, output_root)
    )
    if any((output_root / name).exists() for name in ("formal_learner_pair_results.csv.gz", "formal_learner_system_results.csv.gz", "formal_learner_receipt.json")):
        raise RuntimeError("formal learner outcome artifact already exists; refusing a second inference run")
    reference_config = json.loads(reference_config_path.read_text())
    finalization_config = json.loads(finalization_config_path.read_text())
    if (root / finalization_config["reference_config"]).resolve() != reference_config_path.resolve():
        raise RuntimeError("evaluator configs do not share the frozen reference binding")
    manifest, manifest_receipt = _verify_manifest(manifest_path, manifest_receipt_path, finalization_config_path)

    device = torch.device(device_name)
    model, norms, training_receipt = _load_jepa(root, reference_config, device)
    if training_receipt["checkpoint_hashes"]["jepa"] != manifest_receipt["reference_source_hashes"]["jepa"]:
        raise RuntimeError("formal learner checkpoint differs from the outcome-blind reference checkpoint")
    baseline, baseline_index, candidate, candidate_index = build_unique_learner_rows(root, reference_config, manifest)
    baseline_prediction, _ = _predictions_and_z(model, baseline, norms, json.loads((root / reference_config["s2_config"]).read_text()), device, True)
    candidate_prediction, _ = _predictions_and_z(model, candidate, norms, json.loads((root / reference_config["s2_config"]).read_text()), device, True)
    baseline_target = (baseline.target - norms["target_mean"]) / norms["target_std"]
    candidate_target = (candidate.target - norms["target_mean"]) / norms["target_std"]
    baseline_index["anchor_loss"] = np.mean((baseline_prediction - baseline_target) ** 2, axis=1)
    candidate_index["candidate_loss"] = np.mean((candidate_prediction - candidate_target) ** 2, axis=1)

    selected = candidate_index.rename(columns={"candidate_index": "lqa_selected_candidate", "candidate_loss": "lqa_selected_loss"})
    other = candidate_index.rename(columns={"candidate_index": "other_candidate", "candidate_loss": "other_loss"})
    pair_results = manifest.merge(baseline_index, on=BASE_KEY, validate="many_to_one")
    pair_results = pair_results.merge(selected, on=BASE_KEY[:3] + ["lqa_selected_candidate", "query_index"], validate="many_to_one")
    pair_results = pair_results.merge(other, on=BASE_KEY[:3] + ["other_candidate", "query_index"], validate="many_to_one")
    if len(pair_results) != len(manifest):
        raise RuntimeError("learner losses did not join one-to-one to the frozen pair manifest")
    pair_results["gain_lqa_selected"] = pair_results.anchor_loss - pair_results.lqa_selected_loss
    pair_results["gain_other"] = pair_results.anchor_loss - pair_results.other_loss
    pair_results["delta_gain_lqa_orientation"] = pair_results.gain_lqa_selected - pair_results.gain_other
    if not np.allclose(pair_results.delta_gain_lqa_orientation, pair_results.other_loss - pair_results.lqa_selected_loss, rtol=0.0, atol=1e-12):
        raise RuntimeError("learner-gain orientation invariant failed")

    seed = int(finalization_config["inference"]["bootstrap_seed"])
    reps = int(finalization_config["inference"]["bootstrap_replicates"])
    system_results = pair_results.groupby("system_index", sort=True).agg(
        delta_gain_lqa_orientation=("delta_gain_lqa_orientation", "mean"),
        pairs=("delta_gain_lqa_orientation", "size"),
    ).reset_index()
    primary = _cluster_interval(system_results.delta_gain_lqa_orientation.to_numpy(float), seed, reps)
    if primary["ci_low"] > 0:
        direction = "DIRECTIONAL_SUPPORT"
    elif primary["ci_high"] < 0:
        direction = "DIRECTIONAL_REVERSE"
    else:
        direction = "DIRECTIONAL_INCONCLUSIVE"
    sesoi = float(reference_config["primary"]["sesoi_absolute"])
    if primary["ci_low"] >= sesoi:
        materiality = "PRACTICAL_EFFECT_SUPPORTED"
    elif primary["ci_high"] < sesoi:
        materiality = "PRACTICAL_SIZED_POSITIVE_EFFECT_RULED_OUT"
    else:
        materiality = "MATERIALITY_UNRESOLVED"

    similarity = _similarity_adjusted_interval(pair_results, seed + 101, reps)
    orientation_counts = pair_results.groupby(["candidate_low_id", "candidate_high_id"]).lqa_selected_candidate.nunique()
    reversal_families = orientation_counts[orientation_counts >= 2].index
    reversal_mask = pd.MultiIndex.from_frame(pair_results[["candidate_low_id", "candidate_high_id"]]).isin(reversal_families)
    reversal = _system_interval_from_pairs(pair_results[reversal_mask], "delta_gain_lqa_orientation", seed + 202, reps) if reversal_mask.any() else None
    balance = manifest_receipt.get("residual_bayes") or {}
    realized_residual = max(float(balance.get("global_bound", np.inf)), float(balance.get("candidate_pair_weighted_rms_bound", np.inf)))
    if direction != "DIRECTIONAL_SUPPORT":
        claim_scope = "NOT_APPLICABLE_WITHOUT_DIRECTIONAL_SUPPORT"
    elif not balance.get("strong_attribution_balance_pass", False) or primary["ci_low"] <= realized_residual:
        claim_scope = "LQA_ASSOCIATED_WITHIN_MATCHED_ASSAY"
    elif not similarity["identified"] or similarity["ci_low"] <= 0:
        claim_scope = "BEYOND_RESIDUAL_BAYES_SUPPORTED_QUERY_SIMILARITY_UNRESOLVED"
    elif reversal is None or len(reversal_families) < 3 or reversal["ci_low"] <= 0:
        claim_scope = "BEYOND_RESIDUAL_BAYES_AND_QUERY_SIMILARITY_SUPPORTED_FIXED_WAVEFORM_UNRESOLVED"
    else:
        claim_scope = "BEYOND_RESIDUAL_BAYES_QUERY_SIMILARITY_AND_FIXED_WAVEFORM_SUPPORTED"

    ordered_systems = sorted(system_results.system_index.astype(int), key=lambda value: hashlib.sha256(f"articulated-lqa-repeatability|{value}".encode()).hexdigest())
    halves = []
    for offset in (0, 1):
        values = system_results.set_index("system_index").loc[ordered_systems[offset::2], "delta_gain_lqa_orientation"].to_numpy(float)
        halves.append(_cluster_interval(values, seed + 300 + offset, reps))
    raw_summary = _system_interval_from_pairs(pair_results, "delta_raw_cka_lqa_orientation", seed + 401, reps)
    low_ess = {}
    if "geometry_posterior_ess" in pair_results:
        system_ess = pair_results.groupby("system_index").geometry_posterior_ess.first()
        cutoff = float(system_ess.quantile(0.10))
        kept = set(system_ess[system_ess > cutoff].index)
        subset = pair_results[pair_results.system_index.isin(kept)]
        low_ess = {"p10_cutoff": cutoff, "excluding_lowest_decile": _system_interval_from_pairs(subset, "delta_gain_lqa_orientation", seed + 501, reps)}

    output_root.mkdir(parents=True, exist_ok=True)
    pair_path = output_root / "formal_learner_pair_results.csv.gz"
    system_path = output_root / "formal_learner_system_results.csv.gz"
    _atomic_csv(pair_path, pair_results)
    _atomic_csv(system_path, system_results)
    receipt = {
        "status": "ARTICULATED_LQA_FORMAL_LEARNER_EVALUATION_COMPLETE",
        "assay_validity": "PROSPECTIVE_VALID",
        "direction": direction,
        "materiality": materiality,
        "claim_scope": claim_scope,
        "repeatability": {"hash_halves": halves, "system_positive_fraction": primary["system_positive_fraction"]},
        "primary": primary,
        "supporting": {
            "query_similarity_and_candidate_pair_fe_adjusted": similarity,
            "candidate_pair_reversal_families": int(len(reversal_families)),
            "reversal_family_subset": reversal,
            "raw_cka_lqa_orientation": raw_summary,
            "low_ess_sensitivity": low_ess,
        },
        "residual_bayes": balance,
        "pairs": int(len(pair_results)),
        "contributing_systems": int(len(system_results)),
        "unique_baseline_rows": int(len(baseline_index)),
        "unique_candidate_rows": int(len(candidate_index)),
        "device": device_name,
        "models_retrained": False,
        "formal_inference_runs": 1,
        "interval_semantics": "fixed_design_system_bootstrap_stability_interval",
        "sealed_accessed": False,
        "source_hashes": {
            "reference_config": sha256(reference_config_path),
            "finalization_config": sha256(finalization_config_path),
            "manifest": sha256(manifest_path),
            "manifest_receipt": sha256(manifest_receipt_path),
            "jepa": training_receipt["checkpoint_hashes"]["jepa"],
            "normalization": training_receipt["checkpoint_hashes"]["normalization"],
            "implementation": sha256(Path(__file__)),
            "pair_results": sha256(pair_path),
            "system_results": sha256(system_path),
        },
    }
    _atomic_text(output_root / "formal_learner_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("reference_config", type=Path)
    parser.add_argument("finalization_config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manifest_receipt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    result = evaluate(args.root, args.reference_config, args.finalization_config, args.manifest, args.manifest_receipt, args.output_root, args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
