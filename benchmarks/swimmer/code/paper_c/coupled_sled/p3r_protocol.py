"""Learner-blind gates and orthogonal verdict semantics for Coupled P3-R V3.

V3 keeps hard gates only for identified false-positive paths.  In particular,
reference-QMC uncertainty is separated from real between-system heterogeneity:
the former enters Bayes-balance gates, while the latter is reported only as a
claim-strength diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .accessibility_matching import _reject_learner_outcomes
from .accessibility_power import projected_one_sided_power


@dataclass(frozen=True)
class P3RGates:
    practical_mde: float
    ref_a_proposal_bound_fraction: float
    ref_b_pair_bound_fraction: float
    signed_global_bound_fraction: float
    candidate_pair_weighted_rms_bound_fraction: float
    action_energy_absolute_tolerance: float
    target_maximum_systems: int
    minimum_systems: int
    required_realizations: int
    minimum_power: float
    projected_system_std: float
    repeatability_label_threshold: float
    reference_t_multiplier: float
    reference_scrambles: int
    minimum_reversal_candidate_pairs_for_strong_claim: int


def gates_from_config(config: Mapping) -> P3RGates:
    """Construct the single canonical V3 gate set."""

    matching = config["matching_and_balance"]
    power = config["power"]
    claims = config["claim_strength"]
    reference = config["reference"]
    return P3RGates(
        practical_mde=float(config["practical_mde"]),
        ref_a_proposal_bound_fraction=float(
            matching["ref_a_proposal_bound_fraction_of_mde"]
        ),
        ref_b_pair_bound_fraction=float(
            matching["ref_b_pair_bound_fraction_of_mde"]
        ),
        signed_global_bound_fraction=float(
            matching["signed_global_numerical_bound_fraction_of_mde"]
        ),
        candidate_pair_weighted_rms_bound_fraction=float(
            matching["candidate_pair_weighted_rms_numerical_bound_fraction_of_mde"]
        ),
        action_energy_absolute_tolerance=float(
            matching["action_energy_absolute_tolerance"]
        ),
        target_maximum_systems=int(power["target_maximum_systems"]),
        minimum_systems=int(power["minimum_independent_systems"]),
        required_realizations=int(config["realizations"]),
        minimum_power=float(power["minimum_recomputed_power"]),
        projected_system_std=float(power["frozen_projected_system_std"]),
        repeatability_label_threshold=float(claims["repeatability_label_threshold"]),
        reference_t_multiplier=float(reference["student_t_multiplier_df3"]),
        reference_scrambles=int(reference["independent_owen_scrambles_per_stream"]),
        minimum_reversal_candidate_pairs_for_strong_claim=int(
            claims["minimum_reversal_candidate_pairs_for_strong_claim"]
        ),
    )


def _scramble_columns(frame: pd.DataFrame, stream: str, count: int) -> list[str]:
    columns = [f"reference_gap_{stream}_scramble_{index}" for index in range(count)]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing {stream} QMC scramble columns: {sorted(missing)}")
    return columns


def _row_qmc_mean_and_se(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("reference QMC scramble values must be finite")
    return values.mean(axis=1), values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])


def _aggregate_qmc_mean_and_se(
    frame: pd.DataFrame, columns: Sequence[str], *, system_column: str = "system_index",
) -> tuple[float, float, list[float]]:
    """Aggregate each QMC scramble using equal physical-system weighting.

    The standard error is across the four aggregate scramble estimates.  Real
    between-system dispersion is intentionally absent: the hard balance gate
    asks whether the fixed assay is numerically known to be balanced, not
    whether every physical system has the same signed Bayes gap.
    """

    estimates = []
    for column in columns:
        per_system = frame.groupby(system_column, sort=True)[column].mean()
        if per_system.empty:
            return float("nan"), float("inf"), []
        estimates.append(float(per_system.mean()))
    values = np.asarray(estimates, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))), estimates


def _candidate_pair_balance(
    pairs: pd.DataFrame, columns: Sequence[str], multiplier: float,
) -> tuple[float, list[dict]]:
    rows = []
    for candidate_pair, group in pairs.groupby("candidate_pair", sort=True):
        mean, numerical_se, estimates = _aggregate_qmc_mean_and_se(group, columns)
        per_system = group.assign(
            _ref_mean=group.loc[:, columns].mean(axis=1)
        ).groupby("system_index", sort=True)._ref_mean.mean()
        bound = abs(mean) + multiplier * numerical_se
        rows.append({
            "candidate_pair": str(candidate_pair),
            "systems": int(per_system.size),
            "oriented_gap_mean": mean,
            "qmc_numerical_se": numerical_se,
            "numerical_balance_bound": bound,
            "between_system_sd": float(per_system.std(ddof=1)) if len(per_system) >= 2 else float("nan"),
            "between_system_se_descriptive": (
                float(per_system.std(ddof=1) / np.sqrt(len(per_system)))
                if len(per_system) >= 2 else float("nan")
            ),
            "aggregate_scramble_estimates": estimates,
        })
    if not rows:
        return float("inf"), []
    weights = np.asarray([row["systems"] for row in rows], dtype=float)
    bounds = np.asarray([row["numerical_balance_bound"] for row in rows], dtype=float)
    weighted_rms = float(np.sqrt(np.average(np.square(bounds), weights=weights)))
    return weighted_rms, rows


def _standardized_mean(value: pd.Series, absolute_tolerance: float = 1e-15) -> float:
    mean = abs(float(value.mean()))
    standard_deviation = float(value.std(ddof=1))
    if standard_deviation <= absolute_tolerance:
        return 0.0 if mean <= absolute_tolerance else float("inf")
    return mean / standard_deviation


def filter_ref_a_proposals(
    pairs: pd.DataFrame, gates: P3RGates, *, development_proxy: bool = False,
) -> pd.DataFrame:
    """Apply the permissive ref-A proposal screen without secondary deletion."""

    _reject_learner_outcomes(pairs.columns)
    required = {
        "system_index", "realization", "history_index", "query_index",
        "candidate_pair", "candidate_high_a", "reference_value_abs_gap",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"P3-R proposal table lacks columns: {sorted(missing)}")
    caliper = gates.ref_a_proposal_bound_fraction * gates.practical_mde
    if development_proxy:
        reference_bound = pairs.reference_value_abs_gap.to_numpy(dtype=float)
    else:
        columns = _scramble_columns(pairs, "ref_a", gates.reference_scrambles)
        mean, se = _row_qmc_mean_and_se(pairs, columns)
        reference_bound = np.abs(mean) + gates.reference_t_multiplier * se
    selected = pairs[reference_bound <= caliper].copy()
    return selected.sort_values([
        "system_index", "realization", "history_index", "query_index", "candidate_pair"
    ]).reset_index(drop=True)


def deterministic_system_subset(
    pairs: pd.DataFrame, maximum: int, namespace: str,
) -> tuple[pd.DataFrame, list[int]]:
    """Take at most ``maximum`` complete systems in a frozen hash order."""

    systems = sorted(map(int, pairs.system_index.unique()))
    ordered = sorted(
        systems,
        key=lambda value: hashlib.sha256(f"{namespace}|{value}".encode()).digest(),
    )
    chosen = ordered[: min(maximum, len(ordered))]
    return pairs[pairs.system_index.isin(chosen)].copy(), chosen


def verify_ref_b_and_balance(
    pairs: pd.DataFrame, gates: P3RGates, *, namespace: str = "p3r-formal-system-v3",
) -> tuple[pd.DataFrame, dict]:
    """Ref-B filter, final hash subset, then final-set hard-gate verification."""

    _reject_learner_outcomes(pairs.columns)
    required = {
        "system_index", "realization", "history_index", "query_index",
        "candidate_pair", "candidate_high_a", "action_energy_difference",
        "action_query_similarity_difference", "local_valid",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"P3-R ref-B table lacks columns: {sorted(missing)}")
    columns = _scramble_columns(pairs, "ref_b", gates.reference_scrambles)
    row_mean, row_se = _row_qmc_mean_and_se(pairs, columns)
    working = pairs.assign(reference_gap_ref_b=row_mean, reference_gap_ref_b_se=row_se)
    row_bound = np.abs(row_mean) + gates.reference_t_multiplier * row_se
    working = working[
        (row_bound <= gates.ref_b_pair_bound_fraction * gates.practical_mde)
        & working.local_valid.astype(bool)
    ].copy()

    complete = working.groupby("system_index").realization.nunique()
    complete_systems = set(complete[complete == gates.required_realizations].index)
    working = working[working.system_index.isin(complete_systems)].copy()
    final, chosen = deterministic_system_subset(working, gates.target_maximum_systems, namespace)

    global_mean, global_numerical_se, global_scrambles = _aggregate_qmc_mean_and_se(final, columns)
    global_bound = abs(global_mean) + gates.reference_t_multiplier * global_numerical_se
    pair_rms, pair_rows = _candidate_pair_balance(final, columns, gates.reference_t_multiplier)
    systems = int(final.system_index.nunique())
    projected_power, projected_se = (
        projected_one_sided_power(
            gates.practical_mde, gates.projected_system_std, systems
        ) if systems >= 2 else (0.0, float("inf"))
    )
    energy_max = (
        float(final.action_energy_difference.abs().max()) if len(final) else float("inf")
    )
    action_similarity_smd = (
        _standardized_mean(final.action_query_similarity_difference)
        if len(final) else float("inf")
    )
    orientations = final.groupby("candidate_pair").candidate_high_a.nunique()
    reversals = int((orientations >= 2).sum())
    hard_checks = {
        "minimum_independent_systems": systems >= gates.minimum_systems,
        "recomputed_system_level_power": projected_power >= gates.minimum_power,
        "signed_global_reference_balance": (
            global_bound <= gates.signed_global_bound_fraction * gates.practical_mde
        ),
        "candidate_pair_weighted_rms_balance": (
            pair_rms
            <= gates.candidate_pair_weighted_rms_bound_fraction * gates.practical_mde
        ),
        "action_energy_invariant": energy_max <= gates.action_energy_absolute_tolerance,
    }
    diagnostics = {
        "candidate_pairs": int(final.candidate_pair.nunique()) if len(final) else 0,
        "history_families": int(final.history_index.nunique()) if len(final) else 0,
        "query_families": int(final.query_index.nunique()) if len(final) else 0,
        "history_query_cells": (
            int(final.groupby(["history_index", "query_index"]).ngroups) if len(final) else 0
        ),
        "reversal_candidate_pairs": reversals,
        "reversal_strong_claim_coverage": (
            reversals >= gates.minimum_reversal_candidate_pairs_for_strong_claim
        ),
        "action_query_similarity_smd": action_similarity_smd,
        "candidate_pair_balance": pair_rows,
    }
    summary = {
        "hard_checks": hard_checks,
        "passes": bool(all(hard_checks.values())),
        "learner_inference_unlocked": bool(all(hard_checks.values())),
        "rows": int(len(final)),
        "systems": systems,
        "selected_system_ids": chosen,
        "projected_power": projected_power,
        "projected_standard_error": projected_se,
        "signed_global_reference_mean": global_mean,
        "signed_global_qmc_numerical_se": global_numerical_se,
        "signed_global_numerical_bound": global_bound,
        "global_aggregate_scramble_estimates": global_scrambles,
        "candidate_pair_weighted_rms_numerical_bound": pair_rms,
        "between_system_heterogeneity_is_diagnostic_not_gate": True,
        "action_energy_absolute_maximum": energy_max,
        "diagnostics": diagnostics,
    }
    return final.sort_values([
        "system_index", "realization", "history_index", "query_index", "candidate_pair"
    ]).reset_index(drop=True), summary


def classify_p3r_outcome(
    ci_lower: float, ci_upper: float, practical_mde: float, reliability: float,
    repeatability_threshold: float, *, assay_valid: bool,
    assay_failure: str | None = None, reversal_ci_lower: float | None = None,
    reversal_candidate_pairs: int = 0, minimum_reversal_candidate_pairs: int = 3,
    residual_bayes_balance_bound: float | None = None,
    similarity_adjusted_ci_lower: float | None = None,
) -> Mapping[str, str]:
    """Return five orthogonal V3 result fields."""

    repeatability = (
        "SYSTEM_LEVEL_REPEATABILITY_SUPPORTED"
        if np.isfinite(reliability) and reliability >= repeatability_threshold
        else "LOW_REPEATABILITY"
    )
    if not assay_valid:
        return {
            "assay_status": assay_failure or "ASSAY_NO_GO",
            "direction": "NOT_EVALUATED_CONFIRMATORILY",
            "materiality": "NOT_EVALUATED_CONFIRMATORILY",
            "claim_scope": "NOT_APPLICABLE",
            "repeatability": repeatability,
        }
    if ci_lower > 0:
        direction = "DIRECTIONAL_GO"
    elif ci_upper < 0:
        direction = "REVERSE_EFFECT"
    else:
        direction = "DIRECTIONAL_INCONCLUSIVE"
    if ci_lower >= practical_mde:
        materiality = "MATERIAL_EFFECT_SUPPORTED"
    elif ci_upper < practical_mde:
        materiality = "MATERIAL_EFFECT_RULED_OUT"
    else:
        materiality = "MATERIALITY_INCONCLUSIVE"
    if direction != "DIRECTIONAL_GO":
        claim_scope = "NOT_APPLICABLE"
    elif (
        residual_bayes_balance_bound is not None
        and ci_lower > residual_bayes_balance_bound
        and similarity_adjusted_ci_lower is not None
        and similarity_adjusted_ci_lower > 0
        and reversal_ci_lower is not None and reversal_ci_lower > 0
        and reversal_candidate_pairs >= minimum_reversal_candidate_pairs
    ):
        claim_scope = "BEYOND_FIXED_WAVEFORM_IDENTITY_SUPPORTED"
    else:
        claim_scope = "ACCESSIBILITY_ASSOCIATED_ONLY"
    return {
        "assay_status": "CONFIRMATORY_VALID",
        "direction": direction,
        "materiality": materiality,
        "claim_scope": claim_scope,
        "repeatability": repeatability,
    }


def freeze_v3_pair_manifest(
    pairs: pd.DataFrame, gate_summary: Mapping, output_csv: Path, receipt_path: Path,
    *, source_paths: Sequence[Path], config_path: Path,
) -> dict:
    """Freeze only a final-set-verified V3 manifest before learner inference."""

    _reject_learner_outcomes(pairs.columns)
    if gate_summary.get("passes") is not True or gate_summary.get("learner_inference_unlocked") is not True:
        raise RuntimeError("a failing P3-R V3 final-set gate cannot freeze a formal manifest")
    systems = int(pairs.system_index.nunique())
    observed = pairs.groupby("system_index").realization.agg(lambda value: set(map(int, value)))
    expected = {0, 1, 2, 3}
    if systems != int(gate_summary.get("systems", -1)) or any(value != expected for value in observed):
        raise RuntimeError("V3 manifest lacks complete frozen system-realization coverage")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_csv, index=False, float_format="%.17g")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "status": "PAIR_MANIFEST_FROZEN",
        "protocol_version": "3.0",
        "device": "cpu",
        "learner_outcomes_accessed": False,
        "pair_manifest_precedes_learner_inference": True,
        "learner_inference_unlocked": True,
        "rows": int(len(pairs)),
        "systems": systems,
        "realizations": 4,
        "complete_four_realization_coverage": True,
        "final_set_gate": dict(gate_summary),
        "config_sha256": digest(config_path),
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "pair_manifest_sha256": digest(output_csv),
        "sealed_accessed": False,
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
