"""Outcome-blind feasibility analysis for the Articulated LQA assay."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, t


CONTEXT = ["system_index", "realization", "history_index", "query_index"]


def formal_stopping_decision(processed_systems: int, contributing_systems: int, config: dict) -> dict:
    """Apply the frozen outcome-blind formal-pool stopping rule."""

    formal = config["formal"]
    rule = formal["stopping_rule"]
    initial = int(rule["initial_systems"])
    block = int(rule["additional_block_systems"])
    maximum = int(rule["maximum_systems"])
    desired = int(rule["continue_when_contributing_below"])
    absolute = int(rule["limited_scope_minimum_contributing"])
    if processed_systems < initial or processed_systems > maximum:
        raise ValueError("processed systems must lie within the frozen formal stopping range")
    if processed_systems != initial and (processed_systems - initial) % block:
        raise ValueError("processed systems must end on a frozen formal block boundary")
    if not 0 <= contributing_systems <= processed_systems:
        raise ValueError("contributing systems must be a valid subset of processed systems")
    if contributing_systems >= desired:
        return {"action": "STOP", "scope": "NORMAL", "next_systems": processed_systems}
    if processed_systems < maximum:
        return {"action": "CONTINUE", "scope": "PENDING", "next_systems": min(processed_systems + block, maximum)}
    if contributing_systems >= absolute:
        return {"action": "STOP", "scope": "LIMITED", "next_systems": processed_systems}
    return {"action": "STOP", "scope": "LOW_COVERAGE_ASSAY_NO_GO", "next_systems": processed_systems}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_difference_bound(values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return paired mean, paired SE, and two-sided numerical upper bound."""

    difference = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if difference.ndim != 1 or len(difference) < 2:
        raise ValueError("paired reference values require at least two scrambles")
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    bound = abs(mean) + float(t.ppf(1.0 - alpha / 2.0, len(difference) - 1)) * se
    return mean, se, bound


def pair_table(rows: pd.DataFrame, sesoi: float, ref_a_fraction: float, ref_b_fraction: float) -> pd.DataFrame:
    scramble_a = sorted(column for column in rows if column.startswith("ref_a_vb_scramble"))
    scramble_b = sorted(column for column in rows if column.startswith("ref_b_vb_scramble"))
    if len(scramble_a) < 2 or len(scramble_a) != len(scramble_b):
        raise ValueError("independent ref-A/ref-B scramble columns are incomplete")
    records = []
    for key, cell in rows.groupby(CONTEXT, sort=True):
        cell = cell.sort_values("candidate_index").set_index("candidate_index")
        if list(cell.index) != list(range(6)):
            raise ValueError(f"incomplete candidate axis for context {key}")
        for first, second in combinations(range(6), 2):
            delta_a = float(cell.at[first, "accessibility"] - cell.at[second, "accessibility"])
            delta_lqa = float(cell.at[first, "lqa"] - cell.at[second, "lqa"])
            if delta_a == 0.0 or delta_lqa == 0.0:
                continue
            high_lqa, low_lqa = (first, second) if delta_lqa > 0 else (second, first)
            ref_a_mean, ref_a_se, ref_a_bound = paired_difference_bound(
                cell.loc[high_lqa, scramble_a].to_numpy(float),
                cell.loc[low_lqa, scramble_a].to_numpy(float),
            )
            ref_b_mean, ref_b_se, ref_b_bound = paired_difference_bound(
                cell.loc[high_lqa, scramble_b].to_numpy(float),
                cell.loc[low_lqa, scramble_b].to_numpy(float),
            )
            records.append({
                **dict(zip(CONTEXT, key)),
                "candidate_low_id": min(first, second),
                "candidate_high_id": max(first, second),
                "lqa_selected_candidate": high_lqa,
                "accessibility_selected_candidate": first if delta_a > 0 else second,
                "delta_accessibility_lqa_orientation": float(np.sign(delta_lqa) * delta_a),
                "delta_lqa": abs(delta_lqa),
                "opposing": bool(np.sign(delta_a) != np.sign(delta_lqa)),
                "ref_a_delta_vb": ref_a_mean,
                "ref_a_delta_vb_se_numerical": ref_a_se,
                "ref_a_bound": ref_a_bound,
                "ref_a_proposal": bool(ref_a_bound <= ref_a_fraction * sesoi),
                "ref_b_delta_vb": ref_b_mean,
                "ref_b_delta_vb_se_numerical": ref_b_se,
                "ref_b_bound": ref_b_bound,
                "ref_b_valid": bool(ref_b_bound <= ref_b_fraction * sesoi),
                "local_valid": bool(
                    np.isfinite(cell.loc[[first, second], ["local_value", "accessibility", "lqa"]].to_numpy()).all()
                    and (cell.loc[[first, second], "local_value"].to_numpy() > 0).all()
                ),
            })
    return pd.DataFrame(records)


def analyze(rows_path: Path, config_path: Path, output_root: Path) -> dict:
    rows_path, config_path, output_root = Path(rows_path), Path(config_path), Path(output_root)
    rows = pd.read_csv(rows_path)
    config = json.loads(config_path.read_text())
    sesoi = float(config["primary"]["sesoi_absolute"])
    pairs = pair_table(
        rows,
        sesoi,
        float(config["reference"]["ref_a_proposal_matching_fraction_of_sesoi"]),
        float(config["reference"]["pair_matching_fraction_of_sesoi"]),
    )
    eligible = pairs[pairs.opposing & pairs.ref_a_proposal & pairs.ref_b_valid & pairs.local_valid].copy()
    contributing_systems = int(eligible.system_index.nunique())
    systems_total = int(rows.system_index.nunique())
    yield_interval = binomtest(contributing_systems, systems_total).proportion_ci(confidence_level=0.95, method="wilson")
    formal_target = int(config["formal"]["target_systems"])
    projected_lower = int(np.floor(float(yield_interval.low) * formal_target))
    standardized_effect = float(config["primary"]["sesoi_standardized_effect"])
    z_alpha = float(norm.ppf(0.975))
    projected_power = float(
        1.0 - norm.cdf(z_alpha - standardized_effect * np.sqrt(max(projected_lower, 1)))
        + norm.cdf(-z_alpha - standardized_effect * np.sqrt(max(projected_lower, 1)))
    )
    pair_families = eligible[["candidate_low_id", "candidate_high_id"]].drop_duplicates()
    orientation = eligible.groupby(["candidate_low_id", "candidate_high_id"]).lqa_selected_candidate.nunique()
    receipt = {
        "status": "ARTICULATED_LQA_DEVELOPMENT_FEASIBILITY_COMPLETE_OUTCOME_BLIND",
        "assay_readiness": "READY_FOR_FORMAL_POOL" if (
            contributing_systems > 0
            and projected_lower >= int(config["formal"]["absolute_confirmatory_floor"])
            and not (len(eligible) and eligible.lqa_selected_candidate.nunique() == 1)
        ) else "COMPETING_PAIR_ASSAY_NO_GO",
        "systems_total": systems_total,
        "systems_with_confirmatory_pair": contributing_systems,
        "system_yield_fraction": contributing_systems / max(systems_total, 1),
        "system_yield_wilson_95": [float(yield_interval.low), float(yield_interval.high)],
        "formal_target_systems": formal_target,
        "projected_contributing_systems_at_target_wilson_lower": projected_lower,
        "projected_two_sided_power_at_sesoi_wilson_lower": projected_power,
        "contexts": int(rows[CONTEXT[:-1]].drop_duplicates().shape[0]),
        "context_query_cells": int(rows[CONTEXT].drop_duplicates().shape[0]),
        "candidate_pairs_total": int(len(pairs)),
        "opposing_pairs": int(pairs.opposing.sum()),
        "opposing_fraction": float(pairs.opposing.mean()),
        "ref_a_proposals": int((pairs.opposing & pairs.ref_a_proposal & pairs.local_valid).sum()),
        "ref_b_verified_pairs": int(len(eligible)),
        "candidate_pair_family_coverage": int(len(pair_families)),
        "candidate_pair_families_total": 15,
        "history_family_coverage": int(eligible.history_index.nunique()),
        "query_family_coverage": int(eligible.query_index.nunique()),
        "reversal_candidate_pair_families": int((orientation >= 2).sum()),
        "largest_lqa_candidate_share": float(eligible.lqa_selected_candidate.value_counts(normalize=True).max()) if len(eligible) else None,
        "complete_candidate_identity_aliasing": bool(len(eligible) and eligible.lqa_selected_candidate.nunique() == 1),
        "all_scores_finite": bool(np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all()),
        "sesoi": sesoi,
        "ref_a_bound": float(config["reference"]["ref_a_proposal_matching_fraction_of_sesoi"]) * sesoi,
        "ref_b_bound": float(config["reference"]["pair_matching_fraction_of_sesoi"]) * sesoi,
        "learner_outcomes_read": False,
        "raw_cka_used_for_pair_selection": False,
        "sealed_accessed": False,
        "source_hashes": {"rows": _sha256(rows_path), "config": _sha256(config_path)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_root / "all_candidate_pairs.csv.gz", index=False, compression="gzip")
    eligible.to_csv(output_root / "verified_competing_pairs.csv.gz", index=False, compression="gzip")
    (output_root / "feasibility_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.rows, args.config, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
