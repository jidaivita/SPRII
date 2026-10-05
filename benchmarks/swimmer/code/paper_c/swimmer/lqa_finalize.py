"""Outcome-blind formal-pair finalization for Articulated LQA.

This module consumes only atomic reference artifacts.  It never imports the
formal learner evaluator or reads predictions/losses.  All eligible pairs are
retained, while the physical system remains the independent analysis unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .lqa_feasibility import CONTEXT, formal_stopping_decision, pair_table
from .waveforms import banks


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


def _load_verified_reference(reference_root: Path, processed_systems: int) -> tuple[pd.DataFrame, dict, list[str]]:
    tables, receipt_hashes, source_hashes = [], [], None
    for system_index in range(processed_systems):
        system_root = reference_root / "systems" / f"system_{system_index:04d}"
        rows_path = system_root / "formal_reference_rows.csv.gz"
        receipt_path = system_root / "formal_reference_receipt.json"
        if not rows_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"formal reference system {system_index} is incomplete")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") != "ARTICULATED_LQA_FORMAL_SYSTEM_COMPLETE_OUTCOME_BLIND"
            or receipt.get("system_index") != system_index
            or receipt.get("rows") != 36
            or receipt.get("rows_sha256") != sha256(rows_path)
            or receipt.get("learner_outcomes_read") is not False
            or receipt.get("sealed_accessed") is not False
        ):
            raise RuntimeError(f"formal reference receipt {system_index} failed verification")
        current_hashes = receipt.get("source_hashes")
        if source_hashes is None:
            source_hashes = current_hashes
        elif current_hashes != source_hashes:
            raise RuntimeError("formal reference systems do not share one immutable source binding")
        table = pd.read_csv(rows_path)
        if len(table) != 36 or table.system_index.nunique() != 1 or int(table.system_index.iloc[0]) != system_index:
            raise RuntimeError(f"formal reference rows {system_index} have invalid system semantics")
        tables.append(table)
        receipt_hashes.append(sha256(receipt_path))
    rows = pd.concat(tables, ignore_index=True)
    if rows.duplicated(CONTEXT + ["candidate_index"]).any():
        raise RuntimeError("duplicate named-axis reference cells")
    return rows, source_hashes or {}, receipt_hashes


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=float).reshape(-1), np.asarray(second, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def _enrich_pairs(pairs: pd.DataFrame, rows: pd.DataFrame, root: Path, reference_config: dict) -> pd.DataFrame:
    base = json.loads((root / reference_config["base_config"]).read_text())
    s0 = json.loads((root / reference_config["s0_receipt"]).read_text())
    history, query = banks(float(s0["chosen_horizon_s"]), float(base["model"]["timestep_s"]))
    history_actions = list(history.values())
    query_actions = list(query.values())
    ref_b_scrambles = sorted(column for column in rows if column.startswith("ref_b_vb_scramble"))
    indexed = {
        tuple(key): cell.sort_values("candidate_index").set_index("candidate_index")
        for key, cell in rows.groupby(CONTEXT, sort=True)
    }
    enriched = []
    for row in pairs.itertuples(index=False):
        record = row._asdict()
        key = tuple(record[name] for name in CONTEXT)
        cell = indexed[key]
        selected = int(record["lqa_selected_candidate"])
        low, high = int(record["candidate_low_id"]), int(record["candidate_high_id"])
        other = high if selected == low else low
        if selected not in (low, high):
            raise RuntimeError("LQA orientation is not a member of the unordered candidate pair")
        record["other_candidate"] = other
        record["delta_raw_cka_lqa_orientation"] = float(cell.at[selected, "raw_cka"] - cell.at[other, "raw_cka"])
        selected_similarity = _cosine(history_actions[selected], query_actions[int(record["query_index"])])
        other_similarity = _cosine(history_actions[other], query_actions[int(record["query_index"])])
        record["delta_action_query_similarity_lqa_orientation"] = selected_similarity - other_similarity
        for column in ("geometry_posterior_ess", "ref_a_posterior_ess", "ref_b_posterior_ess"):
            if column in cell:
                selected_value = float(cell.at[selected, column])
                other_value = float(cell.at[other, column])
                if not np.isclose(selected_value, other_value, rtol=0.0, atol=0.0):
                    raise RuntimeError(f"context-level {column} differs across candidates")
                record[column] = selected_value
        for scramble_index, column in enumerate(ref_b_scrambles):
            record[f"ref_b_oriented_scramble{scramble_index}"] = float(cell.at[selected, column] - cell.at[other, column])
        enriched.append(record)
    return pd.DataFrame(enriched)


def _numerical_bound(scramble_estimates: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(scramble_estimates, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("QMC numerical bound requires at least two scramble estimates")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, se, abs(mean) + float(t.ppf(0.975, len(values) - 1)) * se


def residual_bayes_summary(manifest: pd.DataFrame, sesoi: float, budget_fraction: float) -> dict:
    scramble_columns = sorted(column for column in manifest if column.startswith("ref_b_oriented_scramble"))
    # Match the primary estimator exactly: average eligible pairs within each
    # physical system, then give every contributing system equal weight.
    global_scrambles = manifest.groupby("system_index", sort=True)[scramble_columns].mean().mean(axis=0).to_numpy(float)
    global_mean, global_se, global_bound = _numerical_bound(global_scrambles)
    family_rows = []
    for key, family in manifest.groupby(["candidate_low_id", "candidate_high_id"], sort=True):
        system_family = family.groupby("system_index", sort=True)[scramble_columns].mean()
        estimates = system_family.mean(axis=0).to_numpy(float)
        mean, se, bound = _numerical_bound(estimates)
        family_rows.append({
            "candidate_low_id": int(key[0]),
            "candidate_high_id": int(key[1]),
            "pairs": int(len(family)),
            "contributing_systems": int(len(system_family)),
            "mean": mean,
            "se_numerical": se,
            "bound": bound,
        })
    denominator = sum(row["contributing_systems"] for row in family_rows)
    pair_rms = float(np.sqrt(sum(row["contributing_systems"] * row["bound"] ** 2 for row in family_rows) / denominator))
    threshold = float(budget_fraction * sesoi)
    return {
        "global_mean": global_mean,
        "global_se_numerical": global_se,
        "global_bound": global_bound,
        "candidate_pair_weighted_rms_bound": pair_rms,
        "strong_attribution_threshold": threshold,
        "strong_attribution_balance_pass": bool(max(global_bound, pair_rms) <= threshold),
        "candidate_pair_families": family_rows,
    }


def finalize(
    root: Path,
    reference_config_path: Path,
    finalization_config_path: Path,
    reference_root: Path,
    output_root: Path,
    processed_systems: int,
) -> dict:
    root, reference_config_path, finalization_config_path = map(Path, (root, reference_config_path, finalization_config_path))
    reference_root, output_root = Path(reference_root), Path(output_root)
    reference_config = json.loads(reference_config_path.read_text())
    finalization_config = json.loads(finalization_config_path.read_text())
    if finalization_config["status"] != "FROZEN_BEFORE_FORMAL_LEARNER_OUTCOMES" or any(finalization_config["access"].values()):
        raise RuntimeError("formal finalization is not outcome blind")
    configured_reference = (root / finalization_config["reference_config"]).resolve()
    if configured_reference != reference_config_path.resolve():
        raise RuntimeError("finalization config is not bound to the supplied reference config")
    rows, reference_source_hashes, receipt_hashes = _load_verified_reference(reference_root, processed_systems)
    sesoi = float(reference_config["primary"]["sesoi_absolute"])
    pairs = pair_table(
        rows,
        sesoi,
        float(reference_config["reference"]["ref_a_proposal_matching_fraction_of_sesoi"]),
        float(reference_config["reference"]["pair_matching_fraction_of_sesoi"]),
    )
    eligible = pairs[pairs.opposing & pairs.ref_a_proposal & pairs.ref_b_valid & pairs.local_valid].copy()
    eligible = _enrich_pairs(eligible, rows, root, reference_config)
    eligible = eligible.sort_values(CONTEXT + ["candidate_low_id", "candidate_high_id"]).reset_index(drop=True)
    contributing = int(eligible.system_index.nunique()) if len(eligible) else 0
    stopping = formal_stopping_decision(processed_systems, contributing, reference_config)
    balance = residual_bayes_summary(
        eligible,
        sesoi,
        float(finalization_config["residual_bayes"]["strong_attribution_fraction_of_sesoi"]),
    ) if len(eligible) else None
    output_root.mkdir(parents=True, exist_ok=True)
    preview_path = output_root / "formal_pair_preview.csv.gz"
    _atomic_csv(preview_path, eligible)
    receipt = {
        "status": "ARTICULATED_LQA_FORMAL_REFERENCE_BLOCK_COMPLETE_OUTCOME_BLIND",
        "processed_systems": processed_systems,
        "contributing_systems": contributing,
        "eligible_pairs": int(len(eligible)),
        "stopping_decision": stopping,
        "residual_bayes": balance,
        "reference_system_receipt_hashes": receipt_hashes,
        "reference_source_hashes": reference_source_hashes,
        "source_hashes": {
            "reference_config": sha256(reference_config_path),
            "reference_protocol": sha256(root / finalization_config["reference_protocol"]),
            "finalization_config": sha256(finalization_config_path),
            "finalization_protocol": sha256(root / finalization_config["finalization_protocol"]),
            "implementation": sha256(Path(__file__)),
            "preview": sha256(preview_path),
        },
        "learner_outcomes_read": False,
        "sealed_accessed": False,
    }
    if stopping["action"] == "STOP" and stopping["scope"] != "LOW_COVERAGE_ASSAY_NO_GO":
        manifest_path = output_root / "formal_pair_manifest_frozen.csv.gz"
        _atomic_csv(manifest_path, eligible)
        receipt["status"] = "ARTICULATED_LQA_PAIR_MANIFEST_FROZEN_OUTCOME_BLIND"
        receipt["pair_manifest_sha256"] = sha256(manifest_path)
        receipt["source_hashes"]["pair_manifest"] = receipt["pair_manifest_sha256"]
    _atomic_text(output_root / "formal_finalization_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("reference_config", type=Path)
    parser.add_argument("finalization_config", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--processed-systems", type=int, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        args.reference_config,
        args.finalization_config,
        args.reference_root,
        args.output_root,
        args.processed_systems,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
