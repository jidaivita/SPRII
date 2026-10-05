"""Learner-blind matching and manifest freezing for the prospective test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


KEYS = ("system_index", "realization", "history_index", "query_index")
FORBIDDEN_OUTCOME_TOKENS = ("gain", "loss", "prediction", "jepa", "raw", "learner")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_learner_outcomes(columns: Iterable[str]) -> None:
    offending = [
        column for column in columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    ]
    if offending:
        raise ValueError(f"learner outcomes are forbidden during matching: {offending}")


def build_learner_blind_pairs(
    cells: pd.DataFrame,
    *,
    value_caliper: float,
    minimum_accessibility_gap: float,
    maximum_fidelity_error: float,
    utility_floor: float,
    local_value_floor: float,
) -> pd.DataFrame:
    """Create frozen high/low-accessibility pairs without learner outcomes.

    Pair construction is restricted to one physical system, nuisance
    realization, history, and query.  The reference value is the particle
    estimate; local geometry only determines accessibility orientation.
    """

    _reject_learner_outcomes(cells.columns)
    required = set(KEYS) | {
        "candidate_index", "reference_value", "local_value", "accessibility",
        "u_q", "fidelity", "action_energy", "action_query_similarity",
    }
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"missing matching columns: {sorted(missing)}")
    for name, value in {
        "value_caliper": value_caliper,
        "minimum_accessibility_gap": minimum_accessibility_gap,
        "maximum_fidelity_error": maximum_fidelity_error,
        "utility_floor": utility_floor,
        "local_value_floor": local_value_floor,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    eligible = cells[
        (cells.u_q > utility_floor)
        & (cells.local_value > local_value_floor)
        & (cells.fidelity <= maximum_fidelity_error)
    ].copy()
    rows: list[dict] = []
    for context, group in eligible.groupby(list(KEYS), sort=True):
        group = group.sort_values("candidate_index")
        values = list(group.to_dict("records"))
        for left_position, left in enumerate(values):
            for right in values[left_position + 1:]:
                value_gap = abs(float(left["reference_value"]) - float(right["reference_value"]))
                accessibility_gap = abs(float(left["accessibility"]) - float(right["accessibility"]))
                if value_gap > value_caliper or accessibility_gap < minimum_accessibility_gap:
                    continue
                if float(left["accessibility"]) >= float(right["accessibility"]):
                    high, low = left, right
                else:
                    high, low = right, left
                unordered = tuple(sorted((int(high["candidate_index"]), int(low["candidate_index"]))))
                row = dict(zip(KEYS, context))
                row.update({
                    "candidate_pair": f"{unordered[0]}-{unordered[1]}",
                    "candidate_high_a": int(high["candidate_index"]),
                    "candidate_low_a": int(low["candidate_index"]),
                    "reference_value_high_a": float(high["reference_value"]),
                    "reference_value_low_a": float(low["reference_value"]),
                    "reference_value_abs_gap": value_gap,
                    "accessibility_high": float(high["accessibility"]),
                    "accessibility_low": float(low["accessibility"]),
                    "accessibility_gap": accessibility_gap,
                    "local_value_high_a": float(high["local_value"]),
                    "local_value_low_a": float(low["local_value"]),
                    "u_q": float(high["u_q"]),
                    "fidelity_high_a": float(high["fidelity"]),
                    "fidelity_low_a": float(low["fidelity"]),
                    "action_energy_difference": float(high["action_energy"] - low["action_energy"]),
                    "action_query_similarity_difference": float(
                        high["action_query_similarity"] - low["action_query_similarity"]
                    ),
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(list(KEYS) + ["candidate_pair"]).reset_index(drop=True)
        if result.duplicated(list(KEYS) + ["candidate_pair"]).any():
            raise RuntimeError("matching produced duplicate context/pair rows")
    return result


def accessibility_reversal_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Identify unordered pairs for which high-accessibility identity reverses."""

    required = {"candidate_pair", "candidate_high_a", "system_index"}
    if not required.issubset(pairs.columns):
        raise ValueError("pair table lacks reversal columns")
    rows = []
    for candidate_pair, group in pairs.groupby("candidate_pair", sort=True):
        orientations = sorted(group.candidate_high_a.unique().tolist())
        rows.append({
            "candidate_pair": candidate_pair,
            "matched_rows": int(len(group)),
            "systems": int(group.system_index.nunique()),
            "high_a_identities": ",".join(str(value) for value in orientations),
            "has_accessibility_reversal": len(orientations) > 1,
        })
    return pd.DataFrame(rows)


def balance_summary(pairs: pd.DataFrame) -> Mapping[str, float]:
    if pairs.empty:
        return {"matched_rows": 0, "systems": 0, "candidate_pairs": 0}
    return {
        "matched_rows": int(len(pairs)),
        "systems": int(pairs.system_index.nunique()),
        "candidate_pairs": int(pairs.candidate_pair.nunique()),
        "mean_reference_value_abs_gap": float(pairs.reference_value_abs_gap.mean()),
        "maximum_reference_value_abs_gap": float(pairs.reference_value_abs_gap.max()),
        "mean_accessibility_gap": float(pairs.accessibility_gap.mean()),
        "mean_action_energy_difference": float(pairs.action_energy_difference.mean()),
        "mean_action_query_similarity_difference": float(pairs.action_query_similarity_difference.mean()),
    }


def freeze_pair_manifest(
    pairs: pd.DataFrame,
    output_csv: Path,
    receipt_path: Path,
    *,
    source_paths: Iterable[Path],
    thresholds: Mapping[str, float],
) -> dict:
    """Write the immutable learner-blind pair manifest and hash receipt."""

    _reject_learner_outcomes(pairs.columns)
    if pairs.empty:
        raise ValueError("cannot freeze an empty pair manifest")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_csv, index=False, float_format="%.17g")
    payload = {
        "status": "PAIR_MANIFEST_FROZEN",
        "learner_outcomes_accessed": False,
        "device": "cpu",
        "rows": int(len(pairs)),
        "systems": int(pairs.system_index.nunique()),
        "candidate_pairs": int(pairs.candidate_pair.nunique()),
        "thresholds": {key: float(value) for key, value in sorted(thresholds.items())},
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "pair_manifest_sha256": _sha256(output_csv),
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_frozen_pair_manifest(manifest: Path, receipt: Path) -> dict:
    payload = json.loads(receipt.read_text())
    if payload.get("status") != "PAIR_MANIFEST_FROZEN":
        raise ValueError("pair manifest receipt is not frozen")
    if payload.get("learner_outcomes_accessed") is not False:
        raise ValueError("pair manifest was not learner-blind")
    if _sha256(manifest) != payload.get("pair_manifest_sha256"):
        raise ValueError("pair manifest hash mismatch")
    return payload
