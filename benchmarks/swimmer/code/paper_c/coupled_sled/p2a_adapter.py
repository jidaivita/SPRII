"""CPU-only learner-blind P2a adapter for canonical Coupled P0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .accessibility_matching import (
    _reject_learner_outcomes,
    accessibility_reversal_summary,
    balance_summary,
    build_learner_blind_pairs,
)
from .manifests import load_spec
from .waveforms import history_probe_bank, query_bank


DEVELOPMENT_STATUS = "DEVELOPMENT_FEASIBILITY_ONLY"
CANONICAL_AXES = ("system", "realization", "history", "candidate", "query")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_action_similarity(left: np.ndarray, right: np.ndarray, points: int = 128) -> float:
    grid = np.linspace(0.0, 1.0, points)

    def resample(value: np.ndarray) -> np.ndarray:
        source = (np.arange(len(value), dtype=float) + 0.5) / len(value)
        channels = [np.interp(grid, source, value[:, channel]) for channel in range(value.shape[1])]
        return np.stack(channels, axis=1).reshape(-1)

    a = resample(np.asarray(left, dtype=float))
    b = resample(np.asarray(right, dtype=float))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def physical_candidate_features(spec_path: Path, candidate_ids: list[str], query_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    spec = load_spec(spec_path)
    cfg = spec["development_v0_1"]
    dt = spec["dynamics"]["reference_dt_s"]
    candidates = history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"])
    queries = query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"]))
    missing_candidates = set(candidate_ids).difference(candidates)
    missing_queries = set(query_ids).difference(queries)
    if missing_candidates or missing_queries:
        raise ValueError(f"axis IDs are absent from frozen banks: candidates={sorted(missing_candidates)}, queries={sorted(missing_queries)}")
    energy = np.asarray([candidates[name].energy for name in candidate_ids], dtype=float)
    similarity = np.asarray([
        [_normalized_action_similarity(candidates[candidate].actions, queries[query].actions) for query in query_ids]
        for candidate in candidate_ids
    ], dtype=float)
    return energy, similarity


def canonical_npz_to_matching_cells(npz_path: Path, axis_spec_path: Path, spec_path: Path) -> pd.DataFrame:
    """Convert one named-axis P0 artifact to learner-blind matching rows."""

    arrays = np.load(npz_path, allow_pickle=False)
    _reject_learner_outcomes(arrays.files)
    required = {
        "reference_value", "local_value", "accessibility", "u_q",
        "fidelity_normalized_max_value_error",
    }
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"canonical P0 artifact lacks fields: {sorted(missing)}")
    axis = json.loads(axis_spec_path.read_text())
    if tuple(axis.get("canonical_axes", ())) != CANONICAL_AXES:
        raise ValueError("canonical axis declaration is missing or reordered")
    shape = tuple(int(value) for value in axis["shape"])
    if len(shape) != 5:
        raise ValueError("canonical P0 shape must have five unequal-safe axes")
    # NpzFile lazily decompresses a member on every ``__getitem__`` call.
    # Cache each canonical field exactly once before the cell loop; otherwise
    # the five-dimensional expansion repeatedly decompresses the same arrays.
    # This is a storage/read optimization only: values, axes, and row ordering
    # remain byte-for-byte identical to the original definition.
    cached = {field: np.asarray(arrays[field]) for field in required}
    arrays.close()
    for field in required:
        if cached[field].shape != shape:
            raise ValueError(f"{field} does not match declared (S,R,H,E,Q) shape")
    systems, realizations, histories, candidates, queries = shape
    candidate_ids = list(axis["candidate_ids"])
    query_ids = list(axis["query_ids"])
    history_ids = list(axis["history_ids"])
    system_source_indices = list(axis["system_source_indices"])
    if len(candidate_ids) != candidates or len(query_ids) != queries or len(history_ids) != histories or len(system_source_indices) != systems:
        raise ValueError("axis IDs do not match tensor dimensions")
    action_energy, action_query_similarity = physical_candidate_features(spec_path, candidate_ids, query_ids)
    rows = []
    for s in range(systems):
        for r in range(realizations):
            for h in range(histories):
                for e in range(candidates):
                    for q in range(queries):
                        rows.append({
                            "system_index": int(system_source_indices[s]),
                            "local_system_index": s,
                            "realization": r,
                            "history_index": h,
                            "history_id": history_ids[h],
                            "candidate_index": e,
                            "candidate_id": candidate_ids[e],
                            "query_index": q,
                            "query_id": query_ids[q],
                            "reference_value": float(cached["reference_value"][s, r, h, e, q]),
                            "local_value": float(cached["local_value"][s, r, h, e, q]),
                            "accessibility": float(cached["accessibility"][s, r, h, e, q]),
                            "u_q": float(cached["u_q"][s, r, h, e, q]),
                            "fidelity": float(cached["fidelity_normalized_max_value_error"][s, r, h, e, q]),
                            "action_energy": float(action_energy[e]),
                            "action_query_similarity": float(action_query_similarity[e, q]),
                        })
    result = pd.DataFrame(rows)
    expected = systems * realizations * histories * candidates * queries
    if len(result) != expected or result.duplicated([
        "local_system_index", "realization", "history_index", "candidate_index", "query_index"
    ]).any():
        raise RuntimeError("canonical flattening lost or duplicated semantic cells")
    return result


def freeze_development_manifest(
    pairs: pd.DataFrame,
    manifest_path: Path,
    receipt_path: Path,
    *,
    sources: tuple[Path, ...],
    thresholds: Mapping[str, float],
    prospective_systems_generated: bool = False,
) -> dict:
    """Hash a development candidate manifest without unlocking inference."""

    _reject_learner_outcomes(pairs.columns)
    if pairs.empty:
        raise ValueError("development feasibility produced no candidate pairs")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(manifest_path, index=False, float_format="%.17g")
    payload = {
        "status": DEVELOPMENT_STATUS,
        "device": "cpu",
        "learner_outcomes_accessed": False,
        "discovery_accessed": False,
        "validation_accessed": False,
        "sealed_accessed": False,
        "prospective_systems_generated": bool(prospective_systems_generated),
        "prospective_pair_manifest_frozen": False,
        "learner_inference_unlocked": False,
        "rows": int(len(pairs)),
        "systems": int(pairs.system_index.nunique()),
        "candidate_pairs": int(pairs.candidate_pair.nunique()),
        "thresholds": {key: float(value) for key, value in sorted(thresholds.items())},
        "source_sha256": {str(path): _sha256(path) for path in sources},
        "development_manifest_sha256": _sha256(manifest_path),
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_development_manifest(manifest_path: Path, receipt_path: Path) -> dict:
    payload = json.loads(receipt_path.read_text())
    if payload.get("status") != DEVELOPMENT_STATUS:
        raise ValueError("receipt is not development-only feasibility evidence")
    if payload.get("learner_inference_unlocked") is not False or payload.get("prospective_pair_manifest_frozen") is not False:
        raise ValueError("development manifest illegally unlocks prospective inference")
    if _sha256(manifest_path) != payload.get("development_manifest_sha256"):
        raise ValueError("development manifest hash mismatch")
    return payload


def require_prospective_pair_manifest(receipt_path: Path) -> None:
    """Guard for future inference code: development receipts can never unlock it."""

    payload = json.loads(receipt_path.read_text())
    if payload.get("status") != "PAIR_MANIFEST_FROZEN" or payload.get("learner_inference_unlocked") is not True:
        raise RuntimeError("learner inference requires a separately frozen prospective pair manifest")


def run_development_p2a(
    npz_path: Path,
    axis_spec_path: Path,
    spec_path: Path,
    output_root: Path,
    *,
    value_caliper: float,
    minimum_accessibility_gap: float,
    maximum_fidelity_error: float,
    utility_floor: float,
    local_value_floor: float,
    prospective_systems_generated: bool = False,
) -> dict:
    cells = canonical_npz_to_matching_cells(npz_path, axis_spec_path, spec_path)
    thresholds = {
        "value_caliper": value_caliper,
        "minimum_accessibility_gap": minimum_accessibility_gap,
        "maximum_fidelity_error": maximum_fidelity_error,
        "utility_floor": utility_floor,
        "local_value_floor": local_value_floor,
    }
    pairs = build_learner_blind_pairs(cells, **thresholds)
    if pairs.empty:
        raise RuntimeError("P2a development feasibility found no eligible matched pairs")
    # Restore traceable physical IDs without changing matching semantics.
    candidate_map = cells[["candidate_index", "candidate_id"]].drop_duplicates().set_index("candidate_index").candidate_id
    history_map = cells[["history_index", "history_id"]].drop_duplicates().set_index("history_index").history_id
    query_map = cells[["query_index", "query_id"]].drop_duplicates().set_index("query_index").query_id
    pairs.insert(pairs.columns.get_loc("candidate_high_a") + 1, "candidate_high_a_id", pairs.candidate_high_a.map(candidate_map))
    pairs.insert(pairs.columns.get_loc("candidate_low_a") + 1, "candidate_low_a_id", pairs.candidate_low_a.map(candidate_map))
    pairs.insert(pairs.columns.get_loc("history_index") + 1, "history_id", pairs.history_index.map(history_map))
    pairs.insert(pairs.columns.get_loc("query_index") + 1, "query_id", pairs.query_index.map(query_map))
    output_root.mkdir(parents=True, exist_ok=True)
    cells_path = output_root / "p2a_matching_cells.csv"
    pairs_path = output_root / "p2a_candidate_pairs_development.csv"
    reversal_path = output_root / "p2a_accessibility_reversals.csv"
    receipt_path = output_root / "p2a_development_receipt.json"
    cells.to_csv(cells_path, index=False, float_format="%.17g")
    accessibility_reversal_summary(pairs).to_csv(reversal_path, index=False)
    receipt = freeze_development_manifest(
        pairs, pairs_path, receipt_path,
        sources=(npz_path, axis_spec_path, spec_path, cells_path, reversal_path),
        thresholds=thresholds,
        prospective_systems_generated=prospective_systems_generated,
    )
    summary = {
        **receipt,
        "balance": balance_summary(pairs),
        "reversal_candidate_pairs": int(accessibility_reversal_summary(pairs).has_accessibility_reversal.sum()),
        "eligible_cell_rows": int(len(cells[
            (cells.u_q > utility_floor) & (cells.local_value > local_value_floor) & (cells.fidelity <= maximum_fidelity_error)
        ])),
        "total_cell_rows": int(len(cells)),
    }
    (output_root / "p2a_development_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    verify_development_manifest(pairs_path, receipt_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learner-blind P2a development feasibility pairs")
    parser.add_argument("p0_npz", type=Path)
    parser.add_argument("axis_spec", type=Path)
    parser.add_argument("base_spec", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--value-caliper", type=float, required=True)
    parser.add_argument("--minimum-accessibility-gap", type=float, required=True)
    parser.add_argument("--maximum-fidelity-error", type=float, required=True)
    parser.add_argument("--utility-floor", type=float, required=True)
    parser.add_argument("--local-value-floor", type=float, required=True)
    parser.add_argument("--prospective-systems-generated", action="store_true")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    summary = run_development_p2a(
        args.p0_npz, args.axis_spec, args.base_spec, args.output_root,
        value_caliper=args.value_caliper,
        minimum_accessibility_gap=args.minimum_accessibility_gap,
        maximum_fidelity_error=args.maximum_fidelity_error,
        utility_floor=args.utility_floor,
        local_value_floor=args.local_value_floor,
        prospective_systems_generated=args.prospective_systems_generated,
    )
    print(json.dumps({"status": summary["status"], "rows": summary["rows"], "systems": summary["systems"]}, sort_keys=True))


if __name__ == "__main__":
    main()
