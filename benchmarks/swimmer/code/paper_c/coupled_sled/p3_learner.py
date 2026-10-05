"""CPU-only learner realization test for frozen prospective P3 pairs.

The formal entry point is deliberately impossible to call with a development
receipt.  Pair membership and orientation must have been frozen without learner
outcomes, and every file that defines the pool/model/normalization is hash
checked before inference starts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .accessibility_matching import verify_frozen_pair_manifest
from .formal_data import arrays_from_sample_manifest
from .learner import PersistentJEPA, _normalized, _predictions_and_z
from .manifests import load_spec
from .p2a_adapter import require_prospective_pair_manifest, verify_development_manifest
from .p3r_protocol import classify_p3r_outcome


FORMAL_STATUS = "P3_PROSPECTIVE_LEARNER_EVALUATION_COMPLETE"
DRY_STATUS = "P3_LEARNER_ENGINEERING_DRY_RUN_NOT_FORMAL"
REQUIRED_PAIR_COLUMNS = {
    "system_index", "realization", "history_index", "query_index",
    "candidate_high_a", "candidate_low_a", "candidate_pair",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(*values: object) -> int:
    key = "|".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def _pool_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("systems"), list) or not payload["systems"]:
        raise ValueError("system pool has no systems")
    return payload


def _id_columns(pairs: pd.DataFrame, base_spec_path: Path) -> pd.DataFrame:
    """Fill traceable waveform IDs from frozen banks when older manifests lack them."""

    from .waveforms import history_probe_bank, query_bank

    spec = load_spec(base_spec_path)
    cfg = spec["development_v0_1"]
    dt = spec["dynamics"]["reference_dt_s"]
    histories = sorted(history_probe_bank(cfg["experience_duration_s"], dt, cfg["history_energy"]))
    queries = sorted(query_bank(cfg["query_duration_s"], dt, cfg["query_energy"], tuple(cfg["query_chirp_hz"])))
    result = pairs.copy()
    mappings = {
        "history_id": ("history_index", histories),
        "query_id": ("query_index", queries),
        "candidate_high_a_id": ("candidate_high_a", histories),
        "candidate_low_a_id": ("candidate_low_a", histories),
    }
    for output, (index_column, values) in mappings.items():
        expected = result[index_column].astype(int).map(dict(enumerate(values)))
        if expected.isna().any():
            raise ValueError(f"{index_column} points outside the frozen waveform bank")
        if output in result and not np.array_equal(result[output].astype(str).to_numpy(), expected.astype(str).to_numpy()):
            raise ValueError(f"{output} disagrees with its frozen integer axis")
        result[output] = expected
    return result


def build_pair_evaluation_manifest(
    pair_manifest: Path,
    base_spec_path: Path,
    system_pool_path: Path,
    output_path: Path,
    *,
    seed_namespace: str = "coupled-p3-pair-crn-v1",
) -> pd.DataFrame:
    """Create anchor/high/low rows with exact pairwise common random numbers."""

    pairs = pd.read_csv(pair_manifest)
    missing = REQUIRED_PAIR_COLUMNS.difference(pairs.columns)
    if missing:
        raise ValueError(f"pair manifest lacks columns: {sorted(missing)}")
    pairs = _id_columns(pairs, base_spec_path)
    pool = _pool_payload(system_pool_path)
    system_ids = [row["system_id"] for row in pool["systems"]]
    sort_columns = ["system_index", "realization", "history_index", "query_index", "candidate_pair"]
    pairs = pairs.sort_values(sort_columns).reset_index(drop=True)
    rows: list[dict] = []
    for pair_index, row in enumerate(pairs.itertuples(index=False)):
        system_index = int(row.system_index)
        if not 0 <= system_index < len(system_ids):
            raise ValueError("pair system_index is outside the frozen system pool")
        system_id = system_ids[system_index]
        # No candidate identity enters the nuisance seed.  Thus all candidate
        # comparisons within the same (system, realization, H, Q) share exactly
        # the same gain/noise/query target, including across unordered pairs.
        group_seed = _stable_seed(
            seed_namespace, system_id, int(row.realization), row.history_id,
            row.query_id,
        )
        common = {
            "pair_index": pair_index,
            "system_index": system_index,
            "system_id": system_id,
            "anchor_index": int(row.history_index),
            "anchor_probe": row.history_id,
            "query_index": int(row.query_index),
            "query": row.query_id,
            "realization": int(row.realization),
            "group_seed": group_seed,
            "wrong_system_index": -1,
            "candidate_pair": row.candidate_pair,
        }
        for condition_index, (condition, added) in enumerate((
            ("anchor", "NONE"),
            ("candidate_high", row.candidate_high_a_id),
            ("candidate_low", row.candidate_low_a_id),
        )):
            rows.append({**common, "condition_index": condition_index, "condition": condition, "added_probe": added})
    table = pd.DataFrame(rows)
    if table.empty or len(table) != 3 * len(pairs):
        raise RuntimeError("pair evaluation manifest is incomplete")
    for _, group in table.groupby("pair_index", sort=False):
        if len(group) != 3 or group.group_seed.nunique() != 1 or group.system_id.nunique() != 1 or group["query"].nunique() != 1:
            raise RuntimeError("pairwise common-random-number invariant failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _load_frozen_model(
    base_spec_path: Path,
    formal_spec_path: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    arrays,
) -> tuple[PersistentJEPA, dict, dict]:
    base = load_spec(base_spec_path)
    formal = load_spec(formal_spec_path)
    cfg = copy.deepcopy(base["learner_development"])
    cfg.update(formal["learner"])
    norms_npz = np.load(normalization_path, allow_pickle=False)
    norms = {name: norms_npz[name] for name in norms_npz.files}
    model = PersistentJEPA(arrays.history.shape[-1], arrays.query_action.shape[-1], arrays.target.shape[-1], cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, norms, cfg


def pair_losses(table: pd.DataFrame, arrays, prediction: np.ndarray, norms: dict) -> pd.DataFrame:
    target = _normalized(arrays, norms)[3]
    sample_mse = np.mean((prediction - target) ** 2, axis=1)
    values = table.copy()
    values["standardized_query_mse"] = sample_mse
    pivot = values.pivot(index="pair_index", columns="condition", values="standardized_query_mse")
    if set(pivot.columns) != {"anchor", "candidate_high", "candidate_low"} or pivot.isna().any().any():
        raise RuntimeError("loss table is not one complete anchor/high/low triplet per pair")
    metadata = values.drop_duplicates("pair_index").set_index("pair_index")
    result = metadata[[
        "system_index", "system_id", "realization", "anchor_index", "anchor_probe",
        "query_index", "query", "candidate_pair",
    ]].copy()
    pair_source = values[values.condition == "candidate_high"].set_index("pair_index")
    low_source = values[values.condition == "candidate_low"].set_index("pair_index")
    result["candidate_high_a_id"] = pair_source.added_probe
    result["candidate_low_a_id"] = low_source.added_probe
    result["anchor_loss"] = pivot.anchor
    result["high_loss"] = pivot.candidate_high
    result["low_loss"] = pivot.candidate_low
    result["v_l_high"] = result.anchor_loss - result.high_loss
    result["v_l_low"] = result.anchor_loss - result.low_loss
    result["delta_l_pair"] = result.v_l_high - result.v_l_low
    return result.reset_index()


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def split_half_system_reliability(pairs: pd.DataFrame) -> float:
    realizations = sorted(pairs.realization.unique())
    if len(realizations) < 4 or len(realizations) % 2:
        return float("nan")
    midpoint = len(realizations) // 2
    first = pairs[pairs.realization.isin(realizations[:midpoint])].groupby("system_index").delta_l_pair.mean()
    second = pairs[pairs.realization.isin(realizations[midpoint:])].groupby("system_index").delta_l_pair.mean()
    common = first.index.intersection(second.index)
    return _pearson(first.loc[common].to_numpy(), second.loc[common].to_numpy())


def cluster_bootstrap_system_mean(system_values: np.ndarray, replicates: int, seed: int) -> dict:
    values = np.asarray(system_values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("system cluster bootstrap requires at least two finite system means")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_replicates": int(replicates),
    }


def unordered_pair_support(pairs: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Candidate-pair strata and reversal-only system-cluster support."""

    rows = []
    reversal_pairs: list[str] = []
    for name, group in pairs.groupby("candidate_pair"):
        high_identities = sorted(group.candidate_high_a_id.unique().tolist())
        is_reversal = len(high_identities) > 1
        if is_reversal:
            reversal_pairs.append(name)
        system_mean = group.groupby("system_index").delta_l_pair.mean()
        rows.append({
            "candidate_pair": name,
            "systems": int(system_mean.size),
            "rows": int(len(group)),
            "mean_delta_l_pair": float(system_mean.mean()),
            "accessibility_reversal": is_reversal,
        })
    table = pd.DataFrame(rows).sort_values("candidate_pair")
    reversal = pairs[pairs.candidate_pair.isin(reversal_pairs)]
    if reversal.system_index.nunique() >= 2:
        summary = cluster_bootstrap_system_mean(
            reversal.groupby("system_index").delta_l_pair.mean().to_numpy(), replicates, seed
        )
    else:
        summary = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_replicates": replicates}
    summary.update({"candidate_pairs": len(reversal_pairs), "rows": int(len(reversal))})
    return table, summary


def _verify_model_receipt(checkpoint: Path, normalization: Path, receipt: Path) -> dict:
    payload = json.loads(receipt.read_text())
    if payload.get("status") != "FORMAL_LEARNERS_FROZEN":
        raise RuntimeError("learner checkpoint is not covered by the frozen formal receipt")
    hashes = payload.get("hashes", {})
    if hashes.get("jepa") != _sha256(checkpoint) or hashes.get("normalization") != _sha256(normalization):
        raise RuntimeError("checkpoint or train-only normalization hash mismatch")
    return payload


def _verify_formal_inputs(
    pair_manifest: Path, pair_receipt: Path, system_pool: Path, pool_receipt: Path,
    checkpoint: Path, normalization: Path, model_receipt: Path, p2b_receipt: Path,
) -> tuple[dict, dict]:
    require_prospective_pair_manifest(pair_receipt)
    pair_payload = verify_frozen_pair_manifest(pair_manifest, pair_receipt)
    if pair_payload.get("pair_manifest_precedes_learner_inference") is not True:
        raise RuntimeError("pair receipt does not establish prospective ordering")
    is_v3 = pair_payload.get("protocol_version") == "3.0"
    pool_payload = json.loads(pool_receipt.read_text())
    expected_pool_status = (
        "P3R_V3_FRESH_832_SYSTEM_POOL_FROZEN"
        if is_v3 else "PROSPECTIVE_PHYSICAL_SYSTEM_POOL_FROZEN"
    )
    expected_pool_count = 832 if is_v3 else 704
    if (
        pool_payload.get("status") != expected_pool_status
        or pool_payload.get("count") != expected_pool_count
        or pool_payload.get("system_pool_sha256") != _sha256(system_pool)
    ):
        raise RuntimeError("formal prospective system pool hash/status mismatch")
    _verify_model_receipt(checkpoint, normalization, model_receipt)
    p2b = json.loads(p2b_receipt.read_text())
    rec = p2b.get("recommendation", {})
    if p2b.get("status") != "P2B_VARIANCE_ONLY_RESOURCE_CHOICE" or p2b.get("p3_unlocked") is not True:
        raise RuntimeError("P2b variance-only resource gate is not open")
    if is_v3:
        final_gate = pair_payload.get("final_set_gate", {})
        if final_gate.get("passes") is not True or final_gate.get("projected_power", 0.0) < 0.90:
            raise RuntimeError("P3-R V3 final-set identification/power gate is not open")
    elif rec.get("systems") != 704 or rec.get("realizations") != 4 or rec.get("passes") is not True:
        raise RuntimeError("legacy formal resources disagree with the frozen 704x4 choice")
    pairs = pd.read_csv(pair_manifest, usecols=["system_index", "realization"])
    systems = int(pairs.system_index.nunique())
    if len(pairs) != int(pair_payload.get("rows", -1)):
        raise RuntimeError("frozen matched assay row count disagrees with its receipt")
    if is_v3:
        if not (256 <= systems <= 704) or pair_payload.get("complete_four_realization_coverage") is not True:
            raise RuntimeError("P3-R V3 assay violates the frozen 256..704 complete-system rule")
    elif systems != 704 or pair_payload.get("complete_704x4_system_realization_coverage") is not True:
        raise RuntimeError("legacy matched assay does not retain all 704 planned system clusters")
    expected_realizations = {0, 1, 2, 3}
    observed = pairs.groupby("system_index").realization.agg(lambda values: set(map(int, values)))
    if any(values != expected_realizations for values in observed):
        raise RuntimeError("every formal system must retain matched pairs in all four nuisance realizations")
    return pair_payload, p2b


def run_learner_evaluation(
    base_spec: Path, formal_spec: Path, p3_spec: Path, pair_manifest: Path, pair_receipt: Path,
    system_pool: Path, checkpoint: Path, normalization: Path, model_receipt: Path,
    output_root: Path, *, formal: bool, pool_receipt: Path | None = None,
    p2b_receipt: Path | None = None,
) -> dict:
    p3 = load_spec(p3_spec)
    evaluation_cfg = p3["learner_evaluation"]
    if formal:
        if pool_receipt is None or p2b_receipt is None:
            raise ValueError("formal evaluation requires pool and P2b receipts")
        pair_payload, p2b = _verify_formal_inputs(
            pair_manifest, pair_receipt, system_pool, pool_receipt,
            checkpoint, normalization, model_receipt, p2b_receipt,
        )
        mde = float(p2b["practical_mde"])
        minimum_reliability = float(p2b["minimum_reliability"])
    else:
        pair_payload = verify_development_manifest(pair_manifest, pair_receipt)
        _verify_model_receipt(checkpoint, normalization, model_receipt)
        mde = float("nan")
        minimum_reliability = float(evaluation_cfg["minimum_reliability"])
    output_root.mkdir(parents=True, exist_ok=True)
    eval_manifest = output_root / "p3_pair_evaluation_manifest.csv"
    manifest = build_pair_evaluation_manifest(pair_manifest, base_spec, system_pool, eval_manifest)
    arrays = arrays_from_sample_manifest(base_spec, system_pool, eval_manifest)
    # Exact equality matters: all three rows in a pair must share Q target and anchor.
    for pair_index, group in manifest.groupby("pair_index", sort=False):
        idx = group.index.to_numpy()
        if not np.array_equal(arrays.target[idx], np.broadcast_to(arrays.target[idx[0]], arrays.target[idx].shape)):
            raise RuntimeError(f"query target CRN mismatch in pair {pair_index}")
        if not np.array_equal(arrays.history[idx, 0], np.broadcast_to(arrays.history[idx[0], 0], arrays.history[idx, 0].shape)):
            raise RuntimeError(f"anchor history CRN mismatch in pair {pair_index}")
    model, norms, cfg = _load_frozen_model(base_spec, formal_spec, checkpoint, normalization, arrays)
    prediction, _ = _predictions_and_z(model, arrays, norms, cfg, torch.device("cpu"), True)
    pairs = pair_losses(manifest, arrays, prediction, norms)
    # Restore frozen integer orientation for supporting candidate-pair analyses.
    frozen = pd.read_csv(pair_manifest).sort_values([
        "system_index", "realization", "history_index", "query_index", "candidate_pair"
    ]).reset_index(drop=True)
    pairs["candidate_high_a"] = frozen.candidate_high_a.astype(int)
    pairs["candidate_low_a"] = frozen.candidate_low_a.astype(int)
    pairs.to_csv(output_root / "p3_pair_level_contrasts.csv", index=False, float_format="%.17g")
    system = pairs.groupby(["system_index", "system_id"], as_index=False).agg(
        mean_delta_l_pair=("delta_l_pair", "mean"), pairs=("pair_index", "size")
    )
    system.to_csv(output_root / "p3_system_level_contrasts.csv", index=False, float_format="%.17g")
    reliability = split_half_system_reliability(pairs)
    replicates = int(evaluation_cfg["bootstrap_replicates"])
    if len(system) >= 2:
        primary = cluster_bootstrap_system_mean(
            system.mean_delta_l_pair.to_numpy(), replicates, int(evaluation_cfg["bootstrap_seed"])
        )
    else:
        primary = {"mean": float(system.mean_delta_l_pair.mean()), "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_replicates": replicates}
    pair_table, reversal = unordered_pair_support(
        pairs, replicates, int(evaluation_cfg["bootstrap_seed"]) + 1
    )
    pair_table.to_csv(output_root / "p3_unordered_pair_support.csv", index=False, float_format="%.17g")
    outcome = classify_p3r_outcome(
        primary["ci_low"], primary["ci_high"], mde, reliability,
        minimum_reliability, assay_valid=bool(formal),
        reversal_ci_lower=reversal.get("ci_low"),
        reversal_candidate_pairs=int(reversal.get("candidate_pairs", 0)),
        minimum_reversal_candidate_pairs=3,
    )
    formal_pass = bool(formal and outcome["direction"] == "DIRECTIONAL_GO")
    primary.update({
        "split_half_reliability": reliability,
        "minimum_reliability": minimum_reliability,
        "practical_mde": mde,
        "population_direction_pass": formal_pass,
        "outcome_taxonomy": outcome,
        "statistical_unit": "physical_system",
        "reversal_subset": reversal,
    })
    (output_root / "p3_primary_endpoint.json").write_text(json.dumps(primary, indent=2, sort_keys=True, allow_nan=True) + "\n")
    receipt = {
        "status": FORMAL_STATUS if formal else DRY_STATUS,
        "device": "cpu",
        "formal": formal,
        "systems": int(system.system_index.nunique()),
        "pair_rows": int(len(pairs)),
        "realizations": int(pairs.realization.nunique()),
        "pair_manifest_sha256": _sha256(pair_manifest),
        "pair_receipt_sha256": _sha256(pair_receipt),
        "system_pool_sha256": _sha256(system_pool),
        "checkpoint_sha256": _sha256(checkpoint),
        "normalization_sha256": _sha256(normalization),
        "evaluation_manifest_sha256": _sha256(eval_manifest),
        "pair_level_sha256": _sha256(output_root / "p3_pair_level_contrasts.csv"),
        "system_level_sha256": _sha256(output_root / "p3_system_level_contrasts.csv"),
        "primary_sha256": _sha256(output_root / "p3_primary_endpoint.json"),
        "common_random_numbers_within_pair": True,
        "candidate_id_excluded_from_nuisance_seed": True,
        "anchor_and_query_target_array_exact": True,
        "statistical_unit": "physical_system",
        "learner_training_run": False,
        "gpu_used": False,
        "discovery_accessed": False,
        "validation_accessed": False,
        "sealed_accessed": False,
        "source_pair_status": pair_payload["status"],
        "scientific_go": formal_pass,
        "outcome_taxonomy": outcome,
        "receipt_status_is_completion_not_scientific_verdict": True,
    }
    (output_root / "p3_learner_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {**receipt, "primary": primary}


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only frozen-pair P3 learner evaluator")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "run-formal"):
        cmd = sub.add_parser(name)
        cmd.add_argument("base_spec", type=Path); cmd.add_argument("formal_spec", type=Path)
        cmd.add_argument("p3_spec", type=Path); cmd.add_argument("pair_manifest", type=Path)
        cmd.add_argument("pair_receipt", type=Path); cmd.add_argument("system_pool", type=Path)
        cmd.add_argument("checkpoint", type=Path); cmd.add_argument("normalization", type=Path)
        cmd.add_argument("model_receipt", type=Path); cmd.add_argument("output", type=Path)
        cmd.add_argument("--device", choices=("cpu",), default="cpu")
        if name == "run-formal":
            cmd.add_argument("--pool-receipt", type=Path, required=True)
            cmd.add_argument("--p2b-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run_learner_evaluation(
        args.base_spec, args.formal_spec, args.p3_spec, args.pair_manifest, args.pair_receipt,
        args.system_pool, args.checkpoint, args.normalization, args.model_receipt, args.output,
        formal=args.command == "run-formal", pool_receipt=getattr(args, "pool_receipt", None),
        p2b_receipt=getattr(args, "p2b_receipt", None),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
