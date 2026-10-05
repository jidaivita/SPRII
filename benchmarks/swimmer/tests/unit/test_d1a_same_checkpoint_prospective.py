from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_c.extension import d1a_same_checkpoint_prospective as final
from paper_c.extension import prospective_models


def _contexts() -> pd.DataFrame:
    return pd.DataFrame({
        "system_index": np.arange(512, dtype=np.int64),
        "realization": np.arange(512, dtype=np.int64) % 4,
        "history_index": np.arange(512, dtype=np.int64) % 6,
    })


def test_frozen_row_population_and_exact_donor_are_complete() -> None:
    rows = final._row_manifest(_contexts())
    donor = prospective_models.coherent_cell_derangement(rows, 89511)
    final._validate_donor(rows, donor)
    assert len(rows) == 18432
    assert rows.groupby("system_index").size().eq(36).all()


@pytest.mark.parametrize("defect", ["self", "duplicate", "wrong_cell", "out_of_range"])
def test_donor_negative_cases_fail_closed(defect: str) -> None:
    rows = final._row_manifest(_contexts())
    donor = prospective_models.coherent_cell_derangement(rows, 89511)
    broken = donor.copy()
    if defect == "self":
        broken[0] = 0
    elif defect == "duplicate":
        broken[0] = broken[1]
    elif defect == "wrong_cell":
        broken[0], broken[1] = broken[1], broken[0]
    else:
        broken[0] = len(rows)
    with pytest.raises(RuntimeError, match="donor"):
        final._validate_donor(rows, broken)


def test_shard_ownership_rejects_missing_duplicate_and_wrong_shard() -> None:
    even, odd = list(range(0, 512, 2)), list(range(1, 512, 2))
    final._validate_exact_shard_systems(even, 0)
    final._validate_exact_shard_systems(odd, 1)
    final._validate_exact_system_union([even, odd])
    with pytest.raises(RuntimeError, match="shard"):
        final._validate_exact_shard_systems(even[:-1], 0)
    with pytest.raises(RuntimeError, match="shard"):
        final._validate_exact_shard_systems(even + [0], 0)
    with pytest.raises(RuntimeError, match="shard"):
        final._validate_exact_shard_systems([1, *even[1:]], 0)
    with pytest.raises(RuntimeError, match="overlap"):
        final._validate_exact_system_union([even, odd + [0]])
    with pytest.raises(RuntimeError, match="omit"):
        final._validate_exact_system_union([even, odd[:-1]])


def test_bootstrap_requires_exact_physical_system_population() -> None:
    values = np.full(512, 2e-5, dtype=np.float64)
    result = final._bootstrap(values, 200, 89201)
    assert result["estimate"] == pytest.approx(2e-5)
    assert result["ci_low"] > 0
    with pytest.raises(ValueError, match="512"):
        final._bootstrap(values[:-1], 10, 1)


def test_alternate_config_path_and_output_are_rejected(tmp_path: Path) -> None:
    root = tmp_path
    canonical = root / final.CANONICAL_CONFIG
    canonical.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / final.CANONICAL_CONFIG
    cfg = json.loads(source.read_text())
    canonical.write_text(json.dumps(cfg))
    loaded = final._config(root, canonical)
    assert loaded["output_root"] == final.CANONICAL_OUTPUT
    alternate = root / "configs/alternate.json"; alternate.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="canonical config path"):
        final._config(root, alternate)
    cfg["output_root"] = "runs/alternate"
    canonical.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="config/output"):
        final._config(root, canonical)


def test_specificity_identity_is_float64_exact_at_reduction_tolerance() -> None:
    baseline = np.array([0.5, 0.25], dtype=np.float64)
    self_loss = np.array([0.4, 0.20], dtype=np.float64)
    donor_loss = np.array([0.45, 0.23], dtype=np.float64)
    gain_self, gain_donor = baseline - self_loss, baseline - donor_loss
    specificity = donor_loss - self_loss
    assert np.allclose(specificity, gain_self - gain_donor, rtol=0.0, atol=1e-15)


def test_branch_receipt_missing_hash_wrong_status_and_wrong_branch_fail(tmp_path: Path) -> None:
    result_path, receipt_path = tmp_path / "RESULTS.json", tmp_path / "RECEIPT.json"
    result = {"schema_version": "1.0", "status": "RIGHT", "evidence_identity": "RIGHT_EVIDENCE"}
    result_path.write_text(json.dumps(result))
    expected = {"result_sha256": final.sha256(result_path), "result_status": "RIGHT",
                "evidence_identity": "RIGHT_EVIDENCE", "receipt_status": "RIGHT_RECEIPT"}
    base_receipt = {"schema_version": "1.0", "status": "RIGHT_RECEIPT",
                    "implementation_sha256": "a" * 64, "protocol_sha256": "b" * 64}
    receipt_path.write_text(json.dumps(base_receipt))
    with pytest.raises(RuntimeError, match="hash"):
        final._verify_result_receipt(result_path, receipt_path, expected)
    receipt = {**base_receipt, "results_sha256": final.sha256(result_path), "status": "WRONG"}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="status"):
        final._verify_result_receipt(result_path, receipt_path, expected)
    receipt["status"] = "RIGHT_RECEIPT"; receipt_path.write_text(json.dumps(receipt))
    result["evidence_identity"] = "WRONG_BRANCH"; result_path.write_text(json.dumps(result))
    wrong_expected = {**expected, "result_sha256": final.sha256(result_path)}
    receipt["results_sha256"] = final.sha256(result_path); receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="identity"):
        final._verify_result_receipt(result_path, receipt_path, wrong_expected)


def test_stale_result_binds_and_swapped_architecture_fail() -> None:
    expected = {"donor_sha256": "d", "features_sha256": "f", "parity_sha256": "p",
                "checkpoint_hashes": {"86101": "x"}}
    final._require_exact_bindings(dict(expected), expected, "evaluation shard")
    for field in expected:
        stale = dict(expected); stale[field] = "stale"
        with pytest.raises(RuntimeError, match=field):
            final._require_exact_bindings(stale, expected, "evaluation shard")
    canonical = {"architecture_key": "canonical", "primary_for_paper": True,
                 "evidence_identity": "prospective_frozen_same_checkpoint_input_intervention",
                 "fixed_merge_order": [0, 1], "systems": 512,
                 "model_training": False, "checkpoint_selection": False}
    final._validate_result_semantics(canonical, "canonical", True)
    with pytest.raises(RuntimeError, match="semantic"):
        final._validate_result_semantics(canonical, "masked_gru_s64101", False)


def test_gru_panel_cannot_rescue_one_failed_base() -> None:
    assert final._conjunctive_secondary_supported(["SUPPORTED", "SUPPORTED"])
    assert not final._conjunctive_secondary_supported(["SUPPORTED", "NOT_SUPPORTED"])
    with pytest.raises(ValueError, match="both"):
        final._conjunctive_secondary_supported(["SUPPORTED"])


def test_runtime_identity_and_calendar_fail_closed() -> None:
    stop = datetime(2026, 8, 28, 20, 53, 16, tzinfo=timezone.utc)
    final._validate_runtime_authorization("12345", 12345,
                                          datetime(2026, 8, 27, tzinfo=timezone.utc), stop)
    with pytest.raises(RuntimeError, match="run identity"):
        final._validate_runtime_authorization(None, 12345,
                                              datetime(2026, 8, 27, tzinfo=timezone.utc), stop)
    with pytest.raises(RuntimeError, match="calendar stop"):
        final._validate_runtime_authorization("12345", 12345,
                                              datetime(2026, 8, 29, tzinfo=timezone.utc), stop)


def test_mixed_feature_model_identity_fails() -> None:
    identity = {"base_checkpoint_sha256": "x", "normalization_sha256": "n"}
    final._require_same_model_identity([identity, dict(identity)])
    with pytest.raises(RuntimeError, match="mixed"):
        final._require_same_model_identity([identity, {**identity, "base_checkpoint_sha256": "y"}])


def test_parity_stale_donor_checkpoint_coverage_and_tolerance_fail() -> None:
    checkpoints = {str(seed): str(seed) for seed in final.CANONICAL_SEEDS}
    receipt = {"status": final.STATUS_PARITY, "architecture_key": "canonical",
               "features_sha256": "f", "maximum_absolute_prediction_difference": 0.0,
               "tolerance": final.PARITY_TOLERANCE, "systems": list(range(8)), "rows": 8 * 36,
               "fixed_merge_order": [0, 1], "donor_sha256": "d",
               "checkpoint_hashes": checkpoints, "target_values_used": False,
               "scientific_contrast_computed": False}
    final._validate_parity_semantics(receipt, "canonical", "f", "d", checkpoints)
    for field, value in (("donor_sha256", "wrong"), ("checkpoint_hashes", {}),
                         ("systems", list(range(7))), ("rows", 7 * 36), ("tolerance", 1e-5)):
        broken = dict(receipt); broken[field] = value
        with pytest.raises(RuntimeError, match="parity"):
            final._validate_parity_semantics(broken, "canonical", "f", "d", checkpoints)


def test_architecture_result_rejects_stale_evaluation_receipt_lineage(tmp_path: Path) -> None:
    output, key = tmp_path, "canonical"
    paths = []
    for shard in range(2):
        path = output / f"evaluation/{key}/shard_{shard}_of_2/SHARD_RECEIPT.json"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"shard": shard}))
        paths.append(final.sha256(path))
    final._verify_evaluation_receipt_hash_list(output, key, paths)
    with pytest.raises(RuntimeError, match="evaluation shards"):
        final._verify_evaluation_receipt_hash_list(output, key, [paths[0], "stale"])
