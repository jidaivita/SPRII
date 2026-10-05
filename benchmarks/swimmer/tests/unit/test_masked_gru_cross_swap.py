import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_c.stage2 import masked_gru_cross_swap as cross_swap


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/masked_gru_cross_swap_supporting_v3.json"
PROTOCOL = ROOT / "protocol/PAPER_C_MASKED_GRU_CROSS_SWAP_SUPPORTING_V1.md"
EXECUTION_PROTOCOL = ROOT / "protocol/PAPER_C_MASKED_GRU_CROSS_SWAP_SUPPORTING_V3_NUMERICAL_COMPATIBILITY_ADDENDUM.md"


def _config():
    return json.loads(CONFIG.read_text())


def _statistics(systems=(0, 1)):
    rows = []
    for system in systems:
        for cell, gain in {"TT": 4.0, "TS": 2.0, "ST": 3.0, "SS": 2.0}.items():
            for seed in cross_swap.OPTIMIZATION_SEEDS:
                rows.append({
                    "system_index": system, "cell": cell, "seed": seed,
                    "count": cross_swap.ROWS_PER_SYSTEM,
                    "sum_loss": 0.0,
                    "sum_gain": gain * cross_swap.ROWS_PER_SYSTEM,
                    "sum2_loss": 0.0,
                    "sum2_gain": gain * gain * cross_swap.ROWS_PER_SYSTEM,
                })
    return pd.DataFrame(rows)[list(cross_swap.STAT_COLUMNS)]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _historical_report(rows):
    return {
        "status": "TT_SS_HISTORICAL_CSV_COMPATIBLE",
        "rows": int(rows),
        "absolute_tolerance": cross_swap.HISTORICAL_ABSOLUTE_TOLERANCE,
        "relative_tolerance": 0.0,
        "columns": {
            column: {"max_abs": 0.0}
            for column in cross_swap.HISTORICAL_STATISTIC_COLUMNS
        },
    }


def _parity_fixture(tmp_path, cfg, base_seed=64101):
    freeze_path = tmp_path / cfg["output_root"] / "IMPLEMENTATION_FROZEN.json"
    _write_json(freeze_path, {"status": cross_swap.STATUS_FREEZE})
    hashes = {str(seed): str(position + 1) * 64 for position, seed in enumerate(cross_swap.OPTIMIZATION_SEEDS)}
    receipt = {
        "schema_version": "1.0",
        "status": cross_swap.STATUS_PARITY,
        "base_seed": base_seed,
        "systems": [0, 1],
        "cells": list(cross_swap.CELLS),
        "optimization_seeds": list(cross_swap.OPTIMIZATION_SEEDS),
        "fixed_merge_order": [0, 1],
        "statistics_dtype": "float64",
        "content_sha256": "a" * 64,
        "shuffle_map_hashes": hashes,
        "tt_ss_same_runtime_upstream_exact": True,
        "tt_ss_historical_csv_compatible": _historical_report(
            2 * 2 * len(cross_swap.OPTIMIZATION_SEEDS)
        ),
        "implementation_freeze_sha256": cross_swap.sha256(freeze_path),
    }
    path = tmp_path / cfg["output_root"] / f"parity/articulated_s{base_seed}/PARITY.json"
    _write_json(path, receipt)
    return path, receipt


def _shard_fixture(tmp_path, cfg, base_seed=64101, shard=0):
    parity_path, parity = _parity_fixture(tmp_path, cfg, base_seed)
    systems = tuple(range(shard, cross_swap.SYSTEMS, cross_swap.SHARDS))
    frame = _statistics(systems)
    out = tmp_path / cfg["output_root"] / f"formal/articulated_s{base_seed}"
    stats_path = out / f"stats_shard_{shard:02d}_of_{cross_swap.SHARDS:02d}.csv.gz"
    cross_swap._atomic_csv_gz(stats_path, frame)
    receipt = {
        "schema_version": "1.0",
        "status": cross_swap.STATUS_SHARD,
        "base_seed": base_seed,
        "shard_index": shard,
        "shard_count": cross_swap.SHARDS,
        "systems": cross_swap.SYSTEMS // cross_swap.SHARDS,
        "statistics": cross_swap._relative(tmp_path, stats_path),
        "statistics_sha256": cross_swap.sha256(stats_path),
        "semantic_sha256": cross_swap.canonical_frame_sha(frame),
        "parity_sha256": cross_swap.sha256(parity_path),
        "shuffle_map_hashes": parity["shuffle_map_hashes"],
        "tt_ss_historical_csv_compatible": _historical_report(
            (cross_swap.SYSTEMS // cross_swap.SHARDS) * 2 * len(cross_swap.OPTIMIZATION_SEEDS)
        ),
    }
    receipt_path = out / f"receipt_shard_{shard:02d}_of_{cross_swap.SHARDS:02d}.json"
    _write_json(receipt_path, receipt)
    return parity_path, parity, stats_path, receipt_path, receipt, frame


def test_config_freezes_exact_four_cell_no_retrain_design():
    cfg = _config()
    cross_swap.validate_config(cfg)
    assert cfg["design"]["cells"] == ["TT", "TS", "ST", "SS"]
    assert cfg["upstream"]["base_seeds"] == [64101, 64103]
    assert cfg["upstream"]["optimization_seeds"] == [86101, 86103, 86107]
    assert cfg["output_root"] == "runs/supporting/masked_gru_cross_swap_v3"
    assert cfg["execution_revision"] == "v3_dual_numerical_compatibility_gate"
    assert [item["revision"] for item in cfg["superseded_executions"]] == ["v1", "v2"]
    assert all(item["scientific_artifacts_created"] is False for item in cfg["superseded_executions"])
    assert cfg["numerical_compatibility"]["historical_absolute_tolerance"] == 1e-15
    assert cfg["inference"]["bootstrap_seed"] == 86211
    assert "model_training" in cfg["forbidden"]
    assert not any(action.startswith("train") for action in (
        "freeze-implementation", "parity", "evaluate-shard", "merge-base", "summarize", "self-test",
    ))


@pytest.mark.skip(reason="Requires historical protocol/machine-command provenance fixtures, excluded from the source release")
def test_protocol_prevents_metric_selection_and_preserves_evidence_boundary():
    text = PROTOCOL.read_text()
    assert "single primary diagnostic" in text
    assert "E_input" in text and "E_interaction" in text
    assert "Prediction gain is the only outcome" in text
    assert "does not alter the completed masked-GRU" in text
    assert "classification" in text
    assert "No retraining" in text
    addendum = EXECUTION_PROTOCOL.read_text()
    assert "changes no system, row, cell, adapter, base learner, seed, estimand" in addendum
    assert "runs/supporting/masked_gru_cross_swap_v3" in addendum
    assert "same-runtime" in addendum and "1e-15" in addendum
    assert "post-hoc supporting evidence" in addendum


def test_wrong_runtime_root_fails_closed():
    cfg = _config()
    with pytest.raises(RuntimeError, match="isolated runtime root"):
        cross_swap._require_runtime_root(Path("runs/wrong_cross_swap_runtime"), cfg)


def test_factorial_estimands_match_registered_equations():
    systems = cross_swap.system_estimands(_statistics())
    assert np.allclose(systems.G_TT, 4.0)
    assert np.allclose(systems.G_TS, 2.0)
    assert np.allclose(systems.G_ST, 3.0)
    assert np.allclose(systems.G_SS, 2.0)
    assert np.allclose(systems.I_true_train, 2.0)
    assert np.allclose(systems.I_shuffle_train, 1.0)
    assert np.allclose(systems.W_self, 1.0)
    assert np.allclose(systems.W_shuffle, 0.0)
    assert np.allclose(systems.E_input, 1.5)
    assert np.allclose(systems.E_interaction, 1.0)


def test_row_aggregation_is_array_exact_with_completed_formal_evaluator():
    rows = pd.DataFrame([
        {"row_index": row, "system_index": 7, "cell": cell, "seed": 86101,
         "loss": loss, "gain": gain}
        for cell in cross_swap.CELLS
        for row, loss, gain in ((2, 0.7, 0.2), (0, 0.1, -0.3), (1, 0.4, 0.5))
    ]).sample(frac=1.0, random_state=17).reset_index(drop=True)
    observed = cross_swap._aggregate_rows(rows)
    reference_rows = rows.assign(arm=rows.cell).drop(columns="cell").sort_values(
        ["system_index", "row_index", "arm", "seed"], kind="stable",
    ).reset_index(drop=True)
    expected = cross_swap.routing._aggregate_formal_rows(reference_rows).rename(columns={"arm": "cell"})
    expected = expected[list(cross_swap.STAT_COLUMNS)].sort_values(
        ["system_index", "cell", "seed"], kind="stable",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


def test_incomplete_factorial_matrix_fails_closed():
    frame = _statistics()
    frame = frame[~((frame.system_index == 0) & (frame.cell == "TS") & (frame.seed == 86101))]
    with pytest.raises(RuntimeError, match="complete 4x3"):
        cross_swap.system_estimands(frame)


def test_fixed_order_merge_and_duplicate_gate():
    frame = _statistics(systems=(0, 1, 2, 3))
    parts = [frame[frame.system_index % 2 == shard].copy() for shard in (0, 1)]
    merged = cross_swap.merge_statistic_parts(parts)
    expected = frame.sort_values(["system_index", "cell", "seed"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, expected, check_exact=True)
    assert cross_swap.canonical_frame_sha(merged) == cross_swap.canonical_frame_sha(expected)
    with pytest.raises(RuntimeError, match="duplicates"):
        cross_swap.merge_statistic_parts([parts[0], parts[0]])


def test_exact_frame_validation_detects_one_ulp_or_semantic_change():
    expected = _statistics(systems=(0,))
    cross_swap._assert_frames_exact(expected, expected.copy(), "toy")
    changed = expected.copy()
    changed.loc[0, "sum_gain"] = np.nextafter(changed.loc[0, "sum_gain"], np.inf)
    with pytest.raises(RuntimeError, match="not array-exact"):
        cross_swap._assert_frames_exact(expected, changed, "toy")


def test_historical_compatibility_accepts_frozen_tolerance_and_records_maxima():
    expected = _statistics(systems=(0,))
    observed = expected.copy()
    observed.loc[0, "sum_loss"] += 1e-16
    report = cross_swap._assert_historical_compatible(
        observed, expected, "toy", cross_swap.HISTORICAL_ABSOLUTE_TOLERANCE,
    )
    assert report["status"] == "TT_SS_HISTORICAL_CSV_COMPATIBLE"
    assert 0.0 < report["columns"]["sum_loss"]["max_abs"] <= 1e-15


def test_historical_compatibility_rejects_above_tolerance_and_nonfinite_values():
    expected = _statistics(systems=(0,))
    changed = expected.copy()
    changed.loc[0, "sum_loss"] += 2e-15
    with pytest.raises(RuntimeError, match="exceeds frozen historical tolerance"):
        cross_swap._assert_historical_compatible(changed, expected, "toy", 1e-15)
    changed = expected.copy()
    changed.loc[0, "sum_loss"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        cross_swap._assert_historical_compatible(changed, expected, "toy", 1e-15)


def test_common_bootstrap_is_deterministic_and_reuses_one_index_matrix():
    frame = cross_swap.system_estimands(_statistics(systems=tuple(range(16))))
    first = cross_swap.common_bootstrap_summaries(frame, 200, 86211)
    second = cross_swap.common_bootstrap_summaries(frame, 200, 86211)
    assert first == second
    assert first["E_input"]["estimate"] == 1.5
    assert first["E_interaction"]["estimate"] == 1.0
    assert {row["seed"] for row in first.values()} == {86211}


def test_config_rejects_seed_or_metric_hierarchy_drift():
    cfg = _config()
    cfg["inference"]["bootstrap_seed"] = 86212
    with pytest.raises(RuntimeError, match="inference changed"):
        cross_swap.validate_config(cfg)
    cfg = _config()
    cfg["design"]["cells"] = ["TT", "SS"]
    with pytest.raises(RuntimeError, match="four-cell"):
        cross_swap.validate_config(cfg)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("execution", "remote_only"), False, "remote-only"),
        (("execution", "allowed_hosts"), ["worker_a"], "allowed host"),
        (("execution", "allowed_gpu_ids"), [0, 1], "physical GPUs"),
        (("execution", "threads_per_worker"), 2, "thread count"),
        (("execution", "statistics_dtype"), "float32", "dtype"),
        (("execution", "fresh_packet_mutation_forbidden"), False, "execution flag"),
        (("upstream", "final_result_status"), "CHANGED", "final-result status"),
        (("command_matrix",), "work/other.md", "command matrix"),
        (("design", "primary"), "changed", "primary estimand"),
        (("design", "secondary"), "changed", "secondary estimand"),
        (("inference", "scientific_unit"), "row", "scientific unit"),
        (("inference", "interval"), "changed", "interval"),
    ],
)
def test_config_rejects_execution_or_binding_drift(path, value, match):
    cfg = copy.deepcopy(_config())
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match=match):
        cross_swap.validate_config(cfg)


def test_upstream_final_result_binding_fails_after_mutation(tmp_path):
    path = tmp_path / "upstream/FINAL_RESULT.json"
    payload = {
        "status": "MASKED_GRU_COMPETENT_ARCHITECTURE_CONTRADICTION",
        "articulated_formal_results": {"64101": {}, "64103": {}},
    }
    _write_json(path, payload)
    cfg = {
        "upstream": {
            "final_result": str(path.relative_to(tmp_path)),
            "final_result_sha256": cross_swap.sha256(path),
            "final_result_status": payload["status"],
            "articulated_result_keys": ["64101", "64103"],
        }
    }
    binding = cross_swap._upstream_result_binding(tmp_path, cfg)
    assert binding["sha256"] == cfg["upstream"]["final_result_sha256"]
    payload["status"] = "CHANGED"
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match="final result hash changed"):
        cross_swap._upstream_result_binding(tmp_path, cfg)


@pytest.mark.skip(reason="Requires historical protocol/machine-command provenance fixtures, excluded from the source release")
def test_command_matrix_validation_rejects_gpu_remap_or_missing_environment(tmp_path):
    cfg = _config()
    target = tmp_path / cfg["command_matrix"]
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = (ROOT / cfg["command_matrix"]).read_text()
    target.write_text(canonical)
    assert cross_swap._validate_command_matrix(tmp_path, cfg)["cuda_visible_devices_unset"] is True
    target.write_text(canonical.replace("unset CUDA_VISIBLE_DEVICES", "export CUDA_VISIBLE_DEVICES=2,3"))
    with pytest.raises(RuntimeError, match="unset CUDA_VISIBLE_DEVICES"):
        cross_swap._validate_command_matrix(tmp_path, cfg)
    target.write_text(canonical.replace("export CUBLAS_WORKSPACE_CONFIG=:4096:8\n", ""))
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        cross_swap._validate_command_matrix(tmp_path, cfg)
    target.write_text(canonical.replace('export PYTHONPATH="$PAPER_C_ROOT/code"\n', ""))
    with pytest.raises(RuntimeError, match="remote root or Python path"):
        cross_swap._validate_command_matrix(tmp_path, cfg)
    target.write_text(canonical.replace(
        'python3 -m $M "$PAPER_C_ROOT" "$C" freeze-implementation --authorize-reviewed-code\n', "",
    ))
    with pytest.raises(RuntimeError, match="review/freeze command"):
        cross_swap._validate_command_matrix(tmp_path, cfg)


@pytest.mark.skip(reason="Requires historical protocol/machine-command provenance fixtures, excluded from the source release")
def test_freeze_validates_manifest_before_publishing_any_marker(tmp_path, monkeypatch):
    cfg = _config()
    config_path = tmp_path / "config.json"
    _write_json(config_path, cfg)
    command_path = tmp_path / cfg["command_matrix"]
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text((ROOT / cfg["command_matrix"]).read_text())
    protocol_path = tmp_path / cfg["protocol"]
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(EXECUTION_PROTOCOL.read_text())
    scientific_protocol_path = tmp_path / cfg["scientific_protocol"]
    scientific_protocol_path.parent.mkdir(parents=True, exist_ok=True)
    scientific_protocol_path.write_text(PROTOCOL.read_text())

    monkeypatch.setattr(cross_swap, "_load_config", lambda _: cfg)
    monkeypatch.setattr(cross_swap, "_require_runtime_root", lambda *_: None)
    monkeypatch.setattr(cross_swap, "_upstream_config", lambda *_: (tmp_path / "upstream.json", {}))
    monkeypatch.setattr(cross_swap.base, "_require_remote", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cross_swap, "_upstream_result", lambda *_: {})
    monkeypatch.setattr(cross_swap, "_source_paths", lambda *_: [config_path])
    monkeypatch.setattr(cross_swap, "_artifact_manifest", lambda *_: (_ for _ in ()).throw(RuntimeError("manifest failure")))

    with pytest.raises(RuntimeError, match="manifest failure"):
        cross_swap.freeze_implementation(tmp_path, config_path, True)

    out = tmp_path / cfg["output_root"]
    assert not (out / "IMPLEMENTATION_FROZEN.json").exists()
    assert not (out / "COMMAND_MATRIX_FROZEN.json").exists()


def test_artifact_manifest_reads_authorization_from_upstream_formal_pipeline(tmp_path, monkeypatch):
    cfg = {"upstream": {"config": "configs/upstream.json"}}
    upstream_cfg = {"output_root": "runs/upstream"}
    authorization_path = (
        tmp_path / upstream_cfg["output_root"]
        / "formal_pipeline/authorization/ONE_SHOT_FORMAL_AUTHORIZATION.json"
    )
    _write_json(authorization_path, {"status": "AUTHORIZED"})

    monkeypatch.setattr(
        cross_swap.upstream, "_verify_one_shot_authorization",
        lambda *_: {"status": "AUTHORIZED"},
    )
    monkeypatch.setattr(cross_swap, "_upstream_result_binding", lambda *_: {})
    monkeypatch.setattr(
        cross_swap.upstream, "_verify_base",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("past authorization binding")),
    )

    with pytest.raises(RuntimeError, match="past authorization binding"):
        cross_swap._artifact_manifest(tmp_path, cfg, upstream_cfg)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "2.0", "missing or invalid"),
        ("base_seed", 64103, "wrong base seed"),
        ("cells", ["TT", "SS"], "wrong cells"),
        ("optimization_seeds", [86101], "wrong optimization seeds"),
        ("fixed_merge_order", [1, 0], "wrong merge order"),
        ("statistics_dtype", "float32", "wrong statistics dtype"),
        ("content_sha256", "", "lacks a content hash"),
        ("tt_ss_same_runtime_upstream_exact", False, "same-runtime"),
        ("tt_ss_historical_csv_compatible", {}, "historical compatibility"),
        ("implementation_freeze_sha256", "0" * 64, "stale implementation freeze"),
    ],
)
def test_parity_receipt_identity_drift_fails_closed(tmp_path, field, value, match):
    cfg = _config()
    path, receipt = _parity_fixture(tmp_path, cfg)
    receipt[field] = value
    _write_json(path, receipt)
    with pytest.raises(RuntimeError, match=match):
        cross_swap._verify_parity(tmp_path, cfg, 64101)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("base_seed", 64103),
        ("shard_index", 1),
        ("shard_count", 3),
        ("systems", 255),
        ("statistics", "runs/supporting/masked_gru_cross_swap_v1/wrong.csv.gz"),
        ("parity_sha256", "0" * 64),
        ("shuffle_map_hashes", {}),
        ("tt_ss_historical_csv_compatible", {}),
    ],
)
def test_shard_receipt_identity_drift_fails_closed(tmp_path, field, value):
    cfg = _config()
    parity_path, parity, _, receipt_path, receipt, _ = _shard_fixture(tmp_path, cfg)
    receipt[field] = value
    _write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError):
        cross_swap._verify_shard_receipt(tmp_path, cfg, 64101, 0, parity_path, parity)


@pytest.mark.parametrize("mutation", ["ownership", "count"])
def test_shard_statistics_ownership_or_count_drift_fails_closed(tmp_path, mutation):
    cfg = _config()
    parity_path, parity, stats_path, receipt_path, receipt, frame = _shard_fixture(tmp_path, cfg)
    frame = frame.copy()
    if mutation == "ownership":
        frame.loc[frame.system_index == 0, "system_index"] = cross_swap.SYSTEMS + 1
    else:
        frame.loc[0, "count"] = cross_swap.ROWS_PER_SYSTEM - 1
    cross_swap._atomic_csv_gz(stats_path, frame)
    receipt["statistics_sha256"] = cross_swap.sha256(stats_path)
    receipt["semantic_sha256"] = cross_swap.canonical_frame_sha(frame)
    _write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError):
        cross_swap._verify_shard_receipt(tmp_path, cfg, 64101, 0, parity_path, parity)


def test_static_self_test_reads_no_formal_outcome():
    result = cross_swap.self_test()
    assert result["status"] == "MASKED_GRU_CROSS_SWAP_STATIC_PASS"
    assert result["model_training"] is False
    assert result["formal_outcome_read"] is False
