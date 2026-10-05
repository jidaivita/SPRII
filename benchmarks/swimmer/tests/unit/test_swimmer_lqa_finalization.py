import numpy as np
import pandas as pd

from paper_c.swimmer.lqa_evaluate import _cluster_interval, _similarity_adjusted_interval
from paper_c.swimmer.lqa_coordinator import _complete_prefix
from paper_c.swimmer.lqa_finalize import residual_bayes_summary


def test_residual_bayes_summary_uses_qmc_numerical_uncertainty_only():
    rows = []
    # Large between-system heterogeneity with exactly balanced aggregate QMC
    # means must not be mistaken for numerical reference uncertainty.
    for system, sign in enumerate((1.0, -1.0, 1.0, -1.0)):
        rows.append({
            "system_index": system,
            "candidate_low_id": 0,
            "candidate_high_id": 1,
            **{f"ref_b_oriented_scramble{i}": sign for i in range(4)},
        })
    result = residual_bayes_summary(pd.DataFrame(rows), sesoi=1.0, budget_fraction=0.10)
    assert result["global_bound"] == 0.0
    assert result["candidate_pair_weighted_rms_bound"] == 0.0
    assert result["strong_attribution_balance_pass"] is True


def test_residual_global_balance_matches_system_equal_primary_weighting():
    rows = []
    # System 0 has three pairs at +1 and system 1 has one pair at -1.  A raw
    # pair mean would be +0.5, while the frozen system-equal estimand is zero.
    for system, repeats, value in ((0, 3, 1.0), (1, 1, -1.0)):
        for query in range(repeats):
            rows.append({
                "system_index": system,
                "query_index": query,
                "candidate_low_id": 0,
                "candidate_high_id": 1,
                **{f"ref_b_oriented_scramble{i}": value for i in range(4)},
            })
    result = residual_bayes_summary(pd.DataFrame(rows), sesoi=1.0, budget_fraction=0.10)
    assert result["global_mean"] == 0.0
    assert result["global_bound"] == 0.0


def test_cluster_interval_weights_physical_systems_equally():
    result = _cluster_interval(np.asarray([1.0, 3.0]), seed=7, replicates=200)
    assert result["mean"] == 2.0
    assert result["systems"] == 2
    assert result["system_positive_fraction"] == 1.0


def test_similarity_adjustment_preserves_positive_intercept_with_pair_effects():
    rows = []
    for system in range(12):
        for low, high, family_offset in ((0, 1, -0.25), (2, 3, 0.25)):
            similarity = (system - 5.5) / 10.0
            rows.append({
                "system_index": system,
                "candidate_low_id": low,
                "candidate_high_id": high,
                "delta_action_query_similarity_lqa_orientation": similarity,
                "delta_gain_lqa_orientation": 0.5 + 0.4 * similarity + family_offset,
            })
    result = _similarity_adjusted_interval(pd.DataFrame(rows), seed=17, replicates=300)
    assert np.isclose(result["mean"], 0.5)
    assert result["identified"] is True
    assert result["ci_low"] > 0


def test_similarity_adjustment_removes_similarity_only_advantage():
    rows = []
    for system in range(20):
        similarity = 0.2 + system / 100.0
        rows.append({
            "system_index": system,
            "candidate_low_id": 0,
            "candidate_high_id": 1,
            "delta_action_query_similarity_lqa_orientation": similarity,
            "delta_gain_lqa_orientation": 2.0 * similarity,
        })
    result = _similarity_adjusted_interval(pd.DataFrame(rows), seed=19, replicates=300)
    assert result["identified"] is True
    assert abs(result["mean"]) < 1e-10


def test_coordinator_requires_both_atomic_system_artifacts(tmp_path):
    for system in (0, 1):
        root = tmp_path / "systems" / f"system_{system:04d}"
        root.mkdir(parents=True)
        (root / "formal_reference_rows.csv.gz").write_bytes(b"rows")
        if system == 0:
            (root / "formal_reference_receipt.json").write_text("{}")
    complete, missing = _complete_prefix(tmp_path, 2)
    assert complete == 1
    assert missing == [1]
