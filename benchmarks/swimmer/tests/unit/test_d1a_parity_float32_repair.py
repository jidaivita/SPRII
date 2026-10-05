from __future__ import annotations

import pytest

from paper_c.extension import d1a_same_checkpoint_prospective as final


def test_float32_parity_tolerance_accepts_observed_ulp_but_not_large_drift() -> None:
    checkpoints = {str(seed): str(seed) for seed in final.CANONICAL_SEEDS}
    receipt = {"status": final.STATUS_PARITY, "architecture_key": "canonical",
               "features_sha256": "f", "maximum_absolute_prediction_difference": 9.5367431640625e-7,
               "tolerance": final.PARITY_TOLERANCE, "systems": list(range(8)), "rows": 288,
               "fixed_merge_order": [0, 1], "donor_sha256": "d", "checkpoint_hashes": checkpoints,
               "target_values_used": False, "scientific_contrast_computed": False}
    final._validate_parity_semantics(receipt, "canonical", "f", "d", checkpoints)
    broken = dict(receipt); broken["maximum_absolute_prediction_difference"] = 3e-6
    with pytest.raises(RuntimeError, match="parity"):
        final._validate_parity_semantics(broken, "canonical", "f", "d", checkpoints)
