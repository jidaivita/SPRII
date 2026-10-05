import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from persistent_jepa.certificates import r2_score_system, system_gamma_estimates
from persistent_jepa.sampling import (
    VALID_ANCHORS,
    build_collision_free_pseudo_systems,
    deterministic_derangement,
    sample_pseudo_system_pairs,
    sample_same_rollout_nonoverlap_pairs,
    sample_same_system_pairs,
)
from persistent_jepa.simulator import DCleanConfig, exact_step, generate_dataset, generate_rollout
from persistent_jepa.test_seal import TestSealedError, require_test_unsealed, write_immutable_selection


class PreflightTests(unittest.TestCase):
    def small_config(self):
        return DCleanConfig(train_systems=12, val_systems=5, test_systems=4, rollouts_per_system=3)

    def test_exact_zero_force_decay(self):
        state = np.array([0.1, -0.2, 1.5, -0.5])
        out = exact_step(state, np.zeros(2), gamma=2.0, dt=0.05)
        alpha = np.exp(-0.1)
        np.testing.assert_allclose(out[2:], alpha * state[2:], rtol=1e-12, atol=1e-12)

    def test_rollout_shapes_and_determinism(self):
        cfg = self.small_config()
        s1, a1 = generate_rollout(1.2, 99, cfg)
        s2, a2 = generate_rollout(1.2, 99, cfg)
        self.assertEqual(s1.shape, (64, 4))
        self.assertEqual(a1.shape, (63, 2))
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(a1, a2)

    def test_system_splits_and_rollout_independence(self):
        ds = generate_dataset(self.small_config())
        train_ids, val_ids, test_ids = map(set, (ds.system_ids[x].tolist() for x in ("train", "val", "test")))
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertEqual(np.unique(ds.rollout_seeds["train"]).size, ds.rollout_seeds["train"].size)
        self.assertFalse(np.array_equal(ds.states["train"][0, 0], ds.states["train"][0, 1]))

    def test_analytic_certificate(self):
        cfg = DCleanConfig(train_systems=40, val_systems=2, test_systems=2, rollouts_per_system=4)
        ds = generate_dataset(cfg)
        pred, diag = system_gamma_estimates(ds.states["train"], ds.actions["train"], cfg.dt)
        self.assertGreater(diag["valid_system_fraction"], 0.95)
        self.assertGreater(diag["valid_window_fraction"], 0.50)
        self.assertGreater(r2_score_system(ds.gamma["train"], pred), 0.999)

    def test_pairing_invariants(self):
        pairs = sample_same_system_pairs(100, 8, 48, 7)
        self.assertTrue(np.all(pairs[:, 1] != pairs[:, 3]))
        self.assertTrue(np.all(np.isin(pairs[:, 2], VALID_ANCHORS)))
        perm = deterministic_derangement(48, 7)
        self.assertTrue(np.all(perm != np.arange(48)))

    def test_b1_nonoverlap_pairing(self):
        pairs = sample_same_rollout_nonoverlap_pairs(100, 8, 48, 7)
        self.assertTrue(np.all(pairs[:, 1] == pairs[:, 3]))
        self.assertTrue(np.all(np.sort(pairs[:, [2, 4]], axis=1) == np.array([23, 47])))

    def test_fixed_random_relation_is_balanced_and_collision_free(self):
        mapping = build_collision_free_pseudo_systems(100, 8, 20260821)
        self.assertEqual(mapping.shape, (100, 8))
        self.assertTrue(all(np.unique(row).size == 8 for row in mapping))
        self.assertTrue(all(np.unique(mapping[:, col]).size == 100 for col in range(8)))
        pairs = sample_pseudo_system_pairs(mapping, 48, 7)
        self.assertTrue(np.all(pairs[:, 0] != pairs[:, 3]))
        self.assertTrue(np.all(pairs[:, 1] != pairs[:, 4]))

    def test_manifest_determinism(self):
        a = generate_dataset(self.small_config()).manifest()
        b = generate_dataset(self.small_config()).manifest()
        self.assertEqual(a["manifest_sha256"], b["manifest_sha256"])

    def test_test_seal(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selection.json"
            with self.assertRaises(TestSealedError):
                require_test_unsealed(path)
            payload = {
                "variant": "B3",
                "full_config": {},
                "checkpoint_sha256": "a" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "primary_metric": {"name": "h16", "value": 0.1},
                "tie_breakers": [],
                "decoder_ridge": 0.01,
                "pairing_seed": 20260819,
                "code_revision": "dirty",
                "created_at": "2026-08-18T00:00:00+08:00",
            }
            write_immutable_selection(path, payload)
            self.assertEqual(require_test_unsealed(path)["variant"], "B3")
            with self.assertRaises(FileExistsError):
                write_immutable_selection(path, payload)


if __name__ == "__main__":
    unittest.main()
