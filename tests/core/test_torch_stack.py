"""Torch-dependent shape, gradient, and loss contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from persistent_jepa.losses import SIGReg, canonical_vicreg
    from persistent_jepa.model import ModelConfig, PersistentJEPA
    from persistent_jepa.objective import compute_objective
    from persistent_jepa.poke_model import PokeJEPA, poke_objective
    from persistent_jepa.poke_evaluation import fixed_eval_arrays, targeted_systems
    from persistent_jepa.poke_torch import PokeSplit, legal_poke_anchors
    from persistent_jepa.pokeworld import PokeConfig, generate_pokeworld, save_pokeworld
    from persistent_jepa.sampling import build_collision_free_pseudo_systems, deterministic_derangement
    from persistent_jepa.simulator import DCleanConfig, generate_dataset
    from persistent_jepa.torch_data import SplitArrays
except ImportError:  # Local documentation environment intentionally has no torch.
    torch = None


@unittest.skipIf(torch is None, "torch unavailable")
class TorchStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cfg = DCleanConfig(train_systems=8, val_systems=3, test_systems=3, rollouts_per_system=4)
        generate_dataset(cfg).save(Path(cls.temp.name))
        cls.data = SplitArrays(Path(cls.temp.name), "train")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_vectorized_batch_contract(self) -> None:
        batch = self.data.paired_batch(4, 123)
        self.assertEqual(tuple(batch.history_states.shape), (8, 24, 4))
        self.assertEqual(tuple(batch.history_actions.shape), (8, 23, 2))
        self.assertEqual(tuple(batch.target_states.shape), (8, 3, 4))
        self.assertEqual(tuple(batch.future_actions.shape), (8, 3, 16, 2))
        self.assertEqual(tuple(batch.action_masks.shape), (8, 3, 16))
        self.assertTrue(torch.equal(batch.gamma[:4], batch.gamma[4:]))

    def test_dclean_relation_batch_contracts(self) -> None:
        b1 = self.data.paired_batch(4, 123, pairing_mode="same_rollout_nonoverlap")
        self.assertEqual(tuple(b1.history_states.shape), (8, 24, 4))
        mapping = build_collision_free_pseudo_systems(8, 4, 20260821)
        random_batch = self.data.paired_batch(
            4, 123, pairing_mode="random_system_fixed", pseudo_systems=mapping
        )
        self.assertFalse(torch.equal(random_batch.gamma[:4], random_batch.gamma[4:]))

    def test_canonical_vicreg_scaling(self) -> None:
        a = torch.randn(48, 64)
        b = torch.randn(48, 64)
        loss, metrics = canonical_vicreg(a, b)
        reconstructed = (
            25 * metrics["persist_inv"]
            + 25 * metrics["persist_var"]
            + metrics["persist_cov"]
        )
        self.assertTrue(torch.allclose(loss.detach(), reconstructed, atol=1e-5))
        self.assertTrue(torch.isfinite(loss))

    def test_all_registered_objectives_backpropagate(self) -> None:
        batch = self.data.paired_batch(4, 456)
        for variant in ("B0", "Sup", "B0_split", "B1", "B2", "B3"):
            with self.subTest(variant=variant):
                model = PersistentJEPA(
                    ModelConfig(
                        variant=variant,
                        transformer_depth=1,
                        dropout=0.0,
                    )
                )
                loss, metrics = compute_objective(
                    model,
                    batch,
                    SIGReg(num_directions=16),
                    sigreg_weight=0.02,
                    lambda_p=0.3,
                    lambda_x=0.3,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(sum(p.grad is not None for p in model.parameters()), 0)
                self.assertIn("loss_self", metrics)
                if variant == "B3":
                    self.assertIn("loss_cross", metrics)
                if variant == "Sup":
                    self.assertIn("loss_supervised", metrics)

    def test_pokeworld_pixel_objective_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PokeConfig(
                train_systems=3, val_systems=2, test_systems=2, rollouts_per_system=2
            )
            save_pokeworld(root, generate_pokeworld(config), config)
            split = PokeSplit(root, "train")
            batch = split.batch(2, 123)
            self.assertEqual(tuple(batch.history_current.shape), (2, 24, 8))
            self.assertEqual(tuple(batch.target_current.shape), (2, 3, 8))
            self.assertEqual(tuple(batch.future_actions.shape), (2, 3, 16, 2))
            model = PokeJEPA()
            loss, metrics = poke_objective(model, batch, SIGReg(num_directions=8))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("loss_h16", metrics)
            self.assertGreater(sum(parameter.grad is not None for parameter in model.parameters()), 0)
            paired = split.paired_batch(2, 124)
            for variant in ("B0_split", "B2", "B3", "Bx", "Sup"):
                with self.subTest(poke_variant=variant):
                    candidate = PokeJEPA(variant)
                    candidate_loss, candidate_metrics = poke_objective(
                        candidate,
                        paired,
                        SIGReg(num_directions=8),
                        lambda_p=1.0,
                        lambda_x=0.1,
                    )
                    candidate_loss.backward()
                    self.assertTrue(torch.isfinite(candidate_loss))
                    if variant == "B3":
                        self.assertIn("loss_cross", candidate_metrics)
                        self.assertIn("loss_persist", candidate_metrics)
                    if variant == "Bx":
                        self.assertIn("loss_cross", candidate_metrics)
                        self.assertNotIn("loss_persist", candidate_metrics)
                    if variant == "Sup":
                        self.assertIn("loss_supervised", candidate_metrics)

    def test_pokeworld_pairing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PokeConfig(
                train_systems=12, val_systems=3, test_systems=3, rollouts_per_system=4
            )
            save_pokeworld(root, generate_pokeworld(config), config)
            split = PokeSplit(root, "train")
            batch = split.paired_batch(8, 456)
            half = 8
            self.assertTrue(torch.equal(batch.system_index[:half], batch.system_index[half:]))
            self.assertTrue(torch.all(batch.rollout_id[:half] != batch.rollout_id[half:]))
            self.assertTrue(torch.equal(batch.mass[:half], batch.mass[half:]))
            self.assertTrue(torch.equal(batch.gamma[:half], batch.gamma[half:]))
            self.assertTrue(torch.equal(batch.stiffness[:half], batch.stiffness[half:]))
            shuffled = torch.from_numpy(deterministic_derangement(config.train_systems, 789))
            self.assertTrue(torch.all(shuffled != torch.arange(config.train_systems)))
            evaluation = fixed_eval_arrays(split, 790, windows_per_system=2)
            self.assertTrue(
                torch.all(
                    evaluation.target.system_index
                    != evaluation.correct_donor.system_index[evaluation.shuffled_row_index]
                )
            )
            selected, _ = targeted_systems(split)
            for donors in selected.values():
                valid = np.flatnonzero(donors >= 0)
                self.assertTrue(np.all(donors[valid] != valid))

    def test_pokeworld_temporal_difference_anchor_contract(self) -> None:
        expected = {24: (24, 47, 24), 32: (32, 47, 16), 40: (40, 47, 8)}
        for length, (first, last, count) in expected.items():
            anchors = legal_poke_anchors(length)
            self.assertEqual((int(anchors[0]), int(anchors[-1]), anchors.size), (first, last, count))
        self.assertEqual(legal_poke_anchors(48).size, 0)

    def test_pokeworld_variable_history_and_restricted_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PokeConfig(
                train_systems=12, val_systems=3, test_systems=3, rollouts_per_system=4
            )
            save_pokeworld(root, generate_pokeworld(config), config)
            for length, anchor_count in ((24, 24), (32, 16), (40, 8)):
                split = PokeSplit(root, "train", history_length=length)
                self.assertEqual(split.anchors.size, anchor_count)
                batch = split.batch(3, 100 + length)
                self.assertEqual(tuple(batch.history_current.shape), (3, length, 8))
                self.assertEqual(tuple(batch.history_actions.shape), (3, length - 1, 2))
            with self.assertRaises(ValueError):
                PokeSplit(root, "train", history_length=48)

            split = PokeSplit(root, "train", history_length=24)
            same = split.restricted_paired_batch(8, 999, "same_rollout")
            cross = split.restricted_paired_batch(8, 999, "independent_rollout")
            half = 8
            self.assertTrue(torch.equal(same.system_index, cross.system_index))
            self.assertTrue(torch.equal(same.anchor, cross.anchor))
            self.assertTrue(torch.equal(same.rollout_id[half:], cross.rollout_id[half:]))
            self.assertTrue(torch.equal(same.rollout_id[:half], same.rollout_id[half:]))
            self.assertTrue(torch.all(cross.rollout_id[:half] != cross.rollout_id[half:]))
            separation = same.anchor[half:] - same.anchor[:half]
            self.assertTrue(torch.all((separation >= 12) & (separation <= 23)))
            self.assertTrue(torch.all(same.anchor[half:] >= 36))
            overlap = torch.clamp(24 - separation, min=0)
            self.assertTrue(torch.all((overlap >= 1) & (overlap <= 12)))


if __name__ == "__main__":
    unittest.main()
