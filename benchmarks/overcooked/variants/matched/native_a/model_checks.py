"""Small synthetic CPU contract checks; no environment data or optimizer steps.

Run from the repository root: ``python -m native_a.model_checks --out receipt.json``.
Missing JAX/Flax is a BLOCKED check (exit 2), not a successful skipped test.
The tiny fixture checks semantics, not full-model throughput or scientific fit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("jax", "flax"))
if AVAILABLE:
    import jax
    import jax.numpy as jnp
    from benchmarks.baselines.ad.model import ADConfig, ADModel
    from native_a.model import (ADHiddenBackbone, PersistentADConfig, PersistentADModel,
                                apply_batch, batch_model_kwargs, validate_model_batch)
    from native_a.losses import (LossConfig, canonical_components, canonical_vicreg,
                                 loss_from_batch, masked_action_ce, native_objective)


def numpy_components(a, b):
    """Independent transcription of the frozen Paper A branch-wise equations."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    inv = ((a - b) ** 2).mean()
    sa, sb = np.sqrt(a.var(axis=0, ddof=1) + 1e-4), np.sqrt(b.var(axis=0, ddof=1) + 1e-4)
    var = .5 * (np.maximum(1 - sa, 0).mean() + np.maximum(1 - sb, 0).mean())
    cov = 0.0
    for z in (a, b):
        centered = z - z.mean(axis=0, keepdims=True)
        matrix = centered.T @ centered / (len(z) - 1)
        off = matrix[~np.eye(z.shape[1], dtype=bool)]
        cov += np.square(off).sum() / z.shape[1]
    return np.asarray([inv, var, cov])


@unittest.skipUnless(AVAILABLE, "JAX/Flax unavailable")
class NativeModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(d.platform != "cpu" for d in jax.devices()):
            raise RuntimeError("Synthetic model checks must run entirely on CPU")
        ad = ADConfig(obs_shape=(3, 3, 4), embedding_dim=8, hidden_dim=16,
                      num_layers=1, num_heads=2, seq_len=16,
                      attention_dropout=0., residual_dropout=0., embedding_dropout=0.)
        cls.config = PersistentADConfig(ad=ad, return_cross_logits=True)
        cls.model = PersistentADModel(cls.config)
        rng = np.random.default_rng(14)
        cls.batch = {
            "query": {
                "obs": rng.normal(size=(3, 4, 3, 3, 4)).astype(np.float32),
                "prev_actions": rng.integers(0, 6, (3, 4), dtype=np.int32),
                "prev_rewards": rng.normal(size=(3, 4)).astype(np.float32),
                "attention_mask": np.ones((3, 4), dtype=bool),
                "target_actions": rng.integers(0, 6, (3, 4), dtype=np.int32),
            },
            "support": {
                "obs": rng.normal(size=(3, 2, 3, 3, 3, 4)).astype(np.float32),
                "prev_actions": rng.integers(0, 6, (3, 2, 3), dtype=np.int32),
                "prev_rewards": rng.normal(size=(3, 2, 3)).astype(np.float32),
                "attention_mask": np.ones((3, 2, 3), dtype=bool),
            },
            "pair_indices": np.broadcast_to(np.array([0, 1], np.int32), (3, 2)),
        }
        cls.params = cls.model.init(jax.random.PRNGKey(4),
                                   **batch_model_kwargs(cls.batch, train=False))["params"]
        cls.output = apply_batch(cls.model, cls.params, cls.batch, train=False)

    def assertTreeClose(self, a, b, atol=1e-6):
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
            np.testing.assert_allclose(x, y, atol=atol, rtol=1e-6)

    def test_upstream_dynamic_path_unchanged(self):
        ad = ADModel(self.config.ad)
        query = {k: v for k, v in self.batch["query"].items() if k != "target_actions"}
        variables = ad.init(jax.random.PRNGKey(10), **query, train=False)
        expected = ad.apply(variables, **query, train=False)
        h = ADHiddenBackbone(self.config.ad).apply(variables, **query, train=False)
        head = variables["params"]["action_head"]
        actual = h @ head["kernel"] + head["bias"]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_future_query_tokens_cannot_change_past_logits(self):
        changed = copy.deepcopy(self.batch)
        changed["query"]["obs"][:, 2:] += 20
        changed["query"]["prev_actions"][:, 2:] = (changed["query"]["prev_actions"][:, 2:] + 1) % 6
        output = apply_batch(self.model, self.params, changed, train=False)
        np.testing.assert_allclose(output["logits"][:, :2], self.output["logits"][:, :2], atol=1e-5)
        np.testing.assert_allclose(output["support_persistent"], self.output["support_persistent"], atol=1e-6)

    def test_query_targets_do_not_enter_model(self):
        changed = copy.deepcopy(self.batch)
        changed["query"]["target_actions"][:] = (changed["query"]["target_actions"] + 1) % 6
        self.assertTreeClose(apply_batch(self.model, self.params, changed, train=False), self.output)

    def test_supports_independent_and_slot_consumed(self):
        changed = copy.deepcopy(self.batch)
        changed["support"]["obs"][:, 0] += 5
        output = apply_batch(self.model, self.params, changed, train=False)
        np.testing.assert_allclose(output["support_persistent"][:, 1], self.output["support_persistent"][:, 1], atol=1e-6)
        np.testing.assert_allclose(output["query_hidden"], self.output["query_hidden"], atol=1e-6)
        replacement = jnp.ones((3, 32)) * 4
        replaced = apply_batch(self.model, self.params, self.batch, train=False,
                               persistent_override=replacement)
        np.testing.assert_allclose(replaced["query_hidden"], self.output["query_hidden"], atol=1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(replaced["logits"] - self.output["logits"]))), 1e-6)
        gradients = jax.grad(lambda p: loss_from_batch(
            apply_batch(self.model, p, self.batch, train=False), self.batch, LossConfig())[0])(self.params)
        self.assertGreater(float(jnp.abs(gradients["persistent_projection"]["kernel"]).sum()), 0)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(gradients)))

    def test_cold_start_missing_histories_are_zero_and_finite(self):
        empty = copy.deepcopy(self.batch)
        empty["support"]["attention_mask"][:] = False
        with self.assertRaises(ValueError):
            validate_model_batch(empty, self.config)
        validate_model_batch(empty, self.config, allow_empty_supports=True)
        output = apply_batch(self.model, self.params, empty, train=False)
        self.assertTrue(np.isfinite(output["logits"]).all())
        np.testing.assert_array_equal(output["persistent"], np.zeros((3, 32)))
        np.testing.assert_array_equal(output["support_weights"], np.zeros((3, 2)))
        one = copy.deepcopy(empty)
        one["support"]["attention_mask"][:, 0] = True
        output = apply_batch(self.model, self.params, one, train=False)
        np.testing.assert_allclose(output["support_weights"], [[1, 0]] * 3, atol=1e-7)
        np.testing.assert_allclose(output["persistent"], output["support_persistent"][:, 0], atol=1e-7)

    def test_padding_and_support_permutation(self):
        padded = copy.deepcopy(self.batch)
        padded["support"]["attention_mask"][:, :, -1] = False
        first = apply_batch(self.model, self.params, padded, train=False)
        padded["support"]["obs"][:, :, -1] += 50
        second = apply_batch(self.model, self.params, padded, train=False)
        np.testing.assert_allclose(first["persistent"], second["persistent"], atol=1e-5)
        swapped = copy.deepcopy(self.batch)
        swapped["support"] = {k: v[:, ::-1].copy() for k, v in swapped["support"].items()}
        swapped_output = apply_batch(self.model, self.params, swapped, train=False)
        np.testing.assert_allclose(swapped_output["persistent"], self.output["persistent"], atol=1e-5)

    def test_canonical_math_and_invariance_isolated(self):
        rng = np.random.default_rng(77)
        a, b = rng.normal(size=(7, 32)).astype(np.float32), rng.normal(size=(7, 32)).astype(np.float32)
        expected = numpy_components(a, b)
        actual = canonical_components(jnp.array(a), jnp.array(b))
        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-6)
        value, _ = canonical_vicreg(jnp.array(a), jnp.array(b))
        np.testing.assert_allclose(value, np.array([25, 25, 1]) @ expected, rtol=2e-5)
        vc, vm = loss_from_batch(self.output, self.batch, LossConfig("VC", .003))
        both, im = loss_from_batch(self.output, self.batch, LossConfig("I+VC", .003))
        np.testing.assert_allclose(both - vc, .003 * 25 * im["persist_inv"], rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(vm["ego_ce"], im["ego_ce"], atol=0)
        repeated = self.model.init(jax.random.PRNGKey(4), **batch_model_kwargs(self.batch, train=False))["params"]
        self.assertTreeClose(self.params, repeated, atol=0)

    def test_ce_mask_modes_and_optional_cross(self):
        query = self.batch["query"]
        ce, _, _ = masked_action_ce(self.output["logits"], query["target_actions"], query["attention_mask"])
        for mode in ("none", "VC", "I+VC"):
            total, _ = loss_from_batch(self.output, self.batch, LossConfig(mode, 0))
            np.testing.assert_allclose(total, ce, atol=1e-7)
        total, metrics = loss_from_batch(self.output, self.batch, LossConfig("none", 0, .2))
        np.testing.assert_allclose(total, ce + .2 * metrics["cross_ce"], atol=1e-6)
        one_output = {k: v[:1] for k, v in self.output.items()}
        value, _ = native_objective(one_output, query["target_actions"][:1], query["attention_mask"][:1])
        self.assertTrue(np.isfinite(value))
        with self.assertRaises(ValueError):
            native_objective(one_output, query["target_actions"][:1], query["attention_mask"][:1], mode="VC")
        wrong_pairs = np.zeros((3, 2), dtype=np.int32)
        invalid, _ = native_objective(self.output, query["target_actions"], query["attention_mask"], pair_indices=wrong_pairs)
        self.assertFalse(np.isfinite(invalid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    receipt = {"synthetic_only": True, "environment_data_read": False,
               "optimizer_updates": 0, "scientific_training": False,
               "full_model_throughput_measured": False,
               "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                 for name in ("model.py", "losses.py", "model_checks.py")}}
    if not AVAILABLE:
        receipt.update(status="BLOCKED_MISSING_DEPENDENCIES", tests_run=0)
        code = 2
    else:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeModelChecks))
        receipt.update(status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
                       failures=[str(x[0]) for x in result.failures], errors=[str(x[0]) for x in result.errors],
                       jax_version=jax.__version__, devices=[str(x) for x in jax.devices()],
                       fixture={"B": 3, "K": 2, "S": 3, "L": 4, "hidden_dim": 16, "layers": 1})
        code = 0 if result.wasSuccessful() else 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
