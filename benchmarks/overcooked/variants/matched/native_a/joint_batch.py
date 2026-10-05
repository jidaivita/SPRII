"""Exact joint-batch objectives with bounded activation memory, for NEW runs.

These factories replace only native.make_step or upstream._train_step_impl.
The caller retains its sampler, optimizer, scheduler, checkpointing and loop.
Each optimizer update uses effective_batch_size rows. CE is normalized by the
whole batch's valid-token count; gradients are summed BEFORE one clip/Adam.

VC/I+VC use two passes at unchanged parameters: collect all support slots, obtain
the full-batch canonical VICReg slot gradient, replay each microbatch with the
same dropout key and apply its slot VJP alongside CE. The slot cotangent includes
lambda exactly once. Per-microbatch variance/covariance losses are never used.

Randomness is defined by split/fold_in keys per update and microbatch. This is
equivalent to a directly differentiated joint objective using those same keys,
NOT bitwise equivalent to one monolithic dropout call with a differently shaped
random tensor. The externally checkpointed RNG advances as in the old interface.
"""
from __future__ import annotations

from functools import partial
import json
import math

import jax
import jax.numpy as jnp
import optax


def _validate_sizes(effective, micro):
    if not isinstance(effective, int) or not isinstance(micro, int) or not (1 <= micro <= effective):
        raise ValueError("Batch sizes must be positive integers with micro <= effective")
    if effective % micro:
        raise ValueError("The physical microbatch must divide the effective batch exactly")


def microbatch_keys(key, count):
    return [jax.random.fold_in(key, index) for index in range(count)]


def _slice(tree, start, stop):
    return jax.tree_util.tree_map(lambda value: value[start:stop], tree)


@jax.jit
def _add(a, b):
    return jax.tree_util.tree_map(jnp.add, a, b)


@jax.jit
def _apply_once(state, gradients):
    return state.apply_gradients(grads=gradients)


def _ce_sums(logits, targets, mask, *, smoothing=0.0, native=False):
    """The official masked action CE numerator; zero-valid microbatches allowed.

Native's full-batch label/mask validation is retained separately. An empty
microbatch cannot make a nonempty global objective NaN.
"""
    if native:
        logits = logits.astype(jnp.float32)
    targets_onehot = jax.nn.one_hot(targets, logits.shape[-1])
    if smoothing:
        targets_onehot = targets_onehot * (1 - smoothing) + smoothing / logits.shape[-1]
    mask = mask.astype(jnp.float32)
    loss_sum = jnp.sum(-jnp.sum(targets_onehot * jax.nn.log_softmax(logits, axis=-1), axis=-1) * mask)
    correct_sum = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * mask)
    return loss_sum, correct_sum, mask.sum()


def _gradient_metrics(gradients):
    return {"gradient_norm": optax.global_norm(gradients),
            "gradients_finite": jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(gradients)]))}


def make_native_step(model, config, *, microbatch_size=128, effective_batch_size=1024):
    """Drop-in factory for native.make_step, with the same (state,batch,rng) API."""
    from native_a.model import apply_batch
    from native_a.losses import canonical_components

    _validate_sizes(effective_batch_size, microbatch_size)
    mode = config["mode"]
    if mode not in ("none", "VC", "I+VC") or config.get("cross_weight", 0.0) != 0:
        raise ValueError("Only the declared none/VC/I+VC objectives are supported")
    weight = float(config["lambda_p"])
    if not math.isfinite(weight) or weight < 0 or (mode == "none" and weight != 0):
        raise ValueError("Invalid native regularization weight")
    if "batch_size" in config and int(config["batch_size"]) != effective_batch_size:
        raise ValueError("Configuration must record the actual effective batch size")
    parts = effective_batch_size // microbatch_size

    @jax.jit
    def get_slots(params, micro, key):
        return apply_batch(model, params, micro, train=True, dropout_rng=key)["support_persistent"]

    def regularizer(slots, pair_indices):
        rows = jnp.arange(slots.shape[0])
        inv, var, cov = canonical_components(slots[rows, pair_indices[:, 0]],
                                             slots[rows, pair_indices[:, 1]])
        raw = 25.0 * var + cov + (25.0 * inv if mode == "I+VC" else 0.0)
        return weight * raw, {"persistent_loss": raw, "persist_inv": inv,
                              "persist_var": var, "persist_cov": cov}

    joint_regularizer = jax.jit(jax.value_and_grad(regularizer, has_aux=True))

    @jax.jit
    def local_gradient(params, micro, key, global_tokens, slot_cotangent):
        def objective(parameters):
            outputs = apply_batch(model, parameters, micro, train=True, dropout_rng=key)
            loss_sum, correct_sum, count = _ce_sums(outputs["logits"], micro["target_actions"],
                                                    micro["loss_mask"], native=True)
            slots = outputs["support_persistent"]
            surrogate = loss_sum / jnp.maximum(global_tokens, 1.0)
            surrogate += jnp.sum(slots * jax.lax.stop_gradient(slot_cotangent))
            return surrogate, (loss_sum, correct_sum, count, slots)
        (_, stats), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        return gradients, stats

    def gradients(state, batch, dropout_rng):
        if int(batch["target_actions"].shape[0]) != effective_batch_size:
            raise ValueError("Native batch does not contain the full effective row count")
        next_rng, use_rng = jax.random.split(dropout_rng)
        keys = microbatch_keys(use_rng, parts)
        mask = jnp.asarray(batch["loss_mask"])
        targets = jnp.asarray(batch["target_actions"])
        global_tokens = mask.astype(jnp.float32).sum()
        labels_valid = jnp.all(((targets >= 0) & (targets < model.config.ad.num_actions)) | (mask == 0))
        inputs_valid = labels_valid & jnp.all((mask == 0) | (mask == 1)) & (global_tokens > 0)
        pair_indices = jnp.asarray(batch["pair_indices"])
        slot_count = int(batch["support"]["prev_actions"].shape[1])
        if pair_indices.shape != (effective_batch_size, 2) or not jnp.issubdtype(pair_indices.dtype, jnp.integer):
            raise ValueError("Positive pair indices must be integer (effective_batch,2)")
        inputs_valid &= (jnp.all((pair_indices >= 0) & (pair_indices < slot_count))
                         & jnp.all(pair_indices[:, 0] != pair_indices[:, 1]))
        zero = jnp.array(0.0, jnp.float32)
        reg_metrics = {key: zero for key in ("persistent_loss", "persist_inv", "persist_var", "persist_cov")}
        weighted_reg = zero
        if mode != "none":
            slots = jnp.concatenate([
                get_slots(state.params, _slice(batch, i * microbatch_size, (i + 1) * microbatch_size), keys[i])
                for i in range(parts)], axis=0)
            (weighted_reg, reg_metrics), slot_cotangents = joint_regularizer(slots, pair_indices)
            slot_cotangents = jax.lax.stop_gradient(slot_cotangents)
        else:
            slots = None
            slot_cotangents = jnp.zeros((effective_batch_size, slot_count, model.config.persistent_dim), jnp.float32)

        accumulated = None
        loss_sum, correct_sum = zero, zero
        replay_slots = []
        for i in range(parts):
            start, stop = i * microbatch_size, (i + 1) * microbatch_size
            local, stats = local_gradient(state.params, _slice(batch, start, stop), keys[i],
                                          global_tokens, slot_cotangents[start:stop])
            accumulated = local if accumulated is None else _add(accumulated, local)
            loss_sum, correct_sum = loss_sum + stats[0], correct_sum + stats[1]
            replay_slots.append(stats[3])
        replayed = jnp.concatenate(replay_slots, axis=0)
        replay_error = zero if slots is None else jnp.max(jnp.abs(replayed - slots))
        replay_rmse = zero if slots is None else jnp.sqrt(jnp.mean(jnp.square(replayed - slots)))
        reference_rms = jnp.sqrt(jnp.mean(jnp.square(replayed if slots is None else slots)))
        # Return the real joint loss, not the stop-gradient linear surrogate.
        ce = loss_sum / jnp.maximum(global_tokens, 1.0)
        metrics = {"loss": jnp.where(inputs_valid, ce + weighted_reg, jnp.nan), "ego_ce": ce,
                   "accuracy": correct_sum / jnp.maximum(global_tokens, 1.0), "valid_tokens": global_tokens,
                   **reg_metrics, "cross_ce": zero,
                   "persistent_std_mean": jnp.mean(jnp.std(replayed[:, 0].astype(jnp.float32), axis=0)),
                   "slot_replay_max_abs_error": replay_error,
                   "slot_replay_rmse": replay_rmse,
                   "slot_reference_rms": reference_rms,
                   "slot_replay_relative_l2_error": replay_rmse / jnp.maximum(reference_rms, 1e-12),
                   "slot_replay_consistent": replay_error <= 1e-5,
                   "objective_inputs_valid": inputs_valid,
                   "effective_batch_size": jnp.asarray(effective_batch_size),
                   "physical_microbatch_size": jnp.asarray(microbatch_size),
                   **_gradient_metrics(accumulated)}
        return accumulated, metrics, next_rng

    def step(state, batch, dropout_rng):
        gradient, metrics, next_rng = gradients(state, batch, dropout_rng)
        if not bool(jax.device_get(metrics["slot_replay_consistent"])):
            detail = {name: float(value) for name, value in jax.device_get(metrics).items()}
            raise RuntimeError("Support slot replay mismatch before optimizer update: " + json.dumps(detail))
        return _apply_once(state, gradient), metrics, next_rng

    step.gradients = gradients
    return step


@partial(jax.jit, static_argnames=("apply_fn", "num_actions", "label_smoothing", "use_teammate_actions"))
def _upstream_local_gradient(params, batch_data, key, global_tokens, *, apply_fn,
                             num_actions, label_smoothing, use_teammate_actions):
    obs, actions, rewards, targets, mask = batch_data[:5]
    teammate = batch_data[5] if use_teammate_actions else None

    def objective(parameters):
        logits = apply_fn(parameters, obs, actions, rewards, attention_mask=mask,
                          prev_teammate_actions=teammate, train=True, rngs={"dropout": key})
        sums = _ce_sums(logits, targets, mask, smoothing=label_smoothing)
        return sums[0] / jnp.maximum(global_tokens, 1.0), sums
    (_, stats), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    return gradients, stats


def make_upstream_train_step_impl(*, microbatch_size=128, effective_batch_size=1024):
    """Drop-in replacement for original _train_step_impl; outer RNG stays native."""
    _validate_sizes(effective_batch_size, microbatch_size)
    parts = effective_batch_size // microbatch_size

    def gradients(state, batch_data, dropout_rng, num_actions, label_smoothing, use_teammate_actions):
        if int(batch_data[0].shape[0]) != effective_batch_size:
            raise ValueError("Original AD batch does not contain the full effective row count")
        global_tokens = jnp.asarray(batch_data[4]).astype(jnp.float32).sum()
        keys = microbatch_keys(dropout_rng, parts)
        accumulated = None
        loss_sum = correct_sum = jnp.array(0.0, jnp.float32)
        for i in range(parts):
            micro = _slice(batch_data, i * microbatch_size, (i + 1) * microbatch_size)
            local, stats = _upstream_local_gradient(state.params, micro, keys[i], global_tokens,
                apply_fn=state.apply_fn, num_actions=num_actions, label_smoothing=label_smoothing,
                use_teammate_actions=use_teammate_actions)
            accumulated = local if accumulated is None else _add(accumulated, local)
            loss_sum, correct_sum = loss_sum + stats[0], correct_sum + stats[1]
        metrics = {"loss": loss_sum / jnp.maximum(global_tokens, 1.0),
                   "accuracy": correct_sum / jnp.maximum(global_tokens, 1.0),
                   "valid_tokens": global_tokens,
                   "effective_batch_size": jnp.asarray(effective_batch_size),
                   "physical_microbatch_size": jnp.asarray(microbatch_size),
                   **_gradient_metrics(accumulated)}
        return accumulated, metrics

    def step(state, batch_data, dropout_rng, num_actions, label_smoothing, use_teammate_actions):
        gradient, metrics = gradients(state, batch_data, dropout_rng, num_actions,
                                      label_smoothing, use_teammate_actions)
        return _apply_once(state, gradient), metrics

    step.gradients = gradients
    return step
