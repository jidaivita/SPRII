"""Native ego-action CE and the exact branch-wise Paper A VICReg mathematics.

Only the persistent support slots receive relational losses.  No whole-AD-hidden
alignment, partner-label supervision, SIGReg or JEPA target encoder is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


PAPER_A_LOSSES_SHA256 = "700a666484c82f7326502a87cfc7a6f77e4e30c9c8c58d56d9427e9d48994318"
CANONICAL_SOURCE_SHA256 = PAPER_A_LOSSES_SHA256
MODES = ("none", "VC", "I+VC")


@dataclass(frozen=True)
class LossConfig:
    mode: str = "none"
    lambda_p: float = 0.0
    cross_weight: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if any(not math.isfinite(x) or x < 0 for x in (self.lambda_p, self.cross_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")


def canonical_components(z_a, z_b, eps=1e-4):
    if z_a.shape != z_b.shape or z_a.ndim != 2 or min(z_a.shape) < 1:
        raise ValueError("VICReg requires equal nonempty (N,D) branches")
    if z_a.shape[0] < 2:
        raise ValueError("VICReg requires at least two independent partners per branch")
    a, b = z_a.astype(jnp.float32), z_b.astype(jnp.float32)
    inv = jnp.mean(jnp.square(a - b))
    std_a = jnp.sqrt(jnp.var(a, axis=0, ddof=1) + eps)
    std_b = jnp.sqrt(jnp.var(b, axis=0, ddof=1) + eps)
    var = 0.5 * (jnp.maximum(1.0 - std_a, 0).mean() + jnp.maximum(1.0 - std_b, 0).mean())
    ac, bc = a - a.mean(axis=0, keepdims=True), b - b.mean(axis=0, keepdims=True)
    cov_a, cov_b = ac.T @ ac / (a.shape[0] - 1), bc.T @ bc / (b.shape[0] - 1)
    diagonal = jnp.eye(a.shape[1], dtype=bool)
    cov = (jnp.square(jnp.where(diagonal, 0, cov_a)).sum()
           + jnp.square(jnp.where(diagonal, 0, cov_b)).sum()) / a.shape[1]
    return inv, var, cov


def canonical_vicreg(z_a, z_b, eps=1e-4):
    inv, var, cov = canonical_components(z_a, z_b, eps)
    return 25.0 * inv + 25.0 * var + cov, {
        "persist_inv": inv, "persist_var": var, "persist_cov": cov,
    }


def masked_action_ce(logits, targets, mask):
    """The upstream masked ego CE, with invalid labels/all-padding rejected by NaN."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("CE requires logits (B,L,A) and target/mask (B,L)")
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    targets_onehot = jax.nn.one_hot(targets, logits.shape[-1])
    per_token = -jnp.sum(targets_onehot * log_probs, axis=-1)
    mask = mask.astype(jnp.float32)
    valid_count = mask.sum()
    valid = ((targets >= 0) & (targets < logits.shape[-1])) | (mask == 0)
    valid = jnp.all(valid) & jnp.all((mask == 0) | (mask == 1)) & (valid_count > 0)
    ce = jnp.sum(per_token * mask) / jnp.maximum(valid_count, 1.0)
    accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * mask) / jnp.maximum(valid_count, 1.0)
    return jnp.where(valid, ce, jnp.nan), accuracy, valid_count


def native_objective(outputs, target_actions, loss_mask, *, mode="none", lambda_p=0.0,
                     cross_weight=0.0, pair_indices=None):
    """One row = one real fixed partner; query tokens never inflate VICReg N.

    The sampler, not this numeric function, verifies distinct partner identities
    and disjoint preceding support episodes.  ``cross_weight=0`` is the first
    protocol.  A positive weight must be separately registered for all matched
    structure conditions; it supervises replacing the aggregate by each actual
    support slot with the SAME query and ego targets.
    """
    cfg = LossConfig(mode, lambda_p, cross_weight)
    ce, accuracy, count = masked_action_ce(outputs["logits"], target_actions, loss_mask)
    slots = outputs["support_persistent"]
    if slots.ndim != 3 or slots.shape[0] != target_actions.shape[0] or slots.shape[1] < 2:
        raise ValueError("Expected at least two (B,K,D) support slots")
    if pair_indices is None:
        pair_indices = jnp.broadcast_to(jnp.array([0, 1]), (slots.shape[0], 2))
    if pair_indices.shape != (slots.shape[0], 2):
        raise ValueError("pair_indices must have shape (B,2)")
    pair_indices = jnp.asarray(pair_indices)
    if not jnp.issubdtype(pair_indices.dtype, jnp.integer):
        raise ValueError("pair_indices must contain integer indices")
    pair_valid = (jnp.all((pair_indices >= 0) & (pair_indices < slots.shape[1]))
                  & jnp.all(pair_indices[:, 0] != pair_indices[:, 1]))
    zero = jnp.array(0.0, dtype=jnp.float32)
    inv, var, cov, reg = zero, zero, zero, zero
    if cfg.mode != "none":
        rows = jnp.arange(slots.shape[0])
        inv, var, cov = canonical_components(slots[rows, pair_indices[:, 0]],
                                             slots[rows, pair_indices[:, 1]])
        reg = 25.0 * var + cov
        if cfg.mode == "I+VC":
            reg = reg + 25.0 * inv
    cross = zero
    if cfg.cross_weight:
        if "cross_logits" not in outputs:
            raise ValueError("A cross objective requires return_cross_logits=True")
        logits = outputs["cross_logits"]
        if logits.shape != (slots.shape[0], slots.shape[1]) + outputs["logits"].shape[1:]:
            raise ValueError("cross_logits must have shape (B,K,L,A)")
        cross = jnp.mean(jnp.stack([
            masked_action_ce(logits[:, k], target_actions, loss_mask)[0]
            for k in range(slots.shape[1])
        ]))
    total = ce + cfg.lambda_p * reg + cfg.cross_weight * cross
    total = jnp.where(pair_valid, total, jnp.nan)
    return total, {
        "loss": total, "ego_ce": ce, "accuracy": accuracy, "valid_tokens": count,
        "persistent_loss": reg, "persist_inv": inv, "persist_var": var,
        "persist_cov": cov, "cross_ce": cross,
        "persistent_std_mean": jnp.mean(jnp.std(slots[:, 0].astype(jnp.float32), axis=0)),
    }


def loss_from_batch(outputs, batch, config: LossConfig):
    query = batch["query"]
    return native_objective(
        outputs, batch.get("target_actions", query["target_actions"]),
        batch.get("loss_mask", query["attention_mask"]),
        mode=config.mode, lambda_p=config.lambda_p, cross_weight=config.cross_weight,
        pair_indices=batch.get("pair_indices"),
    )
