"""AD with a separate 32-dimensional representation of past partner episodes.

The dynamic path uses the upstream AD CNN, action tokens and causal Transformer.
Each support episode is encoded separately with the same backbone.  A small
attention pool combines support slots, and a fusion head actually consumes the
pooled slot when predicting the ego action.  No method/loss switch changes this
architecture.  The sampler must enforce past-only, disjoint support episodes and
the original observation permissions and total history budget.

The ordinary ego CE already gives an independent episode's slot a functional
role in another episode.  Optional per-support replacement logits are exposed
for an explicitly registered cross loss; the first protocol sets its weight to
zero.  Neither option implements JEPA future-embedding prediction or SIGReg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.baselines.ad.model import ADConfig, ADModel


@dataclass
class PersistentADConfig:
    ad: ADConfig = field(default_factory=ADConfig)
    persistent_dim: int = 32
    fusion_hidden_dim: Optional[int] = None
    return_cross_logits: bool = False

    def __post_init__(self):
        if self.persistent_dim != 32:
            raise ValueError("The native-A protocol fixes persistent_dim=32")
        if self.fusion_hidden_dim is not None and self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive")


class ADHiddenBackbone(ADModel):
    """The upstream AD forward path up to (but excluding) its action head.

    Inherits the actual upstream submodules and parameter names.  With upstream
    parameters, applying the upstream action head to this result reproduces AD.
    No episode mask is introduced on the dynamic path.
    """

    @nn.compact
    def __call__(self, obs, prev_actions, prev_rewards, attention_mask=None,
                 prev_teammate_actions=None, train=True):
        cfg = self.config
        if obs.ndim != 5 or tuple(obs.shape[-3:]) != tuple(cfg.obs_shape):
            raise ValueError("AD observations must have shape (B,L,H,W,C)")
        if prev_actions.shape != obs.shape[:2] or prev_rewards.shape != obs.shape[:2]:
            raise ValueError("AD token arrays must share (B,L)")
        if attention_mask is not None and attention_mask.shape != obs.shape[:2]:
            raise ValueError("AD attention mask must have shape (B,L)")
        if cfg.use_teammate_actions and prev_teammate_actions is None:
            raise ValueError("Configured teammate conditioning requires past teammate actions")
        if not cfg.use_teammate_actions and prev_teammate_actions is not None:
            raise ValueError("Teammate actions were not granted by ADConfig")
        if prev_teammate_actions is not None and prev_teammate_actions.shape != obs.shape[:2]:
            raise ValueError("Past teammate actions must have shape (B,L)")

        obs_emb = self.obs_encoder(obs, train=train)
        action_emb = self.action_embedding(prev_actions)
        reward_emb = prev_rewards[:, :, None]
        pieces = [action_emb]
        if cfg.use_teammate_actions:
            pieces.append(self.teammate_action_embedding(prev_teammate_actions))
        pieces.extend([reward_emb, obs_emb])
        sequence = self.embed_token(jnp.concatenate(pieces, axis=-1))
        sequence = nn.Dropout(rate=cfg.embedding_dropout, deterministic=not train)(sequence)
        length = prev_rewards.shape[1]
        causal_mask = jnp.tril(jnp.ones((length, length)))[None, None, :, :]
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask[:, None, None, :]
        x = sequence
        for block in self.blocks:
            x = block(x, mask=causal_mask, train=train)
        if cfg.pre_norm:
            x = self.final_norm(x)
        return x


class PersistentADModel(nn.Module):
    config: PersistentADConfig

    def setup(self):
        cfg = self.config
        self.backbone = ADHiddenBackbone(cfg.ad)
        self.persistent_projection = nn.Dense(cfg.persistent_dim)
        self.history_pool_score = nn.Dense(1, use_bias=False)
        self.fusion = nn.Dense(cfg.fusion_hidden_dim or cfg.ad.hidden_dim)
        self.action_head = nn.Dense(cfg.ad.num_actions)

    def encode_query(self, obs, prev_actions, prev_rewards, attention_mask=None,
                     prev_teammate_actions=None, train=True):
        return self.backbone(obs, prev_actions, prev_rewards, attention_mask,
                             prev_teammate_actions, train)

    def encode_supports(self, obs, prev_actions, prev_rewards, attention_mask,
                        prev_teammate_actions=None, train=True):
        if obs.ndim != 6:
            raise ValueError("Support observations must have shape (B,K,S,H,W,C)")
        batch, count, length = obs.shape[:3]
        if count < 2:
            raise ValueError("At least two independent support episodes are required")
        if length < 1 or any(x.shape != (batch, count, length)
                             for x in (prev_actions, prev_rewards, attention_mask)):
            raise ValueError("Support token arrays must share nonempty (B,K,S)")
        if prev_teammate_actions is not None and prev_teammate_actions.shape != (batch, count, length):
            raise ValueError("Past support teammate actions must have shape (B,K,S)")
        flatten = lambda x: x.reshape((batch * count, length) + x.shape[3:])
        hidden = self.backbone(
            flatten(obs), flatten(prev_actions), flatten(prev_rewards),
            flatten(attention_mask),
            None if prev_teammate_actions is None else flatten(prev_teammate_actions), train,
        )
        mask = attention_mask.reshape(batch * count, length).astype(bool)
        # Use the last valid causal token of each independent episode/window.
        last = jnp.max(jnp.where(mask, jnp.arange(length)[None, :], -1), axis=1)
        selected = hidden[jnp.arange(batch * count), jnp.maximum(last, 0)]
        slots = self.persistent_projection(selected).reshape(batch, count, -1)
        available = (last >= 0).reshape(batch, count)
        # Online cold start has zero/one completed past episode.  Its missing
        # support is not a real observation and receives zero mass and zero code.
        # Training data validation still requires two nonempty real episodes.
        slots = jnp.where(available[..., None], slots, 0.0)
        scores = self.history_pool_score(jnp.tanh(slots)).squeeze(-1)
        scores = jnp.where(available, scores, -1e30)
        weights = jax.nn.softmax(scores, axis=1) * available
        weights = weights / jnp.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        pooled = jnp.sum(weights[..., None] * slots, axis=1)
        return slots, pooled, weights

    def predict_with_slot(self, query_hidden, persistent):
        if persistent.shape != (query_hidden.shape[0], self.config.persistent_dim):
            raise ValueError("Persistent replacement must have shape (B,32)")
        slot = jnp.broadcast_to(persistent[:, None, :],
                                query_hidden.shape[:2] + (self.config.persistent_dim,))
        fused = nn.gelu(self.fusion(jnp.concatenate([query_hidden, slot], axis=-1)))
        return self.action_head(fused)

    def __call__(self, obs, prev_actions, prev_rewards, attention_mask=None,
                 prev_teammate_actions=None, train=True, *, support_obs,
                 support_prev_actions, support_prev_rewards, support_mask,
                 support_prev_teammate_actions=None, persistent_override=None):
        query_hidden = self.encode_query(obs, prev_actions, prev_rewards, attention_mask,
                                         prev_teammate_actions, train)
        slots, pooled, weights = self.encode_supports(
            support_obs, support_prev_actions, support_prev_rewards, support_mask,
            support_prev_teammate_actions, train,
        )
        if slots.shape[0] != query_hidden.shape[0]:
            raise ValueError("Query and support batch sizes differ")
        consumed = pooled if persistent_override is None else persistent_override
        outputs = {
            "logits": self.predict_with_slot(query_hidden, consumed),
            "persistent": pooled,
            "consumed_persistent": consumed,
            "support_persistent": slots,
            "support_weights": weights,
            "query_hidden": query_hidden,
        }
        if self.config.return_cross_logits:
            # Replace the aggregate by one actual same-partner episode at a time.
            # Query observations/dynamic states/ego labels stay identical.
            outputs["cross_logits"] = jnp.stack([
                self.predict_with_slot(query_hidden, slots[:, k])
                for k in range(slots.shape[1])
            ], axis=1)
        return outputs


def batch_model_kwargs(batch: Mapping[str, Any], *, train: bool) -> dict[str, Any]:
    """Map the sampler's public query/support dictionaries to the model API."""
    query, support = batch["query"], batch["support"]
    return {
        "obs": query["obs"], "prev_actions": query["prev_actions"],
        "prev_rewards": query["prev_rewards"], "attention_mask": query["attention_mask"],
        "prev_teammate_actions": query.get("prev_teammate_actions"), "train": train,
        "support_obs": support["obs"], "support_prev_actions": support["prev_actions"],
        "support_prev_rewards": support["prev_rewards"], "support_mask": support["attention_mask"],
        "support_prev_teammate_actions": support.get("prev_teammate_actions"),
    }


def apply_batch(model, params, batch, *, train: bool, dropout_rng=None, persistent_override=None):
    """Apply raw Flax params (not the outer variables dictionary) to a batch."""
    kwargs = batch_model_kwargs(batch, train=train)
    if persistent_override is not None:
        kwargs["persistent_override"] = persistent_override
    if train and dropout_rng is None:
        raise ValueError("Training forward requires an explicit dropout RNG")
    return model.apply({"params": params}, **kwargs,
                       rngs={"dropout": dropout_rng} if train else None)


def validate_model_batch(batch: Mapping[str, Any], config: PersistentADConfig,
                         *, allow_empty_supports: bool = False) -> None:
    """Host-side array checks before JIT; temporal/partner metadata stays in sampler.

    Call once on each new batch contract, not from a traced loss.  This validates
    padding and permissions, not whether a donor really precedes the query.
    """
    sizes = []
    for role, token_ndim in (("query", 2), ("support", 3)):
        data = batch[role]
        mask = np.asarray(data["attention_mask"])
        obs = np.asarray(data["obs"])
        if mask.ndim != token_ndim or obs.shape != mask.shape + tuple(config.ad.obs_shape):
            raise ValueError(f"{role} observation/mask shape mismatch")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{role} masks must be binary")
        if (role != "support" or not allow_empty_supports) and not (mask.sum(axis=-1) > 0).all():
            raise ValueError(f"{role} masks must be nonempty for training")
        if (np.diff(mask.astype(np.int8), axis=-1) > 0).any():
            raise ValueError(f"{role} masks must be right padded")
        if not np.isfinite(obs).all():
            raise ValueError(f"{role} observations must be finite")
        for key in ("prev_actions", "prev_rewards"):
            values = np.asarray(data[key])
            if values.shape != mask.shape or not np.isfinite(values).all():
                raise ValueError(f"{role}.{key} shape/finiteness mismatch")
        keys = ["prev_actions"]
        if config.ad.use_teammate_actions:
            if "prev_teammate_actions" not in data:
                raise ValueError(f"{role} lacks configured teammate actions")
            keys.append("prev_teammate_actions")
        elif "prev_teammate_actions" in data:
            raise ValueError(f"{role} adds unconfigured teammate information")
        for key in keys:
            values = np.asarray(data[key])
            if values.shape != mask.shape or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{role}.{key} must be integer token IDs")
            if ((values < 0) | (values >= config.ad.num_actions)).any():
                raise ValueError(f"{role}.{key} contains an invalid action")
        sizes.append(mask.shape[0])
    if sizes[0] != sizes[1] or batch["support"]["obs"].shape[1] < 2:
        raise ValueError("Query/support batch mismatch or fewer than two supports")
