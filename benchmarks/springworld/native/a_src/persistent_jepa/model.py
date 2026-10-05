"""D-Clean model family with one frozen z_s/z_p data flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "B3"
    observation_dim: int = 128
    transient_dim: int = 64
    persistent_dim: int = 64
    transformer_width: int = 192
    transformer_depth: int = 4
    transformer_heads: int = 8
    dropout: float = 0.1
    horizon_dim: int = 32
    history_length: int = 24


class MLP(nn.Module):
    def __init__(self, dims: list[int], final_norm: nn.Module | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if final_norm is not None:
            layers.append(final_norm)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.mlp = MLP([4, 192, 192, output_dim])
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape[:-1]
        encoded = self.mlp(state.reshape(-1, 4))
        return self.norm(encoded).reshape(*shape, -1)


class HistoryEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, output_dim: int) -> None:
        super().__init__()
        self.action = MLP([2, 32, 32])
        self.token = nn.Linear(cfg.observation_dim + 32, cfg.transformer_width)
        self.position = nn.Parameter(
            torch.zeros(1, cfg.history_length, cfg.transformer_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_width,
            nhead=cfg.transformer_heads,
            dim_feedforward=4 * cfg.transformer_width,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_depth)
        self.output = MLP([cfg.transformer_width, 128, output_dim])

    def forward(self, history_h: torch.Tensor, history_actions: torch.Tensor) -> torch.Tensor:
        if history_h.shape[1] != self.position.shape[1]:
            raise ValueError(
                f"history length {history_h.shape[1]} does not match configured "
                f"length {self.position.shape[1]}"
            )
        zeros = torch.zeros_like(history_actions[:, :1])
        previous_actions = torch.cat([zeros, history_actions], dim=1)
        token = self.token(torch.cat([history_h, self.action(previous_actions)], dim=-1))
        token = token + self.position[:, : token.shape[1]]
        length = token.shape[1]
        mask = torch.triu(torch.ones(length, length, device=token.device, dtype=torch.bool), diagonal=1)
        return self.output(self.transformer(token, mask=mask)[:, -1])


class Predictor(nn.Module):
    def __init__(self, cfg: ModelConfig, context_dim: int = 128) -> None:
        super().__init__()
        self.horizon = nn.Embedding(3, cfg.horizon_dim)
        input_dim = context_dim + 32 + 16 + cfg.horizon_dim
        self.net = MLP([input_dim, 256, 256, cfg.observation_dim])

    def forward(
        self,
        context: torch.Tensor,
        padded_actions: torch.Tensor,
        action_mask: torch.Tensor,
        horizon_index: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [context, padded_actions.flatten(1), action_mask, self.horizon(horizon_index)], dim=-1
        )
        return self.net(features)


class PersistentJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation = ObservationEncoder(cfg.observation_dim)
        if cfg.variant == "B0":
            self.context = HistoryEncoder(cfg, output_dim=128)
            self.transient = None
            self.persistent = None
        else:
            self.context = None
            self.transient = MLP([cfg.observation_dim, 128, cfg.transient_dim])
            self.persistent = HistoryEncoder(cfg, output_dim=cfg.persistent_dim)
        self.predictor = Predictor(cfg, context_dim=128)
        self.supervised_gamma = MLP([cfg.persistent_dim, 64, 1]) if cfg.variant == "Sup" else None

    def encode_observations(
        self, history_states: torch.Tensor, target_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time = history_states.shape[:2]
        all_states = torch.cat([history_states, target_states], dim=1)
        all_h = self.observation(all_states)
        return all_h[:, :time], all_h[:, time:]

    def codes(
        self, history_h: torch.Tensor, history_actions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.cfg.variant == "B0":
            context = self.context(history_h, history_actions)
            return None, None, context
        z_s = self.transient(history_h[:, -1])
        z_p = self.persistent(history_h, history_actions)
        return z_s, z_p, torch.cat([z_s, z_p], dim=-1)
