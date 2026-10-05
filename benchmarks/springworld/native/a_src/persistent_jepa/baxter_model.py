"""Small split world model for the frozen A1 Baxter tactile replication."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .baxter_data import CONDITIONS
from .losses import SIGReg, canonical_vicreg
from .model import MLP


HORIZONS = (1, 20, 40)


@dataclass(frozen=True)
class BaxterModelConfig:
    condition: str
    history_length: int = 40
    observation_dim: int = 128
    transient_dim: int = 64
    persistent_dim: int = 64
    transformer_width: int = 192
    transformer_depth: int = 4
    transformer_heads: int = 8
    dropout: float = 0.1
    horizon_dim: int = 32

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown Baxter condition {self.condition}")


class BaxterObservationEncoder(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.mlp = MLP([16, 192, 192, output_dim])
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1]
        encoded = self.mlp(value.reshape(-1, 16))
        return self.norm(encoded).reshape(*shape, -1)


class BaxterHistoryEncoder(nn.Module):
    def __init__(self, length: int, output_dim: int, cfg: BaxterModelConfig) -> None:
        super().__init__()
        self.length = length
        self.token = nn.Linear(cfg.observation_dim, cfg.transformer_width)
        self.position = nn.Parameter(torch.zeros(1, length, cfg.transformer_width))
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

    def forward(self, history_h: torch.Tensor) -> torch.Tensor:
        if history_h.shape[1] != self.length:
            raise ValueError(f"history length {history_h.shape[1]} does not match {self.length}")
        token = self.token(history_h) + self.position
        length = token.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=token.device, dtype=torch.bool), diagonal=1
        )
        return self.output(self.transformer(token, mask=mask)[:, -1])


class BaxterPredictor(nn.Module):
    def __init__(self, cfg: BaxterModelConfig) -> None:
        super().__init__()
        self.horizon = nn.Embedding(len(HORIZONS), cfg.horizon_dim)
        self.net = MLP([128 + cfg.horizon_dim, 256, 256, cfg.observation_dim])

    def forward(self, context: torch.Tensor, horizon_index: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([context, self.horizon(horizon_index)], dim=-1))


class BaxterJEPA(nn.Module):
    def __init__(self, cfg: BaxterModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation = BaxterObservationEncoder(cfg.observation_dim)
        self.transient = MLP([cfg.observation_dim, 128, cfg.transient_dim])
        self.persistent = BaxterHistoryEncoder(cfg.history_length, cfg.persistent_dim, cfg)
        self.predictor = BaxterPredictor(cfg)

    def codes(self, history_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s = self.transient(history_h[:, -1])
        z_p = self.persistent(history_h)
        return z_s, z_p, torch.cat([z_s, z_p], dim=-1)


def _prediction_loss(
    model: BaxterJEPA, context: torch.Tensor, target_h: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    for horizon_index in range(len(HORIZONS)):
        index = torch.full(
            (context.shape[0],), horizon_index, dtype=torch.long, device=context.device
        )
        losses.append(F.mse_loss(model.predictor(context, index), target_h[:, horizon_index]))
    total = torch.stack(losses).mean()
    metrics = {f"latent_h{horizon}": loss.detach() for horizon, loss in zip(HORIZONS, losses, strict=True)}
    return total, metrics


def baxter_objective(
    model: BaxterJEPA,
    batch: object,
    sigreg: SIGReg,
    *,
    sigreg_weight: float = 0.02,
    lambda_p: float = 1.0,
    lambda_x: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query_all = torch.cat([batch.query_history, batch.query_targets], dim=1)
    query_all_h = model.observation(query_all)
    history_h = query_all_h[:, : model.cfg.history_length]
    target_h = query_all_h[:, model.cfg.history_length :]
    z_s, z_p, context = model.codes(history_h)
    self_loss, self_parts = _prediction_loss(model, context, target_h)
    sigreg_loss = sigreg(query_all_h.transpose(0, 1))
    donor_h = model.observation(batch.donor_history)
    donor_z_p = model.persistent(donor_h)
    persist_loss, persist_metrics = canonical_vicreg(donor_z_p, z_p)
    cross_loss, cross_parts = _prediction_loss(model, torch.cat([z_s, donor_z_p], dim=-1), target_h)
    total = self_loss + sigreg_weight * sigreg_loss + lambda_p * persist_loss + lambda_x * cross_loss
    return total, {
        "loss_total": total.detach(),
        "loss_self": self_loss.detach(),
        "loss_sigreg": sigreg_loss.detach(),
        "loss_persist": persist_loss.detach(),
        "loss_cross": cross_loss.detach(),
        **{f"self_{key}": value for key, value in self_parts.items()},
        **{f"cross_{key}": value for key, value in cross_parts.items()},
        **persist_metrics,
    }


class BaxterCertificate(nn.Module):
    """Randomly initialized supervised input-accessibility diagnostic."""

    def __init__(self, input_length: int) -> None:
        super().__init__()
        cfg = BaxterModelConfig("Random", history_length=input_length)
        self.input_length = input_length
        self.observation = BaxterObservationEncoder(cfg.observation_dim)
        self.history = BaxterHistoryEncoder(input_length, 128, cfg)
        self.hardness = nn.Linear(128, 3)
        self.shape = nn.Linear(128, 2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.history(self.observation(value))
        return self.hardness(code), self.shape(code)
