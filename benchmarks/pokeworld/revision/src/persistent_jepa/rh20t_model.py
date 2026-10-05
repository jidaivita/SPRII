"""RH20T model family with matched latent, force, and TCP heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .losses import SIGReg, canonical_vicreg
from .model import MLP


HORIZONS = (1, 4, 16)
CONDITIONS = {
    "B0",
    "Bx-SameEp",
    "Bx-Indep",
    "B3-SameEp",
    "B3-Indep",
    "B3-Random",
    "B3-LowDim-Same",
    "B3-LowDim-Random",
}


@dataclass(frozen=True)
class RH20TModelConfig:
    condition: str
    history_length: int = 24
    observation_dim: int = 128
    transient_dim: int = 64
    persistent_dim: int = 64
    transformer_width: int = 192
    transformer_depth: int = 4
    transformer_heads: int = 8
    dropout: float = 0.1
    horizon_dim: int = 32
    lowdim_dim: int = 14
    action_dim: int = 1

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown RH20T condition {self.condition}")

    @property
    def lowdim_only(self) -> bool:
        return "LowDim" in self.condition

    @property
    def split(self) -> bool:
        return self.condition != "B0"

    @property
    def persistence(self) -> bool:
        return self.condition.startswith("B3")

    @property
    def cross(self) -> bool:
        return self.condition.startswith(("Bx", "B3"))


class RHImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (2, 32, 64, 128, 256)
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            layers.extend(
                [nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1), nn.GELU()]
            )
        self.cnn = nn.Sequential(*layers)
        self.project = nn.Linear(256 * 6 * 6, 128)
        self.norm = nn.BatchNorm1d(128)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        shape = image.shape[:-3]
        feature = self.cnn(image.reshape(-1, *image.shape[-3:])).flatten(1)
        return self.norm(self.project(feature)).reshape(*shape, 128)


class RHObservationEncoder(nn.Module):
    def __init__(self, lowdim_only: bool, lowdim_dim: int = 14) -> None:
        super().__init__()
        self.lowdim_only = lowdim_only
        self.lowdim = MLP([lowdim_dim, 192, 192, 128])
        self.lowdim_norm = nn.BatchNorm1d(128)
        self.image = None if lowdim_only else RHImageEncoder()
        self.fusion = MLP([128 if lowdim_only else 256, 192, 128])
        self.fusion_norm = nn.BatchNorm1d(128)

    def forward(self, image: torch.Tensor | None, lowdim: torch.Tensor) -> torch.Tensor:
        shape = lowdim.shape[:-1]
        low = self.lowdim_norm(self.lowdim(lowdim.reshape(-1, lowdim.shape[-1])))
        if self.lowdim_only:
            feature = low
        else:
            if image is None:
                raise ValueError("RGB condition requires image tensors")
            visual = self.image(image).reshape(-1, 128)
            feature = torch.cat([visual, low], dim=-1)
        output = self.fusion_norm(self.fusion(feature))
        return output.reshape(*shape, 128)


class RHHistoryEncoder(nn.Module):
    def __init__(self, cfg: RH20TModelConfig, output_dim: int) -> None:
        super().__init__()
        self.action = MLP([cfg.action_dim, 32, 32])
        self.token = nn.Linear(cfg.observation_dim + 32, cfg.transformer_width)
        self.position = nn.Parameter(torch.zeros(1, cfg.history_length, cfg.transformer_width))
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
            raise ValueError("history length does not match model configuration")
        if history_actions.shape[1] != history_h.shape[1] - 1:
            raise ValueError("history actions must describe the 23 causal transitions")
        zeros = torch.zeros_like(history_actions[:, :1])
        previous_actions = torch.cat([zeros, history_actions], dim=1)
        token = self.token(torch.cat([history_h, self.action(previous_actions)], dim=-1))
        token = token + self.position
        length = token.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=token.device, dtype=torch.bool), diagonal=1
        )
        return self.output(self.transformer(token, mask=mask)[:, -1])


class RHSharedPredictor(nn.Module):
    """One donor-conditioned trunk feeding identical latent/FT/TCP heads."""

    def __init__(self, cfg: RH20TModelConfig) -> None:
        super().__init__()
        self.horizon = nn.Embedding(len(HORIZONS), cfg.horizon_dim)
        input_dim = 128 + 16 * cfg.action_dim + 16 + cfg.horizon_dim
        self.trunk = MLP([input_dim, 256, 256])
        self.latent = nn.Linear(256, cfg.observation_dim)
        self.force = nn.Linear(256, 6)
        self.tcp_xyz = nn.Linear(256, 3)

    def forward(
        self,
        context: torch.Tensor,
        padded_actions: torch.Tensor,
        action_mask: torch.Tensor,
        horizon_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [context, padded_actions.flatten(1), action_mask, self.horizon(horizon_index)], dim=-1
        )
        q_h = self.trunk(features)
        return {
            "q_h": q_h,
            "latent": self.latent(q_h),
            "force": self.force(q_h),
            "tcp_xyz": self.tcp_xyz(q_h),
        }


class RH20TJEPA(nn.Module):
    def __init__(self, cfg: RH20TModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation = RHObservationEncoder(cfg.lowdim_only, cfg.lowdim_dim)
        if cfg.split:
            self.context = None
            self.transient = MLP([128, 128, cfg.transient_dim])
            self.persistent = RHHistoryEncoder(cfg, cfg.persistent_dim)
        else:
            self.context = RHHistoryEncoder(cfg, 128)
            self.transient = None
            self.persistent = None
        self.predictor = RHSharedPredictor(cfg)

    def encode(
        self,
        history_image: torch.Tensor | None,
        history_lowdim: torch.Tensor,
        target_image: torch.Tensor | None,
        target_lowdim: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_h = self.observation(history_image, history_lowdim)
        target_h = self.observation(target_image, target_lowdim)
        return history_h, target_h

    def codes(
        self, history_h: torch.Tensor, history_actions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if not self.cfg.split:
            context = self.context(history_h, history_actions)
            return None, None, context
        z_s = self.transient(history_h[:, -1])
        z_p = self.persistent(history_h, history_actions)
        return z_s, z_p, torch.cat([z_s, z_p], dim=-1)


def _prediction_loss(
    model: RH20TJEPA,
    context: torch.Tensor,
    target_h: torch.Tensor,
    target_force: torch.Tensor,
    target_tcp_xyz: torch.Tensor,
    future_actions: torch.Tensor,
    action_masks: torch.Tensor,
    lambda_force: float,
    lambda_tcp: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent_losses, force_losses, tcp_losses = [], [], []
    for hi, horizon in enumerate(HORIZONS):
        horizon_index = torch.full(
            (context.shape[0],), hi, dtype=torch.long, device=context.device
        )
        output = model.predictor(
            context, future_actions[:, hi], action_masks[:, hi], horizon_index
        )
        latent_losses.append(F.mse_loss(output["latent"], target_h[:, hi]))
        force_losses.append(F.mse_loss(output["force"], target_force[:, hi]))
        tcp_losses.append(F.mse_loss(output["tcp_xyz"], target_tcp_xyz[:, hi]))
    latent = torch.stack(latent_losses).mean()
    force = torch.stack(force_losses).mean()
    tcp = torch.stack(tcp_losses).mean()
    total = latent + lambda_force * force + lambda_tcp * tcp
    metrics = {"latent": latent, "force": force, "tcp": tcp}
    for horizon, value in zip(HORIZONS, force_losses, strict=True):
        metrics[f"force_h{horizon}"] = value
    for horizon, value in zip(HORIZONS, tcp_losses, strict=True):
        metrics[f"tcp_h{horizon}"] = value
    return total, metrics


def rh20t_objective(
    model: RH20TJEPA,
    batch: object,
    sigreg: SIGReg,
    *,
    sigreg_weight: float = 0.02,
    lambda_p: float = 1.0,
    lambda_x: float = 0.1,
    lambda_force: float = 1.0,
    lambda_tcp: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_h, target_h = model.encode(
        batch.history_image, batch.history_lowdim, batch.target_image, batch.target_lowdim
    )
    z_s, z_p, context = model.codes(history_h, batch.history_actions)
    self_loss, self_parts = _prediction_loss(
        model,
        context,
        target_h,
        batch.target_force,
        batch.target_tcp_xyz,
        batch.future_actions,
        batch.action_masks,
        lambda_force,
        lambda_tcp,
    )
    sigreg_loss = sigreg(torch.cat([history_h, target_h], dim=1).transpose(0, 1))
    total = self_loss + sigreg_weight * sigreg_loss
    metrics: dict[str, torch.Tensor] = {
        "loss_self": self_loss.detach(),
        "loss_sigreg": sigreg_loss.detach(),
        **{f"self_{key}": value.detach() for key, value in self_parts.items()},
    }
    if model.cfg.persistence:
        donor_h = model.observation(batch.donor_history_image, batch.donor_history_lowdim)
        donor_z_p = model.persistent(donor_h, batch.donor_history_actions)
        persist_loss, persist_metrics = canonical_vicreg(donor_z_p, z_p)
        total = total + lambda_p * persist_loss
        metrics.update(persist_metrics)
        metrics["loss_persist"] = persist_loss.detach()
    if model.cfg.cross:
        if not model.cfg.persistence:
            donor_h = model.observation(batch.donor_history_image, batch.donor_history_lowdim)
            donor_z_p = model.persistent(donor_h, batch.donor_history_actions)
        cross_context = torch.cat([z_s, donor_z_p], dim=-1)
        cross_loss, cross_parts = _prediction_loss(
            model,
            cross_context,
            target_h,
            batch.target_force,
            batch.target_tcp_xyz,
            batch.future_actions,
            batch.action_masks,
            lambda_force,
            lambda_tcp,
        )
        total = total + lambda_x * cross_loss
        metrics["loss_cross"] = cross_loss.detach()
        metrics.update({f"cross_{key}": value.detach() for key, value in cross_parts.items()})
    metrics["loss_total"] = total.detach()
    return total, metrics
