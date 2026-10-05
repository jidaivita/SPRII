"""Pixel R0 JEPA with published CNN/Transformer dimensions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .losses import SIGReg, canonical_vicreg
from .model import HistoryEncoder, ModelConfig, Predictor
from .poke_torch import PokeBatch
from .torch_data import HORIZONS


class DiskRenderer(nn.Module):
    def __init__(self, resolution: int = 64) -> None:
        super().__init__()
        coordinate = torch.linspace(-1.0, 1.0, resolution)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], dim=-1))
        self.resolution = resolution

    def frame(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape[:-1]
        flat = state.reshape(-1, 8)
        grid = self.grid[None]
        finger_distance = torch.linalg.vector_norm(grid - flat[:, None, None, 0:2], dim=-1)
        object_distance = torch.linalg.vector_norm(grid - flat[:, None, None, 4:6], dim=-1)
        edge_scale = self.resolution * 2.0
        finger = 0.65 * torch.sigmoid((0.06 - finger_distance) * edge_scale)
        obj = torch.sigmoid((0.09 - object_distance) * edge_scale)
        return torch.maximum(finger, obj).reshape(*shape, self.resolution, self.resolution)

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        frame = self.frame(current)
        difference = frame - self.frame(previous)
        return torch.stack([frame, difference], dim=-3)


class PixelObservationEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (2, 32, 64, 128, 256)
        layers = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            layers.extend(
                [nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1), nn.GELU()]
            )
        self.cnn = nn.Sequential(*layers)
        self.project = nn.Linear(256 * 4 * 4, 128)
        self.norm = nn.BatchNorm1d(128)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        shape = image.shape[:-3]
        feature = self.cnn(image.reshape(-1, *image.shape[-3:])).flatten(1)
        return self.norm(self.project(feature)).reshape(*shape, 128)


class PokeJEPA(nn.Module):
    def __init__(self, variant: str = "B0", history_length: int = 24) -> None:
        super().__init__()
        if variant not in {"B0", "B0_split", "B2", "B3", "Bx", "Sup"}:
            raise ValueError(f"unknown PokeWorld variant {variant}")
        cfg = ModelConfig(variant=variant, history_length=history_length)
        self.variant = variant
        self.cfg = cfg
        self.renderer = DiskRenderer()
        self.observation = PixelObservationEncoder()
        if variant == "B0":
            self.context = HistoryEncoder(cfg, output_dim=128)
            self.transient = None
            self.persistent = None
        else:
            self.context = None
            self.transient = nn.Sequential(
                nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64)
            )
            self.persistent = HistoryEncoder(cfg, output_dim=64)
        self.predictor = Predictor(cfg, context_dim=128)
        self.supervised_gamma = (
            nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            if variant == "Sup"
            else None
        )

    def encode_batch(self, batch: PokeBatch) -> tuple[torch.Tensor, torch.Tensor]:
        all_current = torch.cat([batch.history_current, batch.target_current], dim=1)
        all_previous = torch.cat([batch.history_previous, batch.target_previous], dim=1)
        image = self.renderer(all_current, all_previous)
        embeddings = self.observation(image)
        history_length = self.cfg.history_length
        return embeddings[:, :history_length], embeddings[:, history_length:]

    def codes(
        self, history_h: torch.Tensor, history_actions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.variant == "B0":
            context = self.context(history_h, history_actions)
            return None, None, context
        z_s = self.transient(history_h[:, -1])
        z_p = self.persistent(history_h, history_actions)
        return z_s, z_p, torch.cat([z_s, z_p], dim=-1)


def poke_objective(
    model: PokeJEPA,
    batch: PokeBatch,
    sigreg: SIGReg,
    sigreg_weight: float = 0.02,
    lambda_p: float = 0.0,
    lambda_x: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_h, target_h = model.encode_batch(batch)
    z_s, z_p, context = model.codes(history_h, batch.history_actions)
    horizon_losses = []
    for hi, _horizon in enumerate(HORIZONS):
        horizon_index = torch.full((context.shape[0],), hi, device=context.device, dtype=torch.long)
        prediction = model.predictor(
            context, batch.future_actions[:, hi], batch.action_masks[:, hi], horizon_index
        )
        horizon_losses.append(F.mse_loss(prediction, target_h[:, hi]))
    self_loss = torch.stack(horizon_losses).mean()
    sigreg_loss = sigreg(torch.cat([history_h, target_h], dim=1).transpose(0, 1))
    total = self_loss + sigreg_weight * sigreg_loss
    metrics = {
        "loss_total": total.detach(),
        "loss_self": self_loss.detach(),
        "loss_sigreg": sigreg_loss.detach(),
        **{
            f"loss_h{horizon}": loss.detach()
            for horizon, loss in zip(HORIZONS, horizon_losses, strict=True)
        },
    }
    half = context.shape[0] // 2
    if model.variant in {"B2", "B3"}:
        persist_loss, persist_metrics = canonical_vicreg(z_p[:half], z_p[half:])
        total = total + lambda_p * persist_loss
        metrics.update(persist_metrics)
        metrics["loss_persist"] = persist_loss.detach()
    if model.variant in {"Bx", "B3"}:
        cross_context = torch.cat([z_s[half:], z_p[:half]], dim=-1)
        cross_losses = []
        for hi, _horizon in enumerate(HORIZONS):
            horizon_index = torch.full((half,), hi, device=context.device, dtype=torch.long)
            prediction = model.predictor(
                cross_context,
                batch.future_actions[half:, hi],
                batch.action_masks[half:, hi],
                horizon_index,
            )
            cross_losses.append(F.mse_loss(prediction, target_h[half:, hi]))
        cross_loss = torch.stack(cross_losses).mean()
        total = total + lambda_x * cross_loss
        metrics["loss_cross"] = cross_loss.detach()
    if model.variant == "Sup":
        prediction = model.supervised_gamma(z_p).squeeze(-1)
        supervised = F.mse_loss(prediction, batch.gamma)
        total = total + 10.0 * supervised
        metrics["loss_supervised"] = supervised.detach()
    metrics["loss_total"] = total.detach()
    return total, metrics
