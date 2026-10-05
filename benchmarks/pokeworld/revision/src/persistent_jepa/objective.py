"""Variant objectives with explicit self/cross sample counts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import SIGReg, canonical_vicreg
from .model import PersistentJEPA
from .torch_data import HORIZONS, PairedBatch


def compute_objective(
    model: PersistentJEPA,
    batch: PairedBatch,
    sigreg: SIGReg,
    sigreg_weight: float,
    lambda_p: float,
    lambda_x: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_h, target_h = model.encode_observations(batch.history_states, batch.target_states)
    z_s, z_p, context = model.codes(history_h, batch.history_actions)
    predictions, horizon_losses = [], []
    for hi, _h in enumerate(HORIZONS):
        horizon_index = torch.full(
            (context.shape[0],), hi, device=context.device, dtype=torch.long
        )
        pred = model.predictor(
            context, batch.future_actions[:, hi], batch.action_masks[:, hi], horizon_index
        )
        predictions.append(pred)
        horizon_losses.append(F.mse_loss(pred, target_h[:, hi]))
    self_loss = torch.stack(horizon_losses).mean()
    sigreg_input = torch.cat([history_h, target_h], dim=1).transpose(0, 1)
    sigreg_loss = sigreg(sigreg_input)
    total = self_loss + sigreg_weight * sigreg_loss
    metrics = {
        "loss_self": self_loss.detach(),
        "loss_sigreg": sigreg_loss.detach(),
        **{f"loss_h{h}": value.detach() for h, value in zip(HORIZONS, horizon_losses, strict=True)},
    }

    half = context.shape[0] // 2
    if model.cfg.variant in {"B1", "B2", "B3"}:
        persist, persist_metrics = canonical_vicreg(z_p[:half], z_p[half:])
        total = total + lambda_p * persist
        metrics.update(persist_metrics)
        metrics["loss_persist"] = persist.detach()

    if model.cfg.variant == "B3":
        cross_context = torch.cat([z_s[half:], z_p[:half]], dim=-1)
        cross_losses = []
        for hi, _h in enumerate(HORIZONS):
            horizon_index = torch.full((half,), hi, device=context.device, dtype=torch.long)
            cross_pred = model.predictor(
                cross_context,
                batch.future_actions[half:, hi],
                batch.action_masks[half:, hi],
                horizon_index,
            )
            cross_losses.append(F.mse_loss(cross_pred, target_h[half:, hi]))
        cross_loss = torch.stack(cross_losses).mean()
        total = total + lambda_x * cross_loss
        metrics["loss_cross"] = cross_loss.detach()

    if model.cfg.variant == "Sup":
        gamma_pred = model.supervised_gamma(z_p).squeeze(-1)
        supervised = F.mse_loss(gamma_pred, batch.gamma)
        total = total + 10.0 * supervised
        metrics["loss_supervised"] = supervised.detach()

    metrics["loss_total"] = total.detach()
    return total, metrics
