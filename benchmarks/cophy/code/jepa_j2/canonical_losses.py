"""Pinned SIGReg and canonical branch-wise VICReg losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """LeWorldModel-style Epps-Pulley SIGReg for input (T, B, D)."""

    def __init__(self, knots: int = 17, num_directions: int = 1024) -> None:
        super().__init__()
        self.num_directions = num_directions
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"SIGReg requires (T,B,D), got {tuple(embeddings.shape)}")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            z = embeddings.float()
            directions = torch.randn(
                z.shape[-1], self.num_directions, device=z.device, dtype=torch.float32
            )
            directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
            projected_t = (z @ directions).unsqueeze(-1) * self.t
            error = (
                (projected_t.cos().mean(dim=-3) - self.phi).square()
                + projected_t.sin().mean(dim=-3).square()
            )
            statistic = (error @ self.weights) * z.shape[-2]
            return statistic.mean()


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("covariance matrix must be square")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def effective_rank(z: torch.Tensor) -> torch.Tensor:
    singular = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    probabilities = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


def canonical_vicreg(
    z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-4
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if z_a.shape != z_b.shape or z_a.ndim != 2:
        raise ValueError("VICReg branches must have equal (N,D) shape")
    if z_a.shape[0] < 2:
        raise ValueError("VICReg requires at least two samples per branch")
    with torch.autocast(device_type=z_a.device.type, enabled=False):
        a, b = z_a.float(), z_b.float()
        inv = F.mse_loss(a, b)
        std_a = torch.sqrt(a.var(dim=0, correction=1) + eps)
        std_b = torch.sqrt(b.var(dim=0, correction=1) + eps)
        var_a = F.relu(1.0 - std_a).mean()
        var_b = F.relu(1.0 - std_b).mean()
        var = 0.5 * (var_a + var_b)
        a_centered = a - a.mean(dim=0, keepdim=True)
        b_centered = b - b.mean(dim=0, keepdim=True)
        cov_a = a_centered.T @ a_centered / (a.shape[0] - 1)
        cov_b = b_centered.T @ b_centered / (b.shape[0] - 1)
        cov_a_loss = _off_diagonal(cov_a).square().sum() / a.shape[1]
        cov_b_loss = _off_diagonal(cov_b).square().sum() / b.shape[1]
        cov = cov_a_loss + cov_b_loss
        loss = 25.0 * inv + 25.0 * var + cov
        metrics = {
            "persist_inv": inv.detach(),
            "persist_var": var.detach(),
            "persist_cov": cov.detach(),
            "z_p_std_min": torch.minimum(std_a.min(), std_b.min()).detach(),
            "z_p_std_mean": (0.5 * (std_a.mean() + std_b.mean())).detach(),
            "z_p_effective_rank": (0.5 * (effective_rank(a) + effective_rank(b))).detach(),
        }
        return loss, metrics

