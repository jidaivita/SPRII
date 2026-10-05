from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngs.utils import count_params, create_burgers_dataloaders, set_seed
from triple_data import create_burgers_triple_dataloaders
from canonical_losses import canonical_vicreg
import hashlib

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

PREDICTION_HORIZON = 101
PRED_X = 401
PRED_T = 101
TRAIN_QUERY_POINTS = 8192
MIN_MODEL_PARAMS = 5_000_000
ARCH_VERSION = "plain_cnn2d_cond__deeponet_pred_pe__q8192_no_deriv_v2_trimmed"
# 2026-04-29: NOD-PE = NOD ablation with sinusoidal positional encoding on the
# DeepONet trunk input (x, t). Designed to test whether trunk-MLP spectral bias
# is the cause of NOD's poor short-horizon accuracy. Otherwise identical to
# nod.py: plain (no edge) encoder + DeepONet decoder, u-only MSE supervision
# on 8192 sampled queries, no derivative loss, no derivative cache.
COMPAT_ARCH_VERSIONS = (
    ARCH_VERSION,
    # State_dict for the PE variant differs from non-PE in the trunk's first
    # layer (input dim 2 vs 26), so old non-PE checkpoints are NOT loadable
    # here even though many weights overlap. Listed for clarity only.
)
MODEL_DEFAULTS = {
    "context_dim": 1,
    "cond_t": PRED_T,
    "cond_x": PRED_X,
    "pred_x": PRED_X,
    "deeponet_latent_dim": 512,
    "predictor_branch_hidden": 800,
    "predictor_branch_layers": 4,
    "predictor_trunk_hidden": 1100,
    "predictor_trunk_layers": 4,
    "conditioner_base_channels": 46,
    "conditioner_head_hidden": 256,
    "mlp_dropout": 0.0,
    # Sinusoidal (NeRF-style) positional encoding on the trunk input (x, t).
    # L_x=8 matches the spatial Nyquist of the 401-grid (period 2dx=0.01).
    # L_t=4 covers the dominant Burgers time scale; no temporal shocks.
    "pe_num_freqs_x": 8,
    "pe_num_freqs_t": 4,
    "pe_include_input": True,
}


class FourierFeatures(nn.Module):
    """Per-coordinate sinusoidal positional encoding.

    Input  coords: [..., D]
    Output:        [..., sum(1 + 2*L_i for i in 0..D-1)]  (include_input=True)
                   [..., sum(2*L_i for i in 0..D-1)]       (include_input=False)
    """

    def __init__(self, num_freqs_per_dim: tuple[int, ...], include_input: bool = True) -> None:
        super().__init__()
        self.num_freqs_per_dim = tuple(int(L) for L in num_freqs_per_dim)
        self.include_input = bool(include_input)
        for i, L in enumerate(self.num_freqs_per_dim):
            freqs = (2.0 ** torch.arange(L, dtype=torch.float32)) * math.pi
            self.register_buffer(f"freqs_{i}", freqs)

    @property
    def out_dim(self) -> int:
        d = 0
        for L in self.num_freqs_per_dim:
            d += (1 if self.include_input else 0) + 2 * L
        return d

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        D = coords.shape[-1]
        if D != len(self.num_freqs_per_dim):
            raise ValueError(
                f"coords last dim {D} != num_freqs_per_dim length {len(self.num_freqs_per_dim)}"
            )
        outs = []
        for i in range(D):
            xi = coords[..., i:i + 1]
            f = getattr(self, f"freqs_{i}")
            xf = xi * f
            if self.include_input:
                outs.append(xi)
            outs.append(torch.sin(xf))
            outs.append(torch.cos(xf))
        return torch.cat(outs, dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Softplus(beta=1e1))
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus(beta=1e1))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONetBackbone(nn.Module):
    """
    Generic DeepONet backbone:
      y(branch, trunk) = <phi_b(branch), phi_t(trunk)> + b
    """

    def __init__(
        self,
        branch_in_dim: int,
        trunk_in_dim: int,
        latent_dim: int,
        branch_hidden: int,
        branch_layers: int,
        trunk_hidden: int,
        trunk_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.branch_net = MLP(
            in_dim=int(branch_in_dim),
            out_dim=self.latent_dim,
            hidden_dim=int(branch_hidden),
            num_layers=int(branch_layers),
            dropout=float(dropout),
        )
        self.trunk_net = MLP(
            in_dim=int(trunk_in_dim),
            out_dim=self.latent_dim,
            hidden_dim=int(trunk_hidden),
            num_layers=int(trunk_layers),
            dropout=float(dropout),
        )
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        if branch_in.ndim != 2:
            raise ValueError(f"branch_in must be [B,D], got {tuple(branch_in.shape)}")

        branch_feat = self.branch_net(branch_in)  # [B,P]

        if trunk_in.ndim == 2:
            trunk_feat = self.trunk_net(trunk_in)  # [S,P]
            out = torch.einsum("bp,sp->bs", branch_feat, trunk_feat)
        elif trunk_in.ndim == 3:
            if trunk_in.shape[0] != branch_in.shape[0]:
                raise ValueError(
                    f"Batch mismatch: branch B={branch_in.shape[0]}, trunk B={trunk_in.shape[0]}"
                )
            trunk_feat = self.trunk_net(trunk_in)  # [B,S,P]
            out = torch.einsum("bp,bsp->bs", branch_feat, trunk_feat)
        else:
            raise ValueError(f"trunk_in must be [S,Dt] or [B,S,Dt], got {tuple(trunk_in.shape)}")

        return out + self.output_bias


def _pick_group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        groups = _pick_group_count(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EdgeAwareConditioner(nn.Module):
    """
    Plain conditioner (no-edge ablation):
      cond_u [B,1,101,401] -> context [B,1]

    Uses ONLY the raw conditioner field as input (no handcrafted derivative
    channels). Class name kept as 'EdgeAwareConditioner' so checkpoints from
    the legacy derivative-supervision variant load without state_dict-key
    rewrites — despite the name, this is the no-edge variant.
    """

    def __init__(
        self,
        context_dim: int = 1,
        cond_t: int = 101,
        cond_x: int = 401,
        base_channels: int = 64,
        head_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.cond_t = int(cond_t)
        self.cond_x = int(cond_x)

        c0 = int(base_channels)
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8

        # NO-EDGE: 1 input channel (just u, no derivatives).
        self.stem = EdgeConvBlock(1, c0, downsample=False)
        self.stage1 = EdgeConvBlock(c0, c1, downsample=True)
        self.stage2 = EdgeConvBlock(c1, c2, downsample=True)
        self.stage3 = EdgeConvBlock(c2, c3, downsample=True)
        self.stage4 = EdgeConvBlock(c3, c3, downsample=True)

        self.head = nn.Sequential(
            nn.Linear(c3 * 2, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), self.context_dim),
        )

    def forward(self, cond_u: torch.Tensor) -> torch.Tensor:
        if cond_u.ndim != 4:
            raise ValueError(f"cond_u must be [B,1,T,N], got {tuple(cond_u.shape)}")

        b, c, t, n = cond_u.shape
        if c != 1:
            raise ValueError(f"cond_u channel must be 1, got {c}")
        if t != self.cond_t or n != self.cond_x:
            raise ValueError(
                "Conditioner expects fixed [T,N]="
                f"[{self.cond_t},{self.cond_x}], got [{t},{n}]"
            )

        h = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(cond_u)))))
        gap = F.adaptive_avg_pool2d(h, output_size=1).view(b, -1)
        gmp = F.adaptive_max_pool2d(h, output_size=1).view(b, -1)
        return self.head(torch.cat([gap, gmp], dim=1))


class DeepONetPredictor(nn.Module):
    """
    Predictor:
      inputs: u0 [B,1,401], t [B,1], context [B,1]
      branch(u0, context), trunk(x,t) -> u(t,x) [B,1,401]
    """

    def __init__(
        self,
        context_dim: int = 1,
        pred_x: int = 401,
        latent_dim: int = 128,
        branch_hidden: int = 256,
        branch_layers: int = 4,
        trunk_hidden: int = 128,
        trunk_layers: int = 3,
        dropout: float = 0.0,
        pe_num_freqs_x: int = 8,
        pe_num_freqs_t: int = 4,
        pe_include_input: bool = True,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.pred_x = int(pred_x)

        self.pe = FourierFeatures(
            num_freqs_per_dim=(int(pe_num_freqs_x), int(pe_num_freqs_t)),
            include_input=bool(pe_include_input),
        )
        trunk_in_dim = self.pe.out_dim

        self.backbone = DeepONetBackbone(
            branch_in_dim=self.pred_x + self.context_dim,
            trunk_in_dim=trunk_in_dim,
            latent_dim=int(latent_dim),
            branch_hidden=int(branch_hidden),
            branch_layers=int(branch_layers),
            trunk_hidden=int(trunk_hidden),
            trunk_layers=int(trunk_layers),
            dropout=float(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def predict_queries(
        self,
        u0: torch.Tensor,
        query_coords: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if u0.ndim != 3:
            raise ValueError(f"u0 must be [B,1,N], got {tuple(u0.shape)}")
        if query_coords.ndim != 3:
            raise ValueError(f"query_coords must be [B,Q,2], got {tuple(query_coords.shape)}")
        if context.ndim != 2:
            raise ValueError(f"context must be [B,{self.context_dim}], got {tuple(context.shape)}")

        b, c, n = u0.shape
        if c != 1:
            raise ValueError(f"u0 channel must be 1, got {c}")
        if n != self.pred_x:
            raise ValueError(f"u0 spatial size must be {self.pred_x}, got {n}")
        if query_coords.shape[0] != b or query_coords.shape[2] != 2:
            raise ValueError(f"query_coords must be [B,Q,2], got {tuple(query_coords.shape)}")
        if context.shape != (b, self.context_dim):
            raise ValueError(
                f"context shape mismatch. Expected {(b, self.context_dim)}, got {tuple(context.shape)}"
            )

        branch_u0 = u0.squeeze(1)  # [B,N]
        branch_in = torch.cat([branch_u0, context], dim=1)  # [B,N+C]

        # Apply sinusoidal positional encoding to (x, t) before the trunk MLP.
        trunk_in = self.pe(query_coords)  # [B, Q, pe_out_dim]
        delta = self.backbone(branch_in=branch_in, trunk_in=trunk_in)  # [B,Q]

        x_norm = query_coords[..., 0]
        x_idx = torch.round((x_norm + 1.0) * 0.5 * float(n - 1)).long().clamp_(0, n - 1)
        u0_q = torch.gather(branch_u0, dim=1, index=x_idx)  # [B,Q]
        return u0_q + self.res_scale * delta

    def forward(
        self,
        u0: torch.Tensor,
        pred_t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if u0.ndim != 3:
            raise ValueError(f"u0 must be [B,1,N], got {tuple(u0.shape)}")
        if pred_t.ndim != 2:
            raise ValueError(f"pred_t must be [B,1], got {tuple(pred_t.shape)}")
        if context.ndim != 2:
            raise ValueError(f"context must be [B,{self.context_dim}], got {tuple(context.shape)}")

        b, c, n = u0.shape
        if c != 1:
            raise ValueError(f"u0 channel must be 1, got {c}")
        if n != self.pred_x:
            raise ValueError(f"u0 spatial size must be {self.pred_x}, got {n}")
        if pred_t.shape != (b, 1):
            raise ValueError(f"pred_t shape mismatch. Expected {(b, 1)}, got {tuple(pred_t.shape)}")
        if context.shape != (b, self.context_dim):
            raise ValueError(
                f"context shape mismatch. Expected {(b, self.context_dim)}, got {tuple(context.shape)}"
            )

        x_grid = torch.linspace(-1.0, 1.0, n, device=u0.device, dtype=u0.dtype)
        x_grid = x_grid.view(1, n, 1).expand(b, n, 1)
        t_grid = pred_t.view(b, 1, 1).expand(b, n, 1)
        query_coords = torch.cat([x_grid, t_grid], dim=2)  # [B,N,2]

        pred_q = self.predict_queries(u0=u0, query_coords=query_coords, context=context)  # [B,N]
        return pred_q.view(b, 1, n)


class NGS_INR(nn.Module):
    """
    NGS INR:
      Conditioner: cond_u [B,1,101,401] -> context [B,1]
      Predictor (DeepONet): (u0 [B,1,401], t [B,1], context [B,1]) -> u(t) [B,1,401]

    When context_dim=0 (the d=0 ablation), the conditioner is dropped entirely
    and `encode` returns an empty [B,0] tensor; the DeepONet branch then sees
    only u0.
    """

    def __init__(
        self,
        context_dim: int = 1,
        cond_t: int = 101,
        cond_x: int = 401,
        pred_x: int = 401,
        deeponet_latent_dim: int = 512,
        predictor_branch_hidden: int = 3584,
        predictor_branch_layers: int = 4,
        predictor_trunk_hidden: int = 2048,
        predictor_trunk_layers: int = 4,
        conditioner_base_channels: int = 64,
        conditioner_head_hidden: int = 512,
        mlp_dropout: float = 0.0,
        pe_num_freqs_x: int = 8,
        pe_num_freqs_t: int = 4,
        pe_include_input: bool = True,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        if self.context_dim > 0:
            self.conditioner = EdgeAwareConditioner(
                context_dim=self.context_dim,
                cond_t=int(cond_t),
                cond_x=int(cond_x),
                base_channels=int(conditioner_base_channels),
                head_hidden=int(conditioner_head_hidden),
            )
        else:
            self.conditioner = None
        self.predictioner = DeepONetPredictor(
            context_dim=self.context_dim,
            pred_x=int(pred_x),
            latent_dim=int(deeponet_latent_dim),
            branch_hidden=int(predictor_branch_hidden),
            branch_layers=int(predictor_branch_layers),
            trunk_hidden=int(predictor_trunk_hidden),
            trunk_layers=int(predictor_trunk_layers),
            dropout=float(mlp_dropout),
            pe_num_freqs_x=int(pe_num_freqs_x),
            pe_num_freqs_t=int(pe_num_freqs_t),
            pe_include_input=bool(pe_include_input),
        )
        self._init_weights_xavier()

    def _init_weights_xavier(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def encode(self, cond_u: torch.Tensor) -> torch.Tensor:
        if self.conditioner is None:
            return cond_u.new_zeros((cond_u.shape[0], 0))
        return self.conditioner(cond_u)

    def predict(self, pred_u0: torch.Tensor, pred_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.predictioner(pred_u0, pred_t, context)

    def predict_queries(self, pred_u0: torch.Tensor, query_coords: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.predictioner.predict_queries(pred_u0, query_coords, context)

    def forward(self, cond_u: torch.Tensor, pred_u0: torch.Tensor, pred_t: torch.Tensor):
        context = self.encode(cond_u)
        pred_ut = self.predict(pred_u0, pred_t, context)
        return pred_ut, context


# ============================================================================
# Training (no derivative supervision)
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Burgers NGS INR (no-edge encoder + DeepONet decoder). "
            "u-only MSE supervision on randomly sampled (t,x) queries; no "
            "derivative loss, no derivative cache."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/burgers/datagen_python/output_new/task1_viscous_main",
    )
    parser.add_argument("--include_output_add_train", action="store_true")
    parser.add_argument(
        "--output_add_root",
        type=str,
        default="data/burgers/datagen_python/output_add/task1_viscous_main",
    )
    parser.add_argument("--save_dir", type=str, default="runs/burgers")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="cond_init",
        choices=["none", "cond_init", "full"],
    )
    parser.add_argument(
        "--train_query_points",
        type=int,
        default=TRAIN_QUERY_POINTS,
        help="Number of sampled (t,x) queries per sample for train supervision.",
    )
    parser.add_argument(
        "--context_dim",
        type=int,
        default=MODEL_DEFAULTS["context_dim"],
        help="Code dimension d. Used for the d-sweep that supports the dimension-gauging argument.",
    )
    parser.add_argument(
        "--include_alignment_view", action="store_true",
        help="Load matched independent trajectory C for all methods; NOD leaves it unused.",
    )
    parser.add_argument("--lambda_align", type=float, default=0.0)
    parser.add_argument("--random_relation", action="store_true")
    return parser.parse_args()


def resolve_device(arg_device: str) -> torch.device:
    if arg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but CUDA is not available.")
    return torch.device(arg_device)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    tensor_keys: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        should_move = tensor_keys is None or key in tensor_keys
        if should_move and torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def sample_query_indices(
    batch_size: int,
    total_points: int,
    num_query_points: int,
    device: torch.device,
) -> torch.Tensor:
    if num_query_points <= 0:
        raise ValueError(f"num_query_points must be positive, got {num_query_points}")
    if num_query_points > total_points:
        raise ValueError(
            f"num_query_points={num_query_points} exceeds total_points={total_points}"
        )
    weights = torch.ones((batch_size, total_points), device=device)
    return torch.multinomial(weights, num_samples=num_query_points, replacement=False)


def indices_to_coords_and_targets(
    target_seq: torch.Tensor,
    t_idx: torch.Tensor,
    flat_query_idx: torch.Tensor,
    t_norm_denom: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """u-only variant: returns (query_coords [B,Q,2], u_targets [B,Q])."""
    if target_seq.ndim != 4:
        raise ValueError(f"target_seq must be [B,K,1,N], got {tuple(target_seq.shape)}")
    if t_idx.ndim != 2:
        raise ValueError(f"t_idx must be [B,K], got {tuple(t_idx.shape)}")
    if flat_query_idx.ndim != 2:
        raise ValueError(f"flat_query_idx must be [B,Q], got {tuple(flat_query_idx.shape)}")

    b, k, c, n = target_seq.shape
    if c != 1:
        raise ValueError(f"target_seq channel must be 1, got {c}")
    if t_idx.shape != (b, k):
        raise ValueError(f"t_idx shape mismatch. Expected {(b, k)}, got {tuple(t_idx.shape)}")
    if flat_query_idx.shape[0] != b:
        raise ValueError(
            f"flat_query_idx batch mismatch. Expected B={b}, got {flat_query_idx.shape[0]}"
        )

    t_pos = torch.div(flat_query_idx, n, rounding_mode="floor")  # [B,Q]
    x_pos = torch.remainder(flat_query_idx, n)                    # [B,Q]

    x_norm = -1.0 + 2.0 * x_pos.float() / float(max(1, n - 1))
    t_norm = torch.gather(t_idx, dim=1, index=t_pos).float() / float(t_norm_denom)
    query_coords = torch.stack([x_norm, t_norm], dim=-1)          # [B,Q,2]

    target_flat = target_seq.squeeze(2).reshape(b, k * n)         # [B,K*N]
    target_query = torch.gather(target_flat, dim=1, index=flat_query_idx)  # [B,Q]
    return query_coords, target_query


def run_epoch(
    model: NGS_INR,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    clip_grad: float,
    t_norm_denom: float,
    num_query_points: Optional[int] = None,
    split_name: str = "train",
    show_progress: bool = False,
    lambda_align: float = 0.0,
    random_relation: bool = False,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_mse = 0.0
    total_align = 0.0
    align_components = {k: 0.0 for k in ("persist_inv", "persist_var", "persist_cov")}
    pair_hash = hashlib.sha256()
    same_system = pair_count = 0
    donor_counts, recipient_counts = {}, {}
    grad_diagnostics = None
    batch_count = 0

    iterator = loader
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(loader, desc=split_name, leave=False)
        iterator = progress

    with torch.set_grad_enabled(is_train):
        for cpu_batch in iterator:
            batch = move_batch_to_device(
                cpu_batch,
                device=device,
                tensor_keys=("cond_u", "pred_u0", "target_seq", "t_idx"),
            )
            cond_u = batch["cond_u"]            # [B,1,101,401]
            pred_u0 = batch["pred_u0"]          # [B,1,401]
            target_seq = batch["target_seq"]    # [B,101,1,401]
            t_idx = batch["t_idx"]              # [B,101]

            if target_seq.ndim != 4:
                raise ValueError(f"Expected target_seq [B,K,1,N], got {tuple(target_seq.shape)}")
            if t_idx.ndim != 2:
                raise ValueError(f"Expected t_idx [B,K], got {tuple(t_idx.shape)}")

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            context = model.encode(cond_u)  # [B,1]
            b, k_total, _, n = target_seq.shape

            if num_query_points is None:
                # Full-horizon supervision over (K, N) grid (used for eval/test).
                pred_u0_flat = pred_u0.unsqueeze(1).expand(b, k_total, 1, n).reshape(b * k_total, 1, n)
                pred_t_flat = (t_idx.float() / float(t_norm_denom)).reshape(b * k_total, 1)
                context_flat = context.unsqueeze(1).expand(b, k_total, context.shape[-1]).reshape(
                    b * k_total, context.shape[-1]
                )
                pred_flat = model.predict(pred_u0_flat, pred_t_flat, context_flat)  # [B*K,1,N]
                pred_seq = pred_flat.reshape(b, k_total, 1, n)
                loss = criterion(pred_seq, target_seq)
                mse = F.mse_loss(pred_seq, target_seq)
            else:
                total_points = k_total * n
                flat_query_idx = sample_query_indices(
                    batch_size=b,
                    total_points=total_points,
                    num_query_points=int(num_query_points),
                    device=target_seq.device,
                )
                query_coords, target_query = indices_to_coords_and_targets(
                    target_seq=target_seq,
                    t_idx=t_idx,
                    flat_query_idx=flat_query_idx,
                    t_norm_denom=t_norm_denom,
                )  # [B,Q,2], [B,Q]
                pred_query = model.predict_queries(
                    pred_u0=pred_u0, query_coords=query_coords, context=context
                )
                loss = criterion(pred_query, target_query)
                mse = F.mse_loss(pred_query, target_query)

            if is_train and lambda_align != 0.0:
                z_c = model.encode(cpu_batch["align_u"].to(device))
                align, align_stats = canonical_vicreg(context, z_c)
                for name in align_components: align_components[name] += float(align_stats[name])
                src_ids = cpu_batch["recipient_system_idx"].tolist()
                dst_ids = cpu_batch["align_system_idx"].tolist()
                for a_id,b_id,c_id in zip(src_ids,dst_ids,cpu_batch["align_case_idx"].tolist()):
                    same_system += int(a_id == b_id); pair_count += 1
                    donor_counts[b_id] = donor_counts.get(b_id, 0) + 1
                    recipient_counts[a_id] = recipient_counts.get(a_id, 0) + 1
                    pair_hash.update(f"{a_id}:{b_id}:{c_id};".encode())
                if random_relation: assert same_system == 0
                if grad_diagnostics is None:
                    params = list(model.parameters())
                    base_grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    rel_grad = torch.autograd.grad(align, params, retain_graph=True, allow_unused=True)
                    norm = lambda grads: float(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)))
                    grad_diagnostics = {"base":norm(base_grad),"align_raw":norm(rel_grad),"align_weighted":float(lambda_align)*norm(rel_grad)}
                total_align += float(align.detach().item())
                loss = loss + float(lambda_align) * align
            if is_train:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_mse += float(mse.detach().item())
            batch_count += 1

            if progress is not None:
                progress.set_postfix(
                    loss=f"{loss.detach().item():.5f}",
                    avg_mse=f"{total_mse / batch_count:.5f}",
                )

    if progress is not None:
        progress.close()
    if batch_count == 0:
        raise RuntimeError("Dataloader returned zero batches.")

    return {
        "loss": total_loss / batch_count,
        "mse": total_mse / batch_count,
        "align_loss": total_align / batch_count,
        "align_weighted": float(lambda_align) * total_align / batch_count,
        "align_components": {k:v/batch_count for k,v in align_components.items()},
        "gradient_first_batch": grad_diagnostics,
        "pairing": {"same_system_count":same_system,"count":pair_count,"donor_counts":donor_counts,"recipient_counts":recipient_counts,"hash":pair_hash.hexdigest()},
        "num_batches": batch_count,
    }


@torch.no_grad()
def compute_code_spread_per_nu(
    model: NGS_INR,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 64,
) -> dict[float, dict[str, Any]]:
    """Encode samples in `loader`, group encoder codes by ν, return per-ν mean/std/count."""
    import numpy as np
    was_train = model.training
    model.eval()
    by_nu: dict[float, list[np.ndarray]] = {}
    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break
        cond_u = batch["cond_u"].to(device, non_blocking=True)
        nu = batch["nu_value"].squeeze(-1).cpu().numpy()
        ctx = model.encode(cond_u).detach().cpu().numpy()
        for i in range(ctx.shape[0]):
            v = round(float(nu[i]), 8)
            by_nu.setdefault(v, []).append(ctx[i])
    model.train(mode=was_train)
    out: dict[float, dict[str, Any]] = {}
    for v in sorted(by_nu):
        arr = np.stack(by_nu[v], axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        out[v] = {"mean": mean.tolist(), "std": std.tolist(), "n": int(arr.shape[0])}
    return out


def format_code_spread(spread: dict[float, dict[str, Any]]) -> str:
    lines = []
    for v, s in spread.items():
        m, sd, n = s["mean"], s["std"], s["n"]
        if len(m) == 1:
            lines.append(f"  nu={v:.4f}  mu={m[0]:+.4f}  sd={sd[0]:.4f}  (n={n})")
        else:
            m_str = "[" + ",".join(f"{x:+.3f}" for x in m) + "]"
            s_str = "[" + ",".join(f"{x:.3f}" for x in sd) + "]"
            lines.append(f"  nu={v:.4f}  mu={m_str}  sd={s_str}  (n={n})")
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    model: NGS_INR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.lambda_align != 0.0 and not args.include_alignment_view:
        raise ValueError("Alignment requires the matched A/B/C loader.")
    set_seed(args.seed)
    device = resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["clean_train_no_final_loader"] = True
    config["matched_alignment_view"] = bool(args.include_alignment_view)
    config["full_horizon_supervision"] = True
    config["derivative_supervision"] = False
    config["prediction_horizon"] = PREDICTION_HORIZON
    config["arch_version"] = ARCH_VERSION
    config["train_query_points"] = int(args.train_query_points)
    config["model_defaults"] = dict(MODEL_DEFAULTS)
    config.update({k: v for k, v in MODEL_DEFAULTS.items() if k not in config})
    config["min_model_params"] = MIN_MODEL_PARAMS

    loader_num_workers = args.num_workers
    if args.cache_mode == "full" and loader_num_workers > 0:
        print(
            "cache_mode=full duplicates full-shard caches per worker process; "
            "forcing num_workers=0."
        )
        loader_num_workers = 0

    if args.include_alignment_view:
        loaders = create_burgers_triple_dataloaders(
            data_root=args.data_root, output_add_root=args.output_add_root,
            include_output_add_train=args.include_output_add_train,
            prediction_horizon=PREDICTION_HORIZON, cache_mode=args.cache_mode,
            batch_size=args.batch_size, num_workers=loader_num_workers,
            pin_memory=device.type == "cuda", seed=args.seed, include_test=False,
        )
    else:
        loaders = create_burgers_dataloaders(
            data_root=args.data_root, include_output_add_train=args.include_output_add_train,
            output_add_root=args.output_add_root, prediction_horizon=PREDICTION_HORIZON,
            cache_mode=args.cache_mode, batch_size=args.batch_size,
            num_workers=loader_num_workers, pin_memory=device.type == "cuda",
            train_shuffle=True, drop_last_train=False, persistent_workers=None,
            prefetch_factor=4, seed=args.seed, include_test=False,
        )

    dataset_summaries = {
        s: ld.dataset.describe() if hasattr(ld.dataset, "describe") else {}
        for s, ld in loaders.items()
    }
    print("Dataset summaries:")
    print(json.dumps(dataset_summaries, indent=2))

    train_dataset = loaders["train"].dataset
    n_t = int(train_dataset.n_t)
    t_norm_denom = float(max(1, n_t - 1))
    config["t_norm_denom"] = t_norm_denom
    config["dataset_summaries"] = dataset_summaries

    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Override context_dim from CLI for the d-sweep.
    model_kwargs = dict(MODEL_DEFAULTS)
    model_kwargs["context_dim"] = int(args.context_dim)
    config["context_dim_override"] = int(args.context_dim)
    model = NGS_INR(**model_kwargs).to(device)

    param_size = count_params(model)
    # d=0 drops the conditioner (~5M params), so the predictor-only model is
    # roughly half the size; skip the size guard in that case.
    if int(args.context_dim) > 0 and param_size < MIN_MODEL_PARAMS:
        raise RuntimeError(
            f"Model too small: {param_size:,d} params, require at least {MIN_MODEL_PARAMS:,d}."
        )
    model_size_mb = float(param_size * 4.0 / (1024.0 ** 2))
    print(f"Model params: {param_size:,d} ({model_size_mb:.2f} MB @ fp32)")
    config["model_num_params"] = int(param_size)
    config["model_size_mb_fp32"] = model_size_mb
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_eval = float("inf")
    start_epoch = 1

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        ckpt_arch = ckpt_cfg.get("arch_version", "legacy")
        if ckpt_arch not in COMPAT_ARCH_VERSIONS:
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"expected one of {COMPAT_ARCH_VERSIONS}, got '{ckpt_arch}'."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt_arch == ARCH_VERSION and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ckpt_epoch = int(ckpt.get("epoch", 0))
        start_epoch = ckpt_epoch + 1 if ckpt_arch == ARCH_VERSION else 1
        metrics = ckpt.get("metrics", {})
        if isinstance(metrics, dict) and "eval" in metrics and "loss" in metrics["eval"]:
            best_eval = float(metrics["eval"]["loss"])
        history_path = save_dir / "metrics_history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []
        if ckpt_arch == ARCH_VERSION:
            print(f"Resumed from {resume_path}, next epoch={start_epoch}, best_eval={best_eval:.6f}")
        else:
            print(
                f"Loaded weights from compat checkpoint {resume_path} "
                f"(arch='{ckpt_arch}'); restarting from epoch=1 with fresh optimizer."
            )

    if start_epoch > args.epochs:
        print(f"Nothing to run: start_epoch={start_epoch} > epochs={args.epochs}.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        if args.include_alignment_view:
            loaders["train"].dataset.random_relation = bool(args.random_relation)
            loaders["train"].dataset.epoch = epoch
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            clip_grad=args.clip_grad,
            t_norm_denom=t_norm_denom,
            num_query_points=int(args.train_query_points),
            split_name=f"train epoch {epoch}/{args.epochs}",
            show_progress=True,
            lambda_align=args.lambda_align,
            random_relation=args.random_relation,
        )
        with torch.no_grad():
            eval_metrics = run_epoch(
                model=model,
                loader=loaders["eval"],
                criterion=criterion,
                device=device,
                optimizer=None,
                clip_grad=args.clip_grad,
                t_norm_denom=t_norm_denom,
                num_query_points=None,
                split_name="eval",
            )

        epoch_metrics = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(epoch_metrics)
        print(
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_metrics['loss']:.6f} train_mse={train_metrics['mse']:.6f} | "
            f"eval_loss={eval_metrics['loss']:.6f} eval_mse={eval_metrics['mse']:.6f}"
        )

        spread = compute_code_spread_per_nu(model, loaders["eval"], device)
        print("  per-nu code stats (eval split):")
        print(format_code_spread(spread))
        epoch_metrics["code_spread"] = spread

        save_checkpoint(save_dir / "latest.pth", model, optimizer, epoch, epoch_metrics, config)
        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            save_checkpoint(save_dir / "best_eval.pth", model, optimizer, epoch, epoch_metrics, config)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:04d}.pth", model, optimizer, epoch, epoch_metrics, config)

        with (save_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Training complete. Best eval loss: {best_eval:.6f}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
