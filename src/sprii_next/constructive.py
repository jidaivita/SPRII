"""Mechanism-guided frozen-base residual readers.

This module is deliberately separate from the historical ``Reader``.  The
historical Null/Persistent/Decode/Oracle recipes are kept byte-for-byte
compatible; constructive experiments use the classes below and record their
own architecture/configuration hash.
"""

from __future__ import annotations

import torch
from torch import nn


class RecipientBase(nn.Module):
    """Recipient-only predictor used as the frozen B0 base."""

    def __init__(self, seed: int, output_dim: int = 8):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context_adapter = nn.Linear(64, 64)
            self.horizon = nn.Embedding(5, 16)
            self.body = nn.Sequential(
                nn.Linear(256, 256), nn.GELU(),
                nn.Linear(256, 256), nn.GELU(),
            )
            self.head = nn.Linear(256, output_dim)

    def features(self, query, actions, mask, horizon_index):
        if query.ndim != 2 or query.shape[1] != 128:
            raise ValueError("query must have shape [N,128]")
        if actions.ndim != 3 or actions.shape[0] != len(query):
            raise ValueError("actions shape mismatch")
        if mask.shape[0] != len(query) or horizon_index.shape[0] != len(query):
            raise ValueError("conditioning batch mismatch")
        zero_context = torch.zeros((len(query), 64), dtype=query.dtype, device=query.device)
        x = torch.cat((query, self.context_adapter(zero_context),
                       actions.flatten(1), mask, self.horizon(horizon_index)), dim=1)
        return self.body(x)

    def forward(self, query, actions, mask, horizon_index):
        return self.head(self.features(query, actions, mask, horizon_index))


class ResidualReader(nn.Module):
    """Frozen B0 plus a matched generic/persistent residual route.

    M1 and M2 share the same route parameterization.  M1 supplies a learned
    query-only gate; M2 replaces that gate with a persistent-conditioned gate
    whose difference is exactly zero when ``z_p`` is zero.  Thus M2-zero is
    numerically equal to B0 up to floating point evaluation order.
    """

    ARMS = ("m1", "m1_phys", "m2", "m2_phys", "oracle_route")

    def __init__(self, arm: str, seed: int, *, base_state=None,
                 output_dim: int = 8, persistent_dim: int = 64,
                 hidden: int = 128, w=None):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown constructive arm: {arm}")
        if persistent_dim < 1 or hidden < 1:
            raise ValueError("invalid route dimensions")
        self.arm = arm
        self.output_dim = int(output_dim)
        self.persistent_dim = int(persistent_dim)
        self.hidden = int(hidden)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base = RecipientBase(seed, output_dim)
            # Same trainable route for M1 and M2.  The persistent projection is
            # present in both so parameter counts are directly comparable.
            self.value = nn.Linear(256, hidden, bias=False)
            self.query_gate = nn.Linear(256, hidden)
            self.query_proxy = nn.Linear(256, hidden, bias=False)
            self.persistent_gate = nn.Linear(persistent_dim, hidden, bias=False)
            self.out = nn.Linear(hidden, output_dim, bias=False)
            # Oracle reference receives the simulator parameters directly.  It
            # is kept separate from the persistent path so the reference is
            # explicit rather than silently reusing a donor representation.
            self.oracle_projection = nn.Linear(3, persistent_dim, bias=False)
        if base_state is not None:
            self.base.load_state_dict(base_state, strict=True)
        for p in self.base.parameters():
            p.requires_grad_(False)
        if w is None:
            w = torch.ones(output_dim, dtype=torch.float32)
        w = torch.as_tensor(w, dtype=torch.float32)
        if w.ndim != 1 or len(w) != output_dim or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("physics weights must be finite nonnegative [output_dim]")
        self.register_buffer("physics_weight", w.clone(), persistent=True)

    def route(self, h, persistent):
        value = torch.nn.functional.gelu(self.value(h))
        query_base = self.query_gate(h) + self.query_proxy(h)
        base_gate = torch.tanh(query_base)
        if self.arm.startswith("m1"):
            # M1 is the same route family with the persistent input removed.
            # The persistent projection remains registered for parameter-count
            # parity but receives no input/gradient in M1.
            delta = value * base_gate
        else:
            if persistent is None:
                raise ValueError("M2 requires persistent input")
            if persistent.shape != (len(h), self.persistent_dim):
                raise ValueError("persistent shape mismatch")
            conditioned_gate = torch.tanh(query_base + self.persistent_gate(persistent))
            # Exact zero at p=0; no bias in persistent projection.
            delta = value * (conditioned_gate - base_gate)
        delta = self.out(delta)
        if self.arm.endswith("_phys"):
            delta = delta * self.physics_weight.to(delta.dtype)
        return delta

    def forward(self, query, persistent, actions, mask, horizon_index, theta=None):
        h = self.base.features(query, actions, mask, horizon_index)
        base = self.base.head(h)
        if self.arm == "oracle_route":
            if theta is None:
                raise ValueError("oracle_route requires physical parameters")
            route_context = self.oracle_projection(theta)
        elif self.arm.startswith("m1"):
            route_context = None
        else:
            route_context = persistent
        if self.arm.startswith("m1"):
            delta = self.route(h, torch.zeros((len(h), self.persistent_dim), dtype=h.dtype, device=h.device))
        else:
            delta = self.route(h, route_context)
        return base + delta

    def zero_parity(self, query, actions, mask, horizon_index, *, atol=2e-6, rtol=2e-6):
        if not self.arm.startswith("m2"):
            raise ValueError("zero parity is defined for M2 arms")
        h = self.base.features(query, actions, mask, horizon_index)
        base = self.base.head(h)
        zero = torch.zeros((len(h), self.persistent_dim), dtype=h.dtype, device=h.device)
        pred = base + self.route(h, zero)
        if not torch.allclose(pred, base, atol=atol, rtol=rtol):
            raise AssertionError("M2-zero must equal frozen B0")
        return float((pred - base).abs().max().detach().cpu())

    def architecture(self):
        return dict(profile="constructive_frozen_b0_residual_v1", arm=self.arm,
                    output_dim=self.output_dim, persistent_dim=self.persistent_dim,
                    hidden=self.hidden,
                    parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    base_parameters=sum(p.numel() for p in self.base.parameters()),
                    physics_weight=self.physics_weight.detach().cpu().tolist())
