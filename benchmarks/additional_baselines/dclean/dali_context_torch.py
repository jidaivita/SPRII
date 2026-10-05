"""PyTorch port of released DALI CtxEncoder transformer + forward component.

Source: frankroeder/DALI commit 34374fbea258748b03e31977a68693ab040fab72,
dreamerv3_compat/dreamerv3/nets.py. Does not implement the Dreamer world model,
RSSM, actor, critic, or cross-modal objective. No privileged parameter inputs.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    """Released Linear: fan-average variance, optional layer norm then SiLU."""
    def __init__(self, ni, no, act=False, norm=False, winit="uniform"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(no, ni))
        self.bias = None if norm else nn.Parameter(torch.zeros(no))
        self.norm = nn.LayerNorm(no, eps=1e-3) if norm else nn.Identity()
        self.act = act
        std = math.sqrt(1.0 / ((ni + no) / 2.0))
        if winit == "normal":
            std /= .87962566103423978
            nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)
        else:
            assert winit == "uniform"
            nn.init.uniform_(self.weight, -math.sqrt(3) * std, math.sqrt(3) * std)

    def forward(self, x):
        x = self.norm(F.linear(x, self.weight, self.bias))
        return F.silu(x) if self.act else x


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        # Official size is per head; released heads=1, size=256.
        self.width, self.heads = width, heads
        for name in ("query", "key", "value"):
            setattr(self, name, Linear(width, heads * width, winit="normal"))
        self.out = Linear(heads * width, heads * width)

    def forward(self, x):
        shape = (*x.shape[:-1], self.heads, self.width)
        q, k, v = [getattr(self, n)(x).reshape(shape) for n in ("query", "key", "value")]
        weights = (torch.einsum("...thd,...Thd->...htT", q, k) / math.sqrt(self.width)).softmax(-1)
        out = torch.einsum("...htT,...Thd->...thd", weights, v)
        return self.out(out.reshape(*out.shape[:-2], -1))


class DALIContext(nn.Module):
    def __init__(self, obs_dim, action_dim, seq_len, context_dim=8, width=256, heads=1, fw_width=128):
        super().__init__()
        assert heads == 1, "Released transformer residual requires heads=1; do not silently change it."
        self.obs_dim, self.action_dim, self.seq_len, self.context_dim = obs_dim, action_dim, seq_len, context_dim
        self.proj = Linear(obs_dim + action_dim, width, act=True, norm=True, winit="normal")
        self.norm1 = nn.LayerNorm(width, eps=1e-3)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-3)
        self.ff1 = Linear(width, width, act=True, norm=True, winit="normal")
        self.ff2 = Linear(width, width, act=True, norm=True, winit="normal")
        # outnorm=False is unused by official Linear; ctx_out still uses norm+SiLU.
        self.ctx_out = Linear(seq_len * width, context_dim, act=True, norm=True, winit="normal")
        self.forward_h0 = Linear(obs_dim + action_dim + context_dim, fw_width, act=True)
        self.forward_h1 = Linear(fw_width, fw_width, act=True)
        self.forward_out = Linear(fw_width, obs_dim)

    def encode(self, obs, action):
        assert obs.shape[:-1] == action.shape[:-1] and obs.shape[-2:] == (self.seq_len, self.obs_dim)
        assert action.shape[-1] == self.action_dim
        x = self.proj(torch.cat([obs, action], -1))
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(self.ff1(self.norm2(x)))
        return self.ctx_out(x.flatten(1))

    def forward(self, obs, action):
        return self.encode(obs, action)[:, None].expand(-1, self.seq_len, -1)

    def forward_model(self, prev_obs, action, z):
        x = torch.cat([prev_obs, action, z], -1)
        return self.forward_out(self.forward_h1(self.forward_h0(x)))

    def compute_loss(self, obs, action, context=None):
        if context is None:
            context = self(obs, action)
        pred = self.forward_model(obs[:, :-1], action[:, :-1], context[:, :-1])
        return (pred - obs[:, 1:]).square().mean(-1).mean(-1)

    @staticmethod
    def padded_prefix(x, lengths):
        """Exactly np.pad(x[:t], [(T-t,0),(0,0)], mode='edge') per row."""
        t = x.shape[1]
        assert lengths.shape == (len(x),) and bool(((lengths >= 1) & (lengths < t)).all())
        index = (torch.arange(t, device=x.device)[None] - (t - lengths[:, None])).clamp_min(0)
        return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))

    def prefix_loss(self, obs, action, prefix_lengths):
        x, u = self.padded_prefix(obs, prefix_lengths), self.padded_prefix(action, prefix_lengths)
        # Official WorldModel averages T-1 prefix terms plus one zero column over T.
        return self.compute_loss(x, u).mean() * (self.seq_len - 1) / self.seq_len

