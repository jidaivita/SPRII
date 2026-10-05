"""CoPhy v6 dynamics over frozen RGB-derived object features, never GT poses.

All relation arms share an architecture. The same online motion projection
encodes observations and gradient-bearing future targets, as in original A.
The official supervised visual frontend is external and frozen.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


VERSION = 'cophy-latent-v6.2-sig10'


@dataclass(frozen=True)
class ModelConfig:
    family: str = 'JEPA'
    feature_dim: int = 784
    width: int = 128
    persistent_dim: int = 64
    hidden_layers: int = 2
    temperature: float = .1
    sigreg_weight: float = 1.0
    sigreg_directions: int = 1024
    sigreg_times_per_video: int = 3
    lambda_cross: float = 1.
    lambda_align: float = .1
    query_frames: int = 3

    def __post_init__(self):
        if self.family not in ('JEPA', 'CPC', 'RSSM'):
            raise ValueError('Unknown feature dynamics family')
        if (self.feature_dim, self.width, self.persistent_dim) != (784, 128, 64):
            raise ValueError('v6 fixes RGB feature784 and U128=P64+T64')


def time_mask(mask, features):
    if mask.ndim == 2:
        mask = mask[:, None].expand(features.shape[:3])
    if mask.shape != features.shape[:3]:
        raise ValueError('Expected object presence [B,T,O] or [B,O]')
    return mask.bool()


class FrameProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * cfg.feature_dim),
                                 nn.Linear(2 * cfg.feature_dim, 256), nn.GELU(),
                                 nn.Linear(256, cfg.width), nn.LayerNorm(cfg.width))

    def forward(self, features, zero_delta=False):
        if features.ndim != 4 or features.shape[-1] != 784:
            raise ValueError('Motion projection requires [B,T,O,784] from one video')
        current = features.float()
        delta = torch.zeros_like(current)
        if not zero_delta:
            delta[:, 1:] = current[:, 1:] - current[:, :-1]
        return self.net(torch.cat((current, delta), -1))


class Interaction(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                    nn.Linear(width, width), nn.LayerNorm(width))

    def forward(self, tokens, mask):
        k = tokens.shape[-2]
        receiver = tokens[..., :, None, :].expand(*tokens.shape[:-2], k, k, tokens.shape[-1])
        sender = tokens[..., None, :, :].expand_as(receiver)
        edges = self.edge(torch.cat((receiver, sender), -1))
        pair = mask[..., :, None] & mask[..., None, :]
        pair = pair & ~torch.eye(k, device=tokens.device, dtype=torch.bool)
        messages = (edges * pair[..., None]).sum(-2) / pair.sum(-1).clamp_min(1)[..., None]
        return (tokens + self.update(torch.cat((tokens, messages), -1))) * mask[..., None]


class SIGReg(nn.Module):
    """Epps-Pulley statistic across episodes at fixed video/time/object slot."""
    def __init__(self, directions=1024):
        super().__init__()
        self.directions = directions
        t = torch.linspace(0., 3., 17)
        weights = torch.full((17,), 2 * 3 / 16)
        weights[[0, -1]] = 3 / 16
        self.register_buffer('t', t)
        self.register_buffer('phi', torch.exp(-.5 * t.square()))
        self.register_buffer('weights', weights * self.phi)

    def forward(self, embeddings, mask):
        # [groups, episodes, dimensions]. Time and objects NEVER become the
        # independent-sample axis; each group is one fixed time and color slot.
        if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError('SIGReg expects grouped episode samples and mask')
        eligible = mask.sum(1) >= 2
        if not eligible.any():
            return embeddings.sum() * 0
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            z = embeddings[eligible].float(); valid = mask[eligible]
            directions = F.normalize(torch.randn(z.shape[-1], self.directions, device=z.device), dim=0)
            scores = []
            for start in range(0, len(z), 4):
                zz, mm = z[start:start + 4], valid[start:start + 4]
                n = mm.sum(1).float()
                values = (zz @ directions)[..., None] * self.t
                weight = mm[:, :, None, None].float()
                real = (values.cos() * weight).sum(1) / n[:, None, None]
                imaginary = (values.sin() * weight).sum(1) / n[:, None, None]
                error = (real - self.phi).square() + imaginary.square()
                scores.append(((error @ self.weights) * n[:, None]).mean(1))
            return torch.cat(scores).mean()


def relation_loss(a, b):
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Alignment requires paired focal P vectors')
    if len(a) < 2:
        return (a.sum() + b.sum()) * 0, {'align_n': len(a)}
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b = a.float(), b.float()
        inv = F.mse_loss(a, b)
        sa, sb = (x.var(0, correction=1).add(1e-4).sqrt() for x in (a, b))
        var = .5 * (F.relu(1 - sa).mean() + F.relu(1 - sb).mean())
        cov = a.new_zeros(())
        eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.bool)
        for x in (a, b):
            centered = x - x.mean(0)
            covariance = centered.T @ centered / (len(x) - 1)
            cov = cov + covariance[~eye].square().sum() / x.shape[-1]
        loss = inv + var + .04 * cov
        return loss, {'align_n': len(a), 'align_inv': float(inv.detach()),
                      'align_var': float(var.detach()), 'align_cov': float(cov.detach()),
                      'p_std': float((.5 * (sa.mean() + sb.mean())).detach())}


def replace_focal_p(p, rows, focal, donor_p):
    """Only focal P changes; current T is a separate, untouched input."""
    if p.shape[-1] != 64 or donor_p.shape != (len(rows), 64):
        raise ValueError('v6.2 replacement acts on P64 only')
    selected = p[rows]
    onehot = F.one_hot(focal, p.shape[1]).to(p.dtype)[..., None]
    return selected * (1 - onehot) + donor_p[:, None] * onehot


class LatentDynamics(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.project = FrameProjection(config)
        self.history_interaction = Interaction(128)
        self.history = nn.GRU(128, 128, config.hidden_layers, batch_first=True)
        self.history_output = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64))
        self.current_interaction = Interaction(128)
        self.current = nn.GRU(128, 64, config.hidden_layers, batch_first=True)
        self.current_norm = nn.LayerNorm(64)
        self.initial = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128))
        self.dynamics_interaction = Interaction(128)
        self.transition_input = nn.Sequential(nn.Linear(128 + 64, 128), nn.GELU())
        self.recurrent = nn.GRUCell(128, 128)
        self.output = nn.Linear(128, 128)
        self.sigreg = SIGReg(config.sigreg_directions)

    @staticmethod
    def _temporal(sequence, mask, recurrent):
        b, t, k, d = sequence.shape
        sequence = sequence.permute(0, 2, 1, 3).reshape(b * k, t, 128)
        encoded, _ = recurrent(sequence)
        present = mask.permute(0, 2, 1).reshape(b * k, t)
        last = (present * torch.arange(1, t + 1, device=mask.device)).amax(1) - 1
        result = encoded[torch.arange(b * k, device=mask.device), last.clamp_min(0)]
        return result.reshape(b, k, -1), (last >= 0).reshape(b, k, 1)

    def encode(self, features_ab, mask):
        """Public frozen-readout interface: AB only -> P[B,O,64], no donor T."""
        mask = time_mask(mask, features_ab)
        sequence = self.history_interaction(self.project(features_ab), mask)
        result, valid = self._temporal(sequence, mask, self.history)
        return self.history_output(result) * valid

    def encode_current(self, query, mask):
        if query.shape[1] != self.config.query_frames:
            raise ValueError('Current encoder must read exactly the first three CD frames')
        mask = time_mask(mask, query)
        sequence = self.current_interaction(self.project(query), mask)
        result, valid = self._temporal(sequence, mask, self.current)
        return self.current_norm(result) * valid

    def predict(self, p, query, mask_query, steps, sample=False):
        transient = self.encode_current(query, mask_query)
        mask = time_mask(mask_query, query).any(1)
        return self.predict_from_current(p, transient, mask, steps)

    def predict_from_current(self, p, transient, mask, steps):
        b, k, _ = p.shape
        if transient.shape != p.shape or p.shape[-1] != 64:
            raise ValueError('Prediction requires P64 and query-only T64')
        state = self.initial(transient)
        result = []
        for _ in range(steps):
            context = self.dynamics_interaction(state, mask.bool())
            inp = self.transition_input(torch.cat((context, p), -1))
            state = self.recurrent(inp.reshape(b * k, 128), state.reshape(b * k, 128)).reshape(b, k, 128)
            state = state * mask[..., None]
            result.append(self.output(state))
        return torch.stack(result, 1)

    def target(self, full_cd):
        # LIVE gradients, same g as history/current; never used as predictor input.
        return self.project(full_cd)[:, self.config.query_frames:]

    def regularization(self, ab, mask_ab, cd, mask_cd):
        groups, masks = [], []
        for features, mask, first in ((ab, mask_ab, 0), (cd, mask_cd, self.config.query_frames)):
            times = torch.linspace(first, features.shape[1] - 1,
                                   self.config.sigreg_times_per_video, device=features.device).long().unique()
            z = self.project(features)[:, times]
            mm = time_mask(mask, features)[:, times]
            groups.append(z.permute(1, 2, 0, 3).flatten(0, 1))
            masks.append(mm.permute(1, 2, 0).flatten(0, 1))
        return self.sigreg(torch.cat(groups), torch.cat(masks))

    def history_parameters(self):
        # A common parameter set for comparing Self/Cross/Align gradients. The
        # motion projector also serves T/target, so it is deliberately excluded.
        return [p for module in (self.history_interaction, self.history, self.history_output)
                for p in module.parameters()]

    def artifact_config(self):
        return {'version': VERSION, **asdict(self.config)}


def make_model(family='JEPA', config=None):
    if config is None:
        config = ModelConfig(family=family)
    elif isinstance(config, dict):
        config = ModelConfig(**{k: v for k, v in config.items() if k != 'version'})
    if config.family == 'RSSM':
        raise NotImplementedError('RSSM has not passed the v6 feature rollout bridge yet')
    return LatentDynamics(config)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('version') != VERSION:
        raise ValueError('Checkpoint is not v6.2 P-only, query-three, live-target format')
    model = make_model(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint
