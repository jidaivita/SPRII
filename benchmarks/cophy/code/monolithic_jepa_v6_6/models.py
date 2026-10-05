"""Monolithic joint-context JEPA over frozen RGB object features.

One joint encoder consumes AB, a fixed segment boundary, and CD[:3]. There are
no persistent/transient branches or independent-history relation objectives.
The shared motion projector also creates live future targets; it reads full CD
only in target/regularization calls, never in the context encoder.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'cophy-monolithic-jepa-v6.6'


@dataclass(frozen=True)
class ModelConfig:
    family: str = 'Monolithic-JEPA'
    feature_dim: int = 784
    width: int = 128
    hidden_layers: int = 2
    sigreg_weight: float = .2
    sigreg_directions: int = 1024
    sigreg_times_per_video: int = 3
    query_frames: int = 3

    def __post_init__(self):
        if (self.family, self.feature_dim, self.width, self.hidden_layers, self.query_frames) != ('Monolithic-JEPA', 784, 128, 2, 3):
            raise ValueError('Monolithic v6.6 fixes feature784, U128, shared two-layer GRU, and query3')


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



class MonolithicDynamics(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.project = FrameProjection(config)
        self.joint_interaction = Interaction(config.width)
        self.joint_recurrent = nn.GRU(config.width, config.width, config.hidden_layers, batch_first=True)
        self.joint_norm = nn.LayerNorm(config.width)
        # Deterministic, nonlearned segment identifiers. No scene/physics labels.
        dims = torch.arange(1, config.width + 1, dtype=torch.float32)
        segments = torch.sin(torch.arange(1, 4, dtype=torch.float32)[:, None] * dims[None]) / math.sqrt(config.width)
        self.register_buffer('segment_vectors', segments)
        self.initial = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))
        self.dynamics_interaction = Interaction(128)
        self.transition_input = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.recurrent = nn.GRUCell(128, 128)
        self.output = nn.Linear(128, 128)
        self.sigreg = SIGReg(config.sigreg_directions)

    def _segment(self, features, mask, segment):
        mask = time_mask(mask, features)
        # Called separately on AB and CD[:3]; delta never crosses the boundary.
        z = self.project(features) + self.segment_vectors[segment].to(features.device)
        return self.joint_interaction(z, mask), mask

    def encode_history_state(self, ab, am):
        """Cacheable prefix of the SAME joint GRU, not a learned P branch.

        hidden: [layers,B,O,128]; present: [B,O]. Retain all layers.
        """
        sequence, mask = self._segment(ab, am, 0)
        b, t, k, d = sequence.shape
        sequence = sequence.permute(0, 2, 1, 3).reshape(b*k, t, d)
        _, hidden = self.joint_recurrent(sequence)
        return {'hidden': hidden.reshape(self.config.hidden_layers, b, k, d), 'present': mask.any(1)}

    def encode_current_tokens(self, current, cm):
        """Optional frozen-readout cache: query g/interaction, not GRU state."""
        if current.shape[1] != self.config.query_frames:
            raise ValueError('Joint context must see exactly CD[:3]')
        return self._segment(current, cm, 1)[0]

    def encode_joint_from_cached(self, state, current_tokens, cm):
        if current_tokens.ndim != 4 or current_tokens.shape[1] != 3 or current_tokens.shape[-1] != 128:
            raise ValueError('Expected three current-frame joint tokens [B,3,O,128]')
        b, t, k, d = current_tokens.shape
        if state['hidden'].shape != (self.config.hidden_layers, b, k, d) or state['present'].shape != (b, k):
            raise ValueError('History state/query shape mismatch')
        if cm.ndim == 2: cm = cm[:, None].expand(b, t, k)
        if cm.shape != (b, t, k): raise ValueError('Current presence shape mismatch')
        cm = cm.bool()
        boundary_mask = state['present'].bool() | cm.any(1)
        boundary = self.segment_vectors[2].to(current_tokens).view(1, 1, 1, d).expand(b, 1, k, d)
        boundary = boundary * boundary_mask[:, None, :, None]
        suffix = torch.cat((boundary, current_tokens), 1)
        suffix_mask = torch.cat((boundary_mask[:, None], cm), 1)
        h = state['hidden'].reshape(self.config.hidden_layers, b*k, d).contiguous()
        sequence = suffix.permute(0, 2, 1, 3).reshape(b*k, t+1, d)
        encoded, _ = self.joint_recurrent(sequence, h)
        present = suffix_mask.permute(0, 2, 1).reshape(b*k, t+1)
        last = (present * torch.arange(1, t+2, device=present.device)).amax(1)-1
        result = encoded[torch.arange(b*k, device=encoded.device), last.clamp_min(0)].reshape(b, k, d)
        return self.joint_norm(result) * (last >= 0).reshape(b, k, 1)

    def encode_joint_from_state(self, state, current, cm):
        return self.encode_joint_from_cached(state, self.encode_current_tokens(current, cm), cm)

    def encode_joint(self, ab, am, current, cm):
        return self.encode_joint_from_state(self.encode_history_state(ab, am), current, cm)

    def predict_from_context(self, u, active, steps):
        if u.ndim != 3 or u.shape[-1] != 128 or active.shape != u.shape[:2] or steps < 1:
            raise ValueError('Prediction requires joint U[B,O,128], active[B,O], and positive steps')
        b, k, _ = u.shape
        active = active.bool()
        state = self.initial(u) * active[..., None]
        result = []
        for _ in range(steps):
            context = self.dynamics_interaction(state, active)
            inp = self.transition_input(torch.cat((context, u), -1))
            state = self.recurrent(inp.reshape(b*k, 128), state.reshape(b*k, 128)).reshape(b, k, 128)
            state = state * active[..., None]
            result.append(self.output(state))
        return torch.stack(result, 1)

    def target(self, full_cd):
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


    def artifact_config(self):
        return {'version': VERSION, **asdict(self.config)}


def make_model(config=None):
    if config is None: config = ModelConfig()
    elif isinstance(config, dict):
        config = ModelConfig(**{k:v for k,v in config.items() if k != 'version'})
    return MonolithicDynamics(config)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('version') != VERSION:
        raise ValueError('Checkpoint is not monolithic-joint JEPA v6.6')
    model = make_model(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint
