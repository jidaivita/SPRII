"""DALI context/forward component on CoPhy frozen RGB features.

This is deliberately not a complete DALI Dreamer/RSSM reproduction. All AB
frames and all visible objects are available to its scene-level context.
The public graph/GRU reader consumes an 8D context zero-padded to U128;
its existing query path separately receives the three perceived current frames.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dali_context_torch import DALIContext

VERSION = 'dali-context-cophy-adaptation-v1'


@dataclass
class Config:
    scene: str = 'collision'
    frames: int = 15
    slots: int = 4
    feature_dim: int = 784
    context_dim: int = 8
    width: int = 256
    heads: int = 1
    fw_width: int = 128
    representation_dim: int = 128


class CoPhyDALI(nn.Module):
    representation_dim = 128
    history_batch_axes = {'context': 0}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        assert c.representation_dim == 128 and c.context_dim == 8
        self.core = DALIContext(c.slots*c.feature_dim, 1, c.frames,
                                context_dim=c.context_dim, width=c.width,
                                heads=c.heads, fw_width=c.fw_width)

    def scene_observations(self, features, presence):
        c = self.config
        if features.shape[1:] != (c.frames, c.slots, c.feature_dim):
            raise ValueError('Expected complete AB history; no prefix truncation')
        if presence.shape != features.shape[:-1]:
            raise ValueError('Presence dimensions disagree')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite frozen visual feature')
        return (features * presence[..., None]).flatten(2)

    def encode_history_state(self, ab, mask):
        obs = self.scene_observations(ab, mask)
        action = obs.new_zeros(*obs.shape[:2], 1)
        return {'context': self.core.encode(obs, action)}

    def encode_current_tokens(self, current, mask):
        # The shared reader already contains the full perceived query prefix.
        # No extra trainable current branch or future-query posterior is added.
        if current.shape[1:] != (3, self.config.slots, self.config.feature_dim):
            raise ValueError('Only the legal three-frame query prefix is allowed')
        return current.new_zeros(len(current), self.config.slots, 1)

    def encode_joint_from_cached(self, state, tokens, mask):
        context = state['context']
        if context.shape != (len(tokens), self.config.context_dim):
            raise ValueError('Cached context shape changed')
        padded = F.pad(context, (0, self.representation_dim-context.shape[-1]))
        return padded[:, None].expand(-1, self.config.slots, -1).contiguous()

    def encode_joint(self, ab, ab_mask, current, current_mask):
        return self.encode_joint_from_cached(self.encode_history_state(ab, ab_mask),
                                             self.encode_current_tokens(current, current_mask),
                                             current_mask)

    def source_loss(self, features, presence, prefix_lengths):
        obs = self.scene_observations(features, presence)
        actions = obs.new_zeros(*obs.shape[:2], 1)
        return self.core.prefix_loss(obs, actions, prefix_lengths)

    def artifact_config(self):
        return asdict(self.config)


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['version'] != VERSION or ck['method'] != 'DALI-context' or ck['test_read'] is not False:
        raise ValueError('Not a qualified CoPhy DALI component checkpoint')
    model = CoPhyDALI(Config(**ck['model_config']))
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device), ck
