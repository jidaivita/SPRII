"""128/96 input extension; frozen A/strict modules and objectives stay intact.

This module alone is not a runnable study. A native bank, sampler, supervision
bridge, downstream adapter and registered training budget are still required.
"""
from dataclasses import dataclass, fields
import torch
from torch import nn
from strict_model import StrictVisualJEPA, VisualBatch

PROFILE = 'springworld.native128-history96.v1'


@dataclass
class NativeVisualBatch(VisualBatch):
    def validate(self, history_length):
        if history_length != 96:
            raise ValueError('native profile requires 96 raw history frames')
        b = len(self.history_images)
        shapes = dict(history_images=(b,96,2,128,128), history_actions=(b,95,2),
                      target_images=(b,3,2,128,128), future_actions=(b,3,16,2),
                      action_masks=(b,3,16))
        for name, shape in shapes.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(name)
        if b < 4 or b % 2:
            raise ValueError('paired branch batches require even size >=4')
        if torch.any(self.history_images[:,0,1] != 0):
            raise ValueError('history must not expose a hidden predecessor')
        for hi, h in enumerate((1,4,16)):
            if not torch.all(self.action_masks[:,hi,:h] == 1) or torch.any(self.action_masks[:,hi,h:] != 0):
                raise ValueError('future mask does not match horizon')
            if torch.any(self.future_actions[:,hi,h:] != 0):
                raise ValueError('future padding must be zero')

    def to(self, device):
        result=type(self)(**{f.name:getattr(self,f.name).to(device) for f in fields(self)})
        if hasattr(self,'state_targets'):
            result.state_targets=self.state_targets.to(device)
        return result


class Native128JEPA(StrictVisualJEPA):
    observation_profile = PROFILE
    resolution = 128

    def __init__(self, variant='B3', *, projection_seed=0):
        if variant not in ('B0','B0_split','B2','Bx','B3'):
            raise ValueError('only declared JEPA variants; supervised state loss needs its own bridge')
        if type(projection_seed) is not int or not 0 <= projection_seed < 2**32:
            raise ValueError('projection initialization seed must be uint32')
        super().__init__(variant, history_length=96)
        # Four stride-two convolutions produce 8x8 features for128 input.
        # Independent seed avoids variant-dependent RNG offsets in this shared
        # projection. Original CNN, normalization and all temporal widths stay.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(projection_seed)
            self.observation.project = nn.Linear(256*8*8,128)
        self.projection_seed = projection_seed

    def encode_batch(self, batch):
        if not isinstance(batch, NativeVisualBatch):
            raise ValueError('native128 model requires its versioned public batch')
        return super().encode_batch(batch)

    def profile_record(self):
        return dict(profile=PROFILE, resolution=128, history_frames=96,
                    history_transitions=95, history_seconds=4.75,
                    projection_seed=self.projection_seed,
                    normalization=self.normalization_profile,
                    target_encoder='shared, gradient-bearing; no EMA',
                    objective='unmodified strict_objective/poke_objective',
                    parameter_count=sum(p.numel() for p in self.parameters()),
                    compatible_with_legacy64_checkpoint=False)
