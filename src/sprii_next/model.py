"""Interface capacity matched readers, independently fitted per arm."""
import torch
from torch import nn

HORIZONS = (1, 2, 4, 8, 16)


class Reader(nn.Module):
    def __init__(self, arm, seed, physics_dim=3):
        super().__init__()
        if arm not in ('null', 'persistent', 'matched', 'decode', 'oracle', 'decode4', 'oracle4'):
            raise ValueError('unknown arm')
        self.arm = arm
        physics_dim = 4 if arm.endswith('4') else physics_dim
        # Initialize common modules first so their initial values pair across arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context_adapter = nn.Linear(64, 64)
            self.horizon = nn.Embedding(5, 16)
            self.trunk = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 8))
            self.physics_projection = nn.Linear(physics_dim, 64) if arm.startswith(('decode', 'oracle')) else None

    def forward(self, query, context, actions, mask, horizon_index):
        if self.arm == 'null':
            context = torch.zeros((len(query), 64), dtype=query.dtype, device=query.device)
        elif self.physics_projection is not None:
            context = self.physics_projection(context)
        if context.shape != (len(query), 64) or query.shape != (len(query), 128):
            raise ValueError('reader interface shape mismatch')
        x = torch.cat((query, self.context_adapter(context), actions.flatten(1), mask, self.horizon(horizon_index)), dim=1)
        return self.trunk(x)

    def architecture(self):
        return dict(profile='p64_physics_projection_common64_trunk256x2_v1',
                    common_context_adapter=[64, 64], trunk=[256, 256, 256, 8],
                    projection=None if self.physics_projection is None else [self.physics_projection.in_features, 64],
                    parameters=sum(p.numel() for p in self.parameters()),
                    common_parameters=sum(p.numel() for n, p in self.named_parameters() if not n.startswith('physics_projection')))
