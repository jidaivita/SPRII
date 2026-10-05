#!/usr/bin/env python3
"""Amortized histories on the exact frozen CoDA component decoder.

All numerical CoDA definitions and RK4 remain in coda_burgers_components.py.
Only this independent adapter and new history encoder are trained.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def tensor_sha(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def model_sha(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(tensor_sha(v).encode())
    return h.hexdigest()


class HistoryEncoder(nn.Module):
    def __init__(self, obs_mean, obs_std, code_mean, code_std):
        super().__init__()
        assert float(obs_std) > 1e-8
        assert torch.as_tensor(code_std).shape == (2,)
        assert bool((torch.as_tensor(code_std) > 1e-8).all())
        self.register_buffer('obs_mean', torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer('obs_std', torch.as_tensor(obs_std, dtype=torch.float32))
        self.register_buffer('code_mean', torch.as_tensor(code_mean, dtype=torch.float32))
        self.register_buffer('code_std', torch.as_tensor(code_std, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, support):
        assert support.ndim == 4 and tuple(support.shape[1:]) == (1, 401, 101)
        z = self.network(((support - self.obs_mean) / self.obs_std).transpose(-1, -2))
        return z * self.code_std + self.code_mean


class FrozenDecoder:
    """A separate CoDA grouped template; tensors stay connected to the encoder."""
    def __init__(self, components, source, n_contexts):
        self.c = components
        self.model = components.adapted(source, n_contexts)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.shared_before = components.shared_digest(self.model)
        self.source = source
        self.source_before = model_sha(source)
        self.n_contexts = n_contexts
        # nn.Module otherwise disallows replacing registered Parameter with Tensor.
        # The two released references must point at the SAME differentiable tensor.
        for module in (self.model.derivative, self.model.derivative.net_leaf):
            del module._parameters['codes']

    def bind(self, codes):
        assert codes.ndim == 2 and tuple(codes.shape) == (self.n_contexts, 2)
        assert torch.isfinite(codes).all()
        object.__setattr__(self.model.derivative, 'codes', codes)
        object.__setattr__(self.model.derivative.net_leaf, 'codes', codes)
        assert self.model.derivative.codes is self.model.derivative.net_leaf.codes

    def training_reconstruction(self, histories, codes, times, epsilon):
        assert histories.shape[0] == self.n_contexts
        self.bind(codes)
        return self.c.forecast(self.model, histories[:, 0].unsqueeze(0), times, epsilon)[0].unsqueeze(1)

    def predict(self, initial_state, codes, times):
        # There is deliberately no query-future argument in this interface.
        assert initial_state.ndim == 3 and tuple(initial_state.shape) == (self.n_contexts, 1, 401)
        truth_free = initial_state.new_zeros((self.n_contexts, 1, 401, len(times)))
        truth_free[..., 0] = initial_state
        self.bind(codes)
        return self.c.forecast(self.model, truth_free[:, 0].unsqueeze(0), times, 0)[0].unsqueeze(1)

    def audit(self):
        assert self.c.shared_digest(self.model) == self.shared_before
        assert model_sha(self.source) == self.source_before
        assert all(p.grad is None and not p.requires_grad for p in self.source.parameters())
        assert all(p.grad is None and not p.requires_grad for p in self.model.parameters())
        return {'shared_decoder_unchanged': True, 'source_unchanged': True,
                'source_full_digest': self.source_before, 'shared_digest': self.shared_before}


def smoke(components, source=None, device='cpu'):
    """Actual CoDA width64/code2/RK4, small time grid; no data or optimizer on source."""
    torch.set_num_threads(1)
    torch.manual_seed(7264); np.random.seed(7264)
    synthetic_source = source is None
    if source is None:
        source = components.build(9, device)
    source.eval()
    for p in source.parameters(): p.requires_grad_(False)
    before = model_sha(source)
    histories = torch.randn(2, 1, 401, 101, device=device) * .1
    encoder = HistoryEncoder(0., 1., torch.tensor([.01, -.02]), torch.tensor([.1, .2])).to(device)
    codes = encoder(histories)
    codes.retain_grad()
    times = torch.linspace(0., .001, 3, device=device)
    adapter = FrozenDecoder(components, source, 2)
    predicted = adapter.predict(histories[..., 0], codes, times)
    reference = components.adapted(source, 2)
    with torch.no_grad():
        reference.derivative.codes.copy_(codes.detach())
        only_initial = histories.new_zeros((1, 2, 401, 3))
        only_initial[..., 0] = histories[:, 0, :, 0]
        original = components.forecast(reference, only_initial, times, 0)[0].unsqueeze(1)
    torch.testing.assert_close(predicted, original, rtol=2e-6, atol=2e-8)
    target = histories[..., :3]
    loss = (predicted - target).square().mean()
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert codes.grad is not None and float(codes.grad.abs().sum()) > 0
    enc_before = model_sha(encoder)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    opt.step()
    assert model_sha(encoder) != enc_before
    with torch.no_grad():
        z = codes.detach()
        q1 = histories.clone(); q2 = histories.clone(); q2[..., 1:] += 100.
        # Both invocations can access only the same observed query initial state.
        a = adapter.predict(q1[..., 0], z, times)
        b = adapter.predict(q2[..., 0], z, times)
        assert torch.equal(a, b)
    adapter.audit(); assert model_sha(source) == before
    return {'status': 'PASS', 'actual_decoder_width': 64, 'actual_code_dim': 2,
            'solver': 'rk4', 'original_output_parity_max_abs': float((predicted.detach()-original).abs().max()),
            'encoder_gradient_l1': sum(float(g.abs().sum()) for g in grads),
            'code_gradient_l1': float(codes.grad.abs().sum()),
            'encoder_step_changed': True, 'source_parameters_and_codes_unchanged': True,
            'query_future_perturbation_prediction_unchanged': True,
            'source_optimizer_updates': 0, 'synthetic_only': synthetic_source,
            'training_or_test_data_read': False}
