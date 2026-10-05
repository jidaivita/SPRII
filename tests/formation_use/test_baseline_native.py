"""Bounded native-module compatibility check; no dataset or experiment training."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_native_relation_encoder_and_gradient_path():
    native = os.environ.get('SPRII_NATIVE_ROOT')
    if not native:
        pytest.skip('set SPRII_NATIVE_ROOT to the recovered native code tree')
    script = '''
import sys
import torch
from sprii_next.protocol import activate_native
activate_native(sys.argv[1])
from sprii_next.contrastive import new_source, symmetric_infonce, TEMPERATURE
from native_training import new_model
from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec
from persistbench.envs.visual_elastic_coupling.a_fresh_head import AFreshFeatures

torch.set_num_threads(1)
model = new_source(0).train()
reference = new_model('Both', PretrainingSpec(history_frames=96), torch.device('cpu'))
for component in ('observation', 'persistent'):
    for key, value in getattr(model, component).state_dict().items():
        torch.testing.assert_close(value, getattr(reference, component).state_dict()[key], rtol=0, atol=0)
assert model.temperature == TEMPERATURE == .07
assert model.transient is None and model.predictor is None
assert any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.projector)
model.begin_train_step()
# Four frames exercise the actual CNN and statistics lifecycle. Repeating their
# embeddings gives the native temporal shape without a costly full pixel batch.
frames = torch.randn(4, 1, 2, 128, 128)
history = model.observation(frames, train_history=True).expand(-1, 96, -1)
actions = torch.randn(4, 95, 2)
_, persistent, combined = model.codes(history, actions)
assert persistent.shape == (4, 64) and combined.shape == (4, 128)
loss, _ = symmetric_infonce(model.projector(persistent), persistent)
loss.backward()
for component in ('observation', 'persistent', 'projector'):
    gradients = [p.grad for p in getattr(model, component).parameters() if p.grad is not None]
    assert gradients and all(g.isfinite().all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
model.finish_train_step()
model.eval().requires_grad_(False)
features = AFreshFeatures(model)
features.initialize(None)
assert features.representation_dim == 64
assert model.observation._snapshot is None
print('native relation model, fair initialization, gradient path, and frozen adapter: PASS')
'''
    result = subprocess.run([sys.executable, '-c', script, native], text=True,
        capture_output=True, cwd=Path(__file__).resolve().parents[1], timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
