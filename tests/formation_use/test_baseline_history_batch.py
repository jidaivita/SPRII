"""Small native preparation equivalence fixture; the real-batch check is a script."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_native_history_only_inputs_and_receipt():
    native = os.environ.get('SPRII_NATIVE_ROOT')
    if not native:
        pytest.skip('set SPRII_NATIVE_ROOT to the recovered native code tree')
    script = '''
import sys
from types import SimpleNamespace
import numpy as np
import torch
from sprii_next.protocol import activate_native
activate_native(sys.argv[1])
import native_training
from sprii_next.contrastive import make_history_batch

torch.set_num_threads(1)
rng = np.random.default_rng(19)
episodes = {}
rows = {}
for key in ('s0_d', 's1_d', 's0_r', 's1_r'):
    episodes[key] = (rng.integers(0, 256, (128, 128, 128), dtype=np.uint8),
                     rng.uniform(-.4, .4, (127, 2)))
    rows[key] = dict(split='train', episode_key=key, assets={'128': dict(path=key, sha256='fixture')})
pairs = [dict(donor_episode='s0_d', recipient_episode='s0_r', donor_start=0, recipient_start=3, pair_sha256='pair0'),
         dict(donor_episode='s1_d', recipient_episode='s1_r', donor_start=2, recipient_start=5, pair_sha256='pair1')]
plan = dict(configuration={'name':'Both'}, pairs=pairs, plan_sha256='same-plan')
def validate(value): assert value is plan
schedule = SimpleNamespace(batch_pairs=2, rows=rows, validate=validate)
native_training.visible = lambda root, path, expected: episodes[path]
original, reference = native_training.make_batch(schedule, plan, 0, '/fixture')
actual, receipt = make_history_batch(schedule, plan, 0, '/fixture')
actual.validate(96)
for name in ('history_images', 'history_actions'):
    a, b = getattr(actual, name).numpy(), getattr(original, name).numpy()
    assert a.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
for key in ('plan_sha256','batch_index','input_assets','pair_sha256','windows',
            'raw_image_frames_read','presented_history_frames','observed_action_intervals',
            'direction','resolution','physical_labels_read','test_read'):
    assert receipt[key] == reference[key]
assert receipt['target_frames'] == receipt['future_targets_consumed'] == 0
assert receipt['target_horizons'] == [] and not hasattr(actual, 'target_images')
assert actual.to('cpu').history_images.shape == original.history_images.shape
print('native history/action bits, pair identity, and truthful receipt: PASS')
'''
    result = subprocess.run([sys.executable, '-c', script, native], capture_output=True,
        text=True, cwd=Path(__file__).resolve().parents[1], timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
