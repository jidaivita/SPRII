"""Small deterministic test of the actual three-link simulator, not a paper result."""
import json
from pathlib import Path
import numpy as np
from paper_c.swimmer.model import SwimmerModel

cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/swimmer_external_assay_v1.json').read_text())
model = SwimmerModel(cfg['model'])
initial = model.sample_initial_state(np.random.default_rng(23), cfg['transient_initial_state'])
actions = np.stack([0.5 * np.sin(np.arange(40) / 7), 0.3 * np.cos(np.arange(40) / 9)], axis=1)
times = np.array([10, 20, 30, 40])
reference = model.rollout(np.zeros(5), initial, actions, times)
repeat = model.rollout(np.zeros(5), initial, actions, times)
changed = model.rollout(np.log([1.2, 0.8, 1.1, 0.7, 1.3]), initial, actions, times)
assert reference.shape == (32,) and np.isfinite(reference).all()
np.testing.assert_array_equal(reference, repeat)
assert not np.allclose(reference, changed)
print('PASS: deterministic MuJoCo trajectories; persistent physical parameters change the trajectory.')
