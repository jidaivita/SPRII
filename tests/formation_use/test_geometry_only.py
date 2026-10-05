"""Small native synthetic equivalence check; no research source training."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib
import numpy as np
import pytest
import torch


def test_native_donor_only_matches_full_export_and_geometry(tmp_path):
    root = os.environ.get('SPRII_SPRING_NATIVE')
    if root is None:
        pytest.skip('native Spring code snapshot not bound')
    root = Path(root)
    sys.path[:0] = [str(root), str(root / 'src'), str(root / 'a_src'), str(root / 'extension')]
    from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec, _new_model
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan
    from persistbench.envs.visual_elastic_coupling.a_head_features import extract_features, model_state_sha256
    from persistbench.envs.visual_elastic_coupling.schema import Episode
    from sprii_next.io import read
    from sprii_next.providers import SpringCache
    from sprii_next.geometry import source_geometry
    path = Path(__file__).parents[2] / 'benchmarks/formation_use/scripts/export_spring_geometry_only.py'
    spec = importlib.util.spec_from_file_location('geometry_only', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    episodes = []
    groups = [('train', 'train'), ('validation', 'continuous_new_systems'),
              ('validation', 'heldout_factorial_combinations')]
    for group, (split, stratum) in enumerate(groups):
        for index in range(2):
            number = group * 2 + index
            system = f's{number}'
            for kind, count in (('forced', 2), ('cold', 1), ('moving', 1)):
                for rep in range(count):
                    episodes.append(dict(episode_key=f'{system}:{kind}:{rep}', system_key=system,
                        theta=[1 + number * .1, .3 + number * .03, 2 + number * .4], split=split, stratum=stratum,
                        kind=kind, replicate=rep, requested_frames=96, raw_frames=96,
                        anchor=None if kind == 'forced' else 16))
    plan = AHeadCasePlan(dict(episodes=episodes), seed=0, history_frames=96)

    class SyntheticAccess:
        snapshot_sha256 = '1' * 64
        snapshot = {'content_sha256': '2' * 64}

        def __init__(self, donor_only=False):
            self.plan = plan
            self.audit = []
            self.donor_only = donor_only

        def verify_all(self, workers=8):
            return dict(status='SYNTHETIC_FIXTURE_ONLY')

        def public_episode(self, key, purpose):
            if self.donor_only and purpose != 'donor':
                raise AssertionError('unnecessary query access')
            row = plan.rows[key]
            self.audit.append(dict(episode_key=key, split=row['split'], kind='public_observation', purpose=purpose))
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            return Episode(rng.integers(0, 256, size=(96, 128, 128), dtype=np.uint8),
                           np.zeros((95, 2), np.float32), np.arange(96) * .05, np.zeros((96, 8)), {})

    torch.set_num_threads(1)
    model = _new_model('Both', PretrainingSpec(), torch.device('cpu')).eval().requires_grad_(False)
    model_hash = model_state_sha256(model)
    full_access = SyntheticAccess()
    extract_features(full_access, model, tmp_path / 'full', expected_model_state_sha256=model_hash, workers=1)
    full = read(tmp_path / 'full/FEATURES.json')
    expected = np.load(tmp_path / 'full/donor_slot.npy')
    only_access = SyntheticAccess(donor_only=True)
    keys, values, events = helper.extract_donors(only_access, model, model_hash)
    assert keys == full['donor_episodes']
    np.testing.assert_array_equal(values, expected[:, :64])
    assert len(events) == 12 and all(event['purpose'] == 'donor' for event in events)
    descriptor = dict(method='Both', source_seed=0)
    provider = helper.DonorProvider(descriptor, plan, keys, values)
    original = SimpleNamespace(plan=plan, labels={'train': None, 'validation': None},
        features=SimpleNamespace(_donor_index={key: i for i, key in enumerate(keys)}, _arrays={'donor_slot': expected}))
    for split in ('train', 'validation'):
        for actual, reference in zip(provider.donors(split), SpringCache.donors(original, split)):
            np.testing.assert_array_equal(actual, reference)
    class ReferenceProvider:
        environment = 'springworld'
        def __init__(self): self.descriptor = descriptor
        def donors(self, split): return SpringCache.donors(original, split)
    assert source_geometry(provider) == source_geometry(ReferenceProvider())
