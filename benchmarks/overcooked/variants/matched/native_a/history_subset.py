"""Restrict available learning streams without changing the replacement sampler."""
import hashlib
import json
from pathlib import Path


def make_restricted_sampler_class(base):
    class RestrictedHistorySampler(base):
        def __init__(self, *args, history_allowlist=None, **kwargs):
            self.history_allowlist = None
            self.allowed_history_ids = None
            super().__init__(*args, **kwargs)
            if history_allowlist is None:
                return
            path = Path(history_allowlist).resolve()
            manifest = json.loads(path.read_text())
            assert manifest['schema'] == 'overcooked-history-subset/1'
            assert manifest['per_partner_histories'] in (32, 64, 128)
            assert set(manifest['identities']) == set(self.identities)
            assert len(self.identities) == 20
            for name in ('index', 'task_manifest', 'episode_index'):
                ref = manifest['metadata'][name]
                assert Path(ref['path']).resolve() == Path(self.paths[name]).resolve()
                assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
            original = {k: list(v) for k, v in self.identity_histories.items()}
            selected = set()
            for identity, available in original.items():
                ids = manifest['identities'][identity]
                assert len(available) == 128
                assert len(ids) == len(set(ids)) == manifest['per_partner_histories']
                assert all(type(hid) is int for hid in ids)
                assert set(ids) <= set(available)
                # Keep the original order, so the 100% path has identical RNG semantics.
                self.identity_histories[identity] = [hid for hid in available if hid in set(ids)]
                assert not selected.intersection(ids)
                selected.update(ids)
            assert len(selected) == 20 * manifest['per_partner_histories']
            self.allowed_history_ids = frozenset(selected)
            self.history_allowlist = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.available_history_count = len(selected)

        def _window(self, hid, *args, **kwargs):
            if self.allowed_history_ids is not None:
                assert int(hid) in self.allowed_history_ids, 'Attempt to read an excluded learning stream'
            return super()._window(hid, *args, **kwargs)

        def fingerprint(self):
            result = super().fingerprint()
            if self.history_allowlist is not None:
                path = Path(self.history_allowlist['path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == self.history_allowlist['sha256']
                result['history_allowlist'] = dict(self.history_allowlist)
            return result

        def sample(self, rng, batch_size):
            result = super().sample(rng, batch_size)
            if self.history_allowlist is not None:
                assert all(row['history_id'] in self.allowed_history_ids for row in self.last_metadata['rows'])
                self.last_metadata['history_allowlist_sha256'] = self.history_allowlist['sha256']
                self.last_metadata['available_histories'] = self.available_history_count
            return result

    return RestrictedHistorySampler
