"""Consume audited metadata and AB+C caches without loading donor targets.

Index schema: version, scene, split=train, records of
{id, slot, stratum: [...], physical: [...], recipient: bool, donor: bool}.
stratum includes slot/role/type and any audited global gravity. The caller
binds the entire index and field audit in the preflight receipt.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cophy_adapter import ABObservation, DonorPairs


def stable_seed(seed, epoch, batch):
    raw = f'cophy-pairs-v3:{seed}:{epoch}:{batch}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class RelationIndex:
    def __init__(self, data, train_ids, scene):
        if data.get('version') != 'cophy-relation-index-v3' or data.get('split') != 'train' or data.get('scene') != scene:
            raise ValueError('Need an audited v3 index for this scene and training split')
        self.records, self.by_recipient, self.by_relation = {}, defaultdict(list), defaultdict(list)
        train_ids = set(map(str, train_ids))
        for row in data['records']:
            ident, slot = str(row['id']), row['slot']
            if ident not in train_ids or type(slot) is not int or slot < 0:
                raise ValueError('Index includes unknown/nontraining ID or invalid slot')
            if type(row['recipient']) is not bool or type(row['donor']) is not bool:
                raise ValueError('Eligibility must come from the audit')
            if not row['stratum'] or not row['physical']:
                raise ValueError('Missing audited relation or structural stratum')
            key = (ident, slot)
            if key in self.records:
                raise ValueError('Duplicate object in relation index')
            record = dict(row, id=ident, key=key,
                          group=(slot, _key(row['stratum'])), physical_key=_key(row['physical']))
            self.records[key] = record
            if row['recipient']:
                self.by_recipient[ident].append(record)
            if row['donor']:
                self.by_relation[(record['group'], record['physical_key'])].append(record)
        # Eligibility is independent of A/Random and of prediction outcomes.
        self.options = {}
        for ident, records in self.by_recipient.items():
            eligible = []
            for record in records:
                donors = [r for r in self.by_relation[(record['group'], record['physical_key'])]
                          if r['id'] != ident]
                if donors:
                    eligible.append((record, donors))
            self.options[ident] = eligible

    def plan(self, ids, seed, epoch, batch):
        rng = np.random.default_rng(stable_seed(seed, epoch, batch))
        groups = defaultdict(list)
        for i, ident in enumerate(map(str, ids)):
            choices = self.options.get(ident, [])
            if choices:
                recipient, donors = choices[int(rng.integers(len(choices)))]
                donor = donors[int(rng.integers(len(donors)))]
                groups[recipient['group']].append((i, recipient, donor))
        result, skipped = [], 0
        for group in groups.values():
            if len(group) < 2:
                skipped += len(group)
                continue
            # Conditional random permutation, not enforced wrong physics. Identity
            # and incidental same-property matches are allowed. No self episode.
            for _ in range(256):
                permutation = rng.permutation(len(group))
                if all(group[int(j)][2]['id'] != group[i][1]['id'] for i, j in enumerate(permutation)):
                    break
            else:
                # Explicit finite sampling limit; do not claim proven infeasibility.
                skipped += len(group)
                continue
            for i, (row, recipient, correct) in enumerate(group):
                random = group[int(permutation[i])][2]
                result.append({'row': row, 'focal': recipient['slot'], 'correct': correct,
                               'random': random, 'random_same': random['physical_key'] == recipient['physical_key']})
        return sorted(result, key=lambda r: r['row']), skipped


class PairProvider:
    def __init__(self, index, dataset, seed):
        if not hasattr(dataset, 'dict_id2object_properties'):
            raise ValueError('A/Random requires audited AB+C v2 caches')
        self.index, self.cache, self.seed = index, dataset.dict_id2object_properties, seed
        self.ids = list(map(str, dataset.list_ex))
        self.epoch = None
        self.epoch_plan = {}
        self.epoch_randomization_skipped = 0

    def make(self, ids, method, epoch, batch, visual):
        if method not in {'A', 'Random'}:
            raise ValueError('Donor provider is only for A/Random training')
        if epoch != self.epoch:
            # Randomize the complete epoch's donor multiset within each stratum.
            # A minibatch need not contain two objects from a rare gravity/type
            # stratum; do not drop otherwise legal relations just for that reason.
            complete, self.epoch_randomization_skipped = self.index.plan(self.ids, self.seed, epoch, 0)
            self.epoch_plan = {self.ids[r['row']]: r for r in complete}
            self.epoch = epoch
        plan = []
        for i, ident in enumerate(map(str, ids)):
            if ident not in self.cache:
                raise ValueError('Recipient ID outside the audited training cache')
            if ident in self.epoch_plan:
                plan.append(dict(self.epoch_plan[ident], row=i))
        skipped = self.epoch_randomization_skipped if batch == 0 else 0
        device = visual.c.device
        rows = torch.tensor([r['row'] for r in plan], dtype=torch.long, device=device)
        focal = torch.tensor([r['focal'] for r in plan], dtype=torch.long, device=device)
        donor_kind = 'correct' if method == 'A' else 'random'
        poses, masks, slots = [], [], []
        for row in plan:
            donor = row[donor_kind]
            cache = self.cache[donor['id']]
            if cache.get('cache_version') != 'ab_c_float32_v2':
                raise ValueError('Unqualified donor cache')
            # No dataset.__getitem__: no donor CD, targets or labels are loaded.
            poses.append(np.asarray(cache['pose_ab'], dtype=np.float32))
            masks.append(np.asarray(cache['presence_ab'], dtype=np.float32))
            slots.append(donor['slot'])
        if plan:
            ab = ABObservation(torch.as_tensor(np.stack(poses), device=device),
                               torch.as_tensor(np.stack(masks), device=device))
        else:
            ab = ABObservation(visual.ab.pose[:0], visual.ab.presence[:0])
        pairs = DonorPairs(rows, focal, ab, torch.tensor(slots, dtype=torch.long, device=device))
        return pairs, {'randomization_skipped': skipped,
                       'random_same': sum(r['random_same'] for r in plan),
                       'donor_ids': sorted({r[donor_kind]['id'] for r in plan}),
                       'donor_episodes': len({r[donor_kind]['id'] for r in plan})}


class ParameterFeatures:
    """Audited, train-standardized physical values; only Param-known receives them."""
    def __init__(self, data, scene, split):
        if (data.get('version') != 'cophy-parameters-v3' or data.get('scene') != scene or
                data.get('split') != split or data.get('all_varying_parameters_included') is not True):
            raise ValueError('Parameter features must include all reliably audited varying parameters')
        self.fields = data['fields']
        if not self.fields or len({f['name'] for f in self.fields}) != len(self.fields):
            raise ValueError('Missing or duplicate parameter feature fields')
        for field in self.fields:
            if not np.isfinite([field['train_mean'], field['train_scale']]).all() or field['train_scale'] <= 0:
                raise ValueError('Invalid train-only standardization')
        self.schema = _key(self.fields)
        self.examples = data['examples']

    def batch(self, ids, objects, device):
        raw = np.asarray([self.examples[str(i)] for i in ids], dtype=np.float32)
        if raw.shape != (len(ids), objects, len(self.fields)) or not np.isfinite(raw).all():
            raise ValueError('Missing/malformed audited parameter features')
        # Raw values are normalized using the same frozen TRAIN statistics on all splits.
        mean = np.array([f['train_mean'] for f in self.fields], np.float32)
        scale = np.array([f['train_scale'] for f in self.fields], np.float32)
        return torch.as_tensor((raw-mean)/scale, device=device)


def artifact_path(preflight_path, name):
    data = json.loads(Path(preflight_path).read_text())
    try:
        path = Path(data['artifacts'][name]['path'])
    except KeyError as error:
        raise ValueError(f'Missing preflight artifact {name}') from error
    return path if path.is_absolute() else Path(preflight_path).parent / path


def read_artifact(preflight_path, name):
    from cophy_protocol import digest
    path = artifact_path(preflight_path, name)
    item = json.loads(Path(preflight_path).read_text())['artifacts'][name]
    if digest(path) != item['sha256']:
        raise ValueError(f'Artifact changed: {name}')
    return json.loads(path.read_text())
