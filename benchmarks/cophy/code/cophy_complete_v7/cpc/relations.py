"""Exact audited relation-index consumer copied from cophy_relations v3; no pose-target loader."""
import hashlib
import json
from collections import defaultdict
import numpy as np

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

