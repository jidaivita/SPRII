"""Private data/relations: committed RGB caches, no coordinate labels."""
import hashlib, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
SPECS={'balls':(30,9),'collision':(15,4),'blocktower':(30,4)}
def read(path): return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)
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


class FeatureData:
    def __init__(self, path, scene):
        self.path, self.scene = Path(path).resolve(), scene
        self.ids, self.arrays, self.files = {}, {}, {}
        self.frames, self.slots = SPECS[scene]
        marker = self.path / 'scene_COMPLETE.json'
        if read(marker).get('status') != 'COMPLETE':
            raise ValueError('Feature scene is not completely and atomically prepared')
        self.files[str(marker)] = digest(marker)
        for split in ('train', 'val'):
            folder = self.path / split
            for name in ('manifest.json', 'COMPLETE.json', 'ids.json'):
                path = folder / name
                self.files[str(path)] = digest(path)
            if read(folder / 'COMPLETE.json').get('status') != 'COMPLETE':
                raise ValueError('Feature split is incomplete')
            ids = list(map(str, read(folder / 'ids.json')))
            if len(set(ids)) != len(ids) or not ids:
                raise ValueError('Feature IDs must be unique and nonempty')
            self.ids[split] = ids
            self.arrays[split] = {}
            for name in ('features_ab', 'features_cd', 'presence_ab', 'presence_cd'):
                arr = np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                shape = (len(ids), self.frames, self.slots) + ((784,) if name.startswith('features') else ())
                if arr.shape != shape:
                    raise ValueError(f'Unexpected {split}/{name} shape {arr.shape}; expected {shape}')
                if name.startswith('features') and arr.dtype not in (np.float16, np.float32):
                    raise ValueError('RGB features must be floating point')
                if name.startswith('presence') and arr.dtype not in (np.uint8, np.bool_):
                    raise ValueError('Visual presence must be uint8/bool')
                self.arrays[split][name] = arr
        if set(self.ids['train']) & set(self.ids['val']):
            raise ValueError('Train/validation ID overlap')

    def _tensor(self, split, name, indices, device, only_c=False):
        arr = self.arrays[split][name]
        # Index before conversion: no worker loads or copies the whole memmap.
        values = np.array((arr[:, :3] if only_c else arr)[np.asarray(indices)], copy=True)
        if name.startswith('presence'):
            if not np.isin(values, (0, 1)).all():
                raise ValueError('Invalid public visual presence')
            return torch.as_tensor(values, device=device, dtype=torch.bool)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite public RGB feature')
        return torch.as_tensor(values, device=device, dtype=torch.float32)

    def history(self, split, indices, device):
        return (self._tensor(split, 'features_ab', indices, device),
                self._tensor(split, 'presence_ab', indices, device))

    def context(self, split, indices, device):
        ab, mask = self.history(split, indices, device)
        return (ab, mask, self._tensor(split, 'features_cd', indices, device, True),
                self._tensor(split, 'presence_cd', indices, device, True))

    def target(self, split, indices, device):
        # Target-only full video gives index3 its legal index2 predecessor.
        # Context() separately exposes only CD[:3].
        return (self._tensor(split, 'features_cd', indices, device),
                self._tensor(split, 'presence_cd', indices, device))


class EpochPlanner:
    def __init__(self, data, index_path, seed):
        self.ids = data.ids['train']; self.lookup = {ident: i for i, ident in enumerate(self.ids)}
        self.index = RelationIndex(read(index_path), self.ids, data.scene)
        self.seed = seed; self.slots = data.slots
        # Only visible type (first stratum entry) enters CPC negative grouping.
        # Gravity and hidden physical labels remain exclusively in pairing.
        labels = {(slot, str(row['stratum'][0]))
                  for (ident, slot), row in self.index.records.items()}
        names = {key: i for i, key in enumerate(sorted(labels))}
        self.public_groups = np.full((len(self.ids), self.slots), -1, dtype=np.int64)
        for (ident, slot), row in self.index.records.items():
            self.public_groups[self.lookup[ident], slot] = names[(slot, str(row['stratum'][0]))]
        for slot in range(self.slots):
            absent = self.public_groups[:, slot] < 0
            self.public_groups[absent, slot] = len(names) + slot

    def make(self, epoch, batch_size):
        paired, skipped = self.index.plan(self.ids, self.seed, epoch, 0)
        focal = np.full(len(self.ids), -1, dtype=np.int64)
        correct = np.full(len(self.ids), -1, dtype=np.int64)
        randomized = np.full(len(self.ids), -1, dtype=np.int64)
        accidental = 0
        for row in paired:
            i = row['row']; focal[i] = row['focal']
            correct[i] = self.lookup[row['correct']['id']]
            randomized[i] = self.lookup[row['random']['id']]
            if correct[i] == i or randomized[i] == i:
                raise ValueError('Independent donor equals recipient episode')
            accidental += int(row['random_same'])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 6001]))
        order = rng.permutation(len(self.ids))
        batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.r_[batches[-2], batches.pop()]
        h = hashlib.sha256()
        for arr in (order, focal, correct, randomized):
            h.update(arr.tobytes())
        return {'batches': batches, 'focal': focal, 'correct': correct, 'random': randomized,
                'plan_sha256': h.hexdigest(), 'query_exposures': len(order),
                'paired': len(paired), 'randomization_skipped': skipped,
                'random_same': accidental, 'public_groups': self.public_groups}


class SourcePlanner(EpochPlanner):
    """Common eligibility; Random permutes within slot/type/global strata."""
    def __init__(self,data,index_path,seed):
        super().__init__(data,index_path,seed);self.data=data
        self.active=np.asarray(data.arrays['train']['presence_cd'][:,:3],bool).any(1)
        self.visible=np.asarray(data.arrays['train']['presence_ab'],bool).any(1)

    def make(self,epoch,batch_size):
        plan=super().make(epoch,batch_size)
        if self.data.scene!='collision':
            plan['route']='focal';return plan
        n=len(self.ids);k=self.slots
        correct=np.full((n,k),-1,np.int64);randomized=np.full((n,k),-1,np.int64)
        common=plan['focal']>=0
        for i,ident in enumerate(self.ids):
            if not common[i]:continue
            focal=int(plan['focal'][i]);correct[i,focal]=plan['correct'][i];randomized[i,focal]=plan['random'][i]
            for slot in np.flatnonzero(self.active[i]):
                record=self.index.records.get((ident,int(slot)))
                if record is None or not self.visible[i,slot]:common[i]=False;break
                pool=[r for r in self.index.by_relation.get((record['group'],record['physical_key']),[]) if r['id']!=ident and self.visible[self.lookup[r['id']],slot]]
                if not pool:common[i]=False;break
                if slot!=focal:
                    rng=np.random.default_rng(stable_seed(self.seed,epoch,f'rssm-all-{ident}-{slot}'))
                    correct[i,slot]=self.lookup[pool[int(rng.integers(len(pool)))]['id']]
            if not self.active[i,focal] or not self.visible[correct[i,focal],focal] or not self.visible[randomized[i,focal],focal]:common[i]=False
        groups=defaultdict(list)
        for i in np.flatnonzero(common):
            for slot in np.flatnonzero(self.active[i]):
                if slot!=plan['focal'][i]:
                    record=self.index.records[(self.ids[i],int(slot))]
                    groups[record['group']].append((i,int(slot),int(correct[i,slot])))
        skipped=0;accidental=0
        for group,rows in sorted(groups.items(),key=lambda item:repr(item[0])):
            rng=np.random.default_rng(stable_seed(self.seed,epoch,repr(group)+'all-random'))
            permutation=None
            for _ in range(256):
                perm=rng.permutation(len(rows))
                if all(rows[int(j)][2]!=rows[a][0] for a,j in enumerate(perm)):
                    permutation=perm;break
            if permutation is None:
                for i,slot,_ in rows:common[i]=False
                skipped+=len(rows);continue
            for a,(i,slot,_) in enumerate(rows):
                donor=rows[int(permutation[a])][2];randomized[i,slot]=donor
                accidental+=int(self.index.records[(self.ids[i],slot)]['physical_key']==self.index.records[(self.ids[donor],slot)]['physical_key'])
        for i in np.flatnonzero(common):
            if (correct[i,self.active[i]]<0).any() or (randomized[i,self.active[i]]<0).any():raise ValueError('All route missing donor')
        h=hashlib.sha256(plan['plan_sha256'].encode())
        for array in (correct,randomized,common):h.update(array.tobytes())
        plan.update(external_correct=correct,external_random=randomized,common=common,route='all',
                    original_plan_sha256=plan['plan_sha256'],plan_sha256=h.hexdigest(),
                    common_cross_recipients=int(common.sum()),extra_randomization_skipped=skipped,extra_random_same=accidental)
        return plan
