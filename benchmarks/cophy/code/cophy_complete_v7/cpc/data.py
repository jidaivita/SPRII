"""Frozen feature IO, recipient ordering and InfoNCE from CPC v6.3."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as recompute
from .relations import RelationIndex
SPECS={'balls':(30,9),'collision':(15,4),'blocktower':(30,4)}
def read(p): return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''): h.update(b)
 return h.hexdigest()

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


def encode_micro(model, x, mask, microbatch):
    output = []
    for s in range(0, len(x), microbatch):
        xx, mm = x[s:s + microbatch], mask[s:s + microbatch]
        if model.training and microbatch < len(x):
            u = recompute(model.encode, xx, mm, use_reentrant=False)
        else:
            u = model.encode(xx, mm)
        output.append(u)
    return torch.cat(output)


def masked_mse(prediction, target, mask):
    values = (prediction.float() - target.float()).square().mean(-1)
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1)


def cpc_loss(prediction, target_pool, target_mask, groups, anchor_indices, temperature):
    """Full effective-batch negatives, unaffected by forward microbatch size."""
    b, t, k, d = target_pool.shape
    targets = target_pool.permute(1, 0, 2, 3).reshape(t, b * k, d).float()
    masks = target_mask.permute(1, 0, 2).reshape(t, b * k)
    predictions = prediction.transpose(0, 1).float()
    labels = groups.flatten()
    allowed_group = labels[anchor_indices, None] == labels[None, :]
    total, number, hits = predictions.sum() * 0, 0, 0
    eligible_counts = []
    for step in range(t):
        allowed = allowed_group & masks[step][None]
        valid = masks[step, anchor_indices] & (allowed.sum(1) >= 2)
        if not valid.any():
            continue
        logits = F.normalize(predictions[step, valid], dim=-1) @ F.normalize(targets[step], dim=-1).T / temperature
        logits = logits.masked_fill(~allowed[valid], -torch.inf)
        expected = anchor_indices[valid]
        eligible_counts.extend(allowed[valid].sum(1).detach().cpu().tolist())
        total = total + F.cross_entropy(logits, expected, reduction='sum')
        number += len(expected)
        hits += int((logits.detach().argmax(1) == expected).sum())
    return total / max(number, 1), {'cpc_anchors': number, 'cpc_top1': hits / max(number, 1),
        'cpc_candidates_mean': float(np.mean(eligible_counts)) if eligible_counts else 0.,
        'cpc_uniform_chance': float(np.mean(1/np.asarray(eligible_counts))) if eligible_counts else 0.}

