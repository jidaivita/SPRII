"""Resumable v6.2 source training from RGB-feature memmaps, without state labels.

prepare and smoke are bounded prerequisites; train stores exact 10/25/50 epoch
sources, not an independently selected latent-loss winner. Donors read AB only.
"""
import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as recompute

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophy_relations import RelationIndex
from models import VERSION, ModelConfig, make_model, relation_loss, replace_focal_p

METHODS = ('Base', 'Cross', 'Align', 'Both', 'Random-Both')
SPECS = {'balls': (30, 9), 'collision': (15, 4), 'blocktower': (30, 4)}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(data, tmp); os.replace(tmp, path)


def emit(event, **kwargs):
    print(json.dumps({'event': event, 'time': time.time(), **kwargs}, allow_nan=False), flush=True)


def seeded(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
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


def family_methods(family):
    return METHODS if family == 'CPC' else ()


def checked_binding(args, data):
    files = dict(data.files)
    for path in (Path(__file__), Path(__file__).with_name('models.py'),
                 Path(__file__).parents[1] / 'cophy_relations.py', Path(args.relation_index)):
        files[str(path.resolve())] = digest(path)
    if args.protocol:
        files[str(Path(args.protocol).resolve())] = digest(args.protocol)
    body = {'version': VERSION, 'scene': args.scene, 'features': str(data.path),
            'files': files, 'seed': args.seed, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'learning_rate': args.lr, 'weight_decay': 1e-4,
            'clip_norm': 1., 'source_task': 'CPC-style InfoNCE: P(fullAB)+T(CD[:3]) identifies recipient future motion latent among other episodes',
            'negative_rule': 'same future time, same public color-slot and visible object type; independent recipient episodes',
            'same_physical_targets': 'remain valid negatives because future states differ; this is not physical-identity InfoNCE',
            'cpc_temperature': .1, 'sigreg_weight': .2,
            'source_supports': 1, 'queries_per_memory': 1, 'test_read': False,
            'coordinate_labels_read': False, 'frontend': 'frozen supervised official last_cnn784',
            'source_selection': 'fixed epoch50; 10/25 diagnostic snapshots',
            'normalization': 'per-token learned LayerNorm, no cross-example or future pooled statistics',
            'target_gradients': 'live shared online motion projection; no EMA or detached teacher',
            'sigreg_statistics': 'three fixed AB times and three fixed futureCD times; independent episode axis within each time/slot',
            'model_config': asdict(ModelConfig(family=args.family))}
    body['sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return body


def setup(args):
    torch.set_num_threads(args.threads)
    if args.family == 'RSSM':
        raise NotImplementedError('RSSM source bridge is not yet released; do not dispatch')
    data = FeatureData(args.features, args.scene)
    binding = checked_binding(args, data)
    folder = Path(args.out) / args.family
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'binding.json'
    if not path.exists() or read(path) != binding:
        raise ValueError('Run prepare with these exact frozen inputs/configuration first')
    seeded(args.seed)
    model = make_model(args.family).to(args.device)
    planner = EpochPlanner(data, args.relation_index, args.seed)
    return data, binding, model, planner


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


def predict_micro(model, p, current_t, cm, steps, microbatch):
    output = []
    for s in range(0, len(p), microbatch):
        uu, cc, mm = p[s:s + microbatch], current_t[s:s + microbatch], cm[s:s + microbatch]
        if model.training and microbatch < len(p):
            pred = recompute(lambda a, b, d: model.predict_from_current(a, b, d, steps),
                             uu, cc, mm, use_reentrant=False)
        else:
            pred = model.predict_from_current(uu, cc, mm, steps)
        output.append(pred)
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


def batch_objective(model, data, plan, indices, method, microbatch):
    device = next(model.parameters()).device
    ab, am, c, cm = data.context('train', indices, device)
    cd, mask_cd = data.target('train', indices, device)
    active = cm.any(1)
    target_mask = mask_cd[:, 3:] & active[:, None]
    own = encode_micro(model, ab, am, microbatch)
    current_t = model.encode_current(c, cm)
    target = model.target(cd)
    prediction = predict_micro(model, own, current_t, active, data.frames - 3, microbatch)
    self_mse = masked_mse(prediction, target, target_mask)
    groups = torch.as_tensor(plan['public_groups'][indices], device=device)
    b, t, k, d = prediction.shape
    cpc_stats = {}
    if model.config.family == 'CPC':
        self_loss, cpc_stats = cpc_loss(prediction.permute(0, 2, 1, 3).reshape(b * k, t, d),
            target, target_mask, groups, torch.arange(b * k, device=device), model.config.temperature)
    else:
        self_loss = self_mse
    # The common anti-collapse term sees the same recipient observations/targets
    # in every arm. Donor multiplicity never increases this term's sample budget.
    reg = model.regularization(ab, am, cd, mask_cd)
    zero = own.sum() * 0
    cross, align = zero, zero
    donor_p = mixed = None
    rows = torch.empty(0, device=device, dtype=torch.long)
    focal = rows
    alignment_stats = {}
    if method != 'Base':
        local = np.flatnonzero(plan['focal'][indices] >= 0)
        if len(local):
            rows = torch.as_tensor(local, device=device)
            focal = torch.as_tensor(plan['focal'][indices[local]], device=device)
            donor_ids = plan['random' if method == 'Random-Both' else 'correct'][indices[local]]
            donor_ab, donor_mask = data.history('train', donor_ids, device)
            donors = encode_micro(model, donor_ab, donor_mask, microbatch)
            donor_p = donors[torch.arange(len(rows), device=device), focal]
            legal = (am[rows].any(1)[torch.arange(len(rows), device=device), focal]
                     & active[rows, focal]
                     & donor_mask.any(1)[torch.arange(len(rows), device=device), focal])
            rows, focal, donor_p = rows[legal], focal[legal], donor_p[legal]
            if len(rows):
                if method in ('Cross', 'Both', 'Random-Both'):
                    mixed = replace_focal_p(own, rows, focal, donor_p)
                    cross_pred = predict_micro(model, mixed, current_t[rows], active[rows], data.frames - 3, microbatch)
                    fp = cross_pred[torch.arange(len(rows), device=device), :, focal]
                    ft = target[rows, :, focal]
                    mask = target_mask[rows, :, focal]
                    if model.config.family == 'CPC':
                        cross, cross_stats = cpc_loss(fp, target, target_mask, groups, rows * k + focal,
                                                      model.config.temperature)
                        cpc_stats.update({'cross_' + n: v for n, v in cross_stats.items()})
                    else:
                        cross = masked_mse(fp, ft, mask)
                if method in ('Align', 'Both', 'Random-Both'):
                    align, alignment_stats = relation_loss(own[rows, focal], donor_p)
    loss = self_loss + model.config.sigreg_weight * reg + model.config.lambda_cross * cross + model.config.lambda_align * align
    metrics = {'loss': float(loss.detach()), 'self_loss': float(self_loss.detach()),
               'latent_mse': float(self_mse.detach()), 'cross': float(cross.detach()),
               'align': float(align.detach()), 'sigreg': float(reg.detach()),
               'p_std_all': float(own.detach().float().flatten(0, 1).std(0).mean()),
               'target_std': float(target.detach().float().flatten(0, 2).std(0).mean()),
               'paired': len(rows), **cpc_stats, **alignment_stats}
    return loss, metrics, {'own': own, 'mixed': mixed, 'donor_p': donor_p,
                          'rows': rows, 'focal': focal, 'target': target, 'current_t': current_t,
                          'terms': {'self': self_loss, 'cross': model.config.lambda_cross * cross,
                                    'align': model.config.lambda_align * align}}


@torch.no_grad()
def evaluate(model, data, batch_size=32, limit=512):
    model.eval(); device = next(model.parameters()).device
    values, ids, uncovered = [], [], []
    count = min(limit, len(data.ids['val'])) if limit else len(data.ids['val'])
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        ab, am, c, cm = data.context('val', indices, device)
        target, tm = data.target('val', indices, device)
        mask = tm[:, 3:] & cm.any(1)[:, None]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = model.encode(ab, am)
            prediction = model.predict(p, c, cm, data.frames - 3, sample=False)
            truth = model.target(target)
        row_error = (prediction.float() - truth.float()).square().mean(-1)
        denom = mask.sum((1, 2))
        scores = (row_error * mask).sum((1, 2)) / denom.clamp_min(1)
        for index, score, support in zip(indices, scores.tolist(), denom.tolist()):
            if support:
                ids.append(data.ids['val'][int(index)]); values.append(score)
            else:
                uncovered.append(data.ids['val'][int(index)])
    if not values or not np.isfinite(values).all():
        raise ValueError('No finite visual validation coverage')
    return {'mse': float(np.mean(values)), 'ids': ids, 'per_recipient_mse': values,
            'uncovered_ids': uncovered, 'recipients': count, 'scored': len(values),
            'metric': 'live motion-latent MSE diagnostic only, never cross-model ranking or source selection'}


def gradient_diagnostics(model, terms):
    """Same shared history parameter set; no automatic ratio threshold."""
    parameters = model.history_parameters()
    vectors = {}
    for name, loss in terms.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                                  for p, g in zip(parameters, gradients)])
    norms = {name: float(v.norm()) for name, v in vectors.items()}
    cosine = {}
    for name in ('cross', 'align'):
        denominator = norms['self'] * norms[name]
        cosine['self_' + name] = (float(vectors['self'] @ vectors[name]) / denominator
                                  if denominator > 0 else None)
    return {'norms': norms, 'cosines': cosine,
            'parameter_set': 'history_interaction+history_GRU+P_projection; excludes shared motion g/current/decoder',
            'parameter_count': sum(p.numel() for p in parameters),
            'losses_already_weighted': True, 'automatic_weight_change': False}


@torch.no_grad()
def delta_sensitivity(model, data):
    """Fixed train clips; raw-delta energy is frozen input, not a learning score."""
    indices = sorted(range(len(data.ids['train'])),
                     key=lambda i: hashlib.sha256(('v6.2-delta/' + data.ids['train'][i]).encode()).hexdigest())[:16]
    device = next(model.parameters()).device
    ab, mask = data.history('train', indices, device)
    old_mode = model.training; model.eval()
    actual = model.project(ab).float()[mask]
    zeroed = model.project(ab, zero_delta=True).float()[mask]
    if len(actual) < 2:
        raise ValueError('Fixed motion diagnostic has insufficient visible samples')
    numerator = (actual - zeroed).square().sum(-1).mean()
    denominator = (actual - actual.mean(0)).square().sum(-1).mean()
    delta = torch.zeros_like(ab); delta[:, 1:] = ab[:, 1:] - ab[:, :-1]
    raw_energy = delta[mask].square().sum(-1).mean()
    result = {'S_delta': float(numerator / (denominator + 1e-8)),
              'numerator': float(numerator), 'denominator': float(denominator),
              'raw_delta_energy': float(raw_energy), 'visible_tokens': len(actual),
              'ids': [data.ids['train'][i] for i in indices], 'split': 'train',
              'interpretation': 'explicit delta-input reliance, not physical information certificate',
              'test_read': False}
    model.train(old_mode)
    return result


def prepare(args):
    data = FeatureData(args.features, args.scene)
    binding = checked_binding(args, data)
    folder = Path(args.out) / args.family; folder.mkdir(parents=True, exist_ok=True)
    old = folder / 'binding.json'
    if old.exists() and read(old) != binding:
        raise ValueError('Refuse to overwrite a differently bound source family')
    planner = EpochPlanner(data, args.relation_index, args.seed)
    plan = planner.make(1, args.batch_size)
    seeded(args.seed); model = make_model(args.family)
    initialization = {'model_config': model.artifact_config(), 'state_sha256': state_sha(model),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'pretrained_dynamics': False, 'test_read': False}
    write(old, binding); write(folder / 'initialization.json', initialization)
    summary = {key: plan[key] for key in ('plan_sha256', 'query_exposures', 'paired', 'randomization_skipped', 'random_same')}
    write(folder / 'plan_epoch1.json', summary)
    emit('PREPARED', family=args.family, scene=args.scene, initialization=initialization, plan=summary)


def smoke(args):
    data, binding, model, planner = setup(args)
    plan = planner.make(1, args.batch_size)
    batch = plan['batches'][0]
    device = next(model.parameters()).device
    initial = {key: val.detach().clone() for key, val in model.state_dict().items()}
    records = []; common = None
    # Two semantic roles call the exact same online P encoder; no cached P or
    # EMA donor branch is allowed. Identical input must reproduce the same code.
    ab_check, mask_check = data.history('train', batch[:4], device)
    model.eval()
    with torch.no_grad():
        p_recipient = model.encode(ab_check, mask_check)
        p_donor = encode_micro(model, ab_check, mask_check, args.microbatch)
    torch.testing.assert_close(p_recipient, p_donor, atol=2e-5, rtol=2e-5)
    role_error = float((p_recipient - p_donor).abs().max())
    for method in family_methods(args.family):
        model.load_state_dict(initial); seeded(args.seed); model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss, metrics, detail = batch_objective(model, data, plan, batch, method, args.microbatch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite real-batch loss')
        if common is None:
            common = metrics['self_loss']
        elif not np.isclose(common, metrics['self_loss'], atol=1e-6, rtol=1e-5):
            raise ValueError('Same initialized self path differs across objective arms')
        if not detail['target'].requires_grad or hasattr(model, 'target_project'):
            raise ValueError('v6.2 requires live target gradients and no separate target encoder')
        target_gradient = torch.autograd.grad(detail['terms']['self'], detail['target'], retain_graph=True)[0]
        if not torch.isfinite(target_gradient).all() or target_gradient.norm() <= 0:
            raise ValueError('Prediction objective has no finite nonzero future-target gradient')
        own_gradient = torch.autograd.grad(loss, detail['own'], retain_graph=True)[0]
        if own_gradient.norm() <= 0:
            raise ValueError('Recipient P has no training gradient')
        donor_gradient = None
        if detail['donor_p'] is not None and len(detail['donor_p']):
            grad = torch.autograd.grad(loss, detail['donor_p'], retain_graph=True, allow_unused=True)[0]
            donor_gradient = float(grad.norm()) if grad is not None else 0.
            if donor_gradient <= 0:
                raise ValueError('Relation arm lacks a donor gradient on the real batch')
        if detail['mixed'] is not None:
            own = detail['own'][detail['rows']]; mixed = detail['mixed']
            nonfocal = ~F.one_hot(detail['focal'], own.shape[1]).bool()
            torch.testing.assert_close(own[nonfocal], mixed[nonfocal], atol=0, rtol=0)
            if detail['current_t'].shape != detail['own'].shape:
                raise ValueError('Current T and historical P must be separate equal-width inputs')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Real-batch gradient is nonfinite')
        records.append({'method': method, **metrics, 'donor_gradient': donor_gradient,
                        'recipient_p_gradient': float(own_gradient.norm()),
                        'target_gradient': float(target_gradient.norm())})
    result = {'status': 'PASS', 'scene': args.scene, 'family': args.family,
              'binding_sha256': binding['sha256'], 'rows': len(batch), 'records': records,
              'same_input_recipient_donor_max_error': role_error,
              'query_frames': 3, 'target_start_index': 3, 'target_gradient_mode': 'live',
              'peak_cuda_mb': torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else 0,
              'optimizer_updates': 0, 'test_read': False}
    write(Path(args.out) / args.family / 'real_batch_smoke.json', result)
    emit('SMOKE_PASS', **result)


def lock_file(stack, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    file = stack.enter_context(open(path, 'a+'))
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Worker/GPU already locked: ' + str(path)) from exc
    file.seek(0); file.truncate(); file.write(canonical({'pid': os.getpid(), 'host': socket.gethostname()})); file.flush()


def train(args):
    folder = Path(args.out) / args.family / args.method
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_file(stack, folder / 'worker.lock')
        if str(args.device).startswith('cuda'):
            lock_file(stack, Path(args.out).parent / '.gpu_locks' /
                      (socket.gethostname() + '_' + str(args.device).replace(':', '_') + '.lock'))
        data, binding, model, planner = setup(args)
        expected_counts = {'balls': (7000, 2000), 'collision': (14000, 4000), 'blocktower': (28310, 8088)}
        if tuple(len(data.ids[s]) for s in ('train', 'val')) != expected_counts[args.scene]:
            raise ValueError('Training/calibration requires the complete committed scene, not an engineering smoke subset')
        smoke_path = Path(args.out) / args.family / 'real_batch_smoke.json'
        smoke_result = read(smoke_path)
        if smoke_result.get('status') != 'PASS' or smoke_result.get('binding_sha256') != binding['sha256']:
            raise ValueError('Matching real-batch smoke is required')
        if (folder / 'complete.json').exists():
            done = read(folder / 'complete.json')
            if done.get('epochs') == args.epochs and done.get('binding_sha256') == binding['sha256']:
                emit('ALREADY_COMPLETE', **done); return
            raise ValueError('Existing completed run has a different contract')
        initial = read(Path(args.out) / args.family / 'initialization.json')
        if state_sha(model) != initial['state_sha256']:
            raise ValueError('Model initialization differs from family preparation')
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=1e-4)
        epoch, next_batch, step, history, running = 1, 0, 0, [], {}
        started = time.time(); prior_seconds = 0.; microbatch = args.microbatch
        checkpoint_path = folder / 'latest.pt'
        if checkpoint_path.exists():
            if not args.resume:
                raise ValueError('Partial run exists; resume explicitly instead of duplicating it')
            old = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if old['binding_sha256'] != binding['sha256'] or old['method'] != args.method:
                raise ValueError('Resume checkpoint belongs to another run')
            model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
            epoch, next_batch, step = old['next_epoch'], old['next_batch'], old['step']
            history, running = old['history'], old['running']
            microbatch = old.get('microbatch', microbatch); prior_seconds = old.get('seconds', 0.)
            restore_rng(old['rng'])
        elif args.resume:
            emit('RESUME_NEW_RUN', reason='No prior checkpoint; starting common initialization')
        device = next(model.parameters()).device

        def record(next_epoch, next_index):
            return {'version': VERSION, 'model': model.state_dict(), 'model_config': model.artifact_config(),
                    'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'method': args.method,
                    'family': args.family, 'scene': args.scene, 'binding_sha256': binding['sha256'],
                    'binding': binding, 'epoch': next_epoch - 1 if next_index == 0 else next_epoch,
                    'next_epoch': next_epoch, 'next_batch': next_index, 'step': step,
                    'history': history, 'running': running, 'microbatch': microbatch,
                    'initialization_sha256': initial['state_sha256'],
                    'seconds': prior_seconds + time.time() - started, 'test_read': False}

        write(folder / 'worker.json', {'status': 'RUNNING', 'pid': os.getpid(), 'host': socket.gethostname(),
                                      'device': str(device), 'started_at': started, 'binding_sha256': binding['sha256']})
        try:
            if step == 0:
                write(folder / 'delta_step_0.json', {'step': 0, **delta_sensitivity(model, data)})
            if args.max_steps and step >= args.max_steps:
                emit('CALIBRATION_ALREADY_REACHED', step=step, max_steps=args.max_steps,
                     resume_hint='Use --resume with --max-steps 0 for the remaining unchanged source budget')
                return
            while epoch <= args.epochs:
                plan = planner.make(epoch, args.batch_size)
                if next_batch == 0:
                    running = {'weighted': {}, 'examples': 0, 'steps': 0, 'started_at': time.time()}
                elif running.get('plan_sha256') != plan['plan_sha256']:
                    raise ValueError('Resumed pairing plan changed')
                running['plan_sha256'] = plan['plan_sha256']
                model.train()
                for batch_number in range(next_batch, len(plan['batches'])):
                    indices = plan['batches'][batch_number]
                    before_rng = rng_state()
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        try:
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                                loss, metrics, detail = batch_objective(model, data, plan, indices, args.method, microbatch)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite loss')
                            gradient_record = None
                            if step + 1 in (1, 50, 200, 500):
                                gradient_record = gradient_diagnostics(model, detail['terms'])
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                            del loss, detail
                            break
                        except torch.OutOfMemoryError:
                            optimizer.zero_grad(set_to_none=True)
                            if microbatch <= 1:
                                raise
                            microbatch = max(1, microbatch // 2)
                            import gc
                            if 'loss' in locals(): del loss
                            if 'detail' in locals(): del detail
                            gc.collect(); torch.cuda.empty_cache(); restore_rng(before_rng)
                            emit('OOM_MICROBATCH_RETRY', microbatch=microbatch, effective_batch=len(indices),
                                 full_batch_negatives_and_regularizers_preserved=True)
                    step += 1
                    if gradient_record is not None:
                        write(folder / f'gradients_step_{step}.json', {'step': step, 'epoch': epoch,
                              'batch': batch_number, 'metrics': metrics, **gradient_record})
                    if step in (1, 50, 200, 500):
                        write(folder / f'delta_step_{step}.json', {'step': step, **delta_sensitivity(model, data)})
                    running['examples'] += len(indices); running['steps'] += 1
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            running['weighted'][name] = running['weighted'].get(name, 0.) + value * len(indices)
                    next_batch = batch_number + 1
                    if step % 50 == 0 or next_batch == len(plan['batches']):
                        save(checkpoint_path, record(epoch, next_batch))
                        status = {'status': 'RUNNING', 'epoch': epoch, 'batch': next_batch,
                                  'batches': len(plan['batches']), 'step': step, 'history': history,
                                  'latest': metrics, 'target_gradient_mode': 'live', 'microbatch': microbatch,
                                  'grad_norm': float(grad_norm), 'plan_sha256': plan['plan_sha256'],
                                  'seconds': prior_seconds + time.time() - started, 'test_read': False}
                        write(folder / 'progress.json', status)
                        emit('TRAIN_PROGRESS', **{k: v for k, v in status.items() if k != 'history'})
                    if args.max_steps and step >= args.max_steps:
                        save(checkpoint_path, record(epoch, next_batch))
                        validation = evaluate(model, data, min(microbatch, 32))
                        result = {'status': 'CALIBRATION_COMPLETE', 'scene': args.scene,
                                  'method': args.method, 'family': args.family, 'step': step,
                                  'epoch': epoch, 'next_batch': next_batch,
                                  'binding_sha256': binding['sha256'], 'validation': validation,
                                  'checkpoint': str(checkpoint_path), 'full_source_complete': False,
                                  'seconds': prior_seconds + time.time() - started, 'test_read': False,
                                  'weights_changed_automatically': False,
                                  'resume_hint': 'Same objective resumes latest.pt, no restart; remove --max-steps'}
                        write(folder / 'calibration_complete.json', result)
                        write(folder / 'worker.json', {**result, 'pid': os.getpid(), 'exit_code': 0})
                        emit('CALIBRATION_COMPLETE', **result)
                        return
                validation = evaluate(model, data, min(microbatch, 32))
                averages = {name: value / running['examples'] for name, value in running['weighted'].items()}
                row = {'epoch': epoch, 'mse': validation['mse'], 'train': averages,
                       'seconds': time.time() - running['started_at'], 'plan_sha256': plan['plan_sha256'],
                       'query_exposures': running['examples'], 'paired': plan['paired'],
                       'randomization_skipped': plan['randomization_skipped'], 'random_same': plan['random_same']}
                if running['examples'] != len(data.ids['train']):
                    raise ValueError('Epoch did not expose every recipient exactly once')
                history.append(row)
                completed_epoch = epoch; epoch += 1; next_batch = 0; running = {}
                saved = record(epoch, 0)
                save(checkpoint_path, saved)
                if completed_epoch in (10, 25, 50):
                    save(folder / f'checkpoint_{completed_epoch}.pt', saved)
                    write(folder / f'validation_{completed_epoch}.json', {'epoch': completed_epoch, **validation})
                write(folder / 'progress.json', {'status': 'RUNNING', 'epoch': completed_epoch,
                      'step': step, 'history': history, 'microbatch': microbatch, 'test_read': False})
                emit('EPOCH_COMPLETE', method=args.method, family=args.family, **row)
            final = {'status': 'COMPLETE', 'version': VERSION, 'method': args.method, 'family': args.family,
                     'scene': args.scene, 'epochs': args.epochs, 'steps': step,
                     'binding_sha256': binding['sha256'], 'initialization_sha256': initial['state_sha256'],
                     'selected_epoch': args.epochs, 'selection': 'fixed common source budget, no latent-loss selection',
                     'seconds': prior_seconds + time.time() - started, 'coordinate_labels_read': False,
                     'test_read': False, 'finished_at': time.time()}
            if args.epochs not in (10, 25, 50):
                save(folder / f'checkpoint_{args.epochs}.pt', record(epoch, 0))
            final['checkpoint'] = str(folder / f'checkpoint_{args.epochs}.pt')
            final['checkpoint_sha256'] = digest(final['checkpoint'])
            write(folder / 'complete.json', final)
            write(folder / 'worker.json', {**final, 'pid': os.getpid(), 'exit_code': 0})
            emit('TRAIN_COMPLETE', **final)
        except Exception as error:
            failure = {'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(),
                       'epoch': epoch, 'next_batch': next_batch, 'step': step,
                       'binding_sha256': binding['sha256'], 'test_read': False}
            write(folder / 'failure.json', failure); emit('TRAIN_FAILED', **failure)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'smoke', 'train'))
    p.add_argument('--scene', required=True, choices=tuple(SPECS))
    p.add_argument('--features', required=True)
    p.add_argument('--relation-index', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--family', default='CPC', choices=('CPC',))
    p.add_argument('--method', default='Base', choices=METHODS)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--protocol')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int, default=0,
                   help='Pause after this global step with resumable state; 0 runs full source budget')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.method not in family_methods(args.family):
        raise ValueError('Undeclared method for this family')
    if not 1 <= args.epochs <= 50 or min(args.batch_size, args.microbatch, args.threads) < 1 or args.max_steps < 0:
        raise ValueError('Invalid fixed budget or batch/thread limit')
    {'prepare': prepare, 'smoke': smoke, 'train': train}[args.command](args)
