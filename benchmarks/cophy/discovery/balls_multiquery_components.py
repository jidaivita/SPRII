"""Balls4 components for differentiable, independent-history reuse adaptation.

The existing Balls4 files retain nine padded object slots; only up to four are
active. Preserve those slots so the old Native head remains exactly loadable.
History encoding uses the original three visual coordinates. The recipient
supplies only three xy frames and one detection channel; predict 27 xy frames.
Physical attributes and the derived constant public type are sampler metadata.
This module never edits the source artifacts or imports Collision constants.
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from xep_discovery import (PredictHead, digest, read, write, emit, save_torch,
                           save_npz, scores)


VERSION = 'balls-multiquery-components-v4.9-1'
K, D, HORIZON, PREFIX = 9, 2, 27, 3
TYPES = ('ball',)  # Sampler-only constant, never appended to the model input.


def code_statistics(base, manifest=None):
    """Exactly the frozen-readout normalization, fitted to train active codes."""
    base = Path(base)
    manifest = read(base / 'manifest.json') if manifest is None else manifest
    with np.load(base / 'codes/Native_train.npz', allow_pickle=False) as saved:
        if saved['ids'].astype(str).tolist() != manifest['splits']['train']['all_ids']:
            raise ValueError('Native train code order differs from the manifest')
        if saved['u'].shape[1:] != (K, 32):
            raise ValueError('Balls Native codes must retain nine padded slots')
        observed = saved['u'][saved['presence'] > 0]
        mean, scale = observed.mean(0), observed.std(0)
    scale[scale < 1e-6] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite Native code normalization')
    return mean, scale


def prepare_binding(base, head=None):
    """Return immutable inputs for a common Native source + selected S3 head.

    Callers save this into their own new run binding. The default head was
    selected from the completed, standard S3 100-epoch budget, not the S5 run.
    """
    base = Path(base)
    manifest = read(base / 'manifest.json')
    if manifest.get('scene') != 'balls4' or manifest.get('prefix') != PREFIX:
        raise ValueError('Expected the existing Balls4 three-frame query manifest')
    if manifest.get('test_read') is not False:
        raise ValueError('Expected train/validation-only source manifest')
    source = dict(manifest['source_models']['Native'])
    if digest(source['path']) != source['sha256']:
        raise ValueError('Native source checkpoint changed')
    if head is None:
        head = base.parent / 'xep_advantage_v4_2/runs/standard_s3/Native-U/selected.pt'
    head = Path(head)
    state = torch.load(head, map_location='cpu', weights_only=False)
    receipt = read(head.with_name('selected_validation.json'))
    progress = read(head.with_name('progress.json'))
    complete = read(head.with_name('complete.json'))
    if (progress.get('epoch') != 100 or complete.get('epochs') != 100 or
            receipt.get('method') != 'Native-U' or state['epoch'] != receipt['epoch']):
        raise ValueError('Expected the Native head selected from its completed 100-epoch budget')
    if progress.get('setting') != 'standard_s3' or complete.get('setting') != 'standard_s3':
        raise ValueError('Warm-start head must be standard S3, not a selected alternative setting')
    if receipt['ids'] != manifest['splits']['val']['query_ids']:
        raise ValueError('Warm-start validation query cohort differs')
    del state
    mean, scale = code_statistics(base, manifest)
    files = [base / 'manifest.json', Path(source['path']), head,
             head.with_name('selected_validation.json'), head.with_name('progress.json'),
             head.with_name('complete.json'), base / 'codes/Native_train.npz',
             Path(__file__), Path(__file__).with_name('xep_discovery.py')]
    splits = {}
    for split in ('train', 'val'):
        part = manifest['splits'][split]
        if digest(part['cache']) != part['cache_sha256']:
            raise ValueError('Source visual cache changed')
        paths = [Path(part['cache']), base / f'input_{split}.npz',
                 base / f'target_{split}.npz', base / f'codes/Native_{split}.npz']
        files.extend(paths)
        splits[split] = {'queries': len(part['query_ids']), 'cache': part['cache'],
                         'cache_sha256': part['cache_sha256']}
    return {'version': VERSION, 'base': str(base), 'test_read': False,
            'source': source, 'head': {'path': str(head), 'sha256': digest(head),
                'epoch': receipt['epoch'], 'budget': 100, 'setting': 'standard_s3',
                'mse': receipt['mse']},
            'code_mean': mean.tolist(), 'code_scale': scale.tolist(),
            'base_manifest_sha256': digest(base / 'manifest.json'),
            'slots': K, 'maximum_active_objects': 4, 'query_prefix': PREFIX,
            'prediction_dimension': D, 'horizon': HORIZON,
            'public_type_input': False, 'history_coordinates': 3,
            'splits': splits, 'file_sha256': {str(path): digest(path) for path in files}}


class HistoryEncoder(nn.Module):
    """Source GCN/GRU with the time axis batched, retaining edge orientation."""
    def __init__(self, source_state):
        super().__init__()
        self.mlp_inter = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32),
                                       nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.mlp_out = nn.Sequential(nn.Linear(67, 32), nn.ReLU(), nn.Linear(32, 32))
        self.rnn = nn.GRU(32, 32, batch_first=True)
        wanted = {key.removeprefix('backbone.'): value for key, value in source_state.items()
                  if key.startswith(('backbone.mlp_inter.', 'backbone.mlp_out.', 'backbone.rnn.'))}
        self.load_state_dict(wanted, strict=True)

    def forward(self, pose, presence):
        b, t, k, d = pose.shape
        if d != 3 or presence.shape != (b, k):
            raise ValueError('History encoder requires xyz observations and scene presence')
        x = pose.reshape(b * t, k, 3)
        active = presence[:, None].expand(b, t, k).reshape(b * t, k)
        x1 = x[:, None].expand(-1, k, -1, -1)
        x2 = x[:, :, None].expand(-1, -1, k, -1)
        edges = self.mlp_inter(torch.cat([x1, x2], -1))
        pair = active[:, :, None] * active[:, None, :]
        pair = pair * (1 - torch.eye(k, device=x.device)[None])
        e = (edges * pair[..., None]).sum(2) / (.01 + pair.sum(2))[..., None]
        total = (e * active[..., None]).sum(1) / (.01 + active.sum(1))[:, None]
        local = self.mlp_out(torch.cat([x, e, total[:, None].expand(-1, k, -1)], -1))
        seq = local.reshape(b, t, k, 32).permute(0, 2, 1, 3).reshape(b * k, t, 32)
        _, hidden = self.rnn(seq)
        return hidden[0].reshape(b, k, 32)


class ReuseHead(PredictHead):
    """Old Balls head, with a zero-initialized extra 64-memory input per step."""
    def __init__(self, legacy_state):
        super().__init__('Native-U')
        old_input = self.cell.input_size
        self.cell = nn.GRUCell(old_input + 64, self.width)
        updated = dict(legacy_state)
        updated['cell.weight_ih'] = F.pad(legacy_state['cell.weight_ih'], (0, 64))
        self.load_state_dict(updated, strict=True)

    def forward_memory(self, q, det, mask, memory, horizon=HORIZON):
        b, length, k, d = q.shape
        if length != PREFIX or d != D or k != K:
            raise ValueError('Recipient input must be three xy frames with nine padded slots')
        if det.shape == (b, length, k, 1):
            det = det[..., 0]
        if det.shape != (b, length, k) or memory.shape != (b, k, 64):
            raise ValueError('Expected one detection channel and one 64-dimensional memory per slot')
        x = (q - self.xy_mean) / self.xy_scale
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        tokens = torch.cat([x, dx, det[..., None]], -1).permute(0, 2, 1, 3).reshape(b * k, length, 5)
        _, qh = self.query(tokens)
        qh = qh[0].reshape(b, k, 64)
        available = torch.ones(b, k, 1, device=q.device)
        hidden = self.init(torch.cat([qh, memory, available], -1))
        position = x[:, -1]
        velocity = (x[:, -1] - x[:, 0]) / (length - 1)
        pair = mask[:, :, None] * mask[:, None, :] * (1 - torch.eye(k, device=q.device)[None])
        denominator = pair.sum(2).clamp_min(1)[..., None]
        outputs = []
        for _ in range(horizon):
            hi = hidden[:, :, None].expand(b, k, k, self.width)
            hj = hidden[:, None, :].expand(b, k, k, self.width)
            relative = position[:, None] - position[:, :, None]
            relative_velocity = velocity[:, None] - velocity[:, :, None]
            messages = self.message(torch.cat([hi, hj, relative, relative_velocity], -1))
            aggregate = (messages * pair[..., None]).sum(2) / denominator
            step = torch.cat([aggregate, position, velocity, memory], -1)
            hidden = self.cell(step.reshape(b * k, -1), hidden.reshape(b * k, -1)).reshape(b, k, -1)
            movement = velocity + self.delta(hidden)
            position, velocity = position + movement, movement
            outputs.append(position * self.xy_scale + self.xy_mean)
        return torch.stack(outputs, 1)


class Model(nn.Module):
    def __init__(self, binding):
        super().__init__()
        source = torch.load(binding['source']['path'], map_location='cpu', weights_only=False)
        head = torch.load(binding['head']['path'], map_location='cpu', weights_only=False)
        self.encoder = HistoryEncoder(source['model'])
        self.head = ReuseHead(head['model'])
        mean = binding.get('code_mean', head.get('config', {}).get('code_mean'))
        scale = binding.get('code_scale', head.get('config', {}).get('code_scale'))
        if mean is None or scale is None:
            raise ValueError('Binding must contain original train-fitted Native code normalization')
        mean, scale = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)
        if mean.shape != (32,) or scale.shape != (32,) or not (scale > 0).all():
            raise ValueError('Invalid Native U32 code normalization')
        self.register_buffer('code_mean', mean)
        self.register_buffer('code_scale', scale)

    def memory(self, data, split, plan):
        indices = torch.as_tensor(plan, device=self.code_mean.device, dtype=torch.long)
        if indices.ndim != 4 or indices.shape[1] != K:
            raise ValueError('Support indices must have shape batch, padded slots, groups, histories')
        unique, inverse = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        hist = data.hist[split]
        codes = self.encoder(hist['pose'][unique], hist['presence'][unique])
        slots = torch.arange(K, device=indices.device)[None, :, None, None].expand_as(indices)
        selected = codes[inverse, slots.reshape(-1)].reshape(*indices.shape, 32)
        standardized = (selected - self.code_mean) / self.code_scale
        return self.head.support(standardized).mean(3).permute(0, 2, 1, 3)


class Data:
    def __init__(self, base, device):
        self.base = Path(base)
        self.manifest = read(self.base / 'manifest.json')
        if self.manifest.get('scene') != 'balls4' or self.manifest.get('prefix') != PREFIX:
            raise ValueError('Expected the existing three-frame Balls4 data')
        self.rows, self.hist, self.public, self.physical = {}, {}, {}, {}
        self.code_mean, self.code_scale = code_statistics(self.base, self.manifest)
        for split in ('train', 'val'):
            part = self.manifest['splits'][split]
            with np.load(self.base / f'input_{split}.npz', allow_pickle=False) as z:
                row = {'q': z['pose'].copy(), 'det': z['detected'].copy(),
                       'mask': z['presence'].copy(), 'ids': z['ids'].astype(str).tolist()}
            with np.load(self.base / f'target_{split}.npz', allow_pickle=False) as z:
                row['target'] = z['pose'].copy()
            n = len(row['ids'])
            if (row['ids'] != part['query_ids'] or row['q'].shape != (n, PREFIX, K, D)
                    or row['det'].shape != (n, PREFIX, K) or row['mask'].shape != (n, K)
                    or row['target'].shape != (n, HORIZON, K, D)):
                raise ValueError('Balls query/target shapes or row identities differ')
            if not np.isfinite(row['q']).all() or not np.isfinite(row['target']).all():
                raise ValueError('Nonfinite Balls observations or targets')
            if np.any(row['mask'].sum(1) > 4) or np.any(row['mask'].sum(1) <= 0):
                raise ValueError('Balls4 query activity count is invalid')
            expected_mask = np.asarray([part['presence'][ident] for ident in row['ids']])
            if not np.array_equal(row['mask'], expected_mask):
                raise ValueError('Balls activity metadata differ from prediction masks')
            self.rows[split] = row
            with open(part['cache'], 'rb') as f:
                cache = pickle.load(f)
            ids = part['all_ids']
            poses = np.stack([cache[ident]['pose_ab'] for ident in ids])
            presence = np.stack([cache[ident]['presence_ab'] for ident in ids])
            if poses.ndim != 4 or poses.shape[1:] != (30, K, 3) or presence.shape != (len(ids), K):
                raise ValueError('Expected complete 30-frame xyz AB visual histories')
            self.hist[split] = {
                'pose': torch.from_numpy(poses).float().to(device),
                'presence': torch.from_numpy(presence).float().to(device)}
            del cache, poses, presence
            # The generic sampler expects a public type. Balls has one known
            # type; it is added only to this in-memory manifest view.
            part['known_type'] = {ident: [[1] for _ in range(K)] for ident in row['ids']}
            self.public[split] = np.broadcast_to(np.arange(K), (n, K)).copy()
            self.physical[split] = np.asarray([part['physical'][ident] for ident in row['ids']], dtype=np.int64)
            if self.physical[split].shape != (n, K, 3):
                raise ValueError('Expected the three audited Balls attribute labels')
        self.xy_mean = self.rows['train']['q'][np.broadcast_to(
            self.rows['train']['mask'][:, None, :] > 0, self.rows['train']['q'].shape[:-1])].mean(0)

    def plan(self, split, epoch, groups, supports=3):
        row, part = self.rows[split], self.manifest['splits'][split]
        result = np.zeros((len(row['ids']), K, groups, supports), dtype=np.int64)
        index = {ident: i for i, ident in enumerate(part['all_ids'])}
        for i, ident in enumerate(row['ids']):
            seed = int.from_bytes(hashlib.sha256(
                f'xep:20260911:{split}:{ident}:{epoch}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for slot in np.flatnonzero(row['mask'][i] > 0):
                candidates = part['candidates'][ident][slot]
                if index[ident] in candidates or len(candidates) < groups * supports:
                    raise ValueError('Recipient leaked into support pool or insufficient histories')
                result[i, slot] = rng.choice(candidates, groups * supports, replace=False).reshape(groups, supports)
        return result

    def batch(self, split, indices, device):
        row = self.rows[split]
        return tuple(torch.as_tensor(np.asarray(row[key][indices], dtype=np.float32), device=device)
                     for key in ('q', 'det', 'mask', 'target'))


@torch.no_grad()
def evaluate(model, data, plan, device):
    model.eval()
    mse, fde, thirds = [], [], []
    ids = data.rows['val']['ids']
    if plan.shape[:3] != (len(ids), K, 1):
        raise ValueError('Validation requires one fixed correct support set per query object')
    for off in range(0, len(ids), 64):
        ix = np.arange(off, min(off + 64, len(ids)))
        q, det, mask, target = data.batch('val', ix, device)
        memory = model.memory(data, 'val', plan[ix])[:, 0]
        prediction = model.head.forward_memory(q, det, mask, memory)
        mse.extend(scores(prediction, target, mask).cpu().tolist())
        last = (prediction[:, -1] - target[:, -1]).norm(dim=-1)
        fde.extend(((last * mask).sum(1) / mask.sum(1)).cpu().tolist())
        thirds.extend(torch.stack([scores(prediction[:, start:start + 9], target[:, start:start + 9], mask)
                                   for start in (0, 9, 18)], 1).cpu().tolist())
    return {'mse': float(np.mean(mse)), 'fde': float(np.mean(fde)),
            'thirds': np.mean(thirds, 0).tolist(), 'per_recipient_mse': mse,
            'ids': ids, 'recipients': len(ids)}


@torch.no_grad()
def warm_start_equivalence(model, data, binding, plan, device, batch_size=8):
    """One real S5 batch: cached original Native path versus differentiable path.

    Call before any optimizer step. This has no fixed expected S5 score, since
    the old published 1.073004 score was measured with S3. The returned errors
    compare the two implementations on precisely the supplied S5 support IDs.
    """
    model.eval()
    indices = np.arange(min(batch_size, len(data.rows['val']['ids'])))
    support_ids = plan[indices, :, 0]
    q, det, mask, target = data.batch('val', indices, device)
    state = torch.load(binding['head']['path'], map_location=device, weights_only=False)
    original = PredictHead('Native-U').to(device).eval()
    original.load_state_dict(state['model'], strict=True)
    with np.load(data.base / 'codes/Native_val.npz', allow_pickle=False) as saved:
        if saved['ids'].astype(str).tolist() != data.manifest['splits']['val']['all_ids']:
            raise ValueError('Validation source code IDs differ')
        raw = saved['u'].copy()
    support = np.zeros((*support_ids.shape, 32), dtype=np.float32)
    for b, i in enumerate(indices):
        for slot in np.flatnonzero(data.rows['val']['mask'][i] > 0):
            support[b, slot] = (raw[support_ids[b, slot], slot] -
                               model.code_mean.cpu().numpy()) / model.code_scale.cpu().numpy()
    support = torch.as_tensor(support, device=device)
    expected = original(q, det, mask, support)
    memory = model.memory(data, 'val', plan[indices])[:, 0]
    actual = model.head.forward_memory(q, det, mask, memory)
    active = mask > 0
    prediction_error = (actual - expected).abs().permute(0, 2, 1, 3)[active].max().item()
    expected_memory = original.support(support).mean(2)
    memory_error = (memory - expected_memory).abs()[active].max().item()
    unique = np.unique(support_ids)[:16]
    tx = torch.as_tensor(unique, device=device)
    encoded = model.encoder(data.hist['val']['pose'][tx], data.hist['val']['presence'][tx])
    code_error = (encoded - torch.as_tensor(raw[unique], device=device)).abs().max().item()
    new_mse = float(scores(actual, target, mask).mean())
    old_mse = float(scores(expected, target, mask).mean())
    record = {'queries': len(indices), 'supports': support_ids.shape[-1],
              'source_code_max_error': code_error, 'active_memory_max_error': memory_error,
              'active_prediction_max_error': prediction_error,
              'original_batch_mse': old_mse, 'differentiable_batch_mse': new_mse,
              'zero_initialized_step_memory_weights':
                  bool(torch.count_nonzero(model.head.cell.weight_ih[:, -64:]) == 0)}
    if (code_error > 3e-5 or memory_error > 1e-4 or prediction_error > 1e-3 or
            not np.isclose(new_mse, old_mse, atol=2e-5, rtol=2e-5) or
            not record['zero_initialized_step_memory_weights']):
        raise ValueError('Balls warm-start implementation differs: ' + str(record))
    return {'status': 'PASS', **record}


def auxiliary(memory, mask, strata, physical, randomized, seed):
    """Same slot-conditioned v4.9 VICReg definition; no target values used."""
    active = mask.reshape(-1) > 0
    x = memory[:, 0].reshape(-1, 64)[active]
    y = memory[:, 1].reshape(-1, 64)[active]
    labels = np.asarray(strata).reshape(-1)[active.detach().cpu().numpy()]
    properties = np.asarray(physical).reshape(-1, 3)[active.detach().cpu().numpy()]
    order, movable = np.arange(len(labels)), 0
    rng = np.random.default_rng(seed)
    residual_x, residual_y = [], []
    for key in np.unique(labels):
        ix = np.flatnonzero(labels == key)
        if len(ix) > 1:
            movable += len(ix)
            shuffled = rng.permutation(ix)
            order[shuffled] = np.roll(shuffled, 1)
            tx = torch.as_tensor(ix, device=x.device)
            residual_x.append(x[tx] - x[tx].mean(0, keepdim=True))
            residual_y.append(y[tx] - y[tx].mean(0, keepdim=True))
    paired = y[torch.as_tensor(order, device=y.device)] if randomized else y
    invariance = F.mse_loss(x, paired)
    variance = covariance = x.sum() * 0
    if residual_x:
        rx, ry = torch.cat(residual_x), torch.cat(residual_y)
        variance = (F.relu(1 - torch.sqrt(rx.var(0, unbiased=False) + 1e-4)).mean() +
                    F.relu(1 - torch.sqrt(ry.var(0, unbiased=False) + 1e-4)).mean()) / 2
        def cov_penalty(residual):
            cov = residual.T @ residual / max(len(residual) - 1, 1)
            return (cov.square().sum() - cov.diagonal().square().sum()) / residual.shape[1]
        covariance = (cov_penalty(rx) + cov_penalty(ry)) / 2
    moved = order != np.arange(len(order))
    accidental = float(np.mean(np.all(properties[moved] == properties[order[moved]], axis=-1))) if moved.any() else 0.
    metrics = {'inv': float(invariance.detach()), 'var': float(variance.detach()),
               'cov': float(covariance.detach()), 'random_unpermutable': 1 - movable / max(len(labels), 1),
               'random_accidental_same': accidental, 'aux_objects': len(labels)}
    return invariance + variance + .04 * covariance, metrics


# Explicit compatibility names for a new Balls driver; no monkeypatching.
FineTuneData, FineTuneModel = Data, Model
