"""Frozen source-formation encoders: common fresh heads and physical probes.

Native is the original source-only model, without independent-donor source
training. Four v5.1 source models are frozen before all downstream training.
Each S=3/5/8 gets its own randomly initialized original prediction head, with
identical initialization, queries, support plans and 100-epoch budget across
methods. No MQ weights or fine-tuned encoders are imported.
"""

import os
import argparse
import fcntl
import hashlib
import importlib
import json
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import xep_discovery as balls
import collision_xep as collision

VERSION = 'source-formation-frozen-readout-v5.1-1'
METHODS = ('Native', 'Cross-only', 'Align-only', 'Both-new', 'Random-Both-new')
SUPPORTS = (3, 5, 8)
BUDGETS = (20, 60, 100)
read, write, digest = balls.read, balls.write, balls.digest
save_npz, save_torch, emit = balls.save_npz, balls.save_torch, balls.emit


def module_for(scene):
    if scene not in ('balls', 'collision'):
        raise ValueError('Unknown scene')
    return balls if scene == 'balls' else collision


def resolve(path, base):
    path = Path(path)
    if path.exists():
        return path
    prefix = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
    try:
        alternative = Path(base).parent / path.relative_to(prefix)
    except ValueError:
        return path
    return alternative if alternative.exists() else path


def immutable_write(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Existing output bound to different inputs: ' + str(path))
    write(path, value)


def set_runtime_source(base):
    path = Path(base).parent / 'source'
    if not (path / 'cophy_adapter.py').is_file():
        raise FileNotFoundError('Source runtime package unavailable: ' + str(path))
    sys.path.insert(0, str(path))
    return path


def prepare(scene, source_root, base, out):
    base, source_root, out = map(Path, (base, source_root, out))
    out.mkdir(parents=True, exist_ok=True)
    original = read(base / 'manifest.json')
    expected = 'balls4' if scene == 'balls' else 'collision-normal'
    if original['scene'] != expected or original['test_read'] is not False or original['prefix'] != 3:
        raise ValueError('Expected existing independent-experience train/validation task')
    if len(original['splits']['val']['query_ids']) != 512:
        raise ValueError('Readout requires original512 validation queue')
    runtime = set_runtime_source(base)
    sources, files = {}, [base / 'manifest.json', Path(__file__), Path(balls.__file__), Path(collision.__file__)]
    sources['Native'] = dict(original['source_models']['Native'])
    native_path = resolve(sources['Native']['path'], base)
    if digest(native_path) != sources['Native']['sha256']:
        raise ValueError('Original Native source checkpoint differs')
    sources['Native']['path'] = str(native_path)
    native = torch.load(native_path, map_location='cpu', weights_only=False)
    if native['run_config']['method'] != 'Native':
        raise ValueError('Native must be the original no-donor source model')
    sources['Native']['selected_epoch'] = native['epoch']
    files.append(native_path)
    del native
    for method in METHODS[1:]:
        run = source_root / 'runs' / method
        path = run / 'selected.pt'
        complete = read(run / 'complete.json')
        if complete.get('epochs', complete.get('epoch')) != 50:
            raise ValueError('Source model has not completed50: ' + method)
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = state['config']
        if config['method'] != method or config.get('source_pretrained_checkpoint') is not None:
            raise ValueError('Source must be the specified from-scratch run: ' + method)
        if config.get('phase') != 'source_formation' or config.get('visual_frozen') is not True:
            raise ValueError('Unexpected source phase or visual training status')
        receipt = read(run / 'selected_validation.json')
        if receipt['epoch'] != state['epoch']:
            raise ValueError('Source checkpoint/selection receipt epoch differs')
        sources[method] = {'path': str(path), 'sha256': digest(path), 'selected_epoch': state['epoch'],
            'config': config, 'selection_receipt_sha256': digest(run / 'selected_validation.json')}
        files.extend([path, run / 'complete.json', run / 'selected_validation.json'])
        del state
    splits = {}
    for split in ('train', 'val'):
        part = original['splits'][split]
        cache = resolve(part['cache'], base)
        if digest(cache) != part['cache_sha256']:
            raise ValueError('AB visual cache differs')
        minimum = min(len(options) for lists in part['candidates'].values() for options in lists if options)
        if minimum < 8:
            raise ValueError('Cannot cover unchanged query cohort at S8')
        for name in ('input', 'target'):
            files.append(base / f'{name}_{split}.npz')
        files.append(cache)
        splits[split] = {'cache': str(cache), 'queries': len(part['query_ids']), 'minimum_candidate_pool': minimum}
    binding = {'version': VERSION, 'scene': scene, 'base': str(base), 'source_root': str(source_root),
        'test_read': False, 'sources': sources, 'splits': splits, 'runtime_source': str(runtime),
        'slots': 9 if scene == 'balls' else 4, 'dims': 2 if scene == 'balls' else 3,
        'horizon': 27 if scene == 'balls' else 12, 'supports': list(SUPPORTS), 'head_budget': 100,
        'head_architecture': 'original xep PredictHead, Native-U width128; no per-step memory augmentation',
        'source_encoder_frozen': True, 'source_code_views': {'P': [0, 16], 'T': [16, 32], 'U': [0, 32]},
        'file_sha256': {str(path): digest(path) for path in files}}
    immutable_write(out / 'binding.json', binding)
    for directory in ('codes', 'plans', 'probes', 'runs'):
        (out / directory).mkdir(exist_ok=True)
    plan8(out, binding, 'val', 0)
    emit('readout_prepared', scene=scene, sources={m: e['selected_epoch'] for m, e in sources.items()})


def binding_for(out, scene=None):
    binding = read(Path(out) / 'binding.json')
    if binding['version'] != VERSION or (scene is not None and binding['scene'] != scene):
        raise ValueError('Readout binding version/scene differs')
    for path, expected in binding['file_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen readout dependency changed: ' + path)
    sys.path.insert(0, binding['runtime_source'])
    return binding


def plan8(out, binding, split, epoch):
    """One common nested plan per epoch, cached and reused by all methods/S."""
    out = Path(out)
    path = out / 'plans' / f'{split}_{epoch:03d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.lock'), 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = read(Path(binding['base']) / 'manifest.json')
        part = original['splits'][split]
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                if stored['ids'].tolist() != part['query_ids']:
                    raise ValueError('Stored plan query order differs')
                return stored['plan'].copy()
        lookup = {ident: i for i, ident in enumerate(part['all_ids'])}
        plan = np.zeros((len(part['query_ids']), binding['slots'], 8), np.int64)
        for i, ident in enumerate(part['query_ids']):
            seed = int.from_bytes(hashlib.sha256(f'xep:20260911:{split}:{ident}:{epoch}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for slot in range(binding['slots']):
                if part['presence'][ident][slot] <= 0:
                    continue
                options = part['candidates'][ident][slot]
                if len(options) < 8 or lookup[ident] in options:
                    raise ValueError('Invalid independent S8 donor pool')
                first = rng.choice(options, 3, replace=False)
                rest = [v for v in options if v not in set(first.tolist())]
                text = f'source-formation-readout-v51:{split}:{ident}:{slot}:{epoch}'
                extra_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little'))
                plan[i, slot] = np.concatenate([first, extra_rng.choice(rest, 5, replace=False)])
        save_npz(path, plan=plan, ids=np.asarray(part['query_ids']))
        return plan


@torch.no_grad()
def encode(scene, out, method, device):
    out = Path(out); binding = binding_for(out, scene)
    from cophy_adapter import PTCoPhy, ABObservation
    from cf_learning.model import CoPhyNet
    source = binding['sources'][method]
    marker = out / 'codes' / (method + '_complete.json')
    if marker.exists():
        saved = read(marker)
        if saved['source_sha256'] != source['sha256'] or any(digest(path) != sha for path, sha in saved['file_sha256'].items()):
            raise ValueError('Encoded source cache changed')
        return
    torch.set_num_threads(4); torch.manual_seed(0)
    state = torch.load(source['path'], map_location='cpu', weights_only=False)
    model = PTCoPhy(CoPhyNet(binding['slots']), 'Native').to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del state
    original = read(Path(binding['base']) / 'manifest.json')
    artifacts = {}
    for split in ('train', 'val'):
        ids = original['splits'][split]['all_ids']
        with open(binding['splits'][split]['cache'], 'rb') as stream:
            cache = pickle.load(stream)
        codes, presence = [], []
        for off in range(0, len(ids), 128):
            chosen = ids[off:off + 128]
            pose = torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in chosen])).float().to(device)
            mask = torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in chosen])).float().to(device)
            codes.append(model.encode_ab(ABObservation(pose, mask)).cpu().numpy())
            presence.append(mask.cpu().numpy())
        target = out / 'codes' / f'{method}_{split}.npz'
        save_npz(target, ids=np.asarray(ids), u=np.concatenate(codes), presence=np.concatenate(presence))
        artifacts[str(target)] = digest(target)
        del cache
        emit('source_codes_frozen', scene=scene, method=method, split=split, episodes=len(ids))
    write(marker, {'status': 'COMPLETE', 'source_sha256': source['sha256'], 'file_sha256': artifacts, 'test_read': False})


class ReadoutData:
    def __init__(self, out, binding, method, supports):
        self.out, self.binding, self.method, self.supports = Path(out), binding, method, supports
        self.manifest = read(Path(binding['base']) / 'manifest.json')
        self.data = {}; self.train_epoch = None; self.train_plan = None
        marker = read(self.out / 'codes' / (method + '_complete.json'))
        if marker['source_sha256'] != binding['sources'][method]['sha256']:
            raise ValueError('Wrong frozen encoder codes')
        for path, sha in marker['file_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Frozen encoder codes changed')
        mean = scale = None
        for split in ('train', 'val'):
            base = Path(binding['base'])
            with np.load(base / f'input_{split}.npz', allow_pickle=False) as x:
                row = {'ids': x['ids'].tolist(), 'q': x['pose'].copy(), 'det': x['detected'].copy(), 'mask': x['presence'].copy()}
            with np.load(base / f'target_{split}.npz', allow_pickle=False) as y:
                row['target'] = y['pose'].copy()
            with np.load(self.out / 'codes' / f'{method}_{split}.npz', allow_pickle=False) as c:
                if c['ids'].tolist() != self.manifest['splits'][split]['all_ids']:
                    raise ValueError('Code order differs')
                codes = c['u'].copy()
                if split == 'train':
                    observed = codes[c['presence'] > 0]
                    mean, scale = observed.mean(0), observed.std(0)
                    scale[scale < 1e-6] = 1.
            if row['ids'] != self.manifest['splits'][split]['query_ids']:
                raise ValueError('Readout query order differs')
            row['codes'] = (codes - mean) / scale
            self.data[split] = row
        train = self.data['train']
        xy = train['q'][np.broadcast_to(train['mask'][:, None, :] > 0, train['q'].shape[:-1])]
        self.xy_mean, self.xy_scale = xy.mean(0), xy.std(0).clip(.1)
        self.code_mean, self.code_scale = mean, scale
        self.val_plan = plan8(out, binding, 'val', 0)[..., :supports]

    def batch(self, split, indices, epoch, device):
        if split == 'train':
            if self.train_epoch != epoch:
                self.train_plan = plan8(self.out, self.binding, split, epoch)[..., :self.supports]
                self.train_epoch = epoch
            plan = self.train_plan[indices]
        else:
            plan = self.val_plan[indices]
        row = self.data[split]
        slots = np.arange(self.binding['slots'])[None, :, None]
        support = row['codes'][plan, slots] * row['mask'][indices, :, None, None]
        tensor = lambda value: torch.from_numpy(np.asarray(value, np.float32)).to(device)
        return tuple(tensor(value) for value in (row['q'][indices], row['det'][indices], row['mask'][indices], support, row['target'][indices]))


def state_hash(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(scene, out, method, supports, epochs, device):
    out = Path(out); binding = binding_for(out, scene); module = module_for(scene)
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    data = ReadoutData(out, binding, method, supports)
    model = module.PredictHead('Native-U').to(device)
    model.xy_mean.copy_(torch.tensor(data.xy_mean, device=device)); model.xy_scale.copy_(torch.tensor(data.xy_scale, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    folder = out / 'runs' / f'S{supports}' / method; folder.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'scene': scene, 'method': method, 'supports': supports, 'seed': 0,
        'binding_sha256': digest(out / 'binding.json'), 'source_sha256': binding['sources'][method]['sha256'],
        'code_cache_sha256': read(out / 'codes' / (method + '_complete.json'))['file_sha256'],
        'initial_head_sha256': state_hash(model.state_dict()), 'lr': 3e-4, 'weight_decay': 1e-4,
        'batch_size': 128, 'gradient_clip': 1., 'maximum_epochs': 100, 'source_encoder_frozen': True,
        'new_random_head': True, 'uses_MQ_head': False, 'test_read': False,
        'code_mean': data.code_mean.tolist(), 'code_scale': data.code_scale.tolist()}
    immutable_write(folder / 'config.json', config)
    start, best, history = 0, float('inf'), []
    if (folder / 'latest.pt').exists():
        checkpoint = torch.load(folder / 'latest.pt', map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Cannot resume a different readout run')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, history = checkpoint['epoch'], checkpoint['best'], checkpoint['history']
    for epoch in range(start + 1, epochs + 1):
        began = time.perf_counter(); model.train()
        order = np.random.default_rng(771 + epoch).permutation(len(data.data['train']['ids']))
        total = 0.; count = 0
        for off in range(0, len(order), 128):
            indices = order[off:off + 128]
            q, det, mask, support, target = data.batch('train', indices, epoch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = module.scores(model(q, det, mask, support), target, mask).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite readout loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach()) * len(indices); count += len(indices)
        metric = module.evaluate(model, data, device); metric['ids'] = data.data['val']['ids']
        record = {'epoch': epoch, 'train_mse': total / count, 'seconds': time.perf_counter() - began,
            'plan_sha256': hashlib.sha256(data.train_plan.tobytes()).hexdigest(),
            **{key: value for key, value in metric.items() if key not in ('ids', 'per_recipient_mse')}}
        history.append(record)
        if metric['mse'] < best:
            best = metric['mse']
            save_torch(folder / 'selected.pt', {'model': model.state_dict(), 'epoch': epoch, 'config': config})
            write(folder / 'selected_validation.json', {'method': method, 'supports': supports, 'epoch': epoch, **metric})
        save_torch(folder / 'latest.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'best': best, 'history': history, 'config': config})
        write(folder / 'progress.json', {'status': 'RUNNING', 'method': method, 'supports': supports,
            'epoch': epoch, 'best_mse': best, 'history': history, 'test_read': False})
        if epoch in BUDGETS:
            selected = read(folder / 'selected_validation.json')
            write(folder / f'budget_{epoch}.json', {'budget': epoch, 'test_read': False, **selected})
            shutil.copyfile(folder / 'selected.pt', folder / f'selected_{epoch}.pt')
        emit('frozen_readout_epoch', scene=scene, method=method, supports=supports, **record)
    write(folder / 'complete.json', {'status': 'COMPLETE', 'method': method, 'supports': supports,
        'epochs': epochs, 'best_mse': best, 'test_read': False})


def ridge(x, target, evaluation):
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-8] = 1.
    z, e = (x - mean) / scale, (evaluation - mean) / scale
    center = target.mean(0)
    weight = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ (target - center))
    return e @ weight + center


def classification(actual, guessed, classes):
    matrix = np.zeros((classes, classes), np.int64); np.add.at(matrix, (actual, guessed), 1)
    counts = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), counts, out=np.zeros(classes), where=counts > 0)
    return {'accuracy': float((actual == guessed).mean()), 'balanced_accuracy': float(recall[counts > 0].mean()),
        'class_support': counts.tolist(), 'confusion': matrix.tolist(), 'examples': len(actual)}


def numeric_metric(actual, guessed):
    mse = float(np.square(actual - guessed).mean()); variance = float(actual.var())
    return {'r2': 1 - mse / variance if variance > 0 else None, 'mse': mse, 'variance': variance}


def probe(scene, out, method):
    out = Path(out); binding = binding_for(out, scene)
    from cophy_fields import PLANS
    base = Path(binding['base']); original = read(base / 'manifest.json')
    prepath = base.parent / 'prepared_v3' / scene / 'training_preflight.json'
    preflight = read(prepath)
    if digest(prepath) != original['preflight_sha256']:
        raise ValueError('Probe field audit differs')
    spec = PLANS[scene]; fields, levels = spec['fields'], spec['support']
    arrays, provenance = {}, {}
    numeric_available = True
    for split in ('train', 'val'):
        item = preflight['artifacts']['raw_relations_' + split]
        path = resolve(item['path'], base)
        if digest(path) != item['sha256']:
            raise ValueError('Probe raw relation artifact changed')
        provenance[split] = {'path': str(path), 'sha256': item['sha256']}
        raw = {(r['id'], r['slot']): r for r in read(path)}
        part = original['splits'][split]
        with np.load(out / 'codes' / f'{method}_{split}.npz', allow_pickle=False) as saved:
            ids = saved['ids'].tolist(); codes = saved['u'].copy(); visible = saved['presence'].copy()
        if ids != part['all_ids']:
            raise ValueError('Probe source code order differs')
        index = {ident: i for i, ident in enumerate(ids)}
        x, labels, numeric, public, keys = [], [], [], [], []
        for ident in part['query_ids']:
            for slot in np.flatnonzero(np.asarray(part['presence'][ident]) > 0):
                row = raw[(ident, int(slot))]
                if row['split'] != split or not row['in_C'] or row['physical'] != part['physical'][ident][slot]:
                    raise ValueError('Active query labels differ from audited source')
                if visible[index[ident], slot] <= 0:
                    raise ValueError('Probe recipient object not visible in its AB')
                values = row.get('raw_physical')
                if values is None or len(values) != len(fields):
                    numeric_available = False; values = [0.] * len(fields)
                else:
                    for j, value in enumerate(values):
                        matches = [k for k, level in enumerate(levels[j]) if abs(value - level) < .01]
                        if not np.isfinite(value) or matches != [row['physical'][j]]:
                            raise ValueError('Physical category/raw-value mapping failed')
                x.append(codes[index[ident], slot]); labels.append(row['physical']); numeric.append(values)
                public.append((int(slot), row['known_type'])); keys.append((ident, int(slot)))
        arrays[split] = {'x': np.asarray(x, np.float64), 'labels': np.asarray(labels, np.int64),
            'numeric': np.asarray(numeric, np.float64), 'public': public, 'keys': keys}
    tr, va = arrays['train'], arrays['val']
    if set(ident for ident, _ in tr['keys']) & set(ident for ident, _ in va['keys']):
        raise ValueError('Probe experiment split overlap')
    counts = [len(level) for level in levels]
    onehot = np.concatenate([np.eye(n)[tr['labels'][:, j]] for j, n in enumerate(counts)], 1)
    target = np.concatenate([onehot, tr['numeric']], 1) if numeric_available else onehot
    groups = defaultdict(list)
    for i, key in enumerate(tr['public']):
        groups[key].append(i)
    means = {key: target[indices].mean(0) for key, indices in groups.items()}
    prior = np.stack([means.get(key, target.mean(0)) for key in va['public']])
    predictions = {name: ridge(tr['x'][:, left:right], target, va['x'][:, left:right])
                   for name, (left, right) in binding['source_code_views'].items()}
    report = {'status': 'COMPLETE', 'scene': scene, 'method': method, 'test_read': False,
        'source_sha256': binding['sources'][method]['sha256'], 'probe_alpha': 1.,
        'probe_normalization': 'train active objects only; intercept unpenalized; no val hyperparameter selection',
        'numeric_r2_available': numeric_available, 'numeric_target': 'raw_physical values, never category IDs',
        'raw_metadata': provenance, 'fields': fields, 'nominal_levels': levels,
        'train_objects': len(tr['x']), 'val_objects': len(va['x']), 'validation_recipients': 512,
        'representations': {}, 'public_slot_type_prior': {'fields': {}},
        'interpretation': 'Frozen source AB codes, before new-head training; P16/T16/U32 linear accessibility does not itself prove causal predictive use'}
    for view, prediction in {**predictions, 'public_slot_type_prior': prior}.items():
        result = {'fields': {}}; start = 0
        for j, (field, count) in enumerate(zip(fields, counts)):
            entry = classification(va['labels'][:, j], prediction[:, start:start + count].argmax(1), count)
            entry['train_class_support'] = np.bincount(tr['labels'][:, j], minlength=count).tolist()
            if numeric_available:
                entry['numeric_regression'] = numeric_metric(va['numeric'][:, j], prediction[:, sum(counts) + j])
            result['fields'][field] = entry; start += count
        if view == 'public_slot_type_prior':
            report[view] = result
        else:
            report['representations'][view] = result
    write(out / 'probes' / f'{method}.json', report)
    emit('source_probe_complete', scene=scene, method=method)


def summarize(scene, out):
    out = Path(out); binding = binding_for(out, scene)
    result = {'version': VERSION, 'status': 'COMPLETE', 'scene': scene, 'test_read': False,
        'sources': binding['sources'], 'supports': {}, 'probes': {},
        'interpretation': 'Common frozen U32 readout heads; source Native has no independent-donor training; P/T probes are separate representation diagnostics'}
    for supports in SUPPORTS:
        methods = {}; initial = None; ids = None; plans = None
        for method in METHODS:
            folder = out / 'runs' / f'S{supports}' / method
            complete, config = read(folder / 'complete.json'), read(folder / 'config.json')
            if complete['epochs'] != 100 or complete['test_read'] is not False:
                raise ValueError('Readout not complete at100')
            if initial is None:
                initial = config['initial_head_sha256']
            elif config['initial_head_sha256'] != initial:
                raise ValueError('Initial heads differ by source method')
            progress = read(folder / 'progress.json')
            hashes = [r['plan_sha256'] for r in progress['history']]
            if len(hashes) != 100:
                raise ValueError('Missing readout epochs')
            if plans is None: plans = hashes
            elif hashes != plans: raise ValueError('Training support plans differ by source method')
            selected = read(folder / 'selected_validation.json')
            if ids is None: ids = selected['ids']
            elif selected['ids'] != ids: raise ValueError('Selected validation cohorts differ')
            methods[method] = {k: v for k, v in selected.items() if k not in ('ids', 'per_recipient_mse')}
            methods[method]['budget_mse'] = {str(b): read(folder / f'budget_{b}.json')['mse'] for b in BUDGETS}
        comparisons = {method: {'improvement_over_native_percent': 100 * (methods['Native']['mse'] - value['mse']) / methods['Native']['mse']}
                       for method, value in methods.items() if method != 'Native'}
        comparisons['Both-new']['improvement_over_random_percent'] = 100 * (methods['Random-Both-new']['mse'] - methods['Both-new']['mse']) / methods['Random-Both-new']['mse']
        result['supports'][str(supports)] = {'methods': methods, 'comparisons': comparisons}
    for method in METHODS:
        report = read(out / 'probes' / f'{method}.json')
        if report['status'] != 'COMPLETE' or report['test_read'] is not False or set(report['representations']) != {'P', 'T', 'U'}:
            raise ValueError('Formation probe incomplete')
        result['probes'][method] = report
    write(out / 'summary.json', result)
    emit('formation_readout_complete', scene=scene, out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'encode', 'cache', 'train', 'probe', 'summary'])
    parser.add_argument('--scene', choices=['balls', 'collision'], required=True)
    parser.add_argument('--source-root', '--source-run', dest='source_root')
    parser.add_argument('--base'); parser.add_argument('--out', required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--supports', type=int, choices=SUPPORTS, default=3)
    parser.add_argument('--epochs', type=int, choices=BUDGETS, default=100)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.command == 'train': lockname = f'train_S{args.supports}_{args.method}.lock'
    elif args.command in ('encode', 'cache', 'probe'): lockname = f'{args.command}_{args.method}.lock'
    else: lockname = args.command + '.lock'
    with open(Path(args.out) / lockname, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            if not args.base or not args.source_root: raise ValueError('prepare needs --base and --source-root')
            prepare(args.scene, args.source_root, args.base, args.out)
        elif args.command in ('encode', 'cache'):
            encode(args.scene, args.out, args.method, args.device)
        elif args.command == 'train':
            train(args.scene, args.out, args.method, args.supports, args.epochs, args.device)
        elif args.command == 'probe':
            probe(args.scene, args.out, args.method)
        else:
            summarize(args.scene, args.out)
