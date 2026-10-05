"""Source AB+C -> D formation with P16-only independent-history objectives.

Source task networks are initialized from seed 0; only the already-qualified
visual frontend is loaded and frozen. Target packing reads train/val CD states
once on the machine holding raw files, then all workers use a shared data pack.
No new-task prefix/target, trained source model, or test example is consumed.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch


VERSION = 'source-formation-v5.1-1'
DATA_VERSION = 'source-ab-c-full-d-cache-v5.1-1'
ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
METHODS = ('Cross-only', 'Align-only', 'Both-new', 'Random-Both-new')
SUPPORTS, MAX_QUERIES, BATCH_GROUPS = 5, 3, 10
SOURCE_FILES = ('cf_learning/model.py', 'cophy_adapter.py', 'cophy_protocol.py',
                'cophy_relations.py', 'cophy_training.py', 'dataloaders/utils.py',
                'derendering/model.py')
SPECS = {'balls': {'slots': 9, 'frames': 30, 'dims': 2, 't_delta': 2, 'folder': 'ballsCF'},
         'collision': {'slots': 4, 'frames': 15, 'dims': 3, 't_delta': 5, 'folder': 'collisionCF'}}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(tmp, 'wb') as stream:
        np.savez(stream, **values)
    os.replace(tmp, path)


def emit(event, **values):
    print(json.dumps({'time': time.time(), 'event': event, **values}, allow_nan=False), flush=True)


def api(root):
    import sys
    source = str(Path(root) / 'source')
    if source not in sys.path:
        sys.path.insert(0, source)
    return {'adapter': importlib.import_module('cophy_adapter'),
            'model': importlib.import_module('cf_learning.model'),
            'training': importlib.import_module('cophy_training'),
            'protocol': importlib.import_module('cophy_protocol'),
            'utils': importlib.import_module('dataloaders.utils')}


def artifact(preflight, name):
    item = preflight['artifacts'][name]
    path = Path(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError('Audited artifact changed: ' + name)
    return path


def validate_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError('Bound file changed: ' + path)


def state_digest(model, trainable_only=False):
    values = dict(model.named_parameters()) if trainable_only else model.state_dict()
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        if trainable_only and not value.requires_grad:
            continue
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def prepare_data(root, scene):
    """Pack complete source targets; reuse the immutable pack on other hosts."""
    root = Path(root)
    spec = SPECS[scene]
    folder = root / 'source_formation_v51_data' / scene
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / 'prepare.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = folder / 'manifest.json'
        if receipt_path.exists():
            saved = read(receipt_path)
            if saved.get('version') != DATA_VERSION or saved.get('scene') != scene:
                raise ValueError('Shared source cache belongs to another data contract')
            validate_files(saved['files'])
            return saved, receipt_path
        prepath = root / 'prepared_v3' / scene / 'training_preflight.json'
        preflight = read(prepath)
        if (preflight.get('scene') != scene or preflight.get('status') != 'PASS' or
                preflight.get('adapter_version') != 'cophy-pt16-v3' or preflight.get('test_read') is not False):
            raise ValueError('Missing qualified train/val PT16 source data')
        split_path = artifact(preflight, 'splits')
        split_doc = read(split_path)
        relation_path = artifact(preflight, 'relation_index')
        derenderer = artifact(preflight, 'derenderer')
        raw_root = Path(preflight['input_profile']['dataset_dir'])
        if scene == 'balls':
            raw_root = raw_root / str(preflight['input_profile']['num_objects'])
        utils = api(root)['utils']
        files = {str(prepath): digest(prepath), str(split_path): digest(split_path),
                 str(relation_path): digest(relation_path), str(derenderer): digest(derenderer),
                 str(root / 'source/dataloaders/utils.py'): digest(root / 'source/dataloaders/utils.py')}
        splits = {}
        for split in ('train', 'val'):
            ids = list(map(str, split_doc[split]['ids']))
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicated official source IDs')
            cache_path = artifact(preflight, 'cache_' + split)
            with open(cache_path, 'rb') as stream:
                cache = pickle.load(stream)
            if set(cache) != set(ids):
                raise ValueError('Visual cache does not cover the exact official split')
            columns = {name: [] for name in ('pose_ab', 'presence_ab', 'pose_c', 'presence_c',
                                            'target_pose', 'target_stationary', 'target_presence')}
            target_hash = hashlib.sha256()
            for ident in ids:
                item = cache[ident]
                if item.get('cache_version') != 'ab_c_float32_v2':
                    raise ValueError('Legacy future-bearing visual cache is forbidden')
                for name, shape in (('pose_ab', (spec['frames'], spec['slots'], 3)),
                                    ('presence_ab', (spec['slots'],)), ('pose_c', (1, spec['slots'], 3)),
                                    ('presence_c', (spec['slots'],))):
                    value = np.asarray(item[name], dtype=np.float32)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError('Invalid visual cache ' + ident + '/' + name)
                    if name.startswith('presence') and not np.isin(value, (0., 1.)).all():
                        raise ValueError('Nonbinary visual presence')
                    columns[name].append(value)
                state_path = raw_root / ident / 'cd/states.npy'
                if not state_path.exists():
                    raise FileNotFoundError('Complete source CD targets unavailable. Run prepare once on '
                                            'the raw-data host; do not substitute truncated XEP targets: ' + str(state_path))
                states = np.load(state_path, allow_pickle=False)
                pose = np.asarray(states[:, :, :3], dtype=np.float32)
                if pose.shape != (spec['frames'], spec['slots'], 3) or not np.isfinite(pose).all():
                    raise ValueError('Source CD target shape/value mismatch: ' + ident)
                presence = (np.abs(states[0, :, :3]).sum(-1) > 0).astype(np.float32)
                stationary = utils.get_stab(pose, presence, t_delta=spec['t_delta'], eps=.05)
                columns['target_pose'].append(pose[1:])
                columns['target_stationary'].append(stationary[1:])
                columns['target_presence'].append(presence)
                target_hash.update(ident.encode())
                target_hash.update(pose.tobytes())
            path = folder / (split + '.npz')
            save_npz(path, ids=np.asarray(ids), **{name: np.stack(value) for name, value in columns.items()})
            files.update({str(cache_path): digest(cache_path), str(path): digest(path)})
            splits[split] = {'path': str(path), 'sha256': digest(path), 'count': len(ids),
                             'target_xyz_sequence_sha256': target_hash.hexdigest(),
                             'visual_cache': str(cache_path)}
            emit('source_data_packed', scene=scene, split=split, count=len(ids))
            del columns, cache
        if set(split_doc['train']['ids']) & set(split_doc['val']['ids']):
            raise ValueError('Train and validation overlap')
        receipt = {'version': DATA_VERSION, 'scene': scene, 'test_read': False,
                   'preflight': str(prepath), 'relation_index': str(relation_path),
                   'derenderer': str(derenderer), 'spec': spec, 'splits': splits,
                   'files': files, 'target_rule': 'complete xyz CD[1:], original stationary labels',
                   'input_rule': 'full visual AB plus exactly one visual C frame'}
        write(receipt_path, receipt)
        return receipt, receipt_path


class SourceData:
    def __init__(self, packed, device):
        self.packed = packed
        self.rows, self.ids = {}, {}
        for split in ('train', 'val'):
            with np.load(packed['splits'][split]['path'], allow_pickle=False) as data:
                self.ids[split] = data['ids'].astype(str).tolist()
                self.rows[split] = {name: torch.as_tensor(data[name].copy(), dtype=torch.float32, device=device)
                                    for name in data.files if name != 'ids'}
        self.relations = read(packed['relation_index'])

    def visual(self, indices, split, module):
        row = self.rows[split]
        return module.VisualInput(module.ABObservation(row['pose_ab'][indices], row['presence_ab'][indices]),
                                  row['pose_c'][indices], row['presence_c'][indices])

    def target(self, indices, split, module):
        row = self.rows[split]
        return module.Targets(row['target_pose'][indices], row['target_stationary'][indices],
                              row['target_presence'][indices])


class SourceSampler:
    """One native exposure per train episode, ≤3 queries per shared focal P."""
    def __init__(self, data):
        ids = data.ids['train']
        self.ids = ids
        self.index = {ident: i for i, ident in enumerate(ids)}
        document = data.relations
        if document.get('split') != 'train' or document.get('scene') != data.packed['scene']:
            raise ValueError('Relation index is not this source training split')
        self.recipients, self.donors = defaultdict(list), defaultdict(list)
        seen = set()
        for record in document['records']:
            ident, slot = str(record['id']), int(record['slot'])
            if ident not in self.index or (ident, slot) in seen:
                raise ValueError('Unknown or duplicated relation object')
            seen.add((ident, slot))
            public = (slot, canonical(record['stratum']))
            key = public + (canonical(record['physical']),)
            if record['donor']:
                self.donors[key].append(self.index[ident])
            if record['recipient']:
                self.recipients[self.index[ident]].append(key)
        for key, pool in self.donors.items():
            self.donors[key] = np.asarray(sorted(set(pool)), dtype=np.int64)

    def make_epoch(self, epoch):
        seed = int.from_bytes(hashlib.sha256(f'{VERSION}:source-plan:seed0:{epoch}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        assigned, native_only = defaultdict(list), []
        for index in range(len(self.ids)):
            eligible = [key for key in self.recipients[index]
                        if len(self.donors.get(key, [])) - int(index in self.donors.get(key, [])) >= SUPPORTS]
            if eligible:
                assigned[eligible[int(rng.integers(len(eligible)))]].append(index)
            else:
                native_only.append(index)
        groups = []
        for key, indices in sorted(assigned.items()):
            rng.shuffle(indices)
            for offset in range(0, len(indices), MAX_QUERIES):
                query = indices[offset:offset + MAX_QUERIES]
                pool = self.donors[key]
                pool = pool[~np.isin(pool, query)]
                valid = len(pool) >= SUPPORTS
                supports = rng.choice(pool, SUPPORTS, replace=False).tolist() if valid else [0] * SUPPORTS
                groups.append({'query': query, 'focal': key[0], 'public': key[:2],
                               'physical': key[2], 'correct': supports, 'valid': valid})
        for index in native_only:
            groups.append({'query': [index], 'focal': -1, 'public': (-1, ''),
                           'physical': '', 'correct': [0] * SUPPORTS, 'valid': False})
        groups = [groups[int(i)] for i in rng.permutation(len(groups))]
        buckets = defaultdict(list)
        for i, group in enumerate(groups):
            if group['valid']:
                # Matching group size preserves both donor-group marginals and
                # the number of recipients served by each donor segment.
                buckets[(group['public'], len(group['query']))].append(i)
        mapping = np.arange(len(groups), dtype=np.int64)
        skipped = []
        for bucket in buckets.values():
            permutation = None
            if len(bucket) > 1:
                for _ in range(512):
                    order = rng.permutation(bucket)
                    candidate = np.roll(order, 1)
                    if all(not set(groups[int(dst)]['query']) & set(groups[int(src)]['correct'])
                           for dst, src in zip(order, candidate)):
                        permutation = (order, candidate)
                        break
            if permutation is None:
                for i in bucket:
                    groups[i]['valid'] = False
                    skipped.extend(groups[i]['query'])
            else:
                mapping[permutation[0]] = permutation[1]
        offsets = np.r_[0, np.cumsum([len(group['query']) for group in groups])].astype(np.int64)
        query = np.asarray([i for group in groups for i in group['query']], dtype=np.int64)
        if sorted(query.tolist()) != list(range(len(self.ids))):
            raise ValueError('Source epoch must cover every train query exactly once, without padding')
        valid = np.asarray([group['valid'] for group in groups], dtype=bool)
        correct = np.asarray([group['correct'] for group in groups], dtype=np.int64)
        accidental, paired = 0, 0
        for i in np.flatnonzero(valid):
            group, donor = groups[int(i)], groups[int(mapping[i])]
            if i == mapping[i] or set(group['query']) & set(donor['correct']):
                raise ValueError('Random donor assignment retained self group/episode')
            if donor['public'] != group['public'] or len(donor['query']) != len(group['query']):
                raise ValueError('Random assignment changed public or exposure strata')
            paired += len(group['query'])
            accidental += len(group['query']) * int(group['physical'] == donor['physical'])
        if sorted(mapping[valid].tolist()) != np.flatnonzero(valid).tolist():
            raise ValueError('Random assignment failed donor-group marginal preservation')
        arrays = {'query_indices': query, 'group_offsets': offsets,
                  'focal': np.asarray([group['focal'] for group in groups], dtype=np.int64),
                  'correct': correct, 'random_source_group': mapping, 'aux_valid': valid}
        h = hashlib.sha256(f'{VERSION}:{epoch}'.encode())
        for name, value in arrays.items():
            h.update(name.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
        summary = {'epoch': epoch, 'groups': len(groups), 'query_exposures': len(query),
                   'unique_queries': len(query), 'padding': 0, 'paired_recipient_exposures': paired,
                   'unpaired_native_only': len(query) - paired,
                   'randomization_skipped_recipients': len(skipped),
                   'random_accidental_same_recipients': accidental,
                   'random_accidental_same_fraction': accidental / max(paired, 1),
                   'donor_group_multiset_preserved': True, 'donor_served_counts_preserved': True,
                   'group_sizes': {str(size): sum(len(g['query']) == size for g in groups)
                                   for size in range(1, MAX_QUERIES + 1)}}
        return {**arrays, 'summary': summary, 'plan_sha256': h.hexdigest()}


def create_model(root, packed, method, device):
    modules = api(root)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    backbone = modules['model'].CoPhyNet(num_objects=packed['spec']['slots'])
    for parameter in backbone.derendering.parameters():
        parameter.requires_grad_(False)
    # This is the only external checkpoint loaded during initial construction.
    backbone.derendering.load_state_dict(torch.load(packed['derenderer'], map_location='cpu', weights_only=True), strict=True)
    model = modules['adapter'].PTCoPhy(backbone, 'Random' if method == 'Random-Both-new' else 'A').to(device)
    model.derendering.eval()
    return model, modules


def encode_history(model, ab):
    """The original GCN/GRU, batching time/objects without changing parameters."""
    net = model.backbone
    b, t, k, _ = ab.pose.shape
    x = ab.pose.reshape(b * t, k, 3)
    presence = ab.presence[:, None].expand(b, t, k).reshape(b * t, k)
    x1, x2 = x[:, None].expand(-1, k, -1, -1), x[:, :, None].expand(-1, -1, k, -1)
    edges = net.mlp_inter(torch.cat([x1, x2], -1))
    pair = presence[:, :, None] * presence[:, None, :] * (1 - torch.eye(k, device=x.device)[None])
    local_edges = (edges * pair[..., None]).sum(2) / (.01 + pair.sum(2))[..., None]
    global_edges = (local_edges * presence[..., None]).sum(1) / (.01 + presence.sum(1))[:, None]
    local = net.mlp_out(torch.cat([x, local_edges, global_edges[:, None].expand(-1, k, -1)], -1))
    sequence = local.reshape(b, t, k, 32).permute(0, 2, 1, 3).reshape(b * k, t, 32)
    _, hidden = net.rnn(sequence)
    return hidden[0].reshape(b, k, 32)


def batch_objective(model, data, plan, start, end, method, modules):
    adapter = modules['adapter']
    device = next(model.parameters()).device
    lo, hi = int(plan['group_offsets'][start]), int(plan['group_offsets'][end])
    indices = plan['query_indices'][lo:hi]
    tx = torch.as_tensor(indices, device=device)
    visual, target = data.visual(tx, 'train', adapter), data.target(tx, 'train', adapter)
    own = encode_history(model, visual.ab)
    prediction = model.predict_code(own, visual)
    native = adapter.task_loss(prediction, target)
    target_mask = target.presence[:, None].expand_as(target.stationary)
    native_mse = ((prediction[0] - target.pose).square().mean(-1) * target_mask).sum() / target_mask.sum()
    zero = own.sum() * 0
    cross, persist = zero, zero
    groups = np.flatnonzero(plan['aux_valid'][start:end]) + start
    rows, focal, repeated_group = [], [], []
    donor_p = None
    mixed = None
    stats = {'invariance': 0., 'variance': 0., 'covariance': 0., 'skipped': True}
    if len(groups):
        source_groups = plan['random_source_group'][groups] if method == 'Random-Both-new' else groups
        donor_ids = plan['correct'][source_groups]
        unique, inverse = np.unique(donor_ids.reshape(-1), return_inverse=True)
        donor_indices = torch.as_tensor(unique, device=device)
        donor_ab = adapter.ABObservation(data.rows['train']['pose_ab'][donor_indices],
                                         data.rows['train']['presence_ab'][donor_indices])
        donor_u = encode_history(model, donor_ab)
        slots = torch.as_tensor(np.repeat(plan['focal'][groups], SUPPORTS), device=device)
        inverse = torch.as_tensor(inverse, device=device)
        if (donor_ab.presence[inverse, slots] <= 0).any():
            raise ValueError('Focal donor not visible in AB')
        donor_p = donor_u[inverse, slots, :16].reshape(len(groups), SUPPORTS, 16).mean(1)
        for group_number, group in enumerate(groups):
            first, last = int(plan['group_offsets'][group]) - lo, int(plan['group_offsets'][group + 1]) - lo
            rows.extend(range(first, last))
            focal.extend([int(plan['focal'][group])] * (last - first))
            repeated_group.extend([group_number] * (last - first))
        rows = torch.as_tensor(rows, device=device)
        focal = torch.as_tensor(focal, device=device)
        expanded = donor_p[torch.as_tensor(repeated_group, device=device)]
        if (visual.ab.presence[rows, focal] <= 0).any() or (visual.presence_c[rows, focal] <= 0).any():
            raise ValueError('Source focal recipient not visible in AB and C')
        if method != 'Align-only':
            mixed = adapter.replace_p(own[rows], focal, expanded)
            cross = adapter.task_loss(model.predict_code(mixed, visual.select(rows)), target.select(rows))
        if method != 'Cross-only':
            persist, stats = modules['protocol'].vicreg_focal(own[rows, focal, :16], expanded)
    loss = native + cross + .1 * persist
    return loss, {'native': native, 'native_mse': native_mse, 'cross': cross, 'persist': persist,
                  'own': own, 'mixed': mixed, 'donor_p': donor_p,
                  'focal': focal, 'paired_rows': rows, 'indices': indices,
                  'paired': len(rows), 'groups': len(groups),
                  'donor_segments': len(groups) * SUPPORTS, 'stats': stats,
                  'prediction': prediction, 'visual': visual}


@torch.no_grad()
def evaluate(model, data, modules):
    model.eval()
    values, ids, uncovered = [], [], []
    device = next(model.parameters()).device
    for start in range(0, len(data.ids['val']), 64):
        indices = np.arange(start, min(start + 64, len(data.ids['val'])))
        tx = torch.as_tensor(indices, device=device)
        visual = data.visual(tx, 'val', modules['adapter'])
        target = data.target(tx, 'val', modules['adapter'])
        pred, presence, _ = model.predict_code(encode_history(model, visual.ab), visual)
        score = modules['training'].mse_per_recipient(pred, target.pose, presence,
                                                      data.packed['spec']['dims'], allow_uncovered=True)
        for i, value, count in zip(indices, score.cpu().tolist(), presence.sum(1).cpu().tolist()):
            ident = data.ids['val'][int(i)]
            if count > 0:
                ids.append(ident); values.append(value)
            else:
                uncovered.append(ident)
    if not values:
        raise ValueError('No valid visual source validation coverage')
    return {'mse': float(np.mean(values)), 'ids': ids, 'per_recipient_mse': values,
            'recipients': len(data.ids['val']), 'scored_recipients': len(values),
            'uncovered_ids': uncovered, 'visual_coverage': len(values) / len(data.ids['val'])}


def prepare(out, scene, root=ROOT):
    out, root = Path(out), Path(root)
    out.mkdir(parents=True, exist_ok=True)
    packed, data_path = prepare_data(root, scene)
    files = {str(Path(__file__).resolve()): digest(__file__), str(data_path): digest(data_path), **packed['files']}
    files.update({str(root / 'source' / name): digest(root / 'source' / name) for name in SOURCE_FILES})
    binding = {'version': VERSION, 'scene': scene, 'root': str(root), 'data_manifest': str(data_path),
               'file_sha256': files, 'test_read': False, 'seed': 0,
               'source_pretrained_checkpoint': None, 'visual_frontend_checkpoint': packed['derenderer'],
               'methods': list(METHODS), 'supports': SUPPORTS, 'max_queries_per_group': MAX_QUERIES,
               'batch_groups': BATCH_GROUPS, 'lambda_x': 1., 'lambda_p': .1}
    path = out / 'binding.json'
    if path.exists() and read(path) != binding:
        raise ValueError('Source run already bound to different files or configuration')
    write(path, binding)
    data = SourceData(packed, 'cpu')
    plan = SourceSampler(data).make_epoch(1)
    write(out / 'plan_epoch1_summary.json', {**plan['summary'], 'plan_sha256': plan['plan_sha256']})
    model, _ = create_model(root, packed, METHODS[0], 'cpu')
    init = {'full_state_sha256': state_digest(model), 'trainable_state_sha256': state_digest(model, True),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'source_checkpoint_loaded': False, 'visual_checkpoint_loaded': True,
            'torch_version': torch.__version__, 'seed': 0}
    write(out / 'initialization.json', init)
    emit('prepared', scene=scene, splits={s: len(ids) for s, ids in data.ids.items()},
         pairing=plan['summary'], initialization=init)


def setup(out, scene, method, device, root):
    torch.set_num_threads(4)
    binding = read(Path(out) / 'binding.json')
    if binding['version'] != VERSION or binding['scene'] != scene or binding['root'] != str(Path(root)):
        raise ValueError('Source binding differs from requested scene/root/version')
    validate_files(binding['file_sha256'])
    packed = read(binding['data_manifest'])
    data = SourceData(packed, device)
    model, modules = create_model(root, packed, method, device)
    init = read(Path(out) / 'initialization.json')
    if state_digest(model) != init['full_state_sha256']:
        raise ValueError('Fresh source state differs from the common prepared initialization')
    return binding, data, model, modules


def smoke(out, scene, device, root=ROOT):
    binding, data, model, modules = setup(out, scene, METHODS[0], device, root)
    plan = SourceSampler(data).make_epoch(1)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    adapter = modules['adapter']
    tx = torch.arange(min(8, len(data.ids['train'])), device=device)
    visual = data.visual(tx, 'train', adapter)
    with torch.no_grad():
        reference_code = model.encode_ab(visual.ab)
        efficient_code = encode_history(model, visual.ab)
        max_error = float((reference_code - efficient_code).abs().max())
        torch.testing.assert_close(reference_code, efficient_code, atol=3e-5, rtol=3e-5)
    records, reference_loss = [], None
    for method in METHODS:
        model.load_state_dict(original, strict=True)
        model.method = 'Random' if method == 'Random-Both-new' else 'A'
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        before = model.backbone.mlp_inter[0].weight.detach().clone()
        loss, detail = batch_objective(model, data, plan, 0, min(BATCH_GROUPS, len(plan['focal'])), method, modules)
        native = float(detail['native'].detach())
        if reference_loss is None:
            reference_loss = native
        if not np.isclose(native, reference_loss, atol=2e-6, rtol=2e-5):
            raise ValueError('Fresh native source path differs across methods')
        if detail['mixed'] is not None:
            own = detail['own'][detail['paired_rows']]
            torch.testing.assert_close(detail['mixed'][..., 16:], own[..., 16:], atol=0, rtol=0)
            nonfocal = ~torch.nn.functional.one_hot(detail['focal'], own.shape[1]).bool()
            torch.testing.assert_close(detail['mixed'][nonfocal], own[nonfocal], atol=0, rtol=0)
        if method == 'Cross-only' and float(detail['persist']) != 0:
            raise ValueError('Cross-only accidentally applied alignment')
        if method == 'Align-only' and float(detail['cross']) != 0:
            raise ValueError('Align-only accidentally predicted cross targets')
        donor_gradient = float(torch.autograd.grad(loss, detail['donor_p'], retain_graph=True)[0].norm()) if detail['donor_p'] is not None else 0.
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Nonfinite source gradient')
        optimizer.step()
        change = float((model.backbone.mlp_inter[0].weight.detach() - before).abs().max())
        if change <= 0 or donor_gradient <= 0 or any(p.grad is not None for p in model.derendering.parameters()):
            raise ValueError('History failed to learn or frozen visual frontend received gradients')
        records.append({'method': method, 'native': native, 'cross': float(detail['cross'].detach()),
                        'persist': float(detail['persist'].detach()), 'paired': detail['paired'],
                        'donor_mean_p_gradient': donor_gradient, 'encoder_max_update': change})
    model.load_state_dict(original, strict=True)
    result = {'status': 'PASS', 'version': VERSION, 'scene': scene, 'test_read': False,
              'initialization': read(Path(out) / 'initialization.json'),
              'efficient_encoder_max_error': max_error, 'methods': records,
              'pairing': plan['summary'], 'plan_sha256': plan['plan_sha256'],
              'source_weights_reloaded_after_smoke': True, 'visual_frontend_frozen': True}
    write(Path(out) / 'real_batch_smoke.json', result)
    emit('smoke_pass', **result)


def train(out, scene, method, device, epochs=50, root=ROOT):
    out = Path(out)
    if method not in METHODS or not 1 <= epochs <= 50:
        raise ValueError('Unknown source objective or budget exceeds 50')
    if read(out / 'real_batch_smoke.json')['status'] != 'PASS':
        raise ValueError('Real source batch smoke has not passed')
    binding, data, model, modules = setup(out, scene, method, device, root)
    run = out / 'runs' / method
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / 'train.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'version': VERSION, 'phase': 'source_formation', 'scene': scene, 'method': method,
              'adapter_method': model.method, 'seed': 0, 'optimizer': 'Adam', 'lr': 1e-3,
              'lambda_x': 0. if method == 'Align-only' else 1.,
              'lambda_p': 0. if method == 'Cross-only' else .1,
              'vicreg_coefficients': [25, 25, 1], 'supports': SUPPORTS, 'max_queries_per_group': MAX_QUERIES,
              'batch_groups': BATCH_GROUPS, 'P_dim': 16, 'T_dim': 16,
              'binding_sha256': digest(out / 'binding.json'),
              'initialization_sha256': read(out / 'initialization.json')['full_state_sha256'],
              'source_pretrained_checkpoint': None, 'visual_frozen': True, 'test_read': False,
              'selection': 'lowest full official validation MSE including epoch0'}
    if (run / 'config.json').exists() and read(run / 'config.json') != config:
        raise ValueError('Cannot resume a changed source configuration')
    write(run / 'config.json', config)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    start, best, history = 0, float('inf'), []
    if (run / 'latest.pt').exists():
        saved = torch.load(run / 'latest.pt', map_location=device, weights_only=False)
        if saved['config'] != config:
            raise ValueError('Source checkpoint config mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['epoch'], saved['best'], saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.device(device).type == 'cuda':
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    else:
        initial = evaluate(model, data, modules)
        best = initial['mse']
        write(run / 'initial_validation.json', initial)
        save_torch(run / 'selected.pt', {'model': model.state_dict(), 'epoch': 0, 'config': config})
        write(run / 'selected_validation.json', {'method': method, 'epoch': 0, **initial})
        emit('initial_validation', scene=scene, method=method, mse=best)
    sampler = SourceSampler(data)
    for epoch in range(start + 1, epochs + 1):
        began = time.perf_counter()
        model.train()
        plan = sampler.make_epoch(epoch)
        write(run / f'plan_epoch{epoch}_summary.json', {**plan['summary'], 'plan_sha256': plan['plan_sha256']})
        totals, seen, updates, first_grad = defaultdict(float), 0, 0, None
        for first in range(0, len(plan['focal']), BATCH_GROUPS):
            optimizer.zero_grad(set_to_none=True)
            loss, detail = batch_objective(model, data, plan, first, min(first + BATCH_GROUPS, len(plan['focal'])), method, modules)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source objective')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Nonfinite source gradient')
            if first_grad is None:
                first_grad = float(model.backbone.mlp_inter[0].weight.grad.norm())
            optimizer.step()
            n = len(detail['indices'])
            seen += n; updates += 1
            for key in ('native', 'native_mse', 'cross', 'persist'):
                totals[key] += float(detail[key].detach()) * n
            totals['loss'] += float(loss.detach()) * n
            for key in ('invariance', 'variance', 'covariance'):
                value = detail['stats'].get(key, 0.)
                totals[key] += float(value.detach() if torch.is_tensor(value) else value) * n
            totals['donor_segments'] += detail['donor_segments']
            totals['paired'] += detail['paired']
        if seen != len(data.ids['train']):
            raise ValueError('Incomplete source training epoch')
        metric = evaluate(model, data, modules)
        row = {'epoch': epoch, 'seconds': time.perf_counter() - began, 'mse': metric['mse'],
               'visual_coverage': metric['visual_coverage'], 'train_mse': totals['native_mse'] / seen,
               'train_native_loss': totals['native'] / seen, 'train_cross_loss': totals['cross'] / seen,
               'train_persist_loss': totals['persist'] / seen, 'train_total_loss': totals['loss'] / seen,
               'vicreg_invariance': totals['invariance'] / seen, 'vicreg_variance': totals['variance'] / seen,
               'vicreg_covariance': totals['covariance'] / seen, 'first_encoder_gradient': first_grad,
               'query_exposures': seen, 'paired_recipient_exposures': int(totals['paired']),
               'donor_segments': int(totals['donor_segments']), 'updates': updates,
               'plan_sha256': plan['plan_sha256'],
               'random_accidental_same_fraction': plan['summary']['random_accidental_same_fraction']}
        history.append(row)
        checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'config': config}
        if metric['mse'] < best:
            best = metric['mse']
            save_torch(run / 'selected.pt', checkpoint)
            write(run / 'selected_validation.json', {'method': method, 'epoch': epoch, **metric})
        if epoch in (10, 25, 50):
            save_torch(run / f'checkpoint_{epoch}.pt', checkpoint)
        save_torch(run / 'latest.pt', {**checkpoint, 'optimizer': optimizer.state_dict(), 'best': best,
                   'history': history, 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else []})
        write(run / 'progress.json', {'status': 'RUNNING', 'method': method, 'epoch': epoch,
                                      'best_mse': best, 'history': history, 'test_read': False})
        emit('source_epoch', scene=scene, method=method, **row)
    write(run / 'complete.json', {'status': 'COMPLETE', 'method': method, 'epochs': epochs,
                                 'best_mse': best, 'test_read': False})
    emit('source_complete', scene=scene, method=method, epochs=epochs, best_mse=best)
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'train'))
    parser.add_argument('--scene', choices=tuple(SPECS), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.scene, args.root)
    elif args.command == 'smoke':
        smoke(args.out, args.scene, args.device, args.root)
    else:
        if args.method is None:
            parser.error('train requires --method')
        train(args.out, args.scene, args.method, args.device, args.epochs, args.root)
