"""Fixed v4.9 checkpoints on all eligible Collision validation recipients.

The original 512 select checkpoints. The remaining validation recipients are
reported separately; this is not a test split or a new checkpoint-selection
criterion. No optimizer is constructed. Metadata build the support plan before
CD targets are opened. Only missing three-frame visual prefixes are extracted.
"""

import os
import argparse
import concurrent.futures
import fcntl
import hashlib
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import collision_reuse_ft as core
import collision_xep as xep

VERSION = 'collision-multiquery-fullval-v1'
METHODS = ('Native-MQ', 'A-MQ', 'Random-MQ', 'A-MQ-Reg')
SUPPORTS = 5


def immutable_write(path, value):
    path = Path(path)
    if path.exists() and core.read(path) != value:
        raise ValueError('Existing artifact has a different binding: ' + str(path))
    core.write(path, value)


def artifact(preflight, name):
    return xep.artifact(preflight, name)


def prepare(out, base, run):
    out, base, run = map(Path, (out, base, run))
    if out.resolve() in (base.resolve(), run.resolve()):
        raise ValueError('Use a separate expansion output directory')
    out.mkdir(parents=True, exist_ok=True)
    original = core.read(base / 'manifest.json')
    part = original['splits']['val']
    prepath = base.parent / 'prepared_v3/collision/training_preflight.json'
    if core.digest(prepath) != original['preflight_sha256']:
        raise ValueError('Original preflight binding changed')
    preflight = core.read(prepath)
    relation_path = artifact(preflight, 'raw_relations_val')
    cache_path = artifact(preflight, 'cache_val')
    if str(cache_path) != part['cache'] or core.digest(cache_path) != part['cache_sha256']:
        raise ValueError('Original validation AB cache differs')
    ids = part['all_ids']
    lookup = {ident: i for i, ident in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError('Duplicate validation episode IDs')
    rows = core.read(relation_path)
    with open(cache_path, 'rb') as stream:
        cache = pickle.load(stream)
    groups, by_id = defaultdict(list), defaultdict(list)
    for row in rows:
        if row['split'] != 'val' or row['id'] not in lookup:
            raise ValueError('Non-validation relation row')
        by_id[row['id']].append(row)
        slot = row['slot']
        if cache[row['id']]['presence_ab'][slot] > 0:
            groups[(slot, row['known_type'], tuple(row['physical']))].append(lookup[row['id']])
    del cache
    expanded = {'candidates': {}, 'presence': {}, 'physical': {}, 'known_type': {}}
    eligible, excluded = [], []
    for ident in ids:
        active = [row for row in by_id[ident] if row['in_C']]
        candidates = [[] for _ in range(core.K)]
        presence = [0.] * core.K
        physical = [[0] * 3 for _ in range(core.K)]
        known = [[0.] * len(core.TYPES) for _ in range(core.K)]
        for row in active:
            slot = row['slot']
            presence[slot] = 1.
            physical[slot] = row['physical']
            known[slot][core.TYPES.index(row['known_type'])] = 1.
            candidates[slot] = sorted(index for index in groups[
                (slot, row['known_type'], tuple(row['physical']))] if index != lookup[ident])
        # Preserve original eligibility first, then demand S5 without silently dropping queries.
        if not active or any(len(candidates[row['slot']]) < 3 for row in active):
            excluded.append(ident)
            continue
        if any(len(candidates[row['slot']]) < SUPPORTS for row in active):
            raise ValueError('An original eligible query lacks S5 support: ' + ident)
        eligible.append(ident)
        for name, values in [('candidates', candidates), ('presence', presence),
                             ('physical', physical), ('known_type', known)]:
            expanded[name][ident] = values
    if len(eligible) != part['eligible_total'] or sorted(excluded) != sorted(part['excluded_ids']):
        raise ValueError('Rebuilt validation eligibility differs from original audit')
    old_ids = part['query_ids']
    if len(old_ids) != 512 or len(set(old_ids)) != 512 or not set(old_ids) <= set(eligible):
        raise ValueError('Original 512-query cohort differs')
    for name in expanded:
        if any(expanded[name][ident] != part[name][ident] for ident in old_ids):
            raise ValueError('Rebuilt old metadata differ: ' + name)
    old_set = set(old_ids)
    remaining = sorted((ident for ident in eligible if ident not in old_set),
        key=lambda ident: hashlib.sha256(('xep:20260911:' + ident).encode()).digest())
    ordered = old_ids + remaining
    v49 = core.read(run / 'binding.json')
    if v49['base_manifest_sha256'] != core.digest(base / 'manifest.json'):
        raise ValueError('v4.9 run differs from original task manifest')
    old_plan_path = Path(v49['validation_plan'])
    with np.load(old_plan_path, allow_pickle=False) as saved:
        old_plan = saved['plan'].copy()
        if saved['query_ids'].tolist() != old_ids or old_plan.shape != (512, core.K, 1, SUPPORTS):
            raise ValueError('Original S5 plan differs')
    plan = np.zeros((len(ordered), core.K, 1, SUPPORTS), np.int64)
    plan[:512] = old_plan
    for i, ident in enumerate(ordered):
        # Exactly the old S3 seed and old v4.8 extension rule, generalized to new recipients.
        if i >= 512:
            seed = int.from_bytes(hashlib.sha256(
                f'xep:20260911:val:{ident}:0'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for slot in range(core.K):
                if expanded['presence'][ident][slot] <= 0:
                    continue
                options = expanded['candidates'][ident][slot]
                first = rng.choice(options, 3, replace=False)
                rest = [index for index in options if index not in set(first.tolist())]
                # v4.8 sampled five extras, then S5 retained its first two. Keep the same RNG rule.
                if len(rest) < 5:
                    raise ValueError('Cannot reproduce the nested S8-to-S5 support rule: ' + ident)
                extseed = int.from_bytes(hashlib.sha256(
                    f'collision-history-extension:20260911:val:{ident}:{slot}'.encode()).digest()[:8], 'little')
                extra = np.random.default_rng(extseed).choice(rest, 5, replace=False)
                plan[i, slot, 0] = np.concatenate([first, extra[:2]])
        for slot in range(core.K):
            if expanded['presence'][ident][slot] <= 0:
                continue
            chosen = plan[i, slot, 0].tolist()
            if len(set(chosen)) != SUPPORTS or lookup[ident] in chosen or not set(chosen) <= set(expanded['candidates'][ident][slot]):
                raise ValueError('Invalid independent S5 history plan')
    manifest = {'version': VERSION, 'scene': 'collision-normal', 'base': str(base), 'run': str(run),
        'test_read': False, 'inference_only': True, 'optimizer_steps': 0,
        'query_ids': ordered, 'original_query_ids': old_ids, 'remaining_query_ids': remaining,
        'all_history_ids': ids, 'metadata': expanded, 'data_root': original['data_root'],
        'derenderer': original['derenderer'], 'cache': str(cache_path),
        'prefix_frames': [0, 1, 2], 'target_frames': list(range(3, 15)),
        'supports': SUPPORTS, 'selection_rule': 'unchanged v4.9 selection on original 512 only',
        'remaining_cohort_note': 'not used for new-head checkpoint selection; validation, not sealed test',
        'cohort_counts': {'original_selection': 512, 'remaining': len(remaining), 'all_eligible': len(ordered)},
        'support_plan_sha256': hashlib.sha256(plan.tobytes()).hexdigest(),
        'file_sha256': {str(path): core.digest(path) for path in [base / 'manifest.json',
            base / 'input_val.npz', base / 'target_val.npz', prepath, relation_path, cache_path,
            Path(original['derenderer']), run / 'binding.json', old_plan_path,
            Path(__file__), Path(core.__file__), Path(xep.__file__)]}}
    immutable_write(out / 'manifest.json', manifest)
    core.save_npz(out / 'support_plan.npz', plan=plan, query_ids=np.array(ordered))
    core.emit('fullval_prepared', **manifest['cohort_counts'], test_read=False)


def verified_manifest(out):
    out = Path(out)
    manifest = core.read(out / 'manifest.json')
    if manifest['version'] != VERSION:
        raise ValueError('Wrong expansion version')
    for path, sha in manifest['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Bound input changed: ' + path)
    return manifest


@torch.no_grad()
def prefix(out, device, shard, shards, batch_size):
    out = Path(out)
    manifest = verified_manifest(out)
    if not 0 <= shard < shards or batch_size < 1:
        raise ValueError('Invalid shard/batch size')
    target = out / 'prefix' / f'val_{shard}_of_{shards}.npz'
    binding = {'manifest_sha256': core.digest(out / 'manifest.json'), 'shard': shard, 'shards': shards}
    bindpath = target.with_suffix('.json')
    if target.exists():
        if core.read(bindpath) != {**binding, 'artifact_sha256': core.digest(target)}:
            raise ValueError('Existing prefix shard differs')
        return
    ids = manifest['remaining_query_ids'][shard::shards]
    root = Path(manifest['data_root'])
    if not root.is_dir():
        raise FileNotFoundError('Original raw Collision data_root is unavailable: ' + str(root))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    from derendering.model import DeRendering
    visual = DeRendering(core.K).to(device).eval()
    visual.load_state_dict(torch.load(manifest['derenderer'], map_location='cpu', weights_only=True), strict=True)
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    poses, detected, done = [], [], []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for off in range(0, len(ids), batch_size):
            loaded = list(pool.map(xep.load_prefix, [(ident, root) for ident in ids[off:off + batch_size]]))
            rgb = np.stack([item[1] for item in loaded])
            tensor = torch.from_numpy(rgb.reshape(-1, 3, 224, 224)).contiguous().to(device)
            presence, pose, _ = visual(tensor)
            poses.append(pose.reshape(len(loaded), 3, core.K, 3).cpu().numpy())
            detected.append((presence.reshape(len(loaded), 3, core.K) > 0).cpu().numpy().astype(np.float32))
            done.extend(item[0] for item in loaded)
            if off % (batch_size * 10) == 0:
                core.emit('fullval_prefix_progress', shard=shard, done=len(done), total=len(ids), seconds=time.perf_counter() - start)
    pose = np.concatenate(poses) if poses else np.empty((0, 3, core.K, 3), np.float32)
    det = np.concatenate(detected) if detected else np.empty((0, 3, core.K), np.float32)
    core.save_npz(target, ids=np.array(done), pose=pose, detected=det)
    core.write(bindpath, {**binding, 'artifact_sha256': core.digest(target)})
    core.emit('fullval_prefix_complete', shard=shard, queries=len(done), seconds=time.perf_counter() - start)


def merge(out, shards):
    out = Path(out)
    manifest = verified_manifest(out)
    base = Path(manifest['base'])
    missing = {}
    files = {}
    for shard in range(shards):
        path = out / 'prefix' / f'val_{shard}_of_{shards}.npz'
        expected = {'manifest_sha256': core.digest(out / 'manifest.json'), 'shard': shard,
                    'shards': shards, 'artifact_sha256': core.digest(path)}
        if core.read(path.with_suffix('.json')) != expected:
            raise ValueError('Unbound prefix shard')
        files[str(path)] = expected['artifact_sha256']
        with np.load(path, allow_pickle=False) as data:
            if data['ids'].tolist() != manifest['remaining_query_ids'][shard::shards]:
                raise ValueError('Prefix shard order differs')
            for ident, pose, detected in zip(data['ids'].tolist(), data['pose'], data['detected']):
                if ident in missing:
                    raise ValueError('Duplicate prefix recipient')
                missing[ident] = (pose, detected)
    if set(missing) != set(manifest['remaining_query_ids']):
        raise ValueError('Missing/extra validation prefix')
    with np.load(base / 'input_val.npz', allow_pickle=False) as old:
        if old['ids'].tolist() != manifest['original_query_ids']:
            raise ValueError('Original input cohort differs')
        q, det, masks = [old[key].copy() for key in ('pose', 'detected', 'presence')]
    with np.load(base / 'target_val.npz', allow_pickle=False) as old:
        targets = old['pose'].copy()
    newq, newdet, newmask, newtargets = [], [], [], []
    metadata = manifest['metadata']
    # Support sampling is already frozen. Targets are opened only in this merge stage.
    for ident in manifest['remaining_query_ids']:
        pose, seen = missing[ident]
        mask = np.asarray(metadata['presence'][ident], np.float32)
        state = np.load(Path(manifest['data_root']) / ident / 'cd/states.npy', allow_pickle=False)
        if state.shape[:2] != (15, core.K):
            raise ValueError('Unexpected CD target shape: ' + ident)
        actual = (np.abs(state[0, :, :3]).sum(-1) > 0).astype(np.float32)
        if not np.array_equal(actual, mask):
            raise ValueError('Initial object-presence metadata mismatch: ' + ident)
        known = np.asarray(metadata['known_type'][ident], np.float32)
        newq.append(pose)
        newdet.append(np.concatenate([seen[..., None], np.broadcast_to(known[None], (3, core.K, len(core.TYPES)))], -1))
        newmask.append(mask)
        newtargets.append(np.asarray(state[3:, :, :core.D], np.float32))
    if newq:
        q = np.concatenate([q, np.stack(newq)])
        det = np.concatenate([det, np.stack(newdet)])
        masks = np.concatenate([masks, np.stack(newmask)])
        targets = np.concatenate([targets, np.stack(newtargets)])
    if not all(np.isfinite(array).all() for array in (q, det, masks, targets)):
        raise ValueError('Nonfinite expansion inputs/targets')
    core.save_npz(out / 'input_val.npz', ids=np.array(manifest['query_ids']), pose=q, detected=det, presence=masks)
    core.save_npz(out / 'target_val.npz', pose=targets)
    files.update({str(out / name): core.digest(out / name) for name in ('input_val.npz', 'target_val.npz', 'support_plan.npz')})
    immutable_write(out / 'data_ready.json', {'version': VERSION, 'manifest_sha256': core.digest(out / 'manifest.json'),
        'file_sha256': files, 'cohort_counts': manifest['cohort_counts'], 'test_read': False,
        'original_512_prefixes_and_targets_copied_exactly': True})
    core.emit('fullval_data_ready', **manifest['cohort_counts'])


class ValidationData:
    """Only validation AB inputs are loaded; no source train tensors are needed."""
    def __init__(self, out, manifest, device):
        out = Path(out)
        with np.load(out / 'input_val.npz', allow_pickle=False) as data:
            ids = data['ids'].tolist()
            row = {'ids': ids, 'q': data['pose'].copy(), 'det': data['detected'].copy(), 'mask': data['presence'].copy()}
        with np.load(out / 'target_val.npz', allow_pickle=False) as data:
            row['target'] = data['pose'].copy()
        if ids != manifest['query_ids']:
            raise ValueError('Merged query order differs')
        self.rows = {'val': row}
        with open(manifest['cache'], 'rb') as stream:
            cache = pickle.load(stream)
        self.hist = {'val': {name: torch.from_numpy(np.stack([cache[ident][key] for ident in manifest['all_history_ids']])).float().to(device)
            for name, key in [('pose', 'pose_ab'), ('presence', 'presence_ab')]}}

    def batch(self, split, indices, device):
        return tuple(torch.from_numpy(np.asarray(self.rows[split][key][indices], np.float32)).to(device)
                     for key in ('q', 'det', 'mask', 'target'))


def evaluate(out, device):
    out = Path(out)
    manifest = verified_manifest(out)
    ready = core.read(out / 'data_ready.json')
    if ready['manifest_sha256'] != core.digest(out / 'manifest.json'):
        raise ValueError('Merged input binding differs')
    for path, sha in ready['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Merged data changed: ' + path)
    run = Path(manifest['run'])
    status = core.read(run / 'controller_status.json')
    if status.get('status') != 'COMPLETE':
        raise ValueError('Finish the v4.9 controller before freezing final checkpoints')
    binding = core.read(run / 'binding.json')
    for path, sha in binding['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('v4.9 bound dependency changed: ' + path)
    frozen = {'version': VERSION, 'test_read': False, 'selection_cohort_size': 512,
        'data_manifest_sha256': core.digest(out / 'manifest.json'), 'methods': {}}
    for method in METHODS:
        path = run / 'runs' / method / 'selected.pt'
        receipt = run / 'runs' / method / 'selected_validation.json'
        state = torch.load(path, map_location='cpu', weights_only=False)
        metric = core.read(receipt)
        if state['config']['method'] != method or state['config']['binding_sha256'] != core.digest(run / 'binding.json'):
            raise ValueError('Selected checkpoint method/binding differs')
        if state['epoch'] != metric['epoch'] or metric['ids'] != manifest['original_query_ids']:
            raise ValueError('Selected checkpoint/validation receipt differs')
        frozen['methods'][method] = {'checkpoint': str(path), 'checkpoint_sha256': core.digest(path),
            'selected_epoch': state['epoch'], 'receipt': str(receipt), 'receipt_sha256': core.digest(receipt)}
    immutable_write(out / 'checkpoint_freeze.json', frozen)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    data = ValidationData(out, manifest, device)
    with np.load(out / 'support_plan.npz', allow_pickle=False) as saved:
        plan = saved['plan'].copy()
        if saved['query_ids'].tolist() != manifest['query_ids'] or hashlib.sha256(plan.tobytes()).hexdigest() != manifest['support_plan_sha256']:
            raise ValueError('Evaluation support plan differs')
    result = {'version': VERSION, 'test_read': False, 'optimizer_steps': 0,
        'cohort_counts': manifest['cohort_counts'], 'methods': {}, 'comparisons': {},
        'checkpoint_freeze_sha256': core.digest(out / 'checkpoint_freeze.json'),
        'interpretation': 'validation expansion at unchanged selected checkpoints; no re-selection on remaining cohort'}
    started = time.perf_counter()
    for method, source in frozen['methods'].items():
        if core.digest(source['checkpoint']) != source['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after freeze')
        model = core.FineTuneModel(binding['legacy_binding']).to(device)
        state = torch.load(source['checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        metric = core.evaluate(model, data, plan, device)
        values = np.asarray(metric['per_recipient_mse'])
        old = core.read(source['receipt'])
        error = float(np.abs(values[:512] - np.asarray(old['per_recipient_mse'])).max())
        if not np.allclose(values[:512], old['per_recipient_mse'], atol=2e-5, rtol=2e-5):
            raise ValueError(f'{method}: original512 per-recipient reproduction failed, max error {error}')
        cohorts = {'original_selection': float(values[:512].mean()),
                   'remaining': float(values[512:].mean()) if len(values) > 512 else None,
                   'all_eligible': float(values.mean())}
        result['methods'][method] = {'source': source, 'cohort_mse': cohorts,
            'original_512_reproduction_max_absolute_error': error, **metric}
        core.write(out / 'result_snapshot.json', result)
        core.emit('fullval_method_complete', method=method, cohorts=cohorts, selected_epoch=source['selected_epoch'])
        del model, state
    for candidate in ('A-MQ', 'A-MQ-Reg'):
        for reference in ('Native-MQ', 'Random-MQ'):
            ca, re = result['methods'][candidate], result['methods'][reference]
            result['comparisons'][candidate + '_vs_' + reference] = {
                cohort: {'reference_mse': ref, 'candidate_mse': ca['cohort_mse'][cohort],
                    'improvement_percent': 100 * (ref - ca['cohort_mse'][cohort]) / ref if ref else None}
                for cohort, ref in re['cohort_mse'].items()}
    result['seconds'] = time.perf_counter() - started
    result['status'] = 'COMPLETE'
    core.write(out / 'result_snapshot.json', result)
    core.emit('fullval_complete', seconds=result['seconds'], cohorts=manifest['cohort_counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'prefix', 'merge', 'evaluate', 'all'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_discovery_collision_v4_4'))
    parser.add_argument('--run', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_multiquery_v4_9'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=24)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Different prefix shards may proceed concurrently; other operations own a separate lock.
    lockname = f'prefix_{args.shard}_of_{args.shards}.lock' if args.command == 'prefix' else 'operation.lock'
    with open(Path(args.out) / lockname, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command in ('prepare', 'all'):
            prepare(args.out, args.base, args.run)
        if args.command in ('prefix', 'all'):
            if args.command == 'all' and args.shards != 1:
                raise ValueError('Use explicit prefix commands for multiple shards')
            prefix(args.out, args.device, args.shard, args.shards, args.batch_size)
        if args.command in ('merge', 'all'):
            merge(args.out, args.shards)
        if args.command in ('evaluate', 'all'):
            evaluate(args.out, args.device)
