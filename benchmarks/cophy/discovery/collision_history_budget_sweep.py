"""Fixed-checkpoint S=1/3/5/8 sensitivity on the existing Collision val queue.

This is inference only. Heads remain selected using their original S=3,
20-epoch run; no per-S selection or optimization is performed.
"""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import collision_reuse_ft as core


VERSION = 'collision-fixed-history-budget-sweep-v4.8'
SUPPORTS = (1, 3, 5, 8)
MODELS = (('Native-FT', 'v46'), ('A-weak', 'v46'), ('Random-weak', 'v46'),
          ('A-inv1', 'v47'), ('Random-inv1', 'v47'),
          ('A-inv2', 'v47'), ('Random-inv2', 'v47'))
MATCHED = {'A-weak': 'Random-weak', 'A-inv1': 'Random-inv1', 'A-inv2': 'Random-inv2'}


def hash_array(value):
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def fixed_plans(data):
    original = data.plan('val', 0, 1)
    row, part = data.rows['val'], data.manifest['splits']['val']
    if len(row['ids']) != 512:
        raise ValueError('This diagnostic is bound to the existing 512-query queue')
    nested = np.zeros((*original.shape[:-1], 8), dtype=np.int64)
    nested[..., :3] = original
    id_to_index = {ident: i for i, ident in enumerate(part['all_ids'])}
    for i, ident in enumerate(row['ids']):
        for k in range(core.K):
            if row['mask'][i, k] <= 0:
                continue
            first = original[i, k, 0]
            options = part['candidates'][ident][k]
            if len(set(first.tolist())) != 3 or any(int(v) not in options for v in first):
                raise ValueError('Original S3 support plan is invalid')
            used = set(first.tolist())
            remaining = [v for v in options if v not in used]
            if len(remaining) < 5:
                raise ValueError(f'Cannot extend {ident} slot {k} to nested S8')
            seed_text = f'collision-history-extension:20260911:val:{ident}:{k}'
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'little')
            nested[i, k, 0, 3:] = np.random.default_rng(seed).choice(remaining, 5, replace=False)
            if id_to_index[ident] in nested[i, k, 0] or len(set(nested[i, k, 0].tolist())) != 8:
                raise ValueError('Query/support overlap or duplicate nested support')
    plans = {s: nested[..., :s].copy() for s in SUPPORTS}
    if not np.array_equal(plans[3], original):
        raise ValueError('Original S3 sampling changed')
    return plans


def verify_sources(base, v46, v47):
    roots = {'v46': Path(v46), 'v47': Path(v47)}
    bindings, entries = {}, {}
    manifest_sha = core.digest(Path(base) / 'manifest.json')
    for name, root in roots.items():
        status = core.read(root / 'controller_status.json')
        if status.get('status') != 'COMPLETE':
            raise ValueError(f'{name} controller must be COMPLETE before scanning')
        binding = core.read(root / 'binding.json')
        if binding['base_manifest_sha256'] != manifest_sha:
            raise ValueError(f'{name} uses a different base manifest')
        for item in (binding['source'], binding['head']):
            if core.digest(item['path']) != item['sha256']:
                raise ValueError(f'{name} initialization artifact changed')
        bindings[name] = binding
    if bindings['v46']['code_sha256'] != core.digest(core.__file__):
        raise ValueError('Frozen v4.6 loader implementation changed')
    for path, field in ((Path(base) / 'input_val.npz', 'input_sha256'),
                        (Path(base) / 'target_val.npz', 'target_sha256')):
        actual = core.digest(path)
        if any(binding['splits']['val'][field] != actual for binding in bindings.values()):
            raise ValueError('Validation data differ from run bindings: ' + str(path))
    cache_sha = core.digest(bindings['v46']['splits']['val']['cache'])
    if any(binding['splits']['val']['cache_sha256'] != cache_sha for binding in bindings.values()):
        raise ValueError('Validation history cache differs from run bindings')
    for method, family in MODELS:
        root = roots[family]
        run = root / 'runs' / method
        snapshot = core.read(root / 'result_20epochs.json')
        selected = snapshot['methods'][method]
        checkpoint = run / 'selected.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (snapshot.get('budget') != 20 or state['epoch'] != selected['epoch'] or
                not 0 <= state['epoch'] <= 20):
            raise ValueError(f'{method}: current selected checkpoint is not the fixed 20-epoch selection')
        if state['config']['method'] != method:
            raise ValueError('Wrong method checkpoint: ' + method)
        if state['config']['binding_sha256'] != core.digest(root / 'binding.json'):
            raise ValueError('Checkpoint binding mismatch: ' + method)
        progress = core.read(run / 'progress.json')
        if progress.get('epoch', -1) < 20:
            raise ValueError('Source method has not completed 20 epochs: ' + method)
        entries[method] = {'family': family, 'run_path': str(run), 'checkpoint_path': str(checkpoint),
            'checkpoint_sha256': core.digest(checkpoint), 'selected_epoch': state['epoch'],
            'original_s3_mse': selected['mse'], 'original_selection_budget': 20,
            'original_result_path': str(root / 'result_20epochs.json'),
            'original_result_sha256': core.digest(root / 'result_20epochs.json'),
            'binding_sha256': core.digest(root / 'binding.json'),
            'training_code_sha256': state['config']['code_sha256']}
        del state
    return bindings, entries


def comparison(candidate, reference):
    if candidate['ids'] != reference['ids']:
        raise ValueError('Compared methods do not share the same query order')
    a = np.asarray(candidate['per_recipient_mse'], dtype=np.float64)
    b = np.asarray(reference['per_recipient_mse'], dtype=np.float64)
    ref = reference['mse']
    return {'candidate_mse': candidate['mse'], 'reference_mse': ref,
            'paired_mean_difference_reference_minus_candidate': float((b - a).mean()),
            'improvement_percent': 100 * (ref - candidate['mse']) / ref if ref > 0 else None,
            'per_recipient_difference_reference_minus_candidate': (b - a).tolist()}


def run(base, v46, v47, out, device):
    base, v46, v47, out = map(lambda p: Path(p).resolve(), (base, v46, v47, out))
    if any(out == source or out in source.parents for source in (base, v46, v47)):
        raise ValueError('Use a separate output directory for the sensitivity diagnostic')
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'sweep.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    progress = {'status': 'PREPARING', 'version': VERSION, 'test_read': False, 'completed': []}
    core.write(out / 'progress.json', progress)
    try:
        bindings, sources = verify_sources(base, v46, v47)
        data = core.FineTuneData(base, device)
        plans = fixed_plans(data)
        for s, plan in plans.items():
            core.save_npz(out / f'val_support_plan_s{s}.npz', plan=plan,
                          query_ids=np.asarray(data.rows['val']['ids']))
        config = {'version': VERSION, 'device': device, 'base': str(base),
                  'models': sources, 'supports': list(SUPPORTS), 'test_read': False,
                  'inference_only': True, 'optimizer_steps': 0,
                  'checkpoint_selection': 'original S3 within 20 epochs, unchanged across all S',
                  'interpretation': 'fixed-head input sensitivity; not separately optimized S profiles',
                  'plan_rule': 'retain exact original S3, append five independently seeded remaining candidates',
                  'plan_hashes': {str(s): hash_array(plan) for s, plan in plans.items()},
                  'data_manifest_sha256': core.digest(base / 'manifest.json'),
                  'input_val_sha256': core.digest(base / 'input_val.npz'),
                  'target_val_sha256': core.digest(base / 'target_val.npz'),
                  'core_code_sha256': core.digest(core.__file__),
                  'diagnostic_code_sha256': core.digest(__file__),
                  'queries': len(data.rows['val']['ids']), 'query_prefix_frames': 3,
                  'predicted_frames': core.HORIZON, 'metric_dims': core.D}
        core.write(out / 'config.json', config)
        results = {'config': config, 'methods': {}, 'comparisons_by_supports': {}, 'test_read': False}
        progress['status'] = 'RUNNING'
        core.write(out / 'progress.json', progress)
        for method, family in MODELS:
            entry = sources[method]
            if core.digest(entry['checkpoint_path']) != entry['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during inference sweep: ' + method)
            state = torch.load(entry['checkpoint_path'], map_location='cpu', weights_only=False)
            model = core.FineTuneModel(bindings[family]).to(device)
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            del state
            item = {'source': entry, 'supports': {}}
            for s in (3, 1, 5, 8):
                metric = core.evaluate(model, data, plans[s], device)
                if s == 3:
                    difference = abs(metric['mse'] - entry['original_s3_mse'])
                    if difference > 2e-5:
                        raise ValueError(f'{method}: S3 reproduction mismatch {difference:.8g}')
                    item['s3_reproduction_absolute_error'] = difference
                item['supports'][str(s)] = metric
                progress['completed'].append({'method': method, 'supports': s, 'mse': metric['mse']})
                core.write(out / 'progress.json', progress)
            results['methods'][method] = item
            core.write(out / 'result_snapshot.json', results)
            del model
        for s in SUPPORTS:
            key = str(s)
            at_s = {method: item['supports'][key] for method, item in results['methods'].items()}
            comparisons = {'versus_native': {}, 'versus_matched_random': {}}
            for method, metric in at_s.items():
                if method != 'Native-FT':
                    comparisons['versus_native'][method] = comparison(metric, at_s['Native-FT'])
                if method in MATCHED:
                    comparisons['versus_matched_random'][method] = comparison(metric, at_s[MATCHED[method]])
            results['comparisons_by_supports'][key] = comparisons
        results['seconds'] = time.perf_counter() - started
        results['status'] = 'COMPLETE'
        core.write(out / 'result_snapshot.json', results)
        summary = {'status': 'COMPLETE', 'version': VERSION, 'test_read': False,
            'interpretation': config['interpretation'], 'checkpoint_selection': config['checkpoint_selection'],
            'mse_by_supports': {str(s): {method: item['supports'][str(s)]['mse']
                               for method, item in results['methods'].items()} for s in SUPPORTS},
            'improvements_by_supports': {str(s): {kind: {method: value['improvement_percent']
                          for method, value in values.items()}
                          for kind, values in results['comparisons_by_supports'][str(s)].items()} for s in SUPPORTS},
            's3_reproduction_errors': {method: item['s3_reproduction_absolute_error']
                                       for method, item in results['methods'].items()},
            'seconds': results['seconds'], 'result_path': str(out / 'result_snapshot.json')}
        core.write(out / 'summary.json', summary)
        progress.update(status='COMPLETE', seconds=results['seconds'])
        core.write(out / 'progress.json', progress)
        return summary
    except BaseException as exc:
        progress.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        core.write(out / 'progress.json', progress)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--v46', required=True)
    parser.add_argument('--v47', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.v46, args.v47, args.out, args.device),
                     ensure_ascii=False, allow_nan=False))
