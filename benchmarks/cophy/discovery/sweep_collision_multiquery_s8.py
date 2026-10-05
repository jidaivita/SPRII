"""S5/S8 input sensitivity at the fixed completed Collision v4.9 checkpoints.

Reuse the completed 4000-recipient full-validation dataset and checkpoint
freeze. Reproduce all four S5 per-recipient score arrays before evaluating S8.
No training, no new selection, no test data, and no recipient filtering.
"""
import argparse
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

import collision_reuse_ft as core
import evaluate_collision_multiquery_fullval as fullval

VERSION = 'collision-multiquery-fixed-s8-v1'
METHODS = ('Native-MQ', 'A-MQ', 'Random-MQ', 'A-MQ-Reg')


def nested_plan(manifest, original):
    ids = manifest['query_ids']
    if original.shape != (len(ids), core.K, 1, 5):
        raise ValueError('Unexpected fixed S5 plan shape')
    plan = np.zeros((len(ids), core.K, 1, 8), np.int64)
    lookup = {ident: i for i, ident in enumerate(manifest['all_history_ids'])}
    minimum_pool = None
    for i, ident in enumerate(ids):
        for slot in range(core.K):
            if manifest['metadata']['presence'][ident][slot] <= 0:
                continue
            options = manifest['metadata']['candidates'][ident][slot]
            minimum_pool = len(options) if minimum_pool is None else min(minimum_pool, len(options))
            if len(options) < 8:
                raise ValueError(f'Cannot cover all recipients at S8: {ident}, slot {slot}')
            first = original[i, slot, 0, :3]
            if len(set(first.tolist())) != 3 or not set(first.tolist()) <= set(options):
                raise ValueError('Fixed S3 portion is invalid')
            used = set(first.tolist())
            remaining = [index for index in options if index not in used]
            seed = int.from_bytes(hashlib.sha256(
                f'collision-history-extension:20260911:val:{ident}:{slot}'.encode()).digest()[:8], 'little')
            extra = np.random.default_rng(seed).choice(remaining, 5, replace=False)
            if not np.array_equal(extra[:2], original[i, slot, 0, 3:]):
                raise ValueError('v4.8 nested extension does not reproduce the frozen S5 tail')
            chosen = np.concatenate([first, extra])
            if lookup[ident] in chosen or len(set(chosen.tolist())) != 8:
                raise ValueError('Query leakage or duplicate S8 history')
            plan[i, slot, 0] = chosen
    if not np.array_equal(plan[..., :5], original):
        raise ValueError('S8 changed the original five histories')
    return plan, minimum_pool


def cohort_scores(values, original_count):
    return {'original_selection': float(values[:original_count].mean()),
            'remaining': float(values[original_count:].mean()),
            'all_eligible': float(values.mean())}


def compact(result):
    return {key: value for key, value in result.items() if key != 'methods'} | {'methods': {
        name: {'source': value['source'], 's5_reproduction_max_absolute_error': value.get('s5_reproduction_max_absolute_error'),
               'supports': {s: {'mse': metric['mse'], 'cohort_mse': metric['cohort_mse'],
                                'recipients': metric['recipients']} for s, metric in value['supports'].items()}}
        for name, value in result['methods'].items()}}


def run(fullval_dir, out, device):
    fullval_dir, out = Path(fullval_dir), Path(out)
    if fullval_dir.resolve() == out.resolve():
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest = fullval.verified_manifest(fullval_dir)
    if len(manifest['query_ids']) != 4000 or len(manifest['original_query_ids']) != 512:
        raise ValueError('Expected the completed 4000-recipient validation expansion with original512')
    ready = core.read(fullval_dir / 'data_ready.json')
    if ready['manifest_sha256'] != core.digest(fullval_dir / 'manifest.json'):
        raise ValueError('Full-validation data binding changed')
    for path, sha in ready['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Full-validation data changed: ' + path)
    old_result = core.read(fullval_dir / 'result_snapshot.json')
    if old_result.get('status') != 'COMPLETE' or old_result['checkpoint_freeze_sha256'] != core.digest(fullval_dir / 'checkpoint_freeze.json'):
        raise ValueError('Full-validation evaluation/checkpoint freeze is incomplete or changed')
    frozen = core.read(fullval_dir / 'checkpoint_freeze.json')
    if set(frozen['methods']) != set(METHODS) or frozen['data_manifest_sha256'] != core.digest(fullval_dir / 'manifest.json'):
        raise ValueError('Unexpected frozen models/data')
    source_run = Path(manifest['run'])
    if core.read(source_run / 'controller_status.json').get('status') != 'COMPLETE':
        raise ValueError('v4.9 training controller is not terminal COMPLETE')
    binding = core.read(source_run / 'binding.json')
    for path, sha in binding['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('v4.9 dependency changed: ' + path)
    with np.load(fullval_dir / 'support_plan.npz', allow_pickle=False) as saved:
        s5 = saved['plan'].copy()
        if saved['query_ids'].tolist() != manifest['query_ids'] or hashlib.sha256(s5.tobytes()).hexdigest() != manifest['support_plan_sha256']:
            raise ValueError('Frozen full-validation S5 plan changed')
    s8, minimum_pool = nested_plan(manifest, s5)
    plans = {5: s5, 8: s8}
    config = {'version': VERSION, 'test_read': False, 'optimizer_steps': 0, 'inference_only': True,
        'fullval': str(fullval_dir), 'cohort_counts': manifest['cohort_counts'],
        'source_checkpoints': frozen['methods'], 'supports': [5, 8], 'minimum_active_candidate_pool': minimum_pool,
        'checkpoint_selection': 'unchanged final v4.9 checkpoints selected using original512 S5 validation; Random may retain epoch0',
        'plan_rule': 'original S3 plus same five v4.8-seeded extras; exact original S5 is S8 prefix',
        'interpretation': 'fixed S5-trained checkpoint input sensitivity; not an S8 training comparison; no S8 advantage does not rule out S8 adaptation',
        'script_sha256': core.digest(__file__), 'fullval_code_sha256': core.digest(fullval.__file__),
        'checkpoint_freeze_sha256': core.digest(fullval_dir / 'checkpoint_freeze.json'),
        'original_fullval_result_sha256': core.digest(fullval_dir / 'result_snapshot.json'),
        'support_plan_sha256': {str(s): hashlib.sha256(plan.tobytes()).hexdigest() for s, plan in plans.items()}}
    fullval.immutable_write(out / 'config.json', config)
    for supports, plan in plans.items():
        core.save_npz(out / f'support_plan_s{supports}.npz', plan=plan, query_ids=np.array(manifest['query_ids']))
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    data = fullval.ValidationData(fullval_dir, manifest, device)
    result = {'config': config, 'test_read': False, 'status': 'RUNNING',
        'methods': {method: {'source': frozen['methods'][method], 'supports': {}} for method in METHODS},
        'comparisons_by_supports': {}, 'history_gain_by_method': {}}
    began = time.perf_counter()
    for supports in (5, 8):
        # This order verifies every S5 model before opening the S8 result layer.
        for method in METHODS:
            source = frozen['methods'][method]
            if core.digest(source['checkpoint']) != source['checkpoint_sha256'] or core.digest(source['receipt']) != source['receipt_sha256']:
                raise ValueError('Frozen checkpoint/receipt changed: ' + method)
            checkpoint = torch.load(source['checkpoint'], map_location=device, weights_only=False)
            if checkpoint['epoch'] != source['selected_epoch'] or checkpoint['config']['method'] != method:
                raise ValueError('Frozen method/epoch differs')
            model = core.FineTuneModel(binding['legacy_binding']).to(device)
            model.load_state_dict(checkpoint['model'], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            metric = core.evaluate(model, data, plans[supports], device)
            values = np.asarray(metric['per_recipient_mse'])
            if metric['ids'] != manifest['query_ids']:
                raise ValueError('Evaluation recipient order differs')
            if supports == 5:
                previous = old_result['methods'][method]
                if previous['ids'] != metric['ids']:
                    raise ValueError('Original full-validation order differs')
                prior = np.asarray(previous['per_recipient_mse'])
                error = float(np.abs(values - prior).max())
                if not np.allclose(values, prior, atol=2e-5, rtol=2e-5):
                    raise ValueError(f'{method}: all4000 S5 reproduction failed; max error {error}')
                result['methods'][method]['s5_reproduction_max_absolute_error'] = error
            metric['cohort_mse'] = cohort_scores(values, 512)
            result['methods'][method]['supports'][str(supports)] = metric
            core.write(out / 'result_snapshot.json', result)
            core.write(out / 'summary.json', compact(result))
            core.emit('s8_sweep_model_complete', method=method, supports=supports, cohorts=metric['cohort_mse'],
                      selected_epoch=source['selected_epoch'])
            del model, checkpoint
    for supports in (5, 8):
        comparisons = {}
        for candidate in ('A-MQ', 'A-MQ-Reg'):
            for reference in ('Native-MQ', 'Random-MQ'):
                a, b = [result['methods'][name]['supports'][str(supports)]['cohort_mse'] for name in (candidate, reference)]
                comparisons[candidate + '_vs_' + reference] = {cohort: {
                    'candidate_mse': a[cohort], 'reference_mse': value,
                    'improvement_percent': 100 * (value - a[cohort]) / value if value else None}
                    for cohort, value in b.items()}
        result['comparisons_by_supports'][str(supports)] = comparisons
    for method in METHODS:
        a, b = [result['methods'][method]['supports'][str(s)]['cohort_mse'] for s in (8, 5)]
        result['history_gain_by_method'][method] = {cohort: {
            's5_mse': value, 's8_mse': a[cohort], 'improvement_percent': 100 * (value - a[cohort]) / value if value else None}
            for cohort, value in b.items()}
    result['status'] = 'COMPLETE'
    result['seconds'] = time.perf_counter() - began
    core.write(out / 'result_snapshot.json', result)
    core.write(out / 'summary.json', compact(result))
    core.emit('s8_sweep_complete', seconds=result['seconds'], out=str(out), test_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fullval', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / 'sweep.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.fullval, args.out, args.device)
