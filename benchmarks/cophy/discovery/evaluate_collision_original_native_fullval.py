"""Original frozen Native versus completed MQ systems on all4000 validation.

Only two new model evaluations (original Native S5/S8). Reuse hash/plan-bound
completed MQ predictions. No training or checkpoint selection is performed.
"""

import os
import argparse
import fcntl
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import collision_reuse_ft as core
import evaluate_collision_multiquery_fullval as fullval

VERSION = 'collision-original-native-fullval-v1'
METHODS = ('Native-MQ', 'A-MQ', 'A-MQ-Reg')


def compact(result):
    return {**{k: v for k, v in result.items() if k != 'methods'}, 'methods': {
        method: {'supports': {s: {k: v for k, v in metric.items() if k not in ('ids', 'per_recipient_mse')}
                               for s, metric in value['supports'].items()}}
        for method, value in result['methods'].items()}}


def run(fullval_dir, sweep_dir, original_dir, out, device):
    fullval_dir, sweep_dir, original_dir, out = map(Path, (fullval_dir, sweep_dir, original_dir, out))
    out.mkdir(parents=True, exist_ok=True)
    manifest = fullval.verified_manifest(fullval_dir)
    ready = core.read(fullval_dir / 'data_ready.json')
    if ready['manifest_sha256'] != core.digest(fullval_dir / 'manifest.json'):
        raise ValueError('Full-validation binding differs')
    for path, sha in ready['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Full-validation inputs changed: ' + path)
    swept = core.read(sweep_dir / 'result_snapshot.json')
    if swept.get('status') != 'COMPLETE' or swept['config']['checkpoint_freeze_sha256'] != core.digest(fullval_dir / 'checkpoint_freeze.json'):
        raise ValueError('Completed S5/S8 result checkpoint freeze differs')
    if swept['config']['fullval'] != str(fullval_dir):
        raise ValueError('S5/S8 predictions used a different full-validation dataset')
    binding = core.read(Path(manifest['run']) / 'binding.json')
    for path, sha in binding['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Training dependency changed: ' + path)
    previous = core.read(original_dir / 'result_snapshot.json')
    if previous['status'] != 'COMPLETE':
        raise ValueError('Original512 comparison not complete')
    plans = {}
    for supports in (5, 8):
        with np.load(sweep_dir / f'support_plan_s{supports}.npz', allow_pickle=False) as saved:
            plan = saved['plan'].copy()
            if saved['query_ids'].tolist() != manifest['query_ids']:
                raise ValueError('S5/S8 recipient order differs')
        if hashlib.sha256(plan.tobytes()).hexdigest() != swept['config']['support_plan_sha256'][str(supports)]:
            raise ValueError('Completed sweep support plan hash differs')
        if hashlib.sha256(plan[:512].tobytes()).hexdigest() != previous['config']['plan_hashes'][str(supports)]:
            raise ValueError('Original512 support plan differs')
        plans[supports] = plan
    with np.load(fullval_dir / 'support_plan.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['plan'], plans[5]):
            raise ValueError('S5 plan differs from original full-validation evaluation')
    if len(manifest['query_ids']) != 4000 or manifest['query_ids'][:512] != previous['methods']['Original-Native-Frozen']['supports']['5']['ids']:
        raise ValueError('Original/full-validation cohort mismatch')
    config = {'version': VERSION, 'test_read': False, 'optimizer_steps': 0,
        'legacy_source': binding['legacy_binding']['source'], 'legacy_head': binding['legacy_binding']['head'],
        'old_head_training_supports': 3, 'latest_models_training_supports': 5,
        'evaluation_supports': [5, 8], 'prefix_frames': 3, 'predicted_frames': 12,
        'cohort_counts': manifest['cohort_counts'], 'script_sha256': core.digest(__file__),
        'sweep_result_sha256': core.digest(sweep_dir / 'result_snapshot.json'),
        'original512_result_sha256': core.digest(original_dir / 'result_snapshot.json'),
        'support_plan_sha256': {str(s): hashlib.sha256(plan.tobytes()).hexdigest() for s, plan in plans.items()},
        'interpretation': 'Inference-support-matched system comparison; original head trained S3, latest models trained S5. Gains include encoder fine-tuning, recurrent-history architecture and budget changes. A-specific attribution requires comparison with Native-MQ.'}
    fullval.immutable_write(out / 'config.json', config)
    torch.set_num_threads(4); torch.manual_seed(0)
    data = fullval.ValidationData(fullval_dir, manifest, device)
    model = core.FineTuneModel(binding['legacy_binding']).to(device)
    if torch.count_nonzero(model.head.cell.weight_ih[:, -64:]).item() != 0:
        raise ValueError('Original predictor gained a recurrent-history pathway')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = {'config': config, 'methods': {'Original-Native-Frozen': {'supports': {}}},
              'comparisons_by_supports': {}, 'status': 'RUNNING', 'test_read': False}
    for method in METHODS:
        source = swept['methods'][method]['source']
        if core.digest(source['checkpoint']) != source['checkpoint_sha256']:
            raise ValueError('MQ checkpoint changed after completed evaluation')
        result['methods'][method] = {'supports': {}}
        for supports in (5, 8):
            metric = swept['methods'][method]['supports'][str(supports)]
            if metric['ids'] != manifest['query_ids']:
                raise ValueError('Reused per-recipient prediction order differs')
            result['methods'][method]['supports'][str(supports)] = metric
    began = time.perf_counter()
    for supports in (5, 8):
        metric = core.evaluate(model, data, plans[supports], device)
        values = np.asarray(metric['per_recipient_mse'])
        old = np.asarray(previous['methods']['Original-Native-Frozen']['supports'][str(supports)]['per_recipient_mse'])
        error = float(np.abs(values[:512] - old).max())
        if not np.allclose(values[:512], old, atol=2e-5, rtol=2e-5):
            raise ValueError('Original512 baseline did not reproduce')
        metric['original512_reproduction_max_absolute_error'] = error
        metric['cohort_mse'] = {'original_selection': float(values[:512].mean()),
            'remaining': float(values[512:].mean()), 'all_eligible': float(values.mean())}
        result['methods']['Original-Native-Frozen']['supports'][str(supports)] = metric
        comparison = {}
        for candidate in METHODS:
            a = result['methods'][candidate]['supports'][str(supports)]['cohort_mse']
            b = metric['cohort_mse']
            comparison[candidate + '_vs_Original-Native-Frozen'] = {cohort: {
                'candidate_mse': a[cohort], 'reference_mse': reference,
                'improvement_percent': 100 * (reference - a[cohort]) / reference}
                for cohort, reference in b.items()}
        result['comparisons_by_supports'][str(supports)] = comparison
        core.write(out / 'result_snapshot.json', result)
        core.write(out / 'summary.json', compact(result))
        core.emit('original_native_fullval', supports=supports, original_mse=metric['cohort_mse'], comparisons=comparison)
    result['status'] = 'COMPLETE'; result['seconds'] = time.perf_counter() - began
    core.write(out / 'result_snapshot.json', result); core.write(out / 'summary.json', compact(result))
    core.emit('original_native_fullval_complete', seconds=result['seconds'], out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fullval', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_multiquery_v4_9_fullval'))
    parser.add_argument('--sweep', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_multiquery_v4_9_s8'))
    parser.add_argument('--original', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_original_native_comparison_v49'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / 'comparison.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.fullval, args.sweep, args.original, args.out, args.device)
