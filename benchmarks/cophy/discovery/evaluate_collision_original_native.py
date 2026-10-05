"""Compare original frozen Native to adapted v4.9 models, without training.

Same 512 recipients, query prefix3, horizon12 and nested support plans. The
comparison includes adaptation, architecture and budget changes; it does not
isolate A's incremental effect over equally adapted Native-MQ.
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

METHODS = ('Original-Native-Frozen', 'Native-MQ', 'A-MQ', 'A-MQ-Reg')
VERSION = 'collision-original-native-comparison-v1'


def summary(result):
    result = dict(result)
    result['methods'] = {name: {**{k: v for k, v in item.items() if k != 'supports'},
        'supports': {s: {k: v for k, v in metric.items() if k not in ('ids', 'per_recipient_mse')}
                     for s, metric in item['supports'].items()}} for name, item in result['methods'].items()}
    return result


def run(base, source_run, out, device):
    base, source_run, out = map(Path, (base, source_run, out))
    out.mkdir(parents=True, exist_ok=True)
    if core.read(source_run / 'controller_status.json').get('status') != 'COMPLETE':
        raise ValueError('v4.9 must be complete')
    binding = core.read(source_run / 'binding.json')
    if binding['base_manifest_sha256'] != core.digest(base / 'manifest.json'):
        raise ValueError('Base manifest changed')
    for path, sha in binding['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('Dependency changed: ' + path)
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    data = core.FineTuneData(base, device)
    if len(data.rows['val']['ids']) != 512:
        raise ValueError('Expected original512 cohort')
    plans = {}
    for supports in (3, 5, 8):
        path = base.parent / f'xep_collision_history_sweep_v4_8/val_support_plan_s{supports}.npz'
        with np.load(path, allow_pickle=False) as saved:
            if saved['query_ids'].tolist() != data.rows['val']['ids']:
                raise ValueError('Nested support cohort changed')
            plans[supports] = saved['plan'].copy()
    if not np.array_equal(plans[8][..., :5], plans[5]) or not np.array_equal(plans[5][..., :3], plans[3]):
        raise ValueError('Plans are not nested')
    legacy = binding['legacy_binding']
    sources = {'Original-Native-Frozen': {'source': legacy['source'], 'head': legacy['head'],
        'model': 'source Native encoder plus original frozen-representation downstream head',
        'extra_recurrent_memory_columns': 'initialized exactly zero; original predictor function preserved'}}
    for method in METHODS[1:]:
        path = source_run / 'runs' / method / 'selected.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        receipt = core.read(path.parent / 'selected_validation.json')
        if state['epoch'] != receipt['epoch'] or state['config']['method'] != method or receipt['ids'] != data.rows['val']['ids']:
            raise ValueError('Selected checkpoint receipt mismatch')
        sources[method] = {'checkpoint': str(path), 'sha256': core.digest(path),
            'selected_epoch': state['epoch'], 'selection_supports': 5, 'selected_s5_mse': receipt['mse']}
    config = {'version': VERSION, 'test_read': False, 'optimizer_steps': 0, 'sources': sources,
        'base_manifest_sha256': core.digest(base / 'manifest.json'), 'run_binding_sha256': core.digest(source_run / 'binding.json'),
        'script_sha256': core.digest(__file__), 'supports': [3, 5, 8], 'recipients': 512, 'prefix_frames': 3, 'horizon': 12,
        'plan_hashes': {str(s): hashlib.sha256(plan.tobytes()).hexdigest() for s, plan in plans.items()},
        'interpretation': 'Original frozen Native versus adapted models includes fine-tuning, recurrent-history architecture and budget changes; only A versus Native-MQ isolates additional A changes'}
    if (out / 'config.json').exists() and core.read(out / 'config.json') != config:
        raise ValueError('Output binding differs')
    core.write(out / 'config.json', config)
    result = {'config': config, 'status': 'RUNNING', 'methods': {name: {'source': sources[name], 'supports': {}} for name in METHODS},
              'comparisons_by_supports': {}, 'test_read': False}
    began = time.perf_counter()
    for supports in (3, 5, 8):
        for method in METHODS:
            model = core.FineTuneModel(legacy).to(device)
            if method != 'Original-Native-Frozen':
                source = sources[method]
                if core.digest(source['checkpoint']) != source['sha256']:
                    raise ValueError('Checkpoint changed')
                state = torch.load(source['checkpoint'], map_location=device, weights_only=False)
                model.load_state_dict(state['model'], strict=True)
            else:
                # This is the old fixed head, not the v4.6 adapted Native checkpoint.
                if torch.count_nonzero(model.head.cell.weight_ih[:, -64:]).item() != 0:
                    raise ValueError('Original Native acquired a new recurrent history path')
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            metric = core.evaluate(model, data, plans[supports], device)
            if method == 'Original-Native-Frozen' and supports == 3:
                expected = core.read(base / 'runs/Native-U/selected_validation.json')['mse']
                if abs(metric['mse'] - expected) > 2e-5:
                    raise ValueError('Original frozen Native S3 did not reproduce')
                metric['original_s3_reproduction_absolute_error'] = abs(metric['mse'] - expected)
            if method != 'Original-Native-Frozen' and supports == 5:
                if abs(metric['mse'] - sources[method]['selected_s5_mse']) > 2e-5:
                    raise ValueError('Adapted S5 checkpoint did not reproduce')
            result['methods'][method]['supports'][str(supports)] = metric
            core.write(out / 'result_snapshot.json', result)
            core.write(out / 'summary.json', summary(result))
            core.emit('native_reference_eval', method=method, supports=supports, mse=metric['mse'])
            del model
        comparisons = {}
        for candidate in ('Native-MQ', 'A-MQ', 'A-MQ-Reg'):
            for reference in ('Original-Native-Frozen', 'Native-MQ'):
                if candidate == reference:
                    continue
                a, b = [result['methods'][name]['supports'][str(supports)]['mse'] for name in (candidate, reference)]
                comparisons[candidate + '_vs_' + reference] = {'candidate_mse': a, 'reference_mse': b,
                    'improvement_percent': 100 * (b - a) / b}
        result['comparisons_by_supports'][str(supports)] = comparisons
        core.write(out / 'summary.json', summary(result))
        core.emit('support_comparison_complete', supports=supports, comparisons=comparisons)
    result['status'] = 'COMPLETE'; result['seconds'] = time.perf_counter() - began
    core.write(out / 'result_snapshot.json', result)
    core.write(out / 'summary.json', summary(result))
    core.emit('comparison_complete', seconds=result['seconds'], out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_discovery_collision_v4_4'))
    parser.add_argument('--run', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_multiquery_v4_9'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / 'comparison.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.base, args.run, args.out, args.device)
