"""Lightweight read-only collection of Collision v4.9 progress and results.

Only an explicitly requested --write summary is created. Existing training,
20/40-epoch result snapshots, and earlier experiment results are never changed.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path


METHODS = ('Native-MQ', 'A-MQ', 'Random-MQ', 'A-MQ-Reg')


def read(path, errors, read_paths):
    path = Path(path).resolve()
    read_paths.add(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error': str(exc)})
        return None


def valid(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def history(item):
    value = (item.get('progress') or {}).get('history', [])
    return [row for row in value if isinstance(row, dict) and isinstance(row.get('epoch'), int)]


def gain(candidate, reference):
    delta = reference - candidate
    return {'mse_difference_reference_minus_candidate': delta,
            'relative_improvement_percent': 100 * delta / reference if reference > 0 else None}


def common_comparison(items):
    latest = {}
    for method, item in items.items():
        rows = history(item)
        if rows:
            latest[method] = max(row['epoch'] for row in rows)
        elif valid((item.get('initial_validation') or {}).get('mse')):
            latest[method] = 0
        else:
            return {'status': 'WAITING_FOR_ALL_METHODS'}
    budget = min(latest.values())
    best = {}
    for method, item in items.items():
        rows = [row for row in history(item) if 1 <= row['epoch'] <= budget]
        observed = [row['epoch'] for row in rows if valid(row.get('mse'))]
        if sorted(observed) != list(range(1, budget + 1)):
            return {'status': 'INCOMPLETE_OR_DUPLICATED_HISTORY', 'method': method, 'common_epoch': budget}
        candidates = [{'epoch': row['epoch'], 'mse': row['mse']} for row in rows]
        initial = (item.get('initial_validation') or {}).get('mse')
        if valid(initial):
            candidates.append({'epoch': 0, 'mse': initial})
        if not candidates:
            return {'status': 'NO_METRICS', 'method': method, 'common_epoch': budget}
        best[method] = min(candidates, key=lambda row: (row['mse'], row['epoch']))
    native, random = best['Native-MQ']['mse'], best['Random-MQ']['mse']
    return {'status': 'AVAILABLE', 'common_epoch': budget, 'latest_epochs': latest,
            'best_in_common_budget_including_epoch0': best,
            'a_mq_vs_native': gain(best['A-MQ']['mse'], native),
            'a_mq_vs_random': gain(best['A-MQ']['mse'], random),
            'a_mq_reg_vs_native': gain(best['A-MQ-Reg']['mse'], native),
            'a_mq_reg_vs_random_exploratory': gain(best['A-MQ-Reg']['mse'], random),
            'random_mq_vs_native': gain(random, native),
            'a_mq_reg_has_matched_random_regularization': False,
            'random_mq_semantics': 'wrong focal history in second supervised path; not auxiliary-only randomization'}


def check_consistency(items):
    configs = {method: item['config'] for method, item in items.items() if item['config']}
    initials = {method: item['initial_validation'] for method, item in items.items() if item['initial_validation']}
    common_fields = ('binding_sha256', 'seed', 'encoder_lr', 'head_lr', 'weight_decay',
                     'supports', 'queries_per_group', 'batch_groups', 'supervised_predictions_per_query',
                     'history_encoder_trainable', 'visual_frontend_frozen', 'validation_supports')
    differences = {}
    for field in common_fields:
        values = {method: config.get(field) for method, config in configs.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            differences[field] = values
    initial_mses = {method: initial['mse'] for method, initial in initials.items() if valid(initial.get('mse'))}
    spread = max(initial_mses.values()) - min(initial_mses.values()) if initial_mses else None
    ids = [initial.get('ids') for initial in initials.values()]
    same_ids = all(value == ids[0] for value in ids) if ids else None
    per_epoch = {}
    for method, item in items.items():
        for row in history(item):
            per_epoch.setdefault(row['epoch'], {})[method] = {
                'plan_sha256': row.get('plan_sha256'), 'query_exposures': row.get('query_exposures'),
                'prediction_exposures': row.get('prediction_exposures')}
    mismatches = []
    compared = 0
    for epoch, values in sorted(per_epoch.items()):
        if len(values) >= 2:
            compared += 1
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
                mismatches.append({'epoch': epoch, 'values': values})
    complete = len(configs) == len(METHODS) and len(initials) == len(METHODS)
    problem = bool(differences or mismatches or same_ids is False or (spread is not None and spread > 2e-5))
    return {'status': 'MISMATCH' if problem else 'PASS_AVAILABLE' if complete else 'PARTIAL_WAITING',
            'scope': 'declared common bindings, initial scores, and logged plans/exposures',
            'configs_available': len(configs), 'initials_available': len(initials),
            'shared_config_differences': differences, 'initial_mses': initial_mses,
            'initial_mse_spread': spread, 'same_initial_validation_ids': same_ids,
            'epochs_with_plan_and_exposure_comparison': compared,
            'plan_or_exposure_mismatches': mismatches}


def collect(out):
    out = Path(out).resolve()
    if not out.is_dir():
        raise FileNotFoundError('Missing output directory: ' + str(out))
    errors, read_paths = [], set()
    controller = read(out / 'controller_status.json', errors, read_paths) or {}
    binding = read(out / 'binding.json', errors, read_paths) or {}
    smoke = read(out / 'real_batch_smoke.json', errors, read_paths) or {}
    items = {method: {key: read(out / 'runs' / method / (key + '.json'), errors, read_paths)
                      for key in ('progress', 'selected_validation', 'initial_validation', 'config', 'complete')}
             for method in METHODS}
    compact = {}
    for method, item in items.items():
        progress, selected = item['progress'] or {}, item['selected_validation'] or {}
        rows = history(item)
        tail = rows[-5:]
        recent = [row['mse'] for row in tail if valid(row.get('mse'))]
        prior = [row['mse'] for row in rows[-10:-5] if valid(row.get('mse'))]
        current_mean = statistics.fmean(recent) if recent else None
        previous_mean = statistics.fmean(prior) if len(prior) == 5 else None
        compact[method] = {'status': progress.get('status', 'NOT_STARTED'),
            'epoch': progress.get('epoch'), 'actual_selected_epoch': selected.get('epoch'),
            'actual_selected_mse': selected.get('mse'),
            'initial_mse': (item['initial_validation'] or {}).get('mse'),
            'last_scores': [{key: row.get(key) for key in
                             ('epoch', 'mse', 'train_mse', 'main_mse', 'second_mse', 'weighted_aux', 'seconds')}
                            for row in tail],
            'last5_mean_mse': current_mean, 'previous5_mean_mse': previous_mean,
            'last5_relative_improvement_percent':
                100 * (previous_mean - current_mean) / previous_mean
                if previous_mean is not None and previous_mean > 0 and current_mean is not None else None,
            'training_seconds_total': sum(float(row.get('seconds', 0)) for row in rows)}
    snapshots = {}
    for budget in (20, 40):
        path = out / f'result_{budget}epochs.json'
        saved = read(path, errors, read_paths)
        if saved:
            snapshots[str(budget)] = {'path': str(path),
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'matched_budget_verification': saved.get('matched_budget_verification'),
                'selection': {method: {'epoch': value.get('epoch'), 'mse': value.get('mse')}
                              for method, value in saved.get('methods', {}).items()},
                'continuation_decision': saved.get('continuation_decision')}
    now = time.time()
    finished = controller.get('finished_at', now)
    started = controller.get('started_at')
    stages = controller.get('stages') or []
    jobs = stages[-1].get('jobs', []) if stages else []
    summary = {'collected_at': now, 'out': str(out), 'test_read': False,
        'stage': 'development_validation',
        'controller': {key: controller.get(key) for key in
                       ('status', 'pid', 'current_stage', 'completed_budget', 'max_epochs', 'error')},
        'controller_elapsed_seconds': finished - started if isinstance(started, (int, float)) else None,
        'current_jobs': [{key: job.get(key) for key in ('label', 'pid', 'gpu', 'status', 'exit_code', 'log')}
                         for job in jobs],
        'warm_start': {'path': binding.get('warm_start'), 'epoch': binding.get('warm_start_epoch'),
                       'sha256': (binding.get('file_sha256') or {}).get(binding.get('warm_start'))},
        'smoke': {key: smoke.get(key) for key in ('status', 'initial_mse', 'queries',
                  'shared_node_used_by_all_three_queries', 'warm_start_reloaded_after_smoke')},
        'methods': compact, 'common_budget_comparisons': common_comparison(items),
        'consistency': check_consistency(items), 'completed_result_snapshots': snapshots,
        'last_budget_decision': (controller.get('budget_decisions') or [None])[-1],
        'earlier_frozen_results_modified': False, 'read_errors': errors}
    return summary, read_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--write', help='Optional new JSON summary path; existing experiment inputs are protected')
    args = parser.parse_args()
    summary, inputs = collect(args.out)
    if args.write:
        path = Path(args.write).resolve()
        protected = {'binding.json', 'progress.json', 'config.json', 'complete.json', 'selected.pt',
                     'latest.pt', 'selected_validation.json', 'initial_validation.json',
                     'controller_status.json', 'result_20epochs.json', 'result_40epochs.json'}
        if path in inputs or path.name in protected:
            raise ValueError('Refusing to overwrite an experiment artifact with a summary')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        tmp.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
        os.replace(tmp, path)
        summary['summary_path'] = str(path)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
