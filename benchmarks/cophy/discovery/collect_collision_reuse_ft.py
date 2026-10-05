"""Read live Collision v4.6 receipts without changing training state.

Prints compact JSON. Optional --write-snapshot stores complete selected
per-recipient receipts plus progress in result_snapshot.json; --gpu adds a
best-effort machine-level GPU snapshot. Never loads a checkpoint or dataset.
"""
import argparse
import csv
import io
import json
import math
import os
import subprocess
import time
from pathlib import Path


METHODS = ('Native-FT', 'A-weak', 'A-medium', 'Random-weak')


def read_optional(path, errors):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
        return None


def metric(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def gpu_snapshot():
    fields = ['index', 'uuid', 'name', 'utilization.gpu', 'memory.used', 'memory.total']
    result = {'scope': 'whole_machine_including_other_jobs'}
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode:
            return {**result, 'available': False, 'error': proc.stderr.strip()[:500],
                    'exit_code': proc.returncode}
        result.update(available=True, gpus=[dict(zip(fields, [v.strip() for v in row]))
                      for row in csv.reader(io.StringIO(proc.stdout)) if row])
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5, check=False)
        if apps.returncode == 0:
            result['compute_processes'] = [dict(zip(('pid', 'gpu_uuid', 'used_memory'),
                                           [v.strip() for v in row]))
                                          for row in csv.reader(io.StringIO(apps.stdout)) if row]
        else:
            result['process_query_error'] = apps.stderr.strip()[:300]
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(available=False, error=f'{type(exc).__name__}: {exc}')
    return result


def history_rows(item):
    rows = (item.get('progress') or {}).get('history', [])
    return [row for row in rows if isinstance(row, dict) and
            isinstance(row.get('epoch'), int)] if isinstance(rows, list) else []


def plan_agreement(methods, smoke):
    plans = {}
    missing = []
    for method, item in methods.items():
        for row in history_rows(item):
            value = row.get('support_plan_sha256')
            if value:
                plans.setdefault(row['epoch'], {})[method] = value
            else:
                missing.append({'method': method, 'epoch': row['epoch']})
    checked = {epoch: values for epoch, values in plans.items() if len(values) >= 2}
    mismatches = [{'epoch': epoch, 'method_hashes': values}
                  for epoch, values in sorted(checked.items()) if len(set(values.values())) > 1]
    common = [epoch for epoch, values in plans.items() if len(values) == len(METHODS)]
    latest = max(common) if common else None
    return {'status': 'HASH_MISMATCH' if mismatches else
            'MATCH_FOR_AVAILABLE_COMPARISONS' if checked else 'WAITING_FOR_COMPARABLE_HISTORY',
            'scope': 'logged support plan hashes; not a fresh data audit',
            'epochs_compared': len(checked), 'latest_all_method_epoch': latest,
            'latest_all_method_hash': (next(iter(plans[latest].values()))
                                      if latest is not None and len(set(plans[latest].values())) == 1 else None),
            'mismatches': mismatches, 'missing_hashes': missing,
            'smoke_train_plan_sha256': (smoke or {}).get('support_plan_sha256'),
            'smoke_validation_plan_sha256': (smoke or {}).get('validation_plan_sha256')}


def common_budget_comparison(methods):
    """Avoid comparing one worker at epoch 18 against another at epoch 16."""
    latest = {}
    for method, item in methods.items():
        rows = history_rows(item)
        if rows:
            latest[method] = max(row['epoch'] for row in rows)
        elif metric((item.get('initial_validation') or {}).get('mse')):
            latest[method] = 0
        else:
            return {'status': 'WAITING_FOR_ALL_METHODS', 'available_latest_epochs': latest}
    budget = min(latest.values())
    best = {}
    for method, item in methods.items():
        rows = [row for row in history_rows(item) if row['epoch'] <= budget]
        observed = {row['epoch'] for row in rows}
        missing = [epoch for epoch in range(1, budget + 1) if epoch not in observed]
        if missing:
            return {'status': 'INCOMPLETE_HISTORY', 'method': method,
                    'missing_epochs': missing, 'common_epoch': budget}
        candidates = [{'epoch': row['epoch'], 'mse': row['mse']} for row in rows
                      if metric(row.get('mse'))]
        initial = item.get('initial_validation') or {}
        if metric(initial.get('mse')):
            candidates.append({'epoch': 0, 'mse': initial['mse']})
        if not candidates:
            return {'status': 'MISSING_METRICS', 'method': method, 'common_epoch': budget}
        best[method] = min(candidates, key=lambda row: (row['mse'], row['epoch']))
    def improvement(candidate, reference):
        a, b = best[candidate]['mse'], best[reference]['mse']
        return None if b <= 0 else 100 * (b - a) / b
    return {'status': 'AVAILABLE', 'common_epoch': budget,
            'selection': 'best validation mse through common epoch, including epoch0',
            'each_worker_latest_epoch': latest, 'best_at_common_budget': best,
            'a_weak_vs_native_improvement_percent': improvement('A-weak', 'Native-FT'),
            'a_weak_vs_random_improvement_percent': improvement('A-weak', 'Random-weak'),
            'a_medium_vs_native_improvement_percent': improvement('A-medium', 'Native-FT'),
            'a_medium_matched_random_available': False,
            'stage': 'development_validation_not_sealed_test'}


def collect(out, include_gpu=False, write_snapshot=False):
    out = Path(out).resolve()
    if not out.is_dir():
        raise FileNotFoundError('Output directory does not exist: ' + str(out))
    errors = []
    controller = read_optional(out / 'controller_status.json', errors)
    smoke = read_optional(out / 'real_batch_smoke.json', errors)
    binding = read_optional(out / 'binding.json', errors)
    methods = {}
    for method in METHODS:
        run = out / 'runs' / method
        methods[method] = {key: read_optional(run / (key + '.json'), errors)
                           for key in ('config', 'progress', 'selected_validation',
                                       'initial_validation', 'complete')}
    plans = plan_agreement(methods, smoke)
    comparisons = common_budget_comparison(methods)
    compact_methods = {}
    for method, item in methods.items():
        progress, selected = item['progress'] or {}, item['selected_validation'] or {}
        recent = history_rows(item)[-5:]
        compact_methods[method] = {
            'status': progress.get('status', 'NOT_STARTED'), 'epoch': progress.get('epoch'),
            'selected_epoch': selected.get('epoch'), 'selected_mse': selected.get('mse'),
            'initial_mse': (item['initial_validation'] or {}).get('mse'),
            'lambda_target': (item['config'] or {}).get('lambda_target'),
            'recent_history': [{key: row.get(key) for key in
                                ('epoch', 'mse', 'train_mse', 'lambda', 'aux', 'weighted_aux', 'seconds')}
                               for row in recent],
            'latest_first_batch': recent[-1].get('first_batch') if recent else None}
    stages = (controller or {}).get('stages', [])
    current = stages[-1] if stages else {}
    summary = {'collected_at': time.time(), 'out': str(out), 'test_read': False,
               'controller': {key: (controller or {}).get(key) for key in
                              ('status', 'pid', 'current_stage', 'completed_budget', 'max_epochs', 'error')},
               'current_jobs': current.get('jobs', []),
               'last_budget_decision': ((controller or {}).get('budget_decisions') or [None])[-1],
               'smoke': {key: (smoke or {}).get(key) for key in
                          ('status', 'initial_prediction_max_error', 'source_code_max_error',
                           'initial_val_mse', 'old_native_val_mse')},
               'initialization': {key: (binding or {}).get(key) for key in
                                  ('source', 'head', 'code_sha256', 'dependency_sha256')},
               'methods': compact_methods, 'plan_agreement': plans,
               'comparisons': comparisons, 'read_errors': errors}
    # Source config can be large; keep only its actual identity in stdout.
    source = summary['initialization'].get('source')
    if isinstance(source, dict):
        summary['initialization']['source'] = {key: source.get(key) for key in ('path', 'sha256', 'epoch')}
    if include_gpu:
        summary['gpu_snapshot'] = gpu_snapshot()
    if write_snapshot:
        path = out / 'result_snapshot.json'
        snapshot = {'summary': summary, 'controller': controller, 'binding': binding,
                    'smoke': smoke, 'methods': methods, 'test_read': False,
                    'snapshot_is_nonatomic_across_workers': True}
        tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2))
        os.replace(tmp, path)
        summary['snapshot_path'] = str(path)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', action='store_true', help='Best-effort nvidia-smi; does not fail collection')
    parser.add_argument('--write-snapshot', action='store_true', help='Write full per-recipient selected receipts')
    args = parser.parse_args()
    print(json.dumps(collect(args.out, args.gpu, args.write_snapshot),
                     ensure_ascii=False, allow_nan=False))
