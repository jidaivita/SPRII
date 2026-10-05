"""Compact live v4.7 collection, with common-budget Native/Random comparisons.

Reads training receipts only. Writes a separate result_snapshot.json containing
full per-recipient selections; never changes checkpoints, logs, or progress.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path


METHODS = ('A-inv1', 'Random-inv1', 'A-inv2', 'Random-inv2')


def load(path, errors):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error': str(exc)})
        return None


def rows(item):
    history = (item.get('progress') or {}).get('history', [])
    return [row for row in history if isinstance(row, dict) and isinstance(row.get('epoch'), int)]


def usable(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def compare(methods, native):
    all_items = {**methods, 'Native-FT': native}
    epochs = {}
    for name, item in all_items.items():
        history = rows(item)
        if history:
            epochs[name] = max(row['epoch'] for row in history)
        elif usable((item.get('initial_validation') or {}).get('mse')):
            epochs[name] = 0
        else:
            return {'status': 'WAITING_FOR_ALL_METHODS', 'available_epochs': epochs}
    budget = min(20, *epochs.values())
    selected = {}
    for name, item in all_items.items():
        history = [row for row in rows(item) if 1 <= row['epoch'] <= budget]
        observed = {row['epoch'] for row in history if usable(row.get('mse'))}
        missing = sorted(set(range(1, budget + 1)) - observed)
        if missing:
            return {'status': 'INCOMPLETE_HISTORY', 'method': name,
                    'common_epoch': budget, 'missing_epochs': missing}
        candidates = [{'epoch': row['epoch'], 'mse': row['mse']} for row in history
                      if usable(row.get('mse'))]
        initial = (item.get('initial_validation') or {}).get('mse')
        if usable(initial):
            candidates.append({'epoch': 0, 'mse': initial})
        if not candidates:
            return {'status': 'NO_METRICS', 'method': name, 'common_epoch': budget}
        selected[name] = min(candidates, key=lambda row: (row['mse'], row['epoch']))
    def gain(candidate, reference):
        first, second = selected[candidate]['mse'], selected[reference]['mse']
        return {'candidate': candidate, 'reference': reference,
                'mse_difference_reference_minus_candidate': second - first,
                'improvement_percent': 100 * (second - first) / second if second > 0 else None}
    return {'status': 'AVAILABLE', 'common_epoch': budget, 'latest_completed_epochs': epochs,
            'selection': 'best validation mse through common epoch including common epoch0',
            'best_at_common_budget': selected,
            'a_inv1_vs_native': gain('A-inv1', 'Native-FT'),
            'a_inv2_vs_native': gain('A-inv2', 'Native-FT'),
            'a_inv1_vs_random_inv1': gain('A-inv1', 'Random-inv1'),
            'a_inv2_vs_random_inv2': gain('A-inv2', 'Random-inv2')}


def main(out, baseline_out):
    out = Path(out).resolve()
    baseline_out = Path(baseline_out).resolve()
    if not out.is_dir():
        raise FileNotFoundError('Output directory is missing: ' + str(out))
    errors = []
    controller = load(out / 'controller_status.json', errors)
    baseline = load(out / 'baseline_native20.json', errors)
    binding = load(out / 'binding.json', errors)
    smoke = load(out / 'real_batch_smoke.json', errors)
    native_run = Path((baseline or {}).get('run_path', str(baseline_out / 'runs' / 'Native-FT')))
    native = {key: load(native_run / (key + '.json'), errors)
              for key in ('progress', 'initial_validation', 'config')}
    native['selected_validation_at20'] = (baseline or {}).get('selected')
    methods = {}
    compact = {}
    for method in METHODS:
        run = out / 'runs' / method
        item = {key: load(run / (key + '.json'), errors)
                for key in ('progress', 'selected_validation', 'initial_validation', 'config', 'complete')}
        methods[method] = item
        progress, selected = item['progress'] or {}, item['selected_validation'] or {}
        history = rows(item)
        compact[method] = {'status': progress.get('status', 'NOT_STARTED'),
                           'epoch': progress.get('epoch'),
                           'selected_epoch': selected.get('epoch'), 'selected_mse': selected.get('mse'),
                           'initial_mse': (item['initial_validation'] or {}).get('mse'),
                           'last_history_row': history[-1] if history else None}
    plans = (baseline or {}).get('support_plan_sha256_by_epoch', {})
    checks, mismatches = 0, []
    for method, item in methods.items():
        for row in rows(item):
            expected = plans.get(str(row['epoch']))
            if expected:
                checks += 1
                if row.get('support_plan_sha256') != expected:
                    mismatches.append({'method': method, 'epoch': row['epoch'],
                                       'actual': row.get('support_plan_sha256'), 'native': expected})
    summary = {'collected_at': time.time(), 'out': str(out), 'test_read': False,
               'controller': {key: (controller or {}).get(key) for key in
                              ('status', 'pid', 'current_stage', 'completed_budget', 'error')},
               'smoke': {key: (smoke or {}).get(key) for key in
                          ('status', 'initial_val_mse', 'old_native_val_mse',
                           'initial_prediction_max_error', 'source_code_max_error')},
               'native_baseline': {'run_path': str(native_run), 'budget': 20,
                    'selected_epoch': ((baseline or {}).get('selected') or {}).get('epoch'),
                    'selected_mse': ((baseline or {}).get('selected') or {}).get('mse'),
                    'initialization_and_inputs_match': (baseline or {}).get('initialization_and_inputs_match')},
               'methods': compact, 'comparisons': compare(methods, native),
               'support_plans': {'comparisons_against_native': checks,
                    'status': 'MISMATCH' if mismatches else 'MATCH_AVAILABLE' if checks else 'WAITING',
                    'mismatches': mismatches}, 'read_errors': errors,
               'snapshot_path': str(out / 'result_snapshot.json')}
    snapshot = {'summary': summary, 'controller': controller, 'binding': binding,
                'smoke': smoke, 'baseline': baseline, 'native': native,
                'methods': methods, 'test_read': False,
                'snapshot_is_nonatomic_across_workers': True}
    path = out / 'result_snapshot.json'
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2))
    os.replace(tmp, path)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--baseline-out', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_reuse_ft_v4_6'))
    args = parser.parse_args()
    print(json.dumps(main(args.out, args.baseline_out), ensure_ascii=False, allow_nan=False))
