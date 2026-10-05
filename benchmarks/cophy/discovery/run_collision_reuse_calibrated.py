"""Four-GPU Collision v4.7 calibrated relation-weight controller.

Owns only a new output directory and its children. It never inspects, resumes,
or signals previous dispatchers. A worker failure drains the current stage;
there is no automatic parameter change or replacement process.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


VERSION = 'collision-reuse-calibrated-v4.7-controller1'
METHODS = ('A-inv1', 'Random-inv1', 'A-inv2', 'Random-inv2')
IMPROVEMENT_THRESHOLD = 0.005


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    os.replace(tmp, path)


def emit(event, **fields):
    print(json.dumps({'time': time.time(), 'event': event, **fields},
                     ensure_ascii=False, allow_nan=False), flush=True)


def trend(progress, budget, method):
    """Require two complete five-epoch windows ending at this exact budget."""
    history = progress.get('history')
    if not isinstance(history, list):
        raise ValueError(f'{method}: progress history is missing or is not a list')
    indexed = {}
    for row in history:
        epoch = row.get('epoch')
        if not isinstance(epoch, int) or epoch in indexed:
            raise ValueError(f'{method}: invalid or duplicated history epoch {epoch!r}')
        indexed[epoch] = row
    required = list(range(budget - 9, budget + 1))
    missing = [epoch for epoch in required if epoch not in indexed]
    if missing or not indexed or max(indexed) != budget:
        raise ValueError(f'{method}: need complete epochs {required[0]}..{budget}; '
                         f'missing={missing}, latest={max(indexed) if indexed else None}')
    values = []
    for epoch in required:
        value = indexed[epoch].get('mse')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{method}: invalid validation mse at epoch {epoch}: {value!r}')
        values.append(float(value))
    previous = statistics.fmean(values[:5])
    recent = statistics.fmean(values[5:])
    relative = (previous - recent) / previous if previous > 0 else 0.0
    return {'previous_epochs': required[:5], 'recent_epochs': required[5:],
            'previous_mean_mse': previous, 'recent_mean_mse': recent,
            'relative_improvement': relative,
            'relative_improvement_threshold': IMPROVEMENT_THRESHOLD,
            'still_improving': relative >= IMPROVEMENT_THRESHOLD}


def verified_baseline(out, baseline_out):
    """Bind to the immutable Native-FT 20-epoch result, even if Native later resumes."""
    baseline_out = Path(baseline_out).resolve()
    old = read(baseline_out / 'binding.json')
    new = read(Path(out) / 'binding.json')
    comparisons = {
        'source_sha256': (old['source']['sha256'], new['source']['sha256']),
        'head_sha256': (old['head']['sha256'], new['head']['sha256']),
        'base_manifest_sha256': (old['base_manifest_sha256'], new['base_manifest_sha256'])}
    for split in ('train', 'val'):
        for field in ('cache_sha256', 'input_sha256', 'target_sha256'):
            comparisons[f'{split}.{field}'] = (old['splits'][split][field], new['splits'][split][field])
    unequal = {key: values for key, values in comparisons.items() if values[0] != values[1]}
    if unequal:
        raise ValueError('Native baseline initialization/input binding differs: ' + json.dumps(unequal))
    run = baseline_out / 'runs' / 'Native-FT'
    progress = read(run / 'progress.json')
    epoch = progress.get('epoch')
    if not isinstance(epoch, int) or epoch < 20:
        raise ValueError(f'Native baseline must have completed at least 20 epochs; found {epoch!r}')
    snapshot_path = baseline_out / 'result_20epochs.json'
    snapshot = read(snapshot_path)
    if snapshot.get('budget') != 20:
        raise ValueError('Native baseline snapshot is not bound to the 20-epoch budget')
    selected = snapshot['methods']['Native-FT']
    if (not isinstance(selected.get('epoch'), int) or not 0 <= selected['epoch'] <= 20 or
            not isinstance(selected.get('mse'), (int, float)) or
            not math.isfinite(selected['mse']) or selected['mse'] < 0):
        raise ValueError('Invalid Native 20-epoch selected result')
    history = {row['epoch']: row for row in progress['history'] if 1 <= row['epoch'] <= 20}
    if set(history) != set(range(1, 21)):
        raise ValueError('Native baseline lacks complete epoch 1..20 history')
    plans = {str(i): history[i].get('support_plan_sha256') for i in range(1, 21)}
    if not all(plans.values()):
        raise ValueError('Native baseline lacks logged support-plan hashes')
    return {'method': 'Native-FT', 'budget': 20, 'run_path': str(run),
            'snapshot_path': str(snapshot_path), 'snapshot_sha256': digest(snapshot_path),
            'binding_path': str(baseline_out / 'binding.json'),
            'binding_sha256': digest(baseline_out / 'binding.json'),
            'progress_epoch_at_check': epoch, 'selected': selected,
            'initialization_and_inputs_match': True,
            'verified_bindings': {key: values[0] for key, values in comparisons.items()},
            'support_plan_sha256_by_epoch': plans}


def main(out, base, protocol, max_epochs, baseline_out):
    out, base, protocol = Path(out).resolve(), Path(base).resolve(), Path(protocol).resolve()
    if out == base or base in out.parents or out in base.parents:
        raise ValueError('Output and existing base must be separate, non-nested directories')
    code = Path(__file__).resolve().with_name('collision_reuse_calibrated.py')
    dependency = code.with_name('collision_xep.py')
    shared_core = code.with_name('collision_reuse_ft.py')
    if not base.is_dir():
        raise FileNotFoundError(f'Existing Collision base directory is missing: {base}')
    for path in (code, dependency, shared_core, protocol):
        if not path.is_file():
            raise FileNotFoundError(f'Required code/protocol file is missing: {path}')
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'controller.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError('Another calibrated controller owns this output directory') from exc
    status_path = out / 'controller_status.json'
    if status_path.exists() or (out / 'runs').exists():
        lock.close()
        raise RuntimeError('Refusing a duplicate controller in a previously used output. '
                           'Inspect prior jobs/checkpoints before arranging recovery; '
                           'existing workers are not signalled.')
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    bindings = {str(path): digest(path) for path in
                (Path(__file__).resolve(), code, dependency, shared_core, protocol)}
    record = {'version': VERSION, 'pid': os.getpid(), 'started_at': time.time(),
              'out': str(out), 'base': str(base), 'max_epochs': max_epochs,
              'baseline_out': str(Path(baseline_out).resolve()),
              'code_sha256': bindings[str(code)], 'protocol_sha256': bindings[str(protocol)],
              'file_bindings': bindings, 'test_read': False,
              'old_dispatchers_resumed': False, 'old_workers_signalled': False,
              'stages': [], 'budget_decisions': [], 'status': 'RUNNING'}
    write(status_path, record)

    def stage(name, tasks):
        for path, expected in bindings.items():
            if digest(path) != expected:
                raise RuntimeError('Code/protocol changed before dispatch: ' + path)
        entry = {'stage': name, 'started_at': time.time(), 'status': 'RUNNING', 'jobs': []}
        record['stages'].append(entry)
        record['current_stage'] = name
        write(status_path, record)
        live = []
        launch_error = None
        for label, args, gpu in tasks:
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                       CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
            command = [sys.executable, '-u', str(code), *args,
                       '--out', str(out), '--base', str(base)]
            log_path = logs / (label + '.log')
            log = open(log_path, 'a', buffering=1)
            try:
                proc = subprocess.Popen(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, cwd=str(code.parent))
            except Exception as exc:
                log.close()
                launch_error = f'{label}: {type(exc).__name__}: {exc}'
                entry['jobs'].append({'label': label, 'gpu': gpu, 'command': command,
                                      'log': str(log_path), 'status': 'LAUNCH_FAILED',
                                      'exit_code': None, 'error': launch_error})
                write(status_path, record)
                break
            job = {'label': label, 'pid': proc.pid, 'gpu': gpu, 'command': command,
                   'started_at': time.time(), 'status': 'RUNNING', 'log': str(log_path)}
            entry['jobs'].append(job)
            live.append((proc, log, job))
            write(status_path, record)
        emit('stage_started', stage=name, jobs=entry['jobs'])
        # A failure never cancels the other normal workers already in this stage.
        while live:
            for proc, log, job in live[:]:
                rc = proc.poll()
                if rc is not None:
                    job.update(status='COMPLETE' if rc == 0 else 'FAILED',
                               exit_code=rc, finished_at=time.time())
                    log.close()
                    live.remove((proc, log, job))
                    write(status_path, record)
            if live:
                time.sleep(5)
        failed = launch_error is not None or any(job.get('exit_code') != 0 for job in entry['jobs'])
        entry.update(status='FAILED' if failed else 'COMPLETE', finished_at=time.time())
        write(status_path, record)
        write(out / (name + '_stage.json'), entry)
        if failed:
            raise RuntimeError(f'Stage {name} failed; all launched workers have finished. '
                               'Inspect individual logs; no automatic retuning/relaunch.')

    try:
        stage('prepare', [('prepare', ['prepare'], None)])
        baseline = verified_baseline(out, baseline_out)
        write(out / 'baseline_native20.json', baseline)
        record['baseline'] = {key: baseline[key] for key in
                              ('method', 'budget', 'run_path', 'snapshot_path', 'snapshot_sha256')}
        record['baseline']['selected_epoch'] = baseline['selected']['epoch']
        record['baseline']['selected_mse'] = baseline['selected']['mse']
        write(status_path, record)
        emit('baseline_verified', **record['baseline'])
        stage('smoke', [('smoke', ['smoke'], 0)])
        for budget in (20, 40, 60):
            if budget > max_epochs:
                break
            stage('training_' + str(budget), [
                (f'train_{method}_{budget}', ['train', '--method', method,
                 '--device', 'cuda:0', '--epochs', str(budget)], gpu)
                for gpu, method in enumerate(METHODS)])
            selected = {method: read(out / 'runs' / method / 'selected_validation.json')
                        for method in METHODS}
            for method, value in selected.items():
                epoch, mse = value.get('epoch'), value.get('mse')
                if (not isinstance(epoch, int) or not 0 <= epoch <= budget or
                        not isinstance(mse, (float, int)) or not math.isfinite(mse) or mse < 0):
                    raise ValueError(f'{method}: invalid selected validation record at budget {budget}')
            progress = {method: read(out / 'runs' / method / 'progress.json') for method in METHODS}
            # Preserve actual results even if a missing history prevents continuation.
            snapshot = {'version': VERSION, 'budget': budget, 'test_read': False,
                        'file_bindings': bindings, 'methods': selected, 'baseline': baseline,
                        'progress_epochs': {method: item.get('epoch') for method, item in progress.items()}}
            write(out / f'result_{budget}epochs.json', snapshot)
            for method, item in progress.items():
                for row in item.get('history', []):
                    if (1 <= row['epoch'] <= budget and row.get('support_plan_sha256') !=
                            baseline['support_plan_sha256_by_epoch'][str(row['epoch'])]):
                        raise ValueError(f'{method}: support plan differs from Native baseline at epoch {row["epoch"]}')
            base_mse = baseline['selected']['mse']
            snapshot['improvement_over_native_percent'] = {
                method: 100 * (base_mse - value['mse']) / base_mse if base_mse > 0 else None
                for method, value in selected.items()}
            snapshot['support_plans_match_native'] = True
            trends = {method: trend(progress[method], budget, method) for method in METHODS}
            continuing_signal = any(item['still_improving'] for item in trends.values())
            will_continue = continuing_signal and budget < max_epochs
            reason = ('CONTINUE_ALL_METHODS' if will_continue else
                      'STOP_AT_CONFIGURED_CAP' if budget >= max_epochs else 'STOP_PLATEAU')
            decision = {'budget': budget, 'trends': trends,
                        'continue_recommended': continuing_signal,
                        'continue_authorized_by_cap': budget < max_epochs,
                        'will_continue': will_continue, 'decision': reason,
                        'next_budget': budget + 20 if will_continue else None,
                        'uniform_budget_for_all_methods': True}
            snapshot['continuation_decision'] = decision
            write(out / f'result_{budget}epochs.json', snapshot)
            write(out / f'decision_{budget}epochs.json', decision)
            record['budget_decisions'].append(decision)
            record['completed_budget'] = budget
            write(status_path, record)
            emit('budget_complete', budget=budget,
                 methods={method: {'epoch': row['epoch'], 'mse': row['mse']}
                          for method, row in selected.items()}, decision=decision)
            if not will_continue:
                break
        record['status'] = 'COMPLETE'
        record['summary'] = {method: read(out / 'runs' / method / 'complete.json') for method in METHODS}
        emit('adaptation_complete', budget=record['completed_budget'], summary=record['summary'])
    except BaseException as exc:
        record.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at'] = time.time()
        write(status_path, record)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--baseline-out', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_reuse_ft_v4_6'))
    parser.add_argument('--max-epochs', type=int, choices=(20,), default=20)
    args = parser.parse_args()
    main(args.out, args.base, args.protocol, args.max_epochs, args.baseline_out)
