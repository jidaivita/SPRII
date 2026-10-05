"""Run one scene's v5.1 source training and complete frozen-readout chain.

This controller owns only its new output tree. Source and readout programs must
already exist before dispatch, and all dependencies are hash-bound. It never
signals old workers, retrains the existing Native source, or extends source
training beyond the authorized 50 epochs.
"""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


VERSION = 'source-formation-v5.1-controller1'
SOURCE_METHODS = ('Cross-only', 'Align-only', 'Both-new', 'Random-Both-new')
READOUT_METHODS = ('Native', *SOURCE_METHODS)
SUPPORTS = (3, 5, 8)
SOURCE_EPOCHS = 50
HEAD_EPOCHS = 100
SOURCE_DEPENDENCIES = (
    'cf_learning/model.py', 'cophy_adapter.py', 'cophy_protocol.py',
    'cophy_relations.py', 'cophy_training.py', 'cophy_fields.py', 'dataloaders/utils.py',
    'derendering/model.py',
)


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
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    os.replace(temporary, path)


def emit(event, **fields):
    print(json.dumps({'time': time.time(), 'event': event, **fields},
                     ensure_ascii=False, allow_nan=False), flush=True)


def scalar(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def proc_start_ticks(pid):
    path = Path('/proc') / str(pid) / 'stat'
    # The command field can contain spaces or parentheses; fields after it
    # begin at process-stat field 3, and starttime is field 22.
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        # A short-lived child may already have exited; poll() records its code.
        return None


class Controller:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.out = Path(args.out).resolve()
        self.protocol = Path(args.protocol).resolve()
        self.code_dir = Path(__file__).resolve().parent
        self.source_code = self.code_dir / 'source_formation_v51.py'
        self.readout_code = self.code_dir / 'source_formation_readout_v51.py'
        self.base = (Path(args.base).resolve() if args.base else self.root /
                     ('xep_discovery_balls_v4_1' if args.scene == 'balls'
                      else 'xep_discovery_collision_v4_4'))
        if self.out == self.base or self.base in self.out.parents or self.out in self.base.parents:
            raise ValueError('New output must be separate from the existing frozen-readout base')
        if not self.root.is_dir() or not self.base.is_dir():
            raise FileNotFoundError('Existing shared root or scene readout base is missing')
        dependencies = [Path(__file__).resolve(), self.source_code, self.readout_code,
                        self.protocol, self.code_dir / 'xep_discovery.py',
                        self.code_dir / 'collision_xep.py']
        dependencies += [self.root / 'source' / name for name in SOURCE_DEPENDENCIES]
        dependencies += [Path(name).resolve() for name in args.dependency]
        dependencies = list(dict.fromkeys(dependencies))
        for path in dependencies:
            if not path.is_file():
                raise FileNotFoundError('Required complete-pipeline dependency is missing: ' + str(path))
        self.bindings = {str(path): digest(path) for path in dependencies}
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.out / 'controller.lock', 'a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError('A controller already owns this output directory') from exc
        self.status_path = self.out / 'controller_status.json'
        if (self.status_path.exists() or (self.out / 'source').exists() or
                (self.out / 'readout').exists()):
            self.lock.close()
            raise RuntimeError('Refusing duplicate dispatch in a previously used output. '
                               'Inspect existing processes/checkpoints before explicit recovery.')
        self.source_out = self.out / 'source'
        self.readout_out = self.out / 'readout'
        self.logs = self.out / 'logs'
        self.logs.mkdir(exist_ok=True)
        self.record = {
            'version': VERSION, 'scene': args.scene, 'status': 'RUNNING',
            'pid': os.getpid(), 'process_start_ticks': proc_start_ticks(os.getpid()),
            'hostname': socket.gethostname(), 'started_at': time.time(),
            'root': str(self.root), 'out': str(self.out), 'base': str(self.base),
            'source_out': str(self.source_out), 'readout_out': str(self.readout_out),
            'source_methods': SOURCE_METHODS, 'readout_methods': READOUT_METHODS,
            'source_epochs': SOURCE_EPOCHS, 'head_epochs_per_support': HEAD_EPOCHS,
            'supports': SUPPORTS, 'gpus': args.gpus, 'threads_per_worker': args.threads,
            'launcher_handle': args.launcher_handle,
            'file_bindings': self.bindings, 'test_read': False,
            'old_workers_signalled': False, 'native_source_retrained': False,
            'stages': [], 'source_result': None, 'readout_results': {},
        }
        self.save()

    def save(self):
        self.record['last_updated_at'] = time.time()
        write(self.status_path, self.record)

    def verify_bindings(self):
        for path, expected in self.bindings.items():
            if digest(path) != expected:
                raise RuntimeError('A bound dependency changed before dispatch: ' + path)

    def source_task(self, command, label, **options):
        args = [command, '--scene', self.args.scene, '--out', str(self.source_out),
                '--root', str(self.root)]
        for key, value in options.items():
            args.extend(['--' + key.replace('_', '-'), str(value)])
        return {'label': label, 'program': self.source_code, 'args': args}

    def readout_task(self, command, label, **options):
        args = [command, '--scene', self.args.scene, '--out', str(self.readout_out),
                '--source-root', str(self.source_out), '--base', str(self.base)]
        for key, value in options.items():
            args.extend(['--' + key.replace('_', '-'), str(value)])
        return {'label': label, 'program': self.readout_code, 'args': args}

    def stage(self, name, tasks, gpu=True):
        """A bounded queue fills the next free GPU; failures drain this stage."""
        self.verify_bindings()
        tasks = list(tasks)
        if not tasks or len({task['label'] for task in tasks}) != len(tasks):
            raise ValueError('Stage must have nonempty, uniquely labelled tasks')
        entry = {'stage': name, 'status': 'RUNNING', 'started_at': time.time(),
                 'gpu_stage': gpu, 'jobs': [], 'queued_labels': [t['label'] for t in tasks]}
        self.record['stages'].append(entry)
        self.record['current_stage'] = name
        self.save()
        queue = deque(tasks)
        free = deque(self.args.gpus if gpu else [None])
        live = []

        def launch(task, device):
            command = [sys.executable, '-u', str(task['program']), *task['args']]
            if device is not None:
                command.extend(['--device', 'cuda:0'])
            environment = dict(os.environ, OMP_NUM_THREADS=str(self.args.threads),
                MKL_NUM_THREADS=str(self.args.threads), OPENBLAS_NUM_THREADS=str(self.args.threads),
                PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='' if device is None else str(device))
            log_path = self.logs / (task['label'] + '.log')
            job = {'label': task['label'], 'job_id': f'{self.args.scene}/{name}/{task["label"]}',
                   'gpu': device, 'command': command, 'log': str(log_path),
                   'started_at': time.time(), 'status': 'STARTING'}
            entry['jobs'].append(job)
            self.save()
            log = open(log_path, 'a', buffering=1)
            try:
                process = subprocess.Popen(command, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, cwd=str(self.code_dir))
            except Exception as exc:
                log.close()
                job.update(status='LAUNCH_FAILED', exit_code=None, finished_at=time.time(),
                           error=f'{type(exc).__name__}: {exc}')
                self.save()
                free.append(device)
                return
            job.update(status='RUNNING', pid=process.pid,
                       process_start_ticks=proc_start_ticks(process.pid))
            live.append((process, log, job, device))
            self.save()
            emit('worker_started', stage=name, **job)

        while queue or live:
            while queue and free:
                task, device = queue.popleft(), free.popleft()
                entry['queued_labels'] = [t['label'] for t in queue]
                launch(task, device)
            for process, log, job, device in live[:]:
                code = process.poll()
                if code is not None:
                    job.update(status='COMPLETE' if code == 0 else 'FAILED',
                               exit_code=code, finished_at=time.time())
                    log.close()
                    live.remove((process, log, job, device))
                    free.append(device)
                    self.save()
                    emit('worker_finished', stage=name, label=job['label'],
                         pid=job['pid'], exit_code=code)
            entry['last_process_poll_at'] = time.time()
            entry['live_pids'] = [process.pid for process, _, _, _ in live]
            self.save()
            if live:
                time.sleep(5)
        failed = [job['label'] for job in entry['jobs'] if job.get('exit_code') != 0]
        entry.update(status='FAILED' if failed else 'COMPLETE', finished_at=time.time(),
                     failed_jobs=failed)
        self.save()
        write(self.out / (name + '_stage.json'), entry)
        if failed:
            raise RuntimeError(f'Stage {name} failed in {failed}; all sibling jobs in this '
                               'stage have finished. No automatic replacement or retuning.')
        emit('stage_complete', stage=name)

    def collect_source(self):
        methods = {}
        for method in SOURCE_METHODS:
            path = self.source_out / 'runs' / method
            complete = read(path / 'complete.json')
            selected = read(path / 'selected_validation.json')
            if complete.get('status') != 'COMPLETE' or complete.get('epochs') != SOURCE_EPOCHS:
                raise ValueError(f'{method}: source completion does not establish 50 epochs')
            epoch, mse = selected.get('epoch'), selected.get('mse')
            if (not isinstance(epoch, int) or not 0 <= epoch <= SOURCE_EPOCHS or
                    not scalar(mse) or mse < 0 or not (path / 'selected.pt').is_file()):
                raise ValueError(f'{method}: invalid selected source result')
            methods[method] = {'complete': complete, 'selected': selected,
                'checkpoint': str(path / 'selected.pt'),
                'checkpoint_sha256': digest(path / 'selected.pt')}
        result = {'scene': self.args.scene, 'source_epochs': SOURCE_EPOCHS,
                  'test_read': False, 'methods': methods}
        write(self.out / 'source_50epochs.json', result)
        self.record['source_result'] = {
            'path': str(self.out / 'source_50epochs.json'),
            'methods': {method: {'epoch': item['selected']['epoch'],
                                'mse': item['selected']['mse'],
                                'checkpoint': item['checkpoint'],
                                'checkpoint_sha256': item['checkpoint_sha256']}
                        for method, item in methods.items()}}
        self.save()
        return result

    def collect_readout(self, supports):
        methods, ids = {}, None
        for method in READOUT_METHODS:
            path = self.readout_out / 'runs' / f'S{supports}' / method
            complete = read(path / 'complete.json')
            selected = read(path / 'selected_validation.json')
            if complete.get('status') != 'COMPLETE' or complete.get('epochs') != HEAD_EPOCHS:
                raise ValueError(f'{method}/S{supports}: head completion does not establish 100 epochs')
            epoch, mse = selected.get('epoch'), selected.get('mse')
            current_ids, values = selected.get('ids'), selected.get('per_recipient_mse')
            if (not isinstance(epoch, int) or not 0 <= epoch <= HEAD_EPOCHS or
                    not scalar(mse) or mse < 0 or not (path / 'selected.pt').is_file() or
                    not isinstance(current_ids, list) or not current_ids or
                    not isinstance(values, list) or len(values) != len(current_ids) or
                    any(not scalar(value) or value < 0 for value in values)):
                raise ValueError(f'{method}/S{supports}: invalid selected head result')
            if ids is None:
                ids = current_ids
            elif ids != current_ids:
                raise ValueError('Readout methods have different ordered validation queries')
            methods[method] = {'complete': complete, 'selected': selected,
                'checkpoint': str(path / 'selected.pt'),
                'checkpoint_sha256': digest(path / 'selected.pt')}
        result = {'scene': self.args.scene, 'supports': supports, 'head_epochs': HEAD_EPOCHS,
                  'test_read': False, 'methods': methods,
                  'ordered_query_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        write(self.out / f'readout_S{supports}_100epochs.json', result)
        self.record['readout_results'][str(supports)] = {
            'path': str(self.out / f'readout_S{supports}_100epochs.json'),
            'ordered_query_ids_sha256': result['ordered_query_ids_sha256'],
            'methods': {method: {'epoch': item['selected']['epoch'],
                                'mse': item['selected']['mse']}
                        for method, item in methods.items()}}
        self.save()
        emit('readout_budget_complete', supports=supports,
             methods={method: {'epoch': item['selected']['epoch'],
                               'mse': item['selected']['mse']} for method, item in methods.items()})

    def run(self):
        try:
            self.stage('source_prepare', [self.source_task('prepare', 'source_prepare')], gpu=False)
            self.stage('source_smoke', [self.source_task('smoke', 'source_smoke')])
            smoke = read(self.source_out / 'real_batch_smoke.json')
            if smoke.get('status') != 'PASS':
                raise ValueError('Source real-batch smoke did not pass')
            self.record['source_smoke'] = smoke
            self.save()
            self.stage('source_training_50', [self.source_task('train', f'source_{method}_50',
                method=method, epochs=SOURCE_EPOCHS) for method in SOURCE_METHODS])
            self.collect_source()
            self.stage('readout_prepare', [self.readout_task('prepare', 'readout_prepare')], gpu=False)
            self.stage('frozen_codes', [self.readout_task('encode', f'encode_{method}', method=method)
                for method in READOUT_METHODS])
            for supports in SUPPORTS:
                self.stage(f'readout_S{supports}_100', [self.readout_task('train',
                    f'head_S{supports}_{method}_100', method=method, supports=supports,
                    epochs=HEAD_EPOCHS) for method in READOUT_METHODS])
                self.collect_readout(supports)
            self.stage('probes', [self.readout_task('probe', f'probe_{method}', method=method)
                for method in READOUT_METHODS])
            self.stage('summary', [self.readout_task('summary', 'summary')], gpu=False)
            summary = read(self.readout_out / 'summary.json')
            if summary.get('status') != 'COMPLETE' or summary.get('test_read') is not False:
                raise ValueError('Readout summary does not establish complete validation-only execution')
            for method in READOUT_METHODS:
                probe = read(self.readout_out / 'probes' / f'{method}.json')
                representations = probe.get('representations')
                if (probe.get('status') != 'COMPLETE' or probe.get('test_read') is not False or
                        not isinstance(representations, dict) or
                        any(name not in representations for name in ('P', 'T', 'U'))):
                    raise ValueError('Required P/T/U probe result is incomplete: ' + method)
            self.record.update(status='COMPLETE', summary=summary,
                               completed_source_epochs=SOURCE_EPOCHS,
                               completed_supports=SUPPORTS)
            emit('pipeline_complete', scene=self.args.scene,
                 source_methods=SOURCE_METHODS, readout_methods=READOUT_METHODS,
                 supports=SUPPORTS, summary_path=str(self.readout_out / 'summary.json'))
        except BaseException as exc:
            self.record.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                               else 'FAILED', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            self.record['finished_at'] = time.time()
            self.save()
            self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, choices=('collision', 'balls'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--root', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
    parser.add_argument('--base')
    parser.add_argument('--dependency', action='append', default=[])
    parser.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 3])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--launcher-handle', help='Optional externally recorded launcher handle')
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or any(gpu < 0 for gpu in args.gpus):
        parser.error('Exactly four distinct nonnegative GPU indices are required')
    if not 1 <= args.threads <= 8:
        parser.error('Per-worker thread count must be in 1..8')
    Controller(args).run()


if __name__ == '__main__':
    main()
