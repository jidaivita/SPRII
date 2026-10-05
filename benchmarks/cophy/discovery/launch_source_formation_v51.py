"""Remote-only preparation and bounded handoff to the complete v5.1 pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
CODE = Path(__file__).resolve().parent
PREP = ROOT / 'source_formation_v51_prepare'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)

def prepare_all():
    failures = []
    for scene in ('collision', 'balls'):
        folder = PREP / scene
        folder.mkdir(parents=True, exist_ok=True)
        status = folder / 'remote_ready.json'
        write(status, {'status': 'PREPARING', 'pid': os.getpid(), 'scene': scene, 'time': time.time()})
        command = [sys.executable, '-u', str(CODE / 'source_formation_v51.py'),
                   'prepare', '--scene', scene, '--out', str(folder), '--root', str(ROOT)]
        code = subprocess.call(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        write(status, {'status': 'READY' if code == 0 else 'FAILED', 'exit_code': code,
                       'scene': scene, 'time': time.time(), 'pid': os.getpid()})
        if code:
            failures.append(scene)
    if failures:
        raise SystemExit('Preparation failed: ' + ','.join(failures))

def start_scene(scene):
    state = PREP / scene / 'remote_ready.json'
    deadline = time.monotonic() + 1800
    print(json.dumps({'status': 'WAITING_FOR_SHARED_DATA', 'scene': scene,
                      'pid': os.getpid(), 'deadline_seconds': 1800}), flush=True)
    while time.monotonic() < deadline:
        if state.exists():
            row = json.loads(state.read_text())
            if row['status'] == 'FAILED':
                raise RuntimeError('Shared preparation failed; inspect preparation handle: ' + scene)
            if row['status'] == 'READY':
                command = [sys.executable, '-u', str(CODE / 'run_source_formation_v51.py'),
                    '--scene', scene, '--out', str(ROOT / 'source_formation_v5_1' / scene),
                    '--protocol', str(CODE / 'Source_Formation_8GPU_v5_1_EXECUTION.md'),
                    '--root', str(ROOT)]
                os.execv(sys.executable, command)
        time.sleep(5)
    raise TimeoutError('Shared preparation did not become ready within 30 minutes: ' + scene)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare-all', 'start'))
    p.add_argument('--scene', choices=('collision', 'balls'))
    a = p.parse_args()
    if a.command == 'prepare-all':
        prepare_all()
    elif a.scene:
        start_scene(a.scene)
    else:
        p.error('--scene required for start')
