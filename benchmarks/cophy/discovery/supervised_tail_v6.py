"""Existing Blocktower selected models: cached scoring targets and fixed evaluation.

No optimizer or test split. The original evaluator/model/manifest are reused;
only repeated recipient-target disk reads are replaced by an exact shared cache.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = 'supervised-blocktower-tail-v6-2'
METHODS = ('Native', 'A', 'Random', 'Param-known')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def runtime(root):
    root = Path(root)
    registered = read(root / 'runtime_profiles.json')['scenes']['blocktower']
    preflight = Path(registered['training_preflight']['path'])
    if digest(preflight) != registered['training_preflight']['sha256']:
        raise ValueError('Registered Blocktower preflight hash differs')
    bound = read(preflight)
    source = Path(bound['runtime_source_root']).resolve()
    if source != Path(registered['source']).resolve():
        raise ValueError('Blocktower source differs from the registered runtime')
    sys.path.insert(0, str(source))
    import cophy_evaluate
    if Path(cophy_evaluate.__file__).resolve().parent != source:
        raise ValueError('A different scene runtime was already imported')
    cophy_evaluate.verify_adapter_binding(str(preflight))
    return cophy_evaluate


def prepare(root, out):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    marker = out / 'targets_complete.json'
    pre = root / 'prepared_v3/blocktower_gate1/training_preflight.json'
    if marker.exists():
        saved = read(marker)
        if saved['preflight_sha256'] != digest(pre) or saved['targets_sha256'] != digest(out / 'targets.npz'):
            raise ValueError('Changed target cache binding')
        print(json.dumps(saved), flush=True); return
    began = time.time(); module = runtime(root)
    module.verify_preflight(str(pre), require_release=False)
    profile = module.verify_adapter_binding(str(pre))['input_profile']
    splits = module.read_artifact(str(pre), 'splits'); ids = list(map(str, splits['val']['ids']))
    pose, stationary, presence = [], [], []
    for offset in range(0, len(ids), 128):
        target = module.targets(profile, ids[offset:offset + 128], 'cpu')
        pose.append(target.pose.numpy()); stationary.append(target.stationary.numpy()); presence.append(target.presence.numpy())
    path = out / 'targets.npz'; tmp = out / ('targets.tmp.' + str(os.getpid()))
    with open(tmp, 'wb') as stream:
        np.savez(stream, ids=np.asarray(ids), pose=np.concatenate(pose), stationary=np.concatenate(stationary), presence=np.concatenate(presence))
    os.replace(tmp, path)
    receipt = {'version': VERSION, 'status': 'COMPLETE', 'test_read': False, 'optimizer_steps': 0,
               'rows': len(ids), 'preflight_sha256': digest(pre), 'targets_sha256': digest(path),
               'code_sha256': digest(__file__), 'seconds': time.time() - began}
    write(marker, receipt); print(json.dumps(receipt), flush=True)


def evaluate(root, out, method, device):
    import numpy as np
    import torch
    root, out = Path(root), Path(out); run = out / 'runs' / method; run.mkdir(parents=True, exist_ok=True)
    pre = root / 'prepared_v3/blocktower_gate1/training_preflight.json'
    ckpt = root / 'runs_seed0/blocktower_gate1' / method / 'model_state_dict.pt'
    receipt = read(out / 'targets_complete.json')
    if receipt['preflight_sha256'] != digest(pre) or receipt['targets_sha256'] != digest(out / 'targets.npz'):
        raise ValueError('Target cache binding differs')
    selected_hash = digest(ckpt); complete = run / 'complete.json'
    if complete.exists():
        saved = read(complete)
        if saved['checkpoint_sha256'] != selected_hash or not (run / 'validation.json').is_file():
            raise ValueError('Existing evaluation differs')
        print(json.dumps(saved), flush=True); return
    torch.set_num_threads(4); module = runtime(root); z = np.load(out / 'targets.npz', allow_pickle=False)
    lookup = {str(ident): i for i, ident in enumerate(z['ids'])}
    arrays = {k: z[k].copy() for k in ('pose', 'stationary', 'presence')}; z.close()
    def cached_targets(profile, ids, requested_device):
        if profile['scene'] != 'blocktower': raise ValueError('Unexpected target scene')
        index = [lookup[str(ident)] for ident in ids]
        return module.Targets(*(torch.from_numpy(arrays[k][index]).to(requested_device) for k in ('pose', 'stationary', 'presence')))
    module.targets = cached_targets
    config = {'version': VERSION, 'method': method, 'device': device, 'checkpoint': str(ckpt),
              'checkpoint_sha256': selected_hash, 'preflight_sha256': digest(pre), 'test_read': False,
              'optimizer_steps': 0, 'target_cache': receipt, 'pid': os.getpid(),
              'runtime_source': str(Path(module.__file__).resolve().parent),
              'evaluator_sha256': digest(module.__file__)}
    write(run / 'config.json', config); began = time.time()
    args = argparse.Namespace(preflight=str(pre), checkpoint=str(ckpt), split='val', release=None,
                              output=str(run / 'validation.json'), device=device, batch_size=64, threads=4)
    module.run(args)
    result = read(run / 'validation.json')
    final = {**config, 'status': 'COMPLETE', 'epoch': result['epoch'], 'official': result['official'],
             'visual_coverage': result['visual_coverage'], 'seconds': time.time() - began,
             'result_sha256': digest(run / 'validation.json')}
    write(complete, final); print(json.dumps(final), flush=True)


def queue(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'controller.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'version': VERSION, 'status': 'WAITING_TARGETS', 'pid': os.getpid(), 'started_at': time.time(),
              'test_read': False, 'optimizer_steps': 0, 'jobs': []}
    write(out / 'controller_status.json', status)
    while not (out / 'targets_complete.json').exists():
        if time.time() - status['started_at'] > 3600: raise TimeoutError('Recipient target preparation did not complete')
        time.sleep(2)
    children = []
    for gpu, method in enumerate(METHODS):
        run = out / 'runs' / method; run.mkdir(parents=True, exist_ok=True)
        log = open(run / 'worker.log', 'a', buffering=1)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        cmd = [sys.executable, '-u', __file__, 'block-eval', '--root', str(root), '--out', str(out), '--method', method, '--device', 'cuda:0']
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environment)
        job = {'method': method, 'gpu': gpu, 'pid': process.pid, 'status': 'RUNNING', 'started_at': time.time(), 'log': str(run / 'worker.log')}
        status['jobs'].append(job); children.append((process, log, job))
    status['status'] = 'RUNNING'; write(out / 'controller_status.json', status)
    while any(p.poll() is None for p, _, _ in children):
        for p, log, job in children:
            if p.poll() is not None and 'exit_code' not in job:
                job.update(exit_code=p.returncode, status='COMPLETE' if p.returncode == 0 else 'FAILED', finished_at=time.time()); log.close()
        write(out / 'controller_status.json', status); time.sleep(2)
    for p, log, job in children:
        if 'exit_code' not in job: job.update(exit_code=p.returncode, status='COMPLETE' if p.returncode == 0 else 'FAILED', finished_at=time.time()); log.close()
    status['status'] = 'COMPLETE' if all(p.returncode == 0 for p, _, _ in children) else 'FAILED'
    status['finished_at'] = time.time(); write(out / 'controller_status.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] != 'COMPLETE': raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare-block', 'block-eval', 'block-queue'])
    parser.add_argument('--root', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))); parser.add_argument('--out', required=True)
    parser.add_argument('--method', choices=METHODS); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'prepare-block': prepare(args.root, args.out)
    elif args.command == 'block-eval': evaluate(args.root, args.out, args.method, args.device)
    else: queue(args.root, args.out)
