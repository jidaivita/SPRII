"""Finite offline DALI-context source gate; train/dev only, no test reader.

One optimizer step per batch32 of paired train episodes (AB and CD both used).
Uniformly sampled causal prefix per trajectory is an unbiased estimate of the
published rolling-prefix auxiliary forward objective, including its final-zero
time scaling. Every source epoch visits all 14,000 Collision train IDs once.
No physical-parameter or coordinate labels enter this source.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from models import CoPhyDALI, Config, VERSION


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def save(p, x):
    p = Path(p); tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    torch.save(x, tmp); tmp.replace(p)


class FeatureData:
    def __init__(self, path):
        self.path = Path(path)
        if read(self.path/'scene_COMPLETE.json')['status'] != 'COMPLETE':
            raise ValueError('Incomplete feature cache')
        self.files = {}
        self.data = {}
        for split, n in [('train', 14000), ('val', 4000)]:
            folder = self.path/split
            complete = read(folder/'COMPLETE.json')
            assert complete['status'] == 'COMPLETE' and complete['test_read'] is False
            assert complete['scene'] == 'collision'
            assert complete['manifest_sha256'] == sha(folder/'manifest.json')
            for name in ('COMPLETE.json', 'manifest.json', 'ids.json'):
                self.files[str(folder/name)] = sha(folder/name)
            ids = list(map(str, read(folder/'ids.json')))
            assert len(ids) == n and len(set(ids)) == n
            arrays = {}
            for arm in ('ab', 'cd'):
                for kind in ('features', 'presence'):
                    p = folder/f'{kind}_{arm}.npy'
                    array = np.load(p, mmap_mode='r', allow_pickle=False)
                    shape = (n, 15, 4, 784) if kind == 'features' else (n, 15, 4)
                    assert array.shape == shape, (p, array.shape)
                    arrays[kind+'_'+arm] = array
            self.data[split] = {'ids': ids, **arrays}
        assert not set(self.data['train']['ids']) & set(self.data['val']['ids'])

    def batch(self, ix, device):
        # Validation observations are never accessed by the source optimizer.
        split = self.data['train']
        feat = np.concatenate([np.asarray(split['features_'+arm][ix], dtype=np.float32)
                               for arm in ('ab', 'cd')])
        mask = np.concatenate([np.asarray(split['presence_'+arm][ix], dtype=np.bool_)
                               for arm in ('ab', 'cd')])
        assert np.isfinite(feat).all()
        return torch.from_numpy(feat).to(device), torch.from_numpy(mask).to(device)


def rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore(x):
    random.setstate(x['python']); np.random.set_state(x['numpy']); torch.set_rng_state(x['torch'])
    if x['cuda']: torch.cuda.set_rng_state_all(x['cuda'])


def execute(a):
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    out = Path(a.out); data = FeatureData(a.features)
    model = CoPhyDALI(Config()).to(a.device)
    here = Path(__file__).resolve().parent
    binding = {'version': VERSION, 'scene': 'collision', 'method': 'DALI-context', 'seed': a.seed,
               'scope': 'released context/forward component adapted to frozen RGB; not full Dreamer',
               'files': {**data.files, **{str(here/p): sha(here/p) for p in
                                         ('models.py', 'dali_context_torch.py', 'train_source.py')}},
               'config': model.artifact_config(), 'lr': .0001, 'batch_pairs': 32,
               'optimizer': 'Adam', 'weight_decay': 0., 'eps': 1e-8, 'clip_norm': 1000.,
               'prefix_objective': 'uniform t in [1,14], official left-edge padding and (14/15) scaling',
               'episode_visits_per_epoch': 14000, 'visual_trajectories_per_epoch': 28000,
               'updates_per_epoch': 438, 'source_budget_cap_epochs': 50,
               'context_history_frames': 15, 'all_objects': True, 'action': 'constant zero; autonomous task',
               'target': 'next frozen RGB scene features, all train AB/CD trajectories',
               'coordinate_labels_read': False, 'physical_labels_read': False, 'test_read': False}
    bsha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if (out/'BINDING.json').exists(): assert read(out/'BINDING.json') == binding
    else: write(out/'BINDING.json', binding)
    opt = torch.optim.Adam(model.parameters(), lr=.0001, eps=1e-8)
    start_epoch, step, prior_seconds = 1, 0, 0.
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        assert ck['binding_sha256'] == bsha and ck['completed_epoch'] == ck['epoch']
        assert ck['epoch'] < a.epochs
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); restore(ck['rng'])
        start_epoch = ck['epoch']+1; step = ck['step']; prior_seconds = ck['seconds']
    elif (out/'latest.pt').exists():
        raise RuntimeError('Existing partial/complete source: inspect and explicitly resume its checkpoint')
    began = time.monotonic()
    for epoch in range(start_epoch, a.epochs+1):
        model.train(); indices = np.random.default_rng(a.seed+100003*epoch).permutation(14000)
        loss_sum, count = 0., 0
        for start in range(0, len(indices), 32):
            if time.monotonic()-began > a.max_seconds:
                raise TimeoutError('Finite source wall-time budget reached; no implicit restart')
            ix = indices[start:start+32]; features, presence = data.batch(ix, a.device)
            lengths = torch.randint(1, 15, (len(features),), device=a.device)
            opt.zero_grad(set_to_none=True)
            loss = model.source_loss(features, presence, lengths)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite DALI source loss')
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000., error_if_nonfinite=True)
            if float(gn) == 0: raise RuntimeError('No DALI source gradients')
            opt.step(); step += 1; count += len(ix); loss_sum += float(loss.detach())*len(ix)
            if step % 50 == 0:
                write(out/'PROGRESS.json', {'status': 'RUNNING', 'epoch': epoch, 'step': step,
                                           'loss': float(loss.detach()), 'gradient_norm': float(gn),
                                           'elapsed_seconds': prior_seconds+time.monotonic()-began})
        assert count == 14000
        record = {'epoch': epoch, 'step': step, 'train_loss': loss_sum/count,
                  'seconds': prior_seconds+time.monotonic()-began}
        with (out/'TRAIN.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        ck = {'version': VERSION, 'method': 'DALI-context', 'scene': 'collision',
              'epoch': epoch, 'completed_epoch': epoch, 'step': step,
              'model': model.state_dict(), 'model_config': model.artifact_config(),
              'optimizer': opt.state_dict(), 'rng': rng(), 'seconds': record['seconds'],
              'binding': binding, 'binding_sha256': bsha, 'test_read': False}
        save(out/'latest.pt', ck)
        if epoch in (5, 50): save(out/f'checkpoint_{epoch}.pt', ck)
        print(json.dumps(record), flush=True)
    final = out/f'checkpoint_{a.epochs}.pt'
    write(out/f'TRAIN_{a.epochs}_COMPLETE.json', {'status': 'COMPLETE', 'epochs': a.epochs,
          'step': step, 'checkpoint': str(final), 'checkpoint_sha256': sha(final),
          'binding_sha256': bsha, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
          'test_read': False, 'formal_budget_reached': a.epochs == 50})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--features', required=True); p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, choices=(5, 50), default=5); p.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-seconds', type=int, default=7200); p.add_argument('--resume')
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'source.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: execute(a)
        except BaseException as e:
            write(out/f'FAILED_{time.time_ns()}.json', {'status': 'FAILED', 'error': repr(e), 'traceback': traceback.format_exc()})
            raise
