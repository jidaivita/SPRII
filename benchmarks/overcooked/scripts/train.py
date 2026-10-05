"""Launch the archived optimizer with a released configuration and local data paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--data', type=Path, required=True, help='Directory containing the four prepared dataset files')
p.add_argument('--allowlist', type=Path, required=True, help='Allowlist generated against this dataset')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--backend', choices=('cpu','gpu'), default='gpu')
p.add_argument('--threads', type=int, default=8)
p.add_argument('--gpu', default='0', help='Single visible device for GPU execution')
p.add_argument('--resume', action='store_true')
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
cfg = json.loads(a.config.read_text())
repo = a.repo.resolve()
# Preserve the exact variant binding; the method files are unmodified snapshots.
for relative, expected in cfg['code_sha256'].items():
    if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Incorrect source variant: {relative}')
paths = {'h5-path':'histories.h5','index-path':'histories_index.jsonl',
         'task-manifest':'train_manifest.jsonl','episode-index':'native_episode_index.jsonl'}
cmd = [sys.executable, '-m', 'native_a.train']
for key, filename in paths.items():
    cmd += ['--' + key, str((a.data / filename).resolve())]
for key in ('mode','lambda_p','cross_weight','num_steps','batch_size','seed'):
    cmd += ['--' + key.replace('_','-'), str(cfg[key])]
for key in ('learning_rate','warmup_steps','weight_decay'):
    cmd += ['--' + key.replace('_','-'), str(cfg['optimizer'][key])]
for key in ('query_len','support_count','support_len'):
    cmd += ['--' + key.replace('_','-'), str(cfg['sampler'][key])]
cmd += ['--seq-len','500','--history-allowlist',str(a.allowlist.resolve()),
        '--out-dir',str(a.out.resolve()),'--backend',a.backend,'--threads',str(a.threads)]
if cfg['release_condition'] == 'Random':
    cmd += ['--relation-mode', 'random']
if a.resume:
    cmd.append('--resume')
if a.dry_run:
    print(json.dumps(cmd))
else:
    subprocess.run(cmd, cwd=repo, env={**os.environ,'PYTHONPATH':str(repo),
        'JAX_DEFAULT_MATMUL_PRECISION':'highest','JAX_ENABLE_X64':'false',
        'CUDA_VISIBLE_DEVICES':a.gpu if a.backend=='gpu' else ''}, check=True)
