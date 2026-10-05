"""Collect the fixed PPO history budget or pack its original episode metadata."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('operation', choices=('collect','pack'))
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--run-root', type=Path, required=True,
               help='Contains dataset_inputs/train_manifest.jsonl and histories/')
p.add_argument('--task-index', type=int)
p.add_argument('--gpu', default='0')
a = p.parse_args()
repo, root = a.repo.resolve(), a.run_root.resolve()
sys.path.insert(0, str(repo))
if a.operation == 'collect':
    if a.task_index is None or not 0 <= a.task_index < 20:
        p.error('--task-index must select one of the 20 frozen training partners')
    cmd = [sys.executable,'-m','runners.task_runner','--manifest',str(root/'dataset_inputs/train_manifest.jsonl'),
           '--task_idx',str(a.task_index),'--out_dir',str(root/'histories'),
           '--total_steps','60000000','--num_envs','1024','--rollout_length','256',
           '--num_minibatches','64','--update_epochs','4','--record_envs','1024',
           '--record_first_steps','100','--save_interval','0','--max_steps','400',
           '--actor_type','cnn_rnn','--fc_dim_size','128','--gru_hidden_dim','128',
           '--rew_shaping_horizon','60000000','--log_every','10','--csv_log','True',
           '--csv_interval','10','--gpu',a.gpu]
    subprocess.run(cmd,cwd=repo,env={**os.environ,'PYTHONPATH':str(repo)},check=True)
else:
    from native_a.pipeline import pack_worker
    pack_worker({'run_root':str(root),'config':{'max_histories_per_task':128,'technical_only':False}})
