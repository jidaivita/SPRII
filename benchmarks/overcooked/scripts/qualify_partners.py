"""Freeze twenty qualified training policies before collecting any new histories."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
PARTNER_ROOT = Path('data/partners')
SEEDS=[4600,4601,*range(4700,4708)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert read(p)==value,f'Existing frozen artifact differs: {p}';return
 tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)
def source(seed):
 return PARTNER_ROOT / f'seed{seed}'
def verify_source(seed):
 root=source(seed);native=root/'ippo_train_run';done=read(root/'completion.json');binding=read(root/'binding.json')
 pub=read(native/'native_publish.json');config=read(native/'config.json')
 assert done['status']=='PASS' and done['complete'] and done['actual_updates']==done['planned_updates']==457
 assert config['TRAIN_SEED']==binding['config']['TRAIN_SEED']==seed
 assert config['ENV_KWARGS']['layout']=='grounded_coord_simple'
 assert done['native_publish_sha256']==sha(native/'native_publish.json')
 assert pub['actual_updates']==457 and pub['num_checkpoints']==5
 for ref in pub['files']:
  path=(root/ref['path']).resolve();assert path.is_relative_to(native)
  assert path.stat().st_size==ref['size'] and sha(path)==ref['sha256']
 return dict(seed=seed,path=str(root),completion_sha256=sha(root/'completion.json'),publication_sha256=sha(native/'native_publish.json'),binding_sha256=sha(root/'binding.json'))
def make_task(seed,index,role):
 from benchmarks.manifest_schema import TaskEntry,TeammateSpec,generate_task_id
 extra=dict(actor_type='cnn_rnn',activation='relu',fc_dim_size=128,gru_hidden_dim=128,
            use_separated_ckpt=True,checkpoint_idx=index,population_idx=0,seed_idx=0)
 mate=TeammateSpec.rl('ippo',str(source(seed)/'ippo_train_run'),extra)
 task=TaskEntry(generate_task_id('teammate',role,'grounded_coord_simple',mate,4500),
                'teammate','train','grounded_coord_simple',4500,mate).to_json()
 task['split']=role;return task
def main():
 global PARTNER_ROOT
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--partners',type=Path,required=True)
 p.add_argument('--repo',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args(); PARTNER_ROOT=a.partners.resolve()
 os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1')
 sys.path.insert(0,str(a.repo.resolve()))
 import jax,numpy as np
 from benchmarks.manifest_schema import TaskEntry
 from eval_icrl import create_env_for_task,create_teammate
 from native_a.train import tensor_sha
 assert jax.default_backend()=='cpu'
 inputs=a.out.resolve();inputs.mkdir(parents=True,exist_ok=True)
 assert not (inputs/'qualification.json').exists(),'Use a fresh output directory'
 verified=[verify_source(seed) for seed in [*SEEDS,4602]]
 candidates=[];excluded=[];seen={};dev=[]
 for seed in [*SEEDS,4602]:
  returns=read(source(seed)/'ippo_train_run/checkpoint_returns.json')['base_returns'][0]
  for index in (range(5) if seed!=4602 else (3,4)):
   value=float(returns[index]);row=dict(source_seed=seed,checkpoint_idx=index,base_return=value)
   if not math.isfinite(value) or value<=20:
    excluded.append(dict(row,reason='Native checkpoint base return is not strictly greater than20'));continue
   task=make_task(seed,index,'train' if seed!=4602 else 'development')
   legacy=TaskEntry.from_json(dict(task,split='train'));env=create_env_for_task(legacy,max_steps=100)
   partner=create_teammate(legacy,env);digest=tensor_sha(partner._params)
   row.update(task=task,parameter_sha256=digest)
   if digest in seen:
    assert seed!=4602,'Development policy duplicates a training candidate'
    excluded.append(dict(row,reason='Duplicate actual policy parameters',duplicate_of=seen[digest]));continue
   seen[digest]=dict(source_seed=seed,checkpoint_idx=index)
   (dev if seed==4602 else candidates).append(row)
 assert len(dev)==2,'The fixed two development policies must both qualify'
 receipt=dict(status='QUALIFIED_CANDIDATES',entry_sha256=sha(__file__),
   selection_rule='Sort source_seed/checkpoint_idx then numpy.default_rng(0).permutation; take20 and restore source order',
   selection_seed=0,maximum_training_candidates=50,source_verification=verified,
   candidates=candidates,excluded=excluded,development=dev,at=time.time(),official_benchmark_test_read=False)
 if len(candidates)<20:
  receipt['status']='INSUFFICIENT_QUALIFIED_POLICIES';write(inputs/'qualification_failure.json',receipt)
  raise RuntimeError(f'Only {len(candidates)} qualifying policies in the fixed source pool; do not expand or lower the threshold')
 chosen=np.random.default_rng(0).permutation(len(candidates))[:20]
 selected=[candidates[int(i)] for i in sorted(chosen)];receipt.update(status='PASS',selected=selected)
 write(inputs/'qualification.json',receipt)
 for name,rows in (('train_manifest.jsonl',selected),('unseen_manifest.jsonl',dev)):
  content=''.join(json.dumps(row['task'],sort_keys=True)+'\n' for row in rows)
  path=inputs/name
  if path.exists():assert path.read_text()==content
  else:path.write_text(content)
 binding=dict(status='TRAIN_AND_UNSEEN_POLICIES_FROZEN',train_manifest_sha256=sha(inputs/'train_manifest.jsonl'),
  unseen_manifest_sha256=sha(inputs/'unseen_manifest.jsonl'),qualification_sha256=sha(inputs/'qualification.json'),
  number_training_policies=20,number_development_policies=2,disjoint_parameter_hashes=True,
  note='Two unseen checkpoints share one excluded source; they are not independent source seeds',model_fits_started=0)
 write(inputs/'source_isolation.json',binding);print(json.dumps(binding),flush=True)
if __name__=='__main__':main()
