"""Exact-factor mismatch control on the existing factorized development bank."""
import argparse,ast,hashlib,importlib.util,inspect,json,os,pathlib,sys,time
ROOT=pathlib.Path(os.environ.get('SPRII_ROOT', '.'));SOURCE=ROOT/'benchmarks/pokeworld/revision';DATA=ROOT/'data/pokeworld_factorized'
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
import torch
from persistent_jepa import poke_torch
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def prepare():
 out=ROOT/'poke_structured_protocol_v1';out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():return out
 assert sha(DATA/'manifest.json')=='4252ba73eda80e9d819eba6757666f6e4425b48789dec345bd07e4d76ee88638'
 maps={};audit={}
 # Donor substitutions are used for source training only. Geometry/probes use
 # the full common validation population; no validation donor mapping is needed.
 for split in ['train']:
  z=np.load(DATA/(split+'.npz'));theta=np.stack([z[k] for k in ['mass','gamma','stiffness']],1);rows={}
  for axis,label in [(0,'mass50'),(1,'drag50')]:
   other=np.delete(theta,axis,1);groups={}
   for i,t in enumerate(other):groups.setdefault(tuple(t),[]).append(i)
   rng=np.random.default_rng(20260925+axis);mapping=np.empty(len(theta),np.int64)
   for indices in groups.values():
    assert len(indices)>=2 and len(np.unique(theta[indices,axis]))==len(indices)
    order=rng.permutation(indices);mapping[order]=np.roll(order,-1)
   assert len(np.unique(mapping))==len(theta) and np.all(mapping!=np.arange(len(theta)))
   shared=theta[mapping]==theta;assert not shared[:,axis].any() and np.delete(shared,axis,1).all()
   maps[split+'_'+label]=mapping
   delta=np.abs((theta[mapping]-theta)/np.std(theta,axis=0))
   rows[label]=dict(systems=len(theta),coverage=1.,bijection=True,only_factor_changed=axis,match_fraction=shared.mean(0).tolist(),standardized_mean_abs_difference=delta.mean(0).tolist(),candidate_counts=sorted(set(len(v)-1 for v in groups.values())))
  audit[split]=rows
 np.savez_compressed(out/'MAPPINGS.npz',**maps)
 write(out/'COMPLETE.json',dict(status='PASS',data_manifest_sha256=sha(DATA/'manifest.json'),maps_sha256=sha(out/'MAPPINGS.npz'),audit=audit,
    controls='True and Uniform50 retrained on this same factorized bank; prior fidelity bank SHA differs',
    mixture='24 correct and 24 incorrect pair slots per update; original mask, query and window RNG; balanced wrong-system population mapping, not exact minibatch counts',
    matrix=['true','uniform50','drag50','mass50'],seeds=[0,1,2],steps=20000,test_read=False))
 return out

def install(mode):
 if mode in ('true','uniform50'):return
 protocol=prepare()
 with np.load(protocol/'MAPPINGS.npz') as f:mapping=f['train_'+mode].copy()
 source=inspect.getsource(poke_torch.PokeSplit.fidelity_paired_batch)
 import textwrap
 tree=ast.parse(textwrap.dedent(source));fn=tree.body[0];count=0
 for stmt in fn.body:
  if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Name) and stmt.targets[0].id=='random_system':
   stmt.value=ast.parse('_wrong_mapping[query_system]',mode='eval').body;count+=1
 assert count==1;ast.fix_missing_locations(tree);ns=dict(poke_torch.__dict__);ns['_wrong_mapping']=mapping
 exec(compile(tree,'<structured_fidelity_sampler>','exec'),ns)
 poke_torch.PokeSplit.fidelity_paired_batch=ns['fidelity_paired_batch']

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['true','uniform50','drag50','mass50']);p.add_argument('--seed',type=int,default=0);p.add_argument('--output');p.add_argument('--audit',action='store_true');a=p.parse_args()
 protocol=prepare();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if a.audit:
  d=poke_torch.PokeSplit(DATA,'train');original=poke_torch.PokeSplit.fidelity_paired_batch
  for seed in range(3):
   for step in [1,100,20000]:
    ref,mask=d.fidelity_paired_batch(48,seed*10000000+step,20260901,step,.5)
    for mode,axis in [('drag50',1),('mass50',0)]:
     install(mode);b,actual=d.fidelity_paired_batch(48,seed*10000000+step,20260901,step,.5)
     assert np.array_equal(actual,mask) and int(mask.sum())==24
     for name in b.__dict__:assert torch.equal(getattr(b,name)[48:],getattr(ref,name)[48:]),name
     for name in ('rollout_id','anchor'):assert torch.equal(getattr(b,name),getattr(ref,name)),name
     match=torch.stack([getattr(b,k)[:48]==getattr(b,k)[48:] for k in ('mass','gamma','stiffness')],1)
     assert match[mask].all() and not match[~mask,axis].any()
     assert torch.cat([match[~mask,:axis],match[~mask,axis+1:]],1).all()
     poke_torch.PokeSplit.fidelity_paired_batch=original
  write(protocol/'SAMPLER_AUDIT.json',dict(status='PASS',checked_seeds=[0,1,2],checked_steps=[1,100,20000],recipient_and_rollout_and_anchor_equal=True,exact_factor_errors=True,wrong_count=24))
  print('SAMPLER_AUDIT PASS',flush=True);return
 assert a.mode and a.output
 install(a.mode)
 spec=importlib.util.spec_from_file_location('original_poke_training',SOURCE/'scripts/train_pokeworld_revision.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original_atomic=module.atomic_json
 def atomic(path,obj):
  if pathlib.Path(path).name=='config.json':obj.update(relation_mode=a.mode,panel='factorized-structured-errors-v1',structured_protocol_sha256=sha(protocol/'COMPLETE.json'),structured_mapping_sha256=sha(protocol/'MAPPINGS.npz'),wrapper_sha256=sha(__file__),historical_fidelity_bank_reused=False)
  return original_atomic(path,obj)
 module.atomic_json=atomic
 sys.argv=['train_pokeworld_revision.py','--data-root',str(DATA),'--run-dir',a.output,'--family','fidelity','--condition','q1' if a.mode=='true' else 'q05','--seed',str(a.seed),'--steps','20000','--fidelity-q','1' if a.mode=='true' else '.5','--protocol-manifest',str(SOURCE/'PROTOCOL_MANIFEST.json'),'--source-bundle-sha256',sha(SOURCE/'scripts/train_pokeworld_revision.py')]
 module.main()
if __name__=='__main__':main()
