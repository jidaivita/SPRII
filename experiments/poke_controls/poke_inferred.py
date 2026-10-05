"""Train-only episode-history nearest-neighbor relations, with matched random graph."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time
ROOT=pathlib.Path(os.environ.get('SPRII_ROOT', '.'));SRC=ROOT/'benchmarks/pokeworld/revision';DATA=ROOT/'data/pokeworld_factorized'
sys.path.insert(0,str(SRC/'src'))
import numpy as np
import torch
from persistent_jepa.poke_torch import PokeSplit,PokeBatch
from persistent_jepa.poke_model import PokeJEPA
from persistent_jepa.runtime import set_deterministic
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def graph(z):
 # Only unlabelled episode representations enter neighbor selection.
 x=np.asarray(z,dtype=np.float64);mu=x.mean(0);sd=x.std(0).clip(1e-8);x=(x-mu)/sd
 result=[]
 for start in range(0,len(x),128):
  d=((x[start:start+128,None,:]-x[None,:,:])**2).sum(-1)
  d[np.arange(len(d)),np.arange(start,start+len(d))]=np.inf
  result.extend(np.argsort(d,axis=1,kind='stable')[:,:3])
 return np.asarray(result,np.int64),mu,sd

def prepare(seed,out):
 out.mkdir(exist_ok=False);set_deterministic(seed);torch.set_num_threads(1)
 teacher=ROOT/f'runs/pokeworld/refinement_split_s{seed}'
 cfg=json.loads((teacher/'config.json').read_text());assert cfg['objective']=='none' and cfg['lambda_p']==cfg['lambda_x']==0
 assert cfg['dataset_manifest_sha256']==sha(DATA/'manifest.json') and cfg['seed']==seed
 cp=teacher/'checkpoints/step_020000.pt';before=sha(cp);ck=torch.load(cp,map_location='cpu',weights_only=False)
 assert ck['step']==20000 and ck['config']==cfg
 model=PokeJEPA(cfg['model_variant'],history_length=24).cuda().eval();model.load_state_dict(ck['model'])
 data=PokeSplit(DATA,'train');n,r=data.states.shape[:2];assert r==4
 # Labels are replaced before any feature call. No grouping or averaging by a
 # known physical system enters the graph: each rollout is a separate node.
 labels=[data.mass.copy(),data.gamma.copy(),data.stiffness.copy()]
 data.mass[:]=0;data.gamma[:]=0;data.stiffness[:]=0
 ids=np.arange(n*r);s=ids//r;rr=ids%r;zz=[]
 with torch.inference_mode():
  for start in range(0,len(ids),128):
   b=data._from_indices(s[start:start+128],rr[start:start+128],np.full(min(128,len(ids)-start),47)).to(torch.device('cuda'))
   b.target_current.zero_();b.target_previous.zero_()
   h,_=model.encode_batch(b);_,z,_=model.codes(h,b.history_actions);zz.append(z.cpu().numpy())
 z=np.concatenate(zz);assert np.isfinite(z).all();knn,mu,sd=graph(z)
 rng=np.random.default_rng(20260925+seed);random=np.stack([rng.choice(np.concatenate([ids[:i],ids[i+1:]]),3,replace=False) for i in ids])
 assert np.all(knn!=ids[:,None]) and np.all(random!=ids[:,None])
 np.savez_compressed(out/'GRAPH.npz',inferred=knn,random=random,teacher_codes=z,mean=mu,std=sd,query_system=s,query_rollout=rr)
 freeze=dict(status='FROZEN',seed=seed,teacher_sha256=before,teacher_config_sha256=sha(teacher/'config.json'),teacher_objective='native+SIGReg; zero Align/Cross',
  graph_sha256=sha(out/'GRAPH.npz'),data_manifest_sha256=sha(DATA/'manifest.json'),train_npz_sha256=sha(DATA/'train.npz'),nodes=len(ids),k=3,
  inputs='One last-legal 24-frame history per training rollout (anchor47), observation/action only; per-coordinate training standardization; Euclidean kNN excluding only identical rollout',
  factor_labels_used_for_graph=False,system_grouping_or_averaging=False,labels_zeroed_during_teacher_forward=True,targets_zeroed_during_teacher_forward=True,
  validation_or_test_read=False,rule_tuned=False,donor_history='Frozen rollout-level neighbor; donor anchor remains original independent training-window draw',
  wrapper_sha256=sha(__file__),time=time.time())
 write(out/'FROZEN.json',freeze)
 # Ground-truth checks happen only AFTER graph file and recipe have been frozen.
 label=np.stack(labels,1);statistics={}
 for mode,pools in [('inferred',knn),('random',random)]:
  match=label[(pools//r).reshape(-1)].reshape(len(ids),3,3)==label[s,None,:]
  counts=np.bincount(pools.reshape(-1),minlength=len(ids))
  statistics[mode]=dict(factor_match_fraction=dict(zip(['mass','drag','stiffness'],match.mean((0,1)).tolist())),all_factors_match_fraction=float(match.all(-1).mean()),
   same_system_fraction=float((pools//r==s[:,None]).mean()),incoming_degree_min=int(counts.min()),incoming_degree_max=int(counts.max()),incoming_degree_mean=float(counts.mean()),
   unique_donor_rollouts=int((counts>0).sum()))
 assert sha(cp)==before and sha(out/'GRAPH.npz')==freeze['graph_sha256']
 write(out/'MATCH_DIAGNOSTIC.json',dict(status='COMPLETE',statistics=statistics,selection_used_matching_rates=False,old_alpha_curve='Reference only: old fidelity bank and new factorized bank are different; no exact alpha=p interpolation claim'))
 write(out/'COMPLETE.json',dict(status='PASS',graph_sha256=freeze['graph_sha256'],frozen_sha256=sha(out/'FROZEN.json'),diagnostic_sha256=sha(out/'MATCH_DIAGNOSTIC.json')))
 print(json.dumps(dict(status='GRAPH_COMPLETE',seed=seed,statistics=statistics)),flush=True)

def install(seed,mode):
 out=ROOT/f'poke_inferred_graphs_v1/seed{seed}';f=json.loads((out/'FROZEN.json').read_text());assert sha(out/'GRAPH.npz')==f['graph_sha256']
 with np.load(out/'GRAPH.npz') as z:pools=z[mode].copy()
 def paired(self,pairs,seed,relation,donor_pools=None):
  rng=np.random.default_rng(seed);n,r=self.states.shape[:2]
  qs=rng.choice(n,size=pairs,replace=pairs>n);qr=rng.integers(r,size=pairs);qa=rng.choice(self.anchors,size=pairs);da=rng.choice(self.anchors,size=pairs)
  slot=rng.integers(0,3,size=pairs);flat=pools[qs*r+qr,slot]
  rng.integers(r,size=pairs) # consume the original donor-rollout draw
  assert np.all(flat!=qs*r+qr)
  donor=self._from_indices(flat//r,flat%r,da);query=self._from_indices(qs,qr,qa)
  return PokeBatch(**{k:torch.cat([getattr(donor,k),getattr(query,k)]) for k in donor.__dict__})
 PokeSplit.relation_paired_batch=paired
 return out,f,pools

def audit(seed):
 d=PokeSplit(DATA,'train');original=PokeSplit.relation_paired_batch
 pools=np.tile(np.arange(3),(len(d.mass),1));results=[]
 for step in (1,100,20000):
  rngseed=seed*10000000+step;ref=original(d,48,rngseed,'Random',pools)
  batches=[]
  for mode in ['inferred','random']:
   install(seed,mode);b=d.relation_paired_batch(48,rngseed,'G1');batches.append(b)
   for key in b.__dict__:assert torch.equal(getattr(b,key)[48:],getattr(ref,key)[48:]),key
   assert torch.equal(b.anchor,ref.anchor)
  assert torch.equal(batches[0].system_index[48:],batches[1].system_index[48:])
 PokeSplit.relation_paired_batch=original
 p=ROOT/f'poke_inferred_graphs_v1/seed{seed}'
 write(p/'SAMPLER_AUDIT.json',dict(status='PASS',steps=[1,100,20000],recipient_all_tensors_and_anchors_equal_original=True,exact48pairs96views=True,distinct_rollout=True))

def train(seed,mode,output):
 protocol,f,pools=install(seed,mode);assert json.loads((protocol/'SAMPLER_AUDIT.json').read_text())['status']=='PASS'
 module=load_module('original_train',SRC/'scripts/train_pokeworld_revision.py');old=module.atomic_json
 # Parser needs a pool key; the actual sampler uses only the frozen rollout graph.
 dummy=protocol/'PARSER_POOL.npz'
 if not dummy.exists():np.savez(dummy,train_G1=np.zeros((pools.shape[0]//4,3),np.int64))
 def atomic(path,obj):
  if pathlib.Path(path).name=='config.json':obj.update(condition='Inferred' if mode=='inferred' else 'RandomHistory',relation_mode=mode,panel='episode-history-inferred-v1',graph_sha256=f['graph_sha256'],teacher_sha256=f['teacher_sha256'],graph_recipe_sha256=sha(protocol/'FROZEN.json'),wrapper_sha256=sha(__file__),ground_truth_relations_used_in_graph=False)
  return old(path,obj)
 module.atomic_json=atomic
 sys.argv=['train_pokeworld_revision.py','--data-root',str(DATA),'--run-dir',str(output),'--family','refinement','--condition','G1','--donor-pools',str(dummy),'--seed',str(seed),'--steps','20000','--protocol-manifest',str(SRC/'PROTOCOL_MANIFEST.json'),'--source-bundle-sha256',sha(SRC/'scripts/train_pokeworld_revision.py')]
 module.main()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','audit','train']);p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['inferred','random']);p.add_argument('--output',type=pathlib.Path);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if a.phase=='prepare':prepare(a.seed,a.output)
 elif a.phase=='audit':audit(a.seed)
 else:train(a.seed,a.mode,a.output)
