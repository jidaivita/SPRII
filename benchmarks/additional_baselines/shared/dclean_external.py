"""D-Clean released-NOD component adaptation / FCRL paper implementation.
No claims of untouched MassSpring reproduction. Train/validation only.
"""
import argparse,functools,hashlib,importlib.util,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(os.environ.get('SPRII_DCLEAN_ROOT', 'runs/dclean_controls'))
DATA=Path(os.environ.get('SPRII_DCLEAN_DATA', 'data/dclean_v1'))
OFFICIAL=Path(os.environ.get('SPRII_NOD_MODEL', 'external/nod/code/MassSpring/model.py'))
NATIVE=Path(os.environ.get('SPRII_NATIVE_MODEL', str(Path(__file__).resolve().parents[3]/'src/persistent_jepa/model.py')))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
@functools.lru_cache(None)
def data(split):
 assert split in ('train','val')
 assert sha(DATA/'manifest.json')=='120a2e137d1a0a801c447657ad3666b3b939488aecb9df4036ef62c2007bd186'
 expected={'train':'6b59350ce362a4f9a4c39d308c981937b1cf88d7686315120a5112ac728f7ee7','val':'3c26fa37e186152c1a754e9a06e1422a2381ee6633a769dd1ba648b73048e7d7'}
 assert sha(DATA/f'{split}.npz')==expected[split]
 with np.load(DATA/f'{split}.npz',allow_pickle=False) as f:a={k:f[k] for k in ('states','actions','gamma','system_ids','rollout_seeds')}
 assert a['states'].shape==(1000 if split=='train' else 200,8,64,4)
 return a
@functools.lru_cache(None)
def stats():
 a=data('train');x=a['states'].reshape(-1,4).astype('float64');u=a['actions'].reshape(-1,2).astype('float64')
 return dict(state_mean=x.mean(0).tolist(),state_scale=x.std(0).clip(1e-8).tolist(),action_scale=u.std(0).clip(1e-8).tolist())
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def specs(seed,step,h32=False):
 rng=np.random.default_rng((202609160000+seed*100000+step) if h32 else seed*10000000+step)
 n=48;s=rng.choice(1000,n,replace=False);a=rng.integers(8,size=n);b=(a+rng.integers(1,8,n))%8
 return np.stack([s,a,rng.integers(23,32 if h32 else 48,n),b,rng.integers(23,32 if h32 else 48,n)],1)
def batch(split,sp,horizons=(1,4,16),device='cuda:0'):
 d=data(split);sp=np.asarray(sp);assert np.all(sp[:,1]!=sp[:,3]);parts=[]
 for ri,ai in ((1,2),(3,4)):
  s,r,t=sp[:,0,None],sp[:,ri,None],sp[:,ai,None]
  assert np.all(t>=23) and np.all(t+max(horizons)<64)
  hs=d['states'][s,r,t+np.arange(-23,1)];ha=d['actions'][s,r,t+np.arange(-23,0)]
  u=d['actions'][s,r,t+np.arange(max(horizons))];y=d['states'][s,r,t+np.array(horizons)]
  parts.append((hs,ha,u,y))
 return tuple(torch.as_tensor(np.concatenate([p[k] for p in parts]),device=device) for k in range(4))
def swap(z):a,b=z.chunk(2);return torch.cat([b,a])

class Encoder(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  for k,v in stats().items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
  if method=='NOD':
   m=module(OFFICIAL,'dclean_released_nod');self.base=m.NGSMetaNet(latent_dim=4,condition_trajectory_dim=6)
   # Explicit environment adaptation: forced point state4/action2 -> derivative4, integrated at dt=.05.
   self.base.prediction_net[-1]=nn.Linear(32,4);nn.init.xavier_normal_(self.base.prediction_net[-1].weight);nn.init.zeros_(self.base.prediction_net[-1].bias)
  elif method=='FCRL':
   self.point=nn.Sequential(nn.Linear(10,50),nn.ReLU(),nn.Linear(50,50),nn.ReLU(),nn.Linear(50,50))
   self.projector=nn.Sequential(nn.Linear(50,50),nn.BatchNorm1d(50),nn.ReLU(),nn.Linear(50,50))
  else:raise ValueError(method)
 def encode(self,x,u):
  x=(x-self.state_mean)/self.state_scale;u=u/self.action_scale
  if self.method=='NOD':
   # Observed action leaving each state; last unavailable action padded zero.
   inp=torch.cat([x,torch.cat([u,torch.zeros_like(u[:,:1])],1)],-1)
   h,_=self.base.conditioning_net(inp);return self.base.summary_layer(h[:,-1])
  return self.point(torch.cat([x[:,:-1],u,x[:,1:]],-1)).mean(1)
 def objective(self,hs,ha,u,y):
  z=self.encode(hs,ha)
  if self.method=='FCRL':
   a,b=torch.nn.functional.normalize(self.projector(z),dim=1).chunk(2)
   logits=a@b.T/.07;label=torch.arange(len(a),device=a.device)
   return (nn.functional.cross_entropy(logits,label)+nn.functional.cross_entropy(logits.T,label))/2
  donor=swap(z+torch.randn_like(z)*.1);x=hs[:,-1];pred=[]
  for t in range(16):
   inp=torch.cat([(x-self.state_mean)/self.state_scale,u[:,t]/self.action_scale,donor],-1)
   x=x+.05*self.base.prediction_net(inp)*self.state_scale
   if t+1 in (1,4,16):pred.append(x)
  return ((torch.stack(pred,1)-y)/self.state_scale).square().mean()

def contract():
 a,b=data('train'),data('val');assert not set(a['system_ids'])&set(b['system_ids']);assert not set(a['rollout_seeds'].flat)&set(b['rollout_seeds'].flat)
 ids=list(map(int,b['system_ids']));ranked=sorted(ids,key=lambda x:hashlib.sha256(f'sprii-dclean-select-report-20260916:{x}'.encode()).digest());select=set(ranked[:100])
 out=dict(status='FROZEN',methods={'NOD':'released MassSpring conditioning LSTM/ShrinkNet/ResNet; forced state/action and derivative output adaptation; Euler dt=.05','FCRL':'paper implementation; mean-pooled transition set and nonlinear InfoNCE critic','SPRII':'existing B3 frozen source checkpoints'},source_steps=20000,source_seeds=[0,1,2],NOD_latent=4,FCRL_latent=50,NOD_noise_train=.1,NOD_noise_eval=0,source_horizons=[1,4,16],source_pairs=48,reader_seeds=[0,1,2],reader_steps=20000,reader_horizons=[1,4,16,32],primary_horizon=32,selection_ids=[i for i in ids if i in select],report_ids=[i for i in ids if i not in select],normalization=stats(),data_manifest_sha256=sha(DATA/'manifest.json'),official_model_sha256=sha(OFFICIAL),native_model_sha256=sha(NATIVE),code_sha256=sha(__file__),test_read=False,historical_validation=True)
 p=ROOT/'PROTOCOL.json'
 if p.exists():assert read(p)==out,'Protocol changed: use a separate experiment version'
 else:write(p,out)
 return out

def smoke():
 c=contract();seed_all(0);sp=specs(0,1);b=batch('train',sp)
 out={}
 for method in ('NOD','FCRL'):
  m=Encoder(method).cuda().train();loss=m.objective(*b);assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True))
  m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.encode(b[0],b[1]);z2=m.encode(b[0],b[1])
  assert torch.equal(z,z2) and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
  out[method]=dict(loss=float(loss),gradient_norm=gn,latent_shape=list(z.shape),parameters=sum(p.numel() for p in m.parameters()))
 # Data interface contains histories/actions/targets only, no system parameter.
 assert len(b)==4 and b[0].shape==(96,24,4) and b[1].shape==(96,23,2)
 assert sha(OFFICIAL)==c['official_model_sha256'];write(ROOT/'SMOKE.json',dict(status='PASS',checks=out,independent_histories=True,frozen_eval_deterministic=True,test_read=False));print(out,flush=True)

def train(method,seed):
 c=contract();assert read(ROOT/'SMOKE.json')['status']=='PASS';out=ROOT/'sources'/f'{method}_s{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():assert sha(out/'final.pt')==read(out/'COMPLETE.json')['checkpoint_sha256'];return
 assert not (out/'RUN.json').exists(),'Existing unfinished attempt requires explicit resume; do not overwrite'
 seed_all(seed);m=Encoder(method).cuda().train();params=sum(p.numel() for p in m.parameters());lr=.5/math.sqrt(params) if method=='NOD' else 3e-4
 opt=torch.optim.RMSprop(m.parameters(),lr=lr) if method=='NOD' else torch.optim.Adam(m.parameters(),lr=lr)
 sched=torch.optim.lr_scheduler.StepLR(opt,500,.95) if method=='NOD' else torch.optim.lr_scheduler.CosineAnnealingLR(opt,20000)
 write(out/'RUN.json',dict(method=method,seed=seed,protocol_sha256=sha(ROOT/'PROTOCOL.json'),steps=20000,optimizer=type(opt).__name__,initial_lr=lr,parameters=params,selection='fixed_final_20000',test_read=False))
 began=time.monotonic()
 with (out/'train.jsonl').open('x') as log:
  for step in range(1,20001):
   b=batch('train',specs(seed,step));opt.zero_grad(set_to_none=True);loss=m.objective(*b);assert torch.isfinite(loss),step;loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();sched.step()
   if step==1 or step%100==0:
    row=dict(step=step,loss=float(loss),grad_norm=float(norm),seconds=time.monotonic()-began,lr=opt.param_groups[0]['lr']);log.write(json.dumps(row)+'\n');log.flush();write(out/'progress.json',row)
   if step%1000==0:
    p=out/f'step{step:06d}.pt';tmp=p.with_suffix('.tmp');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=step,method=method,seed=seed,protocol_sha256=sha(ROOT/'PROTOCOL.json'),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(p)
 final=out/'final.pt';os.link(out/'step020000.pt',final);assert sha(OFFICIAL)==c['official_model_sha256'];write(out/'COMPLETE.json',dict(status='COMPLETE',method=method,seed=seed,step=20000,checkpoint=str(final),checkpoint_sha256=sha(final),test_read=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);p.add_argument('--method',choices=['NOD','FCRL']);p.add_argument('--seed',type=int,choices=[0,1,2],default=0);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if a.action=='smoke':smoke()
 else:train(a.method,a.seed)
