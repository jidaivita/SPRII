"""Fixed-route JEPA source tracks, strict Random-Cross, epochs 1..150.

Uses the frozen sig02 architecture/self objective and original audited sampler.
No source coordinates. All-object route uses the existing v6.4 objective, with
all extra donors present from epoch1. Random has no correct-trained ancestor.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
import numpy as np
import torch

VERSION='cophy-v7-legacy-fixed-route-source-1'
BUDGETS=(50,100,150)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def immutable(p,x):
 if Path(p).exists() and read(p)!=x:raise ValueError('Changed frozen binding '+str(p))
 write(p,x)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));torch.save(x,q);os.replace(q,p)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def emit(event,**kw):print(json.dumps(dict(event=event,time=time.time(),**kw),allow_nan=False),flush=True)
def rng(device):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
def restore(r,device):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
 if device.type=='cuda':torch.cuda.set_rng_state(r['cuda'].cpu(),device)


class Planner:
 def __init__(self,core,route,data,index,seed,mode):
  self.core,self.route,self.data,self.mode=core,route,data,mode
  self.original=core.EpochPlanner(data,index,seed)
  self.full=route.RoutePlanner(data,self.original,seed) if mode=='all' else None
 def make(self,epoch,batch):
  if self.mode=='focal':return self.original.make(epoch,batch)
  plan=self.full.make(epoch,batch);ext=plan['external'];rand=np.full_like(ext,-1)
  groups={};index=self.original.index;ids=self.original.ids
  for i in np.flatnonzero(plan['common']):
   for slot in np.flatnonzero(ext[i]>=0):
    rec=index.records[(ids[i],int(slot))];groups.setdefault(rec['group'],[]).append((i,int(slot),int(ext[i,slot])))
  seed=int.from_bytes(hashlib.sha256(f'{VERSION}:all-random:{epoch}'.encode()).digest()[:8],'little');rr=np.random.default_rng(seed)
  rejected=set();accidental=0;total=0
  for group in groups.values():
   if len(group)<2:rejected.update(i for i,_,_ in group);continue
   for trial in range(256):
    order=rr.permutation(len(group))
    if all(group[int(j)][2]!=group[k][0] for k,j in enumerate(order)):break
   else:rejected.update(i for i,_,_ in group);continue
   for k,(i,s,_) in enumerate(group):
    donor=group[int(order[k])][2];rand[i,s]=donor
    a=index.records[(ids[i],s)];b=index.records[(ids[donor],s)]
    if a['group']!=b['group'] or donor==i:raise ValueError('Random public stratum/self violation')
    accidental+=int(a['physical_key']==b['physical_key']);total+=1
  if rejected:plan['common'][list(rejected)]=False
  # Removing a failed recipient can affect the multiset only in exceptional
  # unrandomizable strata. Disable the complete full-object experiment if that
  # happens rather than quietly claim matched donor exposure.
  if rejected:raise ValueError('Full-object randomization unsupported for '+str(len(rejected))+' recipients')
  plan['random_external']=rand;plan['random_same']=accidental;plan['random_objects']=total
  h=hashlib.sha256(plan['plan_sha256'].encode());h.update(rand.tobytes());h.update(plan['common'].tobytes());plan['plan_sha256']=h.hexdigest()
  return plan


def runtime(a):
 old=Path(a.runtime_dir).resolve();sys.path.insert(0,str(old.parent));sys.path.insert(0,str(old))
 core=load('_v7_original_sig02',old/'train.py')
 if core.VERSION!='cophy-latent-v6.2-sig02':raise ValueError('Requires qualified common SIGReg .2 runtime')
 route=load('_v7_original_route',Path(a.route_code).resolve()) if a.route=='all' else None
 if route is not None:
  # Initialize its torch/numpy imports without calling its old continuation setup.
  route.np=np;route.torch=torch
 return core,route


def setup(a):
 torch.set_num_threads(a.threads);core,route=runtime(a)
 data=core.FeatureData(a.features,a.scene)
 expected={'balls':(7000,2000),'collision':(14000,4000),'blocktower':(28310,8088)}[a.scene]
 if tuple(len(data.ids[s]) for s in ('train','val'))!=expected:raise ValueError('Incomplete official source data')
 planner=Planner(core,route,data,a.relation_index,a.seed,a.route)
 files=dict(data.files)
 for p in [__file__,str(Path(a.runtime_dir)/'train.py'),str(Path(a.runtime_dir)/'models.py'),str(Path(a.runtime_dir).parent/'cophy_relations.py'),a.relation_index]+([a.route_code] if route else []):files[str(Path(p).resolve())]=sha(p)
 body=dict(version=VERSION,model_version=core.VERSION,scene=a.scene,method=a.method,route=a.route,
  source_budget=150,seed=a.seed,lr=.0003,batch=32,weight_decay=.0001,clip_norm=1.,family='JEPA',
  source_pretrained_checkpoint=None,lambda_cross=1.,lambda_align=0.,sigreg=.2,
  random_policy='conditional permutation of identical correct donor multiset within public slot/type/gravity; accidental matches retained',
  route_schedule='fixed '+a.route+' from epoch1',test_read=False,files=files)
 body['sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 core.seeded(a.seed);model=core.make_model('JEPA').to(a.device)
 model.config=replace(model.config,lambda_align=0.)
 return core,route,data,planner,model,body


def objective(core,route,model,data,plan,ix,method,mode,micro):
 if mode=='focal':
  changed=dict(plan)
  if method=='Random-Cross':changed['correct']=plan['random']
  return core.batch_objective(model,data,changed,ix,'Cross',micro)
 changed=dict(plan)
 if method=='Random-Cross':
  changed['external']=plan['random_external']
  ids=np.arange(len(plan['focal']));focal=plan['focal'].clip(0)
  changed['correct']=changed['external'][ids,focal]
 return route.objective(core,model,data,changed,ix,'all',micro)


def prepare(a):
 core,route,data,planner,model,b=setup(a);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 immutable(out/'binding.json',b)
 initial=dict(status='COMPLETE',model_state_sha256=core.state_sha(model),source_pretrained_checkpoint=None,seed=a.seed,
  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
 immutable(out/'initialization.json',initial);plan=planner.make(1,32)
 write(out/'prepared.json',dict(status='COMPLETE',binding_sha256=b['sha256'],initialization=initial,
  first_plan_sha256=plan['plan_sha256'],paired=plan['paired'],random_same=plan['random_same'],test_read=False))


def checked(a):
 values=setup(a)
 if read(Path(a.out)/'binding.json')!=values[-1]:raise ValueError('Run matching prepare first')
 return values


def smoke(a):
 core,route,data,planner,model,b=checked(a);plan=planner.make(1,32);ix=plan['batches'][0];device=torch.device(a.device)
 initial={k:v.detach().clone() for k,v in model.state_dict().items()};rows=[]
 for method in ('Cross','Random-Cross'):
  model.load_state_dict(initial);core.seeded(a.seed);model.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
   loss,metrics,_=objective(core,route,model,data,plan,ix,method,a.route,a.microbatch)
  if not torch.isfinite(loss):raise ValueError('Nonfinite real loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  if float(norm)<=0:raise ValueError('No source gradient')
  rows.append(dict(method=method,loss=float(loss.detach()),self_loss=metrics['self_loss'],cross=metrics['cross'],grad_norm=float(norm)))
 if not np.isclose(rows[0]['self_loss'],rows[1]['self_loss'],atol=1e-6,rtol=1e-5):raise ValueError('Self paths differ')
 write(Path(a.out)/'smoke.json',dict(status='PASS',binding_sha256=b['sha256'],rows=rows,optimizer_steps=0,test_read=False))


def train(a):
 core,route,data,planner,model,b=checked(a);out=Path(a.out);device=torch.device(a.device)
 if read(out/'smoke.json').get('binding_sha256')!=b['sha256'] or read(out/'smoke.json')['status']!='PASS':raise ValueError('Missing real smoke')
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);epoch,batch,step=1,0,0;history=[];running={};prior=0.
 if (out/'latest.pt').exists():
  ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  if ck['experiment_binding_sha256']!=b['sha256']:raise ValueError('Different source continuation')
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore(ck['active_rng'],device)
  epoch,batch,step=ck['next_epoch'],ck['next_batch'],ck['step'];history=ck['history'];running=ck['running'];prior=ck['seconds']
 began=time.monotonic()
 def snapshot(next_epoch,next_batch):
  return dict(version=core.VERSION,experiment_version=VERSION,model=model.state_dict(),model_config=model.artifact_config(),
   optimizer=opt.state_dict(),active_rng=rng(device),rng_device_index=0,method=a.method,family='JEPA',scene=a.scene,
   epoch=next_epoch-1 if next_batch==0 else next_epoch,next_epoch=next_epoch,next_batch=next_batch,step=step,
   history=history,running=running,seconds=prior+time.monotonic()-began,experiment_binding=b,experiment_binding_sha256=b['sha256'],
   source_pretrained_checkpoint=None,test_read=False)
 while epoch<=a.through:
  plan=planner.make(epoch,32)
  if batch==0:running=dict(plan_sha256=plan['plan_sha256'],examples=0,loss=0.,started=time.time())
  elif running['plan_sha256']!=plan['plan_sha256']:raise ValueError('Resume support plan changed')
  model.train()
  for j in range(batch,len(plan['batches'])):
   ix=plan['batches'][j];opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
    loss,metrics,_=objective(core,route,model,data,plan,ix,a.method,a.route,a.microbatch)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
   step+=1;batch=j+1;running['examples']+=len(ix);running['loss']+=float(loss.detach())*len(ix)
   if step%50==0 or batch==len(plan['batches']):
    save(out/'latest.pt',snapshot(epoch,batch));write(out/'progress.json',dict(status='RUNNING',epoch=epoch,batch=batch,step=step,
      metrics=metrics,grad_norm=float(gn),seconds=prior+time.monotonic()-began,test_read=False))
   if a.max_steps and step>=a.max_steps and batch<len(plan['batches']):
    save(out/'latest.pt',snapshot(epoch,batch));emit('PAUSED',step=step);return
  if running['examples']!=len(data.ids['train']):raise ValueError('Incomplete epoch exposure')
  val=core.evaluate(model,data,min(a.microbatch,32));history.append(dict(epoch=epoch,latent_validation=val,
    train_loss=running['loss']/running['examples'],plan_sha256=plan['plan_sha256'],steps=step,seconds=time.time()-running['started']))
  done=epoch;epoch+=1;batch=0;running={};state=snapshot(epoch,0)
  if done in BUDGETS:
   cp=out/f'checkpoint_{done}.pt';marker=out/f'budget_{done}_complete.json'
   if not marker.exists():save(cp,state);write(marker,dict(status='COMPLETE',scene=a.scene,method=a.method,route=a.route,epochs=done,
      checkpoint=str(cp),checkpoint_sha256=sha(cp),steps=step,binding_sha256=b['sha256'],source_pretrained_checkpoint=None,test_read=False))
  save(out/'latest.pt',state);emit('EPOCH_COMPLETE',epoch=done,step=step,method=a.method,route=a.route)
  if a.max_steps and step>=a.max_steps:return
 write(out/'complete.json',dict(status='COMPLETE',through=a.through,epochs=epoch-1,step=step,binding_sha256=b['sha256'],test_read=False))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','smoke','train'));p.add_argument('--runtime-dir',required=True)
 p.add_argument('--route-code');p.add_argument('--features',required=True);p.add_argument('--relation-index',required=True)
 p.add_argument('--scene',required=True,choices=('balls','collision','blocktower'));p.add_argument('--out',required=True)
 p.add_argument('--method',required=True,choices=('Cross','Random-Cross'));p.add_argument('--route',choices=('focal','all'),default='focal')
 p.add_argument('--through',type=int,choices=BUDGETS,default=50);p.add_argument('--seed',type=int,choices=(0,),default=0)
 p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4);p.add_argument('--microbatch',type=int,default=32);p.add_argument('--max-steps',type=int,default=0)
 a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
 with open(Path(a.out)/'owner.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:{'prepare':prepare,'smoke':smoke,'train':train}[a.command](a)
  except Exception as e:write(Path(a.out)/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),test_read=False));raise
