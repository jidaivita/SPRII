"""Continue qualified Base/Cross50 or Monolithic100 using unchanged objectives.

No Random imports a correctly trained parent. No old artifacts are modified.
The active parent RNG is identified by saved single-stream metadata or an
explicit recorded original worker device, never guessed from tensor contents.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import jepa_source as io

VERSION='cophy-v7-qualified-jepa-continuation-1'


def parent_index(a,ck):
 values=ck['rng']['cuda']
 if len(values)==1 and ck.get('rng_device_index')==0:return 0
 if a.parent_rng_index is not None:
  if not 0<=a.parent_rng_index<len(values):raise ValueError('RNG index outside saved states')
  return a.parent_rng_index
 worker=Path(a.parent_checkpoint).parent/'worker.json'
 if not worker.exists():raise ValueError('Pass audited --parent-rng-index; worker receipt absent')
 row=io.read(worker);dev=torch.device(row['device']);index=dev.index
 if index is None or index>=len(values):raise ValueError('Original worker receipt has ambiguous device')
 # worker.json is emitted inside the original physical-device process, and is
 # hash-bound below. It is not a job-level external CUDA_VISIBLE_DEVICES guess.
 return index


def setup(a):
 torch.set_num_threads(a.threads);runtime=Path(a.runtime_dir).resolve();sys.path.insert(0,str(runtime.parent));sys.path.insert(0,str(runtime))
 core=io.load('_v7_qualified_core',runtime/'train.py');cp=Path(a.parent_checkpoint).resolve();ck=torch.load(cp,map_location='cpu',weights_only=False)
 if ck['scene']!=a.scene or ck['method']!=a.method or ck['family']!='JEPA':raise ValueError('Parent identity differs')
 expected='cophy-monolithic-jepa-v6.6' if a.method=='Monolithic' else 'cophy-latent-v6.2-sig02'
 if ck['version']!=expected or core.VERSION!=expected:raise ValueError('Parent/runtime version differs')
 start=100 if a.method=='Monolithic' else 50
 if (ck['epoch'],ck['next_epoch'],ck['next_batch'])!=(start,start+1,0):raise ValueError('Parent is not exact completed original budget')
 if ck.get('test_read') is not False or not ck.get('optimizer'):raise ValueError('Missing optimizer or invalid split')
 b=ck['binding'];files={str(cp):io.sha(cp)}
 for path,digest in b['files'].items():
  if io.sha(path)!=digest:raise ValueError('Parent bound dependency changed '+path)
  files[path]=digest
 for name in ('train.py','models.py'):
  path=str(runtime/name)
  if files.get(path)!=io.sha(path):raise ValueError('Different original runtime '+name)
 if any(g['lr']!=.0003 or g['weight_decay']!=.0001 for g in ck['optimizer']['param_groups']):raise ValueError('Unexpected original optimizer')
 index=parent_index(a,ck)
 worker=cp.parent/'worker.json'
 if worker.exists():files[str(worker)]=io.sha(worker)
 files[str(Path(__file__).resolve())]=io.sha(__file__);files[str(Path(io.__file__).resolve())]=io.sha(io.__file__)
 data=core.FeatureData(b['features'],a.scene)
 if a.method=='Monolithic':planner=core.EpochPlanner(data,b['seed'])
 else:
  rel=[p for p in b['files'] if Path(p).suffix=='.json' and 'relation' in Path(p).name and io.read(p).get('version')=='cophy-relation-index-v3']
  if len(rel)!=1:raise ValueError('Nonunique audited RelationIndex')
  planner=core.EpochPlanner(data,rel[0],b['seed'])
 body=dict(version=VERSION,scene=a.scene,method=a.method,parent_checkpoint=str(cp),parent_checkpoint_sha256=io.sha(cp),parent_epoch=start,
  parent_step=ck['step'],parent_rng_index=index,parent_binding_sha256=ck['binding_sha256'],final_epoch=150,files=files,
  objective='unchanged original '+a.method,source_selection='fixed budget checkpoints',test_read=False)
 body['sha256']=io.hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return core,ck,data,planner,body


def model_from(core,ck,device,index):
 model=core.make_model(config=ck['model_config']).to(device);model.load_state_dict(ck['model'],strict=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);opt.load_state_dict(ck['optimizer'])
 r=ck['rng'];io.restore(dict(r,cuda=r['cuda'][index] if r['cuda'] else None),device)
 return model,opt


def objective(a,core,model,data,plan,ix,micro):
 if a.method=='Monolithic':return core.batch_objective(model,data,ix,micro)
 loss,metrics,detail=core.batch_objective(model,data,plan,ix,a.method,micro)
 if a.method=='Base' and (metrics['cross']!=0 or metrics['align']!=0 or detail['donor_p'] is not None):raise ValueError('Base has relation path')
 return loss,metrics


def checked(a):
 parts=setup(a)
 if io.read(Path(a.out)/'binding.json')!=parts[-1]:raise ValueError('Prepare exact binding first')
 return parts


def prepare(a):
 core,ck,data,planner,b=setup(a);io.immutable(Path(a.out)/'binding.json',b)
 plan=planner.make(ck['next_epoch'],32)
 io.write(Path(a.out)/'prepared.json',dict(status='COMPLETE',binding_sha256=b['sha256'],next_epoch=ck['next_epoch'],first_plan_sha256=plan.get('plan_sha256',plan.get('order_sha256')),test_read=False))


def smoke(a):
 core,ck,data,planner,b=checked(a);dev=torch.device(a.device);model,opt=model_from(core,ck,dev,b['parent_rng_index']);model.train()
 plan=planner.make(ck['next_epoch'],32)
 with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):loss,metrics=objective(a,core,model,data,plan,plan['batches'][0],ck['microbatch'])
 if not torch.isfinite(loss):raise ValueError('Nonfinite real parent-continuation loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 io.write(Path(a.out)/'smoke.json',dict(status='PASS',binding_sha256=b['sha256'],metrics=metrics,gradient_norm=float(norm),optimizer_steps=0,test_read=False))


def train(a):
 core,parent,data,planner,b=checked(a);out=Path(a.out);dev=torch.device(a.device)
 if io.read(out/'smoke.json') is None or io.read(out/'smoke.json')['status']!='PASS':raise ValueError('Real smoke missing')
 old=parent
 if (out/'latest.pt').exists():
  old=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  if old.get('extension_binding_sha256')!=b['sha256']:raise ValueError('Different continuation')
 model,opt=model_from(core,old,dev,0 if old is not parent else b['parent_rng_index'])
 epoch,batch,step=old['next_epoch'],old['next_batch'],old['step'];history=list(old['history']);running=dict(old.get('running',{}));micro=old['microbatch']
 def record(ne,nb):
  r=io.rng(dev);r['cuda']=[r['cuda']] if r['cuda'] is not None else []
  return dict(version=core.VERSION,experiment_version=VERSION,scene=a.scene,method=a.method,family='JEPA',model=model.state_dict(),model_config=model.artifact_config(),optimizer=opt.state_dict(),rng=r,rng_device_index=0,epoch=ne-1 if nb==0 else ne,next_epoch=ne,next_batch=nb,step=step,history=history,running=running,microbatch=micro,binding=parent['binding'],binding_sha256=parent['binding_sha256'],extension_binding=b,extension_binding_sha256=b['sha256'],initialization_sha256=parent['initialization_sha256'],test_read=False)
 while epoch<=a.through:
  plan=planner.make(epoch,32);digest=plan.get('plan_sha256',plan.get('order_sha256'))
  if batch==0:running=dict(plan_sha256=digest,examples=0,loss=0.,started=time.time())
  elif running['plan_sha256']!=digest:raise ValueError('Resumed plan changed')
  model.train()
  for j in range(batch,len(plan['batches'])):
   ix=plan['batches'][j];opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):loss,metrics=objective(a,core,model,data,plan,ix,micro)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1;batch=j+1
   running['examples']+=len(ix);running['loss']+=float(loss.detach())*len(ix)
   if step%50==0 or batch==len(plan['batches']):
    io.save(out/'latest.pt',record(epoch,batch));io.write(out/'progress.json',dict(status='RUNNING',epoch=epoch,batch=batch,step=step,metrics=metrics,gradient_norm=float(norm),test_read=False))
  if running['examples']!=len(data.ids['train']):raise ValueError('Incomplete source epoch')
  val=core.evaluate(model,data,min(32,micro));history.append(dict(epoch=epoch,latent_validation=val,train_loss=running['loss']/running['examples'],seconds=time.time()-running['started'],steps=step));done=epoch;epoch+=1;batch=0;running={}
  state=record(epoch,0)
  if done in (100,150):
   cp=out/f'checkpoint_{done}.pt';marker=out/f'budget_{done}_complete.json'
   if not marker.exists():io.save(cp,state);io.write(marker,dict(status='COMPLETE',epochs=done,method=a.method,scene=a.scene,steps=step,checkpoint=str(cp),checkpoint_sha256=io.sha(cp),binding_sha256=b['sha256'],test_read=False))
  io.save(out/'latest.pt',state);io.emit('EPOCH_COMPLETE',epoch=done,step=step,method=a.method)
 io.write(out/'complete.json',dict(status='COMPLETE',epochs=epoch-1,through=a.through,steps=step,binding_sha256=b['sha256'],test_read=False))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','smoke','train'));p.add_argument('--runtime-dir',required=True);p.add_argument('--parent-checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--scene',choices=('balls','blocktower'),required=True);p.add_argument('--method',choices=('Monolithic','Base','Cross'),required=True);p.add_argument('--parent-rng-index',type=int);p.add_argument('--through',type=int,choices=(100,150),default=100);p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
 with open(Path(a.out)/'owner.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:{'prepare':prepare,'smoke':smoke,'train':train}[a.command](a)
  except Exception as e:io.write(Path(a.out)/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),test_read=False));raise
