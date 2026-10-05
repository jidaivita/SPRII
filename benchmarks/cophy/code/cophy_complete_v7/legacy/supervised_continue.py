"""Original CoPhy supervised source continuation; never Native-FT or MQ heads.

v3 Native/A/Random keep original adapter.objective and train loader permutation.
v51 Both-new/Random-Both-new keep their exact source sampler and objective.
Validation-only selection extends the same schedule (v3 every5; v51 every1).
"""

import os
import argparse
from collections import defaultdict
import fcntl
import importlib
import json
from pathlib import Path
import pickle
import random
import sys
import time
import traceback
import numpy as np
import torch
import jepa_source as io

VERSION='cophy-v7-original-supervised-continuation-1'
SPECS={'balls':(30,9,2,2),'collision':(15,4,3,5),'blocktower':(30,4,3,2)}


def runtime(a):
 root=Path(a.root);source=Path(a.runtime_source) if a.runtime_source else root/('source_blocktower_gate1' if a.scene=='blocktower' else 'source')
 sys.path.insert(0,str(source));names=('cophy_adapter','cophy_relations','cophy_training','cophy_protocol','cf_learning.model','dataloaders.utils')
 modules={name:importlib.import_module(name) for name in names}
 for m in modules.values():
  if source.resolve() not in Path(m.__file__).resolve().parents:raise ValueError('Mixed source runtimes')
 v51=None
 if a.profile=='v51':
  if a.scene!='collision' or a.method not in ('A','Random'):raise ValueError('v51 preserved only for Collision relation sources')
  v51=io.load('_v7_unchanged_supervised_v51',a.v51_code)
 return source,modules,v51


def prepare_data(a):
 source,modules,v51=runtime(a);out=Path(a.packed);out.mkdir(parents=True,exist_ok=True);marker=out/'manifest.json'
 if marker.exists():
  saved=io.read(marker)
  if saved['scene']!=a.scene or saved['version']!=VERSION:raise ValueError('Different source target pack')
  for path,digest in saved['files'].items():
   if io.sha(path)!=digest:raise ValueError('Source pack file changed')
  return
 root=Path(a.root);pre=root/'prepared_v3'/('blocktower_gate1' if a.scene=='blocktower' else a.scene)/'training_preflight.json';pf=io.read(pre)
 if pf['status']!='PASS' or pf['test_read'] is not False:raise ValueError('Unqualified scene audit')
 rel=modules['cophy_relations'];splits=rel.read_artifact(pre,'splits');relation=rel.artifact_path(pre,'relation_index');derenderer=rel.artifact_path(pre,'derenderer')
 files={str(pre):io.sha(pre),str(relation):io.sha(relation),str(derenderer):io.sha(derenderer)};spec=SPECS[a.scene]
 existing=root/'source_formation_v51_data'/a.scene/'manifest.json';reuse=io.read(existing) if existing.exists() else None
 rawroot=Path(pf['input_profile']['dataset_dir'])
 if a.scene in ('balls','blocktower'):rawroot/=str(pf['input_profile']['num_objects'])
 rows={}
 for split in ('train','val'):
  ids=list(map(str,splits[split]['ids']));cachepath=rel.artifact_path(pre,'cache_'+split)
  files[str(cachepath)]=io.sha(cachepath)
  if reuse is not None:
   sourcepack=Path(reuse['splits'][split]['path'])
   if io.sha(sourcepack)!=reuse['splits'][split]['sha256']:raise ValueError('Existing source target pack altered')
   with np.load(sourcepack,allow_pickle=False) as z:
    if z['ids'].astype(str).tolist()!=ids:raise ValueError('Existing source pack ID order differs')
    if z['target_pose'].shape!=(len(ids),spec[0]-1,spec[1],3):raise ValueError('Truncated source target pack forbidden')
   rows[split]=dict(path=str(sourcepack),count=len(ids),sha256=io.sha(sourcepack));files[str(sourcepack)]=io.sha(sourcepack);continue
  with open(cachepath,'rb') as f:cache=pickle.load(f)
  if set(cache)!=set(ids):raise ValueError('Visual cache/split mismatch')
  col={k:[] for k in ('pose_ab','presence_ab','pose_c','presence_c','target_pose','target_stationary','target_presence')}
  for ident in ids:
   item=cache[ident]
   if item['cache_version']!='ab_c_float32_v2':raise ValueError('Future-bearing visual cache rejected')
   for key in ('pose_ab','presence_ab','pose_c','presence_c'):col[key].append(np.asarray(item[key],np.float32))
   states=np.load(rawroot/ident/'cd/states.npy',allow_pickle=False);pose=np.asarray(states[:,:,:3],np.float32)
   if pose.shape!=(spec[0],spec[1],3) or not np.isfinite(pose).all():raise ValueError('Raw target shape/value')
   presence=(np.abs(pose[0]).sum(-1)>0).astype(np.float32);stationary=modules['dataloaders.utils'].get_stab(pose,presence,t_delta=spec[3],eps=.05)
   col['target_pose'].append(pose[1:]);col['target_stationary'].append(stationary[1:]);col['target_presence'].append(presence)
  path=out/(split+'.npz');tmp=path.with_suffix('.tmp')
  with open(tmp,'wb') as f:np.savez(f,ids=np.asarray(ids),**{k:np.stack(v) for k,v in col.items()})
  tmp.replace(path);rows[split]=dict(path=str(path),count=len(ids),sha256=io.sha(path));files[str(path)]=io.sha(path)
  io.emit('PACKED',scene=a.scene,split=split,rows=len(ids))
 packed=dict(version=VERSION,status='COMPLETE',scene=a.scene,splits=rows,files=files,relation_index=str(relation),derenderer=str(derenderer),preflight=str(pre),spec=dict(frames=spec[0],slots=spec[1],dims=spec[2]),test_read=False,target_rule='original full CD[1:] xyz and stationary',input_rule='audited full AB and exactly one C frame')
 io.write(marker,packed)


class Dataset(torch.utils.data.Dataset):
 def __init__(self,path):
  with np.load(path,allow_pickle=False) as z:self.list_ex=z['ids'].astype(str).tolist();self.rows={k:z[k].copy() for k in z.files if k!='ids'}
  self.is_rgb=False;self.num_objects=self.rows['pose_c'].shape[2]
  self.dict_id2object_properties={ident:dict(cache_version='ab_c_float32_v2',**{k:self.rows[k][i] for k in ('pose_ab','presence_ab','pose_c','presence_c')}) for i,ident in enumerate(self.list_ex)}
 def __len__(self):return len(self.list_ex)
 def __getitem__(self,i):
  r=self.rows
  # First target slot is padding discarded by original targets_from_batch.
  # It never enters input_from_batch, which allowlists pred_* fields only.
  return dict(id=self.list_ex[i],pred_pose_3D_ab=r['pose_ab'][i],pred_presence_ab=r['presence_ab'][i],pred_pose_3D_cd=r['pose_c'][i],pred_presence_cd=r['presence_c'][i],pose_3D_cd=np.concatenate([r['target_pose'][i,:1]*0,r['target_pose'][i]],0),stab_cd=np.concatenate([r['target_stationary'][i,:1]*0,r['target_stationary'][i]],0),presence_cd=r['target_presence'][i])


def setup(a):
 torch.set_num_threads(a.threads);source,modules,v51=runtime(a);manifest=Path(a.packed)/'manifest.json';packed=io.read(manifest)
 if packed['status']!='COMPLETE' or packed['scene']!=a.scene or packed['test_read'] is not False:raise ValueError('Source pack unavailable')
 modules['cophy_protocol'].verify_adapter_binding(packed['preflight'])
 for path,digest in packed['files'].items():
  if io.sha(path)!=digest:raise ValueError('Source pack dependency changed '+path)
 cp=Path(a.parent_checkpoint).resolve();old=torch.load(cp,map_location='cpu',weights_only=False);config=old.get('run_config',old.get('config'))
 expected=('Both-new' if a.method=='A' else 'Random-Both-new') if a.profile=='v51' else a.method
 if config['method']!=expected or config['scene']!=a.scene or old['epoch']!=50 or not old.get('optimizer'):raise ValueError('Wrong original source50 full checkpoint')
 if config.get('phase') not in (None,'source_formation') or any(k.startswith(('encoder.','head.')) for k in old['model']):raise ValueError('FT/MQ checkpoints forbidden')
 if a.profile=='v51' and config['version']!=v51.VERSION:raise ValueError('Wrong v51 configuration')
 slots=SPECS[a.scene][1];model=modules['cophy_adapter'].PTCoPhy(modules['cf_learning.model'].CoPhyNet(slots),a.method).to(a.device)
 for param in model.derendering.parameters():param.requires_grad_(False)
 model.load_state_dict(old['model'],strict=True);model.derendering.eval()
 optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);optimizer.load_state_dict(old['optimizer'])
 if any(group['lr']!=.001 for group in optimizer.param_groups):raise ValueError('Original supervised learning rate changed')
 savedrng=old['cuda_rng']
 index=0 if len(savedrng)==1 else a.parent_rng_index
 if index is None or not 0<=index<len(savedrng):raise ValueError('Explicit original active CUDA RNG index required')
 selected=Path(a.parent_selected).resolve();best=torch.load(selected,map_location='cpu',weights_only=False)
 if best.get('run_config',best.get('config'))!=config:raise ValueError('Selected model from different source run')
 files={str(cp):io.sha(cp),str(selected):io.sha(selected),str(manifest):io.sha(manifest),str(Path(__file__).resolve()):io.sha(__file__),str(Path(io.__file__).resolve()):io.sha(io.__file__),**packed['files']}
 for m in modules.values():files[str(Path(m.__file__).resolve())]=io.sha(m.__file__)
 if v51:files[str(Path(a.v51_code).resolve())]=io.sha(a.v51_code)
 binding=dict(version=VERSION,scene=a.scene,method=a.method,profile=a.profile,parent_checkpoint=str(cp),parent_selected=str(selected),parent_rng_index=index,parent_config=config,start_epoch=50,end_epoch=150,source_selection='original v3 every5 epochs' if a.profile=='v3' else 'original v51 every epoch',files=files,test_read=False)
 binding['sha256']=io.hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return old,best,model,optimizer,modules,v51,packed,binding


def load_rng(old,device,index):
 if 'python_rng' in old:random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng'])
 torch.set_rng_state(old['torch_rng'].cpu())
 if torch.device(device).type=='cuda':torch.cuda.set_rng_state(old['cuda_rng'][index].cpu(),device)


def make_data(a,modules,v51,packed,old):
 if v51:
  data=v51.SourceData(packed,a.device);return data,v51.SourceSampler(data),None
 train=Dataset(packed['splits']['train']['path']);val=Dataset(packed['splits']['val']['path']);generator=torch.Generator();generator.set_state(old['loader_rng'].cpu())
 loader=torch.utils.data.DataLoader(train,batch_size=old['run_config']['batch_size'],shuffle=True,generator=generator,num_workers=0)
 validation=torch.utils.data.DataLoader(val,batch_size=8,generator=torch.Generator().manual_seed(1),num_workers=0)
 pair=None
 if a.method!='Native':pair=modules['cophy_relations'].PairProvider(modules['cophy_relations'].RelationIndex(io.read(packed['relation_index']),train.list_ex,a.scene),train,old['run_config']['seed'])
 return loader,validation,pair


def prepare(a):
 *_,b=setup(a);io.immutable(Path(a.out)/'binding.json',b);io.write(Path(a.out)/'prepared.json',dict(status='COMPLETE',binding_sha256=b['sha256'],test_read=False))


def check(a):
 v=setup(a)
 if io.read(Path(a.out)/'binding.json')!=v[-1]:raise ValueError('Prepare exact source continuation first')
 return v


def smoke(a):
 old,best,model,opt,modules,v51,packed,b=check(a);data,extra,pair=make_data(a,modules,v51,packed,old);load_rng(old,a.device,b['parent_rng_index']);model.train();opt.zero_grad(set_to_none=True)
 if v51:
  plan=extra.make_epoch(51);loss,detail=v51.batch_objective(model,data,plan,0,min(v51.BATCH_GROUPS,len(plan['focal'])),old['config']['method'],{'adapter':modules['cophy_adapter'],'training':modules['cophy_training'],'protocol':modules['cophy_protocol']})
 else:
  batch=next(iter(data));visual=model.input_from_batch(batch,a.device,False);target=modules['cophy_training'].targets_from_batch(batch,a.device);pairs=None if pair is None else pair.make(batch['id'],model.method,51,0,visual)[0]
  loss,_,detail=modules['cophy_adapter'].objective(model,visual,target,pairs,None,old['run_config']['lambda_x'],old['run_config']['lambda_p'])
 if not torch.isfinite(loss):raise ValueError('Nonfinite original real batch loss')
 loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
 if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite/absent source gradient')
 io.write(Path(a.out)/'smoke.json',dict(status='PASS',binding_sha256=b['sha256'],loss=float(loss.detach()),optimizer_steps=0,profile=a.profile,method=a.method,test_read=False))


def train(a):
 parent,selected,model,opt,modules,v51,packed,b=check(a);out=Path(a.out);dev=torch.device(a.device)
 if io.read(out/'smoke.json')['status']!='PASS':raise ValueError('Real smoke required')
 old=parent
 if (out/'latest.pt').exists():
  old=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  if old['extension_binding_sha256']!=b['sha256']:raise ValueError('Different supervised continuation')
  model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
 else:io.save(out/'selected.pt',selected)
 data,extra,pair=make_data(a,modules,v51,packed,old);load_rng(old,dev,b['parent_rng_index'] if old is parent else 0)
 best=old['best'] if v51 else old['best_score'];history=list(old.get('history',[]));start=old['epoch']+1
 config=parent.get('run_config',parent.get('config'))
 def record(epoch):
  x=dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),extension_version=VERSION,extension_binding_sha256=b['sha256'],history=history,torch_rng=torch.get_rng_state(),cuda_rng=[torch.cuda.get_rng_state(dev)] if dev.type=='cuda' else [],rng_device_index=0,test_read=False)
  if v51:x.update(config=config,best=best)
  else:x.update(run_config=config,best_score=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=data.generator.get_state())
  return x
 for epoch in range(start,a.through+1):
  began=time.monotonic();model.train()
  if v51:
   plan=extra.make_epoch(epoch);seen=0;total=0.;updates=0
   for first in range(0,len(plan['focal']),v51.BATCH_GROUPS):
    opt.zero_grad(set_to_none=True);loss,detail=v51.batch_objective(model,data,plan,first,min(first+v51.BATCH_GROUPS,len(plan['focal'])),config['method'],{'adapter':modules['cophy_adapter'],'training':modules['cophy_training'],'protocol':modules['cophy_protocol']})
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite gradients')
    opt.step();n=len(detail['indices']);seen+=n;updates+=1;total+=float(loss.detach())*n
   if seen!=len(data.ids['train']):raise ValueError('Incomplete v51 source epoch')
   metric=v51.evaluate(model,data,{'adapter':modules['cophy_adapter'],'training':modules['cophy_training'],'protocol':modules['cophy_protocol']});score=metric['mse'];row=dict(epoch=epoch,mse=score,loss=total/seen,updates=updates,recipients=seen,plan_sha256=plan['plan_sha256'])
  else:
   row=modules['cophy_training'].train_adapter_epoch(model,dev,data,opt,str(out/'train.jsonl'),epoch=epoch,pair_provider=pair,lambda_x=config['lambda_x'],lambda_p=config['lambda_p'],dims=SPECS[a.scene][2])
   score=modules['cophy_training'].validate_adapter(model,dev,extra,str(out/'validation.jsonl'),dims=SPECS[a.scene][2],epoch=epoch);row['mse']=score
  row['seconds']=time.monotonic()-began;history.append(row)
  if (v51 or epoch%5==0) and score<best:
   best=score;ck=dict(model=model.state_dict(),epoch=epoch,**({'config':config} if v51 else {'run_config':config}))
   io.save(out/'selected.pt',ck);io.write(out/'selected_validation.json',dict(epoch=epoch,mse=best,test_read=False))
  ck=record(epoch)
  if epoch in (100,150) and not (out/f'budget_{epoch}_complete.json').exists():
   cp=out/f'checkpoint_{epoch}.pt';sel=out/f'selected_budget{epoch}.pt';io.save(cp,ck)
   # A frozen copy prevents later training from changing the budget readout input.
   chosen=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);io.save(sel,chosen)
   io.write(out/f'budget_{epoch}_complete.json',dict(status='COMPLETE',epochs=epoch,selected_epoch=chosen['epoch'],checkpoint=str(cp),checkpoint_sha256=io.sha(cp),selected_checkpoint=str(sel),selected_sha256=io.sha(sel),best_mse=best,profile=a.profile,method=a.method,scene=a.scene,binding_sha256=b['sha256'],test_read=False))
  io.save(out/'latest.pt',ck);io.write(out/'progress.json',dict(status='RUNNING',epoch=epoch,best_mse=best,latest=row,test_read=False));io.emit('SOURCE_EPOCH',scene=a.scene,method=a.method,**row)
 io.write(out/'complete.json',dict(status='COMPLETE',through=a.through,epochs=max(a.through,start-1),best_mse=best,binding_sha256=b['sha256'],test_read=False))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare-data','prepare','smoke','train'));p.add_argument('--root',default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")));p.add_argument('--scene',choices=tuple(SPECS),required=True);p.add_argument('--method',choices=('Native','A','Random'),default='Native');p.add_argument('--profile',choices=('v3','v51'),default='v3');p.add_argument('--packed',required=True);p.add_argument('--runtime-source');p.add_argument('--v51-code');p.add_argument('--parent-checkpoint');p.add_argument('--parent-selected');p.add_argument('--parent-rng-index',type=int);p.add_argument('--out',required=True);p.add_argument('--through',choices=(100,150),type=int,default=100);p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
 with open(Path(a.out)/'owner.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:{'prepare-data':prepare_data,'prepare':prepare,'smoke':smoke,'train':train}[a.command](a)
  except Exception as e:io.write(Path(a.out)/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),test_read=False));raise
