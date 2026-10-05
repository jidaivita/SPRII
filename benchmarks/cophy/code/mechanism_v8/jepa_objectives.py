"""Complete the fixed-route JEPA objective factorial, source training from seed0.

Reuses the v7 source trainer, optimizer, exposure and sampler exactly. Align is
focal P alignment in every scene, including Collision's all-object Cross route.
"""
import argparse,fcntl,hashlib,importlib.util,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

VERSION='cophy-mechanism-v8-jepa-objectives-1'

def parent(path):
 s=importlib.util.spec_from_file_location('_v8_parent_source',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','smoke','train'));p.add_argument('--parent-code',required=True)
 p.add_argument('--runtime-dir',required=True);p.add_argument('--route-code');p.add_argument('--features',required=True);p.add_argument('--relation-index',required=True)
 p.add_argument('--scene',required=True);p.add_argument('--method',choices=('Align','Both','Random-Both'),required=True);p.add_argument('--route',choices=('focal','all'),required=True)
 p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--through',type=int,choices=(50,100),default=100)
 a=p.parse_args();a.seed=0;a.threads=2;a.microbatch=32;a.max_steps=0
 m=parent(a.parent_code);original_setup=m.setup
 def setup(args):
  core,route,data,planner,model,b=original_setup(args)
  model.config=replace(model.config,lambda_align=.1,lambda_cross=0. if args.method=='Align' else 1.)
  b.update(mechanism_version=VERSION,source_budget=100,lambda_align=.1,lambda_cross=model.config.lambda_cross,
   alignment_unit='focal P from different episodes; both sides have gradients; no alignment on T',source_pretrained_checkpoint=None)
  b['files'][str(Path(__file__).resolve())]=m.sha(__file__);b.pop('sha256',None)
  b['sha256']=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  return core,route,data,planner,model,b
 def objective(core,route,model,data,plan,ix,method,mode,micro):
  if mode=='focal':return core.batch_objective(model,data,plan,ix,method,micro)
  changed=dict(plan)
  if method=='Random-Both':
   changed['external']=plan['random_external'];rr=np.arange(len(plan['focal']));ff=plan['focal'].clip(0);changed['correct']=changed['external'][rr,ff]
  loss,metrics,detail=route.objective(core,model,data,changed,ix,'all',micro)
  rows=detail['rows'];alignment=detail['own'].sum()*0
  if len(rows):
   original=ix[rows.detach().cpu().numpy()];focal=torch.as_tensor(plan['focal'][original],device=rows.device)
   alignment,stats=core.relation_loss(detail['own'][rows,focal],detail['mixed'][torch.arange(len(rows),device=rows.device),focal]);metrics.update(stats)
  loss=loss+model.config.lambda_align*alignment;metrics.update(align=float(alignment.detach()),loss=float(loss.detach()))
  detail['terms']['align']=model.config.lambda_align*alignment
  return loss,metrics,detail
 def smoke(args):
  core,route,data,planner,model,b=m.checked(args);plan=planner.make(1,32);ix=plan['batches'][0];initial={k:v.detach().clone() for k,v in model.state_dict().items()};rows=[]
  for method in (args.method,'Random-Both'):
   model.load_state_dict(initial);core.seeded(0);model.zero_grad(set_to_none=True)
   with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.bfloat16,enabled=str(args.device).startswith('cuda')):
    loss,metrics,_=objective(core,route,model,data,plan,ix,method,args.route,32)
   if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   if float(norm)<=0:raise ValueError('Missing source gradient')
   rows.append(dict(method=method,loss=float(loss.detach()),self_loss=metrics['self_loss'],cross=metrics['cross'],align=metrics['align'],gradient_norm=float(norm)))
  if not np.isclose(rows[0]['self_loss'],rows[1]['self_loss'],atol=1e-6,rtol=1e-5):raise ValueError('Self path changed')
  m.write(Path(args.out)/'smoke.json',dict(status='PASS',binding_sha256=b['sha256'],rows=rows,optimizer_steps=0,weights_discarded=True,test_read=False))
 m.setup=setup;m.objective=objective;m.smoke=smoke
 Path(a.out).mkdir(parents=True,exist_ok=True)
 with open(Path(a.out)/'owner.lock','a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);{'prepare':m.prepare,'smoke':smoke,'train':m.train}[a.command](a)

if __name__=='__main__':main()
