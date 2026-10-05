#!/usr/bin/env python3
"""Fixed GEPS 5000-update adaptation, with native-compatible historical evaluation.

Train seeds5678/9012 once; seed1234 reuses its verified completed checkpoint.
No NOD training, source modification, test-based selection, or automatic resume.
"""
import argparse, copy, fcntl, hashlib, importlib.util, json, os
from pathlib import Path
import random, signal, subprocess, sys, time, traceback

HERE=Path(__file__).resolve().parent
PILOT_SHA='934fd3c75a15eb77d153685edbe94e7452f1b5cb72f5e2b6cb91a0e442c91d77'
LOADER_SHA='9a9d9f65e549feca695723583ba6b72b1c153a125c255d3eee8af9bf7745e58e'
NATIVE_SHA='466dd40071d8f0c88b7a488e2d14c8acf5db3c59e0e4db5bfab11ec073081417'
EVAL_SHA='709bdb88c5c1484d2b799fbc3f57c3ccf9d5bf6871841a6f95e194ecb84f774b'
REUSE_SHA='46914867b96452a210fda71454e14eb79eccabed7f39a0a433c6104a511cee25'
GROUPS=['id_test','ood_viscous','ood_inviscid']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
 tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest_json(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def native_reduce(values,batch_size=8):
 return sum(sum(values[i:i+batch_size])/len(values[i:i+batch_size]) for i in range(0,len(values),batch_size))/((len(values)+batch_size-1)//batch_size)
def future_free_input(target):
 import torch
 x=torch.zeros_like(target);x[...,0]=target[...,0];return x
def alarm(*_):raise TimeoutError('Finite GEPS phase deadline reached; preserve partial files.')

def configuration(a):
 old=read(HERE/'geps_pilot_retry1/CONFIG.json')
 assert sha(HERE/'geps_burgers_pilot.py')==PILOT_SHA
 cfg=dict(old,seed=a.seed,updates=5000,mode='fixed5000_formal_extension',output=str(a.output),script_sha256=sha(__file__),
  pilot_script_sha256=PILOT_SHA,selection='Fixed update5000; all three seeds; no final-data checkpoint selection',
  formal_result=True,test_read=True,ood_read=True,protocol='Historical public test extension with matched pairing and native batch8 reduction',
  max_seconds_per_phase=7200,train_query_observations_per_update=4,report_batch_size=8,
  source_checkpoint=str(a.checkpoint) if a.checkpoint else None,source_checkpoint_sha256=REUSE_SHA if a.checkpoint else None)
 for k,v in dict(lr=.01,batch_size=4,eval_batch_size=4,code_dim=4,adapt_lr=.01,adapt_steps=50).items():assert cfg[k]==v,(k,cfg[k])
 ns=argparse.Namespace(**cfg)
 for k in ['nod_source','geps_source','data_root']:setattr(ns,k,Path(getattr(ns,k)))
 ns.output=a.output;ns.device='cuda:0'
 assert sha(ns.nod_source/'ngs/utils.py')==LOADER_SHA
 assert sha(ns.nod_source/'train_nod_clean.py')==NATIVE_SHA
 assert sha(ns.nod_source/'evaluate_frozen.py')==EVAL_SHA
 assert {str(p.relative_to(ns.geps_source)):sha(p) for p in sorted(ns.geps_source.rglob('*.py'))}==old['geps_source_sha256']
 assert not a.output.resolve().is_relative_to(ns.geps_source.resolve())
 return ns,cfg

def training(base,model,data,args,out):
 import numpy as np
 import torch
 bank=data['train'];n=len(bank['curves']);assert n==360 and args.batch_size==4
 times=bank['t'].to(args.device);batch=args.batch_size
 smoke=base.smoke(model,bank['curves'][:4].to(args.device),bank['envs'][:4].to(args.device),times)
 write(out/'SMOKE.json',smoke)
 optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999))
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.9,patience=350,threshold=.01,min_lr=1e-5)
 generator=torch.Generator().manual_seed(args.seed);order=torch.randperm(n,generator=generator);cursor=0;losses=[]
 start=time.monotonic();signal.setitimer(signal.ITIMER_REAL,7200)
 for step in range(1,5001):
  if cursor+batch>n:order=torch.randperm(n,generator=generator);cursor=0
  ix=order[cursor:cursor+batch];cursor+=batch
  truth=bank['curves'][ix].to(args.device);env=bank['envs'][ix].to(args.device)
  optimizer.zero_grad(set_to_none=True);loss=base.relative_l2(model(truth,times,env,epsilon=0),truth)
  assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  optimizer.step();losses.append(float(loss.detach()))
  if step%90==0:scheduler.step(float(np.mean(losses[-90:])))
  assert all(group['lr']==.01 for group in optimizer.param_groups),'Unexpected LR transition inside fixed5000 budget'
  if step==1 or step%10==0:
   row=dict(update=step,relative_l2=losses[-1],elapsed_seconds=time.monotonic()-start,lr=.01)
   with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if step%1000==0:
   state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),updates=step,
    config=vars(args),sampler_generator=generator.get_state(),sampler_order=order,sampler_cursor=cursor,
    partial_epoch_losses=losses[-(step%90):] if step%90 else [],python_rng=random.getstate(),numpy_rng=np.random.get_state(),
    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
   tmp=out/'latest.pt.writing';torch.save(state,tmp);tmp.replace(out/'latest.pt')
 signal.setitimer(signal.ITIMER_REAL,0)
 result=dict(status='TRAINING_COMPLETE',updates=5000,seed=args.seed,seconds=time.monotonic()-start,
  relative_l2_first20=float(np.mean(losses[:20])),relative_l2_last20=float(np.mean(losses[-20:])),checkpoint_sha256=sha(out/'latest.pt'),
  final_checkpoint_selection='fixed5000',test_or_ood_used_in_training=False,lr_transitions=0)
 write(out/'TRAIN_COMPLETE.json',result);return out/'latest.pt',result

def groups(args):
 from ngs.utils import BurgersPairedDataset,_BurgersGroupedEval
 # Equivalent ID pairing path to the old evaluator. Only disable its eager unused-case cache.
 ds=BurgersPairedDataset(data_root=str(args.data_root),split='test',prediction_horizon=101,cache_mode='none',seed=0)
 denom=float(ds.n_t-1);assert denom==100 and len(ds)==45
 yield GROUPS[0],ds,denom
 for name,group in zip(GROUPS[1:],['ood','ood_inviscid']):
  yield name,_BurgersGroupedEval(group=group,eval_root=str(args.data_root/'truth'),prediction_horizon=101,deterministic_pairing=True,seed=0),denom

def evaluation(base,model,args,out,checkpoint):
 import numpy as np
 import torch
 from torch import nn
 start=time.monotonic();signal.setitimer(signal.ITIMER_REAL,7200)
 initial=model.derivative.codes.detach().mean(0).clone();source=base.parameter_digest(model);shared=base.parameter_digest(model,exclude_codes=True)
 results={};manifests={};allcodes={}
 for name,ds,denom in groups(args):
  records=[];manifest=[];per_time=[];codes_log=[];adapt_seconds=0.;predict_seconds=0.;time_vector=None
  try:
   for start_idx in range(0,len(ds),4):
    items=[ds[i] for i in range(start_idx,min(start_idx+4,len(ds)))]
    support=torch.stack([x['cond_u'].transpose(-1,-2) for x in items]).to(args.device)
    target=torch.stack([x['target_seq'].permute(1,2,0) for x in items]).to(args.device)
    assert support.shape==target.shape and tuple(support.shape[1:])==(1,401,101)
    assert torch.isfinite(support).all() and torch.isfinite(target).all()
    times=items[0]['t_idx'].to(device=args.device,dtype=torch.float32)/denom
    for item in items:assert torch.equal(item['t_idx'],items[0]['t_idx']) and item['cond_case_idx']!=item['pred_case_idx']
    if time_vector is None:time_vector=times.detach().cpu().tolist()
    else:assert time_vector==times.detach().cpu().tolist()
    for k,item in enumerate(items):
     manifest.append(dict(index=start_idx+k,nu_id=int(item['nu_id']),nu_value=float(item['nu_value'].item()),
      cond_case_idx=int(item['cond_case_idx']),pred_case_idx=int(item['pred_case_idx']),
      support_sha256=base.tensor_digest(support[k]),query_sha256=base.tensor_digest(target[k]),t= time_vector))
    adapted=copy.deepcopy(model);codes=nn.Parameter(initial[None].repeat(len(items),1))
    adapted.derivative.codes=codes;adapted.derivative.model_aug.codes=codes
    for p in adapted.parameters():p.requires_grad_(False)
    codes.requires_grad_(True);opt=torch.optim.Adam([codes],lr=.01,betas=(.9,.999));env=torch.arange(len(items),device=args.device)
    torch.cuda.synchronize();begin=time.monotonic()
    with torch.no_grad():before=float((adapted(support,times,env,epsilon=0)-support).square().mean())
    for _ in range(50):
     opt.zero_grad(set_to_none=True);pred=adapted(support,times,env,epsilon=0);loss=(pred-support).square().mean()
     assert torch.isfinite(loss);loss.backward();assert codes.grad is not None and torch.isfinite(codes.grad).all();opt.step()
    torch.cuda.synchronize();adapt_seconds+=time.monotonic()-begin
    assert base.parameter_digest(adapted,exclude_codes=True)==shared
    with torch.no_grad():
     after=float((adapted(support,times,env,epsilon=0)-support).square().mean())
     torch.cuda.synchronize();begin=time.monotonic()
     # Future query values never enter predictor input or optimizer.
     pred=adapted(future_free_input(target),times,env,epsilon=0)
     torch.cuda.synchronize();predict_seconds+=time.monotonic()-begin
     assert torch.isfinite(pred).all() and torch.isfinite(codes).all()
     assert torch.equal(pred[...,0],target[...,0]),'GEPS initial state differs from supplied query initial condition'
     errors=(pred-target).square();horizon=errors.mean((1,2)).cpu().numpy()
     for j,item in enumerate(items):
      record=dict(manifest[-len(items)+j],mse_all101=float(errors[j].mean()),mse_future100=float(errors[j,...,1:].mean()),
       h1=float(horizon[j,1]),h5=float(horizon[j,5]),h50=float(horizon[j,50]),h100=float(horizon[j,100]),
       support_initial_mse_batch=before,support_final_mse_batch=after,support_steps=50)
      records.append(record);per_time.append(horizon[j]);codes_log.append(codes[j].cpu().tolist())
    del adapted,opt
   assert len(records)==len(ds) and len(records)>0
   a=[r['mse_all101'] for r in records];f=[r['mse_future100'] for r in records]
   results[name]=dict(n_pairs=len(records),native_compatible_mse=native_reduce(a,8),trajectory_mean_mse=float(np.mean(a)),
    future100_mean_mse=float(np.mean(f)),native_reporting_batch_size=8,adaptation_batch_size=4,
    native_last_batch_size=(len(records)-1)%8+1,horizon_mse={str(h):float(np.mean([r['h'+str(h)] for r in records])) for h in [1,5,50,100]},
    support_steps_per_query=50,support_frames=101,adapt_seconds=adapt_seconds,predict_seconds=predict_seconds,
    manifest_sha256=digest_json(manifest),dataset_description=ds.describe() if hasattr(ds,'describe') else dict(group=ds.group,shards=len(ds.shards)))
   write(out/(name+'_PAIRS.json'),records);np.savez_compressed(out/(name+'_ERRORS.npz'),horizon_mse=np.asarray(per_time),adapted_codes=np.asarray(codes_log))
   manifests[name]=manifest;allcodes[name]=dict(initial_code=initial.cpu().tolist(),adapted_codes=codes_log)
   write(out/(name+'_COMPLETE.json'),dict(status='COMPLETE',pairs_sha256=sha(out/(name+'_PAIRS.json')),result=results[name]));print(json.dumps(dict(group=name,result=results[name])),flush=True)
  finally:
   if hasattr(ds,'close'):ds.close()
 assert base.parameter_digest(model)==source
 signal.setitimer(signal.ITIMER_REAL,0);write(out/'MANIFEST.json',manifests);write(out/'CODES.json',allcodes)
 result=dict(status='COMPLETE',seed=args.seed,groups=results,checkpoint_sha256=sha(checkpoint),updates=5000,
  source_parameter_sha256_unchanged=source,shared_parameter_sha256_unchanged=shared,manifest_sha256=digest_json(manifests),
  seconds=time.monotonic()-start,query_future_used_for_adaptation=False,query_future_passed_to_forecaster=False,
  grouping='exact frozen NOD deterministic configuration; prior raw pair export unavailable for retroactive bitwise pairing audit',
  forecast_type='GEPS Euler recurrent integration',scope='Fixed-budget PyTorch GEPS adaptation on historical public test; not untouched or new sealed test',
  primary_metric='Mean of consecutive native batch8 MSE means, including initial frame; matches existing native table reduction')
 write(out/'SUMMARY.json',result);return result

def main(a):
 assert (a.seed==1234)==(a.checkpoint is not None)
 if a.checkpoint:assert sha(a.checkpoint)==REUSE_SHA
 assert not a.output.exists(),'Preserve prior outputs; no implicit resume'
 locks=[]
 for directory in [HERE,HERE/'gpu_locks']:
  directory.mkdir(exist_ok=True);lock=(directory/f'gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 apps=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 assert not apps.strip(),f'GPU busy: {apps}'
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1'
 sys.path.insert(0,str(HERE/'geps_deps'));sys.dont_write_bytecode=True
 import torch,numpy as np
 torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 args,cfg=configuration(a)
 spec=importlib.util.spec_from_file_location('frozen_geps_adapter',HERE/'geps_burgers_pilot.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 a.output.mkdir(parents=True);write(a.output/'CONFIG.json',cfg);write(a.output/'RUN.json',dict(pid=os.getpid(),gpu=a.gpu,time=time.time(),script_sha256=sha(__file__)))
 signal.signal(signal.SIGALRM,alarm)
 try:
  data,meta=base.load_released_data(args);write(a.output/'DATA.json',meta)
  olddata=read(HERE/'geps_pilot_retry1/DATA.json')
  for split in ['train','eval']:assert meta[split]['data_sha256']==olddata[split]['data_sha256']
  model,repaired=base.build_model(args,9,torch.device('cuda:0'))
  write(a.output/'INITIALIZATION.json',dict(seed=a.seed,model_sha256=base.parameter_digest(model),zeroed_released_empty_parameters=repaired))
  if a.checkpoint:
   ck=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False);assert ck['updates']==5000 and ck['config']['seed']==1234
   model.load_state_dict(ck['model'],strict=True);checkpoint=a.checkpoint
   write(a.output/'TRAIN_COMPLETE.json',dict(status='REUSED_COMPLETE',updates=5000,additional_training_updates=0,checkpoint_sha256=sha(checkpoint),source=str(checkpoint)))
  else:checkpoint,_=training(base,model,data,args,a.output)
  model.eval();result=evaluation(base,model,args,a.output,checkpoint)
  assert {str(p.relative_to(args.geps_source)):sha(p) for p in sorted(args.geps_source.rglob('*.py'))}==cfg['geps_source_sha256']
  write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json'),checkpoint_sha256=sha(checkpoint),manifest_sha256=result['manifest_sha256']))
  write(a.output/'EXIT.json',dict(exit_code=0,pid=os.getpid(),time=time.time()))
 except BaseException:
  write(a.output/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));write(a.output/'EXIT.json',dict(exit_code=1,pid=os.getpid(),time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1234,5678,9012],required=True);p.add_argument('--gpu',type=int,choices=[0,1,2,3],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);main(p.parse_args())
