#!/usr/bin/env python3
"""Fixed-budget CoDA component adaptation on the released Burgers protocol.

Five thousand updates, unchanged gate hyperparameters, all three seeds.
Seed1234 resumes the exact saved1000-update state. This is not an official
Burgers configuration: the CoDA release has none. No baseline parameter sweep.
"""
import argparse, fcntl, hashlib, json, os, random, signal, subprocess, sys, time, traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
PILOT_SHA='934fd3c75a15eb77d153685edbe94e7452f1b5cb72f5e2b6cb91a0e442c91d77'
COMPONENT_SHA='acb8bec540220913ed7559c1e16af6e6536fa63e0d66f5f1e0d4d26e6900db3b'
GATE_RUNNER_SHA='45c95cfb4f74f45aa90db99179b8c96021a9e817178c587ea796d5575c680695'
GEPS_FORMAL_SHA='82535b32b57b5779d12379e67a4d79b5267ed08b1b0db8bf5e9583ad01ce371a'
REUSE_SHA='5057809f7a9cd7725b23194782334d36c6ed1a4cd71a228d9cae5df32229637b'
GROUPS=['id_test','ood_viscous','ood_inviscid']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
 tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def json_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def alarm(*_):raise TimeoutError('Fixed CoDA phase deadline reached; preserve partial files.')

def configuration(a):
 import coda_burgers_pilot as gate
 assert sha(gate.__file__)==GATE_RUNNER_SHA
 assert sha(HERE/'coda_burgers_components.py')==COMPONENT_SHA
 assert sha(HERE/'geps_burgers_pilot.py')==PILOT_SHA
 assert sha(HERE/'geps_formal_three_seed.py')==GEPS_FORMAL_SHA
 assert {n:sha(gate.SOURCE/n) for n in gate.EXPECTED}==gate.EXPECTED
 old=read(HERE/'geps_pilot_retry1/CONFIG.json')
 import geps_formal_three_seed as g
 args=argparse.Namespace(seed=a.seed,nod_source=Path(old['nod_source']),data_root=Path(old['data_root']))
 assert sha(args.nod_source/'ngs/utils.py')==g.LOADER_SHA
 assert sha(args.nod_source/'train_nod_clean.py')==g.NATIVE_SHA
 assert sha(args.nod_source/'evaluate_frozen.py')==g.EVAL_SHA
 cfg=dict(read(HERE/'coda_burgers_gate/CONFIG.json'),seed=a.seed,updates=5000,
  runner_sha256=sha(__file__),training_budget='fixed5000; no validation/test checkpoint choice',
  resumed_checkpoint_sha256=REUSE_SHA if a.seed==1234 else None,
  adaptation_batch_size=8,native_report_batch_size=8,max_train_seconds=7200,max_evaluation_seconds=10800,
  protocol='Historical public ID/viscousOOD/inviscidOOD with frozen NOD pairing and native batch8 reduction',
  adaptation_notes=read(HERE/'coda_burgers_gate/CONFIG.json')['adaptation_notes'][:3]+[
   'Fixed5000-update budget; no claim of original120000-epoch convergence.',
   'Code-only50 Adam.001 support steps; adaptation time reported separately.',
   'Seed1234 resumes saved model/Adam/sampler/Python/NumPy/torch/CUDA state at1000.'],
  author_default_Burgers_exists=False,not_untouched_official_reproduction=True)
 assert cfg['lr']==.001 and cfg['factor']==1 and cfg['hidden_c']==64 and cfg['code_c']==2
 return args,cfg

def training(c,base,model,data,a,cfg,out):
 import numpy as np
 import torch
 bank=data['train'];assert tuple(bank['curves'].shape)==(360,1,401,101)
 envs=torch.unique(bank['envs']).tolist();assert envs==list(range(9))
 per_env=[torch.where(bank['envs']==e)[0] for e in envs];assert all(len(x)==40 for x in per_env)
 optimizer=torch.optim.Adam(model.parameters(),lr=.001)
 generator=torch.Generator().manual_seed(a.seed);orders=[torch.randperm(40,generator=generator) for _ in envs]
 cursor=0;start_step=0;epsilon=.99;losses=[]
 if a.seed==1234:
  checkpoint=HERE/'coda_burgers_gate/latest.pt';assert sha(checkpoint)==REUSE_SHA
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==1000 and ck['config']['seed']==1234
  model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
  generator.set_state(ck['sampler_generator']);orders=ck['sampler_orders'];cursor=ck['sampler_cursor'];start_step=1000
  np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
  for _ in range(start_step//30):epsilon*=.99
  assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in ck['model'].items())
  write(out/'RESUME_AUDIT.json',dict(status='PASS',source=str(checkpoint),source_sha256=sha(checkpoint),step=start_step,
   model_exact=True,optimizer_loaded=True,sampler_restored=True,all_recorded_rng_restored=True,epsilon=epsilon))
 times=bank['t'].cuda();start=time.monotonic();signal.setitimer(signal.ITIMER_REAL,7200)
 for step in range(start_step+1,5001):
  if cursor==40:orders=[torch.randperm(40,generator=generator) for _ in envs];cursor=0
  ix=torch.tensor([int(ids[order[cursor]]) for ids,order in zip(per_env,orders)]);cursor+=1
  truth=bank['curves'][ix,0].unsqueeze(0).cuda();optimizer.zero_grad(set_to_none=True)
  output=c.forecast(model,truth,times,epsilon);mse=(output-truth).square().mean();reg=c.regularizer(model);loss=mse+reg
  assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(mse.detach()))
  if step%30==0:epsilon*=.99
  if step==start_step+1 or step%10==0:
   row=dict(update=step,mse=losses[-1],regularizer=float(reg.detach()),epsilon=epsilon,seconds=time.monotonic()-start)
   with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if step%250==0:
   state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=step,config=cfg,
    sampler_generator=generator.get_state(),sampler_orders=orders,sampler_cursor=cursor,epsilon=epsilon,
    numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
   tmp=out/'latest.pt.writing';torch.save(state,tmp);tmp.replace(out/'latest.pt')
 signal.setitimer(signal.ITIMER_REAL,0)
 write(out/'TRAIN_COMPLETE.json',dict(status='COMPLETE',seed=a.seed,updates=5000,additional_updates=5000-start_step,
  seconds=time.monotonic()-start,first20_segment_mse=float(np.mean(losses[:20])),last20_mse=float(np.mean(losses[-20:])),
  checkpoint_sha256=sha(out/'latest.pt'),validation_or_ood_selection=False))
 return out/'latest.pt'

def eval_group(c,base,model,ds,denom,out,name):
 import numpy as np
 import torch
 import geps_formal_three_seed as g
 records=[];manifest=[];curves=[];code_rows=[];adapt_seconds=0.;predict_seconds=0.
 source=c.shared_digest(model)
 for start in range(0,len(ds),8):
  items=[ds[i] for i in range(start,min(start+8,len(ds)))]
  cond=torch.stack([x['cond_u'].transpose(-1,-2) for x in items]).cuda()
  target=torch.stack([x['target_seq'].permute(1,2,0) for x in items]).cuda()
  assert cond.shape==target.shape and tuple(cond.shape[1:])==(1,401,101)
  assert torch.isfinite(cond).all() and torch.isfinite(target).all()
  times=items[0]['t_idx'].to(device='cuda:0',dtype=torch.float32)/denom
  for j,x in enumerate(items):
   assert torch.equal(x['t_idx'],items[0]['t_idx']) and x['cond_case_idx']!=x['pred_case_idx']
   manifest.append(dict(index=start+j,nu_id=int(x['nu_id']),nu_value=float(x['nu_value'].item()),
    cond_case_idx=int(x['cond_case_idx']),pred_case_idx=int(x['pred_case_idx']),
    support_sha256=base.tensor_digest(cond[j]),query_sha256=base.tensor_digest(target[j]),t=times.cpu().tolist()))
  support=cond[:,0].unsqueeze(0);query=target[:,0].unsqueeze(0)
  adapted=c.adapted(model,len(items));opt=torch.optim.Adam([adapted.derivative.codes],lr=.001)
  assert c.shared_digest(adapted)==source
  torch.cuda.synchronize();begin=time.monotonic()
  with torch.no_grad():initial=float((c.forecast(adapted,support,times)-support).square().mean())
  for step in range(50):
   opt.zero_grad(set_to_none=True)
   loss=(c.forecast(adapted,support,times,epsilon=.95*(.95**(step//30)))-support).square().mean()
   assert torch.isfinite(loss);loss.backward();assert torch.isfinite(adapted.derivative.codes.grad).all();opt.step()
  with torch.no_grad():
   final=float((c.forecast(adapted,support,times)-support).square().mean())
   torch.cuda.synchronize();adapt_seconds+=time.monotonic()-begin
   assert c.shared_digest(adapted)==source and c.shared_digest(model)==source
   query_input=torch.zeros_like(query);query_input[...,0]=query[...,0]
   torch.cuda.synchronize();begin=time.monotonic();pred=c.forecast(adapted,query_input,times)
   torch.cuda.synchronize();predict_seconds+=time.monotonic()-begin
   assert torch.isfinite(pred).all() and torch.equal(pred[...,0],query[...,0])
   err=(pred-query).square();horizons=err[0].mean(1).cpu().numpy()
   for j,x in enumerate(items):
    r=dict(manifest[-len(items)+j],mse_all101=float(err[0,j].mean()),mse_future100=float(err[0,j,...,1:].mean()),
     support_initial_mse_batch=initial,support_final_mse_batch=final,support_steps=50)
    r.update({'h'+str(h):float(horizons[j,h]) for h in [1,5,50,100]})
    records.append(r);curves.append(horizons[j]);code_rows.append(adapted.derivative.codes[j].cpu().tolist())
  write(out/(name+'_PARTIAL.json'),dict(completed_pairs=len(records),total_pairs=len(ds),adapt_seconds=adapt_seconds,
    predict_seconds=predict_seconds,records=records,codes=code_rows))
 assert len(records)==len(ds) and source==c.shared_digest(model)
 errors=[x['mse_all101'] for x in records]
 result=dict(n_pairs=len(records),native_compatible_mse=g.native_reduce(errors,8),trajectory_mean_mse=float(np.mean(errors)),
  future100_mean_mse=float(np.mean([x['mse_future100'] for x in records])),native_reporting_batch_size=8,adaptation_batch_size=8,
  native_last_batch_size=(len(records)-1)%8+1,support_steps_per_query=50,support_frames=101,
  adapt_seconds=adapt_seconds,predict_seconds=predict_seconds,manifest_sha256=json_sha(manifest),
  horizon_mse={str(h):float(np.mean([r['h'+str(h)] for r in records])) for h in [1,5,50,100]})
 write(out/(name+'_PAIRS.json'),records);np.savez_compressed(out/(name+'_ERRORS.npz'),horizon_mse=np.asarray(curves),adapted_codes=np.asarray(code_rows))
 write(out/(name+'_COMPLETE.json'),dict(status='COMPLETE',result=result,pairs_sha256=sha(out/(name+'_PAIRS.json'))))
 return result,manifest,code_rows

def evaluation(c,base,model,args,out,checkpoint):
 import geps_formal_three_seed as g
 initial=base.parameter_digest(model);results={};manifests={};codes={};begin=time.monotonic()
 signal.setitimer(signal.ITIMER_REAL,10800)
 for name,ds,denom in g.groups(args):
  try:results[name],manifests[name],codes[name]=eval_group(c,base,model,ds,denom,out,name)
  finally:
   if hasattr(ds,'close'):ds.close()
 assert base.parameter_digest(model)==initial
 signal.setitimer(signal.ITIMER_REAL,0)
 write(out/'MANIFEST.json',manifests);write(out/'CODES.json',codes)
 result=dict(status='COMPLETE',seed=args.seed,updates=5000,groups=results,checkpoint_sha256=sha(checkpoint),
  source_parameter_digest_unchanged=initial,manifest_sha256=json_sha(manifests),seconds=time.monotonic()-begin,
  shared_predictor_frozen_during_support_adaptation=True,query_future_used_for_adaptation=False,
  query_future_passed_to_forecaster=False,author_default_Burgers_exists=False,
  scope='CoDA released components adapted to Burgers, fixed5000 updates; historical public evaluation, not untouched or sealed',
  primary_metric='Native batch8 mean of batch MSE means over101frames, including initial frame',
  grouping='Frozen NOD deterministic configuration. Historical raw NOD pair identities cannot all be retroactively verified.')
 write(out/'SUMMARY.json',result);return result

def main(a):
 locks=[]
 for directory in [HERE,HERE/'gpu_locks']:
  directory.mkdir(exist_ok=True);f=(directory/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
 sys.path.insert(0,str(HERE/'geps_deps'))
 import numpy as np,torch,coda_burgers_components as c,geps_burgers_pilot as base
 args,cfg=configuration(a);assert not a.output.exists();a.output.mkdir(parents=True)
 write(a.output/'CONFIG.json',cfg);write(a.output/'RUN.json',dict(pid=os.getpid(),gpu=a.gpu,time=time.time(),runner_sha256=sha(__file__)))
 signal.signal(signal.SIGALRM,alarm)
 try:
  torch.set_num_threads(1);write(a.output/'SMOKE.json',c.smoke())
  random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
  data,meta=base.load_released_data(args);write(a.output/'DATA_MANIFEST.json',meta)
  old=read(HERE/'coda_burgers_gate/DATA_MANIFEST.json')
  assert all(meta[k]['data_sha256']==old[k]['data_sha256'] for k in ['train','eval'])
  model=c.build(9,'cuda:0');checkpoint=training(c,base,model,data,a,cfg,a.output)
  model.eval();result=evaluation(c,base,model,args,a.output,checkpoint)
  configuration(a)
  write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json'),manifest_sha256=result['manifest_sha256'],checkpoint_sha256=sha(checkpoint)))
  write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
 except BaseException:
  write(a.output/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=[1234,5678,9012]);p.add_argument('--gpu',type=int,required=True,choices=[0,1,2,3]);p.add_argument('--output',type=Path,required=True)
 main(p.parse_args())
