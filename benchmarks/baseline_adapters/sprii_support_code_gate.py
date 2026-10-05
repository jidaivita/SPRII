#!/usr/bin/env python3
"""Two development-only SPRII support-code refinements; all model weights frozen.

Same101-frame independent history, no query-future optimization, no NOD fits.
This adds explicit inference optimization and is not the native one-pass row.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'sprii_support_code_gate'
SOURCE=Path('benchmarks/nod/runs/adapted_tuned/seed1234_refine/latest.pth')
SOURCE_SHA='73f78af412b471ff895c0baee1bf6ad4ddba3e6714c7fd251590fbd03ceb8c37'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def alarm(*_):raise TimeoutError('SPRII development support-refinement exceeded fixed30-minute budget')
def cached_predict(model,u0,z,trunk):
 import torch
 pred=model.predictioner;branch=pred.backbone.branch_net(torch.cat([u0[:,0],z],dim=1))
 delta=branch@trunk.T+pred.backbone.output_bias
 # All401 spatial points at each of101 times, in time-major order.
 return u0[:,0].repeat(1,101)+pred.res_scale*delta
def check_cache(model,u0,z,coords,trunk):
 import torch
 # Numerical parity against the actual frozen full predictor, including dz.
 ix=torch.linspace(0,len(coords)-1,257,device=coords.device).long()
 a=z.clone().detach().requires_grad_();b=z.clone().detach().requires_grad_()
 native=model.predict_queries(u0,coords[ix][None].expand(len(u0),-1,-1),a)
 pred=model.predictioner;branch=pred.backbone.branch_net(torch.cat([u0[:,0],b],dim=1))
 xix=torch.round((coords[ix,0]+1)*.5*400).long().clamp(0,400)
 cached=u0[:,0,xix]+pred.res_scale*(branch@trunk[ix].T+pred.backbone.output_bias)
 torch.testing.assert_close(cached,native,rtol=2e-5,atol=2e-6)
 ga=torch.autograd.grad(native.square().mean(),a)[0];gb=torch.autograd.grad(cached.square().mean(),b)[0]
 torch.testing.assert_close(gb,ga,rtol=2e-4,atol=1e-8)
 return dict(status='PASS',points=257,max_prediction_difference=float((native-cached).abs().max()),max_dz_difference=float((ga-gb).abs().max()))
def run():
 assert not OUT.exists();OUT.mkdir(parents=True)
 write(OUT/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__)))
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,1800)
 try:
  import numpy as np,torch
  import geps_formal_three_seed as g,geps_burgers_pilot as data_api
  assert sha(data_api.__file__)==g.PILOT_SHA
  old=read(HERE/'geps_pilot_retry1/CONFIG.json');nod=Path(old['nod_source']);assert sha(nod/'train_nod_clean.py')==g.NATIVE_SHA
  assert sha(SOURCE)==SOURCE_SHA;sys.path.insert(0,str(nod))
  from train_nod_clean import MODEL_DEFAULTS,NGS_INR
  from ngs.utils import BurgersPairedDataset
  torch.set_num_threads(1);torch.manual_seed(1234);np.random.seed(1234)
  ck=torch.load(SOURCE,map_location='cpu',weights_only=False);cfg=ck['config']
  kwargs=dict(cfg.get('model_defaults',MODEL_DEFAULTS));kwargs['context_dim']=int(cfg.get('context_dim_override',cfg.get('context_dim',1)))
  model=NGS_INR(**kwargs).cuda().eval();model.load_state_dict(ck['model_state_dict'],strict=True)
  assert model.context_dim==1
  for p in model.parameters():p.requires_grad_(False)
  before=data_api.parameter_digest(model)
  args=argparse.Namespace(nod_source=nod,data_root=Path(old['data_root']),seed=1234)
  data,meta=data_api.load_released_data(args);write(OUT/'DATA.json',meta)
  # Training-history statistics only fix the units of the optimization variable.
  train_codes=[]
  with torch.no_grad():
   for start in range(0,360,2):train_codes.append(model.encode(data['train']['curves'][start:start+2].transpose(-1,-2).cuda()).cpu())
  train_codes=torch.cat(train_codes);mean=train_codes.mean(0);std=train_codes.std(0,unbiased=False).clamp_min(1e-8).cuda()
  assert torch.isfinite(train_codes).all() and float(std)>1e-7
  write(OUT/'TRAIN_LATENT_STATS.json',dict(n=360,mean=mean.tolist(),std=std.cpu().tolist(),checkpoint_sha256=SOURCE_SHA,train_only=True))
  times=data['train']['t'].cuda();x=torch.linspace(-1,1,401,device='cuda');tt,xx=torch.meshgrid(times,x,indexing='ij');coords=torch.stack([xx.flatten(),tt.flatten()],dim=-1)
  with torch.no_grad():trunk=torch.cat([model.predictioner.backbone.trunk_net(model.predictioner.pe(coords[i:i+1024])) for i in range(0,len(coords),1024)])
  ds=BurgersPairedDataset(data_root=str(args.data_root),split='eval',prediction_horizon=101,cache_mode='none',seed=0)
  assert len(ds)==45
  rows={str(lr):[] for lr in [0.,.01,.1]};manifest=[];codes={str(lr):[] for lr in [0.,.01,.1]};costs={str(lr):0. for lr in [.01,.1]};beg=time.monotonic()
  write(OUT/'CONFIG.json',dict(seed=1234,checkpoint=str(SOURCE),checkpoint_sha256=SOURCE_SHA,runner_sha256=sha(__file__),
   candidates=[dict(steps=50,normalized_code_lr=v) for v in [.01,.1]],source_optimizer_updates=0,
   parameters_frozen=True,normalizer_frozen=True,shared_predictor_frozen=True,additional_observation_budget=0,
   support_frames=101,support='Original independent conditioning history; fit its own initial-state-to-trajectory prediction',
   selection='Lowest full45-pair native-compatible development query MSE, including unadapted reference; no test/OOD access',
   not_native_one_pass=True,test_read=False,ood_read=False,extra_inference_gradient_steps=50))
  try:
   for start in range(0,45,8):
    items=[ds[i] for i in range(start,min(start+8,45))]
    assert all(40<=int(a['pred_case_idx'])<=44 and 40<=int(a['cond_case_idx'])<=44 and a['pred_case_idx']!=a['cond_case_idx'] for a in items)
    cond=torch.stack([a['cond_u'] for a in items]).cuda();target=torch.stack([a['target_seq'][:,0,:] for a in items]).cuda()
    assert tuple(cond.shape[1:])==(1,101,401) and tuple(target.shape[1:])==(101,401)
    support=cond[:,0].reshape(len(items),-1);su0=cond[:,:,0];qu0=target[:,0,None]
    with torch.no_grad():z0=model.encode(cond)
    if start==0:write(OUT/'CACHE_PARITY.json',check_cache(model,su0,z0,coords,trunk))
    for j,a in enumerate(items):manifest.append(dict(index=start+j,nu_id=int(a['nu_id']),nu_value=float(a['nu_value'].item()),
     cond_case_idx=int(a['cond_case_idx']),pred_case_idx=int(a['pred_case_idx']),support_sha256=data_api.tensor_digest(cond[j]),query_sha256=data_api.tensor_digest(target[j])))
    for lr in [0.,.01,.1]:
     eta=torch.zeros_like(z0,requires_grad=True)
     with torch.no_grad():support_before=((cached_predict(model,su0,z0,trunk)-support)**2).mean(1)
     if lr:
      optimizer=torch.optim.Adam([eta],lr=lr);torch.cuda.synchronize();t0=time.monotonic()
      for step in range(50):
       optimizer.zero_grad(set_to_none=True);pred=cached_predict(model,su0,z0+std*eta,trunk)
       loss=((pred-support)**2).mean(1).sum();assert torch.isfinite(loss);loss.backward()
       assert eta.grad is not None and torch.isfinite(eta.grad).all();optimizer.step()
      torch.cuda.synchronize();costs[str(lr)]+=time.monotonic()-t0
     with torch.no_grad():
      z=z0+std*eta;support_after=((cached_predict(model,su0,z,trunk)-support)**2).mean(1)
      prediction=cached_predict(model,qu0,z,trunk).reshape(len(items),101,401)
      assert torch.isfinite(prediction).all() and torch.isfinite(z).all()
      error=(prediction-target).square();mse=error.mean((1,2));future=error[:,1:].mean((1,2))
      for j,a in enumerate(items):
       rows[str(lr)].append(dict(manifest[-len(items)+j],mse_all101=float(mse[j]),mse_future100=float(future[j]),
        support_before=float(support_before[j]),support_after=float(support_after[j]),eta=float(eta[j]),initial_code=float(z0[j]),refined_code=float(z[j])))
       codes[str(lr)].append(float(z[j]))
    write(OUT/'PARTIAL.json',dict(completed_pairs=start+len(items),total_pairs=45,rows=rows))
  finally:ds.close()
  assert data_api.parameter_digest(model)==before and all(p.grad is None for p in model.parameters()) and sha(SOURCE)==SOURCE_SHA
  result={lr:dict(native_compatible_mse=g.native_reduce([r['mse_all101'] for r in rec],8),
   trajectory_mean_mse=float(np.mean([r['mse_all101'] for r in rec])),future100_mse=float(np.mean([r['mse_future100'] for r in rec])),
   support_mse_before=float(np.mean([r['support_before'] for r in rec])),support_mse_after=float(np.mean([r['support_after'] for r in rec])),
   code_std=float(np.std(codes[lr])),max_abs_normalized_change=max(abs(r['eta']) for r in rec),adapt_seconds=costs.get(lr,0.)) for lr,rec in rows.items()}
  write(OUT/'PAIRS.json',rows);write(OUT/'MANIFEST.json',manifest)
  write(OUT/'SUMMARY.json',dict(status='COMPLETE',results=result,selected_lr=min(result,key=lambda k:result[k]['native_compatible_mse']),
   checkpoint_sha256=SOURCE_SHA,model_digest_unchanged=before,source_optimizer_updates=0,query_future_optimization=False,
   test_read=False,ood_read=False,scope='Development-only inference-code adaptation; not native one-pass or new model training',seconds=time.monotonic()-beg))
  write(OUT/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(OUT/'SUMMARY.json'),pairs_sha256=sha(OUT/'PAIRS.json')))
  write(OUT/'EXIT.json',dict(exit_code=0,time=time.time()))
 except BaseException:
  write(OUT/'FAILED.json',dict(traceback=traceback.format_exc()));write(OUT/'EXIT.json',dict(exit_code=1,time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 locks=[]
 for d in [HERE,HERE/'gpu_locks']:
  d.mkdir(exist_ok=True);f=(d/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 run()
