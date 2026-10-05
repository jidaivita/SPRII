"""Fixed development-selected code refinement: all seeds/groups and bound mechanisms.

No network training. One-pass and refined predictions share each exact query.
"""
import argparse, fcntl, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import sprii_support_code_gate as cache
sha,read,write=cache.sha,cache.read,cache.write
SOURCES={1234:('benchmarks/nod/runs/adapted_tuned/seed1234_refine/latest.pth','73f78af412b471ff895c0baee1bf6ad4ddba3e6714c7fd251590fbd03ceb8c37'),
 5678:('benchmarks/nod/runs/opt_refine/lr5e5/latest.pth','8553c8e142cd32a0141a9238078435cf66e7852b602f43d2942fd293ac5bac64'),
 9012:('benchmarks/nod/runs/adapted_tuned/seed9012_refine/latest.pth','392baa89edec2d9761d1ddd905cf5e6819b89aa68c1b375a6217337d182afb37')}
def refine(model,cond,std,trunk):
 import torch
 with torch.no_grad():z0=model.encode(cond)
 target=cond[:,0].flatten(1);u0=cond[:,:,0];eta=torch.zeros_like(z0,requires_grad=True)
 opt=torch.optim.Adam([eta],lr=.1);torch.cuda.synchronize();begin=time.monotonic()
 for _ in range(50):
  opt.zero_grad(set_to_none=True);loss=(cache.cached_predict(model,u0,z0+std*eta,trunk)-target).square().mean(1).sum()
  assert torch.isfinite(loss);loss.backward();assert torch.isfinite(eta.grad).all();opt.step()
 torch.cuda.synchronize();elapsed=time.monotonic()-begin
 with torch.no_grad():
  z=z0+std*eta
  initial=(cache.cached_predict(model,u0,z0,trunk)-target).square().mean(1)
  final=(cache.cached_predict(model,u0,z,trunk)-target).square().mean(1)
 return z0,z.detach(),initial,final,elapsed
def standardized_geometry(codes,groups,mu,sd):
 import numpy as np
 z=(np.asarray(codes)-mu)/sd;labels=np.asarray(groups);unique=np.unique(labels)
 centers=np.stack([z[labels==u].mean(0) for u in unique]);within=float(np.mean([np.mean(np.sum((z[labels==u]-c)**2,axis=1)) for u,c in zip(unique,centers)]))
 between=float(np.mean(np.sum((centers-centers.mean(0))**2,axis=1)))
 return dict(within=within,between=between,between_within_ratio=between/max(within,1e-12),groups=len(unique),training_statistics=True)
def ranks(x):
 import numpy as np
 _,iv,n=np.unique(x,return_inverse=True,return_counts=True);return (np.cumsum(n)-.5*(n+1))[iv]
def probe(train_codes,train_y,cases,test_codes,test_y):
 import numpy as np
 x=np.asarray(train_codes,dtype=float);y=np.asarray(train_y,dtype=float);c=np.asarray(cases)
 def fit(x,y,alpha):
  mu=x.mean(0);sd=x.std(0).clip(1e-8);z=(x-mu)/sd;ym=y.mean();coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym));return mu,sd,coef,ym
 def predict(f,x):mu,sd,coef,ym=f;return ((x-mu)/sd)@coef+ym
 cv={}
 for alpha in [.0001,.001,.01,.1,1.,10.,100.]:
  errors=[]
  for k in range(5):
   tr=c%5!=k;va=~tr;pred=predict(fit(x[tr],y[tr],alpha),x[va]);errors.extend((pred-y[va])**2)
  cv[str(alpha)]=float(np.mean(errors))
 alpha=float(min(cv,key=cv.get));f=fit(x,y,alpha);test=np.asarray(test_y,dtype=float);pred=predict(f,np.asarray(test_codes))
 denominator=float(np.sum((test-test.mean())**2));assert denominator>0
 a,b=ranks(pred),ranks(test);rho=float(np.corrcoef(a,b)[0,1]) if a.std()>0 else None
 return dict(r2=1-float(np.sum((pred-test)**2))/denominator,spearman=rho,alpha=alpha,training_case5fold_cv=cv,n_train=len(x),n_id_test=len(test),
  predicted_nu=pred.tolist(),training_latent_mean=f[0].tolist(),training_latent_std=f[1].tolist(),selection='Training-case grouped folds only; nu, not log-nu')
def donor_panel(model,targets,zs,manifest,trunk,out):
 import numpy as np,torch
 panels={};labels=np.asarray([r['nu_id'] for r in manifest]);systems=np.unique(labels);assert len(systems)==9
 for mode,codes in zs.items():
  rows=[];codes=torch.as_tensor(codes,device='cuda',dtype=targets.dtype)
  with torch.no_grad():
   for i,rec in enumerate(manifest):
    for env in systems:
     # A matched independent donor cannot be either the recipient trajectory
     # or the already-used conditioning trajectory. Each cell averages only
     # legal single-history substitutions; it never averages input codes.
     js=[j for j,d in enumerate(manifest) if d['nu_id']==env and (env!=rec['nu_id'] or d['cond_case_idx'] not in [rec['pred_case_idx'],rec['cond_case_idx']])]
     assert js
     u0=targets[i:i+1,0,None].expand(len(js),-1,-1)
     prediction=cache.cached_predict(model,u0,codes[js],trunk).reshape(len(js),101,401)
     e=(prediction-targets[i:i+1]).square().mean(2);assert torch.isfinite(e).all()
     rows.append(dict(recipient=i,target_nu_id=rec['nu_id'],donor_nu_id=int(env),donor_indices=js,
      trajectory_mse=float(e.mean()),h1=float(e[:,1].mean()),h5=float(e[:,5].mean()),h50=float(e[:,50].mean()),h100=float(e[:,100].mean())))
  per_target=[]
  for i in range(45):
   r=[v for v in rows if v['recipient']==i];matched=next(v for v in r if v['target_nu_id']==v['donor_nu_id']);wrong=[v for v in r if v['target_nu_id']!=v['donor_nu_id']]
   per_target.append(dict(recipient=i,nu_id=int(labels[i]),**{h+'_wrong_minus_matched':float(np.mean([v[h] for v in wrong]))-matched[h] for h in ['trajectory_mse','h1','h5','h50','h100']}))
  sysdelta=np.asarray([np.mean([v['h50_wrong_minus_matched'] for v in per_target if v['nu_id']==s]) for s in systems])
  rng=np.random.default_rng(834);ci=np.quantile(np.mean(rng.choice(sysdelta,(10000,9),replace=True),1),[.025,.975])
  panels[mode]=dict(rows=rows,per_recipient=per_target,h50_wrong_minus_matched=float(sysdelta.mean()),system_bootstrap_ci95=ci.tolist(),
   matched_h50=float(np.mean([r['h50'] for r in rows if r['target_nu_id']==r['donor_nu_id']])),
   scope='Frozen predictor with already-extracted codes. Nine target systems; one training seed. H50 is direct output, not autoregressive rollout.')
 write(out/'DONOR_PANEL.json',panels);return {k:{n:v for n,v in p.items() if n not in ['rows','per_recipient']} for k,p in panels.items()}
def run(a):
 import numpy as np,torch
 import geps_formal_three_seed as g,geps_burgers_pilot as base
 assert sha(cache.__file__)=='5173857d5c54ab28872be75ea6a68fee3531ec995a614b0bc0bf9f8ce8612c9e'
 gate=read(HERE/'sprii_support_code_gate/SUMMARY.json');assert gate['selected_lr']=='0.1' and gate['results']['0.1']['native_compatible_mse']<gate['results']['0.0']['native_compatible_mse']
 assert read(HERE/'sprii_support_code_queue/EXIT.json')['exit_code']==0
 assert not a.output.exists();a.output.mkdir(parents=True)
 write(a.output/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__)))
 def timeout(*_):raise TimeoutError('Fixed60-minute support-code formal budget expired')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,3600)
 try:
  torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
  old=read(HERE/'geps_pilot_retry1/CONFIG.json');args=argparse.Namespace(nod_source=Path(old['nod_source']),data_root=Path(old['data_root']),seed=a.seed)
  assert sha(args.nod_source/'train_nod_clean.py')==g.NATIVE_SHA and sha(args.nod_source/'ngs/utils.py')==g.LOADER_SHA
  data,meta=base.load_released_data(args);write(a.output/'DATA.json',meta)
  from train_nod_clean import MODEL_DEFAULTS,NGS_INR
  from ngs.utils import BurgersPairedDataset
  source,source_sha=SOURCES[a.seed];assert sha(source)==source_sha
  ck=torch.load(source,map_location='cpu',weights_only=False);cfg=ck['config'];kwargs=dict(cfg.get('model_defaults',MODEL_DEFAULTS));kwargs['context_dim']=int(cfg.get('context_dim_override',cfg.get('context_dim',1)))
  model=NGS_INR(**kwargs).cuda().eval();model.load_state_dict(ck['model_state_dict'],strict=True);assert model.context_dim==1
  for p in model.parameters():p.requires_grad_(False)
  digest=base.parameter_digest(model)
  original=[]
  with torch.no_grad():
   for start in range(0,360,8):original.append(model.encode(data['train']['curves'][start:start+8].transpose(-1,-2).cuda()).cpu())
  original=torch.cat(original);std=original.std(0,unbiased=False).clamp_min(1e-8).cuda()
  assert np.array_equal(np.asarray(meta['train']['temporal_indices'],dtype=np.float32)/meta['train']['time_normalization'],data['train']['t'].numpy())
  # Preserve the released evaluator's exact CUDA float32 normalization order.
  times=torch.tensor(meta['train']['temporal_indices'],device='cuda',dtype=torch.float32)/float(meta['train']['time_normalization']);x=torch.linspace(-1,1,401,device='cuda');tt,xx=torch.meshgrid(times,x,indexing='ij');coords=torch.stack([xx.flatten(),tt.flatten()],-1)
  with torch.no_grad():trunk=torch.cat([model.predictioner.backbone.trunk_net(model.predictioner.pe(coords[i:i+1024])) for i in range(0,len(coords),1024)])
  cond=data['train']['curves'][:8].transpose(-1,-2).cuda()
  write(a.output/'CACHE_PARITY.json',cache.check_cache(model,cond[:,:,0],original[:8].cuda(),coords,trunk))
  train_refined=[]
  for start in range(0,360,8):train_refined.append(refine(model,data['train']['curves'][start:start+8].transpose(-1,-2).cuda(),std,trunk)[1].cpu())
  train_refined=torch.cat(train_refined);traincodes={'one_pass':original.numpy(),'support50':train_refined.numpy()}
  train_ds=BurgersPairedDataset(data_root=str(args.data_root),split='train',cache_mode='none',seed=0)
  try:nu_map={int(s['nu_id']):float(s['nu']) for s in train_ds.shards}
  finally:train_ds.close()
  trainenv=data['train']['envs'].numpy();train_y=np.asarray([nu_map[int(i)] for i in trainenv]);cases=data['train']['cases'].numpy()
  write(a.output/'CONFIG.json',dict(seed=a.seed,checkpoint=source,checkpoint_sha256=source_sha,gate_summary_sha256=sha(HERE/'sprii_support_code_gate/SUMMARY.json'),
   normalized_code_lr=.1,support_steps=50,source_optimizer_updates=0,extra_observations=0,support_frames=101,inference_only=True,
   code_optimization_units='Original encoder training-latent population standard deviation',selection='Frozen complete recipe from seed1234 development45 pairs; all3seeds/groups',
   probe_fit='All360 training histories independently code-refined; only train labels used. Ridge training-case5fold.',historical_public_extension=True,new_sealed=False))
  groups={};manifests={};iddata=None
  for name,ds,denom in g.groups(args):
   rows={k:[] for k in traincodes};codes={k:[] for k in traincodes};manifest=[];targets=[];cost=0.
   try:
    for start in range(0,len(ds),8):
     items=[ds[i] for i in range(start,min(start+8,len(ds)))];cond=torch.stack([p['cond_u'] for p in items]).cuda();target=torch.stack([p['target_seq'][:,0] for p in items]).cuda()
     for p in items:assert torch.equal(p['t_idx'].cuda().float()/denom,times) and p['cond_case_idx']!=p['pred_case_idx']
     z0,z,before,after,sec=refine(model,cond,std,trunk);cost+=sec
     for j,p in enumerate(items):manifest.append(dict(index=start+j,nu_id=int(p['nu_id']),nu_value=float(p['nu_value'].item()),cond_case_idx=int(p['cond_case_idx']),pred_case_idx=int(p['pred_case_idx']),
      support_sha256=base.tensor_digest(cond[j].transpose(-1,-2)),query_sha256=base.tensor_digest(target[j].T[None]),t=times.cpu().tolist()))
     with torch.no_grad():
      for mode,v in [('one_pass',z0),('support50',z)]:
       pred=cache.cached_predict(model,target[:,0,None],v,trunk).reshape(len(items),101,401);e=(pred-target).square().mean(2);assert torch.isfinite(e).all()
       for j,p in enumerate(items):rows[mode].append(dict(index=start+j,mse_all101=float(e[j].mean()),mse_future100=float(e[j,1:].mean()),h1=float(e[j,1]),h5=float(e[j,5]),h50=float(e[j,50]),h100=float(e[j,100]),support_before=float(before[j]),support_after=float(after[j])))
       codes[mode].extend(v.cpu().tolist())
     if name=='id_test':targets.append(target.cpu())
     write(a.output/(name+'_PARTIAL.json'),dict(completed_pairs=start+len(items),total_pairs=len(ds),adapt_seconds=cost))
   finally:
    if hasattr(ds,'close'):ds.close()
   formal=read(HERE/f'geps_formal_three_seed/seed{a.seed}/MANIFEST.json')[name];assert g.digest_json(manifest)==g.digest_json(formal)
   groups[name]={mode:dict(native_compatible_mse=g.native_reduce([r['mse_all101'] for r in rec],8),trajectory_mean_mse=float(np.mean([r['mse_all101'] for r in rec])),future100_mean_mse=float(np.mean([r['mse_future100'] for r in rec])),n_pairs=len(rec),extra_adapt_seconds=cost if mode=='support50' else 0.,horizon_mse={str(h):float(np.mean([r['h'+str(h)] for r in rec])) for h in [1,5,50,100]}) for mode,rec in rows.items()}
   write(a.output/(name+'_PAIRS.json'),dict(manifest=manifest,rows=rows,codes=codes));manifests[name]=manifest
   if name=='id_test':iddata=(codes,manifest,torch.cat(targets))
  codes,manifest,targets=iddata;probes={};geometry={}
  for mode in traincodes:
   probes[mode]=probe(traincodes[mode],train_y,cases,codes[mode],[r['nu_value'] for r in manifest])
   mu=traincodes[mode].mean(0);sd=traincodes[mode].std(0).clip(1e-8)
   geometry[mode]=dict(train=standardized_geometry(traincodes[mode],trainenv,mu,sd),id=standardized_geometry(codes[mode],[r['nu_id'] for r in manifest],mu,sd))
  panels=donor_panel(model,targets.cuda(),codes,manifest,trunk,a.output)
  assert base.parameter_digest(model)==digest and all(p.grad is None for p in model.parameters()) and sha(source)==source_sha
  np.savez_compressed(a.output/'TRAIN_CODES.npz',one_pass=original.numpy(),support50=train_refined.numpy(),nu=train_y,nu_id=trainenv,case=cases)
  write(a.output/'MANIFEST.json',manifests)
  write(a.output/'SUMMARY.json',dict(status='COMPLETE',seed=a.seed,checkpoint_sha256=source_sha,groups=groups,probe=probes,geometry=geometry,donor=panels,
   source_model_unchanged=digest,source_optimizer_updates=0,normalized_code_lr=.1,support_steps=50,manifest_sha256=g.digest_json(manifests),
   scope='Same frozen SPRII checkpoint; support-only inference code optimization. Not a new native training result. Training-normalized geometry, accessibility, and fixed-predictor substitution are distinct.'))
  write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json'),checkpoint_sha256=source_sha));write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
 except BaseException:
  write(a.output/'FAILED.json',dict(traceback=traceback.format_exc()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1234,5678,9012],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 locks=[]
 for d in [HERE,HERE/'gpu_locks']:
  d.mkdir(exist_ok=True);f=(d/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 run(a)
