"""Fill missing frozen NOD per-horizon values; reuse the other9 archived models."""
import argparse,fcntl,os,signal,statistics,subprocess,sys,tarfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import geps_formal_three_seed as g
read,write,sha=g.read,g.write,g.sha
ROOT=HERE/'burgers_native_horizon_completion_retry1'
SEEDS=[1234,5678,9012]
def aggregate(vals):return dict(mean=statistics.mean(vals),seed_sd=statistics.stdev(vals),values=vals,seeds=SEEDS)

def main():
 import torch,numpy as np
 import geps_burgers_pilot as base,sprii_support_code_gate as cache
 import nod_sprii_common_mechanisms_cuda as source
 assert sha(cache.__file__)=='5173857d5c54ab28872be75ea6a68fee3531ec995a614b0bc0bf9f8ce8612c9e'
 assert sha(source.__file__)=='39869d74fa433c3a0086e9f2ce78898c9e1e93a2bbdfb5ef0df1551c14467bb4'
 torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
 old=read(HERE/'geps_pilot_retry1/CONFIG.json');args=argparse.Namespace(nod_source=Path(old['nod_source']),data_root=Path(old['data_root']),seed=0)
 assert sha(args.nod_source/'train_nod_clean.py')==g.NATIVE_SHA and sha(args.nod_source/'ngs/utils.py')==g.LOADER_SHA
 base.load_released_data(args)
 from train_nod_clean import MODEL_DEFAULTS,NGS_INR
 records={k:{} for k in ['NOD_default','SPRII_endpoint','GEPS','CoDA']};sources={}
 for seed in SEEDS:
  out=ROOT/f'seed{seed}';out.mkdir()
  ckpt=Path(f'benchmarks/nod/runs/v3_formal/NOD_clean/seed_{seed}/best_eval.pth');expected=source.NOD_SHA[seed];assert sha(ckpt)==expected
  ck=torch.load(ckpt,map_location='cpu',weights_only=False);cfg=ck['config'];kwargs=dict(cfg.get('model_defaults',MODEL_DEFAULTS));kwargs['context_dim']=int(cfg.get('context_dim_override',cfg.get('context_dim',1)))
  model=NGS_INR(**kwargs).cuda().eval();model.load_state_dict(ck['model_state_dict'],strict=True)
  for p in model.parameters():p.requires_grad_(False)
  digest=base.parameter_digest(model)
  formal=read(HERE/f'geps_formal_three_seed/seed{seed}/MANIFEST.json')
  native_file=Path(f'benchmarks/nod/results/v3_eval/NOD_clean/seed_{seed}.json');native=read(native_file);sources[str(native_file)]=sha(native_file)
  times=torch.tensor(formal['id_test'][0]['t'],dtype=torch.float32,device='cuda');x=torch.linspace(-1,1,401,device='cuda');tt,xx=torch.meshgrid(times,x,indexing='ij');coords=torch.stack([xx.flatten(),tt.flatten()],-1)
  with torch.no_grad():trunk=torch.cat([model.predictioner.backbone.trunk_net(model.predictioner.pe(coords[i:i+1024])) for i in range(0,len(coords),1024)])
  groups={}
  for name,ds,denom in g.groups(args):
   manifest=[];errors=[];zlog=[]
   try:
    for start in range(0,len(ds),8):
     items=[ds[i] for i in range(start,min(start+8,len(ds)))];cond=torch.stack([p['cond_u'] for p in items]).cuda();target=torch.stack([p['target_seq'][:,0] for p in items]).cuda()
     for p in items:assert torch.equal(p['t_idx'].cuda().float()/denom,times) and p['cond_case_idx']!=p['pred_case_idx']
     for j,p in enumerate(items):manifest.append(dict(index=start+j,nu_id=int(p['nu_id']),nu_value=float(p['nu_value'].item()),cond_case_idx=int(p['cond_case_idx']),pred_case_idx=int(p['pred_case_idx']),support_sha256=base.tensor_digest(cond[j].transpose(-1,-2)),query_sha256=base.tensor_digest(target[j].T[None]),t=times.cpu().tolist()))
     with torch.no_grad():
      z=model.encode(cond)
      if start==0:
       with torch.enable_grad():write(out/(name+'_CACHE_PARITY.json'),cache.check_cache(model,target[:,0,None],z,coords,trunk))
      pred=cache.cached_predict(model,target[:,0,None],z,trunk).reshape(len(items),101,401)
      e=(pred-target).square().mean(2);assert torch.isfinite(e).all();errors.append(e.cpu().numpy());zlog.extend(z.cpu().tolist())
   finally:
    if hasattr(ds,'close'):ds.close()
   assert manifest==formal[name]
   e=np.concatenate(errors);score=g.native_reduce(e.mean(1).tolist(),8);oldscore=native['groups'][name]['metrics']['mse']
   assert abs(score-oldscore)<=max(2e-8,2e-5*abs(oldscore)),(seed,name,score,oldscore)
   groups[name]=dict(n_pairs=len(e),native_compatible_mse=score,archived_native_mse=oldscore,trajectory_mean_mse=float(e.mean()),future100_mean_mse=float(e[:,1:].mean()),horizon_mse={str(h):float(e[:,h].mean()) for h in [1,5,50,100]},manifest_sha256=g.digest_json(manifest))
   np.savez_compressed(out/(name+'_ERRORS.npz'),per_pair_time_mse=e,one_pass_codes=np.asarray(zlog));write(out/(name+'_MANIFEST.json'),manifest)
  assert base.parameter_digest(model)==digest and sha(ckpt)==expected and all(p.grad is None for p in model.parameters())
  write(out/'SUMMARY.json',dict(status='COMPLETE',seed=seed,checkpoint_sha256=expected,checkpoint=str(ckpt),groups=groups,source_optimizer_updates=0,code_optimizer_updates=0,source_unchanged=digest))
  write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json')));write(out/'EXIT.json',dict(exit_code=0))
  records['NOD_default'][str(seed)]=read(out/'SUMMARY.json')
  for method,folder in [('SPRII_endpoint','sprii_support_code_formal_three_seed_retry2'),('GEPS','geps_formal_three_seed'),('CoDA','coda_formal_three_seed')]:
   path=HERE/f'{folder}/seed{seed}';s=read(path/'SUMMARY.json');assert read(path/'EXIT.json')['exit_code']==0 and read(path/'COMPLETE.json')['summary_sha256']==sha(path/'SUMMARY.json')
   assert read(path/'MANIFEST.json')==formal
   gg={k:v['one_pass'] for k,v in s['groups'].items()} if method=='SPRII_endpoint' else s['groups']
   records[method][str(seed)]=dict(checkpoint_sha256=s['checkpoint_sha256'],groups=gg)
   for n in ['SUMMARY.json','COMPLETE.json','EXIT.json','MANIFEST.json']:sources[str(path/n)]=sha(path/n)
  print('seed',seed,'COMPLETE',flush=True);del model,ck,trunk
 table={}
 for method,cohort in records.items():
  table[method]={}
  for group in g.GROUPS:
   table[method][group]={k:aggregate([cohort[str(s)]['groups'][group][k] for s in SEEDS]) for k in ['native_compatible_mse','trajectory_mean_mse','future100_mean_mse']}
   table[method][group]['horizon_mse']={str(h):aggregate([cohort[str(s)]['groups'][group]['horizon_mse'][str(h)] for s in SEEDS]) for h in [1,5,50,100]}
 write(ROOT/'SUMMARY.json',dict(status='COMPLETE',records=records,table=table,all_seeds=SEEDS,sources=sources,
  new_source_optimizer_updates=0,new_code_optimizer_updates=0,new_frozen_model_evaluations=3,reused_model_evaluations=9,
  scope='Historical complete3seed one-support prediction horizons. NOD/SPRII direct horizons; GEPS Euler and CoDA RK4 integration with50-step support-code optimization. Individual horizon/trajectory metrics equally average queries; native MSE retains batch8 mean-of-means and101frames including initial. No new sealed evaluation or performance selection.'))
 write(ROOT/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(ROOT/'SUMMARY.json')))
 dest=Path('artifacts/exchange/archives/burgers_native_horizon_completion.tar.gz');assert not dest.exists()
 files=[(p,'results/'+str(p.relative_to(ROOT))) for p in ROOT.rglob('*') if p.is_file() and p.name!='LOCK']
 files += [(Path(p),'sources/'+str(Path(p).relative_to('.'))) for p in sources]
 files.append((Path(__file__),'code/'+Path(__file__).name));write(ROOT/'ARCHIVE_MANIFEST.json',dict(files={n:sha(p) for p,n in files}));files.append((ROOT/'ARCHIVE_MANIFEST.json','ARCHIVE_MANIFEST.json'))
 with tarfile.open(dest.with_suffix('.partial'),'w:gz') as t:
  for p,n in files:t.add(p,arcname=n,recursive=False)
 dest.with_suffix('.partial').replace(dest);write(ROOT/'ARCHIVE.json',dict(status='COMPLETE',path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(files)))
 write(ROOT/'EXIT.json',dict(exit_code=0,time=time.time()))

if __name__=='__main__':
 os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 assert read(HERE/'nod_sprii_common_mechanisms_cuda/EXIT.json')['exit_code']==0
 assert read(HERE/'burgers_native_horizon_completion/EXIT.json')['exit_code']==1
 locks=[]
 for d in [HERE,HERE/'gpu_locks']:
  f=(d/'gpu1.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not ROOT.exists();ROOT.mkdir();write(ROOT/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__),gpu=1))
 def timeout(*_):raise TimeoutError('Fixed15min frozen-inference budget')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,900)
 try:main()
 except BaseException:
  write(ROOT/'FAILED.json',dict(traceback=traceback.format_exc()));write(ROOT/'EXIT.json',dict(exit_code=1,time=time.time()));raise
