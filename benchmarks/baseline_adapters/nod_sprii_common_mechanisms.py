"""Frozen CPU diagnostics on the same ID manifest as GEPS/CoDA.

Unifies geometry definitions and unique support weighting, without any source
or inference-code optimization. Historical diagnostics remain unchanged.
"""
import argparse,json,os,signal,sys,tarfile,time,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
import sprii_support_code_gate as cache
read,write,sha=cache.read,cache.write,cache.sha
ROOT=HERE/'nod_sprii_common_mechanisms'
NOD_SHA={1234:'cb35e84c54458aed628dea1e175c9be144d04915261b36d59ed7998537969fd6',5678:'e575c415f2206a8c263410a80327e48f2ad3e74b3dadb3ab13b5e59addd32c8d',9012:'7ef6f29f0e6df2a18ccd3c6c15518a7952daa2cbe898c64fde052b7ed2f01c3e'}
def main():
 import numpy as np,torch
 import geps_formal_three_seed as g,geps_burgers_pilot as base
 import sprii_support_code_formal_v3 as s
 assert sha(cache.__file__)=='5173857d5c54ab28872be75ea6a68fee3531ec995a614b0bc0bf9f8ce8612c9e'
 assert sha(s.__file__)=='4f7e8f28c36151628fb679fe169e04f7843157cc813dc582537c1ea5d176b3bd'
 assert sha(base.__file__)==g.PILOT_SHA
 torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
 old=read(HERE/'geps_pilot_retry1/CONFIG.json');args=argparse.Namespace(nod_source=Path(old['nod_source']),data_root=Path(old['data_root']),seed=0)
 assert sha(args.nod_source/'train_nod_clean.py')==g.NATIVE_SHA and sha(args.nod_source/'ngs/utils.py')==g.LOADER_SHA
 data,meta=base.load_released_data(args)
 from train_nod_clean import MODEL_DEFAULTS,NGS_INR
 from ngs.utils import BurgersPairedDataset
 write(ROOT/'PLAN.json',dict(device='cpu',threads=1,all_seeds=[1234,5678,9012],methods=['NOD_default','SPRII_endpoint'],
  source_optimizer_updates=0,code_optimizer_updates=0,new_observations=0,scope='Frozen inference only; identical historical45-query bank and unique support-history donor weighting as GEPS/CoDA mechanisms.',
  geometry='Training-population standardization, mean squared distance to system centroids and grand centroid.',probe='360training histories, training-case5fold ridge to nu;45ID queries.',max_seconds=1800))
 ds=BurgersPairedDataset(data_root=str(args.data_root),split='train',cache_mode='none',seed=0)
 try:nu_map={int(x['nu_id']):float(x['nu']) for x in ds.shards}
 finally:ds.close()
 env=data['train']['envs'].numpy();cases=data['train']['cases'].numpy();y=np.asarray([nu_map[int(x)] for x in env]);results={}
 for seed in [1234,5678,9012]:
  manifest=read(HERE/f'geps_formal_three_seed/seed{seed}/MANIFEST.json')['id_test']
  ds=BurgersPairedDataset(data_root=str(args.data_root),split='test',prediction_horizon=101,cache_mode='none',seed=0)
  targets=[];conditions=[]
  try:
   for i,r in enumerate(manifest):
    it=ds[i];t=it['target_seq'][:,0];c=it['cond_u']
    assert r['cond_case_idx']==it['cond_case_idx'] and r['pred_case_idx']==it['pred_case_idx'] and r['nu_id']==it['nu_id']
    assert r['support_sha256']==base.tensor_digest(c.transpose(-1,-2)) and r['query_sha256']==base.tensor_digest(t.T[None])
    targets.append(t);conditions.append(c)
  finally:ds.close()
  targets=torch.stack(targets);conditions=torch.stack(conditions);assert len(targets)==45
  times=torch.tensor(manifest[0]['t'],dtype=torch.float32);xx=torch.linspace(-1,1,401);tt,xx=torch.meshgrid(times,xx,indexing='ij');coords=torch.stack([xx.flatten(),tt.flatten()],-1)
  donors=[];seen=set()
  for j,r in enumerate(manifest):
   key=(r['nu_id'],r['cond_case_idx'],r['support_sha256'])
   if key not in seen:donors.append(j);seen.add(key)
  pairs=[(i,j) for i,r in enumerate(manifest) for j in donors if manifest[j]['nu_id']!=r['nu_id'] or manifest[j]['cond_case_idx'] not in [r['pred_case_idx'],r['cond_case_idx']]]
  for method in ['NOD_default','SPRII_endpoint']:
   out=ROOT/f'{method}/seed{seed}';assert not out.exists();out.mkdir(parents=True);begin=time.monotonic()
   if method=='NOD_default':
    src=Path(f'benchmarks/nod/runs/v3_formal/NOD_clean/seed_{seed}/best_eval.pth');expected_sha=NOD_SHA[seed]
    expected=read(f'benchmarks/nod/results/v3_eval/NOD_clean/seed_{seed}.json')['groups']['id_test']['metrics']['mse']
   else:
    path,expected_sha=s.SOURCES[seed];src=Path(path)
    saved=read(HERE/f'sprii_support_code_formal_three_seed_retry2/seed{seed}/SUMMARY.json');expected=saved['groups']['id_test']['one_pass']['native_compatible_mse']
   assert sha(src)==expected_sha
   ck=torch.load(src,map_location='cpu',weights_only=False);cfg=ck['config'];kwargs=dict(cfg.get('model_defaults',MODEL_DEFAULTS));kwargs['context_dim']=int(cfg.get('context_dim_override',cfg.get('context_dim',1)))
   model=NGS_INR(**kwargs).eval();model.load_state_dict(ck['model_state_dict'],strict=True);assert model.context_dim==1
   for p in model.parameters():p.requires_grad_(False)
   digest=base.parameter_digest(model)
   with torch.no_grad():
    train=torch.cat([model.encode(data['train']['curves'][i:i+8].transpose(-1,-2)) for i in range(0,360,8)])
    codes=torch.cat([model.encode(conditions[i:i+8]) for i in range(0,45,8)])
    trunk=torch.cat([model.predictioner.backbone.trunk_net(model.predictioner.pe(coords[i:i+1024])) for i in range(0,len(coords),1024)])
   parity=cache.check_cache(model,targets[:8,0,None],codes[:8],coords,trunk)
   with torch.no_grad():
    own=torch.cat([(cache.cached_predict(model,targets[i:i+8,0,None],codes[i:i+8],trunk).reshape(-1,101,401)-targets[i:i+8]).square().mean(2) for i in range(0,45,8)])
   native=g.native_reduce(own.mean(1).tolist(),8);assert abs(native-expected)<=max(2e-8,2e-5*abs(expected)),(method,seed,native,expected)
   if method=='SPRII_endpoint':
    savedcodes=read(HERE/f'sprii_support_code_formal_three_seed_retry2/seed{seed}/id_test_PAIRS.json')['codes']['one_pass']
    np.testing.assert_allclose(codes.numpy(),savedcodes,rtol=2e-5,atol=2e-6)
   write(out/'PARITY.json',dict(status='PASS',cache=parity,native_id_mse=native,archived_native_id_mse=expected,checkpoint_sha256=expected_sha,device='CPU; report bank times copied exactly from CUDA manifest'))
   rows=[]
   with torch.no_grad():
    for start in range(0,len(pairs),8):
     b=pairs[start:start+8];ii=[x[0] for x in b];jj=[x[1] for x in b]
     e=(cache.cached_predict(model,targets[ii,0,None],codes[jj],trunk).reshape(-1,101,401)-targets[ii]).square().mean(2);assert torch.isfinite(e).all()
     for (i,j),err in zip(b,e):rows.append(dict(recipient=i,donor=j,target_nu_id=manifest[i]['nu_id'],donor_nu_id=manifest[j]['nu_id'],mse_all101=float(err.mean()),h1=float(err[1]),h5=float(err[5]),h50=float(err[50]),h100=float(err[100])))
   cells=[];delta=[]
   for i,r in enumerate(manifest):
    byenv={k:[v for v in rows if v['recipient']==i and v['donor_nu_id']==k] for k in range(9)};assert all(byenv.values())
    cell={k:{h:float(np.mean([v[h] for v in vals])) for h in ['mse_all101','h1','h5','h50','h100']} for k,vals in byenv.items()}
    cells.extend(dict(recipient=i,target_nu_id=r['nu_id'],donor_nu_id=k,**v) for k,v in cell.items())
    delta.append(dict(recipient=i,nu_id=r['nu_id'],h50_wrong_minus_matched=float(np.mean([v['h50'] for k,v in cell.items() if k!=r['nu_id']]))-cell[r['nu_id']]['h50']))
   means=np.asarray([np.mean([x['h50_wrong_minus_matched'] for x in delta if x['nu_id']==k]) for k in range(9)]);rng=np.random.default_rng(834);ci=np.quantile(rng.choice(means,(10000,9)).mean(1),[.025,.975])
   write(out/'DONOR_PANEL.json',dict(pair_rows=rows,cells=cells,per_recipient=delta,unique_donor_rows=donors,system_bootstrap_ci95=ci.tolist()))
   train=train.numpy();z=codes.numpy();mu=train.mean(0);sd=train.std(0).clip(1e-8)
   rec=dict(status='COMPLETE',method=method,seed=seed,checkpoint=str(src),checkpoint_sha256=expected_sha,source_optimizer_updates=0,code_optimizer_updates=0,
    probe=s.probe(train,y,cases,z,[r['nu_value'] for r in manifest]),geometry=dict(train=s.standardized_geometry(train,env,mu,sd),id=s.standardized_geometry(z,[r['nu_id'] for r in manifest],mu,sd)),
    h50_wrong_minus_matched=float(means.mean()),system_bootstrap_ci95=ci.tolist(),n_legal_donor_pairs=len(rows),n_training_histories=360,n_id_recipients=45,
    manifest_sha256=g.digest_json(manifest),donor_pair_manifest_sha256=g.digest_json(pairs),source_unchanged=digest,seconds=time.monotonic()-begin)
   assert base.parameter_digest(model)==digest and sha(src)==expected_sha and all(p.grad is None for p in model.parameters())
   np.savez_compressed(out/'PROBE_CODES.npz',train=train,train_nu=y,train_case=cases,id=z,id_nu=[r['nu_value'] for r in manifest])
   write(out/'MANIFEST.json',manifest);write(out/'SUMMARY.json',rec);write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json')));write(out/'EXIT.json',dict(exit_code=0))
   results[f'{method}_{seed}']=rec;print(method,seed,'COMPLETE',flush=True);del model,ck,trunk
 write(ROOT/'SUMMARY.json',dict(status='COMPLETE',records=results,all_seeds=[1234,5678,9012],scope='Same45ID recipient bank and unique legal donor histories as GEPS/CoDA; frozen source/code, no parameter search. Old geometry and donor tables preserved.'))
 write(ROOT/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(ROOT/'SUMMARY.json')))
 dest=Path('artifacts/exchange/archives/nod_sprii_common_mechanisms.tar.gz');assert not dest.exists()
 files=[(p,'results/'+str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='LOCK']+[(HERE/'nod_sprii_common_mechanisms.py','code/nod_sprii_common_mechanisms.py')]
 write(ROOT/'ARCHIVE_MANIFEST.json',dict(files={n:sha(p) for p,n in files}));files.append((ROOT/'ARCHIVE_MANIFEST.json','ARCHIVE_MANIFEST.json'))
 tmp=dest.with_suffix('.partial')
 with tarfile.open(tmp,'w:gz') as t:
  for p,n in files:t.add(p,arcname=n,recursive=False)
 tmp.replace(dest);write(ROOT/'ARCHIVE.json',dict(status='COMPLETE',path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(files)));write(ROOT/'EXIT.json',dict(exit_code=0,time=time.time()))
if __name__=='__main__':
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'LOCK').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'RUN.json').exists()
  write(ROOT/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__)))
  def timeout(*_):raise TimeoutError('Fixed30-minute CPU-only diagnostic budget')
  signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,1800)
  try:main()
  except BaseException:
   write(ROOT/'FAILED.json',dict(traceback=traceback.format_exc()));write(ROOT/'EXIT.json',dict(exit_code=1,time=time.time()));raise
