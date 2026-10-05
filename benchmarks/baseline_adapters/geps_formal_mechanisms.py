"""Complete GEPS same-checkpoint accessibility and fixed-code donor diagnostics.

Networks stay frozen. The published50-step code adaptation is applied to the
360 existing training histories solely to fit a training-only diagnostic probe.
The final ID codes and targets reuse the already-completed formal manifest.
"""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import geps_formal_three_seed as g
sha,read,write=g.sha,g.read,g.write
def bind(model,codes):
 import torch
 m=copy.deepcopy(model);v=torch.nn.Parameter(codes.detach().clone())
 m.derivative.codes=v;m.derivative.model_aug.codes=v
 for p in m.parameters():p.requires_grad_(False)
 return m,v
def main(a):
 assert not a.output.exists();a.output.mkdir(parents=True)
 write(a.output/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__)))
 def timeout(*_):raise TimeoutError('GEPS mechanism fixed75-minute per-seed deadline reached')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,4500)
 try:
  import numpy as np,torch
  import geps_burgers_pilot as base
  from sprii_support_code_formal_v3 import probe,standardized_geometry
  assert sha(base.__file__)==g.PILOT_SHA
  assert sha(HERE/'sprii_support_code_formal_v3.py')=='4f7e8f28c36151628fb679fe169e04f7843157cc813dc582537c1ea5d176b3bd'
  torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  src=HERE/f'geps_formal_three_seed/seed{a.seed}';cfg=read(src/'CONFIG.json');summ=read(src/'SUMMARY.json')
  assert read(src/'EXIT.json')['exit_code']==0 and read(src/'COMPLETE.json')['summary_sha256']==sha(src/'SUMMARY.json')
  checkpoint=Path(cfg['source_checkpoint']) if cfg.get('source_checkpoint') else src/'latest.pt'
  assert sha(checkpoint)==summ['checkpoint_sha256']
  args=argparse.Namespace(**cfg)
  for k in ['nod_source','geps_source','data_root']:setattr(args,k,Path(getattr(args,k)))
  args.device='cuda:0';data,meta=base.load_released_data(args)
  model,_=base.build_model(args,9,torch.device('cuda:0'));ck=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
  model.load_state_dict(ck['model'],strict=True);model.eval()
  for p in model.parameters():p.requires_grad_(False)
  source=base.parameter_digest(model);shared=base.parameter_digest(model,exclude_codes=True)
  initial=model.derivative.codes.detach().mean(0);times=data['train']['t'].cuda();traincodes=[];adapt_cost=0.;begin=time.monotonic()
  write(a.output/'CONFIG.json',dict(seed=a.seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_optimizer_updates=0,
   protocol='Unchanged official adaptation: independent code, mean training-code initialization, Adam.01 x50,101-frame support',
   probe_histories=360,probe_data='All training trajectories cases0..39; ridge selected only by training-case5fold',
   donor_data='Archived final45 ID codes and frozen deterministic target manifest; no new code fitting on test',
   source_networks_frozen=True,new_sealed_test=False,finite_budget_seconds=4500))
  for start in range(0,360,4):
   truth=data['train']['curves'][start:start+4].cuda();m,z=bind(model,initial[None].repeat(len(truth),1));z.requires_grad_(True)
   opt=torch.optim.Adam([z],lr=.01,betas=(.9,.999));env=torch.arange(len(truth),device='cuda')
   torch.cuda.synchronize();b=time.monotonic()
   for _ in range(50):
    opt.zero_grad(set_to_none=True);pred=m(truth,times,env,epsilon=0);loss=(pred-truth).square().mean();assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(z.grad).all();opt.step()
   torch.cuda.synchronize();adapt_cost+=time.monotonic()-b
   assert base.parameter_digest(m,exclude_codes=True)==shared and torch.isfinite(z).all();traincodes.extend(z.detach().cpu().tolist())
   write(a.output/'TRAIN_CODES_PARTIAL.json',dict(completed=start+len(truth),total=360,codes=traincodes,adapt_seconds=adapt_cost))
   del m,z,opt
  from ngs.utils import BurgersPairedDataset
  ds=BurgersPairedDataset(data_root=str(args.data_root),split='train',cache_mode='none',seed=0)
  try:nu_map={int(s['nu_id']):float(s['nu']) for s in ds.shards}
  finally:ds.close()
  traincodes=np.asarray(traincodes);trainenv=data['train']['envs'].numpy();cases=data['train']['cases'].numpy();y=np.asarray([nu_map[int(i)] for i in trainenv])
  manifest=read(src/'MANIFEST.json')['id_test'];codes=np.asarray(read(src/'CODES.json')['id_test']['adapted_codes']);old=read(src/'id_test_PAIRS.json')
  assert len(manifest)==len(codes)==len(old)==45
  ds=BurgersPairedDataset(data_root=str(args.data_root),split='test',prediction_horizon=101,cache_mode='none',seed=0)
  targets=[]
  try:
   for i in range(45):
    item=ds[i];target=item['target_seq'].permute(1,2,0);support=item['cond_u'].transpose(-1,-2)
    row=manifest[i]
    assert row['cond_case_idx']==item['cond_case_idx'] and row['pred_case_idx']==item['pred_case_idx'] and row['nu_id']==item['nu_id']
    assert row['support_sha256']==base.tensor_digest(support) and row['query_sha256']==base.tensor_digest(target)
    targets.append(target)
   report_times=item['t_idx'].cuda().float()/100.
  finally:ds.close()
  targets=torch.stack(targets).cuda();code_tensor=torch.as_tensor(codes,device='cuda',dtype=targets.dtype)
  def forecast(ix,js):
   truth=targets[ix];m,z=bind(model,code_tensor[js]);env=torch.arange(len(ix),device='cuda')
   with torch.no_grad():pred=m(g.future_free_input(truth),report_times,env,epsilon=0);errors=(pred-truth).square().mean((1,2))
   assert torch.isfinite(errors).all() and base.parameter_digest(m,exclude_codes=True)==shared
   return errors.cpu().numpy()
  own=[]
  for start in range(0,45,4):
   ids=list(range(start,min(start+4,45)));own.extend(forecast(ids,ids))
  own=np.asarray(own);expected=np.asarray([r['mse_all101'] for r in old]);np.testing.assert_allclose(own.mean(1),expected,rtol=2e-5,atol=2e-8)
  write(a.output/'ARCHIVED_CODE_PARITY.json',dict(status='PASS',n=45,max_mse_difference=float(np.max(np.abs(own.mean(1)-expected))),checkpoint_sha256=sha(checkpoint)))
  # Deterministic unique support histories avoid overweighting a repeated code.
  donors=[];seen=set()
  for j,d in enumerate(manifest):
   key=(d['nu_id'],d['cond_case_idx'],d['support_sha256'])
   if key not in seen:donors.append(j);seen.add(key)
  pairs=[(i,j) for i,r in enumerate(manifest) for j in donors if manifest[j]['nu_id']!=r['nu_id'] or manifest[j]['cond_case_idx'] not in [r['pred_case_idx'],r['cond_case_idx']]]
  records=[]
  for start in range(0,len(pairs),4):
   batch=pairs[start:start+4];errors=forecast([x[0] for x in batch],[x[1] for x in batch])
   for (i,j),error in zip(batch,errors):records.append(dict(recipient=i,donor=j,target_nu_id=manifest[i]['nu_id'],donor_nu_id=manifest[j]['nu_id'],mse_all101=float(error.mean()),h1=float(error[1]),h5=float(error[5]),h50=float(error[50]),h100=float(error[100])))
  cells=[];delta=[]
  for i,r in enumerate(manifest):
   byenv={env:[v for v in records if v['recipient']==i and v['donor_nu_id']==env] for env in range(9)};assert all(byenv.values())
   cell={env:{h:float(np.mean([v[h] for v in vals])) for h in ['mse_all101','h1','h5','h50','h100']} for env,vals in byenv.items()}
   cells.extend(dict(recipient=i,target_nu_id=r['nu_id'],donor_nu_id=env,**v) for env,v in cell.items())
   delta.append(dict(recipient=i,nu_id=r['nu_id'],h50_wrong_minus_matched=float(np.mean([v['h50'] for env,v in cell.items() if env!=r['nu_id']]))-cell[r['nu_id']]['h50']))
  means=np.asarray([np.mean([x['h50_wrong_minus_matched'] for x in delta if x['nu_id']==env]) for env in range(9)])
  rng=np.random.default_rng(834);ci=np.quantile(rng.choice(means,(10000,9)).mean(1),[.025,.975])
  write(a.output/'DONOR_PANEL.json',dict(pair_rows=records,cells=cells,per_recipient=delta,unique_donor_rows=donors,system_bootstrap_ci95=ci.tolist()))
  probe_result=probe(traincodes,y,cases,codes,[r['nu_value'] for r in manifest]);mu=traincodes.mean(0);sd=traincodes.std(0).clip(1e-8)
  geometry={k:standardized_geometry(z,lab,mu,sd) for k,z,lab in [('train',traincodes,trainenv),('id',codes,[r['nu_id'] for r in manifest])]}
  assert base.parameter_digest(model)==source and sha(checkpoint)==summ['checkpoint_sha256'] and all(p.grad is None for p in model.parameters())
  np.savez_compressed(a.output/'PROBE_CODES.npz',train=traincodes,train_nu=y,train_case=cases,id=codes,id_nu=[r['nu_value'] for r in manifest])
  write(a.output/'SUMMARY.json',dict(status='COMPLETE',seed=a.seed,checkpoint_sha256=sha(checkpoint),source_unchanged=source,source_optimizer_updates=0,
   training_history_code_adapt_seconds=adapt_cost,probe=probe_result,geometry=geometry,h50_wrong_minus_matched=float(means.mean()),system_bootstrap_ci95=ci.tolist(),
   n_training_histories=360,n_id_recipients=45,n_legal_donor_pairs=len(records),seconds=time.monotonic()-begin,
   scope='Same GEPS checkpoint and official50-step code protocol; training-only diagnostic probe and frozen-predictor donor use, not new source training or return evidence.'))
  write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json')));write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
 except BaseException:
  write(a.output/'FAILED.json',dict(traceback=traceback.format_exc()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1234,5678,9012],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 locks=[]
 for d in [HERE,HERE/'gpu_locks']:
  d.mkdir(exist_ok=True);f=(d/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(HERE/'geps_deps'))
 main(a)
