"""Original A fresh-reader and physical probes; development only, fixed final source."""
import argparse, dataclasses, hashlib, json, os, pathlib, sys, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
R=pathlib.Path(os.environ.get('SPRII_ROOT', '.'));N=R/'benchmarks/springworld/native';B=R/'data/springworld/research_bank'
sys.path.insert(0,str(R/'experiments/spring_postrun/reader'))
from sprii_next.io import sha,read,write,plain,digest
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)

def activate(lp,lx):
 from sprii_next import protocol
 original=protocol.activate_native
 def patched(root):
  paths=original(root)
  from persistbench.envs.visual_elastic_coupling import a_pairing
  a_pairing.CONFIGURATIONS['Both']=('B3','G3','Independent',lp,lx)
  return paths
 protocol.activate_native=patched
 return protocol

def export(a,out):
 protocol=activate(a.lp,a.lx)
 wrapper=read(a.source.with_name(a.source.name+'_WRAPPER.json'))
 assert (wrapper['seed'],wrapper['lambda_p'],wrapper['lambda_x'],wrapper['steps'])==(a.seed,a.lp,a.lx,10000)
 assert wrapper['test_read'] is False
 if not a.smoke:
  ex=read(a.source.with_name(a.source.name+'_EXIT.json'))
  assert ex['returncode']==0 and ex['actual_child_exited'] and not pathlib.Path(f"/proc/{ex['pid']}").exists()
  diag=read(a.source.with_name(a.source.name+'_DIAGNOSTICS_COMPLETE.json'));assert diag['rows']==100
  descriptor=protocol.export_spring(N,a.source/'COMPLETE.json',B,'Both',a.seed,out/'cache',device='cuda:0')
 else:
  # An intermediate checkpoint is used only in this explicitly labeled engineering preflight.
  paths=protocol.activate_native(N)
  from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec,_new_model
  from persistbench.envs.visual_elastic_coupling.a_head_features import model_state_sha256,extract_features,AHeadFeatureCache
  from persistbench.envs.visual_elastic_coupling.a_head_targets import extract_targets
  from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan,AHeadDataAccess
  from persistbench.envs.visual_elastic_coupling.a_pairing import configuration
  run=read(a.source/'RUN.json');assert run['configuration']==configuration('Both')
  commit=a.source/f'checkpoint_{a.smoke_checkpoint:07d}.COMMIT.json';cr=read(commit)
  checkpoint=a.source/f'checkpoint_{a.smoke_checkpoint:07d}.pt'
  assert cr['step']==a.smoke_checkpoint and cr['run']['sha256']==sha(a.source/'RUN.json')
  assert sha(checkpoint)==cr['files'][checkpoint.name]['sha256']
  ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
  assert ck['step']==a.smoke_checkpoint and ck['binding']==run['binding'] and ck['run_sha256']==sha(a.source/'RUN.json')
  values={f.name:run['spec'][f.name] for f in dataclasses.fields(PretrainingSpec)};values['milestone_steps']=tuple(values['milestone_steps'])
  spec=PretrainingSpec(**values);assert (spec.model_seed,spec.sampling_seed,spec.stochastic_seed)==(a.seed,)*3
  model=_new_model('Both',spec,torch.device('cpu'));assert model_state_sha256(model)==ck['initial_model_sha256']
  model.load_state_dict(ck['model']);assert model_state_sha256(model)==ck['model_state_sha256']
  model.eval().requires_grad_(False).cuda()
  expected=run['binding']['bank_snapshot_sha256'];assert sha(B/'BANK_SNAPSHOT.json')==expected
  plan=AHeadCasePlan(read(B/'MANIFEST.private.json'),seed=0,history_frames=96)
  access=AHeadDataAccess(B,plan,snapshot_sha256=expected)
  cache=out/'cache';cache.mkdir()
  extract_features(access,model,cache/'features',expected_model_state_sha256=ck['model_state_sha256'],workers=2,progress=lambda x:print(json.dumps(x),flush=True))
  fr=cache/'features/FEATURES.json';features=AHeadFeatureCache(fr.parent,plan,receipt_sha256=sha(fr),model_state_sha256=ck['model_state_sha256'],bank_snapshot_sha256=expected)
  extract_targets(access,features,cache/'targets',workers=2)
  descriptor=dict(smoke=True,source_step=a.smoke_checkpoint,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),model_state_sha256=ck['model_state_sha256'],features_receipt=str(fr),features_receipt_sha256=sha(fr),targets_receipt=str(cache/'targets/SUPERVISION.json'),targets_receipt_sha256=sha(cache/'targets/SUPERVISION.json'),manifest=str(B/'MANIFEST.private.json'),native_paths=paths)
  write(cache/'SMOKE_SOURCE.json',descriptor)
 write(out/'EXPORT_COMPLETE.json',dict(status='COMPLETE',source=descriptor,smoke=a.smoke,source_optimization_updates=0,lambda_p=a.lp,lambda_x=a.lx,source_seed=a.seed,test_read=False))

def bound(a,out):
 from sprii_next import protocol
 protocol.activate_native(N)
 from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan
 from persistbench.envs.visual_elastic_coupling.a_head_features import AHeadFeatureCache
 from persistbench.envs.visual_elastic_coupling.a_head_targets import AHeadTargets
 cache=out/'cache';d=read(cache/('SMOKE_SOURCE.json' if a.smoke else 'SOURCE.json'))
 plan=AHeadCasePlan(read(B/'MANIFEST.private.json'),seed=0,history_frames=96);expected=sha(B/'BANK_SNAPSHOT.json')
 f=AHeadFeatureCache(cache/'features',plan,receipt_sha256=sha(cache/'features/FEATURES.json'),model_state_sha256=d['model_state_sha256'],bank_snapshot_sha256=expected)
 labels={s:AHeadTargets(cache/'targets',plan,receipt_sha256=sha(cache/'targets/SUPERVISION.json'),bank_snapshot_sha256=expected,split=s,purpose='fit' if s=='train' else 'score') for s in ['train','validation']}
 return d,f,labels

def fit(a,out):
 d,f,lab=bound(a,out)
 from persistbench.envs.visual_elastic_coupling.a_head_fitting import HeadFitSpec,fit_head
 spec=HeadFitSpec(head_seed=a.reader_seed,sampling_seed=a.reader_seed)
 if a.smoke:spec=dataclasses.replace(spec,steps=100,warmup_steps=99,validation_every=100,save_every=100,log_every=10)
 dest=out/f'reader{a.reader_seed}'
 fit_head(f,lab['train'],lab['validation'],dest,arm='matched',spec=spec,device='cuda:0',progress=lambda x:print(json.dumps(x),flush=True))
 write(out/f'READER{a.reader_seed}_TRAIN_COMPLETE.json',dict(status='COMPLETE',receipt_sha256=sha(dest/'COMPLETE.json'),source_checkpoint_sha256=d['checkpoint_sha256'],smoke=a.smoke,reader_updates=spec.steps,source_optimization_updates=0,test_read=False))

def evaluate(a,out):
 d,f,lab=bound(a,out)
 from persistbench.envs.visual_elastic_coupling.a_head_fitting import load_fitted_head,_math_profile
 dest=out/f'reader{a.reader_seed}';head,_=load_fitted_head(dest,f,lab['train'],lab['validation'],receipt_sha256=sha(dest/'COMPLETE.json'),device='cuda:0')
 H=[1,2,4,8,16];rows=[];pred=[];target=[];cases=[]
 def flush():
  if not cases:return
  x=f.inputs(cases,arm='matched',device=torch.device('cuda:0'))
  with torch.inference_mode():y=head(x).cpu().numpy()
  t=np.asarray(lab['validation'].targets(cases),np.float32)
  assert np.isfinite(y).all() and y.shape==t.shape
  pred.append(y);target.append(t)
  for c in cases:
   rows.append({k:plain(c[k]) for k in ['case_sha256','case_id','system_key','query_episode','matched_episode','q','horizon','kind','stratum','split_weight']})
  cases.clear()
 with _math_profile(torch.device('cuda:0')):
  for i,b in enumerate(f.plan.base):
   if b['split']!='validation':continue
   for q in [0,1]:
    for h in H:
     cases.append(f.plan.case(i,q=q,horizon=h,draw=0))
     if len(cases)==256:flush()
  flush()
 y=np.concatenate(pred);t=np.concatenate(target);err=np.square(y.astype(np.float64)-t.astype(np.float64)).mean(1)
 assert len(rows)==16020 and abs(sum(r['split_weight'] for r in rows)-1)<1e-8
 primary=np.array([r['kind']=='cold' and r['q']==0 and r['horizon']==16 and r['stratum'] in ['continuous_new_systems','heldout_factorial_combinations'] for r in rows])
 ids=np.array([r['system_key'] for r in rows]);primary_system={s:float(err[primary&(ids==s)].mean()) for s in sorted(set(ids[primary]))}
 assert primary.sum()==600 and len(primary_system)==100
 weighted=float(sum(e*r['split_weight'] for e,r in zip(err,rows)))
 original=read(dest/f'validation_{100 if a.smoke else 10000:07d}.json')
 assert abs(weighted-original['validation_loss'])<=1e-6*max(1,abs(weighted))
 ev=out/f'evaluation{a.reader_seed}';ev.mkdir()
 np.savez_compressed(ev/'VECTORS.npz',prediction=y,target=t,mse=err,system=ids,primary=primary)
 manifest=dict(schema='original-afresh-development-v1',rows=rows,plan_sha256=f.plan.plan_sha256,bank_snapshot_sha256=sha(B/'BANK_SNAPSHOT.json'),test_read=False)
 write(ev/'MANIFEST.json',manifest)
 result=dict(status='COMPLETE',source_seed=a.seed,reader_seed=a.reader_seed,lambda_p=a.lp,lambda_x=a.lx,smoke=a.smoke,rows=16020,primary_rows=600,primary_systems=100,primary_system_mse=primary_system,primary_mse=float(np.mean(list(primary_system.values()))),full_mixture_mse=weighted,original_score_parity=True,manifest_sha256=sha(ev/'MANIFEST.json'),head_receipt_sha256=sha(dest/'COMPLETE.json'),source_checkpoint_sha256=d['checkpoint_sha256'],source_optimization_updates=0,test_read=False)
 write(ev/'SUMMARY.json',result);write(ev/'COMPLETE.json',dict(files={n:sha(ev/n) for n in ['VECTORS.npz','MANIFEST.json','SUMMARY.json']},status='COMPLETE'))

def probe(a,out):
 assert not a.smoke
 from sprii_next.providers import SpringCache
 from sprii_next.decoder import RidgeDecoder
 from sprii_next.geometry import source_geometry
 provider=SpringCache(read(out/'cache/SOURCE.json'))
 tr,th,ids,_=provider.donors('train');v,vt,vi,vd=provider.donors('validation')
 decoder=RidgeDecoder.fit(tr,th,ids,split='train',alpha=1.,dim=3)
 write(out/'PROBE.json',dict(result=decoder.score(v,vt,vi),decoder=decoder.record(),source_checkpoint_sha256=provider.descriptor['checkpoint_sha256'],source_optimization_updates=0,test_read=False))
 write(out/'GEOMETRY.json',source_geometry(provider))
 np.savez_compressed(out/'PROBE_VECTORS.npz',code=v,theta=vt,system=vi,donor=vd,prediction=decoder.predict(v),target=decoder.oracle(vt))
 write(out/'PROBE_COMPLETE.json',dict(status='COMPLETE',files={n:sha(out/n) for n in ['PROBE.json','GEOMETRY.json','PROBE_VECTORS.npz']}))

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['export','fit','evaluate','probe']);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--lp',type=float,required=True);p.add_argument('--lx',type=float,required=True);p.add_argument('--reader-seed',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-checkpoint',type=int,default=2500);a=p.parse_args()
 assert (a.lp,a.lx) in [(1,.1),(.25,.1),(4,.1),(1,.025),(1,.4)] and a.seed in [0,1,2] and a.reader_seed in [0,1,2]
 if a.phase=='export':a.output.mkdir(exist_ok=False,parents=True)
 start=time.time();globals()[a.phase](a,a.output)
 print(json.dumps(dict(status='COMPLETE',phase=a.phase,smoke=a.smoke,seconds=time.time()-start,runner_sha256=sha(__file__))),flush=True)
if __name__=='__main__':main()
