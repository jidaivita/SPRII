"""Fixed CoDA decoder, four encoder objectives, three already-trained source seeds."""
import argparse,ast,hashlib,importlib.util,json,os,pathlib,signal,sys,time,traceback
ROOT=pathlib.Path(os.environ.get('SPRII_ROOT', '.'));HERE=pathlib.Path(__file__).resolve().parent
CODE=ROOT/'benchmarks/baseline_adapters';OLD=ROOT/'runs/coda_encoder_gate_seed1234'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def context(a):
 path=HERE/'coda_frozen_encoder_gate.py'
 assert sha(path)=='4c5fe642dadf9d3fa4947c51a3a937fa6b7a9a68deb74f7eecf013b0d94b7e58'
 tree=ast.parse(path.read_text());changes=[]
 class Patch(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str) and node.value in ['coda_formal_three_seed/seed1234','coda_postrun_mechanisms_three_seed/seed1234']:
    changes.append(('seed_path',node.value));return ast.copy_location(ast.Constant(node.value.replace('1234',str(a.seed))),node)
   # The original gate's measured30-minute cap was already extended to50min;
   # keep one declared finite3000s cap here for every encoder arm.
   if node.value==1800 and isinstance(node.value,int):return ast.copy_location(ast.Constant(3000),node)
   return node
  def visit_Assign(self,node):
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
    if node.targets[0].id=='cross' and ast.unparse(node.value)=="a.arm == 'sprii_cross'":
     changes.append(('cross_branch','all except plain'));node.value=ast.parse("a.arm != 'plain'",mode='eval').body
    elif node.targets[0].id=='swapped' and ast.unparse(node.value)=='torch.cat([z_cross[9:], z_cross[:9]])':
     changes.append(('donor_permutation','frozen arm dispatch'));node.value=ast.parse('_donor_codes(z_cross, step, a.arm, SEED)',mode='eval').body
   return self.generic_visit(node)
 tree=Patch().visit(tree);ast.fix_missing_locations(tree)
 assert len(changes)==4,changes
 base=type(sys)('_coda_frozen_external');base.__file__=str(path);exec(compile(tree,'<bounded_external_coda_adapter>','exec'),base.__dict__)
 oldread=base.read
 def remapped_read(p):
  x=oldread(p)
  if pathlib.Path(p)==CODE/'geps_pilot_retry1/CONFIG.json':
   x=dict(x)
   for k in ['nod_source','data_root']:x[k]=str(ROOT/'inputs'/x[k].lstrip('/'))
  return x
 base.read=remapped_read;base.SEED=a.seed
 source=CODE/f'coda_formal_three_seed/seed{a.seed}';teacher=CODE/f'coda_postrun_mechanisms_three_seed/seed{a.seed}'
 base.SOURCE_SHA=read(source/'SUMMARY.json')['checkpoint_sha256'];base.TEACHER_SHA=sha(teacher/'PROBE_CODES.npz')
 oldseed=base.set_seed;base.set_seed=lambda seed=a.seed:oldseed(seed)
 import numpy as np
 generator=np.random.default_rng(a.seed+271828);offsets=np.concatenate([generator.permutation(np.arange(1,9)) for _ in range(125)])
 def donor(z,step,arm,seed):
  import torch
  if arm=='sprii_cross':return torch.cat([z[9:],z[:9]])
  if arm=='duplicate_self':return z
  assert arm=='random_cross';p=(torch.arange(9,device=z.device)+int(offsets[step-1]))%9
  return z[torch.cat([p+9,p])]
 base._donor_codes=donor
 # Original data digest includes absolute shard identities. Relocation may
 # change that digest without changing a byte: verify EVERY trajectory and all
 # time/grid fields, then canonicalize only the input-root prefix for provenance.
 sys.path[:0]=[str(CODE/'geps_deps'),str(CODE)]
 import geps_burgers_pilot as loader
 original_load=loader.load_released_data;path_audit={}
 def relocated_load(args):
  data,meta=original_load(args);oldmeta=oldread(source/'DATA_MANIFEST.json')
  for split in ['train','eval']:
   physical=meta[split]['data_sha256'];canonical=[]
   for row in meta[split]['cases']:
    rr=dict(row);pp=pathlib.Path(rr['path']);assert pp.is_relative_to(ROOT/'inputs')
    rr['path']='/'+str(pp.relative_to(ROOT/'inputs'));canonical.append(rr)
   assert canonical==oldmeta[split]['cases'],'Ordered trajectory bytes or source identity changed'
   for field in ['shape','allowed_cases','temporal_indices','time_normalization']:assert meta[split][field]==oldmeta[split][field]
   h=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==oldmeta[split]['data_sha256']
   path_audit[split]=dict(actual_location_digest=physical,canonical_source_identity_digest=h,trajectories_verified=len(canonical),all_tensor_hashes_equal=True)
   meta[split]['cases']=canonical;meta[split]['data_sha256']=h
  return data,meta
 loader.load_released_data=relocated_load
 a.code_root=CODE;ctx=base.bootstrap(a);ctx['seed']=a.seed;ctx['base_module']=base
 cfg=remapped_read(CODE/'geps_pilot_retry1/CONFIG.json');ctx['args']=argparse.Namespace(nod_source=pathlib.Path(cfg['nod_source']),data_root=pathlib.Path(cfg['data_root']))
 ctx['adaptation_audit']=dict(source_gate_sha256=sha(path),ast_changes=changes,train_budget_seconds=3000,source_optimizer_updates=0,
  random_offsets={str(i):int((offsets==i).sum()) for i in range(1,9)},seed=a.seed,path_relocation_audit=path_audit)
 return ctx,base

def code0_evaluate(a,ctx,base):
 import numpy as np,torch
 import geps_formal_three_seed as g
 source=a.trained;cp=source/'latest.pt';ss=read(source/'SUMMARY.json')
 assert read(source/'EXIT.json')['exit_code']==0 and read(source/'COMPLETE.json')['summary_sha256']==sha(source/'SUMMARY.json')
 assert ss['checkpoint_sha256']==sha(cp)
 enc,shared=base.new_encoder(ctx,a.shared);ck=torch.load(cp,map_location='cpu',weights_only=False)
 assert ck['updates']==1000 and ck['source_sha256']==base.SOURCE_SHA and ck['shared_checkpoint_sha256']==sha(a.shared/'shared_encoder.pt')
 enc.load_state_dict(ck['model']);enc.eval();initial=ctx['impl'].model_sha(enc)
 groups={};allcodes={};allmanifest={};start=time.time()
 for name,ds,denom in g.groups(ctx['args']):
  rows=[];codes=[];manifest=[];errors=[];elapsed=0.
  try:
   for st in range(0,len(ds),8):
    items=[ds[i] for i in range(st,min(st+8,len(ds)))];support=torch.stack([x['cond_u'].transpose(-1,-2) for x in items]).cuda()
    target=torch.stack([x['target_seq'].permute(1,2,0) for x in items]).cuda();times=items[0]['t_idx'].cuda().float()/denom
    assert tuple(support.shape[1:])==(1,401,101) and target.shape==support.shape
    decoder=ctx['impl'].FrozenDecoder(ctx['c'],ctx['source'],len(items))
    with torch.no_grad():
     torch.cuda.synchronize();t=time.monotonic();z=enc(support);prediction=decoder.predict(target[...,0],z,times);torch.cuda.synchronize();elapsed+=time.monotonic()-t
     assert torch.isfinite(z).all() and torch.isfinite(prediction).all() and torch.equal(prediction[...,0],target[...,0])
     e=(prediction-target).square().mean((1,2)).cpu().numpy();zz=z.cpu().numpy()
    decoder.audit()
    for k,item in enumerate(items):
     assert item['cond_case_idx']!=item['pred_case_idx']
     manifest.append(dict(index=st+k,nu_id=int(item['nu_id']),nu_value=float(item['nu_value'].item()),cond_case_idx=int(item['cond_case_idx']),pred_case_idx=int(item['pred_case_idx']),support_sha256=ctx['base'].tensor_digest(support[k]),query_sha256=ctx['base'].tensor_digest(target[k]),t=times.cpu().tolist()))
     rows.append(dict(index=st+k,mse_all101=float(e[k].mean()),mse_future100=float(e[k,1:].mean()),**{'h'+str(h):float(e[k,h]) for h in [1,5,50,100]}));errors.append(e[k]);codes.append(zz[k])
    write(a.output/(name+'_PARTIAL.json'),dict(completed=len(rows),total=len(ds),seconds=elapsed))
  finally:
   if hasattr(ds,'close'):ds.close()
  e=np.asarray(errors);assert e.shape==(len(rows),101)
  groups[name]=dict(n_pairs=len(rows),native_compatible_mse=g.native_reduce([r['mse_all101'] for r in rows]),trajectory_mean_mse=float(e.mean()),future100=float(e[:,1:].mean()),code_adaptation_steps=0,encoder_and_prediction_seconds=elapsed,**{'h'+str(h):float(e[:,h].mean()) for h in [1,5,50,100]})
  write(a.output/(name+'_PAIRS.json'),rows);np.savez_compressed(a.output/(name+'_ERRORS.npz'),errors=e,codes=np.asarray(codes))
  allcodes[name]=np.asarray(codes);allmanifest[name]=manifest
  write(a.output/(name+'_COMPLETE.json'),dict(status='COMPLETE',n_pairs=len(rows),errors_sha256=sha(a.output/(name+'_ERRORS.npz'))))
 # Same frozen encoder: training-only ridge selection, final ID accessibility.
 bank=ctx['data']['train'];traincodes=[]
 with torch.no_grad():
  for st in range(0,360,36):traincodes.extend(enc(bank['curves'][st:st+36].cuda()).cpu().tolist())
 teacher=CODE/f'coda_postrun_mechanisms_three_seed/seed{a.seed}/PROBE_CODES.npz'
 with np.load(teacher) as p:labels=p['train_nu'].copy();cases=p['train_case'].copy()
 # Import only three pure numerical functions; no NOD model or environment.
 path=CODE/'sprii_support_code_formal_v3.py';tree=ast.parse(path.read_text());selected=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['probe','ranks','standardized_geometry']];ns={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
 probe=ns['probe'](np.asarray(traincodes),labels,cases,allcodes['id_test'],[r['nu_value'] for r in allmanifest['id_test']])
 np.savez_compressed(a.output/'PROBE_CODES.npz',train=np.asarray(traincodes),train_nu=labels,train_case=cases,id=allcodes['id_test'],id_nu=[r['nu_value'] for r in allmanifest['id_test']])
 write(a.output/'PROBE.json',probe);write(a.output/'MANIFEST.json',allmanifest)
 assert ctx['impl'].model_sha(enc)==initial;frozen=base.frozen_audit(ctx)
 write(a.output/'SUMMARY.json',dict(status='COMPLETE',seed=a.seed,arm=a.arm,groups=groups,probe=probe,source=frozen,encoder_checkpoint_sha256=sha(cp),encoder_state_unchanged=True,source_optimizer_updates=0,encoder_optimizer_updates=0,
   code_adaptation_steps=0,manifest_sha256=sha(a.output/'MANIFEST.json'),seconds=time.time()-start,historical_report_extension=True,test_read=True,ood_read=True,new_sealed=False))


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','prepare','train','evaluate']);p.add_argument('--seed',type=int,choices=[1234,5678,9012],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--shared',type=pathlib.Path);p.add_argument('--trained',type=pathlib.Path);p.add_argument('--arm',choices=['plain','sprii_cross','random_cross','duplicate_self']);a=p.parse_args()
 a.output.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed bounded phase cap')));signal.setitimer(signal.ITIMER_REAL,3000 if a.phase=='train' else 1800)
 write(a.output/'RUN.json',dict(pid=os.getpid(),time=time.time(),argv=sys.argv,script_sha256=sha(__file__)))
 try:
  ctx,base=context(a);write(a.output/'ADAPTER_AUDIT.json',ctx['adaptation_audit']);write(a.output/'DATA_MANIFEST.json',ctx['meta'])
  if a.phase=='smoke':write(a.output/'SUMMARY.json',ctx['impl'].smoke(ctx['c'],ctx['source'],'cuda:0'))
  elif a.phase=='prepare':base.prepare(ctx,a)
  elif a.phase=='train':base.train(ctx,a)
  else:code0_evaluate(a,ctx,base)
  base.frozen_audit(ctx);write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json')));write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
 except BaseException:
  write(a.output/'FAILED.json',dict(traceback=traceback.format_exc()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
