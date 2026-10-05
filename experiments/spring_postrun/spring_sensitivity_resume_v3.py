"""External wrapper: fixed source recipe, one-axis weights, non-updating gradient diagnostics."""
import argparse, ast, copy, hashlib, inspect, json, os, pathlib, sys, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=pathlib.Path(os.environ.get('SPRII_ROOT', '.'))
NATIVE=ROOT/'benchmarks/springworld/native'
BANK=ROOT/'data/springworld/research_bank'
sys.path[:0]=[str(NATIVE/p) for p in ('','src','a_src','extension')]
import numpy as np
import torch
from persistent_jepa import poke_model
import native_training
from persistbench.envs.visual_elastic_coupling import a_pairing, a_pretraining as engine
from persistbench.envs.visual_elastic_coupling.a_head_fitting import _math_profile
from persistbench.envs.visual_elastic_coupling.a_head_features import model_state_sha256
from persistent_jepa.losses import SIGReg

def write(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class Diagnostic:
 def __init__(self,path,interval=100):self.path=path;self.step=0;self.interval=interval;self.rows=[]
 def __call__(self,model,align,cross):
  self.step+=1
  if self.step%self.interval:return
  params=tuple(model.persistent.parameters())
  cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
  prior=[None if p.grad is None else p.grad.clone() for p in params]
  ga=torch.autograd.grad(align,params,retain_graph=True,allow_unused=False)
  gx=torch.autograd.grad(cross,params,retain_graph=True,allow_unused=False)
  a=torch.cat([x.detach().float().reshape(-1) for x in ga]);x=torch.cat([x.detach().float().reshape(-1) for x in gx])
  an=a.norm();xn=x.norm();cos=(a@x)/(an*xn).clamp_min(1e-30)
  assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
  assert all((old is None and p.grad is None) or (old is not None and torch.equal(old,p.grad)) for p,old in zip(params,prior))
  row=dict(step=self.step,parameter_scope='model.persistent / g_p',parameter_count=sum(p.numel() for p in params),cosine=float(cos),align_norm=float(an),cross_norm=float(xn),align_loss=float(align.detach()),cross_loss=float(cross.detach()),zero_norm=bool(an==0 or xn==0),rng_unchanged=True,parameter_grad_unchanged=True)
  assert all(np.isfinite(row[k]) for k in ('cosine','align_norm','cross_norm','align_loss','cross_loss'))
  self.rows.append(row)
  if self.path:
   with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def instrument(diag):
 # Add exactly one callback to the original function's live graph. No second forward.
 source=inspect.getsource(poke_model.poke_objective);tree=ast.parse(source);function=tree.body[0]
 assert isinstance(function.body[-1],ast.Return)
 function.body.insert(-1,ast.parse('_diagnostic(model, persist_loss, cross_loss)').body[0])
 ast.fix_missing_locations(tree);ns=dict(poke_model.__dict__);ns['_diagnostic']=diag
 exec(compile(tree,'<spring_gradient_diagnostic>','exec'),ns)
 return ns['poke_objective'],hashlib.sha256(source.encode()).hexdigest()

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train','audit'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--lp',type=float,default=1);p.add_argument('--lx',type=float,default=.1);p.add_argument('--output',required=True);p.add_argument('--resume-commit',required=True);p.add_argument('--resume-commit-sha256',required=True);a=p.parse_args()
 assert (a.lp,a.lx) in [(.25,.1),(1,.1),(4,.1),(1,.025),(1,.4)]
 out=pathlib.Path(a.output);assert not out.exists(),str(out)
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 policy=json.loads((NATIVE/'NIGHT_POLICY.json').read_text());values=policy['training_spec']
 assert values['steps']==10000 and values['pairs_per_batch']==48 and values['history_frames']==96
 values.update(model_seed=a.seed,sampling_seed=a.seed,stochastic_seed=a.seed);values['milestone_steps']=tuple(values['milestone_steps'])
 spec=engine.PretrainingSpec(**values);snapshot=sha(BANK/'BANK_SNAPSHOT.json');assert snapshot==policy['bank_snapshot_sha256']
 a_pairing.CONFIGURATIONS['Both']=('B3','G3','Independent',a.lp,a.lx)
 device=torch.device('cuda:0')
 registration=dict(schema='spring-sensitivity-gradient-resume-v3',fast_batch_sha256=sha(ROOT/'spring_fast_batch.py'),seed=a.seed,lambda_p=a.lp,lambda_x=a.lx,steps=10000,checkpoint_selection='fixed10000',gradient_interval=100,gradient_parameters='model.persistent',test_read=False,original_source_unchanged=True,wrapper_sha256=sha(__file__),bank_sha256=snapshot,torch=str(torch.__version__),python=sys.version,gpu=torch.cuda.get_device_name(),original_objective_sha256=hashlib.sha256(inspect.getsource(poke_model.poke_objective).encode()).hexdigest())
 if a.mode=='smoke':
  out.mkdir(parents=True)
  bank=engine.PretrainingBank(BANK,snapshot_sha256=snapshot,spec=spec);bank.verify_all(2)
  plan=bank.schedule.sweep(0,'Both');batch,_=native_training.make_batch(bank.schedule,plan,0,BANK);batch=batch.to(device)
  results=[];started=time.monotonic()
  with _math_profile(device):
   for enabled in (False,True):
    model=engine._new_model('Both',spec,device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.05,foreach=False,fused=False);sig=SIGReg().to(device)
    diag=Diagnostic(None,interval=1);objective=instrument(diag)[0] if enabled else poke_model.poke_objective
    opt.zero_grad(set_to_none=True);model.begin_train_step()
    with engine._step_rng(a.seed,0,device):
     with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,batch,sig,sigreg_weight=.02,lambda_p=a.lp,lambda_x=a.lx)
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();model.finish_train_step(split='train')
     rng=torch.cuda.get_rng_state().cpu().clone()
    def cpu(x):
     if torch.is_tensor(x):return x.detach().cpu().clone()
     if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
     if isinstance(x,list):return [cpu(v) for v in x]
     return x
    results.append(dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),rng=rng,loss=loss.detach().cpu(),metrics=cpu(metrics)))
    del model,opt,sig,loss,metrics;torch.cuda.empty_cache()
  assert equal(results[0],results[1]),'diagnostics changed model/Adam/loss/metrics/RNG'
  write(out/'COMPLETE.json',dict(status='PASS',**registration,full_batch=True,model_optimizer_loss_metrics_rng_equal=True,seconds=time.monotonic()-started,diagnostics=diag.rows))
  print(json.dumps(dict(status='PASS',seconds=time.monotonic()-started,diagnostics=diag.rows)),flush=True);return
 # Preserve the interrupted attempt; only its last committed prefix is accepted.
 parent=pathlib.Path(a.resume_commit).parent
 old_run=json.loads((parent/'RUN.json').read_text())
 assert old_run['configuration']==a_pairing.configuration('Both')
 assert old_run['spec']==json.loads(json.dumps(spec.record()))
 assert old_run['runtime']==engine._runtime_signature(device)
 assert old_run['binding']['source_fingerprint']==engine.source_fingerprint()
 assert old_run['binding']['A_source_fingerprint']==engine.a_source_fingerprint()
 assert old_run['binding']['bank_snapshot_sha256']==snapshot
 prepared=engine._prepare_resume(a.resume_commit,a.resume_commit_sha256,old_run,spec)
 step=prepared['checkpoint']['step'];assert 0<step<10000
 oldgrad=parent.with_name(parent.name+'_GRADIENTS.jsonl')
 raw=oldgrad.read_bytes();lines=raw.splitlines(keepends=True);prefix=[];rows=[]
 for line in lines:
  try:row=json.loads(line)
  except json.JSONDecodeError:break # A torn final uncommitted diagnostic is not evidence.
  if row['step']>step:break
  assert row['rng_unchanged'] and row['parameter_grad_unchanged']
  assert all(np.isfinite(row[k]) for k in ('cosine','align_norm','cross_norm','align_loss','cross_loss'))
  prefix.append(line);rows.append(row)
 assert [r['step'] for r in rows]==list(range(100,step+1,100))
 prefix=b''.join(prefix)
 model=engine._new_model('Both',spec,torch.device('cpu'))
 assert model_state_sha256(model)==prepared['checkpoint']['initial_model_sha256']
 model.load_state_dict(prepared['checkpoint']['model'],strict=True)
 assert model_state_sha256(model)==prepared['checkpoint']['model_state_sha256']
 assert int(model.observation.norm.num_batches_tracked)==step
 opt=torch.optim.AdamW(model.parameters(),lr=spec.learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=spec.weight_decay,foreach=False,fused=False)
 opt.load_state_dict(prepared['checkpoint']['optimizer'])
 assert equal(opt.state_dict(),prepared['checkpoint']['optimizer'])
 assert all(int(x['step'])==step for x in opt.state.values())
 registration['resume']=dict(parent=str(parent),parent_commit=str(a.resume_commit),parent_commit_sha256=a.resume_commit_sha256,
  resumed_from_step=step,new_optimizer_updates=10000-step,gradient_prefix_sha256=hashlib.sha256(prefix).hexdigest(),gradient_prefix_rows=len(rows),
  parent_gradient_file_sha256=sha(oldgrad),parent_wrapper_sha256=sha(parent.with_name(parent.name+'_WRAPPER.json')),
  original_attempt_preserved=True,model_and_adam_restored=True,original_runtime_equal=True)
 if a.mode=='audit':
  write(out.with_name(out.name+'_RESUME_AUDIT.json'),dict(status='PASS',registration=registration,full_checkpoint_and_batch_prefix_verified=True,source_updates_added=0,time=time.time()))
  print(json.dumps(dict(status='PASS',step=step,gradient_rows=len(rows),optimizer_states=len(opt.state))),flush=True);return
 del prepared,model,opt
 # Native engine creates the source folder and binds all original files/checkpoints.
 out.parent.mkdir(parents=True,exist_ok=True)
 write(out.with_name(out.name+'_WRAPPER.json'),registration)
 from spring_fast_batch import install
 assert json.loads((ROOT/'receipts/FAST_BATCH_PARITY.json').read_text())['status']=='PASS'
 _,engine.make_batch,_=install()
 diag=Diagnostic(out.with_name(out.name+'_GRADIENTS.jsonl'))
 assert not diag.path.exists()
 diag.path.write_bytes(prefix);diag.step=step;diag.rows=rows
 observed,_=instrument(diag)
 def objective(model,batch,sigreg,config):
  assert config==a_pairing.configuration('Both') and sigreg.num_directions==1024
  return observed(model,batch,sigreg,sigreg_weight=.02,lambda_p=config['lambda_p'],lambda_x=config['lambda_x'])
 engine.objective=objective
 result=engine.train_pretraining(BANK,out,name='Both',spec=spec,bank_snapshot_sha256=snapshot,device=device,workers=2,progress=lambda e:print(json.dumps(e),flush=True),resume_commit=a.resume_commit,resume_commit_sha256=a.resume_commit_sha256)
 assert diag.step==10000 and len(diag.rows)==100
 write(out.with_name(out.name+'_DIAGNOSTICS_COMPLETE.json'),dict(status='COMPLETE',rows=100,sha256=sha(diag.path),registration=registration,result=result))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
