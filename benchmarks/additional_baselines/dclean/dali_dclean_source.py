#!/usr/bin/env python3
"""Frozen recipe DALI context/forward component on existing D-Clean histories.

PyTorch adaptation, not untouched native Dreamer-DALI. No RSSM/cross-modal loss.
No source checkpoint selection: retain fixed final update (gate1000/formal20000).
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dali_context_torch import DALIContext
sys.dont_write_bytecode=True

OFFICIAL_COMMIT='34374fbea258748b03e31977a68693ab040fab72'
OFFICIAL_NETS='0233a0d7aedce5f137a29a5ce3b49d100db1c0d1322d1c8b6238e66235539b9d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def verify_official(root):
    p=Path(root)/'dreamerv3_compat/dreamerv3/nets.py';assert sha(p)==OFFICIAL_NETS
    return str(p)

class Model(nn.Module):
    def __init__(self,normalization):
        super().__init__()
        for k,v in normalization.items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
        self.context=DALIContext(4,2,24)
    def normalized(self,x,u):
        assert x.shape[1:]==(24,4) and u.shape[1:]==(23,2)
        return (x-self.state_mean)/self.state_scale,torch.cat([u/self.action_scale,torch.zeros_like(u[:,:1])],1)
    def encode(self,x,u):
        x,u=self.normalized(x,u);return self.context.encode(x,u)
    def objective(self,x,u,lengths):
        x,u=self.normalized(x,u);return self.context.prefix_loss(x,u,lengths)

def build_model(config,official_root):
    verify_official(official_root)
    assert config['context_dim']==8 and config['normalization_source']=='training states/actions only'
    assert config['component_code_sha256']==sha(Path(__file__).with_name('dali_context_torch.py'))
    return Model(config['normalization'])

def make_batch(data,sp,device):
    # An independent implementation checked against the unchanged helper.batch.
    parts=[]
    for ri,ti in ((1,2),(3,4)):
        s,r,t=sp[:,0,None],sp[:,ri,None],sp[:,ti,None]
        assert np.all(t>=23) and np.all(sp[:,1]!=sp[:,3])
        parts.append((data['states'][s,r,t+np.arange(-23,1)],data['actions'][s,r,t+np.arange(-23,0)]))
    return tuple(torch.as_tensor(np.concatenate([p[k] for p in parts]),device=device) for k in range(2))

def dev_metrics(model,helper,device):
    # Existing selection100 only. Report100/test never used by source optimizer/selection.
    protocol=helper.read(helper.ROOT/'PROTOCOL.json');d=helper.data('val')
    selection=set(protocol['selection_ids']);systems=np.array([i for i,x in enumerate(d['system_ids']) if int(x) in selection])
    assert len(systems)==100
    pairs=np.array([[s,0,t,1,t] for s in systems for t in (23,31)],dtype=int)
    errs={h:[] for h in (1,4,16)};losses=[];zs=[];model.eval()
    with torch.no_grad():
        for sp in np.array_split(pairs,10):
            x,u=make_batch(d,sp,device);xn,un=model.normalized(x,u);z=model.encode(x,u);zs.append(z.cpu().numpy())
            losses.append(model.context.compute_loss(xn,un).cpu().numpy())
            fut=[];target=[]
            for ri,ti in ((1,2),(3,4)):
                s,r,t=sp[:,0,None],sp[:,ri,None],sp[:,ti,None]
                fut.append(d['actions'][s,r,t+np.arange(16)]);target.append(d['states'][s,r,t+np.arange(1,17)])
            fu=torch.as_tensor(np.concatenate(fut),device=device)/model.action_scale
            y=torch.as_tensor(np.concatenate(target),device=device);pred=xn[:,-1]
            for t in range(16):
                pred=model.context.forward_model(pred,fu[:,t],z)
                if t+1 in errs:errs[t+1].extend((pred*model.state_scale+model.state_mean-y[:,t]).square().mean(-1).cpu().tolist())
    z=np.concatenate(zs)
    out={'within_history_normalized_forward_mse':float(np.concatenate(losses).mean()),'cases':400,
         'fixed_context_autoregressive_raw_state_mse':{str(h):float(np.mean(v)) for h,v in errs.items()},
         'context_std_per_dim':z.std(0).tolist(),'context_finite':bool(np.isfinite(z).all()),
         'systems':100,'split':'existing selection100 validation systems','selection':'none; fixed final budget',
         'closed_loop':False,'test_read':False}
    assert all(np.isfinite(v) for v in out['fixed_context_autoregressive_raw_state_mse'].values())
    return out

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    helper=load(a.data_helper,'dali_dclean_data_helper');official=verify_official(a.official_root)
    assert (helper.ROOT/'PROTOCOL.json').is_file()
    protocol_sha=sha(helper.ROOT/'PROTOCOL.json');contract=helper.contract();assert sha(helper.ROOT/'PROTOCOL.json')==protocol_sha
    helper.seed_all(a.seed);device=torch.device(a.device)
    config={'method':'DALI released-components D-Clean adaptation','official_commit':OFFICIAL_COMMIT,'official_nets_sha256':sha(official),
        'code_sha256':sha(__file__),'component_code_sha256':sha(Path(__file__).with_name('dali_context_torch.py')),
        'data_helper_sha256':sha(a.data_helper),'source_seed':a.seed,'steps':a.steps,'context_dim':8,
        'normalization':helper.stats(),'normalization_source':'training states/actions only',
        'data_hashes':{p:sha(helper.DATA/p) for p in ['manifest.json','train.npz','val.npz']},
        'architecture':{'type':'transformer','attention_width':256,'heads':1,'forward_width':128,'forward_layers':2,'context_dim':8,'norm_epsilon':.001,'symlog_inputs':False},
        'optimizer':{'name':'Adam','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'clip':1000},
        'history_states':24,'history_actions':23,'terminal_action':'zero; unused by forward target',
        'batch_systems':48,'batch_histories':96,'prefix_sampling':'independent uniform t=1..23 per history; edge-left-pad; 23/24 loss multiplier',
        'objective':'unbiased estimator of released WorldModel prefix forward-dynamics loss sum/T',
        'omitted_native_components':['Dreamer RSSM','Dreamer reconstruction/actor/critic','cross-modal context/RSSM loss'],
        'input_adaptation':'train-normalized state4/action2; fixed 24-token sequence replacing native sequence length',
        'selection':'fixed final; no early stopping or checkpoint selection','physical_parameter_input':False,'test_read':False,'closed_loop':False}
    write(out/'CONFIG.json',config);config_sha=sha(out/'CONFIG.json');m=build_model(config,a.official_root).to(device)
    b=make_batch(helper.data('train'),helper.specs(a.seed,1),device)
    hb=helper.batch('train',helper.specs(a.seed,1),device=device)
    for i in range(2):torch.testing.assert_close(b[i],hb[i],rtol=0,atol=0)
    with torch.no_grad():assert m.encode(*b).shape==(96,8)
    write(out/'INTERFACE_SMOKE.json',{'status':'PASS','same_actual_helper_histories':True,'history_shape':[96,24,4],
         'action_shape':[96,23,2],'gamma_input':False,'config_sha256':config_sha})
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,eps=1e-8)
    prefix_rng=np.random.default_rng(202609250000+a.seed)
    write(out/'RUN.json',{'pid':os.getpid(),'config_sha256':config_sha,'parameters':sum(p.numel() for p in m.parameters()),'source_updates':a.steps,'unix_time':time.time()})
    initial=dev_metrics(m,helper,device);write(out/'INITIAL_DEV.json',initial)
    began=time.monotonic();m.train()
    with (out/'train.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            b=make_batch(helper.data('train'),helper.specs(a.seed,step),device)
            lengths=torch.as_tensor(prefix_rng.integers(1,24,96),device=device)
            opt.zero_grad(set_to_none=True);loss=m.objective(*b,lengths);assert torch.isfinite(loss)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1000,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0 or step==a.steps:
                row={'step':step,'loss':float(loss),'gradient_norm':float(gn),'seconds':time.monotonic()-began}
                log.write(json.dumps(row)+'\n');log.flush();write(out/'PROGRESS.json',row)
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'step':a.steps,'source_seed':a.seed,
        'config_sha256':config_sha,'torch_rng':torch.get_rng_state(),'numpy_prefix_rng':prefix_rng.bit_generator.state},out/'final.pt')
    final=dev_metrics(m,helper,device)
    summary={'status':'COMPLETE','steps':a.steps,'source_seed':a.seed,'config_sha256':config_sha,'final_checkpoint_sha256':sha(out/'final.pt'),
             'training_seconds':time.monotonic()-began,'initial_dev':initial,'final_dev':final,'closed_loop':False,'test_read':False}
    write(out/'SUMMARY.json',summary);verify_official(a.official_root)
    assert sha(helper.ROOT/'PROTOCOL.json')==protocol_sha
    write(out/'COMPLETE.json',{'status':'COMPLETE','summary_sha256':sha(out/'SUMMARY.json'),'final_checkpoint_sha256':sha(out/'final.pt')})
    write(out/'EXIT.json',{'exit_code':0,'unix_time':time.time()});print(json.dumps(summary),flush=True)

if __name__=='__main__':
    here=Path(__file__).resolve().parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    p.add_argument('--steps',type=int,choices=(1000,20000),default=20000);p.add_argument('--device',default='cuda:0')
    p.add_argument('--data-helper',default='shared/dclean_external.py')
    p.add_argument('--official-root',default=str(here.parent/'vendor/DALI'));a=p.parse_args()
    # Refuse before exception reporting so an old successful directory remains immutable.
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise SystemExit('Nonempty source output; no implicit overwrite/resume.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:run(a)
    except Exception as e:
        write(out/'FAILED.json',{'error':repr(e),'traceback':traceback.format_exc()});write(out/'EXIT.json',{'exit_code':1});raise
