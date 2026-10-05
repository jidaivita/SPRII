"""Compare released NOD and the shared fork on actual released training batches."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_nod_clean as fork
from ngs.utils import BurgersPairedDataset
from triple_data import BurgersTripleDataset

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

def diff(a,b):
    if torch.is_tensor(a): return float((a.detach().cpu()-b.detach().cpu()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys(); return max([diff(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b); return max([diff(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b,(a,b); return 0.

def checksum(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def rng(): return (torch.get_rng_state(),torch.cuda.get_rng_state())
def restore(s): torch.set_rng_state(s[0]); torch.cuda.set_rng_state(s[1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--steps',type=int,default=10); args=ap.parse_args()
    root=args.root; original=root/'third_party/nod_original/code/Burgers'
    native=module('untouched_train',original/'NOD/train.py'); utils=module('untouched_utils',original/'ngs/utils.py')
    fork.set_seed(1234); device=torch.device('cuda'); torch.set_num_threads(4)
    ds0=utils.BurgersPairedDataset(data_root=str(root/'data/burgers_normalized'),split='train',space_stride=1,seed=1234)
    ds1=BurgersTripleDataset(BurgersPairedDataset(data_root=str(root/'data/burgers_normalized'),split='train',seed=1234),1234)
    assert ds0.n_t==ds1.n_t==101
    loaders=[iter(DataLoader(d,batch_size=8,shuffle=False)) for d in (ds0,ds1)]
    a=native.NGS_INR(**native.MODEL_DEFAULTS).to(device); b=fork.NGS_INR(**fork.MODEL_DEFAULTS).to(device)
    checkpoint=root/'receipts/gate_initialization.pth'; torch.save(a.state_dict(),checkpoint)
    b.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); initial=checksum(a)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=1e-4); ob=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4)
    pred={}
    for name,m in [('original',a),('fork',b)]:
        fn=m.predict_queries
        def wrapped(*args,_name=name,_fn=fn,**kw):
            result=_fn(*args,**kw); pred[_name]=result.detach().cpu(); return result
        m.predict_queries=wrapped
    rows=[]
    for step in range(args.steps):
        x,y=next(loaders[0]),next(loaders[1]); keys=['cond_u','pred_u0','target_seq','t_idx','cond_case_idx','pred_case_idx']
        for key in keys: assert torch.equal(x[key],y[key]),key
        assert torch.all(y['align_case_idx']!=y['cond_case_idx']) and torch.all(y['align_case_idx']!=y['pred_case_idx'])
        state=rng()
        ma=native.run_epoch(a,[x],torch.nn.MSELoss(),device,oa,1.,100.,8192)
        restore(state)
        mb=fork.run_epoch(b,[y],torch.nn.MSELoss(),device,ob,1.,100.,8192,lambda_align=0.)
        rows.append(dict(step=step,base_loss_diff=abs(ma['loss']-mb['loss']),prediction_diff=diff(pred['original'],pred['fork']),parameter_diff=diff(a.state_dict(),b.state_dict()),gradient_diff=diff({k:p.grad for k,p in a.named_parameters()},{k:p.grad for k,p in b.named_parameters()}),optimizer_diff=diff(oa.state_dict(),ob.state_dict()),original_checksum=checksum(a),fork_checksum=checksum(b),A=y['cond_case_idx'].tolist(),B=y['pred_case_idx'].tolist(),C=y['align_case_idx'].tolist()))
    val=utils.BurgersPairedDataset(data_root=str(root/'data/burgers_normalized'),split='eval',space_stride=1,seed=99)
    batch=next(iter(DataLoader(val,batch_size=2,shuffle=False)))
    # Native/fork evaluation on identical examples and selected weights; no test/OOD opened.
    ev0=native.run_epoch(a,[batch],torch.nn.MSELoss(),device,None,1.,100.,None)
    ev1=fork.run_epoch(b,[batch],torch.nn.MSELoss(),device,None,1.,100.,None)
    per=[]
    for i in range(2):
        one={k:(v[i:i+1] if torch.is_tensor(v) else v) for k,v in batch.items()}
        u=native.run_epoch(a,[one],torch.nn.MSELoss(),device,None,1.,100.,None)
        v=fork.run_epoch(b,[one],torch.nn.MSELoss(),device,None,1.,100.,None)
        per.append(abs(u['mse']-v['mse']))
    passed=all(all(v<=1e-6 for k,v in r.items() if k.endswith('_diff')) for r in rows) and abs(ev0['mse']-ev1['mse'])<=1e-6 and max(per)<=1e-6
    out=dict(passed=passed,steps=args.steps,device=str(device),real_data=True,full_model=True,initial_checksum=initial,official_source_sha256=hashlib.sha256((original/'NOD/train.py').read_bytes()).hexdigest(),path_adapter={'space_stride':1,'reason':'released grid already 401'},time_denominator=100.,rows=rows,evaluator={'native':ev0,'fork':ev1,'per_example_abs_diff':per,'split':'validation'},final_test_read=False)
    (root/'receipts/equivalence_real_v2.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); assert passed
if __name__=='__main__': main()
