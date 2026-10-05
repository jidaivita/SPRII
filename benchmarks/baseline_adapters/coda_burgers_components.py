#!/usr/bin/env python3
"""Read-only CoDA components with a singleton spatial axis for 1D Burgers.

Only the dependency-free numerical definitions are loaded from the immutable
release. No replacement hypernetwork, activation, integration or optimizer.
This is an external Burgers adaptation, not an official Burgers configuration.
"""
import ast, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchdiffeq import odeint

SOURCE=Path(__file__).resolve().parent/'third_party/coda_original'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_definitions():
    scope=dict(torch=torch,nn=nn,init=init,F=F,np=np,copy=copy,odeint=odeint)
    selected={'utils.py':['count_parameters','set_requires_grad','init_weights'],
              'network.py':['GroupSwish','GroupActivation','HyperEnvNet','GroupConv'],
              'ode_model.py':['Derivative','Forecaster']}
    for filename,names in selected.items():
        tree=ast.parse((SOURCE/filename).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        # Execute the original numerical AST verbatim; unused plotting imports
        # and unused environment branches do not require optional dependencies.
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/filename),'exec'),scope)
    return scope

OFFICIAL=load_definitions()
def build(n_env,device='cpu',factor=1.0):
    model=OFFICIAL['Forecaster'](state_c=1,hidden_c=64,code_c=2,n_env=n_env,factor=factor,
            method='rk4',nl='swish',dataset='gray',is_ode=False,size=401,is_layer=False,device=device).to(device)
    OFFICIAL['init_weights'](model,init_type='default',init_gain=1.0)
    # The release's ghost template is in a dict, hence not traversed by .to().
    # Derivative.ghost_structure and this template are the same registered object.
    assert next(model.derivative.net_leaf.nets['ghost_structure'].parameters()).device==torch.device(device)
    return model

def forecast(model,curves,times,epsilon=0):
    # curv: batch x environment x 401 x 101 -> official B x E x T x 1 x X
    assert curves.ndim==4 and curves.shape[1]==model.derivative.codes.shape[0]
    packed=curves.permute(0,1,3,2).unsqueeze(3)
    model.derivative.update_ghost()
    return model(packed,times,epsilon=epsilon).squeeze(3).permute(0,1,3,2)

def regularizer(model):
    matrix=model.derivative.net_hyper.weight;codes=model.derivative.codes
    return 1e-6*torch.norm(matrix,dim=1).sum()+1e-4*torch.norm(codes,dim=0).square().sum()

def adapted(source,n_env):
    device=next(source.parameters()).device
    target=build(n_env,str(device),source.derivative.net_root.factor)
    target.derivative.net_root.load_state_dict(source.derivative.net_root.state_dict(),strict=True)
    target.derivative.net_hyper.load_state_dict(source.derivative.net_hyper.state_dict(),strict=True)
    for p in target.parameters():p.requires_grad_(False)
    target.derivative.codes.requires_grad_(True)
    assert torch.count_nonzero(target.derivative.codes)==0
    return target

def shared_digest(model):
    h=hashlib.sha256()
    for prefix,m in [('root',model.derivative.net_root),('hyper',model.derivative.net_hyper)]:
        for n,v in sorted(m.state_dict().items()):h.update((prefix+n).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def smoke():
    torch.set_num_threads(1);torch.manual_seed(1234);np.random.seed(1234)
    model=build(2);times=torch.linspace(0,.02,3)
    truth=torch.randn(1,2,17,3)*.1
    output=forecast(model,truth,times);assert output.shape==truth.shape and torch.isfinite(output).all()
    loss=(output-truth).square().mean()+regularizer(model);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.derivative.codes.grad).all()
    assert model.derivative.codes.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        changed=truth.clone();changed[...,1:]+=100
        assert torch.equal(output,forecast(model,changed,times))
        changed=truth.clone();changed[:,1,:,0]+=1
        changed_out=forecast(model,changed,times)
        assert torch.equal(output[:,0],changed_out[:,0]),'cross-environment coupling'
    target=adapted(model,1);before=shared_digest(target)
    optimizer=torch.optim.Adam([target.derivative.codes],lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);l=(forecast(target,truth[:,:1],times)-truth[:,:1]).square().mean();l.backward();optimizer.step()
    assert shared_digest(target)==before and target.derivative.codes.abs().sum()>0
    return dict(status='PASS',original_definition_sources={n:sha(SOURCE/n) for n in ['utils.py','network.py','ode_model.py']},
                original_forecaster_rk4=True,singleton_spatial_axis=True,finite_gradient=True,
                nonzero_code_gradient=True,future_query_perturbation_unchanged=True,group_independence=True,
                support_adaptation_changes_codes_only=True,initial_codes_zero=True,no_training_or_test_data=True)

if __name__=='__main__':print(json.dumps(smoke(),indent=2))
