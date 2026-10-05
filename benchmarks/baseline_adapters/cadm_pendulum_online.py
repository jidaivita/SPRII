#!/usr/bin/env python3
"""Finite ONLINE Pendulum Vanilla/CaDM deterministic E=1 PyTorch adaptation.

No training runs at import. `smoke` does forward/backward and planner throughput
only; `run --profile gate` is a separate 800-interaction engineering run.
Formal profile is 20 iterations x10 episodes x200 steps =40k training interactions.
Published deterministic E=1 is supported; default probabilistic E=5 is not ported.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


envcode=module(HERE/'cadm_pendulum_engineering_gate.py','cadm_verified_pendulum')
components=module(HERE/'cadm_dclean_pilot.py','cadm_audited_dense_components')
sha,write,Dense=components.sha,components.write,components.Dense
HISTORY=10
GRID=[.75,.8,.85,.9,.95,1.,1.05,1.1,1.15,1.2,1.25]
GRIDS={'ID':(GRID,GRID),'OOD_c0':([.2,.4],[1.6,1.8]),'OOD_c1':([.5,.7],[1.3,1.5]),
       'OOD_c2':([1.3,1.5],[.5,.7]),'OOD_c3':([1.6,1.8],[.2,.4])}


class Context(nn.Module):
    def __init__(self):
        super().__init__();dims=[40,256,128,64,10]
        self.layers=nn.ModuleList([Dense(a,b) for a,b in zip(dims[:-1],dims[1:])])
    def forward(self,x):
        for layer in self.layers[:-1]:x=torch.relu(layer(x))
        return self.layers[-1](x)
    def regularizer(self):
        return sum(.5*c*l.weight.square().sum() for c,l in zip(components.CONTEXT_DECAYS,self.layers))


class Dynamics(nn.Module):
    def __init__(self,context_dim):
        super().__init__();dims=[4+context_dim,200,200,200,200]
        self.layers=nn.ModuleList([Dense(a,b) for a,b in zip(dims[:-1],dims[1:])])
        self.mu=Dense(200,3);self.logvar=Dense(200,3)
    def forward(self,x):
        for layer in self.layers:x=torch.nn.functional.silu(layer(x))
        return self.mu(x)
    def regularizer(self):
        return sum(.5*c*l.weight.square().sum() for c,l in zip(components.DYNAMICS_DECAYS[:-1],self.layers))+.5*components.DYNAMICS_DECAYS[-1]*(self.mu.weight.square().sum()+self.logvar.weight.square().sum())


class Model(nn.Module):
    def __init__(self,method):
        super().__init__();assert method in ('Vanilla','CaDM');self.method=method
        self.forward_model=Dynamics(0 if method=='Vanilla' else 10)
        if method=='CaDM':self.context=Context();self.backward_model=Dynamics(10)
        for name,dim in [('obs',3),('act',1),('delta',3),('back_delta',3),('cp_obs',30),('cp_act',10)]:
            self.register_buffer(name+'_mean',torch.zeros(dim));self.register_buffer(name+'_std',torch.ones(dim))
    def norm(self,x,name):return (x-getattr(self,name+'_mean'))/(getattr(self,name+'_std')+1e-10)
    def encode(self,cp_obs,cp_act):
        assert cp_obs.shape[-2:]==(10,3) and cp_act.shape[-2:]==(10,1)
        if self.method=='Vanilla':return cp_obs.new_zeros((len(cp_obs),0))
        return self.context(torch.cat((self.norm(cp_obs.flatten(1),'cp_obs'),self.norm(cp_act.flatten(1),'cp_act')),-1))
    def next_observation(self,obs,act,z):
        output=self.forward_model(torch.cat((self.norm(obs,'obs'),self.norm(act,'act'),z),-1))
        return obs+output*(self.delta_std+1e-10)+self.delta_mean
    def loss(self,obs,act,nxt,cp_obs,cp_act):
        z=self.encode(cp_obs,cp_act)
        pred=self.forward_model(torch.cat((self.norm(obs,'obs'),self.norm(act,'act'),z),-1))
        forward=(pred-self.norm(nxt-obs,'delta')).square().mean()
        backward=forward.new_zeros(());reg=self.forward_model.regularizer()
        if self.method=='CaDM':
            back=self.backward_model(torch.cat((self.norm(nxt,'obs'),self.norm(act,'act'),z),-1))
            backward=(back-self.norm(obs-nxt,'back_delta')).square().mean()
            reg=reg+self.context.regularizer()+self.backward_model.regularizer()
        # Future SPRII hook must use independent observed donor histories only.
        # It is intentionally not activated in this two-baseline engineering gate.
        return forward+.5*backward+reg,dict(forward=forward,backward=backward,l2=reg)


def update_history(cp_obs,cp_act,counts,obs,action,nxt):
    """In-place, left-filled then sliding; reset caller allocates fresh arrays."""
    for i,n in enumerate(counts):
        if n<10:cp_obs[i,n]=nxt[i]-obs[i];cp_act[i,n]=action[i]
        else:
            cp_obs[i,:-1]=cp_obs[i,1:];cp_act[i,:-1]=cp_act[i,1:]
            cp_obs[i,-1]=nxt[i]-obs[i];cp_act[i,-1]=action[i]
    counts+=1


class TorchCEM:
    def __init__(self,batch,device,seed):
        self.batch,self.device=batch,device
        self.previous=torch.zeros((batch,30,1),device=device)
        self.rng=torch.Generator(device=device).manual_seed(seed)
    @torch.inference_mode()
    def plan(self,model,obs,cp_obs,cp_act):
        model.eval();b=self.batch
        obs=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        cp_obs=torch.as_tensor(cp_obs,dtype=torch.float32,device=self.device)
        cp_act=torch.as_tensor(cp_act,dtype=torch.float32,device=self.device)
        z0=model.encode(cp_obs,cp_act);context_dim=z0.shape[-1]
        z=z0[:,None].expand(b,200,context_dim).reshape(b*200,context_dim)
        mean=self.previous.clone();var=torch.full_like(mean,.25)
        for _ in range(5):
            constrained=torch.minimum(torch.minimum(((mean+1)/2).square(),((1-mean)/2).square()),var)
            # Inverse-CDF sample is the same truncated distribution as TF rejection.
            u=torch.rand((b,200,30,1),device=self.device,generator=self.rng)
            noise=math.sqrt(2.)*torch.erfinv(2*(.02275013194817921+u*.9544997361036416)-1)
            actions=mean[:,None]+constrained.sqrt()[:,None]*noise
            state=obs[:,None].expand(b,200,3).reshape(b*200,3)
            score=torch.zeros(b*200,device=self.device)
            for t in range(30):
                act=actions[:,:,t].reshape(b*200,1)
                theta=torch.atan2(state[:,1],state[:,0]);theta=(theta+math.pi)%(2*math.pi)-math.pi
                score-=theta.square()+.1*state[:,2].square()+.001*act[:,0].square()
                state=model.next_observation(state,act,z)
            assert torch.isfinite(score).all(),'Nonfinite learned planner rollout; no hidden oracle fallback.'
            inds=score.reshape(b,200).topk(50,dim=1).indices
            elites=actions.gather(1,inds[:,:,None,None].expand(b,50,30,1))
            mean=.1*mean+.9*elites.mean(1);var=.1*var+.9*elites.var(1,unbiased=False)
        self.previous[:,:-1]=mean[:,1:];self.previous[:,-1]=0
        result=mean[:,0].cpu().numpy();assert np.max(np.abs(result))<=1.000001
        return result


def manifest(seed,iteration,count,group='ID',evaluation=False):
    rows=[]
    for epi in range(count):
        e_seed=(900000 if evaluation else 10000)+seed*1000000+iteration*100+epi
        m,l=GRIDS[group];env=envcode.Pendulum(e_seed,m,l);env.reset()
        rows.append(dict(seed=e_seed,group=group,mass=env.mass,length=env.length,initial_state=env.state.tolist(),episode=epi))
    return rows


def sample(model,rows,device,seed,random_action=False):
    envs=[]
    for row in rows:
        env=envcode.Pendulum(row['seed'],[row['mass']],[row['length']]);env.state=np.asarray(row['initial_state']).copy()
        env.mass=row['mass'];env.length=row['length'];envs.append(env)
    b=len(envs);obs=np.stack([envcode.observation(e.state) for e in envs])
    cps=np.zeros((b,10,3),np.float32);cpa=np.zeros((b,10,1),np.float32);counts=np.zeros(b,np.int64)
    planner=TorchCEM(b,device,seed);rng=np.random.RandomState(seed)
    trajectories=[dict(obs=[obs[i].copy()],act=[],reward=[],cp_obs=[],cp_act=[],metadata=rows[i]) for i in range(b)]
    start=time.monotonic()
    for t in range(200):
        actions=rng.uniform(-1,1,size=(b,1)) if random_action else planner.plan(model,obs,cps,cpa)
        nxt=[]
        for i,env in enumerate(envs):
            trajectories[i]['cp_obs'].append(cps[i].copy());trajectories[i]['cp_act'].append(cpa[i].copy())
            o,r,done,_=env.step(actions[i]);assert not done
            trajectories[i]['obs'].append(o.copy());trajectories[i]['act'].append(actions[i].copy());trajectories[i]['reward'].append(r);nxt.append(o)
        nxt=np.asarray(nxt);update_history(cps,cpa,counts,obs,actions,nxt);obs=nxt
    for i,path in enumerate(trajectories):
        for key in ('obs','act','reward','cp_obs','cp_act'):
            path[key]=np.asarray(path[key],dtype=np.float64 if key=='reward' else np.float32)
        path['success']=bool(envs[i].success)
    return trajectories,time.monotonic()-start


def prepare(paths,method,rng):
    """Official 199 anchors/path; split anchors before CaDM future expansion."""
    obs=np.concatenate([p['obs'][:199] for p in paths]);nxt=np.concatenate([p['obs'][1:200] for p in paths])
    act=np.concatenate([p['act'][:199] for p in paths]);cp=np.concatenate([p['cp_obs'][:199] for p in paths]);ca=np.concatenate([p['cp_act'][:199] for p in paths])
    # Effective state_diff=1 cp_obs stats are identity in released code.
    norm={}
    for key,value in [('obs',obs),('act',act),('delta',nxt-obs),('back_delta',obs-nxt),('cp_act',ca.reshape(-1,10))]:
        norm[key]=(value.mean(0),value.std(0))
    norm['cp_obs']=(np.zeros(30),np.ones(30))
    n=len(obs);order=rng.permutation(n);nv=min(int(n*.1),5000)
    anchor_train=np.zeros(n,bool);anchor_train[order[nv:]]=True
    data={name:[] for name in ['obs','act','nxt','cp_obs','cp_act','anchor','episode']}
    for ep,path in enumerate(paths):
        for t in range(199):
            offsets=[0] if method=='Vanilla' else range(min(10,199-t))
            if method=='CaDM' and t==0:continue # preserve released concat_bool[-0] behavior
            for k in offsets:
                for name,value in [('obs',path['obs'][t+k]),('nxt',path['obs'][t+k+1]),('act',path['act'][t+k]),('cp_obs',path['cp_obs'][t]),('cp_act',path['cp_act'][t])]:data[name].append(value)
                data['anchor'].append(ep*199+t);data['episode'].append(ep)
    data={key:np.asarray(value) for key,value in data.items()}
    train_mask=anchor_train[data['anchor']]
    assert all(len(v)==len(train_mask) for v in data.values())
    expected=len(paths)*(199 if method=='Vanilla' else 1935);assert len(train_mask)==expected
    return data,train_mask,norm,dict(anchors=n,validation_anchors=nv,expanded_rows=expected,train_rows=int(train_mask.sum()),validation_rows=int((~train_mask).sum()))


def fit(model,optimizer,paths,device,rng,epochs):
    data,mask,norm,receipt=prepare(paths,model.method,rng)
    with torch.no_grad():
        for key,(mu,sd) in norm.items():
            getattr(model,key+'_mean').copy_(torch.as_tensor(mu,device=device));getattr(model,key+'_std').copy_(torch.as_tensor(sd,device=device))
    tensors={k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in data.items() if k in ('obs','act','nxt','cp_obs','cp_act')}
    train=np.flatnonzero(mask);valid=np.flatnonzero(~mask);batch=32 if model.method=='Vanilla' else 256
    logs=[];updates=0;start=time.monotonic()
    for epoch in range(epochs):
        model.train();order=rng.permutation(train);tot=0.
        for i in range(0,len(order),batch):
            ix=torch.as_tensor(order[i:i+batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss,terms=model.loss(*(tensors[k][ix] for k in ('obs','act','nxt','cp_obs','cp_act')))
            assert torch.isfinite(loss);loss.backward()
            grad=sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)
            assert torch.isfinite(grad);optimizer.step();updates+=1;tot+=float(loss)*len(ix)
        model.eval();val=0.
        with torch.inference_mode():
            for i in range(0,len(valid),256):
                ix=torch.as_tensor(valid[i:i+256],device=device)
                _,terms=model.loss(*(tensors[k][ix] for k in ('obs','act','nxt','cp_obs','cp_act')))
                val+=float(terms['forward'])*len(ix)
        logs.append(dict(epoch=epoch+1,train_loss=tot/len(train),validation_forward=val/max(len(valid),1)))
    receipt.update(epochs=epochs,batch_size=batch,optimizer_updates=updates,seconds=time.monotonic()-start,epochs_log=logs)
    return receipt


def save_paths(out,label,paths):
    arrays={}
    for i,p in enumerate(paths):
        for key in ('obs','act','reward','cp_obs','cp_act'):arrays[f'e{i}_{key}']=p[key]
    np.savez_compressed(out/f'{label}.npz',**arrays)
    write(out/f'{label}.json',dict(episodes=[dict(metadata=p['metadata'],return_sum=float(p['reward'].astype('float64').sum()),success=p['success']) for p in paths],
             arrays_sha256=sha(out/f'{label}.npz'),real_environment_steps=len(paths)*200))


def smoke(args):
    components.seed_all(args.seed);model=Model(args.method).to(args.device)
    generator=np.random.RandomState(0)
    obs=torch.as_tensor(generator.normal(size=(32,3)),dtype=torch.float32,device=args.device)
    act=torch.zeros((32,1),device=args.device);cp=torch.zeros((32,10,3),device=args.device);ca=torch.zeros((32,10,1),device=args.device)
    loss,terms=model.loss(obs,act,obs+.03,cp,ca);loss.backward();assert torch.isfinite(loss)
    norms={name:float(sum(p.grad.square().sum() for p in mod.parameters() if p.grad is not None).sqrt()) for name,mod in model.named_children()}
    assert norms['forward_model']>0
    if args.method=='CaDM':assert norms['backward_model']>0 and norms['context']>0
    s=np.zeros((2,10,3),np.float32);a=np.zeros((2,10,1),np.float32);n=np.zeros(2,np.int64)
    for t in range(12):
        old=np.zeros((2,3));new=np.full((2,3),t+1);update_history(s,a,n,old,np.full((2,1),t+1),new)
        if t<9:assert np.all(s[:,t+1:]==0)
    np.testing.assert_array_equal(s[0,:,0],np.arange(3,13));assert np.all(np.zeros_like(s)==0)
    rows=manifest(args.seed,0,2);paths,_=sample(model,rows,args.device,10,random_action=True)
    for path in paths:
        assert np.count_nonzero(path['cp_obs'][0])==np.count_nonzero(path['cp_act'][0])==0
        np.testing.assert_allclose(path['cp_obs'][1,0],path['obs'][1]-path['obs'][0],atol=3e-8)
    data,mask,norm,prep=prepare(paths,args.method,np.random.RandomState(0))
    planner=TorchCEM(1,args.device,0);start=time.monotonic()
    action=planner.plan(model,np.array([[-1.,0.,0.]]),np.zeros((1,10,3)),np.zeros((1,10,1)))
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    seconds=time.monotonic()-start
    result=dict(status='PASS',method=args.method,gradient_norms=norms,shape_boundary=True,history_reset=True,
                prepare=prep,one_learned_CEM_action_seconds=seconds,action=action.tolist(),optimizer_updates=0,
                engineering_random_interactions=400,learned_closed_loop_episode=False,device=args.device,
                code_sha256=sha(__file__),scope='Untrained-model planner throughput only; not a return result.')
    write(Path(args.output)/'SMOKE.json',result);print(json.dumps(result),flush=True)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RUN.json').exists(),'Existing run preserved; no implicit overwrite or restart.'
    if args.action=='smoke':
        assert not (out/'SMOKE.json').exists(),'Existing smoke preserved; use a fresh output.'
        smoke(args);return
    # This port intentionally exposes only the released deterministic E=1 option.
    assert args.ensemble_size==1 and args.deterministic==1,'Probabilistic ensemble path is not ported: do not silently simplify.'
    ev=envcode.validate(argparse.Namespace(official_root=args.official_root,gym_reference=str(HERE/'third_party/gym_0_16_pendulum.py'),output=str(out/'environment_gate')))
    official=components.source_evidence(argparse.Namespace(source_receipt=str(HERE/'CaDM_SOURCE_RECEIPT.json'),official_root=args.official_root))
    iterations,episodes,eval_episodes=(2,2,1) if args.profile=='gate' else (20,10,10)
    config=dict(method=args.method,implementation='Released deterministic E=1 configuration, PyTorch online adaptation',seed=args.seed,
        iterations=iterations,episodes_per_iteration=episodes,episode_steps=200,training_interactions=iterations*episodes*200,
        fit_epochs=5,lr=.001,batch_size=32 if args.method=='Vanilla' else 256,history=10,future=1 if args.method=='Vanilla' else 10,
        context_dims=[40,256,128,64,10],context_final='linear',dynamics_hidden=[200]*4,back_coefficient=.5,
        normalization='Cumulative online train anchors; state_diff cp_obs identity, cp_act slot-wise; normalized flag enabled',
        native_mask='Official 199 anchors and CaDM anchor0 masked; 1935 flattened rows/episode',ensemble_size=1,deterministic=True,
        observation_budget='Own-controller online sampling; same reset manifests and interaction counts across arms',
        parameter_input=False,relation_objective=False,eval_groups=GRIDS,eval_episodes_per_group=eval_episodes,
        evaluation_scope='Fixed development control manifest, not a sealed benchmark test',
        official=official,code_hashes={p.name:sha(p) for p in [Path(__file__),HERE/'cadm_dclean_pilot.py',HERE/'cadm_pendulum_engineering_gate.py']},
        adaptations=['PyTorch and explicit independent episode RNG','E1 deterministic (published README option, not CLI default E5 probabilistic)',
          'Fixed five epochs rather than rolling validation early stopping','Normalization enabled','No model or recipe selection from evaluation returns'])
    write(out/'CONFIG.json',config);write(out/'RUN.json',dict(pid=os.getpid(),start=time.time(),config_sha256=sha(out/'CONFIG.json')))
    components.seed_all(args.seed);model=Model(args.method).to(args.device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.RandomState(args.seed);paths=[];curve=[];total_updates=0;begin=time.monotonic()
    for iteration in range(iterations):
        rows=manifest(args.seed,iteration,episodes)
        fresh,sample_seconds=sample(model,rows,args.device,200000+args.seed*100+iteration,random_action=iteration==0)
        paths.extend(fresh);save_paths(out,f'train_iter{iteration:02d}',fresh)
        receipt=fit(model,optimizer,paths,args.device,rng,5);total_updates+=receipt['optimizer_updates']
        record=dict(iteration=iteration+1,training_interactions=(iteration+1)*episodes*200,
                    collection_policy='uniform_random' if iteration==0 else 'current_learned_model_CEM',sample_seconds=sample_seconds,
                    collection_return_mean=float(np.mean([p['reward'].astype('float64').sum() for p in fresh])),fit=receipt,total_optimizer_updates=total_updates)
        curve.append(record);write(out/f'ITERATION_{iteration+1:02d}.json',record);print(json.dumps(record),flush=True)
        ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration+1,training_interactions=record['training_interactions'],
                config_sha256=sha(out/'CONFIG.json'),numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state())
        torch.save(ck,out/f'checkpoint_iter{iteration+1:02d}.pt')
    evals={}
    for index,group in enumerate(GRIDS):
        rows=manifest(args.seed,index,eval_episodes,group,evaluation=True)
        test,seconds=sample(model,rows,args.device,800000+index,random_action=False)
        save_paths(out,'control_'+group,test)
        evals[group]=dict(mean_return=float(np.mean([p['reward'].astype('float64').sum() for p in test])),
            returns=[float(p['reward'].astype('float64').sum()) for p in test],success_rate=float(np.mean([p['success'] for p in test])),
            seconds=seconds,episodes=eval_episodes,real_environment_steps=eval_episodes*200)
    summary=dict(status='COMPLETE',method=args.method,profile=args.profile,training_interactions=iterations*episodes*200,
                 evaluation_interactions=5*eval_episodes*200,total_optimizer_updates=total_updates,curve=curve,evaluation=evals,
                 final_checkpoint=f'checkpoint_iter{iterations:02d}.pt',final_checkpoint_sha256=sha(out/f'checkpoint_iter{iterations:02d}.pt'),
                 seconds=time.monotonic()-begin,config_sha256=sha(out/'CONFIG.json'),closed_loop=True,
                 scope='Online PyTorch adaptation; gate profile is engineering only, formal profile is a seed0 development comparison.')
    write(out/'SUMMARY.json',summary);write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json'),code_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['smoke','run'])
    p.add_argument('--method',choices=['Vanilla','CaDM'],required=True);p.add_argument('--profile',choices=['gate','formal'],default='gate')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ensemble-size',type=int,default=1);p.add_argument('--deterministic',type=int,default=1)
    p.add_argument('--official-root',default=str(HERE/'third_party/CaDM'))
    args=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        run(args);write(Path(args.output)/f'EXIT_{args.action}.json',dict(exit_code=0,unix_time=time.time()))
    except Exception as exc:
        write(Path(args.output)/f'FAILED_{args.action}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
