#!/usr/bin/env python3
"""Fixed three-seed CaDM / SPRII-Align / Random-Align confirmation.

Independent version; never edits frozen Vanilla/CaDM files. Native CaDM loss,
model, planner, replay, normalizer and observation budget are retained. Only
already-collected training histories supply the relation objective. No Cross.
"""
from __future__ import annotations
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_cadm_online_v1',HERE/'cadm_pendulum_online.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
BASE_SHA='ac458b321d0d3c78c4ab6d7bff193568706d633f680891fc133a32cf9166d9ad'
assert base.sha(HERE/'cadm_pendulum_online.py')==BASE_SHA,'Frozen baseline changed: re-audit explicitly.'
ALIGN_WEIGHT=.003
MIN_SYSTEMS=2
MAX_SYSTEMS=32


def group_key(path):
    """Grouping metadata only; this tuple is never passed to any network."""
    return (float(path['metadata']['mass']),float(path['metadata']['length']))


class RelationBank:
    def __init__(self,paths,data,train_mask,device):
        self.paths=paths;self.device=device
        self.cp=np.concatenate([p['cp_obs'][:199] for p in paths])
        self.ca=np.concatenate([p['cp_act'][:199] for p in paths])
        anchors=np.unique(data['anchor'][train_mask])
        self.legal=set(int(i) for i in anchors if int(i)%199>=10)
        groups=collections.defaultdict(lambda:collections.defaultdict(list))
        for anchor in sorted(self.legal):
            ep=anchor//199;groups[group_key(paths[ep])][ep].append(anchor)
        self.groups={key:{ep:np.asarray(ids) for ep,ids in episodes.items()}
                     for key,episodes in groups.items() if len(episodes)>=2}
        self.keys=sorted(self.groups)
        self.stats=dict(episodes=len(paths),eligible_systems=len(self.keys),
                        eligible_episodes=sum(len(v) for v in self.groups.values()),legal_train_anchors=len(self.legal),
                        activation_eligible=len(self.keys)>=MIN_SYSTEMS,
                        histories='t10..198, training anchors, distinct observed episodes, same exact mass/length',
                        additional_environment_interactions=0,metadata_input_to_model=False)
        self.cp_tensor=torch.as_tensor(self.cp,dtype=torch.float32,device=device)
        self.ca_tensor=torch.as_tensor(self.ca,dtype=torch.float32,device=device)

    def draw(self,rng):
        if len(self.keys)<MIN_SYSTEMS:return None
        selected=rng.choice(len(self.keys),min(MAX_SYSTEMS,len(self.keys)),replace=False)
        left,right=[],[]
        for group in selected:
            choices=self.groups[self.keys[int(group)]]
            ea,eb=rng.choice(list(choices),2,replace=False)
            left.append(int(rng.choice(choices[int(ea)])));right.append(int(rng.choice(choices[int(eb)])))
        # One pair per distinct group makes a nonzero circular permutation a
        # guaranteed cross-system derangement with exactly the same donor bank.
        permutation=np.roll(np.arange(len(left)),int(rng.randint(1,len(left))))
        left,right=np.asarray(left),np.asarray(right)
        for a,b in zip(left,right):
            assert int(a) in self.legal and int(b) in self.legal
            assert a//199!=b//199 and group_key(self.paths[a//199])==group_key(self.paths[b//199])
        for a,b in zip(left,right[permutation]):
            assert group_key(self.paths[a//199])!=group_key(self.paths[b//199])
        return left,right,permutation

    def encode(self,model,pair):
        a,b,permutation=pair
        ai=torch.as_tensor(a,device=self.device);bi=torch.as_tensor(b,device=self.device)
        # No observations outside the saved completed-history arrays enter here.
        za=model.encode(self.cp_tensor[ai],self.ca_tensor[ai])
        zb=model.encode(self.cp_tensor[bi],self.ca_tensor[bi])
        return za,zb,torch.as_tensor(permutation,device=self.device)


def vicreg(za,zb):
    assert len(za)==len(zb)>=2
    inv=(za-zb).square().mean()
    var=.5*(torch.relu(1-torch.sqrt(za.var(0,unbiased=True)+1e-4)).mean()+
             torch.relu(1-torch.sqrt(zb.var(0,unbiased=True)+1e-4)).mean())
    a,b=za-za.mean(0),zb-zb.mean(0)
    ca,cb=a.T@a/(len(a)-1),b.T@b/(len(b)-1)
    off=~torch.eye(za.shape[1],dtype=torch.bool,device=za.device)
    cov=(ca[off].square().sum()+cb[off].square().sum())/za.shape[1]
    return 25*inv+25*var+cov,dict(invariance=inv,variance=var,covariance=cov)


def augment_loss(native,model,bank,pair,arm,weight=ALIGN_WEIGHT):
    # Exact no-op path, including no auxiliary forward, when relation disabled.
    if arm=='CaDM' or weight==0 or pair is None:return native,{}
    za,zb,permutation=bank.encode(model,pair)
    if arm=='RandomAlign':zb=zb[permutation]
    align,terms=vicreg(za,zb)
    return native+weight*align,dict(align=align,**terms)


def fit(model,optimizer,paths,device,rng,epochs,arm,iteration,seed,weight=ALIGN_WEIGHT):
    if arm=='CaDM' or weight==0:return base.fit(model,optimizer,paths,device,rng,epochs)
    data,mask,norm,receipt=base.prepare(paths,'CaDM',rng)
    with torch.no_grad():
        for key,(mu,sd) in norm.items():
            getattr(model,key+'_mean').copy_(torch.as_tensor(mu,device=device));getattr(model,key+'_std').copy_(torch.as_tensor(sd,device=device))
    tensors={k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in data.items() if k in ('obs','act','nxt','cp_obs','cp_act')}
    bank=RelationBank(paths,data,mask,device)
    # Relation sampling cannot perturb native train/validation/permutation RNG.
    relation_rng=np.random.RandomState(420000+100*seed+iteration)
    train=np.flatnonzero(mask);valid=np.flatnonzero(~mask);batch=256
    logs=[];updates=active=pair_count=0;start=time.monotonic();audit=hashlib.sha256()
    group_histogram=collections.Counter();unique_donor_min=None
    for epoch in range(epochs):
        model.train();order=rng.permutation(train);tot=0.;native_tot=0.;relation_tot=0.
        component_tot=dict(invariance=0.,variance=0.,covariance=0.)
        for i in range(0,len(order),batch):
            ix=torch.as_tensor(order[i:i+batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            native,_=model.loss(*(tensors[k][ix] for k in ('obs','act','nxt','cp_obs','cp_act')))
            pair=bank.draw(relation_rng)
            loss,terms=augment_loss(native,model,bank,pair,arm,weight=weight)
            assert torch.isfinite(loss);loss.backward()
            grad=sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)
            assert torch.isfinite(grad);optimizer.step();updates+=1
            tot+=float(loss)*len(ix);native_tot+=float(native)*len(ix)
            if pair is not None:
                active+=1;pair_count+=len(pair[0]);relation_tot+=float(terms['align'])*len(ix)
                group_histogram[str(len(pair[0]))]+=1
                donor_unique=len(np.unique(pair[1]));assert donor_unique==len(pair[0])
                unique_donor_min=donor_unique if unique_donor_min is None else min(unique_donor_min,donor_unique)
                for key in component_tot:component_tot[key]+=float(terms[key])*len(ix)
                for a in pair:audit.update(a.astype('int64').tobytes())
        model.eval();val=0.
        with torch.inference_mode():
            for i in range(0,len(valid),256):
                ix=torch.as_tensor(valid[i:i+256],device=device)
                _,terms=model.loss(*(tensors[k][ix] for k in ('obs','act','nxt','cp_obs','cp_act')))
                val+=float(terms['forward'])*len(ix)
        logs.append(dict(epoch=epoch+1,train_loss=tot/len(train),native_loss=native_tot/len(train),
                         unweighted_alignment=relation_tot/len(train),alignment_components={k:v/len(train) for k,v in component_tot.items()},
                         validation_forward=val/max(len(valid),1)))
    receipt.update(epochs=epochs,batch_size=batch,optimizer_updates=updates,seconds=time.monotonic()-start,
                   epochs_log=logs,relation_bank=bank.stats,relation_updates=active,relation_pairs=pair_count,
                   relation_pair_sequence_sha256=audit.hexdigest(),alignment_weight=weight,
                   relation_skipped_updates=updates-active,relation_unique_system_count_histogram=dict(group_histogram),
                   minimum_unique_donor_histories_per_active_batch=unique_donor_min,
                   randomization='Only cross-system permutation of identical donor bank' if arm=='RandomAlign' else 'Correct independent histories')
    return receipt


def diagnose(out):
    result={}
    for profile,iterations,episodes in [('original_gate',2,2),('relation_gate',2,10),('formal',20,10)]:
        rows=[];timeline=[]
        for it in range(iterations):
            rows+=base.manifest(0,it,episodes)
            counts=collections.Counter((r['mass'],r['length']) for r in rows)
            paired=sum(v>=2 for v in counts.values());eligible=sum(v for v in counts.values() if v>=2)
            timeline.append(dict(iteration=it+1,episodes=len(rows),unique_systems=len(counts),paired_systems=paired,
                                  eligible_episodes=eligible,fraction=eligible/len(rows),align_activated=paired>=MIN_SYSTEMS))
        result[profile]=timeline
    base.write(out/'AVAILABILITY.json',dict(status='COMPLETE',seed=0,profiles=result,environment_steps=0,optimizer_updates=0,
                 note='Metadata-only maximum availability; exact training-anchor availability recorded during each fit.'))
    return result


def smoke(args):
    out=Path(args.output);base.components.seed_all(0);model=base.Model('CaDM').to(args.device)
    # Six independent random engineering episodes, three systems. No training.
    rows=base.manifest(0,0,6)
    for i,row in enumerate(rows):row['mass']=[.8,1.,1.2][i//2];row['length']=[1.2,1.,.8][i//2]
    paths,_=base.sample(model,rows,args.device,345,random_action=True)
    data,mask,norm,_=base.prepare(paths,'CaDM',np.random.RandomState(0));bank=RelationBank(paths,data,mask,args.device)
    pair=bank.draw(np.random.RandomState(7));assert pair is not None
    za,zb,perm=bank.encode(model,pair)
    correct,ct=vicreg(za,zb);random,rt=vicreg(za,zb[perm])
    torch.testing.assert_close(ct['variance'],rt['variance'],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(ct['covariance'],rt['covariance'],rtol=1e-5,atol=1e-10)
    np.testing.assert_array_equal(np.sort(pair[1]),np.sort(pair[1][pair[2]]))
    correct.backward();relation_gradient=float(sum(p.grad.square().sum() for p in model.context.parameters() if p.grad is not None).sqrt())
    assert relation_gradient>0 and torch.isfinite(correct) and torch.isfinite(random)
    # Future modifications cannot change already-saved history pair tensors.
    future_changed=copy.deepcopy(paths)
    for path in future_changed:path['obs'][:]=12345;path['act'][:]=-12345
    other=RelationBank(future_changed,data,mask,args.device)
    za2,zb2,_=other.encode(model,pair)
    torch.testing.assert_close(za,za2,atol=0,rtol=0);torch.testing.assert_close(zb,zb2,atol=0,rtol=0)
    # Two microscopic optimizer steps per clone through the actual fit path.
    # No performance training or learned controller evaluation occurs.
    base.components.seed_all(99);a=base.Model('CaDM').to(args.device);b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    tiny={k:v[:544] for k,v in data.items()};tiny_mask=np.arange(544)<512
    saved_prepare=base.prepare
    try:
        base.prepare=lambda *_args:(tiny,tiny_mask,norm,{'engineering_subset':True})
        ra,rb=np.random.RandomState(55),np.random.RandomState(55)
        aa=base.fit(a,oa,paths,args.device,ra,1)
        bb=fit(b,ob,paths,args.device,rb,1,'SPRIIAlign',0,0,weight=0)
    finally:base.prepare=saved_prepare
    assert aa['optimizer_updates']==bb['optimizer_updates']==2
    assert aa['epochs_log']==bb['epochs_log']
    for va,vb in zip(a.state_dict().values(),b.state_dict().values()):torch.testing.assert_close(va,vb,rtol=0,atol=0)
    np.testing.assert_array_equal(ra.get_state()[1],rb.get_state()[1]);assert ra.get_state()[2:]==rb.get_state()[2:]
    # Relation draws use an isolated RNG and cannot move the native shuffle RNG.
    before=copy.deepcopy(ra.get_state());rel=np.random.RandomState(11)
    for _ in range(3):bank.draw(rel)
    np.testing.assert_array_equal(before[1],ra.get_state()[1]);assert before[2:]==ra.get_state()[2:]
    # One eligible group makes strict Random impossible: both must skip.
    d,m,_,_=base.prepare(paths[:2],'CaDM',np.random.RandomState(0));small=RelationBank(paths[:2],d,m,args.device)
    assert small.draw(np.random.RandomState(0)) is None
    for p in paths:
        assert np.count_nonzero(p['cp_obs'][0])==0
        np.testing.assert_allclose(p['cp_obs'][10],p['obs'][1:11]-p['obs'][:10],atol=2e-7)
    result=dict(status='PASS',arm=args.arm,bank=bank.stats,valid_pair_ids=[p.tolist() for p in pair],
        correct_align=float(correct.detach()),random_align=float(random.detach()),relation_gradient_norm=relation_gradient,
        random_preserves_donor_marginal=True,one_group_common_skip=True,native_unchanged=True,
        disabled_relation_two_step_parameter_equality=True,metadata_never_model_input=True,history_future_boundary=True,
        disabled_relation_actual_fit_equivalence=True,native_shuffle_rng_equal=True,auxiliary_rng_isolated=True,
        engineering_random_interactions=1200,equivalence_optimizer_steps_per_clone=2,performance_training=False,
        code_sha256=base.sha(__file__),base_sha256=BASE_SHA)
    base.write(out/'SMOKE.json',result);print(json.dumps(result),flush=True)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.action=='diagnose':diagnose(out);return
    if args.action=='smoke':
        assert not (out/'SMOKE.json').exists(),'Preserve earlier smoke.'
        smoke(args);return
    assert args.seed in (0,1,2),'Fixed confirmation seeds only.'
    assert not (out/'RUN.json').exists(),'Preserve incomplete or complete runs; no implicit restart.'
    iterations,episodes,eval_episodes=(2,10,1) if args.profile=='relation_gate' else (20,10,10)
    official=base.components.source_evidence(argparse.Namespace(source_receipt=str(HERE/'CaDM_SOURCE_RECEIPT.json'),official_root=args.official_root))
    config=dict(method=args.arm,base_method='CaDM deterministic E1 PyTorch online adaptation',seed=args.seed,
        profile=args.profile,iterations=iterations,episodes_per_iteration=episodes,episode_steps=200,
        training_interactions=iterations*episodes*200,evaluation_interactions=5*eval_episodes*200,
        fit_epochs=5,batch_size=256,lr=.001,history=10,future=10,alignment_weight=0 if args.arm=='CaDM' else args.align_weight,
        objective='Native forward + .5 backward + official L2; plus lambda*(25inv+25var+cov) on the 10D context',
        cross_objective=False,alignment_min_systems=MIN_SYSTEMS,alignment_max_systems=MAX_SYSTEMS,
        grouping='Exact simulator mass/length equality, only for pairing independent episodes; never model input',
        bank='Already-collected cumulative replay, training anchors t10..198, one pair per distinct eligible system',
        random_control='Only derange the same donor minibank across systems; native batch/loss and marginal histories unchanged',
        budget='Same reset manifests, own-controller online sampling and interaction count as frozen baseline formal recipe',
        extra_environment_observations=0,extra_relation_compute=True,selection='Fixed selected lambda .0003, seeds0/1/2 all included, final checkpoint; no OOD selection',
        base_sha256=BASE_SHA,code_sha256=base.sha(__file__),official=official,
        scope='Align-only method adaptation, not the complete Align+Cross recipe; relation_gate is engineering only; no new sealed test')
    base.write(out/'CONFIG.json',config);base.write(out/'RUN.json',dict(pid=os.getpid(),time=time.time(),config_sha256=base.sha(out/'CONFIG.json')))
    base.components.seed_all(args.seed);model=base.Model('CaDM').to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.RandomState(args.seed)
    paths=[];curve=[];updates=0;active=0;begin=time.monotonic()
    for iteration in range(iterations):
        rows=base.manifest(args.seed,iteration,episodes)
        fresh,seconds=base.sample(model,rows,args.device,200000+args.seed*100+iteration,random_action=iteration==0)
        paths.extend(fresh);base.save_paths(out,f'train_iter{iteration:02d}',fresh)
        receipt=fit(model,optimizer,paths,args.device,rng,5,args.arm,iteration,args.seed,weight=args.align_weight)
        updates+=receipt['optimizer_updates'];active+=receipt.get('relation_updates',0)
        record=dict(iteration=iteration+1,training_interactions=(iteration+1)*episodes*200,
             collection_policy='uniform_random' if iteration==0 else 'current_learned_model_CEM',sample_seconds=seconds,
             collection_return_mean=float(np.mean([p['reward'].sum() for p in fresh])),fit=receipt,total_optimizer_updates=updates)
        curve.append(record);base.write(out/f'ITERATION_{iteration+1:02d}.json',record);print(json.dumps(record),flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration+1,
            training_interactions=record['training_interactions'],config_sha256=base.sha(out/'CONFIG.json'),
            numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state()),out/f'checkpoint_iter{iteration+1:02d}.pt')
    if args.arm!='CaDM':assert active>0,'No relation updates: cannot call this a relation experiment.'
    evals={}
    for index,group in enumerate(base.GRIDS):
        rows=base.manifest(args.seed,index,eval_episodes,group,evaluation=True)
        test,seconds=base.sample(model,rows,args.device,800000+index,random_action=False)
        base.save_paths(out,'control_'+group,test)
        evals[group]=dict(mean_return=float(np.mean([p['reward'].sum() for p in test])),returns=[float(p['reward'].sum()) for p in test],
                         success_rate=float(np.mean([p['success'] for p in test])),seconds=seconds,episodes=eval_episodes)
    final=out/f'checkpoint_iter{iterations:02d}.pt'
    summary=dict(status='COMPLETE',method=args.arm,profile=args.profile,curve=curve,evaluation=evals,
         training_interactions=iterations*episodes*200,evaluation_interactions=5*eval_episodes*200,
         total_optimizer_updates=updates,total_relation_updates=active,final_checkpoint=str(final),final_checkpoint_sha256=base.sha(final),
         config_sha256=base.sha(out/'CONFIG.json'),seconds=time.monotonic()-begin,closed_loop=True,comparison_ready=False,
         scope='Needs same-profile controls and full trace comparison. Align-only adaptation; historical development, not sealed test.')
    base.write(out/'SUMMARY.json',summary);base.write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=base.sha(out/'SUMMARY.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['diagnose','smoke','run'])
    p.add_argument('--arm',choices=['CaDM','SPRIIAlign','RandomAlign'],default='SPRIIAlign')
    p.add_argument('--profile',choices=['relation_gate','formal'],default='relation_gate')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--official-root',default=str(HERE/'third_party/CaDM'))
    p.add_argument('--align-weight',type=float,choices=[0.0,.0003,.001,.003],default=.003)
    args=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args);base.write(out/f'EXIT_{args.action}.json',dict(exit_code=0,unix_time=time.time()))
        except Exception as exc:
            base.write(out/f'FAILED_{args.action}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
