#!/usr/bin/env python3
"""Read-only frozen-context probes and logged-action donor inference.

No simulator interaction, source optimizer, checkpoint selection or Vanilla
probe. Ridge is fitted only on the saved training bank with physical-group CV.
"""
from __future__ import annotations
import argparse
import collections
import csv
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

ARMS=['CaDM','SPRIIAlign','RandomAlign']
GROUPS=['ID','OOD_c0','OOD_c1','OOD_c2','OOD_c3']
ANCHORS=list(range(10,191,20))
ALPHAS=[1e-4,1e-3,1e-2,.1,1.,10.]
BASE_SHA='ac458b321d0d3c78c4ab6d7bff193568706d633f680891fc133a32cf9166d9ad'
V2_SHA='1f0734f33f86d3da5a5cf2c49a2c25d5314426b5bafbfb22d683d7ceb063bf9b'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def physical(row):return (float(row['mass']),float(row['length']))
def state_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def csv_write(path,rows):
    assert rows
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_arm(root,arm,code):
    out=root/arm;c,r,s,d=[read(out/n) for n in ['CONFIG.json','RUN.json','SUMMARY.json','COMPLETE.json']]
    assert read(out/'EXIT_run.json')['exit_code']==0 and d['status']==s['status']=='COMPLETE'
    assert d['summary_sha256']==sha(out/'SUMMARY.json')
    assert sha(out/'CONFIG.json')==r['config_sha256']==s['config_sha256']
    assert c['method']==s['method']==arm and c['seed']==0 and c['profile']==s['profile']=='formal'
    assert c['base_sha256']==sha(code/'cadm_pendulum_online.py')==BASE_SHA
    assert c['code_sha256']==sha(code/'cadm_pendulum_relation_v2.py')==V2_SHA
    assert c['iterations']==20 and c['episodes_per_iteration']==10 and c['episode_steps']==200
    assert c['training_interactions']==s['training_interactions']==40000
    assert c['evaluation_interactions']==s['evaluation_interactions']==10000
    assert len(s['curve'])==20 and set(s['evaluation'])==set(GROUPS)
    for i,row in enumerate(s['curve'],1):
        assert row==read(out/f'ITERATION_{i:02d}.json') and row['training_interactions']==i*2000
    checkpoint=out/Path(s['final_checkpoint']).name
    assert checkpoint.name=='checkpoint_iter20.pt' and sha(checkpoint)==s['final_checkpoint_sha256']
    return dict(path=str(out),config_sha256=sha(out/'CONFIG.json'),checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),summary_sha256=sha(out/'SUMMARY.json')),s


def make_bank(root,out,summaries):
    episodes=[];rows=[];excluded=[];traces=[]
    arrays={k:[] for k in ['cp_obs','cp_act','obs','actions','targets']}
    for arm in ARMS:
        labels=[(f'train_iter{i:02d}','train','ID',i) for i in range(20)]+[('control_'+g,'control',g,None) for g in GROUPS]
        for label,split,group,iteration in labels:
            p=root/arm/(label+'.npz');j=root/arm/(label+'.json');record=read(j)
            assert record['arrays_sha256']==sha(p) and record['real_environment_steps']==2000
            assert len(record['episodes'])==10
            traces.append(dict(collector=arm,label=label,json_path=str(j),json_sha256=sha(j),npz_path=str(p),npz_sha256=sha(p)))
            with np.load(p,allow_pickle=False) as data:
                for e,entry in enumerate(record['episodes']):
                    meta=entry['metadata'];assert meta['group']==group
                    episode=dict(episode_id=f'{arm}/{label}/e{e}/reset{meta["seed"]}',collector=arm,label=label,split=split,
                        group=group,iteration=iteration,episode_index=e,reset_seed=int(meta['seed']),mass=float(meta['mass']),
                        length=float(meta['length']),initial_state=meta['initial_state'],trace_index=len(traces)-1)
                    ei=len(episodes);episodes.append(episode)
                    a={key:np.asarray(data[f'e{e}_{key}']) for key in ['obs','act','reward','cp_obs','cp_act']}
                    expected={'obs':(201,3),'act':(200,1),'reward':(200,),'cp_obs':(200,10,3),'cp_act':(200,10,1)}
                    for key,shape in expected.items():assert a[key].shape==shape and np.isfinite(a[key]).all(),(arm,label,e,key)
                    assert np.count_nonzero(a['cp_obs'][0])==0 and np.count_nonzero(a['cp_act'][0])==0
                    np.testing.assert_allclose(a['reward'].astype('float64').sum(),entry['return_sum'],atol=1e-10,rtol=0)
                    if split=='control':
                        np.testing.assert_allclose(entry['return_sum'],summaries[arm]['evaluation'][group]['returns'][e],rtol=0,atol=1e-10)
                    for t in ANCHORS:
                        if not np.allclose(a['cp_obs'][t],a['obs'][t-9:t+1]-a['obs'][t-10:t],atol=2e-6,rtol=1e-5):
                            excluded.append(dict(episode_id=episode['episode_id'],t=t,reason='saved_history_delta_mismatch'));continue
                        if not np.allclose(a['cp_act'][t],a['act'][t-10:t],atol=1e-7,rtol=0):
                            excluded.append(dict(episode_id=episode['episode_id'],t=t,reason='saved_action_history_mismatch'));continue
                        rows.append(dict(row=len(rows),episode=ei,t=t,**episode))
                        for key,value in [('cp_obs',a['cp_obs'][t]),('cp_act',a['cp_act'][t]),('obs',a['obs'][t]),
                                          ('actions',a['act'][t:t+10]),('targets',a['obs'][t+1:t+11])]:arrays[key].append(value)
    assert len(episodes)==750
    # The common reset manifest must agree across all three collector arms.
    reset_views={arm:[{k:e[k] for k in ['label','episode_index','reset_seed','mass','length','initial_state']} for e in episodes if e['collector']==arm] for arm in ARMS}
    assert reset_views['CaDM']==reset_views['SPRIIAlign']==reset_views['RandomAlign']
    arrays={k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()}
    np.savez_compressed(out/'BANK.npz',**arrays)
    write(out/'BANK_ROWS.json',rows);write(out/'BANK_EPISODES.json',episodes);write(out/'BANK_EXCLUSIONS.json',excluded)
    bank=dict(status='COMPLETE',anchor_times=ANCHORS,history_length=10,query_length=10,expected_rows=7500,
        actual_rows=len(rows),training_rows=sum(r['split']=='train' for r in rows),control_rows=sum(r['split']=='control' for r in rows),
        collector_counts={a:sum(r['collector']==a for r in rows) for a in ARMS},same_reset_manifests=True,
        weighting='equal episode, then equal collector; reset seed clusters across collector arms',
        history_check_atol=2e-6,files={p.name:sha(p) for p in [out/'BANK.npz',out/'BANK_ROWS.json',out/'BANK_EPISODES.json',out/'BANK_EXCLUSIONS.json']})
    write(out/'BANK.json',bank)
    return rows,arrays,traces,sha(out/'BANK.json')


def episode_weights(rows):
    if not rows:return np.array([],dtype=np.float64)
    sizes=collections.Counter(r['episode_id'] for r in rows)
    per_col=collections.defaultdict(set)
    for r in rows:per_col[r['collector']].add(r['episode_id'])
    w=np.array([1/(len(per_col)*len(per_col[r['collector']])*sizes[r['episode_id']]) for r in rows])
    np.testing.assert_allclose(w.sum(),1.,atol=1e-12)
    return w


def physical_folds(rows):
    groups=sorted(set(physical(r) for r in rows),key=lambda x:digest(list(x)))
    assert len(groups)>=5,'Five group folds require at least five training physical systems.'
    mapping={g:i%5 for i,g in enumerate(groups)}
    return np.array([mapping[physical(r)] for r in rows]),[dict(mass=g[0],length=g[1],fold=mapping[g]) for g in groups]


def ridge_fit(x,y,rows,alpha):
    assert len(x)==len(y)==len(rows) and all(r['split']=='train' for r in rows)
    w=episode_weights(rows);xm=np.sum(w[:,None]*x,0);ym=np.sum(w[:,None]*y,0)
    xs=np.maximum(np.sqrt(np.sum(w[:,None]*(x-xm)**2,0)),1e-8)
    ys=np.maximum(np.sqrt(np.sum(w[:,None]*(y-ym)**2,0)),1e-8)
    xx=(x-xm)/xs;yy=(y-ym)/ys
    # Centering supplies the unpenalized intercept; only slopes receive ridge.
    coef=np.linalg.solve(xx.T@(w[:,None]*xx)+alpha*np.eye(x.shape[1]),xx.T@(w[:,None]*yy))
    return dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,coef=coef,alpha=alpha)


def ridge_predict(fit,x):return ((x-fit['x_mean'])/fit['x_std'])@fit['coef']*fit['y_std']+fit['y_mean']
def serial_fit(fit):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}


def ranks(x):
    order=np.argsort(x,kind='stable');out=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def weighted_corr(a,b,w):
    a=a-np.sum(w*a);b=b-np.sum(w*b);den=np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
    return None if den<=1e-15 else float(np.sum(w*a*b)/den)


def cluster_mean_ci(values,rows,seed=62024):
    """Errors first average within episode; resets resample all collectors jointly."""
    if not rows:return dict(mean=None,ci95=None,reset_clusters=0,episodes=0)
    episodes=collections.defaultdict(list);meta={}
    for v,r in zip(values,rows):episodes[r['episode_id']].append(float(v));meta[r['episode_id']]=r
    keys=list(episodes);means=np.array([np.mean(episodes[k]) for k in keys])
    collectors=np.array([meta[k]['collector'] for k in keys]);seeds=np.array([meta[k]['reset_seed'] for k in keys])
    def aggregate(multiplicity):
        vals=[]
        for arm in sorted(set(collectors)):
            ix=collectors==arm;weights=np.array([multiplicity.get(int(s),0) for s in seeds[ix]])
            if weights.sum():vals.append(float(np.average(means[ix],weights=weights)))
        return float(np.mean(vals)) if vals else np.nan
    unique=np.unique(seeds);point=aggregate({int(s):1 for s in unique});boot=[]
    if len(unique)>=2:
        rng=np.random.RandomState(seed)
        for _ in range(1000):boot.append(aggregate(collections.Counter(map(int,rng.choice(unique,len(unique),replace=True)))))
    return dict(mean=point,ci95=None if not boot else np.percentile(boot,[2.5,97.5]).tolist(),reset_clusters=len(unique),episodes=len(keys),
        bootstrap='1000 paired reset-cluster resamples; collector balanced; one source-training seed')


def probe_metrics(pred,truth,rows):
    if not rows:return dict(coverage=0,rmse=None,r2=None,spearman=None)
    w=episode_weights(rows);squared=(pred-truth)**2;error=cluster_mean_ci(squared,rows)
    mean=np.sum(w*truth);var=np.sum(w*(truth-mean)**2)
    return dict(coverage=len(rows),episodes=error['episodes'],reset_clusters=error['reset_clusters'],
        unique_target_values=len(np.unique(truth)),rmse=float(np.sqrt(np.sum(w*squared))),
        rmse_ci95=None if error['ci95'] is None else np.sqrt(np.maximum(error['ci95'],0)).tolist(),
        r2=None if var<=1e-15 else float(1-np.sum(w*squared)/var),
        spearman=weighted_corr(ranks(pred),ranks(truth),w),
        metric_unit='single-history predictions; episode/collector weights; uncertainty clusters saved reset seeds')


def probe(arm,z,rows,out,inputs,bank_sha):
    train=np.array([r['split']=='train' for r in rows]);control=~train
    tr=[r for r in rows if r['split']=='train'];ev=[r for r in rows if r['split']=='control']
    x=z[train].astype('float64');xe=z[control].astype('float64')
    y=np.array([physical(r) for r in tr]);ye=np.array([physical(r) for r in ev]);folds,mapping=physical_folds(tr)
    fits={};results={};pred=np.zeros_like(ye);cv={}
    for factor,name in enumerate(['mass','length']):
        scores=[]
        for alpha in ALPHAS:
            fold_scores=[]
            for fold in range(5):
                ti=folds!=fold;vi=~ti
                assert set(physical(r) for r,m in zip(tr,ti) if m).isdisjoint(physical(r) for r,m in zip(tr,vi) if m)
                fit=ridge_fit(x[ti],y[ti,factor:factor+1],[r for r,m in zip(tr,ti) if m],alpha)
                val=ridge_predict(fit,x[vi])[:,0]
                w=episode_weights([r for r,m in zip(tr,vi) if m])
                fold_scores.append(float(np.sum(w*((val-y[vi,factor])/fit['y_std'][0])**2)))
            scores.append(dict(alpha=alpha,fold_standardized_mse=fold_scores,mean=float(np.mean(fold_scores))))
        chosen=min(scores,key=lambda s:(s['mean'],s['alpha']))['alpha']
        fit=ridge_fit(x,y[:,factor:factor+1],tr,chosen);pred[:,factor]=ridge_predict(fit,xe)[:,0]
        fits[name]=serial_fit(fit);cv[name]=scores
        results[name]={}
        for group in GROUPS:
            ix=np.array([r['group']==group for r in ev]);rs=[r for r,m in zip(ev,ix) if m]
            results[name][group]=probe_metrics(pred[ix,factor],ye[ix,factor],rs)
            results[name][group]['by_collector']={}
            for collector in ARMS:
                ci=ix&np.array([r['collector']==collector for r in ev])
                results[name][group]['by_collector'][collector]=probe_metrics(pred[ci,factor],ye[ci,factor],[r for r,m in zip(ev,ci) if m])
    np.savez_compressed(out/f'PROBE_{arm}.npz',z=z,prediction=pred,truth=ye,train_mask=train,control_row_indices=np.flatnonzero(control),folds=folds)
    result=dict(model=arm,checkpoint_sha256=inputs['checkpoint_sha256'],config_sha256=inputs['config_sha256'],bank_sha256=bank_sha,
        train_rows=len(tr),evaluation_rows=len(ev),training_physical_group_folds=mapping,alphas=ALPHAS,cv=cv,fits=fits,metrics=results,
        train_latent_std=np.std(x,axis=0).tolist(),near_constant_columns=np.flatnonzero(np.std(x,axis=0)<1e-8).tolist(),
        no_control_data_used_for_fitting_or_selection=True,scope='Parameter accessibility only, not predictor or planner use')
    output=[]
    for r,p,ytrue in zip(ev,pred,ye):
        output.append(dict(model=arm,row=r['row'],collector=r['collector'],group=r['group'],reset_seed=r['reset_seed'],episode_id=r['episode_id'],t=r['t'],
            true_mass=float(ytrue[0]),predicted_mass=float(p[0]),true_length=float(ytrue[1]),predicted_length=float(p[1]),
            checkpoint_sha256=inputs['checkpoint_sha256'],config_sha256=inputs['config_sha256'],bank_sha256=bank_sha))
    return result,output


def pairs(rows):
    """Select only from IDs/metadata, never from predictions or errors."""
    strata=collections.defaultdict(list)
    for r in rows:strata[(r['collector'],r['split'],r['group'],r['t'])].append(r)
    output=[]
    for q in rows:
        if q['split']!='control':continue
        pool=strata[(q['collector'],'train' if q['group']=='ID' else 'control',q['group'],q['t'])]
        independent=[r for r in pool if r['reset_seed']!=q['reset_seed']]
        matches=[r for r in independent if physical(r)==physical(q)]
        choices=[]
        for m in matches:
            wrong=[r for r in independent if r['mass']!=q['mass'] and r['length']!=q['length']
                   and r['reset_seed']!=m['reset_seed'] and (q['group']!='ID' or r['iteration']==m['iteration'])]
            for w in wrong:choices.append((digest([q['episode_id'],q['t'],m['episode_id'],w['episode_id']]),m,w))
        record=dict(recipient=q['row'],recipient_id=q['episode_id'],collector=q['collector'],group=q['group'],reset_seed=q['reset_seed'],t=q['t'],
                    mass=q['mass'],length=q['length'],matched=None,wrong=None,eligible=False,reason=None)
        if not choices:record['reason']='no_independent_same_system_history' if not matches else 'no_two_factor_wrong_in_same_policy_stratum'
        else:
            _,m,w=min(choices,key=lambda x:x[0])
            assert m['collector']==w['collector']==q['collector'] and m['t']==w['t']==q['t']
            assert len({q['reset_seed'],m['reset_seed'],w['reset_seed']})==3
            assert physical(m)==physical(q) and w['mass']!=q['mass'] and w['length']!=q['length']
            if q['group']=='ID':assert m['split']==w['split']=='train' and m['iteration']==w['iteration']
            else:assert m['group']==w['group']==q['group'] and m['split']==w['split']=='control'
            record.update(matched=m['row'],wrong=w['row'],matched_id=m['episode_id'],wrong_id=w['episode_id'],
                matched_reset_seed=m['reset_seed'],wrong_reset_seed=w['reset_seed'],donor_iteration=m['iteration'],
                wrong_mass=w['mass'],wrong_length=w['length'],eligible=True)
        output.append(record)
    return output


@torch.inference_mode()
def encode(model,arrays,device):
    zs=[]
    for i in range(0,len(arrays['cp_obs']),256):
        z=model.encode(torch.as_tensor(arrays['cp_obs'][i:i+256],device=device),torch.as_tensor(arrays['cp_act'][i:i+256],device=device))
        zs.append(z.cpu().numpy())
    result=np.concatenate(zs);assert result.shape==(len(arrays['cp_obs']),10) and np.isfinite(result).all()
    return result


@torch.inference_mode()
def rollout(model,arrays,z,recipients,donors,device):
    result=[]
    for i in range(0,len(recipients),256):
        qi=np.asarray(recipients[i:i+256]);di=np.asarray(donors[i:i+256])
        state=torch.as_tensor(arrays['obs'][qi],device=device);fixed_z=torch.as_tensor(z[di],device=device)
        actions=torch.as_tensor(arrays['actions'][qi],device=device);truth=torch.as_tensor(arrays['targets'][qi],device=device)
        horizons=[]
        for t in range(10):
            state=model.next_observation(state,actions[:,t],fixed_z)
            if t in (0,9):horizons.append((state-truth[:,t]).square().mean(-1).cpu().numpy())
        result.append(np.stack(horizons,axis=1))
    return np.concatenate(result) if result else np.empty((0,2))


def donor_summary(errors,pair_records,rows,valid):
    selected=[p for p in pair_records if p['eligible']];results={}
    for group in GROUPS:
        results[group]={}
        for collector in ['ALL']+ARMS:
            ix=np.array([p['group']==group and (collector=='ALL' or p['collector']==collector) for p in selected])&valid
            rs=[rows[p['recipient']] for p,m in zip(selected,ix) if m]
            cell=dict(recipients=len(rs),episodes=len(set(r['episode_id'] for r in rs)),reset_clusters=len(set(r['reset_seed'] for r in rs)),horizons={})
            for h,label in enumerate(['H1','H10']):
                own,matched,wrong=errors[ix,0,h],errors[ix,1,h],errors[ix,2,h]
                cell['horizons'][label]=dict(own=cluster_mean_ci(own,rs),matched=cluster_mean_ci(matched,rs),wrong=cluster_mean_ci(wrong,rs),
                    wrong_minus_matched=cluster_mean_ci(wrong-matched,rs),own_minus_matched=cluster_mean_ci(own-matched,rs))
            results[group][collector]=cell
    return results


def main(args):
    code,root,out=Path(args.code_root),Path(args.relation_root),Path(args.output)
    assert out.resolve()!=root.resolve() and root.resolve() not in out.resolve().parents,'Use a separate analysis output.'
    assert not any(out.iterdir()),'Analysis outputs preserved; use a fresh output or inspect existing COMPLETE.'
    assert sha(code/'cadm_pendulum_online.py')==BASE_SHA and sha(code/'cadm_pendulum_relation_v2.py')==V2_SHA
    sys.dont_write_bytecode=True;base=module(code/'cadm_pendulum_online.py','frozen_pendulum_analysis_base')
    references={};summaries={}
    for arm in ARMS:references[arm],summaries[arm]=verify_arm(root,arm,code)
    rows,arrays,traces,bank_sha=make_bank(root,out,summaries)
    input_record=dict(status='VERIFIED',relation_root=str(root),code_root=str(code),analysis_sha256=sha(__file__),
        base_sha256=BASE_SHA,relation_code_sha256=V2_SHA,models=references,traces=traces,bank_sha256=bank_sha,
        source_optimizer_updates=0,new_environment_interactions=0,device=args.device)
    write(out/'INPUTS.json',input_record)
    pair_records=pairs(rows);write(out/'DONOR_PAIRS.json',pair_records)
    eligible=[p for p in pair_records if p['eligible']]
    recipients=[p['recipient'] for p in eligible];matched=[p['matched'] for p in eligible];wrong=[p['wrong'] for p in eligible]
    probes={};probe_rows=[];errors={};audits={}
    for arm in ARMS:
        checkpoint=torch.load(references[arm]['checkpoint'],map_location='cpu',weights_only=False)
        assert checkpoint['iteration']==20 and checkpoint['training_interactions']==40000 and checkpoint['config_sha256']==references[arm]['config_sha256']
        model=base.Model('CaDM').to(args.device);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        before=state_sha(model);z=encode(model,arrays,args.device)
        probes[arm],new_rows=probe(arm,z,rows,out,references[arm],bank_sha);probe_rows+=new_rows
        errors[arm]=np.stack([rollout(model,arrays,z,recipients,d,args.device) for d in [recipients,matched,wrong]],axis=1)
        after=state_sha(model);assert before==after and not any(p.requires_grad for p in model.parameters())
        audits[arm]=dict(before_state_sha256=before,after_state_sha256=after,identical=True,normalizer_buffers_preserved=True,
                         gradients_disabled=True,checkpoint_sha256=references[arm]['checkpoint_sha256'])
        print(json.dumps(dict(model=arm,bank_rows=len(rows),eligible_donor_rows=len(eligible),state_unchanged=True)),flush=True)
        del model,checkpoint
    valid=np.ones(len(eligible),dtype=bool);failures=[]
    for arm in ARMS:
        finite=np.isfinite(errors[arm]).all(axis=(1,2));valid&=finite
        for i in np.flatnonzero(~finite):failures.append(dict(model=arm,recipient=eligible[i]['recipient'],reason='nonfinite_frozen_prediction',finite_condition_horizon_mask=np.isfinite(errors[arm][i]).tolist()))
    donor=dict(models={a:donor_summary(errors[a],pair_records,rows,valid) for a in ARMS},bank_sha256=bank_sha,
        candidate_control_rows=len(pair_records),structurally_eligible_rows=len(eligible),common_finite_rows=int(valid.sum()),
        exclusions=dict(collections.Counter(p['reason'] for p in pair_records if not p['eligible'])),prediction_failures=failures,
        masks='Same structural donor pairs for every model; exclude a recipient from all conditions/models if any prediction is nonfinite',
        horizon_definition='Endpoint raw observation MSE after1 or10 recurrent steps with logged normalized actions, fixed donor code; no context update or unit-circle projection',
        scope='Frozen predictor dependence on compatible histories; not pure parameter intervention or a new closed-loop return')
    donor_rows=[]
    for arm in ARMS:
        for i,p in enumerate(eligible):
            row=dict(model=arm,recipient=p['recipient'],collector=p['collector'],group=p['group'],reset_seed=p['reset_seed'],t=p['t'],
                matched=p['matched'],wrong=p['wrong'],matched_id=p['matched_id'],wrong_id=p['wrong_id'],common_finite_eligible=bool(valid[i]),
                checkpoint_sha256=references[arm]['checkpoint_sha256'],config_sha256=references[arm]['config_sha256'],bank_sha256=bank_sha)
            for h,label in enumerate(['H1','H10']):
                for c,name in enumerate(['own','matched','wrong']):
                    value=float(errors[arm][i,c,h]);row[f'{name}_{label}']=value if np.isfinite(value) else ''
                delta=float(errors[arm][i,2,h]-errors[arm][i,1,h]);row[f'wrong_minus_matched_{label}']=delta if np.isfinite(delta) else ''
            donor_rows.append(row)
    np.savez_compressed(out/'DONOR_ERRORS.npz',**errors,common_finite_mask=valid,recipient_rows=np.asarray(recipients),matched_rows=np.asarray(matched),wrong_rows=np.asarray(wrong))
    write(out/'PROBE.json',probes);write(out/'DONOR.json',donor);write(out/'MODEL_STATE_AUDIT.json',audits)
    csv_write(out/'PROBE_ROWS.csv',probe_rows)
    if donor_rows:csv_write(out/'DONOR_ROWS.csv',donor_rows)
    write(out/'SUMMARY.json',dict(status='COMPLETE',models=ARMS,training_source_seeds=1,bank_sha256=bank_sha,
        bank_rows=len(rows),donor_eligible_rows=len(eligible),donor_common_finite_rows=int(valid.sum()),
        probe_uses_all_valid_control_histories=True,probe_training_only_physical_group_cv=True,source_weights_unchanged=True,
        new_environment_interactions=0,source_optimizer_updates=0,no_vanilla_probe=True,
        distinction='Probe accessibility, frozen logged-action donor sensitivity, and already-executed returns answer separate questions',
        actual_return_traces_verified=True,episode_intervals_do_not_measure_training_seed_robustness=True))
    write(out/'COMPLETE.json',dict(status='COMPLETE',files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['COMPLETE.json','EXIT.json','LOCK']}))


def selftest(args):
    """Synthetic checks only: no source fit, simulator, controller or real bank."""
    # Fold isolation and fold-only centering are checked without any control fit.
    rs=[]
    for c in ARMS:
        for g in range(10):
            for t in (10,30):rs.append(dict(collector=c,episode_id=f'{c}/{g}',reset_seed=g,mass=float(g),length=float(g+1),split='train',t=t))
    folds,_=physical_folds(rs)
    for f in range(5):assert set(physical(r) for r,m in zip(rs,folds!=f) if m).isdisjoint(physical(r) for r,m in zip(rs,folds==f) if m)
    x=np.arange(len(rs)*2,dtype=float).reshape(-1,2);y=np.array([[r['mass']] for r in rs]);ti=folds!=0
    fit=ridge_fit(x[ti],y[ti],[r for r,m in zip(rs,ti) if m],.1)
    np.testing.assert_allclose(fit['x_mean'],np.sum(episode_weights([r for r,m in zip(rs,ti) if m])[:,None]*x[ti],0))
    frozen=digest(serial_fit(fit));ridge_predict(fit,np.full((2,2),1e10));assert frozen==digest(serial_fit(fit))
    # ID matched and wrong must come from the same training iteration; same
    # reset under another collector cannot supply an independent matched donor.
    pr=[]
    def add(split,group,seed,mass,length,iteration=None,collector='CaDM'):
        i=len(pr);pr.append(dict(row=i,episode_id=f'{collector}/{i}',collector=collector,split=split,group=group,reset_seed=seed,
                                mass=mass,length=length,iteration=iteration,t=190,episode=i));return i
    q=add('control','ID',900,1.,1.);add('train','ID',10,1.,1.,0);add('train','ID',11,2.,2.,0)
    add('train','ID',12,1.,1.,1);add('train','ID',13,2.,1.,1);add('train','ID',900,1.,1.,0)
    oq=add('control','OOD_c0',901,1.,1.);add('control','OOD_c0',902,1.,1.);add('control','OOD_c0',903,2.,2.)
    add('control','ID',904,4.,4.);add('train','ID',905,4.,4.,0,collector='SPRIIAlign')
    pp=pairs(pr);p=next(v for v in pp if v['recipient']==q);assert p['eligible'] and p['donor_iteration']==0
    assert next(v for v in pp if v['recipient']==oq)['eligible']
    assert not next(v for v in pp if v['reset_seed']==904)['eligible']
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('marker',torch.tensor([.2]));self.seen=[]
        def next_observation(self,obs,action,z):self.seen.append(z.clone());return obs+action+z[:,:3]
    toy=Toy();arr=dict(obs=np.zeros((1,3),np.float32),actions=np.ones((1,10,1),np.float32)*.1,targets=np.zeros((1,10,3),np.float32))
    z=np.ones((1,10),np.float32)*.2;before=state_sha(toy);err=rollout(toy,arr,z,[0],[0],'cpu')
    np.testing.assert_allclose(err,[[.09,9.]],rtol=2e-6);assert state_sha(toy)==before
    assert all(torch.equal(toy.seen[0],v) for v in toy.seen)
    assert not any(p.requires_grad for p in toy.parameters())
    result=dict(status='PASS',code_sha256=sha(__file__),environment_interactions=0,source_optimizer_updates=0,
        physical_group_cv_disjoint=True,train_only_stats_preserved_under_extreme_eval=True,
        strict_independent_same_collector_same_time_donors=True,id_same_training_iteration=True,
        wrong_both_factors_different=True,no_cross_collector_match_fallback=True,
        fixed_z_all10_steps=True,normalized_actions_no_extra_torque_scaling=True,h1_h10_endpoint_checked=True,
        model_buffers_unchanged=True)
    write(Path(args.output)/'SELFTEST.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--relation-root');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--self-test',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Lock resides beside output, so an empty output remains detectable.
    with out.with_name(out.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.self_test:selftest(args)
            else:assert args.relation_root;main(args)
            write(out/'EXIT.json',dict(exit_code=0,time=time.time(),code_sha256=sha(__file__)))
        except Exception as exc:
            write(out/f'FAILED_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            write(out/'EXIT.json',dict(exit_code=1,time=time.time(),code_sha256=sha(__file__)));raise
