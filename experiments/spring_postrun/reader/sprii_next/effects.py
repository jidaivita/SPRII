"""Conditional predictor sensitivity and optional physical effect alignment."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .io import digest,read,sha,write,npz
from .model import Reader,HORIZONS
from .poke_simulator import replay
from .statistics import verify_gate


def controlled_pairs(theta,ids,factor,max_pairs=64,seed=20260919):
    groups=defaultdict(list)
    for i,t in enumerate(theta):
        key=tuple(float(t[j]).hex() for j in range(3) if j!=factor)
        groups[key].append(i)
    pairs=[]
    for g in groups.values():
        # Preserve a range of factor distances: adjacent levels alone make rho
        # undefined on the native log-spaced mass grid.
        ordered=sorted(g,key=lambda i:(theta[i,factor],str(ids[i])))
        levels={}
        for i in ordered:levels.setdefault(float(theta[i,factor]),i)
        values=list(levels.values())
        pairs.extend((values[a],values[b]) for a in range(len(values)) for b in range(a+1,len(values)))
    if not pairs:raise ValueError('no exact one-factor-controlled donor pairs; do not relax controls silently')
    rng=np.random.default_rng(seed)
    chosen=np.sort(rng.choice(len(pairs),min(max_pairs,len(pairs)),replace=False))
    return [pairs[i] for i in chosen]


def effect_analysis(provider,run_dir,gate_path,output,*,device='cpu',max_pairs=64,recipient_systems=32,
                    target_alignment_review=None,simulator_config=None):
    root=Path(run_dir);run=read(root/'RUN.json');complete=read(root/'COMPLETE.json')
    verify_gate(gate_path,run['protocol_sha256'])
    if complete.get('smoke') or run['job']['arm'] not in ('matched','persistent'):
        raise ValueError('completed non-smoke matched reader required')
    if complete['checkpoint_sha256']!=sha(root/'head.pt') or complete['run_sha256']!=sha(root/'RUN.json'):
        raise ValueError('reader changed')
    if run['provider_sha256']!=provider.identity:raise ValueError('effect source differs from trained reader')
    if provider.environment!='pokeworld':raise ValueError('controlled simulator analysis is PokeWorld only')
    if target_alignment_review is not None:
        review=read(target_alignment_review)
        if review.get('decision')!='run_target_alignment' or not review.get('reason'):
            raise PermissionError('explicit review of clear sensitivity pattern required')
        if sha(review['sensitivity_result'])!=review['sensitivity_result_sha256']:
            raise ValueError('sensitivity review changed')
        prior=read(review['sensitivity_result'])
        if prior['run_sha256']!=sha(root/'RUN.json'):raise ValueError('sensitivity review belongs to another reader')
        if simulator_config is None:raise ValueError('native simulator configuration required')
    ck=torch.load(root/'head.pt',map_location='cpu',weights_only=False)
    head=Reader(run['job']['arm'],run['job']['reader_seed']).to(device)
    head.load_state_dict(ck['model']);head.eval().requires_grad_(False)
    p,theta,systems,donors=provider.donors('validation')
    # One deterministic recipient window per physical system; common across encoders.
    rows=provider.rows['validation'];first={}
    for i,r in enumerate(rows):first.setdefault(r['system_id'],i)
    ordered=sorted(first)
    rng=np.random.default_rng(20260919)
    chosen=np.sort(rng.choice(len(ordered),min(recipient_systems,len(ordered)),replace=False))
    ri=np.asarray([first[ordered[i]] for i in chosen])
    b=provider.batch('validation',ri,np.full(len(ri),4))
    results=[];vectors=[];identities=[]
    with torch.inference_mode():
        common=[torch.tensor(v,device=device,dtype=torch.float32) for v in (b.query,b.actions,b.mask)]
        hi=torch.tensor(b.horizon_index,device=device)
        for factor,name in enumerate(('mass','drag')):
            pairs=controlled_pairs(theta,donors,factor,max_pairs)
            distances=[]
            for i,j in pairs:
                predictions=[]
                for idx in (i,j):
                    code=torch.tensor(np.repeat(p[idx:idx+1],len(ri),axis=0),device=device,dtype=torch.float32)
                    predictions.append(head(common[0],code,common[1],common[2],hi).cpu().numpy())
                effect=predictions[1]-predictions[0];norm=np.linalg.norm(effect,axis=1)
                distances.append(float(norm.mean()))
                for n,r in enumerate(b.rows):
                    entry=dict(r,factor=name,source_method=run['job']['method'],source_seed=run['job']['source_seed'],
                        reader_seed=run['job']['reader_seed'],stage='development',
                        donor_i=donors[i],donor_j=donors[j],donor_system_i=systems[i],donor_system_j=systems[j],
                        theta_i=theta[i].tolist(),theta_j=theta[j].tolist(),sensitivity=float(norm[n]),
                        error_i=float(np.square(predictions[0][n]-b.target[n]).mean()),error_j=float(np.square(predictions[1][n]-b.target[n]).mean()))
                    true=np.full(8,np.nan)
                    if target_alignment_review is not None:
                        initial=np.asarray(r['initial_state'])[None,:];actions=b.actions[n:n+1]
                        y0,_=replay(initial,actions,theta[i:i+1],simulator_config)
                        y1,_=replay(initial,actions,theta[j:j+1],simulator_config)
                        scale=np.asarray(provider.normalization['target_scale'])[4]
                        true=(y1[0,-1]-y0[0,-1])/scale
                        denom=float(np.linalg.norm(true));pred_norm=float(norm[n]);epsilon=1e-8
                        entry.update(target_effect_norm=denom,cosine=None if denom<epsilon or pred_norm<epsilon else float(effect[n]@true/(denom*pred_norm)),
                                     relative_cf_error=None if denom<epsilon else float(np.linalg.norm(effect[n]-true)/(denom+epsilon)))
                    identities.append(entry);vectors.append((effect[n],true,p[i],p[j],predictions[0][n],predictions[1][n],b.target[n],b.query[n],b.actions[n]))
            from scipy.stats import spearmanr
            fd=np.array([abs(np.log(theta[j,factor])-np.log(theta[i,factor])) for i,j in pairs])
            rho=None if fd.std()<1e-12 or np.std(distances)<1e-12 else float(spearmanr(fd,distances).statistic)
            results.append(dict(factor=name,mean_sensitivity=float(np.mean(distances)),pairs=len(pairs),rho_sensitivity=rho))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    npz(out/'vectors.npz',prediction_effect=np.asarray([v[0] for v in vectors]),true_counterfactual_effect=np.asarray([v[1] for v in vectors]),
        persistent_i=np.asarray([v[2] for v in vectors]),persistent_j=np.asarray([v[3] for v in vectors]),
        prediction_i=np.asarray([v[4] for v in vectors]),prediction_j=np.asarray([v[5] for v in vectors]),target_vector=np.asarray([v[6] for v in vectors]),
        error_per_dimension_i=np.asarray([(v[4]-v[6])**2 for v in vectors]),error_per_dimension_j=np.asarray([(v[5]-v[6])**2 for v in vectors]),
        query_embedding=np.asarray([v[7] for v in vectors]),future_actions=np.asarray([v[8] for v in vectors]))
    write(out/'rows.json',identities)
    write(out/'RESULT.json',dict(name='predictor_induced_sensitivity_geometry',stage='development',test_read=False,
        run_sha256=sha(root/'RUN.json'),provider_sha256=provider.identity,gate_sha256=sha(gate_path),
        target_alignment=target_alignment_review is not None,results=results,recipients=len(ri),
        selection='fixed seed, one window per system, exact one-factor pairs',
        vectors_sha256=sha(out/'vectors.npz'),rows_sha256=sha(out/'rows.json')))


def balls_reanalysis(path):
    """Existing prediction vectors only. No inference or source training."""
    with np.load(path,allow_pickle=False) as f:
        needed=('matched_prediction','mass_prediction','friction_prediction','restitution_prediction','target','system_id')
        if any(k not in f for k in needed):raise ValueError('full vectors unavailable; keep existing error summaries, do not rerun')
        matched=f['matched_prediction'];ids=f['system_id'];out={}
        for factor in ('mass','friction','restitution'):
            changed=f[factor+'_prediction']
            if changed.shape!=matched.shape:raise ValueError('prediction shapes differ')
            values=np.linalg.norm((changed-matched).reshape(len(ids),-1),axis=1)
            out[factor]=float(np.mean([values[ids==s].mean() for s in np.unique(ids)]))
    return dict(stage='development',source_optimizer_steps=0,new_inference=False,
        definition='mean Euclidean prediction change; physical-system macro average',sensitivity=out)
