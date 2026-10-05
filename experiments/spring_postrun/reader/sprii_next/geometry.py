"""Frozen source geometry. Reader checkpoints are not accepted by this API."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata


def partial_spearman(x,y,controls):
    a=np.column_stack([np.ones(len(x))]+[rankdata(c) for c in controls])
    def residual(v):
        r=rankdata(v)
        return r-a@np.linalg.lstsq(a,r,rcond=None)[0]
    rx,ry=residual(x),residual(y)
    denom=np.linalg.norm(rx)*np.linalg.norm(ry)
    return None if denom<1e-12 else float(rx@ry/denom)


def source_geometry(provider):
    if provider.environment!='springworld' or provider.descriptor['method'] not in ('Structure','Align','Cross','Both'):
        raise ValueError('light geometry permits only four SpringWorld frozen source interfaces')
    train,_,_,_=provider.donors('train')
    p,theta,systems,donors=provider.donors('validation')
    # Source history normalization is fitted on training histories only.
    p=(p-train.mean(0))/train.std(0).clip(1e-6)
    ids=np.unique(systems);centroids=[];within=[];parameters=[]
    for sid in ids:
        sel=np.flatnonzero(systems==sid)
        if len(sel)<2:raise ValueError('within-system geometry requires independent histories')
        if len(np.unique(donors[sel]))!=len(sel):raise ValueError('duplicate donor histories')
        within.append(np.square(pdist(p[sel])).mean())
        centroids.append(p[sel].mean(0));parameters.append(theta[sel[0]])
    distances=pdist(np.asarray(centroids));theta=np.asarray(parameters)
    factors=dict(mass=pdist(np.log(theta[:,0,None])),drag=pdist(np.log(theta[:,1,None])),
                 stiffness=pdist(np.log(theta[:,2,None])),k_over_m=pdist(np.log((theta[:,2]/theta[:,0])[:,None])))
    # Do not condition rho(k/m) on BOTH m and k: it is their deterministic ratio.
    # Original three factors control one another; the ratio controls drag only.
    controls={'mass':['drag','stiffness'],'drag':['mass','stiffness'],
              'stiffness':['mass','drag'],'k_over_m':['drag']}
    within=float(np.mean(within));between=float(np.square(distances).mean())
    return dict(source=provider.descriptor,representation='frozen_source_P64',reader_used=False,
        source_optimizer_steps=0,test_read=False,systems=len(ids),histories=len(p),
        D_within=within,D_between=between,between_within_ratio=None if within<=1e-12 else between/within,
        factor_coordinates='log_m, log_gamma, log_k, log_k_over_m',
        rho={k:partial_spearman(distances,v,[factors[c] for c in controls[k]]) for k,v in factors.items()},
        rho_controls=controls,normalization='training donor histories only',
        distance_definition='squared Euclidean within independent histories; squared Euclidean between system centroids')
