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
    train_mean=train.mean(0);train_scale=train.std(0).clip(1e-6)
    p=(p-train_mean)/train_scale
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
    rho={k:partial_spearman(distances,v,[factors[c] for c in controls[k]]) for k,v in factors.items()}
    return dict(source=provider.descriptor,representation='frozen_source_P64',reader_used=False,
        source_optimizer_steps=0,test_read=False,systems=len(ids),histories=len(p),
        D_within=within,D_between=between,between_within_ratio=None if within<=1e-12 else between/within,
        factor_coordinates='log_m, log_gamma, log_k, log_k_over_m',
        rho=rho,rho_m=rho['mass'],rho_gamma=rho['drag'],rho_k=rho['stiffness'],rho_k_over_m=rho['k_over_m'],
        rho_controls=controls,normalization='training donor histories only',
        train_mean=train_mean.tolist(),train_scale=train_scale.tolist(),
        distance_definition='squared Euclidean within independent histories; squared Euclidean between system centroids')


def geometry_summary(results):
    """Report the fixed four-by-three source grid without reader outputs."""
    from pathlib import Path
    from .io import read,sha
    cells={};population=None
    for path in results:
        path=Path(path);record=read(path);source=record['source']
        key=(source['method'],source['source_seed'])
        if key in cells:raise ValueError('duplicate geometry source cell')
        if record['reader_used'] is not False or record['test_read'] is not False:raise ValueError('frozen development geometry required')
        vector_path=path.parent/'source_vectors.npz'
        if sha(vector_path)!=record['source_vectors_sha256']:raise ValueError('geometry vectors changed')
        with np.load(vector_path,allow_pickle=False) as vectors:
            identity=(vectors['system_id'],vectors['donor_id'],vectors['physical_parameters'])
            if population is None:population=tuple(v.copy() for v in identity)
            elif any(not np.array_equal(a,b) for a,b in zip(population,identity)):
                raise ValueError('geometry source populations differ')
        cells[key]=record
    methods=('Structure','Align','Cross','Both')
    if set(cells)!={(m,s) for m in methods for s in range(3)}:raise ValueError('complete four-method by three-source geometry required')
    fields=('D_within','D_between','between_within_ratio','rho_m','rho_gamma','rho_k','rho_k_over_m')
    def average(values):return None if any(v is None for v in values) else float(np.mean(values))
    return dict(test_read=False,reader_used=False,representation='frozen_source_P64',complete_sources=12,
        by_source=[dict(method=m,source_seed=s,**{k:cells[m,s][k] for k in fields}) for m in methods for s in range(3)],
        mean_by_method={m:{k:average([cells[m,s][k] for s in range(3)]) for k in fields} for m in methods},
        aggregation='equal source seeds; within distances equal physical systems; between/rho use unordered system pairs',
        systems=cells['Both',0]['systems'],histories=cells['Both',0]['histories'])
