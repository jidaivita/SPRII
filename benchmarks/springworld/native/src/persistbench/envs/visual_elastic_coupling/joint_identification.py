"""Shared-parameter fit with separate initial-state nuisance per episode.

This is the full-history calibration reference, not a compressed-memory agent.
Duplicate observations are not counted as independent likelihood observations.
"""
import hashlib
import numpy as np
from scipy.optimize import least_squares,minimize
from .schema import Config
from .identification import fit_positions,reference_rollout
from .optimization import select_solution


def history_digest(positions,actions):
    h=hashlib.sha256()
    for value in (positions,actions):
        array=np.ascontiguousarray(value)
        h.update(str((array.shape,str(array.dtype))).encode());h.update(array.tobytes())
    return h.hexdigest()


def information_factor(fit,config=Config(),noise_floor_m=.0006):
    """Local Gaussian factor in global log(m,gamma,k) coordinates.

    Noise is a declared image-tracking approximation. These factors are not
    exact pixel likelihoods, and require predictive/coverage calibration.
    """
    if not fit['optimizer_success']:return np.zeros((3,3)),np.zeros(3)
    mode=np.asarray(fit['parameter_log_mode'])
    raw=np.asarray(fit['parameter_log_information_per_pixel'])
    sigma=max(noise_floor_m,fit['position_residual_rmse_m'])/(config.field_width/config.resolution)
    precision=raw/sigma**2
    mapping=np.eye(3) if fit['parameter_dimension']==3 else np.array([[0.,1.,0.],[-1.,0.,1.]])
    return mapping.T@precision@mapping,mapping.T@precision@mode


def fit_histories(histories,config=Config(),*,starts=3,max_nfev=150,
                  parameter_bounds=((.5,.25,4.),(2.,1.5,25.))):
    if not histories:raise ValueError('at least one observed history is required')
    unique={}
    for history in histories:
        x=np.asarray(history['positions'],float);a=np.asarray(history['actions'],float)
        if x.ndim!=2 or x.shape[1]!=4 or a.shape!=(len(x)-1,2) or len(x)<3:
            raise ValueError('invalid history support')
        digest=history.get('fingerprint',history_digest(x,a))
        unique.setdefault(digest,dict(positions=x,actions=a))
    observed=list(unique.values());forced=any(np.any(np.abs(h['actions'])>1e-12) for h in observed)
    fits=[fit_positions(h['positions'],h['actions'],config,starts=starts,parameter_bounds=parameter_bounds) for h in observed]
    low,high=np.asarray(parameter_bounds,float);pixel=config.field_width/config.resolution
    plow=low if forced else np.array([low[1],low[2]/high[0]])
    phigh=high if forced else np.array([high[1],high[2]/low[0]])
    ntheta=len(plow)
    if len(observed)==1:
        result=dict(fits[0]);result.update(unique_histories=1,processed_histories=len(histories),deduplicated=len(histories)-1)
        result['initial_states']=[result['initial_state']]
        return result
    lower=np.concatenate([np.log(plow)]+[np.r_[h['positions'][0]-2*pixel,[-2.]*4] for h in observed])
    upper=np.concatenate([np.log(phigh)]+[np.r_[h['positions'][0]+2*pixel,[2.]*4] for h in observed])
    nuisance=np.concatenate([np.asarray(f['initial_state']) for f in fits])
    H=np.zeros((3,3));b=np.zeros(3)
    for f in fits:
        fh,fb=information_factor(f,config);H+=fh;b+=fb
    lo=np.log(low);hi=np.log(high)
    mode=minimize(lambda x:.5*x@H@x-b@x,(lo+hi)/2,jac=lambda x:H@x-b,
                  bounds=list(zip(lo,hi)),method='L-BFGS-B').x
    first=mode if forced else np.array([mode[1],mode[2]-mode[0]])
    seeds=[first]
    for fraction in np.linspace(0.,1.,starts-1) if starts>1 else []:
        raw=np.array([high[0]*(low[0]/high[0])**fraction,np.sqrt(low[1]*high[1]),low[2]*(high[2]/low[2])**fraction])
        seeds.append(np.log(raw) if forced else np.log([raw[1],raw[2]/raw[0]]))
    def decode(z):
        p=np.exp(z[:ntheta]);return p if forced else np.array([1.,p[0],p[1]])
    def residual(z):
        p=decode(z);result=[]
        for i,h in enumerate(observed):
            initial=z[ntheta+8*i:ntheta+8*(i+1)]
            predicted=reference_rollout(p,initial,h['actions'],config)
            result.append(((predicted[:,:4]-h['positions'])/pixel).ravel())
        return np.concatenate(result)
    attempts=[];solutions=[]
    for seed in seeds:
        z=np.clip(np.r_[seed,nuisance],lower,upper)
        try:
            fit=least_squares(residual,z,bounds=(lower,upper),
                x_scale=np.r_[np.ones(ntheta),np.tile([pixel]*4+[.2]*4,len(observed))],
                max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
            attempts.append(dict(success=bool(fit.success),cost=float(fit.cost),nfev=int(fit.nfev)))
            solutions.append(fit)
        except (ValueError,FloatingPointError) as exc:attempts.append(dict(success=False,error=str(exc)))
    if not solutions:raise ValueError('all joint fit starts failed')
    best=select_solution(solutions);j=best.jac;jn=j[:,ntheta:]
    projected=j[:,:ntheta]-jn@np.linalg.lstsq(jn,j[:,:ntheta],rcond=1e-9)[0]
    sv=np.linalg.svd(projected,compute_uv=False);rank=int(np.count_nonzero(sv>max(1e-6,sv[0]*1e-6)))
    p=decode(best.x);identified=rank==ntheta and best.success
    return dict(m=float(p[0]) if forced and identified else None,gamma=float(p[1]),
        k=float(p[2]) if forced and identified else None,k_over_m=float(p[2]/p[0]),
        initial_states=best.x[ntheta:].reshape(-1,8).tolist(),
        position_residual_rmse_m=float(np.sqrt(np.mean(best.fun**2))*pixel),
        local_parameter_rank=rank,parameter_dimension=ntheta,parameter_log_mode=best.x[:ntheta].tolist(),
        parameter_log_information_per_pixel=(projected.T@projected).tolist(),
        parameter_log_singular_values_per_pixel=sv.tolist(),parameter_bounds=[low.tolist(),high.tolist()],
        structural_mass_stiffness_ambiguity=not forced,optimizer_success=bool(best.success),
        status='locally_identified' if identified else 'weak_or_unidentified',
        unique_histories=len(observed),processed_histories=len(histories),deduplicated=len(histories)-len(observed),
        attempts=attempts,uncertainty='local sensitivity; separate initial state per unique episode; no global identifiability claim')
