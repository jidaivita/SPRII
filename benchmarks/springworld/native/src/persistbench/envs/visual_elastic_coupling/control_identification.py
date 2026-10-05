"""Online physical inference from the actual, causally available control prefix.

The public start is rest at natural length. All four measured object coordinates
and executed force intervals enter a full nonlinear trajectory fit. Donor
information is included once; growing overlapping query prefixes replace the
previous query likelihood instead of being added repeatedly.
"""
import numpy as np
from scipy.optimize import least_squares
from .schema import Config
from .calibration import identify_positions
from .fast_reference import reference_rollout_fast
from .optimization import select_solution


def rest_state(center,angle,config):
    relative=config.ell0*np.array([np.cos(angle),np.sin(angle)])
    return np.r_[center-relative/2,center+relative/2,np.zeros(4)]


def information_residual(H,b):
    H=np.asarray(H,float);b=np.asarray(b,float)
    if H.shape!=(3,3) or b.shape!=(3,) or not np.isfinite(H).all() or not np.isfinite(b).all():raise ValueError('invalid donor information')
    if not np.allclose(H,H.T,rtol=1e-8,atol=1e-8):raise ValueError('asymmetric donor information')
    values,vectors=np.linalg.eigh((H+H.T)/2)
    if values.min()<-1e-8*max(1.,values.max()):raise ValueError('negative donor precision')
    active=values>max(1e-12,values.max()*1e-10)
    design=np.sqrt(values[active])[:,None]*vectors[:,active].T
    target=(vectors[:,active].T@b)/np.sqrt(values[active])
    return design,target


def fit_control_prefix(positions,actions,H,b,config=Config(resolution=128),*,noise_m=.0006,max_nfev=100,initial_parameters=None):
    measured=np.asarray(positions,float);actions=np.asarray(actions,float)
    if measured.ndim!=2 or measured.shape[1]!=4 or len(measured)<6 or not np.isfinite(measured).all():raise ValueError('insufficient online visual support')
    if actions.shape!=(len(measured)-1,2) or not np.isfinite(actions).all():raise ValueError('online actions must cover only the observed prefix')
    if noise_m<=0 or max_nfev<1:raise ValueError('invalid online fitting budget')
    design,target=information_residual(H,b);low=np.array([.5,.25,4.]);high=np.array([2.,1.5,25.])
    center=(measured[0,:2]+measured[0,2:])/2;relative=measured[0,2:]-measured[0,:2];angle=float(np.arctan2(relative[1],relative[0]));pixel=config.field_width/config.resolution
    lower=np.r_[np.log(low),center-2*pixel,angle-np.pi/2];upper=np.r_[np.log(high),center+2*pixel,angle+np.pi/2]
    diagnostic=identify_positions(measured,actions,config);beta=np.asarray(diagnostic['beta'])
    mass=np.clip(1/beta[0] if beta[0]>1e-10 else 1.25,low[0],high[0]);warm=np.array([mass,np.clip(beta[1],low[1],high[1]),np.clip(mass*beta[2],low[2],high[2])])
    starts=[warm,np.array([2.,np.sqrt(.25*1.5),4.]),np.array([.5,np.sqrt(.25*1.5),25.])]
    if initial_parameters is not None:starts.insert(0,np.clip(np.asarray(initial_parameters,float),low,high))
    def residual(z):
        initial=rest_state(z[3:5],z[5],config)
        prediction=reference_rollout_fast(np.exp(z[:3]),initial,actions,config)
        visual=((prediction[:,:4]-measured)/noise_m).ravel()
        return np.r_[visual,design@z[:3]-target]
    fits=[];attempts=[]
    for parameters in starts:
        seed=np.clip(np.r_[np.log(parameters),center,angle],lower,upper)
        try:
            fit=least_squares(residual,seed,bounds=(lower,upper),x_scale=[1,1,1,pixel,pixel,.1],
                max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
            fits.append(fit);attempts.append(dict(success=bool(fit.success),cost=float(fit.cost),nfev=int(fit.nfev)))
        except (ValueError,FloatingPointError) as exc:attempts.append(dict(success=False,error=str(exc)))
    if not fits:raise ValueError('all online visual fit attempts failed')
    chosen=select_solution(fits);parameters=np.exp(chosen.x[:3]);initial=rest_state(chosen.x[3:5],chosen.x[5],config)
    path=reference_rollout_fast(parameters,initial,actions,config)
    return dict(parameters=parameters.tolist(),current_state=path[-1].tolist(),optimizer_success=bool(chosen.success),
        visual_rmse_m=float(np.sqrt(np.mean((path[:,:4]-measured)**2))),cost=float(chosen.cost),attempts=attempts,
        observed_frames=len(measured),executed_actions=len(actions),donor_factor_rank=int(len(target)),
        estimator='bounded physical-coordinate MAP with local donor log-likelihood and full current visual prefix; known public rest/natural-length initial state; no repeated-prefix likelihood accumulation')
