"""Vectorized explicit-reference forecasts, checked against independent DOP853.

This is an inference accelerator. It never generates benchmark observations;
the official stage-checked MuJoCo generator remains the only data kernel.
"""
import numpy as np
from .schema import Config


def reference_rollout_batch(parameters,initial,actions,config=Config()):
    p=np.asarray(parameters,float);z=np.asarray(initial,float).copy();actions=np.asarray(actions,float)
    if p.ndim!=2 or p.shape[1]!=3 or z.shape!=(len(p),8) or actions.ndim!=2 or actions.shape[1]!=2:
        raise ValueError('batched reference shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(z).all() or not np.isfinite(actions).all() or np.any(p[:,0]<=0) or np.any(p[:,1]<0) or np.any(p[:,2]<=0):
        raise ValueError('invalid batched reference values')
    substeps=round(config.control_dt/config.dt)
    if substeps<1 or not np.isclose(substeps*config.dt,config.control_dt,atol=1e-12,rtol=0):raise ValueError('noninteger integration interval')
    mass=p[:,0,None];gamma=p[:,1,None];ratio=(p[:,2]/p[:,0])[:,None];dt=config.dt
    result=np.empty((len(z),len(actions)+1,8));result[:,0]=z
    def rhs(state,force):
        r=state[:,2:4]-state[:,:2];length=np.linalg.norm(r,axis=1)[:,None]
        if np.any(length<1e-7) or not np.isfinite(state).all():raise ValueError('batched reference spring singularity')
        spring=ratio*(1-config.ell0/length)*r
        return np.concatenate((state[:,4:],force/mass-gamma*state[:,4:6]+spring,-gamma*state[:,6:8]-spring),axis=1)
    for t,action in enumerate(actions):
        force=action[None]*config.force_max
        for _ in range(substeps):
            a=rhs(z,force);b=rhs(z+dt*a/2,force);c=rhs(z+dt*b/2,force);d=rhs(z+dt*c,force)
            z=z+dt/6*(a+2*b+2*c+d)
        result[:,t+1]=z
    return result
