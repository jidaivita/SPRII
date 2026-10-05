"""Compiled float64 RK4 reference for repeated online inference, not data.

Benchmark states/images still come only from the checked MuJoCo generator.
This accelerator must agree with the independent DOP853 reference before use.
"""
import numpy as np
from .schema import Config

try:
    import numba
    njit=numba.njit
    COMPILER_VERSION=numba.__version__
except ImportError:
    def njit(*args,**kwargs):return lambda function:function
    COMPILER_VERSION=None


@njit(cache=True)
def _rhs(state,mass,gamma,ratio,ell0,ux,uy):
    out=np.empty(8,np.float64);rx=state[2]-state[0];ry=state[3]-state[1];length=np.sqrt(rx*rx+ry*ry)
    if length<1e-7:raise ValueError('reference spring singularity')
    spring=ratio*(1.-ell0/length)
    for i in range(4):out[i]=state[i+4]
    out[4]=ux/mass-gamma*state[4]+spring*rx;out[5]=uy/mass-gamma*state[5]+spring*ry
    out[6]=-gamma*state[6]-spring*rx;out[7]=-gamma*state[7]-spring*ry
    return out


@njit(cache=True)
def _rollout(parameters,initial,actions,dt,substeps,ell0,force_max):
    state=initial.copy();out=np.empty((len(actions)+1,8),np.float64);out[0]=state
    mass,gamma,stiffness=parameters;ratio=stiffness/mass
    for t in range(len(actions)):
        ux,uy=actions[t]*force_max
        for _ in range(substeps):
            a=_rhs(state,mass,gamma,ratio,ell0,ux,uy)
            b=_rhs(state+dt*a/2,mass,gamma,ratio,ell0,ux,uy)
            c=_rhs(state+dt*b/2,mass,gamma,ratio,ell0,ux,uy)
            d=_rhs(state+dt*c,mass,gamma,ratio,ell0,ux,uy)
            state=state+dt/6*(a+2*b+2*c+d)
        out[t+1]=state
    return out


def reference_rollout_fast(parameters,initial,actions,config=Config()):
    p=np.asarray(parameters,np.float64);s=np.asarray(initial,np.float64);a=np.asarray(actions,np.float64)
    if p.shape!=(3,) or s.shape!=(8,) or a.ndim!=2 or a.shape[1]!=2:raise ValueError('reference shape mismatch')
    if not all(np.isfinite(x).all() for x in (p,s,a)) or p[0]<=0 or p[1]<0 or p[2]<=0:raise ValueError('invalid reference inputs')
    substeps=round(config.control_dt/config.dt)
    if substeps<1 or not np.isclose(substeps*config.dt,config.control_dt,rtol=0,atol=1e-12):raise ValueError('noninteger reference interval')
    result=_rollout(p,s,a,float(config.dt),substeps,float(config.ell0),float(config.force_max))
    if not np.isfinite(result).all():raise ValueError('nonfinite reference result')
    return result
