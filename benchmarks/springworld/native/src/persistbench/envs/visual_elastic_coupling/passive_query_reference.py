"""Conditional visual reference for the registered passive-prefix95 query.

This particular prefix has zero force and zero initial relative velocity, so
orientation is constant and its radial motion is exactly a damped oscillator.
The forced future still uses the full nonlinear central-spring equations.
Only public pixels and the declared initial-state distribution are used. A
Gaussian tracking likelihood and plug-in visible orientation are approximations.
This is not a general-query Bayes oracle or an exact pixel likelihood.
"""
import numpy as np
from scipy.special import log_ndtr,logsumexp
from scipy.stats import qmc,truncnorm
from numpy.polynomial.legendre import leggauss
from .schema import Config
from .calibration import track
from .identification import reference_rollout
from .batch_prediction import reference_rollout_batch
from .observations import public_query_images


def log_normal_interval(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    if np.any(a>=b):raise ValueError('empty normal interval')
    # Use the opposite tail when both endpoints are large and positive.
    flip=a+b>0;lo=np.where(flip,-b,a);hi=np.where(flip,-a,b)
    upper=log_ndtr(hi);lower=log_ndtr(lo)
    return upper+np.log(-np.expm1(np.minimum(lower-upper,0.)))


def radial_solution(gamma,ratio,times):
    g=np.atleast_1d(gamma)[:,None];q=np.atleast_1d(ratio)[:,None];t=np.asarray(times)[None]
    frequency2=2*q-g*g/4
    if np.any(frequency2<=0):raise ValueError('passive profile requires underdamped radial support')
    w=np.sqrt(frequency2);decay=np.exp(-g*t/2)
    return decay*(np.cos(w*t)+g/(2*w)*np.sin(w*t)), -decay*(2*q/w)*np.sin(w*t)


def mass_factor(H,b,gamma,ratio,bounds=((.5,.25,4.),(2.,1.5,25.))):
    """Integrate m analytically, keeping k=m*ratio and the physical prior Jacobian.

    Density in (m,gamma,ratio) is proportional to m. In log(m) integration the
    exponent consequently has +2 log(m), not +1 or an independent flat mass.
    """
    gamma,ratio=np.broadcast_arrays(gamma,ratio);low,high=np.asarray(bounds,float)
    lo=np.log(np.maximum(low[0],low[2]/ratio));hi=np.log(np.minimum(high[0],high[2]/ratio))
    if np.any(lo>=hi):raise ValueError('ratio outside continuous physical-prior support')
    v=np.array([1.,0.,1.]);w=np.stack((np.zeros_like(gamma),np.log(gamma),np.log(ratio)),axis=-1)
    a=float(v@H@v);linear=2+v@b-np.einsum('...i,i->...',w,H@v)
    constant=np.einsum('...i,i->...',w,b)-.5*np.einsum('...i,ij,...j->...',w,H,w)
    if a>1e-7:
        mu=linear/a;sd=1/np.sqrt(a)
        integral=constant+.5*linear**2/a+.5*np.log(2*np.pi/a)+log_normal_interval((lo-mu)/sd,(hi-mu)/sd)
    else:
        integral=np.empty_like(linear);positive=linear>1e-7;negative=linear< -1e-7;zero=~(positive|negative)
        integral[positive]=linear[positive]*hi[positive]+np.log(-np.expm1(linear[positive]*(lo[positive]-hi[positive])))-np.log(linear[positive])
        integral[negative]=linear[negative]*lo[negative]+np.log(-np.expm1(linear[negative]*(hi[negative]-lo[negative])))-np.log(-linear[negative])
        integral[zero]=np.log(hi[zero]-lo[zero]);integral+=constant
    return integral,dict(curvature=a,linear=linear,lo=lo,hi=hi)


def draw_mass(factor,u):
    a=factor['curvature'];p=factor['linear'];lo=factor['lo'];hi=factor['hi'];u=np.asarray(u)
    if a>1e-7:
        mu=p/a;sd=1/np.sqrt(a)
        return np.exp(truncnorm.ppf(u,(lo-mu)/sd,(hi-mu)/sd,loc=mu,scale=sd))
    logmass=np.empty_like(p);positive=p>1e-7;negative=p< -1e-7;zero=~(positive|negative)
    logmass[positive]=hi[positive]+np.log(u[positive]+(1-u[positive])*np.exp(p[positive]*(lo[positive]-hi[positive])))/p[positive]
    logmass[negative]=lo[negative]+np.log(1-u[negative]+u[negative]*np.exp(p[negative]*(hi[negative]-lo[negative])))/p[negative]
    logmass[zero]=lo[zero]+u[zero]*(hi[zero]-lo[zero])
    return np.exp(logmass)


def center_likelihood(gammas,centers,times,direction,noise_m=.0006,speed_nodes=64):
    """Integrate the public uniform initial center and speed distributions.

    Speed quadrature follows its Gaussian conditional instead of a fixed speed
    grid, so long queries cannot gain spurious gamma evidence from coarse speeds.
    """
    g=np.atleast_1d(gammas);n=len(centers);sigma=noise_m/np.sqrt(2)
    response=-np.expm1(-g[:,None]*np.asarray(times)[None])/g[:,None]
    mean_g=response.mean(1);centered_g=response-mean_g[:,None]
    observed_mean=centers.mean(0);centered_y=centers-observed_mean
    a=np.sum(centered_g**2,axis=1);cross=np.sum(centered_g*(centered_y@direction)[None],axis=1)
    constant=np.sum(centered_y**2);z,weights=leggauss(speed_nodes);u=(z+1)/2;weights=weights/2
    speeds=np.empty((len(g),speed_nodes));logbase=np.zeros(len(g));active=a>1e-20
    mu=cross[active]/a[active];sd=sigma/np.sqrt(a[active])
    lo=(.01-mu)/sd;hi=(.04-mu)/sd
    speeds[active]=truncnorm.ppf(u[None],lo[:,None],hi[:,None],loc=mu[:,None],scale=sd[:,None])
    logbase[active]=-(constant-cross[active]**2/a[active])/(2*sigma**2)+np.log(np.sqrt(2*np.pi)*sd/.03)+log_normal_interval(lo,hi)
    speeds[~active]=.01+.03*u[None];logbase[~active]=-constant/(2*sigma**2)
    initial_center_mean=observed_mean[None,None,:]-speeds[:,:,None]*mean_g[:,None,None]*direction[None,None,:]
    center_sd=sigma/np.sqrt(n)
    boxes=log_normal_interval((-.08-initial_center_mean)/center_sd,(.08-initial_center_mean)/center_sd).sum(-1)
    weighted=boxes+np.log(weights)[None];normalizer=logsumexp(weighted,axis=1)
    expected_speed=np.sum(np.exp(weighted-normalizer[:,None])*speeds,axis=1)
    return logbase+normalizer,expected_speed


def _grid(positions,times,H,b,gamma_bounds,ratio_bounds,shape,noise_m):
    ng,nr=shape
    gammas=np.linspace(*gamma_bounds,ng,endpoint=False)+np.diff(gamma_bounds)[0]/(2*ng)
    ratios=np.linspace(*ratio_bounds,nr,endpoint=False)+np.diff(ratio_bounds)[0]/(2*nr)
    relative=positions[:,2:]-positions[:,:2];n=relative.sum(0);n=n/np.linalg.norm(n)
    direction=np.array([n[0]*np.cos(.5)-n[1]*np.sin(.5),n[0]*np.sin(.5)+n[1]*np.cos(.5)])
    center=(positions[:,:2]+positions[:,2:])/2
    log_center,speeds=center_likelihood(gammas,center,times,direction,noise_m)
    gg=np.repeat(gammas,nr);rr=np.tile(ratios,ng)
    log_mass,_=mass_factor(H,b,gg,rr)
    logweights=np.repeat(log_center,nr)+log_mass;observed=np.linalg.norm(relative,axis=1)-.35
    for start in range(0,len(gg),4096):
        stop=min(start+4096,len(gg));f,_=radial_solution(gg[start:stop],rr[start:stop],times)
        plus=-np.sum((observed[None]-.08*f)**2,axis=1)/(4*noise_m**2)
        minus=-np.sum((observed[None]+.08*f)**2,axis=1)/(4*noise_m**2)
        logweights[start:stop]+=np.logaddexp(plus,minus)-np.log(2)
    maximum=float(logweights.max());weights=np.exp(logweights-maximum);weights/=weights.sum()
    return dict(gamma=gg,ratio=rr,weights=weights,logweights=logweights,speed=np.repeat(speeds,nr),
        orientation=n,shape=shape,gamma_bounds=gamma_bounds,ratio_bounds=ratio_bounds)


def conditional_parameters(positions,times,H,b,*,shape=(192,3072),samples=256,noise_m=.0006,seed=983741):
    """Global gamma/ratio quadrature with separate continuous mass integration."""
    result=_grid(positions,times,H,b,(.25,1.5),(2.,50.),shape,noise_m);coarse=result
    active=result['logweights']>result['logweights'].max()-35
    dg=1.25/shape[0];dr=48/shape[1]
    gb=(max(.25,float(result['gamma'][active].min())-3*dg),min(1.5,float(result['gamma'][active].max())+3*dg))
    rb=(max(2.,float(result['ratio'][active].min())-3*dr),min(50.,float(result['ratio'][active].max())+3*dr))
    zoom=(gb[1]-gb[0])*(rb[1]-rb[0])<.2*1.25*48
    if zoom:result=_grid(positions,times,H,b,gb,rb,(192,1024),noise_m)
    # Keep the actual integration dimensions separate. Flattening a 2D grid
    # into one inverse CDF destroys useful Sobol stratification across gamma
    # and ratio and creates avoidable prediction integration variance.
    power=int(np.ceil(np.log2(samples)));u=qmc.Sobol(3,scramble=True,seed=seed).random_base2(power)[:samples]
    ng,nr=result['shape'];joint=result['weights'].reshape(ng,nr);marginal=joint.sum(1)
    gamma_index=np.minimum(np.searchsorted(np.cumsum(marginal),u[:,0],side='right'),ng-1)
    ratio_index=np.zeros(samples,int)
    for gi in np.unique(gamma_index):
        selected=gamma_index==gi
        ratio_index[selected]=np.minimum(np.searchsorted(np.cumsum(joint[gi]/marginal[gi]),u[selected,1],side='right'),nr-1)
    indices=gamma_index*nr+ratio_index
    g=result['gamma'][indices];ratio=result['ratio'][indices]
    _,factor=mass_factor(H,b,g,ratio);m=draw_mass(factor,u[:,2]);parameters=np.stack((m,g,m*ratio),axis=1)
    f,_=radial_solution(g,ratio,times);relative=positions[:,2:]-positions[:,:2];observed=np.linalg.norm(relative,axis=1)-.35
    plus=-np.sum((observed[None]-.08*f)**2,axis=1)/(4*noise_m**2)
    minus=-np.sum((observed[None]+.08*f)**2,axis=1)/(4*noise_m**2)
    sign_probability=np.exp(plus-np.logaddexp(plus,minus))
    diagnostic=dict(coarse_shape=shape,zoomed=zoom,final_shape=result['shape'],largest_grid_weight=float(result['weights'].max()),
        grid_effective_nodes=float(1/np.sum(result['weights']**2)),coarse_largest_weight=float(coarse['weights'].max()),
        marginal_gamma_mean=float(np.sum(result['weights']*result['gamma'])),marginal_ratio_mean=float(np.sum(result['weights']*result['ratio'])),
        likelihood='independent Gaussian tracking residual; visible orientation point estimate; initial-state prior marginalized',
        quadrature='global gamma/k-over-m midpoint grid; local refinement; 3D Sobol gamma/conditional-ratio/conditional-mass sampling')
    return parameters,result['speed'][indices],sign_probability,result['orientation'],diagnostic


def predict_passive(query,H,b,config=Config(resolution=128),*,shape=(192,3072),samples=256,theta=None,return_path=False,seed=983741,return_parameter_samples=False):
    if config.control_dt!=.05 or config.ell0!=.35 or config.force_max!=1.:
        raise ValueError('passive-prefix95 prior does not match this physical profile')
    obs,actions=public_query_images(query,config,allowed_targets=('passive_prefix95_joint_state_delta_8d',))
    if len(obs)>96:raise ValueError('passive query prefix exceeds declared anchor')
    images=np.rint(obs[:,0]*255).astype(np.uint8);positions=track(images,config)
    times=(95-len(obs)+1+np.arange(len(obs)))*config.control_dt
    if theta is None:
        parameters,speeds,positive,n,diagnostic=conditional_parameters(positions,times,H,b,shape=shape,samples=samples,seed=seed)
    else:
        parameters=np.asarray(theta,float)[None];g=parameters[:,1];ratio=parameters[:,2]/parameters[:,0]
        relative=positions[:,2:]-positions[:,:2];n=relative.sum(0);n/=np.linalg.norm(n)
        direction=np.array([n[0]*np.cos(.5)-n[1]*np.sin(.5),n[0]*np.sin(.5)+n[1]*np.cos(.5)])
        _,speeds=center_likelihood(g,(positions[:,:2]+positions[:,2:])/2,times,direction)
        f,_=radial_solution(g,ratio,times);observed=np.linalg.norm(relative,axis=1)-config.ell0
        plus=-np.sum((observed[None]-.08*f)**2,axis=1)/(4*.0006**2);minus=-np.sum((observed[None]+.08*f)**2,axis=1)/(4*.0006**2)
        positive=np.exp(plus-np.logaddexp(plus,minus));diagnostic=dict(reference='same visible query plus true theta; initial state remains visually conditioned')
    f,derivative=radial_solution(parameters[:,1],parameters[:,2]/parameters[:,0],[95*config.control_dt])
    direction=np.array([n[0]*np.cos(.5)-n[1]*np.sin(.5),n[0]*np.sin(.5)+n[1]*np.cos(.5)])
    vc=(speeds*np.exp(-parameters[:,1]*95*config.control_dt))[:,None]*direction
    states=[]
    for sign in (1,-1):
        r=(config.ell0+sign*.08*f[:,0])[:,None]*n;rv=sign*.08*derivative[:,0,None]*n
        states.append(np.concatenate((-r/2,r/2,vc-rv/2,vc+rv/2),axis=1))
    initial=np.concatenate(states,axis=0);p=np.concatenate((parameters,parameters),axis=0)
    weights=np.r_[positive,1-positive]/len(parameters);active=weights>1e-12/len(parameters)
    initial=initial[active];path=reference_rollout_batch(p[active],initial,actions,config)-initial[:,None]
    prediction=np.einsum('n,ntk->tk',weights[active],path)
    diagnostic['future_solver']='batched RK4 at .002s; separately checked against DOP853'
    if return_parameter_samples:diagnostic['parameter_samples']=parameters.tolist()
    return (prediction if return_path else prediction[-1]),diagnostic
