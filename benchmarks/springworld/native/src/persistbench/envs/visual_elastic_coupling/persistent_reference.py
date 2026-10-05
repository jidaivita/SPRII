"""Bounded continuous parameter posterior stored as compressed local factors.

The prior is uniform in physical m/gamma/k, not in their logarithms. Likelihood
factors are local Gaussian approximations from visual trajectory fitting. Raw
history is discarded after ingest; initialization and reset have distinct roles.
"""
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc,truncnorm
from .schema import Config
from .calibration import track
from .identification import fit_positions,reference_rollout
from .joint_identification import information_factor
from .cold_opportunity import visual_rest_state
from .observations import image_history,public_query_images


def posterior_samples(H,b,bounds=((.5,.25,4.),(2.,1.5,25.)),*,samples=128,burn=96,seed=71893):
    """Directional Gibbs sampler for a box-truncated log-concave posterior.

    Eigenvector directions traverse the free-motion m/k scale null direction
    jointly. The +1 per log coordinate is the Jacobian of the physical prior.
    This avoids a discrete particle grid losing a narrow continuous posterior.
    """
    H=np.asarray(H,float);b=np.asarray(b,float);low,high=np.asarray(bounds,float)
    if H.shape!=(3,3) or b.shape!=(3,) or not np.isfinite(H).all() or not np.isfinite(b).all():
        raise ValueError('invalid parameter information')
    if np.max(np.abs(H-H.T))>1e-6*max(1.,np.max(np.abs(H))):raise ValueError('nonsymmetric information')
    H=(H+H.T)/2
    eigen,vectors=np.linalg.eigh(H)
    if eigen.min() < -1e-7*max(1.,eigen.max()):raise ValueError('negative information curvature')
    eigen=np.maximum(eigen,0.);H=(vectors*eigen)@vectors.T
    if np.max(eigen)<1e-12:
        power=int(np.ceil(np.log2(samples)))
        return qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(power)[:samples],low,high)
    lo=np.log(low);hi=np.log(high);linear=b+1.
    mode=minimize(lambda x:.5*x@H@x-linear@x,(lo+hi)/2,
        jac=lambda x:H@x-linear,bounds=list(zip(lo,hi)),method='L-BFGS-B').x
    x=np.clip(mode,lo,hi);rng=np.random.default_rng(seed);draws=[]
    for sweep in range(burn+samples):
        for vi in rng.permutation(3):
            v=vectors[:,vi];active=np.abs(v)>1e-12
            first=(lo[active]-x[active])/v[active];last=(hi[active]-x[active])/v[active]
            lower=float(np.minimum(first,last).max());upper=float(np.maximum(first,last).min())
            if upper-lower<1e-13:continue
            curvature=float(v@H@v);rate=float(v@(linear-H@x));u=float(rng.uniform(1e-12,1-1e-12))
            if curvature>1e-8:
                mean=rate/curvature;sd=1/np.sqrt(curvature)
                step=float(truncnorm.ppf(u,(lower-mean)/sd,(upper-mean)/sd,loc=mean,scale=sd))
            elif abs(rate)*(upper-lower)<1e-8:
                step=lower+u*(upper-lower)
            elif rate>0:
                step=upper+np.log(u+(1-u)*np.exp(rate*(lower-upper)))/rate
            else:
                step=lower+np.log((1-u)+u*np.exp(rate*(upper-lower)))/rate
            if not np.isfinite(step):raise ValueError('nonfinite posterior sampler step')
            x=np.clip(x+step*v,lo,hi)
        if sweep>=burn:draws.append(np.exp(x))
    return np.asarray(draws)




class CompressedVisualReference:
    """Ingest pixels once, retain only H/b and bounded duplicate fingerprints."""
    def __init__(self,config=Config(resolution=128),*,max_histories=8,samples=128,noise_floor_m=.0006):
        self.config=config;self.max_histories=max_histories;self.samples=samples;self.noise_floor_m=noise_floor_m
        self._clear()

    def _clear(self):
        self.H=np.zeros((3,3));self.b=np.zeros(3);self.fingerprints=set()
        self.processed_histories=0;self.unique_frames=0;self.fit_failures=0;self.episode_token=None

    def initialize(self,context):
        from persistbench.contracts import CapabilityDeclaration,OutputType
        self._clear();self.seed=int(context.seed)
        return CapabilityDeclaration((OutputType.PREDICTION,OutputType.REPRESENTATION),representation_dim=12,
            output_keys=('joint_state_delta','representation'))

    def reset(self,context):self.episode_token=context.episode_token

    def ingest(self,experience):
        images,actions=image_history(experience,self.config)
        fingerprint=hashlib.sha256(images.tobytes()+actions.tobytes()).digest()
        self.processed_histories+=1
        if fingerprint in self.fingerprints:return
        if len(self.fingerprints)>=self.max_histories:raise ValueError('declared persistent episode capacity exceeded')
        self.fingerprints.add(fingerprint);self.unique_frames+=len(images)
        if len(images)<3 or (not np.any(actions) and np.array_equal(images,np.broadcast_to(images[:1],images.shape))):return
        try:
            fit=fit_positions(track(images,self.config),actions,self.config,starts=3)
            H,b=information_factor(fit,self.config,self.noise_floor_m)
            if not fit['optimizer_success']:self.fit_failures+=1
            self.H+=H;self.b+=b
        except (ValueError,FloatingPointError):self.fit_failures+=1

    def memory_state(self):
        return dict(H=self.H.copy(),b=self.b.copy(),fingerprints=tuple(sorted(self.fingerprints)),
            processed_histories=self.processed_histories,unique_frames=self.unique_frames,fit_failures=self.fit_failures)

    def mutable_state_bytes(self):
        return self.H.nbytes+self.b.nbytes+32*len(self.fingerprints)+3*8

    def parameter_samples(self):return posterior_samples(self.H,self.b,samples=self.samples,seed=self.seed)

    def respond(self,query):
        from persistbench.contracts import AgentOutput,OutputType
        query.validate();payload=query.observations
        allowed={'observations','relative_times','horizon_seconds','target_spec','past_actions','past_action_mask'}
        if not isinstance(payload,dict) or set(payload)-allowed:raise ValueError('non-public query payload')
        if payload.get('target_spec')=='parameter_information_representation':
            return AgentOutput(OutputType.REPRESENTATION,{'representation':np.r_[self.H.ravel(),self.b]})
        if payload.get('target_spec')=='passive_prefix95_joint_state_delta_8d':
            from .passive_query_reference import predict_passive
            prediction,diagnostic=predict_passive(query,self.H,self.b,self.config,samples=self.samples)
            diagnostic.update(persistent_numeric_payload_bytes=self.mutable_state_bytes(),fit_failures=self.fit_failures)
            return AgentOutput(OutputType.PREDICTION,{'joint_state_delta':prediction},diagnostics=diagnostic)
        # Only the independently calibrated cold profile is handled here. Moving
        # queries need their own state/posterior adapter, not a silent zero velocity.
        if payload.get('target_spec')!='cold_rest_joint_state_delta_8d':
            raise ValueError('this reference requires the declared cold-rest query profile')
        if 'past_actions' in payload:raise ValueError('cold profile exposes no past query actions')
        obs,actions=public_query_images(query,self.config,allowed_targets=('cold_rest_joint_state_delta_8d',))
        images=np.rint(obs[:,0]*255).astype(np.uint8)
        if len(images) not in (1,2):raise ValueError('cold query requires one or two frames')
        if len(images)==2 and not np.array_equal(images[0],images[1]):raise ValueError('query does not satisfy rest observation profile')
        initial=visual_rest_state(track(images,self.config)[-1],self.config)
        samples=self.parameter_samples()
        from .batch_prediction import reference_rollout_batch
        paths=reference_rollout_batch(samples,np.broadcast_to(initial,(len(samples),8)),actions,self.config)
        prediction=np.mean(paths[:,-1]-initial,axis=0)
        return AgentOutput(OutputType.PREDICTION,{'joint_state_delta':prediction},
            diagnostics=dict(persistent_numeric_payload_bytes=self.mutable_state_bytes(),unique_histories=len(self.fingerprints),
                processed_histories=self.processed_histories,fit_failures=self.fit_failures,
                likelihood='local visual trajectory-fit Gaussian; physical-uniform bounded prior',
                forecast_solver='vectorized float64 RK4 at0.002s, independently checked against DOP853; official observations remain MuJoCo'))
