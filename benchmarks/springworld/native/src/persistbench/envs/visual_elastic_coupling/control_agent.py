"""Causal visual feedback controller with historical and online physical belief.

The shared certainty-equivalent PD controller is a competent reference class,
not a Bayes-optimal policy. No future actions or simulator states enter this API.
"""
import numpy as np
from persistbench.contracts import AgentOutput,CapabilityDeclaration,OutputType
from .schema import Config
from .calibration import track
from .cold_opportunity import visual_rest_state
from .control_calibration import center_feedback
from .control_identification import fit_control_prefix
from .fast_reference import reference_rollout_fast

FIT_STEPS=(10,20,40,80,160,240)


class ExplicitParameterPrior:
    def __init__(self,config):
        from .persistent_reference import CompressedVisualReference
        self.reference=CompressedVisualReference(config,samples=256)
    def initialize(self,context):self.reference.initialize(context)
    def reset(self,context):self.reference.reset(context)
    def ingest(self,experience):self.reference.ingest(experience)
    def prior(self):
        r=self.reference
        point=r.parameter_samples().mean(0) if np.any(r.H) else np.array([1.25,.875,14.5])
        return r.H.copy(),r.b.copy(),point
    def mutable_state_bytes(self):return self.reference.mutable_state_bytes()


class OnlineVisualControlAgent:
    def __init__(self,config=Config(resolution=128),*,prior_adapter=None,known_parameters=None,adaptive=True,fit_steps=FIT_STEPS,max_fit_evaluations=100):
        self.config=config;self.prior=prior_adapter or ExplicitParameterPrior(config)
        self.known_parameters=None if known_parameters is None else np.asarray(known_parameters,float).copy()
        if self.known_parameters is not None and (self.known_parameters.shape!=(3,) or not np.isfinite(self.known_parameters).all() or np.any(self.known_parameters<=0)):
            raise ValueError('invalid declared privileged parameter reference')
        self.adaptive=bool(adaptive and known_parameters is None);self.fit_steps=tuple(fit_steps);self.max_fit_evaluations=max_fit_evaluations
        if any(s<5 or s>=320 for s in self.fit_steps) or tuple(sorted(set(self.fit_steps)))!=self.fit_steps:raise ValueError('invalid causal fit schedule')
        self._clear_online()

    def _clear_online(self):
        self.positions=[];self.actions=[];self.estimate=None;self.parameters=None;self.last_action=None;self.goal=None
        self.step=-1;self.fit_calls=0;self.fit_failures=0

    def initialize(self,context):
        self.prior.initialize(context);self._clear_online()
        return CapabilityDeclaration((OutputType.ACTION,),output_keys=('action',))

    def reset(self,context):
        self.prior.reset(context);self._clear_online()

    def ingest(self,experience):
        if self.step>=0:raise ValueError('historical ingest is closed after control begins')
        self.prior.ingest(experience)

    def mutable_state_bytes(self):
        persistent=self.prior.mutable_state_bytes()
        arrays=self.positions+self.actions+[x for x in (self.estimate,self.parameters,self.last_action,self.goal) if x is not None]
        return persistent+sum(np.asarray(a).nbytes for a in arrays)+3*8

    def respond(self,query):
        query.validate();payload=query.observations
        allowed={'images','elapsed_seconds','goal_xy','target_spec','initial_profile'}
        if not isinstance(payload,dict) or set(payload)!=allowed:raise ValueError('control packet contains missing or non-public fields')
        if payload['target_spec']!='move_and_stabilize_v1' or payload['initial_profile']!='natural_length_rest':raise ValueError('unregistered control profile')
        if query.horizons is not None:raise ValueError('control receives no future action horizon')
        images=np.asarray(payload['images']);expected_frames=2 if self.step==-1 else 1
        if images.dtype!=np.uint8 or images.shape!=(expected_frames,self.config.resolution,self.config.resolution):raise ValueError('invalid causal control image support')
        if self.step==-1 and not np.array_equal(images[0],images[1]):raise ValueError('initial control profile is not visually at rest')
        elapsed=float(payload['elapsed_seconds']);next_step=self.step+1
        if not np.isfinite(elapsed) or abs(elapsed-next_step*self.config.control_dt)>1e-10:raise ValueError('control time skipped, repeated or outside observed support')
        goal=np.asarray(payload['goal_xy'],float)
        if goal.shape!=(2,) or not np.isfinite(goal).all() or not np.array_equal(goal,[.2,0.]):raise ValueError('unregistered movement goal')
        executed=np.asarray(query.actions,float)
        if executed.shape!=((0,2) if next_step==0 else (1,2)) or not np.isfinite(executed).all():raise ValueError('expected only the previous executed action')
        if next_step and not np.array_equal(executed[0],self.last_action):raise ValueError('executed action differs from the declared deterministic actuator')
        measured=track(images[-1:],self.config)[0];fit=None
        if next_step==0:
            self.parameters=self.known_parameters.copy() if self.known_parameters is not None else self.prior.prior()[2]
            self.estimate=visual_rest_state(measured,self.config);self.goal=goal.copy()
        else:
            prediction=reference_rollout_fast(self.parameters,self.estimate,executed,self.config)[-1]
            innovation=measured-prediction[:4];prediction[:4]+=.35*innovation;prediction[4:]+=.08/self.config.control_dt*innovation
            self.estimate=prediction;self.actions.append(executed[0].copy())
        self.positions.append(measured.copy());self.step=next_step
        if self.adaptive and next_step in self.fit_steps:
            self.fit_calls+=1;H,b,_=self.prior.prior()
            try:
                fit=fit_control_prefix(np.asarray(self.positions),np.asarray(self.actions),H,b,self.config,
                    initial_parameters=self.parameters,max_nfev=self.max_fit_evaluations)
                if fit['optimizer_success']:
                    self.parameters=np.asarray(fit['parameters']);self.estimate=np.asarray(fit['current_state'])
                else:self.fit_failures+=1
            except (ValueError,FloatingPointError) as exc:self.fit_failures+=1;fit=dict(error=str(exc),observed_frames=len(self.positions))
        self.last_action=center_feedback(self.estimate,self.parameters,self.goal,self.config)
        return AgentOutput(OutputType.ACTION,{'action':self.last_action.copy()},diagnostics=dict(
            parameters=self.parameters.tolist(),estimate=self.estimate.tolist(),step=self.step,fit_calls=self.fit_calls,fit_failures=self.fit_failures,
            current_fit=fit,persistent_numeric_bytes=self.prior.mutable_state_bytes(),total_mutable_numeric_bytes=self.mutable_state_bytes(),
            raw_donor_retained=False,raw_query_images_retained=False,online_observed_positions=len(self.positions)))
