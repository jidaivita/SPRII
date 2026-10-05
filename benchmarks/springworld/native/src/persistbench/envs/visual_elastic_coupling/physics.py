"""MuJoCo 3.3.5 is the only episode dynamics backend. SI units throughout."""
from dataclasses import replace
from pathlib import Path
import json
import threading
import numpy as np
import mujoco
from .schema import Config, Parameters

_MAX_STAGE_ERROR = 0.0
_LOCK = threading.RLock()  # MuJoCo's Python control callback is process-global.

class InvalidTrajectory(ValueError):
    def __init__(self, reason, time, last_valid_state):
        self.reason, self.time = str(reason), float(time)
        self.last_valid_state = np.asarray(last_valid_state).copy()
        super().__init__(f'{reason} at t={time:.9g}')


def validate_state(state, config=Config()):
    s = np.asarray(state, dtype=float)
    if s.shape != (8,) or not np.isfinite(s).all():
        raise ValueError('nonfinite_or_invalid_state')
    if np.linalg.norm(s[2:4]-s[:2]) <= max(config.ell_min, 2*config.radius):
        raise ValueError('minimum_separation')
    if np.any(np.abs(s[:4]) > config.field_width/2-config.margin-config.radius):
        raise ValueError('outside_complete_visibility')
    return s


def derivative(state, force, parameters, config=Config()):
    """Independent equation reference; never used to generate episodes."""
    s = validate_state(state, config); parameters.validate()
    force = np.asarray(force, dtype=float)
    if force.shape != (2,) or not np.isfinite(force).all():
        raise ValueError('invalid force')
    r = s[2:4]-s[:2]; length = np.linalg.norm(r)
    fs = parameters.k*(length-config.ell0)*r/length
    return np.r_[s[4:], (force+fs)/parameters.m-parameters.gamma*s[4:6],
                 -fs/parameters.m-parameters.gamma*s[6:8]]


def build_model(parameters, config=Config()):
    parameters.validate()
    if mujoco.__version__ != '3.3.5':
        raise RuntimeError('formal physics requires mujoco==3.3.5')
    if config.dt <= 0 or config.control_dt <= 0 or not np.isclose(config.control_dt/config.dt, round(config.control_dt/config.dt)):
        raise ValueError('control_dt must be a positive integer multiple of dt')
    m, damping, k = parameters.m, parameters.m*parameters.gamma, parameters.k
    bodies = ''
    for i, shade in ((1,.35),(2,.8)):
        bodies += f'''<body name="object{i}"><inertial pos="0 0 0" mass="{m}" diaginertia="0.001 0.001 0.001"/>
        <joint name="x{i}" type="slide" axis="1 0 0" damping="{damping}"/><joint name="y{i}" type="slide" axis="0 1 0" damping="{damping}"/>
        <geom type="sphere" size="{config.radius}" rgba="{shade} {shade} {shade} 1"/><site name="center{i}" size="0.001"/></body>'''
    xml = f'''<mujoco><option timestep="{config.dt}" integrator="RK4" gravity="0 0 0"/>
    <default><joint stiffness="0" frictionloss="0" armature="0"/><geom contype="0" conaffinity="0"/></default>
    <worldbody>{bodies}</worldbody><tendon><spatial name="spring" stiffness="{k}" damping="0" frictionloss="0" springlength="{config.ell0}"><site site="center1"/><site site="center2"/></spatial></tendon>
    <actuator><motor joint="x1" gear="{config.force_max}"/><motor joint="y1" gear="{config.force_max}"/></actuator></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def trajectory(parameters, initial_state, actions, config=Config(), *, return_substeps=False, diagnostics=None):
    global _MAX_STAGE_ERROR
    actions = np.asarray(actions,dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-7):
        raise ValueError('actions must be finite normalized [T,2], without clipping')
    try: initial_state = validate_state(initial_state, config)
    except ValueError as exc: raise InvalidTrajectory(str(exc),0,initial_state) from exc
    with _LOCK:
        model = build_model(parameters, config); data = mujoco.MjData(model)
        data.qpos[:] = initial_state[:4]; data.qvel[:] = initial_state[4:]
        states = [initial_state.copy()]; substeps = [initial_state.copy()]
        visibility_envelopes = []
        previous = mujoco.get_mjcb_control()
        last = initial_state.copy()
        expected_stages = []
        stage_index = 0
        def stage_check(m, d):
            nonlocal stage_index
            global _MAX_STAGE_ERROR
            actual = np.r_[d.qpos,d.qvel]
            if stage_index >= len(expected_stages):
                raise InvalidTrajectory("unexpected_RK4_stage", d.time, last)
            error = float(np.max(np.abs(actual-expected_stages[stage_index])))
            _MAX_STAGE_ERROR = max(_MAX_STAGE_ERROR,error)
            stage_index += 1
            if error > 1e-10:
                raise InvalidTrajectory("reference_native_stage_mismatch", d.time, last)
            try: validate_state(np.r_[d.qpos,d.qvel],config)
            except ValueError as exc: raise InvalidTrajectory('RK4_stage_'+str(exc),d.time,last) from exc
        try:
            mujoco.set_mjcb_control(stage_check)
            for action in actions:
                data.ctrl[:] = action
                envelope = np.abs(last[:4]).copy()
                for _ in range(round(config.control_dt/config.dt)):
                    # Safety preflight only; all accepted states still come from MuJoCo.
                    # Every derivative validates before forming a spring direction.
                    expected_stages = [last.copy()]
                    preflight_time = data.time
                    try:
                        k1 = derivative(last, action*config.force_max, parameters, config)
                        expected_stages.append(last+config.dt*.5*k1)
                        preflight_time = data.time+config.dt*.5
                        k2 = derivative(expected_stages[-1],action*config.force_max,parameters,config)
                        expected_stages.append(last+config.dt*.5*k2)
                        k3 = derivative(expected_stages[-1],action*config.force_max,parameters,config)
                        expected_stages.append(last+config.dt*k3)
                        preflight_time = data.time+config.dt
                        k4 = derivative(expected_stages[-1],action*config.force_max,parameters,config)
                        validate_state(last+config.dt*(k1+2*k2+2*k3+k4)/6,config)
                    except ValueError as exc:
                        raise InvalidTrajectory('RK4_preflight_'+str(exc),preflight_time,last) from exc
                    stage_index = 0
                    mujoco.mj_step(model,data)
                    if stage_index != 4:
                        raise InvalidTrajectory('incomplete_RK4_stage_checks',data.time,last)
                    current = np.r_[data.qpos,data.qvel].copy()
                    try: validate_state(current,config)
                    except ValueError as exc: raise InvalidTrajectory(str(exc),data.time,last) from exc
                    if np.any(data.warning.number):
                        raise InvalidTrajectory('mujoco_warning',data.time,last)
                    last = current
                    envelope = np.maximum(envelope, np.max(np.abs(np.asarray(expected_stages)[:, :4]), axis=0))
                    envelope = np.maximum(envelope, np.abs(current[:4]))
                    if return_substeps: substeps.append(current)
                states.append(current)
                visibility_envelopes.append(envelope)
        finally:
            mujoco.set_mjcb_control(previous)
            if diagnostics is not None:
                # Evaluator-only diagnostics retain the accepted prefix on a
                # rejected full episode, so short-window coverage is measurable.
                diagnostics["accepted_observed_states"] = np.asarray(states)
                diagnostics["transition_position_envelope"] = np.asarray(visibility_envelopes).reshape(-1, 4)
                diagnostics["requested_transitions"] = len(actions)
                diagnostics["completed_transitions"] = len(states)-1
    result = np.asarray(states)
    return (result,np.asarray(substeps)) if return_substeps else result


def run_checks(report_path):
    from scipy.integrate import solve_ivp
    p=Parameters(1.,.7,9.); c=Config(); s=np.array([-.22,0,.22,0,.08,.03,-.03,-.01])
    actions=np.tile([.15,.08],(30,1)); checks={}
    def check(name, fn):
        try:
            values,tolerance=fn(); error=float(np.max(np.abs(values)))
            checks[name]={'status':'PASS' if error<=tolerance else 'FAIL','max_error':error,'tolerance':tolerance}
        except Exception as exc: checks[name]={'status':'FAIL','error':str(exc)}
    states,sub=trajectory(p,s,actions,c,return_substeps=True)
    check('deterministic_replay',lambda:(states-trajectory(p,s,actions,c),0.))
    check('restart_replay',lambda:(states[15:]-trajectory(p,states[15],actions[15:],c),1e-12))
    check('COM_k_invariance',lambda:((states[:,:2]+states[:,2:4])/2-(trajectory(replace(p,k=15),s,actions,c)[:,:2]+trajectory(replace(p,k=15),s,actions,c)[:,2:4])/2,1e-10))
    zeros=np.zeros_like(actions)
    check('free_mass_stiffness_scaling',lambda:(trajectory(p,s,zeros,c)-trajectory(replace(p,m=2,k=18),s,zeros,c),1e-10))
    time=np.arange(len(states))*c.control_dt; c0=(s[:2]+s[2:4])/2; v0=(s[4:6]+s[6:])/2; veq=actions[0]*c.force_max/(2*p.m*p.gamma)
    expected=c0+veq*time[:,None]+(v0-veq)*(1-np.exp(-p.gamma*time[:,None]))/p.gamma
    check('COM_analytic_forced',lambda:((states[:,:2]+states[:,2:4])/2-expected,1e-10))
    model=build_model(p,c); data=mujoco.MjData(model); data.qpos[:]=s[:4]; data.qvel[:]=s[4:]; data.ctrl[:]=actions[0]; mujoco.mj_forward(model,data)
    check('instant_acceleration_reference',lambda:(data.qacc-derivative(s,c.force_max*actions[0],p,c)[4:],1e-12))
    check('actuator_force_mapping',lambda:(data.qfrc_actuator-np.r_[actions[0]*c.force_max,0,0],1e-12))
    check('compiled_damping',lambda:(model.dof_damping-p.m*p.gamma,0.))
    check('compiled_mass',lambda:(model.body_mass[1:]-p.m,0.))
    check('spring_reaction_pair',lambda:((data.qfrc_passive+model.dof_damping*s[4:])[:2]+(data.qfrc_passive+model.dof_damping*s[4:])[2:],1e-12))
    _,free=trajectory(p,s,zeros,c,return_substeps=True)
    energy=p.m/2*np.sum(free[:,4:]**2,axis=1)+p.k/2*(np.linalg.norm(free[:,2:4]-free[:,:2],axis=1)-c.ell0)**2
    check('unforced_energy_nonincrease',lambda:(max(0,float(np.diff(energy).max())),1e-12))
    glide=np.array([-.175,0,.175,0,.08,.04,.08,.04]); g=trajectory(p,glide,zeros,c)
    check('natural_length_glide',lambda:(g[:,2:4]-g[:,:2]-[c.ell0,0],1e-12))
    still=glide.copy(); still[4:]=0
    check('stationary_no_information',lambda:(trajectory(p,still,zeros,c)-still,1e-12))
    ref=solve_ivp(lambda t,y:derivative(y,c.force_max*actions[0],p,c),(0,time[-1]),s,t_eval=time,rtol=1e-11,atol=1e-13,method='DOP853').y.T
    check('independent_reference',lambda:(states-ref,1e-8))
    fine=trajectory(p,s,actions,replace(c,dt=.001)); finer=trajectory(p,s,actions,replace(c,dt=.0005))
    check('step_convergence',lambda:(fine-finer,1e-9))
    d=derivative(s,actions[0]*c.force_max,p,c); r=s[2:4]-s[:2]; fs=p.k*(np.linalg.norm(r)-c.ell0)*r/np.linalg.norm(r)
    measured_power=p.m*np.dot(s[4:],d[4:])+np.dot(fs,s[6:8]-s[4:6])
    expected_power=np.dot(actions[0]*c.force_max,s[4:6])-p.m*p.gamma*np.dot(s[4:],s[4:])
    check('instant_energy_power_identity',lambda:(measured_power-expected_power,1e-12))
    fast=s.copy(); fast[4]=1000
    try: trajectory(p,fast,zeros[:1],c); checks['internal_stage_invalid']={'status':'FAIL'}
    except InvalidTrajectory as exc: checks['internal_stage_invalid']={'status':'PASS','reason':exc.reason,'time':exc.time,'last_valid_matches_initial':bool(np.array_equal(exc.last_valid_state,fast))}
    for name,bad in [('overlap',np.zeros(8)),('outside',np.r_[1.,0,.5,0,np.zeros(4)])]:
        try: trajectory(p,bad,zeros,c); checks[name]={'status':'FAIL'}
        except InvalidTrajectory as exc: checks[name]={'status':'PASS','reason':exc.reason,'time':exc.time}
    result={'status':'executed','scientific_claims':'NOT_ESTABLISHED','backend':'mujoco','version':mujoco.__version__,'integrator':'RK4','dt':c.dt,'checks':checks,'step_errors':{'dt_vs_reference':float(np.max(abs(states-ref))),'half_dt_vs_reference':float(np.max(abs(fine-ref))),'quarter_dt_vs_reference':float(np.max(abs(finer-ref)))},'sleep':'MuJoCo 3.3.5 has no body sleep API; stationary and low-amplitude trajectories checked','stage_validation':'independent RK4 safety preflight validates all stages before spring direction and before mj_step; actual native stages checked against preflight at tolerance 1e-10; accepted substeps independently checked','max_native_preflight_stage_error':_MAX_STAGE_ERROR ,'initial_failed_attempts':[{'error':'ModuleNotFoundError: scipy','resolution':'installed scipy==1.13.1; reran checks'}]}
    path=Path(report_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+'\n'); return result

if __name__=='__main__':
    import sys
    print(json.dumps(run_checks(sys.argv[1]),indent=2))
