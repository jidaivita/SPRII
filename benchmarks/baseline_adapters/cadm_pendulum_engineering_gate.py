#!/usr/bin/env python3
"""Zero-learning CaDM Pendulum dynamics/reset/reward/CEM engineering gate.

Executes unchanged official method ASTs behind minimal dependency stubs and
compares them to an independent NumPy port. Optional analytic-oracle CEM rollouts
are real simulator returns, but are engineering references, never learned results.
No Gym, TensorFlow, GPU, training, or remote access is needed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def observation(state):
    state = np.asarray(state)
    return np.stack((np.cos(state[..., 0]), np.sin(state[..., 0]), state[..., 1]), axis=-1)


def dynamics(state, action, mass, length):
    """State is (theta, velocity); action is normalized scalar in [-1,1]."""
    state = np.asarray(state, dtype=np.float64)
    a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
    th, vel = state[..., 0], state[..., 1]
    u = 2. * a
    cost = angle(th) ** 2 + .1 * vel ** 2 + .001 * a ** 2
    raw_vel = vel + (-15. / length * np.sin(th + np.pi) + 3. / (mass * length ** 2) * u) * .05
    # Official code updates theta before clipping the new velocity.
    new_th = th + raw_vel * .05
    new_vel = np.clip(raw_vel, -8., 8.)
    return np.stack((new_th, new_vel), axis=-1), -cost


def known_reward(obs, actions):
    return -(angle(np.arctan2(obs[..., 1], obs[..., 0])) ** 2 + .1 * obs[..., 2] ** 2
             + .001 * np.clip(actions[..., 0], -1., 1.) ** 2)


class Pendulum:
    def __init__(self, seed=0, masses=None, lengths=None):
        self.rng = np.random.RandomState(seed)
        self.masses = np.array(masses if masses is not None else [.75 + .05*i for i in range(11)])
        self.lengths = np.array(lengths if lengths is not None else [.75 + .05*i for i in range(11)])
        self.state = np.zeros(2); self.mass = self.length = 1.
        self.nsteps = self.nsteps_vertical = 0; self.success = False

    def reset(self):
        self.mass = float(self.masses[self.rng.randint(len(self.masses))])
        self.length = float(self.lengths[self.rng.randint(len(self.lengths))])
        x = self.rng.uniform(low=np.array([7*np.pi/8, -.2]), high=np.array([9*np.pi/8, .2]))
        self.state = np.array([angle(x[0]), x[1]])
        self.nsteps = self.nsteps_vertical = 0; self.success = False
        return observation(self.state)

    def step(self, normalized_action):
        self.state, reward = dynamics(self.state, float(np.asarray(normalized_action).reshape(-1)[0]), self.mass, self.length)
        self.nsteps += 1
        self.nsteps_vertical = self.nsteps_vertical + 1 if abs(float(angle(self.state[0]))) <= np.pi/3 else 0
        self.success = self.nsteps_vertical >= 100
        return observation(self.state), float(reward), False, {}


class BoxStub:
    def __init__(self):
        self.low = np.array([-2.]); self.high = np.array([2.]); self.shape = (1,)


def official_reference(official, gym_file):
    """Keep official reset/step/reward bodies literal; stub unused frameworks."""
    gym_tree = ast.parse(gym_file.read_text())
    gym_cls = next(n for n in gym_tree.body if isinstance(n, ast.ClassDef) and n.name == 'PendulumEnv')
    init = next(n for n in gym_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    constants = {}
    for statement in init.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Attribute):
            name = statement.targets[0].attr
            if name in ('dt', 'max_speed', 'max_torque'):
                constants[name] = ast.literal_eval(statement.value)
    assert constants == {'max_speed': 8, 'max_torque': 2., 'dt': .05}
    obs_method = next(n for n in gym_cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_obs')
    obs_namespace = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[obs_method], type_ignores=[])), str(gym_file), 'exec'), obs_namespace)

    class GymBase:
        def __init__(self):
            self.__dict__.update(constants)
            self.observation_space = SimpleNamespace(shape=(3,))
            self.action_space = BoxStub()
            self.np_random = np.random.RandomState(0)
        _get_obs = obs_namespace['_get_obs']

    path = official / 'cadm/envs/classic_control.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('ModifiablePendulumEnv', 'RandomPendulumAll')]
    ns = {'np': np, 'PendulumEnv': GymBase}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), ns)
    norm_path = official / 'cadm/envs/normalized_env.py'
    norm_tree = ast.parse(norm_path.read_text())
    norm_cls = next(n for n in norm_tree.body if isinstance(n, ast.ClassDef) and n.name == 'NormalizedEnv')
    step = next(n for n in norm_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    ns_step = {'np': np, 'Box': BoxStub, 'CustomBox': BoxStub}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[])), str(norm_path), 'exec'), ns_step)
    class NormalizeStub:
        def __init__(self, env):
            self._wrapped_env = env; self._normalization_scale = 1.; self._scale_reward = 1.
        step = ns_step['step']
    return ns['RandomPendulumAll'], NormalizeStub


def validate(args):
    official, gym_file = Path(args.official_root), Path(args.gym_reference)
    Cls, Wrapper = official_reference(official, gym_file)
    grid = [.75, .8, .85, .9, .95, 1., 1.05, 1.1, 1.15, 1.2, 1.25]
    ref, port = Cls(mass_set=grid, length_set=grid), Pendulum(masses=grid, lengths=grid)
    reset_max = 0.
    for seed in range(10):
        ref.np_random = np.random.RandomState(seed); port.rng = np.random.RandomState(seed)
        for _ in range(20):
            a, b = ref.reset(), port.reset()
            reset_max = max(reset_max, float(np.max(np.abs(a-b))))
            np.testing.assert_array_equal(a, b)
            assert ref.mass == port.mass and ref.length == port.length
    rng = np.random.RandomState(20260924)
    step_max = reward_max = 0.; clipped = count = 0
    for mass, length in [(1., 1.), (.75, 1.25), (1.25, .75), (.2, 1.8), (.4, 1.6), (.7, 1.3), (1.3, .7), (1.6, .4), (1.8, .2)]:
        for initial in ([np.pi, 0.], [0., 7.99], [-.5, -7.99], [7*np.pi, 4.]):
            ref.mass = port.mass = mass; ref.length = port.length = length
            ref.state = port.state = np.array(initial, dtype=float)
            ref.nsteps = port.nsteps = ref.nsteps_vertical = port.nsteps_vertical = 0
            normalized = Wrapper(ref)
            actions = np.concatenate(([0., -1., 1., -2., 2.], rng.uniform(-1., 1., 95)))
            for a in actions:
                before = observation(port.state)
                ro, rr, rd, _ = normalized.step(np.array([a]))
                po, pr, pd, _ = port.step(a)
                np.testing.assert_allclose(ro, po, rtol=0, atol=2e-12)
                np.testing.assert_allclose(ref.state, port.state, rtol=0, atol=2e-12)
                np.testing.assert_allclose(rr, pr, rtol=0, atol=2e-12)
                assert rd == pd == False and ref.success == port.success
                step_max = max(step_max, float(np.max(np.abs(ro-po)))); reward_max = max(reward_max, abs(rr-pr))
                if abs(a) <= 1:
                    # Official planner's reward consumes normalized, bounded actions.
                    np.testing.assert_allclose(ref.reward(before, np.array([a]), ro), pr, atol=2e-12, rtol=0)
                    np.testing.assert_allclose(known_reward(before, np.array([a])), pr, atol=2e-12, rtol=0)
                clipped += abs(po[2]) == 8.; count += 1
    assert clipped > 0
    result = dict(status='PASS', reset_cases=200, trajectory_steps=count, speed_clip_cases=int(clipped),
                  max_observation_abs_error=step_max, max_reward_abs_error=reward_max, reset_max_abs_error=reset_max,
                  reward_pre_action_state=True, angle_before_velocity_clip=True, normalized_action_to_torque='u=2*clip(a,-1,1)',
                  source_execution='Unchanged official method ASTs; Gym constants/get_obs from pinned Gym0.16 source.',
                  rng_scope='Equal injected RandomState; does not claim Gym seed-to-RNG mapping reproduction.',
                  sources={str(p):sha(p) for p in [official/'cadm/envs/classic_control.py', official/'cadm/envs/normalized_env.py', official/'cadm/envs/config.py', gym_file]},
                  learning=False, trained_policy=False, code_sha256=sha(__file__))
    write(Path(args.output)/'ENV_EQUIVALENCE.json', result)
    return result


def truncated_normal(rng, shape):
    z = rng.normal(size=shape)
    bad = np.abs(z) > 2.
    while bad.any():
        z[bad] = rng.normal(size=int(bad.sum())); bad = np.abs(z) > 2.
    return z


class CEM:
    """Isolated deterministic CEM; replace transition callback for learned models."""
    def __init__(self, seed=0, horizon=30, candidates=200, elites=50, iterations=5, alpha=.1):
        self.rng = np.random.RandomState(seed)
        self.horizon, self.candidates, self.elites, self.iterations, self.alpha = horizon, candidates, elites, iterations, alpha
        self.previous = np.zeros((horizon, 1))

    def plan(self, obs, transition):
        mean = self.previous.copy(); var = np.full_like(mean, .25)
        for _ in range(self.iterations):
            constrained = np.minimum(np.minimum(((mean+1)/2)**2, ((1-mean)/2)**2), var)
            samples = mean[None] + np.sqrt(constrained)[None]*truncated_normal(self.rng, (self.candidates, self.horizon, 1))
            assert np.max(np.abs(samples)) <= 1.00000000001
            state = np.repeat(np.asarray(obs)[None], self.candidates, axis=0)
            reward = np.zeros(self.candidates)
            for t in range(self.horizon):
                action = samples[:, t]
                reward += known_reward(state, action)
                state = transition(state, action)
            elites = samples[np.argsort(reward)[-self.elites:]]
            mean = self.alpha*mean + (1-self.alpha)*elites.mean(0)
            var = self.alpha*var + (1-self.alpha)*elites.var(0)
        self.previous[:-1] = mean[1:]; self.previous[-1] = 0.
        return mean[0].copy()


def oracle_transition(mass, length):
    def f(obs, action):
        state = np.stack((np.arctan2(obs[:, 1], obs[:, 0]), obs[:, 2]), -1)
        next_state, _ = dynamics(state, action[:, 0], mass, length)
        return observation(next_state)
    return f


def control_smoke(args):
    out = Path(args.output); rows=[]; traces={}; start=time.monotonic()
    # Deliberately small engineering manifest; neither model tuning nor benchmark test.
    cases = [('ID', 1., 1.), ('moderate_OOD', .5, 1.5), ('extreme_OOD', 1.8, .2)]
    if args.episodes == 1: cases = cases[:1]
    for i, (group, mass, length) in enumerate(cases):
        env=Pendulum(seed=args.seed+i, masses=[mass], lengths=[length]); initial=env.reset(); initial_state=env.state.copy()
        for policy in ('zero_action', 'random_action', 'analytic_CEM'):
            env.state=initial_state.copy(); env.nsteps=env.nsteps_vertical=0; env.success=False
            rng=np.random.RandomState(args.seed+i+10); planner=CEM(args.seed+i+100)
            obs=initial.copy(); observations=[obs]; actions=[]; rewards=[]; successes=[]; before=time.monotonic()
            for _ in range(200):
                a = np.array([0.]) if policy=='zero_action' else rng.uniform(-1,1,size=1) if policy=='random_action' else planner.plan(obs, oracle_transition(mass,length))
                obs,reward,done,_=env.step(a)
                assert not done
                observations.append(obs);actions.append(a);rewards.append(reward);successes.append(env.success)
            key=f'{group}_{policy}'; seconds=time.monotonic()-before
            traces[key+'_obs']=np.asarray(observations);traces[key+'_actions']=np.asarray(actions);traces[key+'_rewards']=np.asarray(rewards)
            row=dict(group=group,mass=mass,length=length,policy=policy,return_sum=float(sum(rewards)),
                     final_success=bool(env.success),ever_success=bool(any(successes)),seconds=seconds,
                     real_environment_steps=200,steps_per_second=200/max(seconds,1e-9),initial_state=initial_state.tolist())
            rows.append(row); print(json.dumps(row), flush=True)
    np.savez_compressed(out/'CONTROL_ENGINEERING_TRACES.npz',**traces)
    idoracle=next(x for x in rows if x['group']=='ID' and x['policy']=='analytic_CEM')
    idrandom=next(x for x in rows if x['group']=='ID' and x['policy']=='random_action')
    result=dict(status='COMPLETE',rows=rows,seconds=time.monotonic()-start,
                id_oracle_improves_random=idoracle['return_sum']>idrandom['return_sum'],
                planner_gate_pass=idoracle['return_sum']>idrandom['return_sum']+200 and idoracle['final_success'],
                planner=dict(horizon=30,candidates=200,elites=50,iterations=5,alpha=.1,truncated_normal='rejection [-2,2]',variance_reset_each_step=.25),
                oracle_receives_true_parameters=True,learned_models_receive_true_parameters=False,
                learning=False,trained_policy=False,scope='Engineering analytic-oracle sanity; not learned-method evidence.',
                trace_sha256=sha(out/'CONTROL_ENGINEERING_TRACES.npz'))
    write(out/'CONTROL_ENGINEERING.json',result);return result


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','all'])
    p.add_argument('--output',required=True)
    p.add_argument('--official-root',default=str(here/'third_party/CaDM'))
    p.add_argument('--gym-reference',default=str(here/'third_party/gym_0_16_pendulum.py'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes',type=int,choices=[1,3],default=1)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Completed engineering run: preserve and use a fresh output.'
    result=validate(args); print(json.dumps(result),flush=True)
    if args.action=='all':control_smoke(args)
    write(out/'COMPLETE.json',dict(status='COMPLETE',action=args.action,code_sha256=sha(__file__),learning=False,
            files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
