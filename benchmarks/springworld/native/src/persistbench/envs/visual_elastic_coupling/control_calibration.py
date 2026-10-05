"""Competency calibration for one bounded move-and-stabilize control task.

This uses feedback PD on the exactly decoupled center equation, with a force
cap and passive internal damping. It makes no global optimal-control claim.
The visual reference uses a causal model-prediction/measurement observer.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .schema import Config,Parameters
from .identification import reference_rollout
from .calibration import track
from .cold_opportunity import visual_rest_state
from .calibration_queries import query_initial
from .control_scoring import control_metrics


def center_feedback(estimate,parameters,goal,config=Config(),force_cap=.4):
    m,gamma,_=parameters
    center=(estimate[:2]+estimate[2:4])/2
    velocity=(estimate[4:6]+estimate[6:8])/2
    force=2*m*((goal-center)-(2.-gamma)*velocity)
    norm=np.linalg.norm(force)
    if norm>force_cap:force=force*(force_cap/norm)
    return np.asarray(force/config.force_max,np.float32)


class VisualObserver:
    def __init__(self,parameters,first_images,config):
        self.parameters=np.asarray(parameters,float).copy();self.config=config
        self.state=visual_rest_state(track(first_images,config)[-1],config)

    def update(self,image,previous_action):
        predicted=reference_rollout(self.parameters,self.state,np.asarray(previous_action)[None],self.config)[-1]
        measured=track(np.asarray(image)[None],self.config)[0]
        innovation=measured-predicted[:4]
        predicted[:4]+=.35*innovation
        predicted[4:]+=.08/self.config.control_dt*innovation
        self.state=predicted
        return predicted.copy()


def run_case(job):
    from .physics import trajectory,InvalidTrajectory
    from .rendering import VisualRenderer
    index,theta,replicate,condition,out=job
    cfg=replace(Config(),resolution=128);goal=np.array([.2,0.])
    root=Path(out)/'cases'/f'{index:05d}';root.mkdir(parents=True)
    state=query_initial(replicate,cfg)
    query_states=trajectory(Parameters(*theta),state,np.zeros((1,2)),cfg)
    state=query_states[-1];states=[state.copy()];estimates=[];actions=[];failure=None
    start=time.monotonic();sample_frames=[]
    with VisualRenderer(cfg) as renderer:
        query_images=np.stack([renderer.frame(s) for s in query_states])
        observer=None if condition=='privileged_state_theta' else VisualObserver(theta,query_images,cfg)
        estimate=state.copy() if observer is None else observer.state.copy()
        for step in range(320):
            action=center_feedback(estimate,theta,goal,cfg)
            estimates.append(estimate.copy());actions.append(action)
            try:state=trajectory(Parameters(*theta),state,action[None],cfg)[-1]
            except InvalidTrajectory as exc:
                failure=dict(reason=exc.reason,time_s=step*cfg.control_dt+exc.time);break
            states.append(state.copy())
            frame=renderer.frame(state)
            if index<6 and step%4==0:sample_frames.append(frame)
            estimate=state.copy() if observer is None else observer.update(frame,action)
    states=np.asarray(states);actions=np.asarray(actions)
    np.savez_compressed(root/'private_rollout.npz',state=states,actions=actions,estimates=estimates,goal=goal,
                        query_images=query_images,preview_frames=np.asarray(sample_frames))
    row=dict(index=index,theta=dict(zip(('m','gamma','k'),theta)),replicate=replicate,condition=condition,
        status='EXECUTED' if failure is None else 'PHYSICS_REJECTED',failure=failure,
        completed_transitions=len(states)-1,metrics=control_metrics(states,actions,goal,cfg),seconds=time.monotonic()-start)
    (root/'report.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.output.exists():raise ValueError('use a fresh attempt')
    args.output.mkdir(parents=True)
    bank=json.loads((args.bank/'BANK_CONFIG.json').read_text())
    if bank['split']!='development':raise ValueError('development only')
    jobs=[]
    for theta in bank['parameter_grid']:
        for replicate in range(3):
            for condition in ('privileged_state_theta','same_visual_theta'):
                jobs.append((len(jobs),theta,replicate,condition,str(args.output)))
    if args.smoke:jobs=[jobs[i] for i in (0,1,96,97,288,289,382,383)]
    config=dict(schema='vec.control-competency.v1.1',bank=str(args.bank),test_read=False,formal_training=False,
        job_count=len(jobs),force_cap_n=.4,feedback_interval_s=.05,goal=[.2,0.],
        controller='center PD with known m/gamma compensation, Kp=1, Kd=2; force norm bounded',
        visual_observer='causal model prediction plus position .35 and velocity .08/dt innovation',
        public_initial_profile='natural length and rest; two initial images; no prior query actions exposed',
        candidate_horizons_s=[8,12,16],success='all four tolerances held for final .5 seconds',
        purpose='independent competency check; not global optimality or historical utility',smoke=args.smoke)
    (args.output/'CONTROL_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[];errors=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,j):j[0] for j in jobs}
        with (args.output/'progress.jsonl').open('w') as f:
            for future in as_completed(futures):
                try:row=future.result();rows.append(row);event=dict(index=row['index'],status=row['status'],success=[m['success'] for m in row['metrics']])
                except Exception as exc:event=dict(index=futures[future],error=repr(exc));errors.append(event)
                f.write(json.dumps(event)+'\n');f.flush()
                if (len(rows)+len(errors))%16==0 or args.smoke:print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),last=event)),flush=True)
    (args.output/'CONTROL_REPORT.json').write_text(json.dumps(dict(config=config,cases=rows,errors=errors,
        elapsed_seconds=time.monotonic()-start,qualification='PENDING_ANALYSIS'),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
