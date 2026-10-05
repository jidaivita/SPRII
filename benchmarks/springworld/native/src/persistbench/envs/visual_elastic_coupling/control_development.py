"""Paired closed-loop development evaluation of historical and online evidence."""
import argparse,hashlib,itertools,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from persistbench.contracts import RunContext,EpisodeContext,QueryBatch,ComputeTier,Split
from .schema import Config,Parameters,Episode,history_payload
from .adapters import z_experience,_digest
from .control_agent import OnlineVisualControlAgent,FIT_STEPS
from .control_calibration import center_feedback,control_metrics

CONDITIONS=('query_only_online','matched_online','wrong_online','query_only_fixed','same_visual_theta','privileged_state_theta')


def boundary_sample(bank):
    keys=bank.selection_keys;values=np.asarray([bank.systems[k]['theta'] for k in keys]);normalized=(values-[.5,.25,4.])/[1.5,1.25,21.]
    chosen=[]
    for corner in itertools.product((0.,1.),repeat=3):
        distances=np.sum((normalized-corner)**2,axis=1)
        selected=next(i for i in np.argsort(distances,kind='stable') if i not in chosen);chosen.append(selected)
    return [keys[i] for i in chosen]


def run_case(job,*,split='validation',prior_adapter=None):
    from .physics import trajectory,InvalidTrajectory
    from .rendering import VisualRenderer
    root,output,system,query_row,donor_row,replicate,condition,index=job[:8];root=Path(root);cfg=Config(resolution=128)
    prior_spec=job[8] if len(job)==9 else None
    if len(job) not in (8,9):raise ValueError('invalid control job')
    if split not in ('validation','test'):raise ValueError('control requires an evaluation split')
    if prior_spec and prior_adapter is not None:raise ValueError('ambiguous control prior')
    theta=np.asarray(system['theta']);case_root=Path(output)/'cases'/f'{index:05d}';case_root.mkdir(parents=True)
    with np.load(root/query_row['private_state_path'],allow_pickle=False) as data:state=data['state'][query_row['anchor']].copy()
    with np.load(root/query_row['assets']['128']['path'],allow_pickle=False) as data:initial_images=data['images'][:2].copy()
    original_initial=state.copy();query_hash=hashlib.sha256(initial_images.tobytes()).hexdigest();goal=np.array([.2,0.]);began=time.monotonic()
    privileged=condition=='privileged_state_theta';agent=None;donor_key=None;prior=prior_adapter;frozen_before=None
    if prior_spec:
        if condition not in ('query_only_online','matched_online','wrong_online'):raise ValueError('learned control adapter condition differs')
        from .learned_control_prior import load_prior
        prior=load_prior(**prior_spec)
    if prior is not None:
        if condition not in ('query_only_online','matched_online','wrong_online'):raise ValueError('learned control adapter condition differs')
        frozen_before=prior.frozen_fingerprint()
    if not privileged:
        agent=OnlineVisualControlAgent(cfg,prior_adapter=prior,known_parameters=theta if condition=='same_visual_theta' else None,adaptive=condition!='query_only_fixed')
        agent.initialize(RunContext('visual_elastic_coupling/control','1.1',Split(split),ComputeTier.STANDARD,991045))
        if condition in ('matched_online','wrong_online'):
            agent.reset(EpisodeContext('opaque_donor'))
            with np.load(root/donor_row['assets']['128']['path'],allow_pickle=False) as data:images=data['images'][:96];actions=data['actions'][:95]
            ep=Episode(images,actions,np.arange(96)*.05,np.zeros((96,8)),{})
            experience=z_experience(history_payload(ep,0,95));agent.ingest(experience)
            experience.observations.fill(0);experience.actions.fill(0);donor_key=donor_row['episode_key']
        agent.reset(EpisodeContext('opaque_fresh_control'))
    states=[state.copy()];actions=[];frames=[initial_images[-1].copy()];estimates=[];parameters=[];fits=[];failure=None;max_memory=0;fit_failures=0
    with VisualRenderer(cfg) as renderer:
        if not np.array_equal(renderer.frame(state),initial_images[-1]):raise ValueError('online control renderer differs from stored initial query')
        for step in range(320):
            if privileged:action=center_feedback(state,theta,goal,cfg);estimate=state.copy();point=theta
            else:
                payload=dict(images=initial_images.copy() if step==0 else frames[-1][None].copy(),elapsed_seconds=step*.05,goal_xy=goal.copy(),
                    target_spec='move_and_stabilize_v1',initial_profile='natural_length_rest')
                executed=np.empty((0,2)) if step==0 else np.asarray(actions[-1])[None].copy()
                incoming=QueryBatch(('opaque_control_query',),('opaque_fresh_control',),payload,executed,metadata={'observation_schema':'causal_control_gray_v1','action_schema':'previous_executed_action'})
                before=(_digest(incoming.observations),_digest(incoming.actions));response=agent.respond(incoming)
                if before!=(_digest(incoming.observations),_digest(incoming.actions)):raise ValueError('controller mutated public observation')
                action=np.asarray(response.values['action']);diagnostic=response.diagnostics;estimate=np.asarray(diagnostic['estimate']);point=np.asarray(diagnostic['parameters'])
                if diagnostic['current_fit'] is not None:fits.append(dict(step=step,**diagnostic['current_fit']))
                max_memory=max(max_memory,diagnostic['total_mutable_numeric_bytes']);fit_failures=diagnostic['fit_failures']
            if action.shape!=(2,) or not np.isfinite(action).all() or np.linalg.norm(action)>.400001:raise ValueError('controller exceeded shared force cap')
            actions.append(action.copy());estimates.append(estimate.copy());parameters.append(point.copy())
            try:state=trajectory(Parameters(*theta),state,action[None],cfg)[-1]
            except InvalidTrajectory as exc:failure=dict(reason=exc.reason,time_s=step*.05+exc.time);break
            states.append(state.copy());frames.append(renderer.frame(state))
    states=np.asarray(states);actions=np.asarray(actions);frames=np.asarray(frames)
    if prior is not None and prior.frozen_fingerprint()!=frozen_before:raise ValueError('control evaluation changed frozen learned artifact')
    np.savez_compressed(case_root/'public_rollout.npz',images=frames,actions=actions[:len(frames)-1],timestamps=np.arange(len(frames))*.05)
    np.savez_compressed(case_root/'private_rollout.npz',states=states,attempted_actions=actions,estimates=estimates,parameters=parameters,goal=goal)
    result=dict(index=index,split=split,system_key=system['system_key'],theta=theta.tolist(),stratum=system['stratum'],replicate=replicate,condition=condition,
        source_query_episode=query_row['episode_key'],source_donor_episode=donor_key,initial_query_sha256=query_hash,initial_state_sha256=hashlib.sha256(original_initial.tobytes()).hexdigest(),
        completed_transitions=len(states)-1,failure=failure,metrics=control_metrics(states,actions,goal,cfg),online_fits=fits,fit_failures=fit_failures,
        max_mutable_numeric_bytes=max_memory,seconds=time.monotonic()-began,privileged_parameters=condition in ('same_visual_theta','privileged_state_theta'),privileged_state=privileged,
        history_method=prior.calibration['family'] if prior is not None else 'explicit',frozen_artifact_fingerprint=frozen_before)
    (case_root/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--replicates',type=int,default=1);p.add_argument('--full',action='store_true');a=p.parse_args()
    from .pixel_training import TrainingBank
    from .prediction_assay import donor_assignment
    from .training_protocol import source_fingerprint
    from .fast_reference import COMPILER_VERSION
    if a.output.exists():raise ValueError('control attempt exists')
    if a.replicates not in (1,2) or a.workers<1:raise ValueError('invalid development control budget')
    bank=TrainingBank(a.bank);keys=bank.selection_keys if a.full else boundary_sample(bank);jobs=[];assignments=[]
    for rep in range(a.replicates):
        wrong=donor_assignment(keys,991055+rep);assignments.append({key[1]:keys[wrong[i]][1] for i,key in enumerate(keys)})
        for i,key in enumerate(keys):
            query_rows=[r for r in bank.by_system[key]['cold'] if r['template']=='pulse_050'];query=query_rows[rep]
            matched=bank.eligible(key,'forced',96)[rep];mismatched=bank.eligible(keys[wrong[i]],'forced',96)[rep]
            for condition in CONDITIONS:
                jobs.append((str(a.bank),str(a.output),bank.systems[key],query,mismatched if condition=='wrong_online' else matched,rep,condition,len(jobs)))
    a.output.mkdir(parents=True);config=dict(schema='vec.control-development.v1.1',status='DEVELOPMENT_NOT_FROZEN',source_fingerprint=source_fingerprint(),
        bank_manifest_sha256=bank.manifest_sha256,systems=len(keys),cases=len(jobs),replicates=a.replicates,conditions=CONDITIONS,assignments=assignments,
        population='all100 main validation systems' if a.full else '8 result-blind nearest distinct systems to normalized parameter-box corners',
        initial_profile='actual independent cold episode, public rest/natural length and two frames; old future actions discarded',
        task='center goal[.2,0],320 steps at.05s,forcecap.4; same sustained4tolerances as competency010',
        controller='shared certainty-equivalent centerPD Kp1 Kd2, causal full-state visual observer gains .35 and.08/dt',
        online='full nonlinear trajectory fit of actual observed4D positions and executedactions; known public initial-rest/natural-length constraint; growing prefix replaces old query likelihood; static donor factor included once',
        fit_steps=FIT_STEPS,fit_max_nfev=100,compiler_version=COMPILER_VERSION,
        limitations='competent restricted controller/inference family, not Bayes optimality or a normative control information-value theorem; adaptive query-only must be checked against fixed prior and theta references before qualification',
        formal_results=False,test_read=False)
    (a.output/'CONTROL_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');rows=[];errors=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,j):j[-1] for j in jobs}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:errors.append(dict(index=futures[future],error=repr(exc)))
            print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors))),flush=True)
    (a.output/'CONTROL_REPORT.json').write_text(json.dumps(dict(config=config,cases=rows,errors=errors,seconds=time.monotonic()-began),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
