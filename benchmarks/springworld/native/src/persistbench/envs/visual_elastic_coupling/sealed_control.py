"""Frozen closed-loop control on exact planned episodes and source commitments.

The same physical runner, visual observer and certainty-equivalent controller
as development are used. Only evaluator-owned reference conditions receive
true parameters/state; learned and explicit methods receive causal pixels and
their previous executed action. Physical failures remain scored outcomes.
"""
import argparse,hashlib,json,multiprocessing,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .sealed_access import SealedAuthorization
from .sealed_bank import SealedBank
from .sealed_formation import validate_readout_bundle
from .formation import EXPLICIT_CONFIGURATION,SELECTION_STRATA
from .dataset_snapshot import stable_digest
from .training_protocol import source_fingerprint
from .evaluation_blocks import digest
from .control_scoring import CENTER_INTEGRAL_RULE

MAIN_CONDITIONS=('query_only_online','matched_online','wrong_online')
REFERENCE_CONDITIONS=('query_only_fixed','same_visual_theta','privileged_state_theta')
CONTRACT=dict(schema='vec.frozen-control-contract.v1',resolution=128,initial_frames=2,initial_profile='natural_length_rest',
    goal_xy=[.2,0.],transitions=320,feedback_interval_s=.05,force_cap_n=.4,horizons_s=[8.,12.,16.],
    controller='certainty_equivalent_center_pd',Kp=1.,Kd=2.,observer_position_gain=.35,observer_velocity_gain=.08,
    fit_steps=[10,20,40,80,160,240],fit_max_nfev=100,evaluation_seed=991045,
    success_tolerances=dict(center_error_m=.02,center_speed_m_s=.02,relative_speed_m_s=.03,length_error_m=.02),
    sustained_span_s=.5,sustained_frames=11,center_error_integral_rule=CENTER_INTEGRAL_RULE,
    physics_failure='retain prefix; unavailable-horizon success is false; incomplete continuous costs are undefined',
    execution_failure='retain all errors; incomplete assay remains unqualified; no replacement or successful-case-only analysis')
METRICS=('sustained_success','integrated_center_error_m2_s','effort_n2_s','sustained_maxima','physical_failure','fit_failures')


def validate_profile(profile,*,kind):
    fields={'stratum','conditions','contract','metrics','compute_tier','workers','formation_profile','prior_role'}
    if set(profile)!=fields or profile['contract']!=CONTRACT or profile['metrics']!=list(METRICS):raise ValueError('unregistered control task, budget, metrics or failure policy')
    conditions=tuple(profile['conditions'])
    if conditions not in (MAIN_CONDITIONS,REFERENCE_CONDITIONS,MAIN_CONDITIONS+REFERENCE_CONDITIONS):raise ValueError('unregistered control condition family')
    if kind not in ('learned','explicit'):raise ValueError('unregistered control method kind')
    if kind=='learned' and (conditions!=MAIN_CONDITIONS or not profile['formation_profile'] or not profile['prior_role']):raise ValueError('learned control requires its frozen readout/calibration and nonprivileged conditions')
    if kind=='explicit' and (profile['formation_profile'] is not None or profile['prior_role'] is not None):raise ValueError('explicit control does not use learned readout roles')
    if not isinstance(profile['workers'],int) or profile['workers']<1:raise ValueError('positive frozen control worker budget required')
    if profile['compute_tier']!='standard':raise ValueError('control reference uses the registered standard compute tier')
    return profile


def planned_control_cases(bank,profile,*,kind):
    if not isinstance(bank,SealedBank) or bank.closed:raise PermissionError('open authorized sealed control bank required')
    validate_profile(profile,kind=kind)
    if bank.resolution!=128 or profile['stratum'] not in bank.plans:raise ValueError('control observation/population differs')
    plan=bank.plans[profile['stratum']]['plan'];slots={(r['system_key'],r['kind'],r['replicate']):r for r in bank.rows.values() if r['stratum']==profile['stratum']}
    cases=[]
    for case in plan['cases']:
        query=slots[(case['system_key'],'cold',case['episode_slots']['control_cold'][0])]
        if query['raw_frames']<2 or query['anchor']!=1:raise ValueError('planned causal control start is unavailable; no replacement')
        if query['episode_key']==slots[(case['system_key'],'cold',case['episode_slots']['cold'][0])]['episode_key']:raise ValueError('prediction and control use the same query episode')
        images=bank.visible(query['episode_key'])[0][:2];state=bank.labels(query['episode_key'])[1]
        if len(images)!=2 or not np.array_equal(images[0],images[1]) or np.linalg.norm(state[4:])>1e-8 or abs(np.linalg.norm(state[2:4]-state[:2])-.35)>1e-10:
            raise ValueError('stored control start violates public natural-length/rest profile')
        commitments=dict(initial_query_sha256=hashlib.sha256(images.tobytes()).hexdigest(),initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        for condition in profile['conditions']:
            donor=None
            if condition in ('matched_online','wrong_online'):
                key=case['wrong_system_key'] if condition=='wrong_online' else case['system_key']
                donor=slots[(key,'forced',case['episode_slots']['forced'][0])]
                if donor['raw_frames']<96 or donor['episode_key']==query['episode_key']:raise ValueError('planned control donor support unavailable or overlapping')
            cases.append(dict(case_id=case['case_id'],block_id=case['block_id'],system_key=case['system_key'],replicate=case['replicate'],
                condition=condition,system=bank.systems[('test',case['system_key'])],query=query,donor=donor,index=len(cases),**commitments))
    return cases


def expected_calibration(prediction,truth,systems):
    """Independent group-mean calculation of the frozen calibration recipe."""
    prediction=np.asarray(prediction,float);truth=np.asarray(truth,float);systems=np.asarray(systems)
    if prediction.shape!=truth.shape or prediction.ndim!=2 or prediction.shape[1]!=3 or systems.shape!=(len(prediction),):raise ValueError('invalid control calibration sample shapes')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all() or len(np.unique(systems))<2:raise ValueError('insufficient finite calibration systems')
    residual=truth-prediction;groups=[residual[systems==key] for key in np.unique(systems)]
    bias=np.mean([g.mean(0) for g in groups],axis=0)
    covariance=np.mean([np.einsum('ni,nj->ij',g-bias,g-bias)/len(g) for g in groups],axis=0)+np.eye(3)*.05**2
    return bias,covariance


def check_calibration(calibration,*,slot,paths,prediction,truth,systems,reused_readout=False):
    bindings=dict(schema='vec.learned-control-prior.v1.1',status='DEVELOPMENT_CALIBRATED',family=slot['family'],frames=96,test_read=False,
        checkpoint_sha256=stable_digest(paths['predictor'])['sha256'],mlp_sha256=stable_digest(paths['mlp'])['sha256'],
        feature_sha256=stable_digest(paths['features'])['sha256'],extraction_sha256=stable_digest(paths['extraction'])['sha256'],
        probe_report_sha256=stable_digest(paths['report'])['sha256'],source_fingerprint=source_fingerprint(),
        physical_prior_bounds=[[.5,.25,4.],[2.,1.5,25.]],log_std_floor=.05,systems=len(np.unique(systems)),histories=len(systems))
    if reused_readout:
        # Original files predate added source/sidecar fields. The caller has
        # independently audited those selected originals; do not invent them.
        bindings.pop('source_fingerprint')
        for key in ('extraction_sha256','probe_report_sha256'):
            if key not in calibration:bindings.pop(key)
    if any(calibration.get(k)!=v for k,v in bindings.items()):raise ValueError('control calibration artifact/profile binding differs')
    bias,covariance=expected_calibration(prediction,truth,systems)
    if np.asarray(calibration.get('bias')).shape!=(3,) or np.asarray(calibration.get('covariance')).shape!=(3,3):raise ValueError('invalid frozen calibration shape')
    if not np.allclose(calibration['bias'],bias,atol=1e-10,rtol=1e-10) or not np.allclose(calibration['covariance'],covariance,atol=1e-10,rtol=1e-10):raise ValueError('control calibration is not the registered system-balanced validation recipe')
    return calibration


def selected_worker_settings(authorization,slot,profile):
    """All fitting data are train/validation artifacts, checked before test open."""
    selected=authorization.selected_paths[slot['slot_id']]
    if slot['kind']=='explicit':
        if json.loads(selected['configuration'].read_text())!=EXPLICIT_CONFIGURATION:raise ValueError('explicit control configuration differs from its calibrated reference')
        return dict(kind='explicit')
    from .formation_readout import FrozenMLPReadout
    formation=authorization.protocol['assays']['formation']['profiles'][profile['formation_profile']]
    if formation['feature_condition']!='trained' or formation['frames']!=96 or formation['resolution']!=128:raise ValueError('control needs the frozen trained96frame readout')
    paths={role:selected[name] for role,name in formation['readout_roles'].items()};paths['predictor']=selected['predictor']
    from .sealed_formation import resolve_reuse_paths
    reuse_paths=resolve_reuse_paths(formation,selected,slot['kind'])
    checked=validate_readout_bundle({k:v for k,v in paths.items() if k!='predictor'},profile=formation,
        bank_manifest_sha256=authorization.protocol['training_bank']['manifest_sha256'],bank_content_sha256=authorization.protocol['training_bank']['content_sha256'],
        bank_snapshot_sha256=authorization.protocol['training_bank']['snapshot_sha256'],checkpoint_sha256=stable_digest(selected['predictor'])['sha256'],
        reuse_paths=reuse_paths,checkpoint_path=selected['predictor'])
    readout=FrozenMLPReadout(paths['mlp'])
    for name,values in checked['normalization'].items():
        if not np.array_equal(getattr(readout,name),values):raise ValueError('control MLP normalization differs from train-only values')
    with np.load(paths['features'],allow_pickle=False) as data:
        select=(data['split']=='validation')&np.isin(data['stratum'],SELECTION_STRATA)
        features=data['features'][select];truth=data['labels'][select,:3];systems=data['system_key'][select]
    calibration_path=selected[profile['prior_role']];calibration=json.loads(calibration_path.read_text())
    if calibration.get('bank_manifest_sha256')!=authorization.protocol['training_bank']['manifest_sha256']:raise ValueError('control calibration bank differs')
    check_calibration(calibration,slot=slot,paths=paths,prediction=readout.predict(features)[:,:3],truth=truth,systems=systems,reused_readout=checked['reused_original_artifacts'])
    paths['calibration']=calibration_path
    return dict(kind='learned',slot=slot,training_bank=authorization.protocol['training_bank'],
        paths={k:str(v) for k,v in paths.items() if k in ('predictor','mlp','calibration')},
        sha256={k:stable_digest(v)['sha256'] for k,v in paths.items() if k in ('predictor','mlp','calibration')})


def initialize_worker(settings):
    global WORKER_PRIOR
    WORKER_PRIOR=None
    if settings['kind']=='explicit':return
    import torch
    from .pixel_models import PixelDynamicsModel,PixelModelConfig
    from .pixel_training import runtime_policy
    from .formation_readout import FrozenMLPReadout
    from .learned_control_prior import LearnedParameterPrior
    paths={k:Path(v) for k,v in settings['paths'].items()}
    if any(stable_digest(paths[k])['sha256']!=sha for k,sha in settings['sha256'].items()):raise ValueError('worker frozen control files differ')
    slot=settings['slot'];artifact=torch.load(paths['predictor'],map_location='cpu',weights_only=True);config=artifact['config']
    reused=slot.get('admission_mode','prospective')=='pretest_existing'
    identity_ok=config.get('stage')=='development_training' if reused else config.get('stage')=='formal' and config.get('protocol_sha256')==slot['training_protocol_sha256']
    if not identity_ok or config['family']!=slot['family'] or config['seed']!=slot['seed']:
        raise ValueError('control checkpoint embedded formal identity differs')
    if config['bank_manifest_sha256']!=settings['training_bank']['manifest_sha256'] or config['model']['resolution']!=128:raise ValueError('control model observation/data differs')
    runtime_policy(slot['seed']);torch.set_num_threads(1);model=PixelDynamicsModel(PixelModelConfig(**config['model']))
    model.load_state_dict(artifact['model'],strict=True);model.eval().requires_grad_(False)
    WORKER_PRIOR=LearnedParameterPrior(model,FrozenMLPReadout(paths['mlp']),json.loads(paths['calibration'].read_text()))


def run_worker(job):
    from .control_development import run_case
    return run_case(job,split='test',prior_adapter=WORKER_PRIOR)


def check_paired_outcomes(rows,cases,*,split='test'):
    if split not in ('validation','test'):raise ValueError('unregistered control result split')
    expected={(c['index'],c['case_id'],c['condition']):c for c in cases};seen=set();groups={}
    for row in rows:
        identity=(row['index'],row['case_id'],row['condition'])
        if identity not in expected or identity in seen:raise ValueError('unexpected or duplicate control result')
        seen.add(identity);case=expected[identity]
        if row.get('split')!=split or row['system_key']!=case['system_key'] or row['replicate']!=case['replicate'] or row['theta']!=case['system']['theta']:raise ValueError('control result system/split differs')
        if row['source_query_episode']!=case['query']['episode_key'] or row['source_donor_episode']!=(case['donor']['episode_key'] if case['donor'] else None):raise ValueError('control result source pairing differs')
        if any(row[k]!=case[k] for k in ('initial_query_sha256','initial_state_sha256')):raise ValueError('control initial pixels/state differ from committed source')
        if row['privileged_parameters']!=(row['condition'] in ('same_visual_theta','privileged_state_theta')) or row['privileged_state']!=(row['condition']=='privileged_state_theta'):
            raise ValueError('control privilege declaration differs')
        if not 0<=row['completed_transitions']<=320 or (row['failure'] is None and row['completed_transitions']!=320):raise ValueError('control completion/failure declaration differs')
        groups.setdefault(case['case_id'],set()).add((row['initial_query_sha256'],row['initial_state_sha256']))
        if [m['horizon_s'] for m in row['metrics']]!=CONTRACT['horizons_s']:raise ValueError('control scoring horizons differ')
        for metric in row['metrics']:
            complete=row['completed_transitions']>=round(metric['horizon_s']/.05)
            if complete!=(metric['status']=='EXECUTED'):raise ValueError('control metric support differs from completed trajectory')
            if metric['status']=='INCOMPLETE':
                if metric['success'] is not False or any(k in metric for k in ('integrated_center_error_m2_s','effort_n2_s')):raise ValueError('incomplete control got an invented full-horizon score')
            elif metric['status']=='EXECUTED':
                if metric.get('center_error_integral_rule')!=CENTER_INTEGRAL_RULE:raise ValueError('control integral convention differs')
                if metric.get('sustained_frames')!=11 or metric.get('sustained_observation_span_s')!=.5 or metric.get('tolerances')!=CONTRACT['success_tolerances']:raise ValueError('control stability criterion differs')
                if metric['success']!=all(metric['sustained_maxima'][k]<=v for k,v in CONTRACT['success_tolerances'].items()):raise ValueError('control success inconsistent with all tolerances')
                values=[metric['integrated_center_error_m2_s'],metric['effort_n2_s'],*metric['sustained_maxima'].values()]
                if not np.isfinite(values).all() or min(values)<0:raise ValueError('nonfinite or negative control metric')
            else:raise ValueError('unregistered control outcome')
    if set(expected)!=seen or any(len(v)!=1 for v in groups.values()):raise ValueError('incomplete control results or changed initial conditions')
    return rows


def score_records(rows,*,horizon,metric,model_seed,profile_key):
    """Success includes physical failures; partial continuous costs are refused."""
    if horizon not in CONTRACT['horizons_s'] or metric not in ('sustained_success','integrated_center_error_m2_s','effort_n2_s'):raise ValueError('unknown control scoring endpoint')
    out=[]
    for row in rows:
        values=next(m for m in row['metrics'] if m['horizon_s']==horizon)
        if metric=='sustained_success':value=float(values['success'])
        elif values['status']!='EXECUTED':raise ValueError('continuous control contrast unavailable with incomplete cases; no successful-case-only reduction')
        else:value=float(values[metric])
        out.append(dict(case_id=row['case_id'],model_seed=model_seed,condition=row['condition'],profile_key=profile_key,status='OK',
            outcome_status='PHYSICS_FAILURE' if row['failure'] else 'EXECUTED',metrics={metric:value},query_fingerprint=row['initial_query_sha256'],
            target_fingerprint=digest([row['initial_state_sha256'],row['theta'],CONTRACT['goal_xy'],CONTRACT])))
    return out


def run(args):
    authorization=SealedAuthorization(args.protocol,args.selection,protocol_sha256=args.protocol_sha256,selection_sha256=args.selection_sha256)
    slot=next(s for s in authorization.protocol['method_slots'] if s['slot_id']==args.slot)
    profile=validate_profile(authorization.protocol['assays']['closed_loop_control']['profiles'][args.profile],kind=slot['kind'])
    settings=selected_worker_settings(authorization,slot,profile)
    runtime=authorization.protocol['render_runtime']
    if version('mujoco')!=runtime['mujoco_version'] or os.environ.get('MUJOCO_GL')!=runtime['backend']:raise ValueError('online control runtime differs from frozen physics/renderer')
    from .control_agent import FIT_STEPS
    if list(FIT_STEPS)!=CONTRACT['fit_steps']:raise ValueError('online control inference schedule changed')
    if args.output.exists():raise ValueError('control attempt exists; do not overwrite')
    args.output.mkdir(parents=True);rows=[];errors=[];test_opened=False;began=time.monotonic()
    try:
        with SealedBank(args.bank,authorization,admission_path=args.admission,admission_sha256=args.admission_sha256,
                        audit_path=args.output/'ACCESS.private.jsonl',allow_labels=True,resolution=128) as bank:
            test_opened=True;cases=planned_control_cases(bank,profile,kind=slot['kind'])
            bank._audit('CONTROL_DISPATCH',cases=len(cases),conditions=profile['conditions'],scope='private state supplies the physical initial condition; only named privileged references receive state/parameters')
            jobs=[(str(bank.root),str(args.output),c['system'],c['query'],c['donor'],c['replicate'],c['condition'],c['index']) for c in cases]
            with ProcessPoolExecutor(max_workers=profile['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(settings,)) as pool:
                futures={pool.submit(run_worker,j):c for j,c in zip(jobs,cases)}
                with (args.output/'progress.private.jsonl').open('x') as progress:
                    for future in as_completed(futures):
                        case=futures[future]
                        try:
                            row=future.result();row.update(case_id=case['case_id'],block_id=case['block_id']);rows.append(row)
                            event=dict(index=case['index'],case_id=case['case_id'],condition=case['condition'],physical_failure=row['failure'])
                        except Exception as exc:
                            event=dict(index=case['index'],case_id=case['case_id'],condition=case['condition'],error=repr(exc));errors.append(event)
                        progress.write(json.dumps(event)+'\n');progress.flush()
                        if (len(rows)+len(errors))%32==0:print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors))),flush=True)
            if not errors:check_paired_outcomes(rows,cases)
            bank._audit('CONTROL_COMPLETE',cases=len(rows),execution_errors=len(errors),physical_failures=sum(r['failure'] is not None for r in rows))
        # Includes every public/private saved rollout, and retained errors.
        rows.sort(key=lambda r:r['index']);assets={str(p.relative_to(args.output)):stable_digest(p) for p in sorted((args.output/'cases').rglob('*')) if p.is_file()}
        report=dict(schema='vec.formal-control-result.v1',status='PASS' if not errors else 'FAILURES_RETAINED',profile=profile,method_slot=args.slot,
            cases=rows,errors=errors,planned_cases=len(cases),physical_failures=sum(r['failure'] is not None for r in rows),assets=assets,
            protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256,bank_admission_sha256=args.admission_sha256,
            source_fingerprint=source_fingerprint(),test_read=True,formal_results=not errors,seconds=time.monotonic()-began,
            interpretation='utility of retained history under the same calibrated online estimator and bounded PD controller; not Bayes-optimal or autonomous learned control',
            statistical_status='complete paired trajectory outcomes; independent-block inference is separate; incomplete continuous metrics cannot be reduced over successes')
        (args.output/'RESULT.private.json').write_text(json.dumps(report,indent=2)+'\n')
        public={k:v for k,v in report.items() if k not in ('cases','errors','assets')};public['execution_error_count']=len(errors)
        (args.output/'PUBLIC_RESULT.json').write_text(json.dumps(public,indent=2)+'\n')
        completion=dict(schema='vec.formal-control-completion.v1',status=report['status'],test_read=True,formal_results=not errors,
            cases=len(rows),planned_cases=len(cases),execution_errors=len(errors),result_sha256=stable_digest(args.output/'RESULT.private.json')['sha256'])
    except Exception as exc:
        (args.output/'FAILED_PARTIAL.private.json').write_text(json.dumps(dict(cases=rows,errors=errors,error=repr(exc)),indent=2)+'\n')
        (args.output/'COMPLETION.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc),test_read=test_opened,formal_results=False),indent=2)+'\n');raise
    (args.output/'COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n')
    if errors:raise RuntimeError('control execution errors retained; formal assay is unqualified')
    return completion


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','selection','bank','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','selection-sha256','admission-sha256','slot','profile'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a)),flush=True)


if __name__=='__main__':main()
