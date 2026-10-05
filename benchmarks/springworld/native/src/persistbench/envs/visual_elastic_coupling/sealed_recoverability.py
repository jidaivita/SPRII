"""Raw visual evidence before a separate evaluator-only scoring phase.

No fitted weights are learned here. The fixed, declared physical inference
recipe measures raw evidence, including negative/ambiguous histories. This
does not replace compressed formation, prediction or lifecycle evaluations.
"""
import argparse,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from .sealed_access import SealedAuthorization
from .sealed_bank import SealedBank
from .raw_evidence import CONFIGURATION,PARAMETERS,estimate,array_digest
from .dataset_snapshot import stable_digest
from .training_protocol import source_fingerprint
from .evaluation_blocks import digest

DONOR_CONDITIONS=('forced24','forced48','forced96','free24','free96','glide24','glide96','static24','static96',
    'M','F','MM','FF','MF','repeat_M','repeat_F')
QUERY_CONDITIONS=('query_only','query_matched24','query_matched48','query_matched96','query_wrong96')
METRICS=('parameter_log_squared_error','parameter_relative_error','approximate_interval80_coverage',
    'tracking_rmse_m','prediction_component_errors','failure_and_missing_counts','local_sensitivity')


def validate_profile(profile):
    fields={'stratum','resolution','mode','conditions','query_kind','query_budget','horizon','reference_layers','configuration','workers','compute_tier','metrics'}
    if set(profile)!=fields or profile['configuration']!=CONFIGURATION or profile['metrics']!=list(METRICS):raise ValueError('raw evidence recipe or fields differ')
    if profile['resolution'] not in (64,128) or not isinstance(profile['workers'],int) or profile['workers']<1 or profile['compute_tier']!='standard':raise ValueError('unregistered raw evidence resource/observation budget')
    if profile['mode'] not in ('donor_only','conditional'):raise ValueError('raw evidence mode differs')
    allowed=DONOR_CONDITIONS if profile['mode']=='donor_only' else QUERY_CONDITIONS
    conditions=profile['conditions']
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions).issubset(allowed):raise ValueError('raw evidence condition family differs')
    if profile['mode']=='donor_only':
        if any(profile[k] is not None for k in ('query_kind','query_budget','horizon')) or profile['reference_layers']!=[]:raise ValueError('donor-only profile has a query')
    else:
        kind=profile['query_kind'];budget=profile['query_budget']
        if kind not in ('cold','moving') or budget not in ((0,1) if kind=='cold' else (0,1,3,7,15,31,63,95)) or profile['horizon'] not in (1,4,16,32):raise ValueError('conditional query budget differs')
        if profile['reference_layers']!=['visual_theta','state_theta']:raise ValueError('both distinct evaluator-only reference layers required')
    return profile


def planned_cases(bank,profile):
    validate_profile(profile)
    if not isinstance(bank,SealedBank) or bank.closed:raise PermissionError('authorized raw evidence bank required')
    if bank.resolution!=profile['resolution'] or profile['stratum'] not in bank.plans:raise ValueError('raw evidence population/resolution differs')
    plan=bank.plans[profile['stratum']]['plan'];slots={(r['system_key'],r['kind'],r['replicate']):r for r in bank.rows.values() if r['stratum']==profile['stratum']}
    result=[]
    for case in plan['cases']:
        key=case['system_key'];s=case['episode_slots'];forced=(key,'forced',s['forced'][0]);wrong=(case['wrong_system_key'],'forced',s['forced'][0])
        mass=[(key,'mass',i) for i in s['mass']];free=[(key,'free',i) for i in s['free']]
        members={**{'forced'+str(n):[(forced,n)] for n in (24,48,96)},**{'free'+str(n):[(free[0],n)] for n in (24,96)},
            'M':[(mass[0],48)],'F':[(free[0],48)],'MM':[(x,48) for x in mass],'FF':[(x,48) for x in free],
            'MF':[(mass[0],48),(free[0],48)],'repeat_M':[(mass[0],48)]*2,'repeat_F':[(free[0],48)]*2,
            'query_only':[],**{'query_matched'+str(n):[(forced,n)] for n in (24,48,96)},'query_wrong96':[(wrong,96)]}
        for kind in ('glide','static'):
            if kind in s:
                for n in (24,96):members[kind+str(n)]=[((key,kind,s[kind][0]),n)]
        for condition in profile['conditions']:
            if condition not in members:raise ValueError('certificate slots absent from frozen plan; cannot silently add episodes')
            sources=[]
            for slot,n in members[condition]:
                row=slots[slot];sources.append(dict(episode_key=row['episode_key'],start=0,frames=n,available=row['raw_frames']>=n))
            query=None
            if profile['mode']=='conditional':
                kind=profile['query_kind'];row=slots[(key,kind,s[kind][0])];anchor=1 if kind=='cold' else 95
                if row['anchor']!=anchor:raise ValueError('raw query anchor differs')
                query=dict(episode_key=row['episode_key'],anchor=anchor,start=anchor-profile['query_budget'],frames=profile['query_budget']+1,
                    available=row['raw_frames']>=anchor+1,future_available=row['raw_frames']>=anchor+profile['horizon']+1)
                if query['episode_key'] in {r['episode_key'] for r in sources}:raise ValueError('raw certificate donor/query overlap')
            result.append(dict(index=len(result),case_id=case['case_id'],block_id=case['block_id'],system_key=key,replicate=case['replicate'],condition=condition,
                sources=sources,query=query,observed_support_available=all(r['available'] for r in sources) and (query is None or query['available'])))
    return result


def public_inputs(bank,case,profile):
    """Return only authorized arrays; metadata/labels stay outside the worker."""
    if not case['observed_support_available']:return None,None,None
    histories=[];commitments=[]
    for source in case['sources']:
        images,actions=bank.visible(source['episode_key']);start=source['start'];end=start+source['frames']
        h=dict(images=images[start:end].copy(),actions=actions[start:end-1].copy());histories.append(h)
        commitments.append(array_digest(h['images'],h['actions']))
    query=None;source=case['query']
    if source:
        images,actions=bank.visible(source['episode_key']);start=source['start'];anchor=source['anchor']
        query=dict(images=images[start:anchor+1].copy(),kind=profile['query_kind'],budget=profile['query_budget'],horizon=profile['horizon'],
            future_actions=actions[anchor:anchor+profile['horizon']].copy() if source['future_available'] else None)
    query_digest=None if query is None else array_digest(query['images'],query['future_actions'] if query['future_actions'] is not None else np.empty((0,2),np.float32))
    return histories,query,dict(histories=commitments,query=query_digest)


def estimate_job(job):
    histories,query,resolution,configuration=job
    before=[array_digest(h['images'],h['actions']) for h in histories]
    query_before=None if query is None else array_digest(query['images'],query['future_actions'] if query['future_actions'] is not None else np.empty((0,2),np.float32))
    result=estimate(histories,query,resolution=resolution,configuration=configuration)
    if before!=[array_digest(h['images'],h['actions']) for h in histories] or query_before!=(None if query is None else array_digest(query['images'],query['future_actions'] if query['future_actions'] is not None else np.empty((0,2),np.float32))):raise ValueError('raw estimator mutated public evidence')
    return result


def parameter_errors(summary,theta):
    if summary is None:return None
    theta=np.asarray(theta,float)
    if theta.shape!=(3,) or not np.isfinite(theta).all() or np.any(theta<=0) or summary.get('parameters')!=list(PARAMETERS):raise ValueError('physical target order or support differs')
    truth=np.r_[theta,theta[2]/theta[0]]
    mean=np.asarray(summary['mean']);interval=np.asarray(summary['interval80'])
    if mean.shape!=(4,) or interval.shape!=(2,4) or not np.isfinite(mean).all() or not np.isfinite(interval).all() or np.any(mean<=0) or np.any(interval<=0) or np.any(interval[0]>interval[1]):raise ValueError('invalid raw parameter estimate')
    return {name:dict(relative_error=float(abs(mean[i]-truth[i])/truth[i]),log_squared_error=float(np.log(mean[i]/truth[i])**2),
        covered80=bool(interval[0,i]<=truth[i]<=interval[1,i])) for i,name in enumerate(PARAMETERS)}


def summarize(rows):
    """Equal system weight; missing cases never disappear from full means."""
    result={}
    for condition in sorted({r['condition'] for r in rows}):
        selected=[r for r in rows if r['condition']==condition];keys=sorted({r['system_key'] for r in selected})
        missing=sum(r['parameter_errors'] is None for r in selected);failures=sum(r['outcome_status']!='EXECUTED' for r in selected)
        entry=dict(cases=len(selected),systems=len(keys),missing_parameter_cases=missing,method_or_support_failures=failures,parameters={})
        for name in PARAMETERS:
            values=[r['parameter_errors'][name] for r in selected if r['parameter_errors'] is not None]
            full={}
            for metric in ('relative_error','log_squared_error','covered80'):
                full[metric]=None if missing else float(np.mean([np.mean([r['parameter_errors'][name][metric] for r in selected if r['system_key']==key]) for key in keys]))
            available=np.asarray([v['relative_error'] for v in values])
            entry['parameters'][name]=dict(full_system_weighted_mean=full,available_case_count=len(values),
                available_only_relative_error_quantiles=None if not len(values) else np.quantile(available,[.5,.9,.95,.99]).tolist(),
                tail_qualification='descriptive conditional on available estimates; inspect missing and fallback counts')
        result[condition]=entry
    return result


def run(args):
    authority=SealedAuthorization(args.protocol,args.selection,protocol_sha256=args.protocol_sha256,selection_sha256=args.selection_sha256)
    profile=validate_profile(authority.protocol['assays']['raw_recoverability']['profiles'][args.profile])
    if args.output.exists():raise ValueError('raw evidence attempt exists; retain old attempts')
    args.output.mkdir(parents=True);prediction_path=args.output/'ESTIMATES_BEFORE_LABELS.private.jsonl';rows=[];execution_errors=[];test_opened=False;began=time.monotonic()
    try:
        with SealedBank(args.bank,authority,admission_path=args.admission,admission_sha256=args.admission_sha256,audit_path=args.output/'EXTRACTION_ACCESS.private.jsonl',allow_labels=False,resolution=profile['resolution']) as bank:
            test_opened=True;cases=planned_cases(bank,profile)
            (args.output/'SOURCES.private.json').write_text(json.dumps(cases,indent=2)+'\n');case_digest=digest(cases)
            with prediction_path.open('x') as output,ProcessPoolExecutor(max_workers=profile['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                pending={}
                def record(case,result,commitment):
                    output.write(json.dumps(dict(index=case['index'],result=result,input_commitment=commitment),allow_nan=False)+'\n');output.flush()
                def drain():
                    ready,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in ready:
                        case,commitment=pending.pop(future)
                        try:result=future.result()
                        except Exception as exc:
                            result=dict(status='EXECUTION_FAILED',error=repr(exc),parameter_estimate=None,prediction=None,tracked=[]);execution_errors.append(dict(index=case['index'],error=repr(exc)))
                        record(case,result,commitment)
                for case in cases:
                    histories,query,commitment=public_inputs(bank,case,profile)
                    if histories is None:record(case,dict(status='SUPPORT_UNAVAILABLE',parameter_estimate=None,prediction=None,tracked=[]),None);continue
                    pending[pool.submit(estimate_job,(histories,query,profile['resolution'],profile['configuration']))]=(case,commitment)
                    if len(pending)>=2*profile['workers']:drain()
                while pending:drain()
            bank._audit('ALL_UNKNOWN_ESTIMATES_SAVED',cases=len(cases),execution_errors=len(execution_errors))
        commitment=stable_digest(prediction_path);(args.output/'ESTIMATE_COMMITMENT.json').write_text(json.dumps(commitment)+'\n')
        # The unknown estimator is finished before evaluator labels/references.
        with SealedBank(args.bank,authority,admission_path=args.admission,admission_sha256=args.admission_sha256,audit_path=args.output/'SCORING_ACCESS.private.jsonl',allow_labels=True,resolution=profile['resolution']) as bank:
            if digest(planned_cases(bank,profile))!=case_digest:raise ValueError('raw scoring plan changed')
            seen=set();reference_cache={}
            from .calibration import component_errors
            from .raw_evidence import privileged_references
            for line in prediction_path.read_text().splitlines():
                item=json.loads(line);index=item['index']
                if index in seen or index not in range(len(cases)):raise ValueError('duplicate or unplanned raw estimate')
                seen.add(index);case=cases[index];result=item['result'];theta=bank.systems[('test',case['system_key'])]['theta']
                histories,query,current=public_inputs(bank,case,profile)
                if current!=item['input_commitment']:raise ValueError('scored public inputs differ from the estimation phase')
                row=dict(**{k:case[k] for k in ('index','case_id','block_id','system_key','replicate','condition')},outcome_status=result['status'],
                    parameter_errors=parameter_errors(result.get('parameter_estimate'),theta),theta=theta,input_commitment=current,tracking=[],prediction_errors=None,references=None)
                fit=result.get('fit') or {}
                row['evidence_diagnostics']=dict(fit_status=fit.get('status'),optimizer_success=fit.get('optimizer_success'),
                    local_parameter_rank=fit.get('local_parameter_rank'),parameter_dimension=fit.get('parameter_dimension'),
                    fit_error=result.get('fit_error'),zero_force_scale_ambiguity=result.get('zero_force_scale_ambiguity'),
                    no_observed_history_excitation=result.get('no_observed_history_excitation'),processed_frames=result.get('processed_frames'),unique_frames=result.get('unique_frames'))
                for source,tracked in zip(case['sources'],result['tracked']):
                    states=bank.labels(source['episode_key'])[source['start']:source['start']+source['frames']]
                    difference=np.asarray(tracked['positions'])-states[:,:4]
                    row['tracking'].append(dict(episode_key=source['episode_key'],position_rmse_m=float(np.sqrt(np.mean(difference**2)))))
                source=case['query']
                if source and result.get('query_positions') is not None:
                    observed=bank.labels(source['episode_key'])[source['start']:source['anchor']+1,:4]
                    row['query_tracking_rmse_m']=float(np.sqrt(np.mean((np.asarray(result['query_positions'])-observed)**2)))
                if source and source['future_available'] and query is not None:
                    state=bank.labels(source['episode_key']);initial=state[source['anchor']];target=state[source['anchor']+profile['horizon']]-initial
                    def errors(prediction):
                        values=component_errors(prediction,target);scale=np.asarray(bank.statistics['scale'][str(profile['horizon'])])
                        values['joint_train_standardized_mse']=float(np.mean(((np.asarray(prediction)-target)/scale)**2));return values
                    if result.get('prediction') is not None:row['prediction_errors']=errors(result['prediction'])
                    refkey=source['episode_key']
                    if refkey not in reference_cache:reference_cache[refkey]=privileged_references(query,theta,initial,resolution=profile['resolution'])
                    reference=reference_cache[refkey];row['references']={name:None if reference[name] is None else errors(reference[name]) for name in profile['reference_layers']}
                    row['visual_theta_error']=reference.get('visual_theta_error')
                    row['target_sha256']=array_digest(target)
                rows.append(row)
            if len(seen)!=len(cases):raise ValueError('raw certificate estimates incomplete')
        if stable_digest(prediction_path)!=commitment:raise ValueError('unknown estimates changed after truth access')
        report=dict(schema='vec.formal-raw-recoverability-result.v1',status='PASS' if not execution_errors else 'FAILURES_RETAINED',profile=profile,cases=rows,
            summary=summarize(rows),execution_errors=execution_errors,protocol_sha256=authority.protocol_sha256,selection_sha256=authority.selection_sha256,
            admission_sha256=args.admission_sha256,estimates_commitment=commitment,source_fingerprint=source_fingerprint(),test_read=True,formal_results=not execution_errors,
            scope='raw pixel/action parameter and combination recovery, raw joint-donor composition, query-conditioned recovery and distinct visual-theta/state-theta references',
            interpretation='local trajectory-fit/Gaussian likelihood and bounded continuous prior; no global or discrete-bank identifiability claim; zero-force m/k scale ambiguity survives prior-dependent point estimates',
            statistical_status='descriptive equal-system recovery; independent-block/multiseed confidence statements require registered downstream analysis',seconds=time.monotonic()-began)
        (args.output/'RESULT.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output/'PUBLIC_RESULT.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ('cases','execution_errors')},indent=2,allow_nan=False)+'\n')
        completion=dict(status=report['status'],formal_results=not execution_errors,test_read=True,cases=len(rows),result_sha256=stable_digest(args.output/'RESULT.private.json')['sha256'])
    except Exception as exc:
        (args.output/'FAILED_PARTIAL.private.json').write_text(json.dumps(dict(cases=rows,errors=execution_errors,error=repr(exc)),indent=2)+'\n')
        (args.output/'COMPLETION.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc),test_read=test_opened,formal_results=False),indent=2)+'\n');raise
    (args.output/'COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n')
    if execution_errors:raise RuntimeError('raw evidence execution errors retained; assay incomplete')
    return completion


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','selection','bank','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','selection-sha256','admission-sha256','profile'):p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args())),flush=True)


if __name__=='__main__':main()
