"""Frozen formation probes, with representation/prediction before test labels.

Probe fitting remains in formation.py and rejects test bundles. This entry
only verifies already-selected train/validation artifacts, extracts from the
declared retained state, predicts, and then opens a separate scoring phase.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from .formation import LABELS,SELECTION_STRATA,FEATURE_DEFINITIONS,EXPLICIT_CONFIGURATION,train_standardize,grouped_scores,private_labels
from .formation_readout import ridge_predict
from .dataset_snapshot import stable_digest
from .training_protocol import source_fingerprint
from .evaluation_blocks import digest
from .sealed_access import SealedAuthorization
from .sealed_bank import SealedBank

ROLES=('features','extraction','report','ridge','mlp')
METRICS=('squared_error','train_standardized_squared_error','system_weighted_r2')


def validate_profile(profile):
    fields={'stratum','frames','resolution','feature_condition','feature_seed','compute_tier','decoders','metrics','readout_roles','probe_updates','probe_seed'}
    if not fields<=set(profile) or set(profile)-fields-{'reuse_roles','numerical_runtime','neural_source_checkpoint_role'}:raise ValueError('unrecognized or incomplete frozen formation profile')
    if profile['frames'] not in (24,48,96) or profile['resolution']!=128:raise ValueError('unimplemented formation observation profile')
    if profile['feature_condition'] not in ('trained','random_initialization'):raise ValueError('unregistered formation condition')
    if profile['feature_seed']!=990017 or profile['decoders']!=['ridge','mlp'] or profile['metrics']!=list(METRICS):raise ValueError('formation seed/decoders/metrics differ')
    if set(profile['readout_roles'])!=set(ROLES) or len(set(profile['readout_roles'].values()))!=len(ROLES):raise ValueError('five distinct frozen readout roles required')
    if not isinstance(profile['probe_updates'],int) or profile['probe_updates']<1 or not isinstance(profile['probe_seed'],int):raise ValueError('unregistered probe optimization budget')
    if 'reuse_roles' in profile:
        from .readout_reuse import ROLES as REUSE_ROLES
        roles=profile['reuse_roles']
        if set(roles) not in (set(REUSE_ROLES),set(REUSE_ROLES)|{'source_checkpoint'}):raise ValueError('incomplete readout reuse role registration')
        values=[*roles.values(),*profile['readout_roles'].values()]
        if len(values)!=len(set(values)):raise ValueError('readout reuse roles must be distinct')
    if 'neural_source_checkpoint_role' in profile:
        role=profile['neural_source_checkpoint_role']
        if 'reuse_roles' not in profile or 'source_checkpoint' in profile['reuse_roles'] or not isinstance(role,str) or not role or role in values:
            raise ValueError('invalid neural-only source checkpoint role')
    if 'numerical_runtime' in profile:
        runtime=profile['numerical_runtime']
        if set(runtime)!={'device_type','torch_version','feature_batch_size','precision','tf32','mha_fastpath','readout_device'}:
            raise ValueError('incomplete formation numerical runtime')
        if runtime['device_type'] not in ('cpu','cuda') or not runtime['torch_version'] or runtime['feature_batch_size']!=1 or runtime['precision']!='float32' or runtime['tf32'] is not False or runtime['mha_fastpath'] is not True or runtime['readout_device']!='cpu':
            raise ValueError('unsupported formation numerical runtime')
    from persistbench.contracts import ComputeTier
    ComputeTier(profile['compute_tier']);return profile


def resolve_reuse_paths(profile,selected,kind):
    """Common measurement profiles can bind an origin model only for learners."""
    if 'reuse_roles' not in profile:return None
    roles=dict(profile['reuse_roles'])
    if kind=='learned' and 'neural_source_checkpoint_role' in profile:
        roles['source_checkpoint']=profile['neural_source_checkpoint_role']
    return {role:selected[name] for role,name in roles.items()}


def validate_readout_bundle(paths,*,profile,bank_manifest_sha256,bank_content_sha256,bank_snapshot_sha256,checkpoint_sha256=None,explicit_configuration=None,reuse_paths=None,checkpoint_path=None):
    """No test bank is available here; normalization is independently recomputed."""
    validate_profile(profile)
    if set(paths)!=set(ROLES):raise ValueError('formation readout roles incomplete')
    extraction=json.loads(Path(paths['extraction']).read_text());report=json.loads(Path(paths['report']).read_text())
    reused='reuse_roles' in profile
    if reused:
        from .readout_reuse import validate
        roles=set(profile['reuse_roles'])
        if checkpoint_sha256 is not None and 'neural_source_checkpoint_role' in profile:roles.add('source_checkpoint')
        if reuse_paths is None or set(reuse_paths)!=roles:raise ValueError('registered readout reuse evidence missing')
        extraction,report=validate(paths,reuse_paths,profile=profile,bank_manifest_sha256=bank_manifest_sha256,
            bank_content_sha256=bank_content_sha256,bank_snapshot_sha256=bank_snapshot_sha256,
            checkpoint_sha256=checkpoint_sha256,checkpoint_path=checkpoint_path,explicit_configuration=explicit_configuration)
    elif reuse_paths is not None:raise ValueError('unregistered readout reuse route')
    method='neural' if checkpoint_sha256 is not None else 'explicit'
    bindings=dict(bank_manifest_sha256=bank_manifest_sha256,frames=profile['frames'],resolution=profile['resolution'],kind='forced',
        feature_seed=profile['feature_seed'],feature_definition=FEATURE_DEFINITIONS[method],method=method,frozen=True,test_read=False)
    if extraction.get('schema')!='vec.formation-feature.v1.1' or any(extraction.get(k)!=v for k,v in bindings.items()):raise ValueError('formation extraction profile/model/data differs')
    if not reused:
        if extraction.get('source_fingerprint')!=source_fingerprint() or report.get('source_fingerprint')!=source_fingerprint():raise ValueError('formation implementation differs from selected artifacts')
        content=extraction.get('bank_content_verification',{})
        if content.get('snapshot_sha256')!=bank_snapshot_sha256:raise ValueError('formation training snapshot differs')
        for boundary in ('before','after'):
            check=content.get(boundary,{})
            if check.get('status')!='PASS' or check.get('content_sha256')!=bank_content_sha256 or check.get('test_read') is not False:
                raise ValueError('formation extraction lacks before/after training content verification')
    if method=='neural':
        if extraction.get('checkpoint_sha256')!=checkpoint_sha256 or extraction.get('random_initialization_control')!=(profile['feature_condition']=='random_initialization'):
            raise ValueError('formation backbone or initialization condition differs')
    elif profile['feature_condition']!='trained' or extraction.get('explicit_configuration')!=explicit_configuration or explicit_configuration!=EXPLICIT_CONFIGURATION:
        raise ValueError('explicit posterior readout configuration differs')
    feature_sha=stable_digest(paths['features'])['sha256']
    if extraction.get('feature_sha256')!=feature_sha or report.get('features_sha256')!=feature_sha or report.get('extraction_sha256')!=stable_digest(paths['extraction'])['sha256']:
        raise ValueError('formation feature lineage differs')
    if report.get('schema')!='vec.formation-probes.v1.1' or report.get('test_read') is not False or report.get('labels')!=list(LABELS) or extraction.get('labels')!=list(LABELS):
        raise ValueError('formation target definition or fit split differs')
    for decoder in ('ridge','mlp'):
        filename='ridge.npz' if decoder=='ridge' else 'mlp.pt'
        if report.get('artifacts',{}).get(filename)!=stable_digest(paths[decoder])['sha256']:raise ValueError('selected probe weights differ from fit report')
    for key,value in dict(updates=profile['probe_updates'],seed=profile['probe_seed'],lr=.001,weight_decay=.0001,hidden_widths=[128,128]).items():
        if report.get('mlp',{}).get(key)!=value:raise ValueError('probe fitting budget differs')
    with np.load(paths['features'],allow_pickle=False) as data:
        x=data['features'];y=data['labels'];splits=data['split'];strata=data['stratum'];systems=data['system_key'];episodes=data['episode_key']
    if any(a.shape!=(len(x),) for a in (splits,strata,systems,episodes)) or x.ndim!=2 or y.shape!=(len(x),len(LABELS)):
        raise ValueError('malformed formation feature bundle')
    if np.any(~np.isin(splits,['train','validation'])):raise PermissionError('test observations or labels cannot be used for fitting formation probes')
    if len(set(episodes.tolist()))!=len(episodes) or set(systems[splits=='train'])&set(systems[splits=='validation']):raise ValueError('formation training/validation identities overlap')
    train=splits=='train';select=(splits=='validation')&np.isin(strata,SELECTION_STRATA)
    if not train.any() or not select.any() or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('missing finite train/validation formation support')
    dim=128 if method=='neural' else 12
    if x.shape[1]!=dim or extraction.get('representation_dim')!=dim or extraction.get('rows')!=len(x):raise ValueError('registered representation shape differs')
    _,xm,xs,xa=train_standardize(x,train);_,ym,ys,ya=train_standardize(y,train)
    expected=dict(xmean=xm,xscale=xs,xactive=xa,ymean=ym,yscale=ys)
    with np.load(paths['ridge'],allow_pickle=False) as ridge:
        for name,values in expected.items():
            if ridge[name].shape!=values.shape or not np.array_equal(ridge[name],values):raise ValueError('probe normalization does not match training-only data: '+name)
        if ridge['weights'].shape!=(int(xa.sum()),len(LABELS)) or not np.isfinite(ridge['weights']).all():raise ValueError('invalid frozen ridge support')
    return dict(normalization=expected,rows=len(x),train_rows=int(train.sum()),validation_rows=int(select.sum()),dimension=dim,reused_original_artifacts=reused)


def check_numerical_runtime(profile,device):
    import torch
    runtime=profile.get('numerical_runtime')
    if runtime is None:return
    if torch.device(device).type!=runtime['device_type'] or torch.__version__!=runtime['torch_version']:
        raise ValueError('formation device or Torch runtime differs from frozen profile')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.mha.get_fastpath_enabled()!=runtime['mha_fastpath']:
        raise ValueError('formation numerical backend flags differ from frozen profile')


def planned_rows(bank,profile):
    if not isinstance(bank,SealedBank) or bank.closed:raise PermissionError('open authorized formation bank required')
    validate_profile(profile)
    if bank.resolution!=profile['resolution']:raise ValueError('formation bank resolution differs')
    if profile['stratum'] not in bank.plans:raise ValueError('formation population not frozen')
    plan=bank.plans[profile['stratum']]['plan'];slots={(r['system_key'],r['kind'],r['replicate']):r for r in bank.rows.values() if r['stratum']==profile['stratum']}
    rows=[]
    for case in plan['cases']:
        row=slots[(case['system_key'],'forced',case['episode_slots']['forced'][0])]
        if row['raw_frames']<profile['frames']:raise ValueError('planned formation history support unavailable; no replacement')
        rows.append((case,row))
    if len({r['episode_key'] for _,r in rows})!=len(rows):raise ValueError('formation cases reused a source episode')
    return rows


def extract_feature(inner,history,*,kind,context):
    """Only the public history reaches the method. Passed raw arrays are erased."""
    from persistbench.contracts import EpisodeContext
    from .adapters import z_experience
    inner.initialize(context);inner.reset(EpisodeContext('opaque_donor'))
    payload=z_experience(history);began=time.monotonic();inner.ingest(payload)
    payload.observations.fill(0);payload.actions.fill(0);inner.reset(EpisodeContext('opaque_readout'))
    if kind=='neural':
        code=inner.memory()[0].detach().cpu().numpy()[0].copy()
    elif kind=='explicit':
        theta=inner.parameter_samples();physical=np.c_[theta,theta[:,2]/theta[:,0]];logs=np.log(physical)
        code=np.r_[logs.mean(0),logs.std(0),physical.mean(0)]
    else:raise ValueError('unknown frozen feature definition')
    if code.shape!=((128,) if kind=='neural' else (12,)) or not np.isfinite(code).all():raise ValueError('invalid retained representation')
    return code,dict(seconds=time.monotonic()-began,memory_bytes=inner.mutable_state_bytes(),fit_failures=getattr(inner,'fit_failures',None))


def run(args):
    from persistbench.contracts import RunContext,ComputeTier,Split
    from .sealed_evaluate import load_method
    from .formation_readout import FrozenMLPReadout
    from .prediction_assay import history_case,artifact_digest
    from .adapters import _digest
    from .pixel_training import runtime_policy
    import torch
    authorization=SealedAuthorization(args.protocol,args.selection,protocol_sha256=args.protocol_sha256,selection_sha256=args.selection_sha256)
    profile=validate_profile(authorization.protocol['assays']['formation']['profiles'][args.profile])
    selected=authorization.selected_paths[args.slot];slot=next(s for s in authorization.protocol['method_slots'] if s['slot_id']==args.slot)
    paths={role:selected[name] for role,name in profile['readout_roles'].items()}
    reuse_paths=resolve_reuse_paths(profile,selected,slot['kind'])
    checked=validate_readout_bundle(paths,profile=profile,bank_manifest_sha256=authorization.protocol['training_bank']['manifest_sha256'],
        bank_content_sha256=authorization.protocol['training_bank']['content_sha256'],bank_snapshot_sha256=authorization.protocol['training_bank']['snapshot_sha256'],
        checkpoint_sha256=stable_digest(selected['predictor'])['sha256'] if slot['kind']=='learned' else None,
        explicit_configuration=json.loads(selected['configuration'].read_text()) if slot['kind']=='explicit' else None,
        reuse_paths=reuse_paths,checkpoint_path=selected.get('predictor'))
    runtime_policy(profile['feature_seed']);torch.set_num_threads(1)
    # The device field binds neural history encoding. Explicit posterior
    # features and both probe families use their fixed CPU implementations.
    if slot['kind']=='learned':check_numerical_runtime(profile,args.device)
    inner=load_method(authorization,args.slot,profile['resolution'],args.device).inner
    if profile['feature_condition']=='random_initialization':
        from .pixel_models import PixelDynamicsModel
        from .pixel_agent import PixelMemoryAgent
        runtime_policy(slot['seed']);model=PixelDynamicsModel(inner.model.cfg);model.to(args.device).eval().requires_grad_(False);inner=PixelMemoryAgent(model)
    model_before=artifact_digest(inner);mlp=FrozenMLPReadout(paths['mlp'])
    for name,values in checked['normalization'].items():
        if not np.array_equal(getattr(mlp,name),values):raise ValueError('MLP normalization differs from train-only features/labels')
    context=RunContext('visual_elastic_coupling/formation','1.1',Split.TEST,ComputeTier(profile['compute_tier']),profile['feature_seed'])
    if args.output.exists():raise ValueError('formation test attempt exists; do not overwrite')
    args.output.mkdir(parents=True);test_opened=False
    common=dict(admission_path=args.admission,admission_sha256=args.admission_sha256,resolution=profile['resolution'])
    try:
        with SealedBank(args.bank,authorization,audit_path=args.output/'FEATURE_ACCESS.private.jsonl',allow_labels=False,**common) as bank:
            test_opened=True;planned=planned_rows(bank,profile);features=[];cases=[]
            for case,row in planned:
                history=history_case(bank,row,profile['frames']);fingerprint=digest(_digest(history))
                feature,diagnostic=extract_feature(inner,history,kind='neural' if slot['kind']=='learned' else 'explicit',context=context)
                features.append(feature);cases.append(dict(case_id=case['case_id'],block_id=case['block_id'],system_key=case['system_key'],replicate=case['replicate'],
                    episode_key=row['episode_key'],history_fingerprint=fingerprint,diagnostic=diagnostic))
            features=np.asarray(features);predictions=dict(ridge=ridge_predict(paths['ridge'],features),mlp=mlp.predict(features))
            if any(v.shape!=(len(cases),len(LABELS)) or not np.isfinite(v).all() for v in predictions.values()):raise ValueError('incomplete finite frozen formation predictions')
            if model_before!=artifact_digest(inner):raise ValueError('formation changed the frozen backbone')
        # Commit every prediction before granting access to test scoring labels.
        prediction_path=args.output/'PREDICTIONS_BEFORE_LABELS.npz';np.savez_compressed(prediction_path,features=features,**predictions)
        prediction_sha=stable_digest(prediction_path)['sha256'];predictions_before={k:_digest(v) for k,v in predictions.items()}
        labels=[]
        with SealedBank(args.bank,authorization,audit_path=args.output/'SCORE_ACCESS.private.jsonl',allow_labels=True,**common) as bank:
            scoring=planned_rows(bank,profile)
            if [(c['case_id'],r['episode_key']) for c,r in scoring]!=[(c['case_id'],c['episode_key']) for c in cases]:raise ValueError('formation scoring changed planned cases')
            for case,row in scoring:labels.append(private_labels(bank,row,profile['frames']))
        labels=np.asarray(labels)
        if labels.shape!=(len(cases),len(LABELS)) or not np.isfinite(labels).all():raise ValueError('invalid formation scoring targets')
        if stable_digest(prediction_path)['sha256']!=prediction_sha or predictions_before!={k:_digest(v) for k,v in predictions.items()} or model_before!=artifact_digest(inner):raise ValueError('formation predictions/model changed after labels were opened')
        records=[];systems=np.asarray([c['system_key'] for c in cases]);scale=checked['normalization']['yscale']
        for i,case in enumerate(cases):
            for decoder,pred in predictions.items():records.append(dict(**case,decoder=decoder,target=labels[i].tolist(),prediction=pred[i].tolist(),
                target_fingerprint=digest(_digest(labels[i])),squared_error=((pred[i]-labels[i])**2).tolist(),train_standardized_squared_error=(((pred[i]-labels[i])/scale)**2).tolist()))
        report=dict(schema='vec.formal-formation-result.v1',profile=profile,labels=LABELS,method_slot=args.slot,
            condition=profile['feature_condition'],cases=len(cases),records=records,
            descriptive_scores={k:grouped_scores(v,labels,systems,scale) for k,v in predictions.items()},
            interpretation='frozen linear/nonlinear decoder-specific decodability; failure of a decoder is not proof of missing information',
            statistical_status='complete paired cases retained; independent-block/fixed-model-seed inferential analysis remains separate',
            protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256,bank_admission_sha256=args.admission_sha256,
            predictions_committed_before_labels_sha256=prediction_sha,test_read=True,formal_results=True)
        (args.output/'RESULT.private.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'PUBLIC_RESULT.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2)+'\n')
        completion=dict(schema='vec.formal-formation-completion.v1',status='PASS',test_read=True,
            result_sha256=stable_digest(args.output/'RESULT.private.json')['sha256'],predictions_before_labels_sha256=prediction_sha,
            cases=len(cases),source_fingerprint=source_fingerprint(),protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256)
    except Exception as exc:
        (args.output/'COMPLETION.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc),test_read=test_opened,formal_results=False),indent=2)+'\n');raise
    (args.output/'COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n');return completion


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','selection','bank','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','selection-sha256','admission-sha256','slot','profile'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();print(json.dumps(run(a)),flush=True)


if __name__=='__main__':main()
