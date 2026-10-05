"""Explicit asset binding. No discovery of test assets, no training on bind."""
from pathlib import Path
import sys
import numpy as np
from .io import read,write,sha,digest,development_path,plain


def default_protocol(sources):
    return dict(schema='sprii-next.development.v1',version='2026-09-19.1',test_read=False,
        allowed_splits=['train','validation'],source_seeds=[0,1,2],reader_seeds=[0,1,2],
        physics_coordinates='train_standardized_log_m_log_gamma_log_k',ridge_alpha=1.,
        decoder_selection='fixed ridge, no downstream selection',
        reader=dict(steps=10000,batch_size=256,learning_rate=3e-4,weight_decay=.05,warmup_steps=500,gradient_clip=1.),
        springworld=dict(primary='cold q0 h16, 600 cases, 100 systems (64 continuous + 36 heldout factorial)',
            secondary='complete original mixture and all five horizons',donor='same frozen P64, identical rows across arms'),
        pokeworld=dict(stage='pilot_first',windows_per_system=16,query_budget=1,horizons=[1,2,4,8,16],primary_horizon=16,
            sensitivity_fraction=.01,dominance='train z(S_mass) - train z(S_drag)',
            utility='(Null-Matched)_G1 - (Null-Matched)_G2',query_policy='own frozen source observation encoder',
            gate='at least 2/3 same-direction slopes plus reviewed stable, nonflat binned curve; record reversed direction; no p-value gate'),
        statistics=dict(unit='physical_system',bootstrap_seed=20260919,bootstrap_draws=10000,
            uncertainty='conditional on the fitted source/reader grid; seed-level cells retained'),sources=sources)


def native_paths(root):
    root=development_path(root).resolve()
    paths=[root,root/'src',root/'a_src',root/'extension']
    return [str(p) for p in paths if p.is_dir()]


def activate_native(root):
    paths=native_paths(root)
    if 'persistbench' in sys.modules:raise RuntimeError('bind/export native Spring assets in a fresh process')
    sys.path[:0]=paths
    return paths


def verify_spring_source(completion,method,seed):
    import torch
    from dataclasses import fields
    from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec,_new_model
    from persistbench.envs.visual_elastic_coupling.a_head_features import model_state_sha256
    from persistbench.envs.visual_elastic_coupling.a_pairing import configuration
    cp=development_path(completion).resolve();r=read(cp)
    if method not in ('Structure','Align','Cross','Both') or seed not in (0,1,2):raise ValueError('registered source grid required')
    native_name='Split' if method=='Structure' else method
    if r['status']!='TRAINING_COMPLETE' or r['name']!=native_name or r['selected_step']!=10000 or r['optimizer_updates']!=10000:
        raise ValueError('unqualified final-only source')
    if r['test_read'] is not False or r['validation_used'] is not False or r['selection_rule']!='final_step_only':
        raise ValueError('source selection changed')
    if r['configuration']!=configuration(native_name):raise ValueError('source mechanism mislabeled')
    spec_values={f.name:r['spec'][f.name] for f in fields(PretrainingSpec)}
    spec_values['milestone_steps']=tuple(spec_values['milestone_steps']);spec=PretrainingSpec(**spec_values)
    if (spec.model_seed,spec.sampling_seed,spec.stochastic_seed)!=(seed,seed,seed):raise ValueError('source seeds differ')
    path=development_path(cp.parent/r['selected_checkpoint'])
    if sha(path)!=r['files'][path.name]['sha256']:raise ValueError('source checkpoint changed')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['step']!=10000 or ck['binding']!=r['binding']:raise ValueError('checkpoint lineage changed')
    with torch.random.fork_rng(devices=[]):model=_new_model(native_name,spec,torch.device('cpu'))
    if model_state_sha256(model)!=r['initial_model_sha256']:raise ValueError('source architecture/initialization changed')
    model.load_state_dict(ck['model'],strict=True)
    if model_state_sha256(model)!=r['model_state_sha256']:raise ValueError('source tensor content differs')
    model.eval().requires_grad_(False)
    return model,r,path


def bind_spring(native_root,completion,manifest,features,targets,method,seed,output):
    paths=activate_native(native_root)
    model,r,checkpoint=verify_spring_source(completion,method,seed)
    for p in (manifest,features,targets):development_path(p)
    if any(e['split'] not in ('train','validation') for e in read(manifest)['episodes']):raise PermissionError('development bank required')
    fr=read(features)
    if fr['model_state_sha256']!=r['model_state_sha256'] or fr['bank_snapshot_sha256']!=r['binding']['bank_snapshot_sha256']:
        raise ValueError('source/cache/bank lineage differs')
    native_files={str(p.resolve()):sha(p) for p in sorted(Path(native_root).rglob('*.py'))}
    descriptor=dict(environment='springworld',method=method,source_seed=seed,native_paths=paths,native_files=native_files,
        model_state_sha256=r['model_state_sha256'],source_completion=str(Path(completion).resolve()),source_completion_sha256=sha(completion))
    for key,value in dict(manifest=manifest,features_receipt=features,targets_receipt=targets,checkpoint=checkpoint).items():
        descriptor[key]=str(Path(value).resolve());descriptor[key+'_sha256']=sha(value)
    from .providers import SpringCache
    provider=SpringCache(descriptor)
    write(output,descriptor)
    return dict(status='BOUND',source=method,seed=seed,identity=provider.identity,test_read=False)


def export_spring(native_root,completion,bank,method,seed,output,device='cuda:0'):
    paths=activate_native(native_root)
    model,r,checkpoint=verify_spring_source(completion,method,seed)
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan,AHeadDataAccess
    from persistbench.envs.visual_elastic_coupling.a_head_features import extract_features,AHeadFeatureCache
    from persistbench.envs.visual_elastic_coupling.a_head_targets import extract_targets
    bank=development_path(bank);manifest=bank/'MANIFEST.private.json'
    plan=AHeadCasePlan(read(manifest),seed=0,history_frames=96)
    expected=r['binding']['bank_snapshot_sha256']
    if sha(bank/'BANK_SNAPSHOT.json')!=expected:raise ValueError('bank differs from trained source')
    access=AHeadDataAccess(bank,plan,snapshot_sha256=expected)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model.to(device)
    extract_features(access,model,out/'features',expected_model_state_sha256=r['model_state_sha256'],workers=8)
    fr=out/'features/FEATURES.json'
    features=AHeadFeatureCache(fr.parent,plan,receipt_sha256=sha(fr),model_state_sha256=r['model_state_sha256'],bank_snapshot_sha256=expected)
    extract_targets(access,features,out/'targets',workers=8)
    # Bind in the same verified native process without changing import provenance.
    descriptor=dict(environment='springworld',method=method,source_seed=seed,native_paths=paths,
        native_files={str(p.resolve()):sha(p) for p in sorted(Path(native_root).rglob('*.py'))},
        model_state_sha256=r['model_state_sha256'],source_completion=str(Path(completion).resolve()),source_completion_sha256=sha(completion))
    for key,value in dict(manifest=manifest,features_receipt=fr,targets_receipt=out/'targets/SUPERVISION.json',checkpoint=checkpoint).items():
        descriptor[key]=str(Path(value).resolve());descriptor[key+'_sha256']=sha(value)
    from .providers import SpringCache
    SpringCache(descriptor)
    write(out/'SOURCE.json',descriptor)
    return descriptor


def preflight(cfg,environment,stage=None):
    from .providers import provider
    from .engine import jobs
    required=[('Both',s) for s in range(3)] if environment=='springworld' else [(m,s) for m in ('G1','G2') for s in range(3)]
    if stage=='baseline':required=[('RelInfoNCE',s) for s in range(3)]
    results=[];reference=None;norm=None
    for method,seed in required:
        p=provider(cfg,environment,method,seed)
        populations={}
        for split in ('train','validation'):
            code,theta,ids,donors=p.donors(split)
            populations[split]=dict(systems=len(set(ids)),histories=len(code),persistent_dim=code.shape[1])
        # Compare actual evaluation identities and target values, not just config labels.
        import hashlib
        h=hashlib.sha256()
        for b in p.evaluation_batches():
            h.update(digest([{k:r[k] for k in ('system_id','query_id','donor_id','horizon')} for r in b.rows]).encode())
            h.update(b.target.tobytes());h.update(b.theta.tobytes())
        identity=h.hexdigest()
        if reference is None:reference=identity;norm=digest(p.normalization)
        if reference!=identity or norm!=digest(p.normalization):raise ValueError('cross-source population/targets/normalization changed')
        results.append(dict(method=method,seed=seed,populations=populations,provider_sha256=p.identity))
    return dict(status='PASS',test_read=False,protocol_sha256=digest(cfg),sources=results,
        jobs=len(jobs(environment,stage or ('development' if environment=='springworld' else 'pilot'))),
        evaluation_pairing_sha256=reference,code_only_smoke=False)
