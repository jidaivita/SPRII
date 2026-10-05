"""Audit original train/validation probes without rewriting or refitting them.

Legacy metadata omissions are recorded in a separate admission. A selected
admission is recomputed from original files, source, predictions and scores.
It establishes pretest reuse, not historical before/after verification.
"""
import ast
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

SCHEMA='vec.pretest-readout-reuse.v1'
ROLES=('admission','source_archive','source_completion','content_admission','predictions')
PACKAGE='src/persistbench/envs/visual_elastic_coupling/'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_recipe(path):
    with tarfile.open(path) as archive:
        files={}
        for member in archive.getmembers():
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not member.isfile() or member.name in files:
                raise ValueError('invalid readout source archive')
            files[member.name]=archive.extractfile(member).read()
    current=Path(__file__).parent
    old=ast.parse(files[PACKAGE+'formation.py']);new=ast.parse((current/'formation.py').read_text())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    previous,active=functions(old),functions(new)
    dump=lambda nodes:ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
    for name in ('private_labels','explicit_feature','train_standardize','grouped_scores'):
        if ast.dump(previous[name],include_attributes=False)!=ast.dump(active[name],include_attributes=False):
            raise ValueError('original formation algorithm differs: '+name)
    def fit_core(node):
        begin=next(i for i,n in enumerate(node.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.output.exists()')
        end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets))
        return dump(node.body[begin:end])
    if fit_core(previous['fit_probes'])!=fit_core(active['fit_probes']):raise ValueError('original probe fitting recipe differs')
    def encoding_core(node):
        branch=next(n for n in node.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.explicit')
        # Explicit posterior routine is checked separately. Only descriptive
        # metadata were added to its branch; the neural branch is unchanged.
        added_guard=ast.parse("if checkpoint['config'].get('bank_manifest_sha256')!=bank.manifest_sha256:\n    raise ValueError('formation training/validation bank differs from checkpoint training bank')").body[0]
        return dump([n for n in branch.orelse if ast.dump(n,include_attributes=False)!=ast.dump(added_guard,include_attributes=False)])
    if encoding_core(previous['extract'])!=encoding_core(active['extract']):raise ValueError('original representation encoding differs')
    for name in ('LABELS','SELECTION_STRATA'):
        value=lambda tree:next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        if ast.dump(value(old))!=ast.dump(value(new)):raise ValueError('original label/selection definition differs')
    for name in ('pixel_models.py','pixel_training.py'):
        if files.get(PACKAGE+name)!=(current/name).read_bytes():raise ValueError('original model/readout implementation differs: '+name)
    if PACKAGE+'formation_readout.py' in files and files[PACKAGE+'formation_readout.py']!=(current/'formation_readout.py').read_bytes():
        raise ValueError('original inference-only readout implementation differs')
    fingerprint=hashlib.sha256()
    for name,value in sorted(files.items()):
        relative=name.removeprefix(PACKAGE)
        if name.startswith(PACKAGE) and '/' not in relative and relative.endswith('.py'):
            fingerprint.update(relative.encode()+b'\0'+value)
    return dict(original_source_fingerprint=fingerprint.hexdigest(),formation_file_sha256=hashlib.sha256(files[PACKAGE+'formation.py']).hexdigest(),
                fitting_core_sha256=hashlib.sha256(fit_core(previous['fit_probes']).encode()).hexdigest())


def audit(paths,reuse,*,profile,bank_manifest_sha256,bank_content_sha256,bank_snapshot_sha256,
          checkpoint_sha256=None,checkpoint_path=None,explicit_configuration=None):
    from .formation import LABELS,FEATURE_DEFINITIONS,EXPLICIT_CONFIGURATION,train_standardize,grouped_scores
    from .formation_readout import FrozenMLPReadout,ridge_predict
    required=set(ROLES)-{'admission'}
    if checkpoint_sha256 is not None:required.add('source_checkpoint')
    if not required<=set(reuse):raise ValueError('readout reuse evidence incomplete')
    bound={**{key:Path(value) for key,value in paths.items()},**{key:Path(reuse[key]) for key in required}}
    before={key:sha(value) for key,value in bound.items()}
    source=source_recipe(bound['source_archive'])
    completion=json.loads(bound['source_completion'].read_text());content=json.loads(bound['content_admission'].read_text())
    if completion.get('status')!='EXECUTED' or completion.get('test_read') is not False or completion.get('source_sha256')!=before['source_archive']:
        raise ValueError('original readout execution/source did not complete')
    if not completion.get('steps') or any(row.get('returncode')!=0 for row in completion['steps']):raise ValueError('original readout execution contains failure')
    if content.get('status')!='PASS' or content.get('test_read') is not False or content.get('original_archive_files_verified',0)<1:
        raise ValueError('original bank content audit missing')
    if content.get('snapshot_sha256')!=bank_snapshot_sha256 or content.get('content_sha256')!=bank_content_sha256:
        raise ValueError('original readout data-content binding differs')
    extraction=json.loads(bound['extraction'].read_text());report=json.loads(bound['report'].read_text())
    method='neural' if checkpoint_sha256 is not None else 'explicit'
    for name in ('schema','method','frames','kind','bank_manifest_sha256','frozen','test_read'):
        expected=dict(schema='vec.formation-feature.v1.1',method=method,frames=profile['frames'],kind='forced',bank_manifest_sha256=bank_manifest_sha256,frozen=True,test_read=False)[name]
        if extraction.get(name)!=expected:raise ValueError('original readout extraction differs: '+name)
    if report.get('schema')!='vec.formation-probes.v1.1' or report.get('test_read') is not False or report.get('labels')!=list(LABELS) or extraction.get('labels')!=list(LABELS):
        raise ValueError('original readout targets or test access differ')
    for item in (extraction,report):
        if 'source_fingerprint' in item and item['source_fingerprint']!=source['original_source_fingerprint']:
            raise ValueError('original readout source fingerprint differs')
    if extraction.get('feature_sha256')!=before['features'] or report.get('features_sha256')!=before['features']:
        raise ValueError('original readout feature binding differs')
    if method=='neural':
        import torch
        if checkpoint_path is None or sha(checkpoint_path)!=checkpoint_sha256:raise ValueError('selected backbone evidence missing')
        original=torch.load(bound['source_checkpoint'],map_location='cpu',weights_only=True)
        selected=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if original['config'].get('bank_manifest_sha256')!=bank_manifest_sha256 or original['config'].get('model',{}).get('resolution')!=128:
            raise ValueError('original feature backbone bank/resolution differs')
        if extraction.get('checkpoint_sha256')!=before['source_checkpoint']:raise ValueError('original extraction backbone differs')
        for key in ('model','seed','family','bank_manifest_sha256'):
            if original['config'].get(key)!=selected['config'].get(key):raise ValueError('random initialization recipe differs from selected backbone')
        random=profile['feature_condition']=='random_initialization'
        if extraction.get('random_initialization_control')!=random or (not random and before['source_checkpoint']!=checkpoint_sha256):
            raise ValueError('readout trained/random backbone mismatch')
        if extraction.get('selected_update')!=(0 if random else original['update']):raise ValueError('original backbone update differs')
    elif profile['feature_condition']!='trained' or explicit_configuration!=EXPLICIT_CONFIGURATION:
        raise ValueError('explicit readout configuration differs')
    with np.load(bound['features'],allow_pickle=False) as data:
        x=data['features'];y=data['labels'];splits=data['split'];strata=data['stratum'];systems=data['system_key']
    if np.any(~np.isin(splits,['train','validation'])):raise PermissionError('test rows in original readout fit')
    train=splits=='train'
    _,xm,xs,xa=train_standardize(x,train);_,ym,ys,_=train_standardize(y,train)
    import torch
    with torch.random.fork_rng(devices=[]):readout=FrozenMLPReadout(bound['mlp'])
    for name,value in dict(xmean=xm,xscale=xs,xactive=xa,ymean=ym,yscale=ys).items():
        if not np.array_equal(getattr(readout,name),value):raise ValueError('MLP normalization does not use original train-only values')
    predictions=dict(ridge=ridge_predict(bound['ridge'],x),mlp=readout.predict(x));differences={}
    with np.load(bound['predictions'],allow_pickle=False) as saved:
        for decoder,prediction in predictions.items():
            if prediction.shape!=saved[decoder].shape:raise ValueError('original probe prediction support differs')
            # The existing declared readout tolerance is retained. This check
            # uses exactly the saved features, without changing device/batching
            # of the history encoder or requiring cross-backend equivalence.
            np.testing.assert_allclose(prediction,saved[decoder],rtol=2e-5,atol=2e-4)
            differences[decoder]=float(np.max(np.abs(prediction-saved[decoder])))
            for split in ('train','validation'):
                for stratum in np.unique(strata[splits==split]):
                    mask=(splits==split)&(strata==stratum)
                    score=grouped_scores(saved[decoder][mask],y[mask],systems[mask],ys)
                    previous=report['scores'][decoder][split+'/'+stratum]
                    for key in ('mse','train_standardized_mse','r2'):
                        np.testing.assert_allclose(np.asarray(score[key],float),np.asarray(previous[key],float),rtol=1e-10,atol=1e-12,equal_nan=True)
    if before!={key:sha(value) for key,value in bound.items()}:raise ValueError('original readout evidence changed during audit')
    record=dict(schema=SCHEMA,status='PASS',files=before,source_recipe=source,feature_condition=profile['feature_condition'],
                selected_checkpoint_sha256=checkpoint_sha256,bank_manifest_sha256=bank_manifest_sha256,
                bank_snapshot_sha256=bank_snapshot_sha256,bank_content_sha256=bank_content_sha256,
                prediction_max_abs_difference=differences,rows=len(x),test_read=False,optimization_performed=False,
                historical_before_after_verification_claimed=False)
    # Effective fields are derived only for the reader. Original files and the
    # admission keep their true historical contents and origins.
    extraction=dict(extraction);report=dict(report)
    for key,value in dict(resolution=128,feature_seed=990017,feature_definition=FEATURE_DEFINITIONS[method]).items():
        if key in extraction and extraction[key]!=value:raise ValueError('original optional extraction field differs')
        extraction[key]=value
    if method=='neural':extraction['checkpoint_sha256']=checkpoint_sha256
    else:extraction['explicit_configuration']=EXPLICIT_CONFIGURATION
    for key,value in dict(extraction_sha256=before['extraction'],artifacts={'ridge.npz':before['ridge'],'mlp.pt':before['mlp']}).items():
        if key in report and report[key]!=value:raise ValueError('original optional probe binding differs')
        report[key]=value
    return record,extraction,report


def validate(paths,reuse,**kwargs):
    expected,extraction,report=audit(paths,reuse,**kwargs)
    if 'admission' not in reuse or json.loads(Path(reuse['admission']).read_text())!=expected:
        raise ValueError('readout reuse admission differs from original evidence')
    return extraction,report
