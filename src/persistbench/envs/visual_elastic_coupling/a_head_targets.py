"""Separate A training/validation supervision after public features are committed.

Training moments are finalized before validation labels are parsed. Incomplete
training support never yields successful-case-only normalization. Cache readers
have a fixed split/purpose and cannot load validation arrays for fitting.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .a_head_data import _immutable
from .a_head_statistics import FitAHeadStatistics,AHeadStatistics,HORIZONS
from .dataset_snapshot import checked_asset,stable_digest

SCHEMA='vec.A-head-targets.v1'


def _indices(plan,split):return [i for i,b in enumerate(plan.base) if b['split']==split]
def _support(plan,indices):
    return np.asarray([[plan.rows[plan.base[i]['query_episode']]['raw_frames']>plan.base[i]['anchor']+h for h in HORIZONS] for i in indices],bool)


def _fit_complete_train(plan,indices,targets):
    """Recompute exact declared training statistics, refusing subsets before values."""
    plan._guard()
    if list(indices)!=_indices(plan,'train') or not _support(plan,indices).all():
        raise PermissionError('complete training population required before normalization')
    values=np.asarray(targets,np.float64)
    if values.shape!=(len(indices),len(HORIZONS),8) or not np.isfinite(values).all():raise ValueError('invalid complete A training targets')
    fit=FitAHeadStatistics()
    for j,i in enumerate(indices):
        for hi,h in enumerate(HORIZONS):fit.add_target(split='train',case_key=plan.base[i]['query_episode'],horizon=h,delta=values[j,hi])
    for key,entry in sorted(plan.systems.items()):
        if entry['split']=='train':fit.add_system(split='train',system_key=key,theta=entry['theta'])
    return fit.export()


def _save_array(root,name,value):
    path=root/(name+'.npy')
    with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    return dict(path=path.name,shape=list(value.shape),dtype=value.dtype.str,**stable_digest(path))


def _save_json(root,name,value):
    path=root/name
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return dict(path=path.name,**stable_digest(path))


def extract_targets(reader,features,output,*,workers=16,progress=None):
    """Materialize both splits without fitting any model or opening test data."""
    from .a_head_features import AHeadFeatureCache
    from .training_protocol import source_fingerprint
    if not isinstance(features,AHeadFeatureCache):raise ValueError('committed A public feature cache required first')
    plan=reader.plan;plan._guard();features._guard()
    if features.plan.plan_sha256!=plan.plan_sha256 or features._record['bank_snapshot_sha256']!=reader.snapshot_sha256:
        raise ValueError('public features and target source do not share a committed bank/plan')
    output=Path(output)
    if output.exists():raise FileExistsError('A target output already exists; preserve previous artifact')
    feature_before=features.verify_all();bank_before=reader.verify_all(workers=workers);source=source_fingerprint()
    start=len(reader.audit);output.mkdir(parents=True,exist_ok=False);files={};splits={};statistics=None
    normalization_status=None;train_complete=None;phase=None
    for split in ('train','validation'):
        indices=_indices(plan,split);targets=np.zeros((len(indices),len(HORIZONS),8),np.float64);support=_support(plan,indices)
        purpose='fit' if split=='train' else 'score';phase_start=len(reader.audit)
        for j,i in enumerate(indices):
            row,actual=reader.target_row(i,purpose=purpose)
            if not np.array_equal(actual,support[j]):raise ValueError('target support changed during materialization')
            targets[j]=row
            if progress is not None and ((j+1)%256==0 or j+1==len(indices)):progress(dict(phase=split,completed=j+1,total=len(indices)))
        expected_reads=int(support.any(1).sum())
        events=reader.audit[phase_start:]
        if len(events)!=expected_reads or any(e['kind']!='private_label' or e['split']!=split or e['purpose']!=purpose for e in events):
            raise ValueError('label read order/split differs from training then validation phases')
        files[split+'_delta']=_save_array(output,split+'_delta',targets)
        files[split+'_support']=_save_array(output,split+'_support',support)
        splits[split]=dict(indices=indices,queries=len(indices),support_counts=support.sum(0).tolist(),complete=bool(support.all()))
        if split=='train':
            train_complete=bool(support.all())
            if train_complete:
                statistics=_fit_complete_train(plan,indices,targets)
                files['statistics']=_save_json(output,'TRAIN_STATISTICS.json',statistics)
                normalization_status='TRAIN_ONLY_COMPLETE'
            else:normalization_status='UNAVAILABLE_INCOMPLETE_TRAIN_SUPPORT'
            phase=dict(status='COMPLETE',plan_sha256=plan.plan_sha256,training_targets_complete=train_complete,
                normalization_status=normalization_status,train_files={k:v for k,v in files.items()},
                statistics_sha256=None if statistics is None else statistics['statistics_sha256'],validation_labels_parsed=0)
            files['training_phase']=_save_json(output,'TRAINING_PHASE.json',phase)
    # Recheck the pre-validation commitment, including statistics, after scoring
    # labels were parsed; the validation phase must never update training moments.
    for entry in phase['train_files'].values():
        if stable_digest(checked_asset(output,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:
            raise ValueError('training targets/statistics changed after validation reads')
    feature_after=features.verify_all();bank_after=reader.verify_all(workers=workers);plan._guard()
    if source_fingerprint()!=source:raise ValueError('A target extraction source changed')
    for entry in files.values():
        if stable_digest(checked_asset(output,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:
            raise ValueError('A supervision output changed before completion')
    record=dict(schema=SCHEMA,status='COMPLETE',plan_sha256=plan.plan_sha256,manifest_semantic_sha256=plan.manifest_semantic_sha256,
        bank_snapshot_sha256=reader.snapshot_sha256,bank_content_sha256=reader.snapshot['content_sha256'],extractor_source_fingerprint=source,
        preceding_feature_receipt_sha256=features.receipt_sha256,
        preceding_feature_model_state_sha256=features._record['model_state_sha256'],
        feature_binding_scope='preceding public extraction; labels are common to any encoder using this exact bank and plan',
        splits=splits,horizons=list(HORIZONS),target_definition='physical_state_delta_8d',units=['m']*4+['m/s']*4,files=files,
        training_targets_complete=train_complete,normalization_status=normalization_status,
        statistics_sha256=None if statistics is None else statistics['statistics_sha256'],
        normalization_population='unique training query episode/horizon, irrespective of q or donor draws; Oracle once per physical training system',
        training_phase_committed_before_validation=True,parsed_label_files=len(reader.audit)-start,
        feature_before=feature_before,feature_after=feature_after,bank_before=bank_before,bank_after=bank_after,
        formal_training=False,formal_results=False,test_read=False)
    _save_json(output,'SUPERVISION.json',record)
    return dict(status='COMPLETE',receipt_sha256=stable_digest(output/'SUPERVISION.json')['sha256'],
        training_targets_complete=train_complete,normalization_status=normalization_status,
        statistics_sha256=record['statistics_sha256'],formal_results=False,test_read=False)


class AHeadTargets:
    """Fixed split/purpose label access; fit rejects validation before any read."""
    def __init__(self,root,plan,*,receipt_sha256,bank_snapshot_sha256,split,purpose):
        if split not in ('train','validation') or purpose not in ('fit','score') or (purpose=='fit' and split!='train'):
            raise PermissionError('A target reader permits training fit or explicit split scoring only')
        self.root=Path(root);self.plan=plan;self.receipt_sha256=receipt_sha256;self.split=split;self.purpose=purpose;plan._guard()
        path=checked_asset(root,'SUPERVISION.json')
        if stable_digest(path)['sha256']!=receipt_sha256:raise ValueError('A supervision receipt differs from commitment')
        record=json.loads(path.read_text())
        if record.get('schema')!=SCHEMA or record.get('status')!='COMPLETE' or record.get('test_read') or not record.get('training_phase_committed_before_validation'):
            raise ValueError('unqualified A target artifact')
        for key,value in (('plan_sha256',plan.plan_sha256),('manifest_semantic_sha256',plan.manifest_semantic_sha256),('bank_snapshot_sha256',bank_snapshot_sha256)):
            if record.get(key)!=value:raise ValueError('A supervision bank or plan differs')
        if record.get('horizons')!=list(HORIZONS) or record.get('target_definition')!='physical_state_delta_8d' or record.get('units')!=['m']*4+['m/s']*4:
            raise ValueError('A physical target definition differs')
        train_support=_support(plan,_indices(plan,'train'));complete=bool(train_support.all())
        status='TRAIN_ONLY_COMPLETE' if complete else 'UNAVAILABLE_INCOMPLETE_TRAIN_SUPPORT'
        if record.get('training_targets_complete')!=complete or record.get('normalization_status')!=status:
            raise ValueError('A training support or normalization admission differs')
        required={'train_delta','train_support','validation_delta','validation_support','training_phase'}|({'statistics'} if complete else set())
        if set(record.get('files',{}))!=required:raise ValueError('A target file manifest differs')
        for key,entry in record['files'].items():
            canonical={'training_phase':'TRAINING_PHASE.json','statistics':'TRAIN_STATISTICS.json'}.get(key,key+'.npy')
            if entry.get('path')!=canonical:raise ValueError('A target file path does not match its permission role')
        if purpose=='fit' and not complete:raise PermissionError('incomplete training support cannot produce a normalized A head fit')
        for source_split in ('train','validation'):
            indices=_indices(plan,source_split);support=_support(plan,indices)
            expected=dict(indices=indices,queries=len(indices),support_counts=support.sum(0).tolist(),complete=bool(support.all()))
            if record['splits'].get(source_split)!=expected:raise ValueError('A target cache shrank or reordered a declared split')
        self._record=_immutable(record);self._record_identity=id(self._record)
        self._allowed=(split+'_delta',split+'_support','training_phase')+(('statistics',) if complete else ())
        self._settings=(self.split,self.purpose,self.receipt_sha256,self._allowed)
        self.verify_all()
        phase=json.loads(checked_asset(root,record['files']['training_phase']['path']).read_text())
        expected_train_files={k:record['files'][k] for k in ('train_delta','train_support')+(('statistics',) if complete else ())}
        if phase!=dict(status='COMPLETE',plan_sha256=plan.plan_sha256,training_targets_complete=complete,
                       normalization_status=status,train_files=expected_train_files,statistics_sha256=record['statistics_sha256'],validation_labels_parsed=0):
            raise ValueError('training phase commitment differs from supervision')
        indices=_indices(plan,split);self._row_index=_immutable({value:i for i,value in enumerate(indices)})
        arrays={}
        for suffix,shape,dtype in (('delta',(len(indices),5,8),np.dtype(np.float64)),('support',(len(indices),5),np.dtype(bool))):
            entry=record['files'][split+'_'+suffix]
            with checked_asset(root,entry['path']).open('rb') as stream:value=np.load(stream,allow_pickle=False)
            if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all() or entry['shape']!=list(shape) or entry['dtype']!=dtype.str:
                raise ValueError('invalid A target array')
            value.setflags(write=False);arrays[suffix]=value
        if not np.array_equal(arrays['support'],_support(plan,indices)) or np.any(arrays['delta'][~arrays['support']]!=0):
            raise ValueError('A target support or missing placeholder differs')
        self.statistics=None
        if complete:
            self.statistics=AHeadStatistics(json.loads(checked_asset(root,record['files']['statistics']['path']).read_text()))
            if self.statistics.sha256!=record['statistics_sha256']:raise ValueError('A target statistics binding differs')
            counts=[self.statistics.record['target'][str(h)]['count'] for h in HORIZONS]
            if counts!=[len(_indices(plan,'train'))]*5 or self.statistics.record['oracle']['count']!=sum(s['split']=='train' for s in plan.systems.values()):
                raise ValueError('A normalization counts differ from full training population')
            if split=='train' and _fit_complete_train(plan,indices,arrays['delta'])!=self.statistics.record:
                raise ValueError('A normalization does not match actual complete training targets')
        elif record['statistics_sha256'] is not None:raise ValueError('partial training cannot carry fitted statistics')
        self._arrays=_immutable(arrays);self._arrays_identity=tuple((k,id(v)) for k,v in arrays.items())
        self._array_content={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        self._statistics_identity=id(self.statistics);self._row_identity=id(self._row_index);self.verify_all()

    def _guard(self):
        self.plan._guard()
        if (self.plan.plan_sha256!=self._record['plan_sha256'] or (self.split,self.purpose,self.receipt_sha256,self._allowed)!=self._settings or id(self._record)!=self._record_identity or
            id(self._row_index)!=self._row_identity or id(self.statistics)!=self._statistics_identity or
            tuple((k,id(v)) for k,v in self._arrays.items())!=self._arrays_identity or any(v.flags.writeable for v in self._arrays.values())):
            raise ValueError('A target access scope or cached content changed')
        if self.statistics is not None:
            self.statistics._check()
            if self.statistics.record['statistics_sha256']!=self._record['statistics_sha256']:raise ValueError('A train statistics changed')

    def verify_all(self):
        if self.plan.plan_sha256!=self._record['plan_sha256'] or (self.split,self.purpose,self.receipt_sha256,self._allowed)!=self._settings or id(self._record)!=self._record_identity:
            raise ValueError('A target permission scope changed')
        if stable_digest(checked_asset(self.root,'SUPERVISION.json'))['sha256']!=self.receipt_sha256:raise ValueError('A supervision receipt changed')
        for name in self._allowed:
            entry=self._record['files'][name]
            if stable_digest(checked_asset(self.root,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:
                raise ValueError('A target or training statistics file changed')
        if hasattr(self,'_arrays'):
            self._guard()
            if any(hashlib.sha256(v.tobytes()).hexdigest()!=self._array_content[k] for k,v in self._arrays.items()):raise ValueError('in-memory A targets changed')
        return dict(status='PASS',split=self.split,purpose=self.purpose,files=len(self._allowed),receipt_sha256=self.receipt_sha256)

    def _cases(self,cases):
        self._guard()
        if not cases:raise ValueError('empty A supervised batch')
        if any(c['split']!=self.split for c in cases):raise PermissionError('A target case is outside the fixed reader split')
        for case in cases:self.plan.validate_case(case)
        return np.array([self._row_index[c['index']] for c in cases]),np.array([HORIZONS.index(c['horizon']) for c in cases])

    def targets(self,cases,*,standardized=True):
        if type(standardized) is not bool:raise ValueError('explicit physical or standardized target requested')
        rows,hi=self._cases(cases)
        if not self._arrays['support'][rows,hi].all():raise ValueError('planned A target missing; no replacement')
        values=self._arrays['delta'][rows,hi].copy()
        if not standardized:return values
        if self.statistics is None:raise ValueError('no complete training normalization for A standardized targets')
        result=np.empty(values.shape,np.float32)
        for i,h in enumerate(HORIZONS):
            selected=hi==i
            if selected.any():result[selected]=self.statistics.standardize(values[selected],h)
        self._guard();return result

    def oracle_slots(self,cases):
        self._cases(cases)
        if self.statistics is None:raise ValueError('no complete training Oracle normalization')
        theta=np.asarray([self.plan.systems[c['system_key']]['theta'] for c in cases],np.float64)
        return self.statistics.oracle_slot(theta)
