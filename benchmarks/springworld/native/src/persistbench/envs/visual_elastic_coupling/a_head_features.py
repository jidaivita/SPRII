"""Frozen A features shared by independently fitted heads, without labels.

Each query is encoded for q0/1 once; each forced donor once. Short failed prefixes
retain their slots and support masks. Cache files are private evaluator artifacts,
but the assembled head inputs contain only public image/action features.
This module does not train, fit normalization, or authorize sealed-test access.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
import torch
from .a_fresh_head import AFreshFeatures,INPUT_KEYS
from .a_head_data import _immutable
from .a_head_statistics import HORIZONS
from .dataset_snapshot import checked_asset,stable_digest

SCHEMA='vec.A-head-features.v1'
ARRAY_NAMES=('query_embedding','future_actions','query_support','donor_slot','donor_support')


def model_state_sha256(model):
    """Canonical tensor content, independent of pickle/checkpoint serialization."""
    value=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        header=json.dumps([key,array.dtype.str,list(array.shape)],separators=(',',':')).encode()
        value.update(len(header).to_bytes(8,'big'));value.update(header)
        value.update(array.tobytes(order='C'))
    return value.hexdigest()


def _donors(plan):
    return sorted(key for pools in plan.pools.values() for key in pools['forced'])


def _support(plan):
    return np.asarray([[plan.rows[b['query_episode']]['raw_frames']>b['anchor']+h for h in HORIZONS]
                       for b in plan.base],dtype=bool)


def _identity(plan,model_sha):
    return dict(plan_sha256=plan.plan_sha256,manifest_semantic_sha256=plan.manifest_semantic_sha256,
                model_state_sha256=model_sha,history_frames=plan.frames)


def extract_features(reader,model,output,*,expected_model_state_sha256,workers=16,progress=None):
    """Create a new full train/validation public-feature cache, never overwrite.

    Hash verification reads source bytes but never parses physical label arrays.
    The final receipt is written only after extraction and full source recheck.
    Failed support is represented explicitly; unexpected model/file errors abort.
    """
    from .adapters import z_query,z_experience
    from .schema import query_packet,history_payload
    from .training_protocol import source_fingerprint
    plan=reader.plan;plan._guard()
    if model_state_sha256(model)!=expected_model_state_sha256:raise ValueError('frozen A model content differs from commitment')
    if model.cfg.history_length!=plan.frames:raise ValueError('A model and donor plan frame budgets differ')
    if getattr(model,'normalization_profile',None)!='train_history_running_statistics_pre_step_v1':
        raise ValueError('A head cache requires the registered strict observation normalization')
    output=Path(output)
    if output.exists():raise FileExistsError('A feature output already exists; preserve previous artifact')
    before=reader.verify_all(workers=workers);features=AFreshFeatures(model);features.initialize(None)
    source=source_fingerprint();donors=_donors(plan);n=len(plan.base);d=len(donors)
    arrays=dict(query_embedding=np.zeros((n,2,128),np.float32),future_actions=np.zeros((n,16,2),np.float32),
                query_support=_support(plan),donor_slot=np.zeros((d,128),np.float32),donor_support=np.zeros(d,bool))
    audit_start=len(reader.audit);output.mkdir(parents=True,exist_ok=False)
    for i,base in enumerate(plan.base):
        supported=np.flatnonzero(arrays['query_support'][i])
        if len(supported):
            episode=reader.public_episode(base['query_episode'],purpose='query')
            h=HORIZONS[int(supported[-1])]
            for q in (0,1):
                inputs=features.head_inputs(z_query(query_packet(episode,base['anchor'],q,h)))
                arrays['query_embedding'][i,q]=inputs['query_embedding'].cpu().numpy()[0]
                if q==0:arrays['future_actions'][i]=inputs['future_actions'].cpu().numpy()[0]
            del episode
        if progress is not None and ((i+1)%256==0 or i+1==n):progress(dict(phase='query',completed=i+1,total=n))
    for i,key in enumerate(donors):
        if plan.rows[key]['raw_frames']>=plan.frames:
            features._guard();features.initialize(None)
            episode=reader.public_episode(key,purpose='donor')
            features.ingest(z_experience(history_payload(episode,0,plan.frames-1)))
            arrays['donor_slot'][i,:features.representation_dim]=features.history_code().cpu().numpy()[0]
            arrays['donor_support'][i]=True;del episode
        if progress is not None and ((i+1)%128==0 or i+1==d):progress(dict(phase='donor',completed=i+1,total=d))
    features._guard();plan._guard()
    if model_state_sha256(model)!=expected_model_state_sha256 or source_fingerprint()!=source:
        raise ValueError('A encoder or extraction source changed during caching')
    if any(event['kind']!='public_observation' for event in reader.audit[audit_start:]):
        raise ValueError('nonpublic source parsed during A feature extraction')
    files={}
    for name,array in arrays.items():
        if not np.isfinite(array).all():raise ValueError('nonfinite frozen A feature; no complete cache')
        path=output/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        files[name]=dict(path=path.name,shape=list(array.shape),dtype=array.dtype.str,**stable_digest(path))
    after=reader.verify_all(workers=workers);features._guard()
    if model_state_sha256(model)!=expected_model_state_sha256 or source_fingerprint()!=source:
        raise ValueError('A encoder or extraction source changed before completion')
    record=dict(schema=SCHEMA,status='COMPLETE',**_identity(plan,expected_model_state_sha256),
        bank_snapshot_sha256=reader.snapshot_sha256,bank_content_sha256=reader.snapshot['content_sha256'],
        extractor_source_fingerprint=source,variant=model.variant,normalization_profile=model.normalization_profile,
        native_context_dim=features.representation_dim,query_episodes=[b['query_episode'] for b in plan.base],donor_episodes=donors,
        query_budgets=[0,1],horizons=list(HORIZONS),files=files,
        query_supported_by_horizon=arrays['query_support'].sum(0).tolist(),donors_supported=int(arrays['donor_support'].sum()),
        public_reads=len(reader.audit)-audit_start,parsed_label_arrays=0,bank_before=before,bank_after=after,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device=str(next(model.parameters()).device),dtype='float32'),
        input_profile='image features and actions; query protocol labels are not additional head inputs',
        failure_policy='all query/donor slots retained; missing support raises per requested arm/horizon',
        formal_checkpoint_provenance_verified=False,formal_results=False,test_read=False)
    with (output/'FEATURES.json').open('x') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return dict(status='COMPLETE',receipt_sha256=stable_digest(output/'FEATURES.json')['sha256'],
                query_count=n,donor_count=d,model_state_sha256=expected_model_state_sha256,
                formal_results=False,test_read=False)


class AHeadFeatureCache:
    """Content-checked, read-only in-memory features; labels are absent entirely.

    Verify every file when opening and at fitting/scoring completion. Batches use
    detached copies, so head optimizers cannot alter shared cached features.
    """
    def __init__(self,root,plan,*,receipt_sha256,model_state_sha256,bank_snapshot_sha256):
        self.root=Path(root);self.plan=plan;self.receipt_sha256=receipt_sha256;self._receipt_commitment=receipt_sha256;plan._guard()
        receipt=checked_asset(root,'FEATURES.json')
        if stable_digest(receipt)['sha256']!=receipt_sha256:raise ValueError('A feature receipt differs from commitment')
        record=json.loads(receipt.read_text())
        if record.get('schema')!=SCHEMA or record.get('status')!='COMPLETE' or record.get('test_read') or record.get('parsed_label_arrays')!=0:
            raise ValueError('unqualified A public-feature cache')
        for key,value in _identity(plan,model_state_sha256).items():
            if record.get(key)!=value:raise ValueError('A feature model or case plan binding differs')
        if record.get('bank_snapshot_sha256')!=bank_snapshot_sha256:raise ValueError('A feature bank binding differs')
        if record.get('query_budgets')!=[0,1] or record.get('horizons')!=list(HORIZONS):raise ValueError('A feature budget profile differs')
        donors=_donors(plan)
        if record.get('query_episodes')!=[b['query_episode'] for b in plan.base] or record.get('donor_episodes')!=donors:
            raise ValueError('A feature population order differs')
        native=128 if record.get('variant')=='B0' else 64
        if record.get('variant') not in ('B0','B0_split','B2','Bx','B3') or record.get('native_context_dim')!=native:
            raise ValueError('A native context width differs')
        if record.get('normalization_profile')!='train_history_running_statistics_pre_step_v1':raise ValueError('A normalization profile differs')
        if set(record['files'])!=set(ARRAY_NAMES):raise ValueError('unexpected A feature array or private labels')
        self._record=_immutable(record);self._record_identity=id(self._record);self.verify_all()
        n=len(plan.base);d=len(donors)
        shapes=dict(query_embedding=(n,2,128),future_actions=(n,16,2),query_support=(n,5),donor_slot=(d,128),donor_support=(d,))
        arrays={}
        for name in ARRAY_NAMES:
            entry=record['files'][name]
            with checked_asset(root,entry['path']).open('rb') as stream:value=np.load(stream,allow_pickle=False)
            expected_dtype=np.dtype(bool if name.endswith('_support') else np.float32)
            if value.shape!=shapes[name] or value.dtype!=expected_dtype or not np.isfinite(value).all():raise ValueError('invalid A feature array')
            if entry['shape']!=list(value.shape) or entry['dtype']!=value.dtype.str:raise ValueError('A feature metadata differs from array')
            value.setflags(write=False);arrays[name]=value
        expected_donor=np.asarray([plan.rows[k]['raw_frames']>=plan.frames for k in donors],bool)
        if not np.array_equal(arrays['query_support'],_support(plan)) or not np.array_equal(arrays['donor_support'],expected_donor):
            raise ValueError('A feature support mask changed the declared population')
        if np.any(arrays['donor_slot'][~expected_donor]!=0) or np.any(arrays['donor_slot'][:,native:]!=0):raise ValueError('A donor placeholders/padding differ')
        if np.any(arrays['query_embedding'][~arrays['query_support'].any(1)]!=0):raise ValueError('unsupported A query placeholder differs')
        lengths=np.array([max((h for h,ok in zip(HORIZONS,row) if ok),default=0) for row in arrays['query_support']])
        if np.any(arrays['future_actions'][np.arange(16)[None]>=lengths[:,None]]!=0) or np.any(np.linalg.norm(arrays['future_actions'],axis=-1)>1+1e-7):
            raise ValueError('A cached future action support/padding differs')
        self._arrays=MappingProxyType(arrays);self._donor_index=MappingProxyType({key:i for i,key in enumerate(donors)})
        self._donor_identity=id(self._donor_index)
        self._array_identity=tuple((name,id(value)) for name,value in arrays.items())
        self._array_content={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in arrays.items()}
        self.verify_all()

    def _guard(self):
        self.plan._guard()
        if self.receipt_sha256!=self._receipt_commitment or self.plan.plan_sha256!=self._record['plan_sha256'] or id(self._record)!=self._record_identity or id(self._donor_index)!=self._donor_identity or tuple((n,id(v)) for n,v in self._arrays.items())!=self._array_identity or any(v.flags.writeable for v in self._arrays.values()):
            raise ValueError('shared A feature cache changed')

    def verify_all(self):
        if self.receipt_sha256!=self._receipt_commitment or self.plan.plan_sha256!=self._record['plan_sha256']:raise ValueError('A feature receipt/plan commitment changed')
        if stable_digest(checked_asset(self.root,'FEATURES.json'))['sha256']!=self.receipt_sha256:
            raise ValueError('A feature receipt changed')
        if id(self._record)!=self._record_identity:raise ValueError('A feature metadata changed')
        for entry in self._record['files'].values():
            if stable_digest(checked_asset(self.root,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:
                raise ValueError('A cached feature file changed')
        if hasattr(self,'_arrays'):
            self._guard()
            if any(hashlib.sha256(v.tobytes()).hexdigest()!=self._array_content[n] for n,v in self._arrays.items()):
                raise ValueError('in-memory A feature content changed')
        return dict(status='PASS',receipt_sha256=self.receipt_sha256,files=len(self._record['files']))

    def inputs(self,cases,*,arm,device='cpu'):
        if arm not in ('null','matched','wrong'):raise ValueError('A public cache arm must be null/matched/wrong; Oracle is separate')
        self._guard()
        if not cases:raise ValueError('empty A cached head batch')
        for case in cases:self.plan.validate_case(case)
        indices=np.array([c['index'] for c in cases]);qs=np.array([c['q'] for c in cases]);hi=np.array([HORIZONS.index(c['horizon']) for c in cases])
        if not self._arrays['query_support'][indices,hi].all():raise ValueError('planned A query/future support missing; no replacement')
        slot=np.zeros((len(cases),128),np.float32)
        if arm!='null':
            donor=np.array([self._donor_index[c[arm+'_episode']] for c in cases])
            if not self._arrays['donor_support'][donor].all():raise ValueError('planned A donor support missing; no replacement')
            slot[:]=self._arrays['donor_slot'][donor]
        mask=(np.arange(16)[None]<np.asarray(HORIZONS)[hi,None]).astype(np.float32)
        result=dict(query_embedding=torch.tensor(self._arrays['query_embedding'][indices,qs],device=device),
            context_slot=torch.tensor(slot,device=device),future_actions=torch.tensor(self._arrays['future_actions'][indices]*mask[:,:,None],device=device),
            action_mask=torch.tensor(mask,device=device),horizon_index=torch.tensor(hi,device=device,dtype=torch.long))
        if set(result)!=INPUT_KEYS:raise AssertionError('private metadata escaped A feature projection')
        self._guard();return result
