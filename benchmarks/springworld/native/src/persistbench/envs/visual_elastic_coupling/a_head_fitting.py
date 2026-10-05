"""Independent A head optimization with paired sampling and final-step selection.

This is the reusable train/validation engine, not formal experiment admission.
It does not load an encoder, open raw trajectories, or authorize sealed data.
The fixed final checkpoint rule follows the existing A follow-up; development
validation curves never change updates, stopping time, or checkpoint selection.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass,fields
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .a_fresh_head import AFreshHead,AFreshStateAgent,HEAD_PROFILE,_stamp
from .a_head_data import _immutable
from .a_head_features import AHeadFeatureCache,model_state_sha256
from .a_head_targets import AHeadTargets
from .a_head_statistics import HORIZONS
from .a_pairing import digest
from .dataset_snapshot import checked_asset,stable_digest
from .training_protocol import source_fingerprint

ARMS=('null','matched','wrong','oracle')
SCHEMA='vec.A-head-fit.v1'


@dataclass(frozen=True)
class HeadFitSpec:
    steps:int=10000
    batch_size:int=256
    learning_rate:float=3e-4
    weight_decay:float=.05
    warmup_steps:int=500
    gradient_clip:float=1.
    head_seed:int=0
    sampling_seed:int=0
    validation_every:int=10000
    validation_draws:tuple=(0,)
    validation_batch_size:int=256
    save_every:int=2500
    log_every:int=100

    def validate(self):
        for name in ('steps','batch_size','validation_every','validation_batch_size','save_every','log_every'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('positive integer A head budget required: '+name)
        if type(self.warmup_steps) is not int or not 0<=self.warmup_steps<self.steps:raise ValueError('warmup must precede final optimizer update')
        for name in ('head_seed','sampling_seed'):
            if type(getattr(self,name)) is not int or not 0<=getattr(self,name)<2**32:raise ValueError('uint32 paired A head seed required')
        if not isinstance(self.validation_draws,tuple) or not self.validation_draws or len(set(self.validation_draws))!=len(self.validation_draws) or any(type(d) is not int or not 0<=d<2**32 for d in self.validation_draws):
            raise ValueError('nonempty unique registered validation donor draws required')
        for name in ('learning_rate','weight_decay','gradient_clip'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (name!='weight_decay' and value==0):raise ValueError('invalid A head optimization coefficient')

    def record(self):
        self.validate();value=asdict(self);value['validation_draws']=list(self.validation_draws)
        value.update(optimizer='AdamW',betas=[.9,.999],epsilon=1e-8,foreach=False,fused=False,
            schedule='linear_warmup_cosine_per_update_v1',checkpoint_selection='final_step_only',
            loss='mean train-standardized squared error over 8 coordinates',precision='float32_no_tf32',deterministic_algorithms=True,cudnn_benchmark=False,cudnn_deterministic=True,
            head_profile=HEAD_PROFILE,query_budgets=[0,1],horizons=list(HORIZONS))
        return value


def learning_rate(spec,completed_updates):
    spec.validate()
    if type(completed_updates) is not int or not 0<=completed_updates<spec.steps:raise ValueError('optimizer update index outside budget')
    if completed_updates<spec.warmup_steps:return spec.learning_rate*(completed_updates+1)/spec.warmup_steps
    progress=(completed_updates-spec.warmup_steps)/(spec.steps-spec.warmup_steps)
    return spec.learning_rate*.5*(1+math.cos(math.pi*progress))


@contextmanager
def _math_profile(device):
    if device.type=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('deterministic A CUDA fitting requires CUBLAS_WORKSPACE_CONFIG before starting the process')
    before=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:yield
    finally:
        torch.use_deterministic_algorithms(before[0],warn_only=before[1]);torch.backends.cuda.matmul.allow_tf32=before[2];torch.backends.cudnn.allow_tf32=before[3]
        torch.backends.cudnn.benchmark=before[4];torch.backends.cudnn.deterministic=before[5]


def _new_head(seed,device):
    # Initialize on CPU with a local RNG state; do not reseed other CUDA devices.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        head=AFreshHead()
    return head.to(device=device,dtype=torch.float32)


def _binding(features,train,validation):
    if not isinstance(features,AHeadFeatureCache) or not isinstance(train,AHeadTargets) or not isinstance(validation,AHeadTargets):
        raise ValueError('bound A public features and separate target readers required')
    if (train.split,train.purpose)!=('train','fit') or (validation.split,validation.purpose)!=('validation','score'):
        raise PermissionError('A optimizer and validation require distinct fixed permission scopes')
    features._guard();train._guard();validation._guard()
    if len({x.plan.plan_sha256 for x in (features,train,validation)})!=1 or train.receipt_sha256!=validation.receipt_sha256:
        raise ValueError('A head feature/target populations differ')
    if features._record['bank_snapshot_sha256']!=train._record['bank_snapshot_sha256'] or train.statistics is None or validation.statistics is None or train.statistics.sha256!=validation.statistics.sha256:
        raise ValueError('A head bank or normalization binding differs')
    return dict(feature_receipt_sha256=features.receipt_sha256,supervision_receipt_sha256=train.receipt_sha256,
        plan_sha256=features.plan.plan_sha256,bank_snapshot_sha256=features._record['bank_snapshot_sha256'],
        encoder_state_sha256=features._record['model_state_sha256'],encoder_variant=features._record['variant'],
        history_frames=features.plan.frames,statistics_sha256=train.statistics.sha256,head_profile=HEAD_PROFILE)


def _inputs(features,labels,cases,arm,device):
    inputs=features.inputs(cases,arm='null' if arm=='oracle' else arm,device=device)
    if arm=='oracle':inputs['context_slot']=torch.tensor(labels.oracle_slots(cases),device=device,dtype=torch.float32)
    return inputs


def _missing(features,labels,case,arm):
    hi=HORIZONS.index(case['horizon'])
    if not features._arrays['query_support'][case['index'],hi] or not labels._arrays['support'][labels._row_index[case['index']],hi]:return 'query_or_target_support'
    if arm in ('matched','wrong') and not features._arrays['donor_support'][features._donor_index[case[arm+'_episode']]]:return 'donor_support'
    return None


def score_head(head,features,validation,*,arm,draws,batch_size):
    """Full validation denominators; diagnostic strata never alter selection loss."""
    if arm not in ARMS or validation.split!='validation' or validation.purpose!='score':raise PermissionError('explicit A validation scoring scope required')
    if type(batch_size) is not int or batch_size<1 or not isinstance(draws,tuple) or not draws or len(set(draws))!=len(draws):raise ValueError('invalid A validation batching/draws')
    features._guard();validation._guard()
    if features.plan.plan_sha256!=validation.plan.plan_sha256:raise ValueError('A validation source plan mismatch')
    device=next(head.parameters()).device;was_training=head.training;head.eval();groups={};pending=[];missing_cases=[];case_digest=hashlib.sha256()
    total=dict(weight=0.,observed_weight=0.,selection_weight=0.,observed_selection_weight=0.,loss_sum=0.,selection_sum=0.,cases=0,missing=0,selection_missing=0)
    def flush():
        if not pending:return
        cases=[x[0] for x in pending];inputs=_inputs(features,validation,cases,arm,device)
        target=torch.tensor(validation.targets(cases),device=device,dtype=torch.float32)
        with torch.no_grad():loss=(head(inputs)-target).square().mean(1).cpu().double().numpy()
        if not np.isfinite(loss).all():raise ValueError('nonfinite A validation prediction/loss')
        for (_,key,weight,selected),value in zip(pending,loss):
            group=groups[key];group['observed_weight']+=weight;group['loss_sum']+=weight*float(value)
            total['observed_weight']+=weight;total['observed_selection_weight']+=selected
            total['loss_sum']+=weight*float(value);total['selection_sum']+=selected*float(value)
        pending.clear()
    try:
        for index,base in enumerate(features.plan.base):
            if base['split']!='validation':continue
            for draw in draws:
                for q in (0,1):
                    for h in HORIZONS:
                        case=features.plan.case(index,q=q,horizon=h,draw=draw);case_digest.update(case['case_sha256'].encode())
                        key=(base['stratum'],base['kind'],q,h);weight=case['split_weight']/len(draws);selected=case['selection_weight']/len(draws)
                        group=groups.setdefault(key,dict(weight=0.,observed_weight=0.,loss_sum=0.,cases=0,missing=0))
                        group['weight']+=weight;group['cases']+=1;total['weight']+=weight;total['selection_weight']+=selected;total['cases']+=1
                        reason=_missing(features,validation,case,arm)
                        if reason:
                            group['missing']+=1;total['missing']+=1
                            if selected>0:total['selection_missing']+=1
                            missing_cases.append(dict(case_id=case['case_id'],index=index,q=q,horizon=h,draw=draw,reason=reason,weight=weight,selection_weight=selected))
                        else:pending.append((case,key,weight,selected))
                        if len(pending)>=batch_size:flush()
        flush()
        if not np.isclose(total['weight'],1.) or not np.isclose(total['selection_weight'],1.):raise ValueError('A validation population weights incomplete')
        rows=[]
        for key,group in sorted(groups.items()):
            rows.append(dict(stratum=key[0],query_kind=key[1],q=key[2],horizon=key[3],**group,
                loss=None if group['missing'] else group['loss_sum']/group['weight']))
        selection_complete=total['selection_missing']==0
        features._guard();validation._guard()
        return dict(status='COMPLETE' if not total['missing'] else 'MISSING_SUPPORT_RETAINED',cases=total['cases'],missing=total['missing'],
            expected_weight=total['weight'],observed_weight=total['observed_weight'],
            selection_expected_weight=total['selection_weight'],selection_observed_weight=total['observed_selection_weight'],selection_missing=total['selection_missing'],missing_cases=missing_cases,
            validation_loss=None if total['missing'] else total['loss_sum']/total['weight'],
            selection_loss=total['selection_sum']/total['selection_weight'] if selection_complete else None,
            groups=rows,case_sequence_sha256=case_digest.hexdigest(),donor_draws=list(draws),
            checkpoint_selection='final_step_only; validation scores cannot change selection',formal_results=False,test_read=False)
    finally:head.train(was_training)


def _write_json(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return dict(path=path.name,**stable_digest(path))


def fit_head(features,train,validation,output,*,arm,spec,device='cpu',progress=None):
    """Train one independent arm for its complete fixed budget and save final head.

    Caller must register the common spec before an experiment and obtain formal
    admission separately. A completed optimizer run is not benchmark acceptance.
    """
    if arm not in ARMS or not isinstance(spec,HeadFitSpec):raise ValueError('registered A head arm/spec required')
    config=spec.record();binding=_binding(features,train,validation);device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('A fitting supports declared CPU/CUDA float32 profiles')
    output=Path(output)
    if output.exists():raise FileExistsError('A head run exists; preserve previous attempt')
    # No rejection-resampling when a training donor is unavailable. Null/Oracle
    # have no donor-support requirement and remain independent valid fits.
    if arm in ('matched','wrong'):
        for key,index in features._donor_index.items():
            if features.plan.rows[key]['split']=='train' and not features._arrays['donor_support'][index]:raise ValueError('complete training donor support required for this A arm')
    before=dict(features=features.verify_all(),train=train.verify_all(),validation=validation.verify_all())
    source=source_fingerprint();started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    run=dict(schema=SCHEMA,arm=arm,config=config,binding=binding,source_fingerprint=source,
        selection_rule='final_step_only',formal_admission_verified=False,test_read=False)
    run_file=_write_json(output/'RUN.json',run);updates=0;files={};window_loss=0.;window_count=0;case_digest=hashlib.sha256();validation_records=[]
    try:
        with _math_profile(device):
            head=_new_head(spec.head_seed,device);initial_sha=model_state_sha256(head)
            optimizer=torch.optim.AdamW(head.parameters(),lr=spec.learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=spec.weight_decay,foreach=False,fused=False)
            head.train()
            with (output/'training.jsonl').open('x') as log:
                for completed in range(spec.steps):
                    cases=features.plan.draw_training_batch(sampling_seed=spec.sampling_seed,step=completed,batch_size=spec.batch_size)
                    for case in cases:case_digest.update(case['case_sha256'].encode())
                    inputs=_inputs(features,train,cases,arm,device);target=torch.tensor(train.targets(cases),device=device,dtype=torch.float32)
                    rate=learning_rate(spec,completed)
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.zero_grad(set_to_none=True);prediction=head(inputs);loss=(prediction-target).square().mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite A training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),spec.gradient_clip,error_if_nonfinite=True);optimizer.step();updates=completed+1
                    if not torch.stack([torch.isfinite(p).all() for p in head.parameters()]).all():raise ValueError('nonfinite A head parameter after update')
                    window_loss+=float(loss.detach());window_count+=1
                    event=dict(step=updates,learning_rate=rate,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started)
                    if updates%spec.validation_every==0 or updates==spec.steps:
                        scored=score_head(head,features,validation,arm=arm,draws=spec.validation_draws,batch_size=spec.validation_batch_size)
                        name=f'validation_{updates:07d}.json';files[name]=_write_json(output/name,scored)
                        validation_records.append(dict(step=updates,path=name,selection_loss=scored['selection_loss'],validation_loss=scored['validation_loss']))
                        event['validation']=validation_records[-1]
                    if updates%spec.save_every==0 or updates==spec.steps:
                        name=f'checkpoint_{updates:07d}.pt';path=output/name
                        checkpoint=dict(schema=SCHEMA,completed_updates=updates,run_sha256=run_file['sha256'],binding=binding,
                            head_state_dict={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer_state_dict=optimizer.state_dict(),
                            head_state_sha256=model_state_sha256(head),training_case_sequence_sha256=case_digest.hexdigest(),initial_head_sha256=initial_sha)
                        with path.open('xb') as stream:torch.save(checkpoint,stream)
                        files[name]=dict(path=name,**stable_digest(path))
                    if updates%spec.log_every==0 or 'validation' in event or updates==spec.steps:
                        event.update(window_mean_loss=window_loss/window_count,window_updates=window_count)
                        log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();window_loss=0.;window_count=0
                        if progress is not None:progress(event)
            head.eval();after=dict(features=features.verify_all(),train=train.verify_all(),validation=validation.verify_all())
            if _binding(features,train,validation)!=binding or source_fingerprint()!=source:raise ValueError('A fitting source/bindings changed')
            if stable_digest(output/'RUN.json')!={k:run_file[k] for k in ('bytes','sha256')}:raise ValueError('A run registration changed')
            for entry in files.values():
                if stable_digest(checked_asset(output,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('A checkpoint/validation artifact changed during fitting')
            final_name=f'checkpoint_{spec.steps:07d}.pt'
            result=dict(schema=SCHEMA,status='TRAINING_COMPLETE',arm=arm,run=run_file,binding=binding,config=config,
                optimizer_updates=updates,selected_checkpoint=final_name,selected_step=spec.steps,selection_rule='final_step_only',
                final_head_state_sha256=model_state_sha256(head),initial_head_sha256=initial_sha,head_architecture=head.architecture(),
                training_case_sequence_sha256=case_digest.hexdigest(),validation_records=validation_records,files=files,
                training_log=dict(path='training.jsonl',**stable_digest(output/'training.jsonl')),before=before,after=after,
                seconds=time.monotonic()-started,device=str(device),torch_version=torch.__version__,numpy_version=np.__version__,
                formal_admission_verified=False,formal_results=False,test_read=False)
            _write_json(output/'COMPLETE.json',result)
            return dict(status='TRAINING_COMPLETE',receipt_sha256=stable_digest(output/'COMPLETE.json')['sha256'],
                selected_step=spec.steps,selected_checkpoint=final_name,head_state_sha256=result['final_head_state_sha256'],formal_results=False)
    except Exception as error:
        _write_json(output/'FAILURE.json',dict(status='FAILED_RETAINED',completed_updates=updates,error_type=type(error).__name__,error=str(error),formal_results=False,test_read=False))
        raise


def load_fitted_head(root,features,train,validation,*,receipt_sha256,device='cpu'):
    """Reload final head only after actual feature/target/checkpoint binding checks."""
    root=Path(root);binding=_binding(features,train,validation);path=checked_asset(root,'COMPLETE.json')
    if stable_digest(path)['sha256']!=receipt_sha256:raise ValueError('A fitted-head receipt differs')
    record=json.loads(path.read_text());run_path=checked_asset(root,record['run']['path'])
    if stable_digest(run_path)!={k:record['run'][k] for k in ('bytes','sha256')}:raise ValueError('A fitted-head run registration changed')
    run=json.loads(run_path.read_text())
    if record.get('schema')!=SCHEMA or record.get('status')!='TRAINING_COMPLETE' or record.get('test_read') or record.get('binding')!=binding or run.get('binding')!=binding:
        raise ValueError('A fitted-head population/model binding differs')
    if run.get('source_fingerprint')!=source_fingerprint() or record.get('config')!=run.get('config') or record.get('arm')!=run.get('arm'):
        raise ValueError('A fitted-head source/configuration differs')
    spec_values={f.name:record['config'][f.name] for f in fields(HeadFitSpec)};spec_values['validation_draws']=tuple(spec_values['validation_draws'])
    spec=HeadFitSpec(**spec_values)
    if spec.record()!=record['config'] or record['arm'] not in ARMS:raise ValueError('A fitted-head registered settings differ')
    steps=record['config']['steps'];name=f'checkpoint_{steps:07d}.pt'
    if record.get('selection_rule')!='final_step_only' or record.get('selected_step')!=steps or record.get('optimizer_updates')!=steps or record.get('selected_checkpoint')!=name:
        raise ValueError('A fitted-head final-step selection differs')
    expected_checkpoints={f'checkpoint_{s:07d}.pt' for s in range(1,steps+1) if s%spec.save_every==0 or s==steps}
    expected_validations={f'validation_{s:07d}.json' for s in range(1,steps+1) if s%spec.validation_every==0 or s==steps}
    if set(record['files'])!=expected_checkpoints|expected_validations:raise ValueError('A fitted-head scheduled artifacts missing')
    for filename,asset in record['files'].items():
        if asset['path']!=filename or stable_digest(checked_asset(root,filename))!={k:asset[k] for k in ('bytes','sha256')}:
            raise ValueError('A fitted-head artifact changed')
    if record['training_log']['path']!='training.jsonl' or stable_digest(checked_asset(root,'training.jsonl'))!={k:record['training_log'][k] for k in ('bytes','sha256')}:
        raise ValueError('A fitted-head training log changed')
    checkpoint_path=checked_asset(root,name);entry=record['files'][name]
    if stable_digest(checkpoint_path)!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('A fitted head checkpoint changed')
    with checkpoint_path.open('rb') as stream:checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=SCHEMA or checkpoint.get('completed_updates')!=steps or checkpoint.get('binding')!=binding or checkpoint.get('run_sha256')!=record['run']['sha256']:
        raise ValueError('A checkpoint content does not match the registered fit')
    head=_new_head(record['config']['head_seed'],torch.device(device))
    if model_state_sha256(head)!=record['initial_head_sha256'] or checkpoint.get('initial_head_sha256')!=record['initial_head_sha256'] or checkpoint.get('training_case_sequence_sha256')!=record['training_case_sequence_sha256']:
        raise ValueError('A fitted-head paired initialization or sampling differs')
    head.load_state_dict(checkpoint['head_state_dict'],strict=True)
    if model_state_sha256(head)!=record['final_head_state_sha256'] or checkpoint['head_state_sha256']!=record['final_head_state_sha256']:
        raise ValueError('A reloaded head tensors differ from the completed fit')
    if stable_digest(checkpoint_path)!={k:entry[k] for k in ('bytes','sha256')} or stable_digest(path)['sha256']!=receipt_sha256:
        raise ValueError('A fitted-head artifact changed during loading')
    features.verify_all();train.verify_all();validation.verify_all();head.eval();head.requires_grad_(False)
    return head,_immutable(record)


class _BoundFittedAgent(AFreshStateAgent):
    def __init__(self,model,head,statistics,*,arm,receipt_sha256):
        super().__init__(model,head,statistics,context_policy='null_only' if arm=='null' else 'history_or_zero')
        self._fit_model_stamp=_stamp(model);self._fit_head_stamp=_stamp(head)
        self._fit_descriptor=self.features._descriptor();self._fit_statistics_sha=self.statistics.sha256
        self._fit_policy=(self.context_policy,self.features.max_histories,self.features.aggregation,self.features.no_history_policy)
        self._fit_receipt=receipt_sha256

    def _fit_guard(self):
        if (_stamp(self.model)!=self._fit_model_stamp or _stamp(self.head)!=self._fit_head_stamp or self.features._descriptor()!=self._fit_descriptor or
            (self.context_policy,self.features.max_histories,self.features.aggregation,self.features.no_history_policy)!=self._fit_policy):
            raise ValueError('fitted A backbone/head no longer matches the loaded artifact')
        self.statistics._check()
        if self.statistics.record['statistics_sha256']!=self._fit_statistics_sha:raise ValueError('fitted A statistics changed')

    def initialize(self,context):self._fit_guard();return super().initialize(context)
    def _guard(self):self._fit_guard();super()._guard()

    def respond(self,query):
        from persistbench.contracts import AgentOutput
        result=super().respond(query)
        return AgentOutput(result.output_type,result.values,dict(result.diagnostics,
            fitted_head_provenance_verified=True,fitted_run_receipt_sha256=self._fit_receipt,formal_head_provenance_verified=False))


def load_fitted_state_agent(model,root,features,train,validation,*,receipt_sha256):
    """Bind the actual backbone to its head; Oracle retains its privileged route."""
    binding=_binding(features,train,validation)
    if model_state_sha256(model)!=binding['encoder_state_sha256'] or model.variant!=binding['encoder_variant'] or model.cfg.history_length!=binding['history_frames']:
        raise ValueError('actual A backbone does not match the fitted head source')
    device=next(model.parameters()).device
    head,record=load_fitted_head(root,features,train,validation,receipt_sha256=receipt_sha256,device=device)
    if record['arm']=='oracle':raise PermissionError('Oracle head must use the separate evaluator parameter route')
    return _BoundFittedAgent(model,head,train.statistics,arm=record['arm'],receipt_sha256=receipt_sha256)
