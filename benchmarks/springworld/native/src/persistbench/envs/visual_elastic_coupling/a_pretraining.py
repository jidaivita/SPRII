"""Train-only nine-configuration A optimizer and bound final model artifacts.

This engine does not confer formal admission. It parses public training images
and actions only; physical files are hashed for provenance, never loaded as
supervision. Validation cannot select a checkpoint or update normalization.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass,fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from .a_pairing import CONFIGURATIONS,configuration,digest
from native_training import NativePairSchedule as APairSchedule
from native_training import make_batch,objective
from .a_head_fitting import _math_profile,_write_json
from .a_head_features import model_state_sha256
from .a_head_data import _immutable
from .dataset_snapshot import checked_asset,stable_digest,verify
from .training_protocol import source_fingerprint

SCHEMA='springworld.A-native128-pretraining.v1'


@dataclass(frozen=True)
class PretrainingSpec:
    steps:int=20000
    pairs_per_batch:int=48
    history_frames:int=96
    learning_rate:float=3e-4
    weight_decay:float=.05
    warmup_steps:int=500
    gradient_clip:float=1.
    model_seed:int=0
    sampling_seed:int=0
    stochastic_seed:int=0
    precision:str='bfloat16_cuda'
    save_every:int=1000
    log_every:int=100
    milestone_steps:tuple=(3000,)

    def record(self):
        for key in ('steps','pairs_per_batch','save_every','log_every'):
            value=getattr(self,key)
            if type(value) is not int or value<1:raise ValueError('positive integer A pretraining budget required: '+key)
        if self.pairs_per_batch<2:raise ValueError('A VICReg requires at least two pairs per batch')
        if type(self.history_frames) is not int or self.history_frames not in (24,48,96):raise ValueError('unregistered A history length')
        if type(self.warmup_steps) is not int or not 0<=self.warmup_steps<self.steps:raise ValueError('A warmup must precede the final step')
        for key in ('model_seed','sampling_seed','stochastic_seed'):
            value=getattr(self,key)
            if type(value) is not int or not 0<=value<2**32:raise ValueError('registered uint32 A pretraining seed required')
        for key in ('learning_rate','weight_decay','gradient_clip'):
            value=getattr(self,key)
            if type(value) not in (float,int) or not math.isfinite(value) or value<0 or (key!='weight_decay' and value==0):raise ValueError('invalid A pretraining optimization coefficient')
        if self.precision not in ('float32','bfloat16_cuda'):raise ValueError('unregistered A pretraining precision')
        if not isinstance(self.milestone_steps,tuple) or len(set(self.milestone_steps))!=len(self.milestone_steps) or any(type(s) is not int or s<1 for s in self.milestone_steps):
            raise ValueError('unique positive checkpoint milestones required')
        result=asdict(self);result['milestone_steps']=list(self.milestone_steps)
        result.update(optimizer='AdamW',betas=[.9,.999],epsilon=1e-8,foreach=False,fused=False,
            schedule='A_original_zero_first_update_linear_warmup_cosine_v1',checkpoint_selection='final_step_only',
            stochastic_policy='local per-update device RNG seeded independently of model initialization and pairing',
            normalization='train_history_running_statistics_pre_step_v1',horizons=[1,4,16],sigreg_directions=1024,
            gradient_accumulation=1,batch_windows=2*self.pairs_per_batch,deterministic_algorithms=True,tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
            strict_history_compatible=self.history_frames==24,validation_used=False)
        result.update(observation_profile='native128-history96',resolution=128,strict_history_compatible=False)
        return result


def pretraining_learning_rate(spec,completed_updates):
    """Match original LambdaLR: scheduler(0) is applied before first update."""
    spec.record()
    if type(completed_updates) is not int or not 0<=completed_updates<spec.steps:raise ValueError('A update outside registered budget')
    if spec.warmup_steps and completed_updates<=spec.warmup_steps:return spec.learning_rate*completed_updates/spec.warmup_steps
    return spec.learning_rate*.5*(1+math.cos(math.pi*(completed_updates-spec.warmup_steps)/(spec.steps-spec.warmup_steps)))


def a_source_fingerprint():
    import persistent_jepa,strict_model
    folder=Path(persistent_jepa.__file__).parent;paths=sorted(folder.glob('*.py'))
    rows=[dict(path='persistent_jepa/'+p.name,sha256=stable_digest(p)['sha256']) for p in paths]
    rows.append(dict(path='extension/strict_model.py',sha256=stable_digest(Path(strict_model.__file__))['sha256']))
    import native128_model,native_training
    for module in (native128_model,native_training):
        p=Path(module.__file__);rows.append(dict(path='native/'+p.name,sha256=stable_digest(p)['sha256']))
    return digest(rows)


@contextmanager
def _step_rng(seed,step,device):
    actual=int(np.random.SeedSequence([seed,step,719]).generate_state(1,dtype=np.uint64)[0])%(2**63-1)
    devices=[] if device.type=='cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        if device.type=='cpu':torch.set_rng_state(torch.Generator(device='cpu').manual_seed(actual).get_state())
        else:torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(actual).get_state(),device=device)
        yield actual


def _new_model(name,spec,device):
    from native_training import new_model
    return new_model(name,spec,device)


class PretrainingBank:
    def __init__(self,root,*,snapshot_sha256,spec):
        spec.record();self.root=Path(root);self.snapshot_sha256=snapshot_sha256
        path=checked_asset(self.root,'BANK_SNAPSHOT.json')
        if stable_digest(path)['sha256']!=snapshot_sha256:raise ValueError('A pretraining bank snapshot differs')
        self.snapshot=json.loads(path.read_text())
        manifest=json.loads(checked_asset(self.root,'MANIFEST.private.json').read_text())
        if any(r['split'] not in ('train','validation') for r in manifest['episodes']):raise PermissionError('test rows forbidden in A pretraining bank')
        self.schedule=APairSchedule(manifest,seed=spec.sampling_seed,history_frames=spec.history_frames,pairs_per_batch=spec.pairs_per_batch)
        self._inventory=self.schedule.inventory();self._snapshot_content=digest(self.snapshot)

    def verify_all(self,workers):
        if stable_digest(checked_asset(self.root,'BANK_SNAPSHOT.json'))['sha256']!=self.snapshot_sha256 or digest(self.snapshot)!=self._snapshot_content:
            raise ValueError('A pretraining snapshot changed')
        if self.schedule.inventory()!=self._inventory:raise ValueError('A pretraining schedule changed')
        if digest(json.loads(checked_asset(self.root,'MANIFEST.private.json').read_text()))!=self.schedule.manifest_sha256:
            raise ValueError('A pretraining manifest differs from schedule')
        return verify(self.root,self.snapshot,workers=workers)


def _saved_steps(spec):
    return {s for s in range(1,spec.steps+1) if s%spec.save_every==0 or s in spec.milestone_steps or s==spec.steps}


def _runtime_signature(device):
    result=dict(torch_version=str(torch.__version__),device_type=device.type,machine=platform.machine(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if device.type=='cuda':
        result.update(cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
            gpu_name=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return result


def _prefix_bytes(root,entry,filename):
    if entry['path']!=filename or type(entry['bytes']) is not int or entry['bytes']<1:
        raise ValueError('invalid A resume log prefix')
    with checked_asset(root,filename).open('rb') as stream:content=stream.read(entry['bytes'])
    if len(content)!=entry['bytes'] or hashlib.sha256(content).hexdigest()!=entry['sha256'] or not content.endswith(b'\n'):
        raise ValueError('A resume log prefix changed')
    return content


def _prepare_resume(path,sha256,run,spec):
    """Validate a committed prefix without reading episode arrays or updating a model.

    The launcher must first establish that the parent process is terminal. A new
    attempt keeps its parent intact and cannot extend the original step budget.
    """
    path=Path(path);root=path.parent;path=checked_asset(root,path.name)
    if stable_digest(path)['sha256']!=sha256:raise ValueError('A resume commit hash differs')
    if (root/'COMPLETE.json').exists():raise ValueError('completed A training cannot be resumed')
    commit=json.loads(path.read_text());step=commit.get('step')
    if (commit.get('schema')!='vec.A-pretraining-checkpoint.v1' or commit.get('status')!='CHECKPOINT_COMMITTED'
        or type(step) is not int or step not in _saved_steps(spec) or path.name!=f'checkpoint_{step:07d}.COMMIT.json'):
        raise ValueError('invalid A committed checkpoint')
    for candidate in root.glob('checkpoint_*.COMMIT.json'):
        try:other=json.loads(candidate.read_text())
        except json.JSONDecodeError:continue  # A torn receipt is not a commit.
        if other.get('status')=='CHECKPOINT_COMMITTED' and other.get('step',0)>step:
            raise ValueError('A resume must use latest committed checkpoint')
    runpath=checked_asset(root,'RUN.json');run_bytes=runpath.read_bytes()
    if (commit['run']['path']!='RUN.json' or stable_digest(runpath)!={k:commit['run'][k] for k in ('bytes','sha256')}
        or json.loads(run_bytes)!=run):raise ValueError('A resume settings, budget, bank, source or runtime differ')
    expected={f'checkpoint_{s:07d}.pt' for s in _saved_steps(spec) if s<=step}
    if set(commit['files'])!=expected:raise ValueError('A resume checkpoint prefix is incomplete')
    for name,entry in commit['files'].items():
        if entry['path']!=name or stable_digest(checked_asset(root,name))!={k:entry[k] for k in ('bytes','sha256')}:
            raise ValueError('A resume checkpoint changed')
    logs={key:_prefix_bytes(root,commit[key],filename) for key,filename in
        (('training_log','training.jsonl'),('batch_log','BATCHES.private.jsonl'))}
    rows=[json.loads(line) for line in logs['batch_log'].splitlines()]
    if [r['step'] for r in rows]!=list(range(1,step+1)):raise ValueError('A resume batch sequence is incomplete')
    case_chain=hashlib.sha256();recipient_chain=hashlib.sha256()
    exposure=dict(pairs=0,history_frames=0,target_frames=0,action_intervals=0)
    for row in rows:
        if (len(row['pair_sha256'])!=spec.pairs_per_batch or len(row['recipient_sha256'])!=spec.pairs_per_batch
            or row['physical_labels_read'] != (run['configuration']['name']=='Supervised-Split') or row['test_read']):raise ValueError('invalid A resume batch evidence')
        for value in row['pair_sha256']:case_chain.update(value.encode())
        for value in row['recipient_sha256']:recipient_chain.update(value.encode())
        exposure['pairs']+=spec.pairs_per_batch;exposure['history_frames']+=row['presented_history_frames']
        exposure['target_frames']+=row['target_frames'];exposure['action_intervals']+=row['observed_action_intervals']
    events=[json.loads(line) for line in logs['training_log'].splitlines()]
    if not events or events[-1]['step']!=step or events[-1].get('checkpoint')!=f'checkpoint_{step:07d}.pt':
        raise ValueError('A resume checkpoint lacks committed training event')
    checkpoint_path=checked_asset(root,f'checkpoint_{step:07d}.pt')
    with checkpoint_path.open('rb') as stream:checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
    if (checkpoint.get('schema')!=SCHEMA or checkpoint.get('step')!=step or checkpoint.get('binding')!=run['binding']
        or checkpoint.get('run_sha256')!=commit['run']['sha256'] or checkpoint.get('training_pair_sequence_sha256')!=case_chain.hexdigest()
        or checkpoint.get('recipient_sequence_sha256')!=recipient_chain.hexdigest() or commit['exposure']!=exposure):
        raise ValueError('A resume checkpoint sequence binding differs')
    provenance=dict(schema='vec.A-pretraining-resume.v1',parent_directory=str(root.resolve()),parent_commit=path.name,
        parent_commit_sha256=sha256,resumed_from_step=step,original_budget=spec.steps,new_optimizer_updates=spec.steps-step,
        policy='latest committed prefix, identical runtime and fixed budget; parent retained; no image or optimizer replay',test_read=False)
    return dict(root=root,commit=commit,checkpoint=checkpoint,logs=logs,run_bytes=run_bytes,
        case_chain=case_chain,recipient_chain=recipient_chain,exposure=exposure,provenance=provenance)


def train_pretraining(bank_root,output,*,name,spec,bank_snapshot_sha256,device='cuda',workers=16,progress=None,
                      resume_commit=None,resume_commit_sha256=None):
    """Run a fixed train-only budget. Formal Z/A protocol authorization is external."""
    from persistent_jepa.losses import SIGReg
    if not isinstance(spec,PretrainingSpec):raise ValueError('registered A pretraining specification required')
    config=spec.record();objective_config=configuration(name);device=torch.device(device);save_steps=_saved_steps(spec)
    if device.type not in ('cpu','cuda') or (spec.precision=='bfloat16_cuda' and device.type!='cuda'):raise ValueError('A pretraining device/precision mismatch')
    if device.type=='cuda' and spec.precision=='bfloat16_cuda' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 support required')
    if type(workers) is not int or workers<1:raise ValueError('positive bank verification workers required')
    if (resume_commit is None)!=(resume_commit_sha256 is None):raise ValueError('A resume requires commit path and hash together')
    output=Path(output)
    if output.exists():raise FileExistsError('A pretraining attempt exists; preserve it')
    bank=PretrainingBank(bank_root,snapshot_sha256=bank_snapshot_sha256,spec=spec);before=bank.verify_all(workers)
    inventory=bank.schedule.inventory();batches=inventory['batches_per_sweep']
    binding=dict(bank_snapshot_sha256=bank_snapshot_sha256,manifest_sha256=inventory['manifest_sha256'],
        selected_manifest_sha256=inventory['selected_manifest_sha256'],schedule_sha256=digest(inventory),
        source_fingerprint=source_fingerprint(),A_source_fingerprint=a_source_fingerprint())
    run=dict(schema=SCHEMA,configuration=objective_config,spec=config,binding=binding,schedule=inventory,runtime=_runtime_signature(device),
        full_sweeps=spec.steps//batches,final_sweep_batches=spec.steps%batches,
        stopping_rule='complete fixed updates; final partial sweep retained, no dropped or replacement batch',
        formal_admission_verified=False,formal_training=False,test_read=False)
    resume=None if resume_commit is None else _prepare_resume(resume_commit,resume_commit_sha256,run,spec)
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();updates=0;start_step=0;files={}
    case_chain=hashlib.sha256();recipient_chain=hashlib.sha256();resume_file=None
    if resume is None:run_file=_write_json(output/'RUN.json',run)
    else:
        (output/'RUN.json').write_bytes(resume['run_bytes']);run_file=dict(path='RUN.json',**stable_digest(output/'RUN.json'))
    try:
        with _math_profile(device):
            model=_new_model(name,spec,device);model.train();initial=model_state_sha256(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=spec.learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=spec.weight_decay,foreach=False,fused=False)
            sigreg=SIGReg().to(device);plan=None;window_losses=[];exposure=dict(pairs=0,history_frames=0,target_frames=0,action_intervals=0)
            if resume is not None:
                checkpoint=resume['checkpoint'];start_step=updates=checkpoint['step']
                if checkpoint['initial_model_sha256']!=initial:raise ValueError('A resume initialization differs')
                model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
                if model_state_sha256(model)!=checkpoint['model_state_sha256'] or int(model.observation.norm.num_batches_tracked)!=start_step:
                    raise ValueError('A resume model/statistic state differs')
                files=resume['commit']['files'];case_chain=resume['case_chain'];recipient_chain=resume['recipient_chain'];exposure=resume['exposure']
                for filename,entry in files.items():
                    shutil.copyfile(checked_asset(resume['root'],filename),output/filename)
                    if stable_digest(output/filename)!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('A resume checkpoint changed while copying')
                # Keep the parent's last commit so another immediate interruption is resumable.
                commit_name=Path(resume_commit).name;shutil.copyfile(checked_asset(resume['root'],commit_name),output/commit_name)
                if stable_digest(output/commit_name)['sha256']!=resume_commit_sha256:raise ValueError('A resume commit changed while copying')
                resume_file=_write_json(output/'RESUME.json',resume['provenance'])
            with (output/'training.jsonl').open('x') as log,(output/'BATCHES.private.jsonl').open('x') as batch_log:
                if resume is not None:
                    log.write(resume['logs']['training_log'].decode());log.flush()
                    batch_log.write(resume['logs']['batch_log'].decode());batch_log.flush()
                for completed in range(start_step,spec.steps):
                    sweep,batch_index=divmod(completed,batches)
                    if plan is None or batch_index==0:plan=bank.schedule.sweep(sweep,name)
                    batch,batch_receipt=make_batch(bank.schedule,plan,batch_index,bank.root);batch=batch.to(device)
                    selected=plan['pairs'][batch_index*spec.pairs_per_batch:(batch_index+1)*spec.pairs_per_batch]
                    recipient_digests=[]
                    for pair in selected:
                        case_chain.update(pair['pair_sha256'].encode())
                        value=digest({key:pair[key] for key in ('recipient_system','recipient_episode','donor_start','recipient_start','history_frames','horizons')})
                        recipient_chain.update(value.encode());recipient_digests.append(value)
                    rate=pretraining_learning_rate(spec,completed)
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.zero_grad(set_to_none=True);model.begin_train_step()
                    try:
                        with _step_rng(spec.stochastic_seed,completed,device) as random_seed:
                            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=spec.precision=='bfloat16_cuda'):
                                loss,metrics=objective(model,batch,sigreg,objective_config)
                            if not torch.isfinite(loss):raise ValueError('nonfinite A pretraining loss')
                            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec.gradient_clip,error_if_nonfinite=True)
                            optimizer.step();model.finish_train_step(split='train');updates=completed+1
                    except Exception:
                        model.discard_train_step();raise
                    if not torch.stack([torch.isfinite(t).all() for t in model.state_dict().values()]).all():raise ValueError('nonfinite A model/buffer after update')
                    batch_log.write(json.dumps(dict(step=updates,stochastic_seed=random_seed,recipient_sha256=recipient_digests,**batch_receipt),allow_nan=False)+'\n');batch_log.flush()
                    exposure['pairs']+=spec.pairs_per_batch;exposure['history_frames']+=batch_receipt['presented_history_frames']
                    exposure['target_frames']+=batch_receipt['target_frames'];exposure['action_intervals']+=batch_receipt['observed_action_intervals']
                    window_losses.append(float(loss.detach()))
                    event=dict(step=updates,sweep=sweep,batch_index=batch_index,learning_rate=rate,
                        gradient_norm=float(norm),seconds=time.monotonic()-started,**{k:float(v) for k,v in metrics.items()})
                    if updates in save_steps:
                        filename=f'checkpoint_{updates:07d}.pt';path=output/filename
                        checkpoint=dict(schema=SCHEMA,step=updates,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            optimizer=optimizer.state_dict(),model_state_sha256=model_state_sha256(model),initial_model_sha256=initial,
                            run_sha256=run_file['sha256'],binding=binding,training_pair_sequence_sha256=case_chain.hexdigest(),recipient_sequence_sha256=recipient_chain.hexdigest())
                        with path.open('xb') as stream:torch.save(checkpoint,stream)
                        files[filename]=dict(path=filename,**stable_digest(path));event['checkpoint']=filename
                    if updates%spec.log_every==0 or 'checkpoint' in event or updates==spec.steps:
                        event.update(window_mean_loss=sum(window_losses)/len(window_losses),window_updates=len(window_losses));window_losses=[]
                        log.write(json.dumps(event,allow_nan=False)+'\n');log.flush()
                        if 'checkpoint' in event:
                            _write_json(output/f'checkpoint_{updates:07d}.COMMIT.json',dict(schema='vec.A-pretraining-checkpoint.v1',status='CHECKPOINT_COMMITTED',
                                step=updates,run=run_file,files=files,exposure=exposure,
                                training_log=dict(path='training.jsonl',**stable_digest(output/'training.jsonl')),
                                batch_log=dict(path='BATCHES.private.jsonl',**stable_digest(output/'BATCHES.private.jsonl'))))
                        if progress is not None:progress(event)
            after=bank.verify_all(workers)
            if source_fingerprint()!=binding['source_fingerprint'] or a_source_fingerprint()!=binding['A_source_fingerprint']:raise ValueError('A pretraining source changed')
            if stable_digest(output/'RUN.json')!={k:run_file[k] for k in ('bytes','sha256')}:raise ValueError('A pretraining registration changed')
            for entry in files.values():
                if stable_digest(checked_asset(output,entry['path']))!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('A saved checkpoint changed during training')
            if int(model.observation.norm.num_batches_tracked)!=spec.steps or model.observation._snapshot is not None:
                raise ValueError('A training-history normalization updates incomplete')
            model.eval();result=dict(schema=SCHEMA,status='TRAINING_COMPLETE',name=name,run=run_file,configuration=objective_config,spec=config,binding=binding,
                selected_step=spec.steps,selected_checkpoint=f'checkpoint_{spec.steps:07d}.pt',selection_rule='final_step_only',optimizer_updates=updates,
                resumed_from_step=start_step,new_optimizer_updates=updates-start_step,resume=resume_file,
                initial_model_sha256=initial,model_state_sha256=model_state_sha256(model),train_history_statistic_updates=int(model.observation.norm.num_batches_tracked),
                parameters=sum(p.numel() for p in model.parameters()),training_pair_sequence_sha256=case_chain.hexdigest(),recipient_sequence_sha256=recipient_chain.hexdigest(),
                files=files,training_log=dict(path='training.jsonl',**stable_digest(output/'training.jsonl')),
                batch_log=dict(path='BATCHES.private.jsonl',**stable_digest(output/'BATCHES.private.jsonl')),exposure=exposure,before=before,after=after,
                device=str(device),torch_version=torch.__version__,seconds=time.monotonic()-started,physical_labels_parsed=(name=='Supervised-Split'),validation_used=False,
                formal_admission_verified=False,formal_training=False,formal_results=False,test_read=False)
            _write_json(output/'COMPLETE.json',result)
            return dict(status='TRAINING_COMPLETE',receipt_sha256=stable_digest(output/'COMPLETE.json')['sha256'],
                model_state_sha256=result['model_state_sha256'],selected_step=spec.steps,formal_results=False)
    except Exception as error:
        _write_json(output/'FAILURE.json',dict(status='FAILED_RETAINED',completed_updates=updates,error_type=type(error).__name__,error=str(error),formal_results=False,test_read=False))
        raise


def load_pretrained_model(root,*,receipt_sha256,bank_snapshot_sha256,device='cpu'):
    """Read a bound final model without loading train/validation state labels."""
    root=Path(root);path=checked_asset(root,'COMPLETE.json')
    if stable_digest(path)['sha256']!=receipt_sha256:raise ValueError('A pretraining completion receipt differs')
    record=json.loads(path.read_text());runpath=checked_asset(root,'RUN.json')
    if record['run']['path']!='RUN.json' or stable_digest(runpath)!={k:record['run'][k] for k in ('bytes','sha256')}:raise ValueError('A pretraining run registration differs')
    run=json.loads(runpath.read_text());binding=record['binding']
    if record.get('schema')!=SCHEMA or record.get('status')!='TRAINING_COMPLETE' or record.get('test_read') or record.get('validation_used'):
        raise ValueError('incomplete/unpermitted A pretraining model')
    if binding!=run['binding'] or binding['bank_snapshot_sha256']!=bank_snapshot_sha256 or binding['source_fingerprint']!=source_fingerprint() or binding['A_source_fingerprint']!=a_source_fingerprint():
        raise ValueError('A pretrained model bank/source differs')
    spec_values={f.name:record['spec'][f.name] for f in fields(PretrainingSpec)};spec_values['milestone_steps']=tuple(spec_values['milestone_steps']);spec=PretrainingSpec(**spec_values)
    if spec.record()!=record['spec'] or run['spec']!=record['spec'] or configuration(record['name'])!=record['configuration'] or run['configuration']!=record['configuration']:
        raise ValueError('A pretraining registered settings differ')
    final=f'checkpoint_{spec.steps:07d}.pt'
    if record.get('selected_step')!=spec.steps or record.get('optimizer_updates')!=spec.steps or record.get('selection_rule')!='final_step_only' or record.get('selected_checkpoint')!=final:
        raise ValueError('A pretraining final-step selection differs')
    start=record.get('resumed_from_step',0)
    if type(start) is not int or start<0 or start>spec.steps or record.get('new_optimizer_updates',spec.steps)!=spec.steps-start:
        raise ValueError('A resumed update accounting differs')
    if record.get('resume') is not None:
        entry=record['resume'];path_resume=checked_asset(root,'RESUME.json')
        if entry['path']!='RESUME.json' or stable_digest(path_resume)!={k:entry[k] for k in ('bytes','sha256')}:
            raise ValueError('A resume provenance changed')
        provenance=json.loads(path_resume.read_text());commit_name=f'checkpoint_{start:07d}.COMMIT.json'
        if (start not in _saved_steps(spec) or provenance.get('schema')!='vec.A-pretraining-resume.v1'
            or provenance.get('resumed_from_step')!=start or provenance.get('original_budget')!=spec.steps
            or provenance.get('new_optimizer_updates')!=spec.steps-start or provenance.get('parent_commit')!=commit_name
            or stable_digest(checked_asset(root,commit_name))['sha256']!=provenance.get('parent_commit_sha256')):
            raise ValueError('A resume provenance binding differs')
    elif start!=0:raise ValueError('A resumed model lacks provenance')
    if set(record['files'])!={f'checkpoint_{s:07d}.pt' for s in _saved_steps(spec)}:raise ValueError('A pretraining scheduled checkpoints missing')
    for filename,entry in record['files'].items():
        if entry['path']!=filename or stable_digest(checked_asset(root,filename))!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('A pretrained checkpoint changed')
    for key,filename in (('training_log','training.jsonl'),('batch_log','BATCHES.private.jsonl')):
        if record[key]['path']!=filename or stable_digest(checked_asset(root,filename))!={k:record[key][k] for k in ('bytes','sha256')}:raise ValueError('A pretraining log changed')
    with checked_asset(root,final).open('rb') as stream:checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=SCHEMA or checkpoint.get('step')!=spec.steps or checkpoint.get('binding')!=binding or checkpoint.get('run_sha256')!=record['run']['sha256']:
        raise ValueError('A pretrained checkpoint content binding differs')
    model=_new_model(record['name'],spec,torch.device(device))
    if model_state_sha256(model)!=record['initial_model_sha256'] or checkpoint.get('initial_model_sha256')!=record['initial_model_sha256']:raise ValueError('A initialization binding differs')
    for key in ('training_pair_sequence_sha256','recipient_sequence_sha256'):
        if checkpoint.get(key)!=record[key]:raise ValueError('A sample sequence binding differs')
    model.load_state_dict(checkpoint['model'],strict=True)
    if model_state_sha256(model)!=record['model_state_sha256'] or checkpoint['model_state_sha256']!=record['model_state_sha256'] or int(model.observation.norm.num_batches_tracked)!=spec.steps:
        raise ValueError('A pretrained model tensors/statistics differ')
    entry=record['files'][final]
    if stable_digest(checked_asset(root,final))!={k:entry[k] for k in ('bytes','sha256')} or stable_digest(path)['sha256']!=receipt_sha256:
        raise ValueError('A pretrained artifacts changed during loading')
    model.eval();model.requires_grad_(False);return model,_immutable(record)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='A train-only engine; formal admission must be provided by the experiment launcher')
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--configuration',choices=list(CONFIGURATIONS),required=True);parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--bank-snapshot-sha256',required=True);parser.add_argument('--device',default='cuda');parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--resume-commit',type=Path);parser.add_argument('--resume-commit-sha256')
    args=parser.parse_args();values=json.loads(args.spec.read_text());values['milestone_steps']=tuple(values.get('milestone_steps',(3000,)))
    result=train_pretraining(args.bank,args.output,name=args.configuration,spec=PretrainingSpec(**values),bank_snapshot_sha256=args.bank_snapshot_sha256,
        device=args.device,workers=args.workers,progress=lambda event:print(json.dumps(event,allow_nan=False),flush=True),
        resume_commit=args.resume_commit,resume_commit_sha256=args.resume_commit_sha256)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
