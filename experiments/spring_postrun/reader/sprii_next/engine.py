"""Fixed-budget reader training and full-vector evaluation."""
import fcntl
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .decoder import RidgeDecoder
from .io import code_hashes, digest, read, sha, write, npz
from .model import Reader


def context(batch, arm, decoder):
    if arm in ('persistent','matched'):return batch.persistent
    if arm=='null':return np.zeros_like(batch.persistent)
    if arm.startswith('decode'):return decoder.predict(batch.persistent)
    if arm.startswith('oracle'):return decoder.oracle(batch.theta)
    raise ValueError(arm)


def tensors(batch, arm, decoder, device):
    floating=(batch.query,context(batch,arm,decoder),batch.actions,batch.mask)
    return [torch.as_tensor(x,dtype=torch.float32,device=device) for x in floating]+[
        torch.as_tensor(batch.horizon_index,dtype=torch.long,device=device)]


def get_decoder(provider, dim=3, alpha=1.):
    p,theta,systems,_=provider.donors('train')
    return RidgeDecoder.fit(p,theta,systems,split='train',alpha=alpha,dim=dim)


def jobs(environment, stage='development', secondary=False):
    if environment=='springworld':
        if stage not in ('development','baseline'):raise ValueError('SpringWorld development/baseline stage required')
        methods=['Both'] if stage=='development' else ['RelInfoNCE']
        arms=['null','persistent','decode','oracle'] if stage=='development' else ['null','persistent'];readers=range(3)
        if secondary and stage=='baseline':raise ValueError('no extra baseline arms')
        if secondary:arms+=['decode4','oracle4']
    elif environment=='pokeworld':
        if stage not in ('pilot','formal'):raise ValueError('PokeWorld pilot/formal stage required')
        methods=['G1','G2'];arms=['null','matched'];readers=[0] if stage=='pilot' else [1,2]
    else:raise ValueError('unknown environment')
    return [dict(environment=environment,stage=stage,method=m,source_seed=s,reader_seed=r,arm=a)
            for m in methods for s in range(3) for r in readers for a in arms]


def job_name(job):
    return f"{job['environment']}/{job['stage']}/{job['method']}/source{job['source_seed']}/reader{job['reader_seed']}/{job['arm']}"


def math_profile(device):
    if device.type=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA process startup')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def evaluate(head,provider,decoder,output,job,run_hash,*,batch_size=256):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    head.eval();device=next(head.parameters()).device
    shards=[];systems={};full=[];row_keys=set()
    with torch.inference_mode():
        for n,b in enumerate(provider.evaluation_batches(batch_size)):
            pred=head(*tensors(b,job['arm'],decoder,device)).cpu().numpy()
            target=np.asarray(b.target,np.float32)
            error=np.square(pred.astype(np.float64)-target.astype(np.float64))
            if pred.shape!=target.shape or not np.isfinite(error).all():raise ValueError('invalid prediction')
            rows=[]
            for i,row in enumerate(b.rows):
                key=(row['system_id'],row['query_id'])
                if key in row_keys:raise ValueError('duplicate evaluation query')
                row_keys.add(key)
                r=dict(row,source_method=job['method'],relation=job['method'],source_seed=job['source_seed'],
                    reader_seed=job['reader_seed'],reader_arm=job['arm'],stage=job['stage'],
                    aggregate_error=float(error[i].mean()),vector_index=i,
                    supplied_context='null' if job['arm']=='null' else job['arm'])
                rows.append(r)
                if r['primary']:systems.setdefault(r['system_id'],[]).append(r['aggregate_error'])
                full.append((r['aggregate_error'],r.get('split_weight')))
            tensor_path=output/f'vectors_{n:05d}.npz'
            npz(tensor_path,persistent_code=b.persistent.astype(np.float32),prediction_vector=pred,
                target_vector=target,error_per_dimension=error,context_vector=context(b,job['arm'],decoder),
                query_embedding=b.query, future_actions=b.actions,action_mask=b.mask,
                physical_parameters=b.theta, horizon_index=b.horizon_index)
            row_path=output/f'rows_{n:05d}.json';write(row_path,rows)
            shards.append(dict(vectors=tensor_path.name,vectors_sha256=sha(tensor_path),rows=row_path.name,
                               rows_sha256=sha(row_path),count=len(rows)))
    if not systems:raise ValueError('primary endpoint empty')
    means={str(k):float(np.mean(v)) for k,v in sorted(systems.items())}
    result=dict(schema='sprii-next.evaluation.v1',job=job,run_sha256=run_hash,provider_sha256=provider.identity,
        normalization=provider.normalization,primary_system_mse=means,primary_mse=float(np.mean(list(means.values()))),
        primary_cases=sum(map(len,systems.values())),all_cases=len(row_keys),shards=shards,test_read=False,
        interpretation='development diagnostic' if job['stage']=='pilot' else 'development; not sealed confirmation')
    if provider.environment=='springworld':
        if len(systems)!=100 or result['primary_cases']!=600 or len(row_keys)!=16020:
            raise ValueError('SpringWorld profile changed')
        if not np.isclose(sum(w for _,w in full),1.):raise ValueError('full-mixture weights changed')
        result['secondary_full_mixture_mse']=sum(e*w for e,w in full)
    write(output/'RESULT.json',result)
    return result


def run_job(provider,cfg,job,output,*,device='cuda:0',smoke_steps=None,gate_path=None):
    if smoke_steps is None and getattr(provider,'synthetic_fixture',False):raise ValueError('synthetic assets require explicitly marked smoke mode')
    if provider.environment!=job['environment'] and not (smoke_steps is not None and provider.environment=='fixture'):
        raise ValueError('provider belongs to another environment')
    if provider.descriptor['method']!=job['method'] or provider.descriptor['source_seed']!=job['source_seed']:
        raise ValueError('source method/seed differs from job')
    if job['environment']=='pokeworld' and job['stage']=='formal':
        from .statistics import verify_gate
        if gate_path is None:raise PermissionError('pilot gate required before additional readers')
        verify_gate(gate_path,digest(cfg))
    device=torch.device(device)
    if device.type not in ('cuda','cpu'):raise ValueError('supported device: CUDA or CPU')
    math_profile(device)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # One process owns one job; independent GPUs can consume distinct job indices.
    lock_path=output.parent/(output.name+'.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise FileExistsError('preserve existing run: '+str(output))
        output.mkdir()
        steps=cfg['reader']['steps'] if smoke_steps is None else smoke_steps
        if steps<1:raise ValueError('positive update count required')
        spec=dict(cfg['reader']);spec['steps']=steps
        if smoke_steps is not None:spec['warmup_steps']=min(spec['warmup_steps'],steps-1)
        dim=4 if job['arm'].endswith('4') else 3
        decoder=get_decoder(provider,dim,cfg['ridge_alpha'])
        head=Reader(job['arm'],job['reader_seed'],dim).to(device)
        initial_common=digest({n:p.detach().cpu().tolist() for n,p in head.named_parameters() if not n.startswith('physics_projection')})
        run=dict(schema='sprii-next.run.v1',job=job,protocol_sha256=digest(cfg),provider_sha256=provider.identity,
            provider_descriptor=provider.descriptor,reader=spec,architecture=head.architecture(),
            initial_common_parameters_sha256=initial_common,decoder=decoder.record(),code_sha256=code_hashes(),
            source_optimizer_steps=0,checkpoint_rule='final_step_only',normalization=provider.normalization,
            smoke=smoke_steps is not None,test_read=False)
        import platform,socket
        run['runtime']=dict(host=socket.gethostname(),python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
            device=str(device),cuda=torch.version.cuda,gpu=None if device.type!='cuda' else torch.cuda.get_device_name(device),
            precision='float32',deterministic_algorithms=True,tf32=False)
        write(output/'RUN.json',run)
        p,theta,ids,donor_ids=provider.donors('validation')
        write(output/'PROBE.json',decoder.score(p,theta,ids))
        from .decoder import physical_coordinates
        npz(output/'probe_vectors.npz',persistent_code=p,physical_parameters=theta,system_id=ids,donor_id=donor_ids,
            prediction_vector=decoder.predict(p)*decoder.scale_y+decoder.mean_y,target_vector=physical_coordinates(theta,dim))
        optimizer=torch.optim.AdamW(head.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'],foreach=False)
        sequence=[];start=time.monotonic();completed=0
        try:
            head.train()
            with (output/'training.jsonl').open('x') as log:
                for step in range(steps):
                    b=provider.training_batch(job['reader_seed'],step,spec['batch_size'])
                    sequence.append(digest([(r['query_id'],r['donor_id']) for r in b.rows]))
                    warm=spec['warmup_steps']
                    factor=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
                    for group in optimizer.param_groups:group['lr']=spec['learning_rate']*factor
                    optimizer.zero_grad(set_to_none=True)
                    pred=head(*tensors(b,job['arm'],decoder,device))
                    target=torch.as_tensor(b.target,dtype=torch.float32,device=device)
                    loss=(pred-target).square().mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),spec['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();completed=step+1
                    if completed%100==0 or completed==steps:
                        import json
                        event=dict(step=completed,loss=float(loss.detach()),seconds=time.monotonic()-start)
                        log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
            if code_hashes()!=run['code_sha256']:raise ValueError('code changed during run')
            torch.save(dict(model=head.state_dict(),job=job,run_sha256=sha(output/'RUN.json'),decoder=decoder.record()),output/'head.pt')
            result=evaluate(head,provider,decoder,output/'evaluation',job,sha(output/'RUN.json'))
            write(output/'COMPLETE.json',dict(status='COMPLETE',job=job,optimizer_updates=completed,
                protocol_sha256=digest(cfg),run_sha256=sha(output/'RUN.json'),checkpoint_sha256=sha(output/'head.pt'),
                evaluation_sha256=sha(output/'evaluation/RESULT.json'),training_sequence_sha256=digest(sequence),
                probe_sha256=sha(output/'PROBE.json'),probe_vectors_sha256=sha(output/'probe_vectors.npz'),
                smoke=smoke_steps is not None,test_read=False,primary_mse=result['primary_mse']))
        except Exception as e:
            write(output/'FAILURE.json',dict(completed_updates=completed,error=repr(e),test_read=False))
            raise
