"""One history-encoder-matched relation-contrastive repair, no label objective.

Single shared observation encoder for query and donor. This is a matched
Rel-InfoNCE adaptation, not a claim to reproduce the published FCRL pipeline.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .io import read,write,sha,digest,code_hashes,development_path

TEMPERATURES=(.05,.1,.2)
THRESHOLDS=dict(uniform_ce_margin=.1,positive_negative_gap=.01,mean_normalized_std=.015,effective_rank=4.)


def symmetric_infonce(projected,raw,temperature):
    if len(projected)%2 or len(projected)<4:raise ValueError('paired batch with at least two systems required')
    # .float() alone is insufficient when an outer autocast is active.
    with torch.autocast(device_type=projected.device.type,enabled=False):
        z=F.normalize(projected.float(),dim=-1);p=F.normalize(raw.float(),dim=-1)
        n=len(z)//2;cos=z[:n]@z[n:].T;logits=cos/temperature
        labels=torch.arange(n,device=z.device)
        loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))*.5
        with torch.no_grad():
            diagonal=torch.eye(n,device=z.device,dtype=torch.bool)
            eig=torch.linalg.svdvals(p-p.mean(0)).square();prob=eig/eig.sum().clamp_min(1e-20)
            rank=torch.exp(-(prob*prob.clamp_min(1e-20).log()).sum())
            metrics=dict(loss=float(loss),uniform_ce=math.log(n),uniform_ce_margin=math.log(n)-float(loss),
                positive_negative_gap=float(cos.diag().mean()-cos[~diagonal].mean()),
                mean_normalized_std=float(p.std(0,unbiased=False).mean()),effective_rank=float(rank),
                similarity_dtype=str(logits.dtype))
    return loss,metrics


def objective(model,batch,keys):
    batch.validate(96);half=len(batch.history_images)//2
    if len(keys)!=half or len(set(keys))!=half:raise ValueError('same-system false negatives forbidden')
    history=model.observation(batch.history_images,train_history=model.training)
    code=model.persistent_code(history,batch.history_actions)
    return symmetric_infonce(model.projector(code),code,model.temperature)


def qualify(events,steps):
    if not events:raise ValueError('no optimization records')
    tail=events[-min(100,len(events)):]
    values={k:float(np.mean([e[k] for e in tail])) for k in THRESHOLDS}
    return dict(status='PASS' if steps==10000 and all(values[k]>=v for k,v in THRESHOLDS.items()) else 'FAIL',
        thresholds=THRESHOLDS,tail_mean=values,updates=steps,
        criterion='predeclared optimization margins, not a statistical significance test',test_read=False)


def setup(legacy_root):
    root=development_path(legacy_root).resolve()
    if 'runtime' in sys.modules:raise RuntimeError('use a fresh process for native baseline adapter')
    os.environ['SPRII_BASELINE_ROOT']=str(root);sys.path.insert(0,str(root))
    import runtime
    runtime.verify_native()
    files={str(p.resolve()):sha(p) for p in sorted((root/'native').rglob('*.py'))}
    for name in ('runtime.py','baseline_model.py'):files[str(root/name)]=sha(root/name)
    return runtime,files


def verify_selection(path):
    r=read(development_path(path))
    if r.get('status')!='FROZEN_ONCE' or r['selection_seed']!=90 or r['test_read'] is not False:
        raise PermissionError('frozen development selection required')
    for p,h in r['artifacts'].items():
        if sha(p)!=h:raise ValueError('temperature selection evidence changed')
    if r['code_sha256']!=code_hashes():raise ValueError('code changed after temperature freeze')
    return r


def train(legacy_root,output,seed,temperature=None,selection=None,device='cuda:0',smoke_steps=None):
    if seed==90:
        if selection is not None or temperature not in TEMPERATURES:raise ValueError('dev seed90 and fixed temperature grid required')
    elif seed in (0,1,2):
        if selection is None:raise PermissionError('final source seeds require frozen temperature selection')
        chosen=verify_selection(selection)
        if temperature is not None and temperature!=chosen['temperature']:raise ValueError('temperature differs from selection')
        temperature=chosen['temperature']
    else:raise ValueError('source seeds are dev90 or final0,1,2')
    runtime,source_files=setup(legacy_root)
    if selection is not None and chosen['source_files']!=source_files:raise ValueError('native source changed after selection')
    from baseline_model import new_source
    from native_training import NativePairSchedule,make_batch
    from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec,pretraining_learning_rate,_step_rng
    from persistbench.envs.visual_elastic_coupling.a_head_features import model_state_sha256
    from persistbench.envs.visual_elastic_coupling.dataset_snapshot import verify
    from .engine import math_profile
    device=torch.device(device);math_profile(device)
    torch.set_num_threads(1)
    manifest=read(development_path(runtime.BANK/'MANIFEST.private.json'))
    if any(r['split'] not in ('train','validation') for r in manifest['episodes']):raise PermissionError('development bank required')
    snapshot=read(runtime.BANK/'BANK_SNAPSHOT.json');verify(runtime.BANK,snapshot,workers=8)
    steps=10000 if smoke_steps is None else smoke_steps
    spec=PretrainingSpec(steps=10000,pairs_per_batch=48,history_frames=96,model_seed=seed,sampling_seed=seed,
        stochastic_seed=seed,save_every=2500,log_every=100,milestone_steps=(1000,2500,5000,10000))
    schedule=NativePairSchedule(manifest,seed=seed,pairs_per_batch=48)
    count=len({r['system_key'] for r in manifest['episodes'] if r['split']=='train'})
    if count!=144:raise ValueError('registered source bank requires 144 physical training systems')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model=new_source('fcrl_history',seed,dimension=64,temperature=temperature,device=device).train()
    run=dict(recipe='RelInfoNCE',native_recipe='fcrl_history',seed=seed,temperature=temperature,history_frames=96,
        history_encoder='unchanged native HistoryEncoder P64',query_encoder='same observation encoder as history',
        source_updates=steps,source_files=source_files,code_sha256=code_hashes(),spec=spec.record(),
        thresholds=THRESHOLDS,selection=None if selection is None else dict(path=str(Path(selection).resolve()),sha256=sha(selection)),
        bank_snapshot_sha256=runtime.SNAP,test_read=False,smoke=smoke_steps is not None)
    write(out/'RUN.json',run)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec.learning_rate,weight_decay=spec.weight_decay,foreach=False,fused=False)
    events=[];plan=None;sequence=[]
    with (out/'training.jsonl').open('x') as log:
        for step in range(steps):
            sweep,bi=divmod(step,3)
            if plan is None or bi==0:plan=schedule.sweep(sweep,'Both')
            batch,receipt=make_batch(schedule,plan,bi,runtime.BANK)
            if receipt['physical_labels_read'] or receipt['test_read']:raise PermissionError('contrastive source saw private evidence')
            keys=[x['recipient_system'] for x in plan['pairs'][bi*48:(bi+1)*48]]
            batch=batch.to(device);sequence.append(digest(receipt))
            for group in optimizer.param_groups:group['lr']=pretraining_learning_rate(spec,step)
            optimizer.zero_grad(set_to_none=True);model.begin_train_step()
            with _step_rng(seed,step,device):
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    loss,metrics=objective(model,batch,keys)
                if not torch.isfinite(loss):raise ValueError('nonfinite contrastive objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec.gradient_clip,error_if_nonfinite=True)
                optimizer.step();model.finish_train_step()
            event=dict(step=step+1,**metrics);events.append(event)
            if (step+1)%100==0 or step+1==steps:
                log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
    qualification=qualify(events,steps);write(out/'QUALIFICATION.json',qualification)
    if code_hashes()!=run['code_sha256'] or any(sha(p)!=h for p,h in source_files.items()):raise ValueError('source implementation changed')
    runtime.verify_native();verify(runtime.BANK,snapshot,workers=8)
    ck=dict(settings=dict(recipe='fcrl_history',seed=seed,dimension=64,temperature=temperature),step=steps,
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_sha256=model_state_sha256(model))
    torch.save(ck,out/'source.pt')
    complete=dict(status='COMPLETE',method='RelInfoNCE',source_seed=seed,selected_step=steps,checkpoint=str((out/'source.pt').resolve()),
        checkpoint_sha256=sha(out/'source.pt'),model_state_sha256=ck['model_sha256'],run_sha256=sha(out/'RUN.json'),
        qualification_sha256=sha(out/'QUALIFICATION.json'),qualified=qualification['status']=='PASS',temperature=temperature,
        training_sequence_sha256=digest(sequence),test_read=False,smoke=smoke_steps is not None)
    write(out/'COMPLETE.json',complete)
    return complete


def select(completions,output):
    rows=[];artifacts={};source_files=None
    for path in completions:
        cp=Path(path).resolve();c=read(cp);r=read(cp.parent/'RUN.json');q=read(cp.parent/'QUALIFICATION.json')
        if c['source_seed']!=90 or c['smoke'] or c['selected_step']!=10000 or c['run_sha256']!=sha(cp.parent/'RUN.json'):
            raise ValueError('complete development seed90 sources required')
        if r['code_sha256']!=code_hashes():raise ValueError('candidate was trained by another implementation')
        if c['checkpoint_sha256']!=sha(c['checkpoint']) or c['qualification_sha256']!=sha(cp.parent/'QUALIFICATION.json'):
            raise ValueError('candidate artifacts changed')
        if source_files is None:source_files=r['source_files']
        if source_files!=r['source_files']:raise ValueError('native source versions differ across temperatures')
        for p in (cp,cp.parent/'RUN.json',cp.parent/'QUALIFICATION.json',Path(c['checkpoint'])):artifacts[str(p)]=sha(p)
        rows.append((c,q))
    if sorted(c['temperature'] for c,q in rows)!=list(TEMPERATURES):raise ValueError('complete three-temperature grid required')
    passed=[(c,q) for c,q in rows if q['status']=='PASS' and c['qualified']]
    if not passed:raise ValueError('no successful contrastive candidate; task evaluation stays closed')
    best=min(passed,key=lambda x:(-x[1]['tail_mean']['uniform_ce_margin'],x[0]['temperature']))[0]
    result=dict(status='FROZEN_ONCE',selection_seed=90,temperature=best['temperature'],artifacts=artifacts,
        source_files=source_files,code_sha256=code_hashes(),rule='lowest final-tail CE among optimization-qualified candidates; tie lowest temperature',test_read=False)
    write(output,result);return result


def export(legacy_root,completion,output,device='cuda:0'):
    runtime,source_files=setup(legacy_root)
    from baseline_model import new_source
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan,AHeadDataAccess
    from persistbench.envs.visual_elastic_coupling.a_head_features import extract_features,AHeadFeatureCache,model_state_sha256
    from persistbench.envs.visual_elastic_coupling.a_head_targets import extract_targets
    cp=Path(completion).resolve();c=read(cp);r=read(cp.parent/'RUN.json');q=read(cp.parent/'QUALIFICATION.json')
    if c['source_seed'] not in (0,1,2) or c['smoke'] or not c['qualified'] or q['status']!='PASS':raise PermissionError('qualified final source required')
    if sha(cp.parent/'QUALIFICATION.json')!=c['qualification_sha256'] or sha(cp.parent/'RUN.json')!=c['run_sha256']:
        raise ValueError('source receipts changed')
    selection=verify_selection(r['selection']['path'])
    if sha(r['selection']['path'])!=r['selection']['sha256'] or r['source_files']!=source_files or selection['source_files']!=source_files:
        raise ValueError('source lineage differs from frozen selection')
    if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('source checkpoint changed')
    ck=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)
    model=new_source(**ck['settings'],device=device);model.load_state_dict(ck['model']);model.eval().requires_grad_(False)
    if ck['step']!=10000 or model_state_sha256(model)!=c['model_state_sha256']:raise ValueError('source content differs')
    plan=AHeadCasePlan(read(runtime.BANK/'MANIFEST.private.json'),seed=0,history_frames=96)
    access=AHeadDataAccess(runtime.BANK,plan,snapshot_sha256=runtime.SNAP)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    extract_features(access,model,out/'features',expected_model_state_sha256=c['model_state_sha256'],workers=8)
    fr=out/'features/FEATURES.json'
    features=AHeadFeatureCache(fr.parent,plan,receipt_sha256=sha(fr),model_state_sha256=c['model_state_sha256'],bank_snapshot_sha256=runtime.SNAP)
    extract_targets(access,features,out/'targets',workers=8)
    descriptor=dict(environment='springworld',method='RelInfoNCE',source_seed=c['source_seed'],native_files=source_files,
        native_paths=[str(runtime.ROOT),str(runtime.NATIVE),*[str(runtime.NATIVE/k) for k in ('src','a_src','extension')]],
        model_state_sha256=c['model_state_sha256'],source_completion=str(cp),source_completion_sha256=sha(cp))
    for key,path in dict(manifest=runtime.BANK/'MANIFEST.private.json',features_receipt=fr,targets_receipt=out/'targets/SUPERVISION.json',checkpoint=Path(c['checkpoint'])).items():
        descriptor[key]=str(path.resolve());descriptor[key+'_sha256']=sha(path)
    from .providers import SpringCache
    SpringCache(descriptor);write(out/'SOURCE.json',descriptor);return descriptor


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('train');c.add_argument('--legacy-root',required=True);c.add_argument('--output',required=True)
    c.add_argument('--seed',type=int,required=True);c.add_argument('--temperature',type=float);c.add_argument('--selection')
    c.add_argument('--device',default='cuda:0');c.add_argument('--smoke-steps',type=int)
    c=sub.add_parser('select');c.add_argument('--completions',nargs=3,required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('export');c.add_argument('--legacy-root',required=True);c.add_argument('--completion',required=True)
    c.add_argument('--output',required=True);c.add_argument('--device',default='cuda:0')
    a=vars(p.parse_args());cmd=a.pop('command');print(json.dumps(dict(train=train,select=select,export=export)[cmd](**a),indent=2))


if __name__=='__main__':main()
