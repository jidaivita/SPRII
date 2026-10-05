"""Native/structured RSSM source formation; immutable 50/100/150 snapshots.

Frozen RGB features only. Native has no P tower. Relation variants use the same
structured network, with independent AB donors and focal-only auxiliary loss.
No source or readout training runs when this module is imported.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
# Distinct names avoid contaminating a dynamically loaded shared readout core.
def local_module(name):
    key='_cophy_v7_rssm_'+name
    spec=importlib.util.spec_from_file_location(key,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module);return module
M=local_module('models');D=local_module('data')
VERSION=M.VERSION
METHODS=('Native','Structure','Cross','Random-Cross')
read,write,digest,canonical=D.read,D.write,D.digest,D.canonical
BUDGETS=(50,100,150)

def emit(event,**values):print(json.dumps(dict(event=event,time=time.time(),**values),allow_nan=False),flush=True)
def seeded(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        x=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));torch.save(value,tmp);os.replace(tmp,path)
def rng_state(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
def restore_rng(state,device,legacy_index=None):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    cuda=state.get('cuda')
    if device.type=='cuda' and cuda is not None:
        if isinstance(cuda,list):
            if legacy_index is None or not 0<=legacy_index<len(cuda):raise ValueError('Explicit parent CUDA index needed for legacy RNG')
            cuda=cuda[legacy_index]
        torch.cuda.set_rng_state(cuda,device)
def mse(pred,target,mask):
    error=(pred.float()-target.float()).square().mean(-1)
    return (error*mask).sum()/mask.sum().clamp_min(1)
def mean_mask(values,mask):return (values.float()*mask).sum()/mask.sum().clamp_min(1)

def normalization(data,out):
    dest=out/'normalization.npz';meta=out/'normalization.json';cfg=M.ModelConfig()
    chosen=sorted(range(len(data.ids['train'])),key=lambda i:hashlib.sha256(('rssm-normalize/'+data.ids['train'][i]).encode()).hexdigest())[:256]
    definition=dict(split='train',ids=[data.ids['train'][i] for i in chosen],streams=['features_ab','features_cd'],
                    feature_channels=784,scale_floor=.001,visible_objects_only=True,test_read=False)
    if meta.exists():
        old=read(meta)
        if old['definition']!=definition or old['normalization_sha256']!=digest(dest):raise ValueError('Normalization binding changed')
        return old
    total=np.zeros(784,np.float64);squares=np.zeros(784,np.float64);count=0
    for first in range(0,len(chosen),8):
        ix=np.asarray(chosen[first:first+8])
        for suffix in ('ab','cd'):
            mask=np.asarray(data.arrays['train']['presence_'+suffix][ix],bool)
            values=np.asarray(data.arrays['train']['features_'+suffix][ix],np.float64)[mask]
            total+=values.sum(0);squares+=np.square(values).sum(0);count+=len(values)
    if count<2:raise ValueError('No normalization tokens')
    center=total/count;scale=np.sqrt(np.maximum(squares/count-center*center,0)).clip(.001)
    tmp=dest.with_suffix('.tmp.npz');np.savez(tmp,mean=center.astype(np.float32),scale=scale.astype(np.float32));os.replace(tmp,dest)
    result=dict(status='COMPLETE',definition=definition,observed_tokens=count,normalization_sha256=digest(dest),test_read=False)
    write(meta,result);return result

def build_binding(args,data):
    files=dict(data.files)
    for file in (HERE/'models.py',HERE/'train.py',HERE/'data.py',Path(args.relation_index),Path(args.protocol),Path(args.out)/'normalization.json'):
        files[str(file.resolve())]=digest(file)
    architecture='native' if args.method=='Native' else 'structured'
    cfg=asdict(M.ModelConfig(architecture=architecture))
    result=dict(version=VERSION,scene=args.scene,family='RSSM',method=args.method,seed=args.seed,
                model_config=cfg,batch_size=args.batch_size,learning_rate=args.lr,weight_decay=1e-4,clip_norm=1.,
                max_source_epochs=150,budget_snapshots=list(BUDGETS),files=files,features=str(data.path),
                source_supports=1,queries_per_memory=1,query_frames=3,
                route='all' if args.scene=='collision' else 'focal',route_fixed_from_epoch1=True,
                source_target='fixed normalized frozen RGB feature784; never GT coordinates',
                common_loss='prior rollout MSE + posterior reconstruction MSE + 0.01 balanced KL',
                cross_loss='1.0 focal prior MSE' if args.method in ('Cross','Random-Cross') else None,
                random_semantics='conditional permutation within slot/type/global gravity; accidental matches allowed',
                source_history='AB through same RSSM then boundary/query' if architecture=='native' else 'AB->P64; independent query h/z',
                full_context_dim=160 if architecture=='native' else 224,
                parent=None,coordinate_labels_read=False,test_read=False)
    if args.parent_checkpoint:
        parent_path=Path(args.parent_checkpoint).resolve()
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        expected={'Structure':'Base','Cross':'Cross'}.get(args.method)
        if parent.get('version')!=M.LEGACY_VERSION or parent.get('method')!=expected or expected is None:
            raise ValueError('Legacy continuation only accepts same-objective Structure/Base or Cross; never correct-parent Random')
        if args.method=='Cross' and args.scene=='collision':raise ValueError('Old Collision focal source cannot initialize epoch1-all budget curve')
        if parent.get('scene')!=args.scene or parent.get('epoch')!=50 or parent.get('next_epoch')!=51 or parent.get('next_batch')!=0:
            raise ValueError('Expected the exact matching legacy source50 end-of-epoch checkpoint')
        if parent.get('test_read') is not False or not parent.get('optimizer') or len(parent.get('history',[]))!=50:
            raise ValueError('Legacy parent lacks complete optimizer/history')
        if parent['binding']['learning_rate']!=args.lr or parent['binding']['batch_size']!=args.batch_size:
            raise ValueError('Legacy optimizer/batch budget differs')
        if parent['binding'].get('seed')!=args.seed:
            raise ValueError('Legacy continuation must retain the source seed')
        if Path(parent['binding']['features']).resolve()!=data.path:
            raise ValueError('Legacy continuation must retain the exact committed feature root')
        parent_files={str(Path(p).resolve()):h for p,h in parent['binding']['files'].items()}
        for file,expected_hash in data.files.items():
            if parent_files.get(str(Path(file).resolve()))!=expected_hash:
                raise ValueError('Current feature manifest/IDs differ from the legacy parent: '+file)
        index_path=str(Path(args.relation_index).resolve());index_sha=digest(index_path)
        if parent_files.get(index_path)!=index_sha:
            raise ValueError('Current RelationIndex differs from the legacy parent')
        for file,expected_hash in parent['binding']['files'].items():
            if digest(file)!=expected_hash:raise ValueError('Legacy bound artifact changed: '+file)
        result['parent']=dict(path=str(parent_path),sha256=digest(parent_path),version=parent['version'],
                              method=parent['method'],epoch=50,binding_sha256=parent['binding_sha256'],
                              seed=args.seed,features=str(data.path),feature_files_sha256=dict(data.files),
                              relation_index=index_path,relation_index_sha256=index_sha,
                              cuda_index=args.parent_device_index,reason='same exact structured computation and objective; extend budget')
    result['sha256']=hashlib.sha256(canonical(result).encode()).hexdigest();return result

def setup(args,create=False):
    torch.set_num_threads(args.threads)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=D.FeatureData(args.features,args.scene)
    if create:normalization(data,out)
    binding=build_binding(args,data)
    if not create and read(out/'binding.json')!=binding:raise ValueError('Source run binding changed')
    seeded(args.seed);model=M.make_model(config=binding['model_config']).to(args.device)
    with np.load(out/'normalization.npz') as z:model.set_normalization(z['mean'],z['scale'])
    planner=D.SourcePlanner(data,args.relation_index,args.seed)
    return data,binding,model,planner

def prepare(args):
    data,binding,model,planner=setup(args,True);out=Path(args.out)
    if (out/'binding.json').exists() and read(out/'binding.json')!=binding:raise ValueError('Use independent output for a changed source contract')
    write(out/'binding.json',binding)
    init=dict(version=VERSION,method=args.method,state_sha256=state_sha(model),
              trainable_parameters=sum(p.numel() for p in model.parameters()),context_dim=model.context_dim,
              has_p_tower=hasattr(model,'history_output'),test_read=False)
    if args.method=='Native' and (init['has_p_tower'] or hasattr(model,'history')):raise ValueError('Native unexpectedly has an A history tower')
    if (out/'initialization.json').exists() and read(out/'initialization.json')!=init:raise ValueError('Changed initialization')
    write(out/'initialization.json',init)
    plan=planner.make(1,args.batch_size)
    record={k:v for k,v in plan.items() if k in ('route','plan_sha256','query_exposures','paired','random_same','randomization_skipped','common_cross_recipients','extra_random_same')}
    write(out/'plan_epoch1.json',record)
    write(out/'prepared.json',dict(status='COMPLETE',version=VERSION,binding_sha256=binding['sha256'],initialization=init,test_read=False))
    emit('PREPARED',method=args.method,scene=args.scene,**record)


def objective(model,data,plan,indices,method):
    device=next(model.parameters()).device
    ab,am,c,cm=data.context('train',indices,device)
    full,mask_cd=data.target('train',indices,device)
    target=model.target(full);active=cm.any(1);mask=mask_cd[:,3:]&active[:,None]
    own=None
    if method=='Native':
        prefix,active=model.source_prefix(ab,am,c,cm,sample=True)
        prior=model.imagine(prefix,active,data.frames-3,sample=True)
        reconstructed,kl,raw=model.observe_future(prefix,active,full,mask_cd,sample=True)
    else:
        own=model.encode(ab,am);prefix,active=model.filter_prefix(c,cm,sample=True)
        prior=model.imagine(own,prefix,active,data.frames-3,sample=True)
        reconstructed,kl,raw=model.observe_future(own,prefix,active,full,mask_cd,sample=True)
    rollout=mse(prior,target,mask);reconstruction=mse(reconstructed,target,mask);kl_loss=mean_mask(kl,mask)
    common=rollout+reconstruction+model.config.kl_weight*kl_loss
    cross=common*0;paired=0;donor_vectors=None
    if method in ('Cross','Random-Cross'):
        local=np.flatnonzero((plan['common'] if plan['route']=='all' else plan['focal']>=0)[indices])
        rows=torch.as_tensor(local,device=device);focal=torch.as_tensor(plan['focal'][indices[local]],device=device)
        if len(rows):
            if plan['route']=='all':
                external=plan['external_random' if method=='Random-Cross' else 'external_correct'][indices[local]]
                rr,ss=np.where(external>=0);donors=external[rr,ss]
                da,dm=data.history('train',donors,device);dp=model.encode(da,dm)
                ri=torch.as_tensor(rr,device=device);si=torch.as_tensor(ss,device=device)
                donor_vectors=dp[torch.arange(len(dp),device=device),si]
                legal=dm.any(1)[torch.arange(len(dp),device=device),si]
                if not legal.all():raise ValueError('All-route paired donor became visually absent')
                mixed=own[rows].clone();mixed[ri,si]=donor_vectors
                if not torch.equal(torch.as_tensor(external>=0,device=device),active[rows]):raise ValueError('All active P must be external')
            else:
                donors=plan['random' if method=='Random-Cross' else 'correct'][indices[local]]
                da,dm=data.history('train',donors,device);dp=model.encode(da,dm)
                donor_vectors=dp[torch.arange(len(dp),device=device),focal]
                legal=am[rows].any(1)[torch.arange(len(rows),device=device),focal]&active[rows,focal]&dm.any(1)[torch.arange(len(rows),device=device),focal]
                rows,focal,donor_vectors=rows[legal],focal[legal],donor_vectors[legal]
                mixed=own[rows].clone();mixed[torch.arange(len(rows),device=device),focal]=donor_vectors
            if len(rows):
                prediction=model.imagine(mixed,{k:v[rows] for k,v in prefix.items()},active[rows],data.frames-3,sample=True)
                ri=torch.arange(len(rows),device=device)
                cross=mse(prediction[ri,:,focal],target[rows,:,focal],mask[rows,:,focal]);paired=len(rows)
    loss=common+model.config.lambda_cross*cross
    metrics=dict(loss=float(loss.detach()),self_loss=float(common.detach()),prior_rollout_mse=float(rollout.detach()),
                 posterior_reconstruction_mse=float(reconstruction.detach()),raw_kl_nats=float(mean_mask(raw,mask).detach()),
                 free_kl_nats=float(kl_loss.detach()),cross=float(cross.detach()),paired=paired,
                 h_std=float(prefix['h'].detach().float().flatten(0,1).std(0).mean()),
                 z_std=float(prefix['z'].detach().float().flatten(0,1).std(0).mean()))
    return loss,metrics,dict(own=own,donor_vectors=donor_vectors,terms={'common':common,'cross':cross})

@torch.no_grad()
def evaluate(model,data,batch_size=32,limit=512):
    model.eval();device=next(model.parameters()).device;values=[];ids=[]
    for first in range(0,min(limit,len(data.ids['val'])),batch_size):
        ix=np.arange(first,min(first+batch_size,limit,len(data.ids['val'])))
        ab,am,c,cm=data.context('val',ix,device)
        if model.config.architecture=='native':pred=model.predict_observed(ab,am,c,cm,data.frames-3,False)
        else:pred=model.predict(model.encode(ab,am),c,cm,data.frames-3,False)
        full,mask_cd=data.target('val',ix,device);mask=mask_cd[:,3:]&cm.any(1)[:,None]
        error=(pred.float()-model.target(full)).square().mean(-1)
        count=mask.sum((1,2));score=(error*mask).sum((1,2))/count.clamp_min(1)
        for i,v,n in zip(ix,score.tolist(),count.tolist()):
            if n:ids.append(data.ids['val'][i]);values.append(v)
    return dict(mse=float(np.mean(values)),ids=ids,per_recipient_mse=values,future_posterior_calls=0,test_read=False)

@torch.no_grad()
def diagnostic(model,data):
    model.eval();device=next(model.parameters()).device
    ix=sorted(range(len(data.ids['train'])),key=lambda i:hashlib.sha256(('rssm-v7-delta/'+data.ids['train'][i]).encode()).hexdigest())[:16]
    ab,am,c,cm=data.context('train',np.asarray(ix),device)
    real=model.encode_joint(ab,am,c,cm).float()
    forward=model.project.forward
    try:
        model.project.forward=lambda features,zero_delta=False:forward(features,zero_delta=True)
        zero=model.encode_joint(ab,am,c,cm).float()
    finally:model.project.forward=forward
    active=cm.any(1);a=real[active];z=zero[active]
    numerator=float((a-z).square().mean());denominator=float(a.square().mean())
    return dict(split='fixed_train16',ids=[data.ids['train'][i] for i in ix],context_dim=model.context_dim,
                context_std=float(a.std(0).mean()),delta_zero_numerator=numerator,delta_zero_denominator=denominator,
                relative_delta_response=numerator/max(denominator,1e-12),
                raw_ab_delta_energy=float((ab[:,1:]-ab[:,:-1]).square().mean()),
                target_mode='fixed feature784, no learned target collapse',test_read=False)


def smoke(args):
    data,binding,model,planner=setup(args);out=Path(args.out);plan=planner.make(1,args.batch_size);ix=plan['batches'][0]
    seeded(args.seed);model.train();loss,metrics,details=objective(model,data,plan,ix,args.method)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real RSSM batch')
    if args.method in ('Cross','Random-Cross') and metrics['paired']==0:raise ValueError('Smoke needs a nontrivial relation batch')
    donor_gradient=None
    if args.method in ('Cross','Random-Cross'):
        grad=torch.autograd.grad(loss,details['donor_vectors'],retain_graph=True,allow_unused=True)[0]
        if grad is None or not torch.isfinite(grad).all() or float(grad.norm())<=0:raise ValueError('Missing live external-history gradient')
        donor_gradient=float(grad.norm())
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not float(norm)>0:raise ValueError('No RSSM gradient')
    model.eval();ab,am,c,cm=data.context('train',ix[:4],args.device)
    with torch.no_grad():
        state=model.encode_history_state(ab,am);tokens=model.encode_current_tokens(c,cm)
        direct=model.encode_joint(ab,am,c,cm);cached=model.encode_joint_from_cached(state,tokens,cm)
        torch.testing.assert_close(direct,cached,atol=2e-5,rtol=2e-5)
        null=model.encode_joint_from_cached({k:torch.zeros_like(v) for k,v in state.items()},tokens,cm)
    result=dict(status='PASS',version=VERSION,binding_sha256=binding['sha256'],method=args.method,scene=args.scene,
                rows=len(ix),metrics=metrics,gradient_norm=float(norm),donor_gradient_norm=donor_gradient,context_dim=model.context_dim,
                cache_max_error=float((direct-cached).abs().max()),null_finite=bool(torch.isfinite(null).all()),
                context_uses_only_observed_AB_query=True,cache_shapes={k:list(v.shape) for k,v in state.items()},
                native_has_p_tower=hasattr(model,'history_output') if args.method=='Native' else None,
                weights_discarded_after_smoke=True,optimizer_steps=0,test_read=False)
    if not result['null_finite']:raise FloatingPointError('Nonfinite Null context')
    write(out/'real_batch_smoke.json',result);emit('SMOKE_PASS',**result)


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def train(args):
    out=Path(args.out);device=torch.device(args.device)
    with ExitStack() as stack:
        stack.enter_context(lock(out/'worker.lock'))
        if device.type=='cuda':stack.enter_context(lock(out.parent/'.rssm_gpu_locks'/(socket.gethostname()+'_'+str(device).replace(':','_')+'.lock')))
        data,binding,model,planner=setup(args)
        if tuple(len(data.ids[s]) for s in ('train','val'))!={'balls':(7000,2000),'collision':(14000,4000),'blocktower':(28310,8088)}[args.scene]:
            raise ValueError('Full committed scene required for source training')
        smoke_result=read(out/'real_batch_smoke.json')
        if smoke_result.get('status')!='PASS' or smoke_result['binding_sha256']!=binding['sha256']:raise ValueError('Matching real smoke required')
        initial=read(out/'initialization.json')
        if state_sha(model)!=initial['state_sha256']:raise ValueError('Initialization changed')
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        epoch=1;batch_index=0;step=0;history=[];running={};prior_seconds=0.;latest=out/'latest.pt'
        imported=False
        if latest.exists():
            if not args.resume:raise ValueError('Existing run; use --resume')
            ck=torch.load(latest,map_location='cpu',weights_only=False)
            if ck['binding_sha256']!=binding['sha256'] or ck['method']!=args.method:raise ValueError('Wrong resume parent')
        elif binding['parent']:
            ck=torch.load(binding['parent']['path'],map_location='cpu',weights_only=False);imported=True
        else:ck=None
        if ck:
            model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
            epoch=ck['next_epoch'];batch_index=ck['next_batch'];step=ck['step'];history=ck['history'];running=ck.get('running',{})
            prior_seconds=ck.get('seconds',0.);restore_rng(ck['rng'],device,args.parent_device_index if imported else None)
        began=time.monotonic()
        def checkpoint():
            return dict(version=VERSION,model=model.state_dict(),model_config=model.artifact_config(),optimizer=optimizer.state_dict(),rng=rng_state(device),
                        method=args.method,family='RSSM',scene=args.scene,epoch=epoch-1 if batch_index==0 else epoch,next_epoch=epoch,next_batch=batch_index,
                        step=step,history=history,running=running,seconds=prior_seconds+time.monotonic()-began,
                        binding=binding,binding_sha256=binding['sha256'],initialization_sha256=initial['state_sha256'],test_read=False)
        if imported:
            # Publish an exact reusable budget50 under the explicit parent chain.
            path=out/'checkpoint_50.pt';save(path,checkpoint());save(latest,checkpoint())
            done=dict(status='COMPLETE',version=VERSION,scene=args.scene,family='RSSM',method=args.method,
                      epochs=50,steps=step,checkpoint=str(path),checkpoint_sha256=digest(path),
                      binding_sha256=binding['sha256'],context_dim=model.context_dim,
                      initialization_sha256=initial['state_sha256'],seconds=prior_seconds,
                      reused_legacy_source=True,parent_checkpoint=binding['parent'],
                      selection='fixed source epoch',coordinate_labels_read=False,test_read=False)
            write(out/'complete_50.json',done);write(out/'complete.json',done)
        if epoch>args.epochs:
            emit('ALREADY_REACHED_BUDGET',epoch=epoch-1,requested=args.epochs,step=step);return
        if args.max_steps and step>=args.max_steps:
            emit('CALIBRATION_ALREADY_REACHED',step=step,resume_hint='remove --max-steps');return
        write(out/'worker.json',dict(status='RUNNING',host=socket.gethostname(),pid=os.getpid(),device=str(device),method=args.method,target_epoch=args.epochs,step=step,test_read=False))
        try:
            if step==0:write(out/'diagnostic_step_0.json',dict(step=0,**diagnostic(model,data)))
            while epoch<=args.epochs:
                plan=planner.make(epoch,args.batch_size)
                if batch_index==0:running=dict(examples=0,steps=0,weighted={},plan_sha256=plan['plan_sha256'],started=time.time())
                elif running['plan_sha256']!=plan['plan_sha256']:raise ValueError('Resume changed paired plan')
                model.train()
                for at in range(batch_index,len(plan['batches'])):
                    ix=plan['batches'][at];optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        loss,metrics,details=objective(model,data,plan,ix,args.method)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RSSM loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    del loss,details
                    step+=1;batch_index=at+1;running['examples']+=len(ix);running['steps']+=1
                    for key,value in metrics.items():running['weighted'][key]=running['weighted'].get(key,0.)+value*len(ix)
                    if step in (1,50,200,500):
                        write(out/f'diagnostic_step_{step}.json',dict(step=step,metrics=metrics,**diagnostic(model,data)));model.train()
                    if step%50==0 or batch_index==len(plan['batches']):
                        save(latest,checkpoint());write(out/'progress.json',dict(status='RUNNING',epoch=epoch,batch=batch_index,step=step,history=history,latest=metrics,grad_norm=float(norm),test_read=False))
                        emit('TRAIN_PROGRESS',epoch=epoch,batch=batch_index,step=step,method=args.method,**metrics)
                    if args.max_steps and step>=args.max_steps:
                        save(latest,checkpoint())
                        result=dict(status='CALIBRATION_COMPLETE',step=step,epoch=epoch,full_source_complete=False,binding_sha256=binding['sha256'],test_read=False)
                        write(out/'calibration_complete.json',result);write(out/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('CALIBRATION_COMPLETE',**result);return
                if running['examples']!=len(data.ids['train']):raise ValueError('Incomplete source epoch')
                validation=evaluate(model,data,args.batch_size)
                row=dict(epoch=epoch,mse=validation['mse'],train={k:v/running['examples'] for k,v in running['weighted'].items()},
                         seconds=time.time()-running['started'],plan_sha256=plan['plan_sha256'],query_exposures=running['examples'],
                         paired=plan['paired'],random_same=plan['random_same'],route=plan['route'])
                history.append(row);completed=epoch;epoch+=1;batch_index=0;running={}
                save(latest,checkpoint())
                if completed in BUDGETS:
                    path=out/f'checkpoint_{completed}.pt';save(path,checkpoint())
                    write(out/f'validation_{completed}.json',dict(epoch=completed,**validation))
                    done=dict(status='COMPLETE',version=VERSION,scene=args.scene,family='RSSM',method=args.method,epochs=completed,steps=step,
                              checkpoint=str(path),checkpoint_sha256=digest(path),binding_sha256=binding['sha256'],context_dim=model.context_dim,
                              initialization_sha256=initial['state_sha256'],seconds=prior_seconds+time.monotonic()-began,
                              selection='fixed source epoch',coordinate_labels_read=False,test_read=False)
                    write(out/f'complete_{completed}.json',done);write(out/'complete.json',done)
                write(out/'progress.json',dict(status='RUNNING' if completed<args.epochs else 'COMPLETE',epoch=completed,step=step,history=history,test_read=False))
                emit('EPOCH_COMPLETE',method=args.method,**row)
            write(out/'worker.json',dict(status='COMPLETE',epochs=epoch-1,steps=step,pid=os.getpid(),exit_code=0,test_read=False))
            emit('TRAIN_COMPLETE',epochs=epoch-1,steps=step,method=args.method)
        except Exception as error:
            failed=dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),epoch=epoch,batch=batch_index,step=step,test_read=False)
            write(out/'failure.json',failed);write(out/'worker.json',dict(failed,pid=os.getpid(),exit_code=1));raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--scene',required=True,choices=tuple(D.SPECS));p.add_argument('--method',required=True,choices=METHODS)
    for name in ('features','relation-index','out','protocol'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=50,choices=BUDGETS)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--parent-checkpoint');p.add_argument('--parent-device-index',type=int)
    return p
if __name__=='__main__':
    args=parser().parse_args()
    if args.seed!=0 or args.batch_size!=32 or args.lr!=.0003 or args.max_steps<0:raise ValueError('Fixed seed0, batch32, lr3e-4')
    globals()[args.command](args)
