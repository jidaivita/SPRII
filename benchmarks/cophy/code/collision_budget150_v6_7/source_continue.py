"""Isolated Collision source100 -> source150 continuations, with old objectives.

No old trainer/checkpoint/result is modified. The old model-format version is
retained; this wrapper's experiment provenance is recorded separately.
"""
import argparse
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

VERSION = 'collision-source-budget150-v6.7-1'
MONO_VERSION = 'cophy-monolithic-jepa-v6.6'
SPLIT_VERSION = 'cophy-latent-v6.2-sig02'
METHODS = ('Monolithic', 'Base', 'Cross-all')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)


def immutable(p,value):
    if Path(p).exists() and read(p)!=value:raise ValueError('Changed frozen artifact: '+str(p))
    write(p,value)


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));torch.save(value,tmp);os.replace(tmp,p)


def emit(event,**kw):print(json.dumps(dict(event=event,time=time.time(),**kw),allow_nan=False),flush=True)


def load_module(alias,path):
    path=Path(path).resolve()
    if alias in sys.modules:
        module=sys.modules[alias]
        if Path(module.__file__).resolve()!=path:raise ValueError('Conflicting imported module '+alias)
        return module
    spec=importlib.util.spec_from_file_location(alias,path);module=importlib.util.module_from_spec(spec)
    sys.modules[alias]=module;spec.loader.exec_module(module);return module


def tree_digest(value):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
        elif isinstance(x,(tuple,list)):
            for y in x:add(y)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()


def verify_binding(ck):
    binding=ck['binding'];body={k:v for k,v in binding.items() if k!='sha256'}
    value=hashlib.sha256(canonical(body).encode()).hexdigest()
    if value!=ck['binding_sha256'] or value!=binding['sha256']:raise ValueError('Parent binding digest differs')
    return binding


def runtime(args):
    global np,torch
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    old=Path(args.runtime_dir).resolve();trainer=Path(args.parent_trainer).resolve()
    if args.method=='Monolithic':
        if trainer!=old/'train.py':raise ValueError('Monolithic parent trainer must be the bound model runtime train.py')
        core=load_module('_collision_budget150_mono_core',trainer)
        if core.VERSION!=MONO_VERSION:raise ValueError('Wrong original Monolithic runtime')
        return core,None
    if 'models' in sys.modules and Path(sys.modules['models'].__file__).resolve()!=old/'models.py':
        raise ValueError('A different split model runtime is already imported')
    sys.path.insert(0,str(old))
    core=load_module('_collision_budget150_sig02_core',old/'train.py')
    if core.VERSION!=SPLIT_VERSION:raise ValueError('Wrong original SIG02 runtime')
    route=None
    if args.method=='Cross-all':
        route=load_module('_collision_budget150_original_route',trainer)
        # The old module's globals/version are retained: VERSION is part of the
        # nonfocal donor RNG seed. Never replace it with this wrapper VERSION.
        route.runtime(args)
        if route.CORE_VERSION!=SPLIT_VERSION or route.VERSION!='collision-cross-source-routing-v6.4-1':
            raise ValueError('Wrong registered all-external routing implementation')
    return core,route


def setup(args,core,route):
    source=Path(args.parent_checkpoint).resolve();out=Path(args.out).resolve()
    if out==source.parent or out in source.parents:raise ValueError('New output may not overwrite an old source directory')
    expected_version=MONO_VERSION if args.method=='Monolithic' else SPLIT_VERSION
    expected_method='Cross' if args.method=='Cross-all' else args.method
    done=read(source.parent/'complete.json')
    if (done.get('status'),done.get('scene'),done.get('method'),done.get('epochs'),done.get('steps'))!=('COMPLETE','collision',expected_method,100,43800):
        raise ValueError('Parent must be this completed Collision source100, step43800')
    if done.get('checkpoint_sha256')!=sha(source) or done.get('test_read') is not False:
        raise ValueError('Parent complete/checkpoint hash or test status differs')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    if (ck.get('version'),ck.get('scene'),ck.get('method'),ck.get('family'),ck.get('epoch'),ck.get('next_epoch'),ck.get('next_batch'),ck.get('step'))!=(expected_version,'collision',expected_method,'JEPA',100,101,0,43800):
        raise ValueError('Parent must be the exact end-of-epoch100 model/optimizer/RNG checkpoint')
    if len(ck.get('history',[]))!=100 or ck.get('test_read') is not False:raise ValueError('Incomplete parent history/test status')
    if len(ck.get('rng',{}).get('cuda',[]))!=1 or ck.get('rng_device_index')!=0:
        raise ValueError('Expected the single saved active-device CUDA RNG at index0; do not use old physical GPU indices')
    if args.method=='Cross-all' and (ck.get('route')!='all' or done.get('route')!='all'):
        raise ValueError('Cross parent is not the all-external route')
    binding100=verify_binding(ck)
    dependencies={str(source):sha(source),str(source.parent/'complete.json'):sha(source.parent/'complete.json')}
    if args.method=='Monolithic':
        config=binding100;dependencies.update(config['files'])
        if config['files'].get(str(Path(args.parent_trainer).resolve()))!=sha(args.parent_trainer):
            raise ValueError('Monolithic parent trainer hash differs')
    else:
        if binding100['implementation_sha256']!=sha(args.parent_trainer):raise ValueError('Parent continuation implementation changed')
        ancestor=Path(binding100['source_checkpoint'])
        if sha(ancestor)!=binding100['source_checkpoint_sha256']:raise ValueError('Bound source50 ancestor changed')
        ck50=torch.load(ancestor,map_location='cpu',weights_only=False);config=verify_binding(ck50)
        if ck50['binding_sha256']!=binding100['parent_binding_sha256'] or ck50['epoch']!=50:
            raise ValueError('Wrong source50 ancestry/configuration')
        if ck50['model_config']!=ck['model_config']:raise ValueError('Source100 architecture/config differs from source50')
        dependencies.update(binding100['parent_dependencies'])
        dependencies[str(ancestor)]=sha(ancestor);dependencies[str(Path(args.parent_trainer).resolve())]=sha(args.parent_trainer)
        if config['files']!=binding100['parent_dependencies']:raise ValueError('Source100 inherited data/runtime binding differs')
    for path,expected in dependencies.items():
        if sha(path)!=expected:raise ValueError('Changed parent dependency: '+path)
    for filename in ('train.py','models.py'):
        values=[v for p,v in config['files'].items() if Path(p).name==filename]
        if values!=[sha(Path(args.runtime_dir)/filename)]:raise ValueError('Runtime is not the original hash-bound '+filename)
    if (config['batch_size'],config['learning_rate'],config['seed'],config['weight_decay'],config['clip_norm'])!=(32,.0003,0,1e-4,1.):
        raise ValueError('Unexpected inherited optimizer/sampling settings')
    if ck['model_config']['sigreg_weight']!=.2:raise ValueError('Expected inherited SIGReg0.2')
    groups=ck.get('optimizer',{}).get('param_groups',[])
    if not groups or any(g['lr']!=.0003 or g['weight_decay']!=1e-4 for g in groups):
        raise ValueError('Missing optimizer or changed effective learning rate/weight decay')
    optimizer_steps=[int(value['step'].item()) if torch.is_tensor(value.get('step')) else int(value['step'])
                     for value in ck['optimizer']['state'].values() if 'step' in value]
    if not optimizer_steps or any(value!=43800 for value in optimizer_steps):
        raise ValueError('Parent optimizer moments are not at the source100 update count')
    data=core.FeatureData(config['features'],'collision')
    if len(data.ids['train'])!=14000:raise ValueError('Unexpected recipient exposure')
    if args.method=='Monolithic':
        planner=core.EpochPlanner(data,config['seed']);relation_path=None
    else:
        possible=[p for p in config['files'] if Path(p).suffix=='.json' and 'relation' in Path(p).name.lower()
                  and read(p).get('version')=='cophy-relation-index-v3']
        if len(possible)!=1:raise ValueError('Original bound relation index is not unique')
        relation_path=possible[0];original=core.EpochPlanner(data,relation_path,config['seed'])
        planner=route.RoutePlanner(data,original,config['seed']) if route else original
    body=dict(version=VERSION,model_version=expected_version,method=args.method,source_method=expected_method,
        scene='collision',parent_checkpoint=str(source),parent_checkpoint_sha256=sha(source),
        parent_binding_sha256=ck['binding_sha256'],parent_dependencies=dependencies,
        parent_model_sha256=tree_digest(ck['model']),parent_optimizer_sha256=tree_digest(ck['optimizer']),
        parent_rng_sha256=tree_digest(ck['rng']),parent_initialization_sha256=ck['initialization_sha256'],
        parent_trainer=str(Path(args.parent_trainer).resolve()),parent_trainer_sha256=sha(args.parent_trainer),
        runtime_dir=str(Path(args.runtime_dir).resolve()),implementation_sha256=sha(__file__),
        protocol_sha256=sha(args.protocol) if args.protocol else None,
        start_epoch=100,end_epoch=150,additional_epochs=50,parent_steps=43800,expected_final_steps=65700,
        batch_size=32,microbatch=ck['microbatch'],learning_rate=.0003,weight_decay=1e-4,clip_norm=1.,
        seed=config['seed'],features=config['features'],relation_index=relation_path,model_config=ck['model_config'],
        route_sampler_version=route.VERSION if route else None,query_frames=3,history_frames=15,
        source_selection='fixed source150; old source50/source100 remain untouched',
        objective='original self + SIGReg0.2' if not route else 'original self + SIGReg0.2 + original lambda_cross * focal-only Cross; all active P external; no Align',
        loss_implementation='old mono.batch_objective' if args.method=='Monolithic' else ('old sig02.batch_objective(method=Base)' if args.method=='Base' else 'old source_route.objective(route=all)'),
        sampler='unchanged original implementation at global epochs101..150; not restarted at epoch1',
        test_read=False,coordinate_labels_read=False)
    body['sha256']=hashlib.sha256(canonical(body).encode()).hexdigest()
    return ck,data,planner,body


def plan_digest(plan):return plan.get('plan_sha256',plan.get('order_sha256'))


def prepare(args,core,route):
    parent,data,planner,binding=setup(args,core,route);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    immutable(out/'binding.json',binding);plans=[]
    for epoch in range(101,151):
        plan=planner.make(epoch,32);order=np.concatenate(plan['batches'])
        if len(plan['batches'])!=438 or not np.array_equal(np.sort(order),np.arange(14000)):
            raise ValueError('Every additional epoch must expose all14000 queries in438 updates')
        row=dict(epoch=epoch,plan_sha256=plan_digest(plan),recipients=len(order),steps=438)
        if route:row.update(common_cross_recipients=plan['common_cross_recipients'],
            original_plan_sha256=plan['original_plan_sha256'],additional_nonfocal_objects=plan['additional_nonfocal_objects'])
        plans.append(row)
    receipt=dict(status='COMPLETE',version=VERSION,binding_sha256=binding['sha256'],epochs=plans,test_read=False)
    if route:receipt['coverage']=planner.coverage
    immutable(out/'plans.json',receipt)
    result=dict(status='COMPLETE',version=VERSION,method=args.method,binding_sha256=binding['sha256'],
        plans_sha256=sha(out/'plans.json'),additional_epochs=50,additional_steps=21900,expected_final_steps=65700,
        parent_model_unchanged=True,optimizer_steps=0,test_read=False)
    immutable(out/'prepared.json',result);emit('PREPARED',**result)


def checked_setup(args,core,route):
    parent,data,planner,binding=setup(args,core,route);out=Path(args.out)
    if read(out/'binding.json')!=binding or read(out/'prepared.json')['binding_sha256']!=binding['sha256']:
        raise ValueError('Prepare the exact new continuation binding first')
    if read(out/'prepared.json')['plans_sha256']!=sha(out/'plans.json'):raise ValueError('Changed continuation plans')
    return parent,data,planner,binding


def checked_plan(args,planner,epoch):
    plan=planner.make(epoch,32);expected=read(Path(args.out)/'plans.json')['epochs'][epoch-101]
    if expected['plan_sha256']!=plan_digest(plan):raise ValueError('Changed epoch continuation sampling')
    return plan


def restore_rng(value,device):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if device.type=='cuda':
        if len(value['cuda'])!=1:raise ValueError('Expected one active-device RNG state')
        torch.cuda.set_rng_state(value['cuda'][0],device=device)


def capture_rng(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=[torch.cuda.get_rng_state(device)] if device.type=='cuda' else [])


def objective(args,core,route,model,data,plan,indices,microbatch):
    if args.method=='Monolithic':return core.batch_objective(model,data,indices,microbatch)
    if args.method=='Base':
        loss,metrics,detail=core.batch_objective(model,data,plan,indices,'Base',microbatch)
        if metrics['cross']!=0. or metrics['align']!=0. or detail['donor_p'] is not None:
            raise ValueError('Base activated a donor/relation path')
    else:loss,metrics,detail=route.objective(core,model,data,plan,indices,'all',microbatch)
    del detail
    return loss,metrics


def device_for(args):
    device=torch.device(args.device)
    if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
    return device


def restored_model(parent,core,device):
    model=core.make_model(config=parent['model_config']).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0003,weight_decay=1e-4)
    model.load_state_dict(parent['model']);optimizer.load_state_dict(parent['optimizer'])
    restore_rng(parent['rng'],device)
    return model,optimizer


def smoke(args,core,route):
    parent,data,planner,binding=checked_setup(args,core,route);device=device_for(args)
    model,optimizer=restored_model(parent,core,device);model.train()
    if tree_digest(model.state_dict())!=binding['parent_model_sha256'] or tree_digest(optimizer.state_dict())!=binding['parent_optimizer_sha256']:
        raise ValueError('Smoke did not restore parent model/optimizer exactly')
    plan=checked_plan(args,planner,101);indices=plan['batches'][0]
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        loss,metrics=objective(args,core,route,model,data,plan,indices,parent['microbatch'])
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real first continuation loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if tree_digest(model.state_dict())!=binding['parent_model_sha256'] or tree_digest(optimizer.state_dict())!=binding['parent_optimizer_sha256']:
        raise ValueError('Smoke modified parent state')
    result=dict(status='PASS',version=VERSION,method=args.method,binding_sha256=binding['sha256'],
        epoch=101,plan_sha256=plan_digest(plan),metrics=metrics,gradient_norm=float(norm),
        model_unchanged=True,optimizer_unchanged=True,optimizer_steps=0,active_cuda_rng_index=0,test_read=False)
    write(Path(args.out)/'smoke.json',result);emit('SMOKE_PASS',**result)


def train(args,core,route):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'owner.lock','a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Continuation already running') from exc
        parent,data,planner,binding=checked_setup(args,core,route)
        receipt=read(out/'smoke.json')
        if receipt.get('status')!='PASS' or receipt['binding_sha256']!=binding['sha256']:raise ValueError('Matching real smoke required')
        if (out/'complete.json').exists():
            done=read(out/'complete.json')
            if done.get('binding_sha256')!=binding['sha256'] or done.get('epochs')!=150:raise ValueError('Different completed continuation')
            emit('ALREADY_COMPLETE',**done);return
        latest=out/'latest.pt';old=parent
        if latest.exists():
            if not args.resume:raise ValueError('Existing continuation requires --resume')
            old=torch.load(latest,map_location='cpu',weights_only=False)
            if old.get('extension_binding_sha256')!=binding['sha256'] or old.get('budget_method')!=args.method:
                raise ValueError('Different resumed continuation checkpoint')
        device=device_for(args);model,optimizer=restored_model(old,core,device)
        if not latest.exists() and (tree_digest(model.state_dict())!=binding['parent_model_sha256']
                or tree_digest(optimizer.state_dict())!=binding['parent_optimizer_sha256']):
            raise ValueError('Parent optimizer/model restore differs')
        epoch,next_batch,step=old['next_epoch'],old['next_batch'],old['step']
        history=list(old['history']);running=dict(old.get('running',{}));microbatch=old['microbatch']
        prior_seconds=old.get('additional_seconds',0.) if latest.exists() else 0.;started=time.time()
        if not latest.exists():write(out/'fork.json',dict(status='COMPLETE',version=VERSION,binding_sha256=binding['sha256'],
            parent_model_sha256=binding['parent_model_sha256'],parent_optimizer_sha256=binding['parent_optimizer_sha256'],
            parent_rng_sha256=binding['parent_rng_sha256'],active_cuda_rng_index=0,device=str(device),epoch=100,step=43800,test_read=False))
        def record(ne,nb):
            value=dict(version=binding['model_version'],experiment_version=VERSION,budget_method=args.method,
                method=binding['source_method'],family='JEPA',scene='collision',model=model.state_dict(),
                model_config=model.artifact_config(),optimizer=optimizer.state_dict(),rng=capture_rng(device),
                rng_device_index=0 if device.type=='cuda' else None,epoch=ne-1 if nb==0 else ne,
                next_epoch=ne,next_batch=nb,step=step,history=history,running=running,microbatch=microbatch,
                binding=binding,binding_sha256=binding['sha256'],extension_binding_sha256=binding['sha256'],
                parent_checkpoint=binding['parent_checkpoint'],parent_checkpoint_sha256=binding['parent_checkpoint_sha256'],
                initialization_sha256=parent['initialization_sha256'],additional_seconds=prior_seconds+time.time()-started,
                test_read=False,coordinate_labels_read=False)
            if route:value['route']='all';value['route_sampler_version']=route.VERSION
            return value
        def publish150(validation):
            checkpoint=out/'checkpoint_150.pt';marker=out/'checkpoint_150_complete.json'
            if marker.exists():
                done=read(marker)
                if done['binding_sha256']!=binding['sha256'] or done['checkpoint_sha256']!=sha(checkpoint):
                    raise ValueError('Changed already-published source150')
                return
            if step!=65700 or len(history)!=150:raise ValueError('Source150 budget differs')
            save(checkpoint,record(151,0));write(out/'validation_150.json',dict(epoch=150,**validation))
            write(marker,dict(status='COMPLETE',version=VERSION,model_version=binding['model_version'],method=binding['source_method'],
                budget_method=args.method,route='all' if route else None,scene='collision',epoch=150,epochs=150,steps=step,additional_epochs=50,additional_steps=21900,
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),binding_sha256=binding['sha256'],
                parent_checkpoint=binding['parent_checkpoint'],parent_checkpoint_sha256=binding['parent_checkpoint_sha256'],
                selected_epoch=150,selection='fixed epoch150; source50/source100 untouched',test_read=False,coordinate_labels_read=False))
        write(out/'worker.json',dict(status='RUNNING',pid=os.getpid(),host=socket.gethostname(),device=str(device),method=args.method,
            started_at=started,binding_sha256=binding['sha256']))
        try:
            if args.max_additional_steps and step-43800>=args.max_additional_steps:
                emit('CHUNK_ALREADY_REACHED',step=step,max_additional_steps=args.max_additional_steps);return
            while epoch<=150:
                plan=checked_plan(args,planner,epoch)
                if next_batch==0:running=dict(weighted={},examples=0,steps=0,started_at=time.time(),plan_sha256=plan_digest(plan))
                elif running.get('plan_sha256')!=plan_digest(plan):raise ValueError('Resumed epoch plan differs')
                model.train()
                for batch_number in range(next_batch,len(plan['batches'])):
                    indices=plan['batches'][batch_number];optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        loss,metrics=objective(args,core,route,model,data,plan,indices,microbatch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();del loss;step+=1;next_batch=batch_number+1
                    running['examples']+=len(indices);running['steps']+=1
                    for key,value in metrics.items():running['weighted'][key]=running['weighted'].get(key,0.)+value*len(indices)
                    if step%50==0 or next_batch==len(plan['batches']):
                        save(latest,record(epoch,next_batch))
                        state=dict(status='RUNNING',method=args.method,epoch=epoch,batch=next_batch,step=step,additional_steps=step-43800,
                            history=history,latest=metrics,gradient_norm=float(norm),plan_sha256=plan_digest(plan),test_read=False)
                        write(out/'progress.json',state);emit('PROGRESS',**{k:v for k,v in state.items() if k!='history'})
                    if args.max_additional_steps and step-43800>=args.max_additional_steps:
                        save(latest,record(epoch,next_batch));result=dict(status='PAUSED',method=args.method,step=step,
                            next_epoch=epoch,next_batch=next_batch,binding_sha256=binding['sha256'],test_read=False)
                        write(out/'paused.json',result);write(out/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('PAUSED',**result);return
                validation=core.evaluate(model,data,min(microbatch,32))
                if running['examples']!=14000:raise ValueError('Incomplete recipient exposure')
                row=dict(epoch=epoch,mse=validation['mse'],train={k:v/14000 for k,v in running['weighted'].items()},
                    seconds=time.time()-running['started_at'],query_exposures=14000,plan_sha256=plan_digest(plan),
                    method=args.method,metric='live latent diagnostic only, not source selection or cross-model ranking')
                if route:row.update(paired=plan['common_cross_recipients'],original_plan_sha256=plan['original_plan_sha256'],
                    additional_nonfocal_encoded=plan['additional_nonfocal_objects'],additional_nonfocal_used=plan['additional_nonfocal_objects'])
                history.append(row);completed=epoch;epoch+=1;next_batch=0;running={}
                if completed==150:publish150(validation)
                save(latest,record(epoch,0))
                write(out/'progress.json',dict(status='RUNNING' if completed<150 else 'COMPLETE',method=args.method,
                    epoch=completed,step=step,history=history,test_read=False));emit('EPOCH_COMPLETE',**row)
            if not (out/'checkpoint_150_complete.json').exists():publish150(core.evaluate(model,data,min(microbatch,32)))
            result=dict(read(out/'checkpoint_150_complete.json'),original_100_history_preserved=history[:100]==parent['history'],
                additional_seconds=prior_seconds+time.time()-started,finished_at=time.time())
            write(out/'complete.json',result);write(out/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('COMPLETE',**result)
        except Exception as exc:
            result=dict(status='FAILED',method=args.method,error=repr(exc),traceback=traceback.format_exc(),epoch=epoch,next_batch=next_batch,
                step=step,binding_sha256=binding['sha256'],test_read=False)
            write(out/'failure.json',result);emit('FAILED',**result);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--parent-checkpoint',required=True)
    p.add_argument('--runtime-dir',required=True);p.add_argument('--parent-trainer',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,choices=(150,),default=150);p.add_argument('--device',default='cuda:0')
    p.add_argument('--protocol');p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--max-additional-steps',type=int,default=0)
    args=p.parse_args()
    if args.threads<1 or args.max_additional_steps<0:p.error('Invalid numeric argument')
    core,route=runtime(args);globals()[args.command](args,core,route)


if __name__=='__main__':main()
