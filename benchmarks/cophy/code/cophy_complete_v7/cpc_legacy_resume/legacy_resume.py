"""Continue immutable CPC v6.3 source50 with its exact old objective/sampler.

Allowed: legacy Base -> Structure in all scenes; legacy Cross -> Cross in Balls
or Blocktower. Collision Cross and every Random parent are deliberately refused.
The old model format remains readable by the new complete-context CPC loader.
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

VERSION='cophy-cpc-legacy50-budget150-v7'
MODEL_VERSION='cophy-cpc-v6.3-sig02'
REVIEWED_V7_TRAIN_SHA='3592a8e56ea5e76547a618f7fbea6d14ee99d132962bcc8473b8c0f42833ba04'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)


def immutable(p,value):
    if Path(p).exists() and read(p)!=value: raise ValueError('Changed continuation binding: '+str(p))
    write(p,value)


def emit(event,**kw): print(json.dumps(dict(event=event,time=time.time(),**kw),allow_nan=False),flush=True)


def runtime(args):
    global torch,np
    import numpy as np
    import torch
    root=Path(args.legacy_runtime_dir).resolve()
    if 'models' in sys.modules and Path(sys.modules['models'].__file__).resolve()!=root/'models.py':
        raise RuntimeError('Different models module already imported; use this standalone CLI')
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('_cophy_cpc_v7_legacy_budget_core',root/'train.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    if core.VERSION!=MODEL_VERSION: raise ValueError('Not the original CPC v6.3 runtime')
    torch.set_num_threads(args.threads)
    return core


def tree_sha(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,np.ndarray): h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for y in x: visit(y)
        else: h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def active_rng(old,index):
    cuda=old['cuda']
    if not cuda or not 0<=index<len(cuda): raise ValueError('Unavailable parent active CUDA RNG index')
    return dict(python=old['python'],numpy=old['numpy'],torch=old['torch'],cuda=[cuda[index]],rng_device_index=0)


def capture(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(device)] if device.type=='cuda' else [],rng_device_index=0)


def restore(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type!='cuda' or len(state['cuda'])!=1:
        raise ValueError('Legacy trained source continues on CUDA with one selected active RNG stream')
    torch.cuda.set_rng_state(state['cuda'][0],device)


def load_parent(args,core):
    if args.method=='Cross' and args.scene=='collision':
        raise ValueError('Collision changes focal to all; legacy Cross50 is not a constant-all parent')
    parent_method='Base' if args.method=='Structure' else 'Cross'
    path=Path(args.parent_checkpoint).resolve();complete=read(path.parent/'complete.json')
    destination=Path(args.out).resolve()
    if destination==path.parent or path.parent in destination.parents or destination in path.parents:
        raise ValueError('Continuation output must be independent of the old source tree')
    if (destination/'latest.pt').exists():
        previous=torch.load(destination/'latest.pt',map_location='cpu',weights_only=False)
        if previous.get('continuation_version')!=VERSION: raise ValueError('Output contains a different source run')
    if (complete.get('status'),complete.get('family'),complete.get('scene'),complete.get('method'),complete.get('epochs'))!=('COMPLETE','CPC',args.scene,parent_method,50):
        raise ValueError('Need the completed exact legacy CPC Base/Cross50 source')
    if complete.get('checkpoint_sha256')!=sha(path): raise ValueError('Parent checkpoint differs from completion receipt')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if (ck.get('version'),ck.get('family'),ck.get('scene'),ck.get('method'),ck.get('epoch'),ck.get('next_epoch'),ck.get('next_batch'))!=(MODEL_VERSION,'CPC',args.scene,parent_method,50,51,0):
        raise ValueError('Parent must be exact source50, next51/batch0, not a best-only checkpoint')
    if len(ck.get('history',[]))!=50 or ck.get('test_read') is not False: raise ValueError('Incomplete source history')
    old=ck['binding'];body={k:v for k,v in old.items() if k!='sha256'}
    if hashlib.sha256(canonical(body).encode()).hexdigest()!=old['sha256'] or old['sha256']!=ck['binding_sha256']:
        raise ValueError('Parent binding digest mismatch')
    if (old['epochs'],old['seed'],old['batch_size'],old['learning_rate'],old['weight_decay'],old['clip_norm'])!=(50,0,32,.0003,.0001,1.):
        raise ValueError('Unexpected parent training recipe')
    expected=dict(family='CPC',feature_dim=784,width=128,persistent_dim=64,hidden_layers=2,
                  temperature=.1,sigreg_weight=.2,lambda_cross=1.,lambda_align=.1,query_frames=3)
    if any(ck['model_config'].get(k)!=v for k,v in expected.items()): raise ValueError('Parent objective/model differs')
    # Every original dependency remains read-only and hash-identical.
    for name,digest in old['files'].items():
        if sha(name)!=digest: raise ValueError('Changed parent dependency: '+name)
    for name in ('train.py','models.py'):
        candidates=[v for k,v in old['files'].items() if Path(k).name==name]
        if candidates!=[sha(Path(args.legacy_runtime_dir)/name)]: raise ValueError('Runtime is not the parent source version')
    relation=str(Path(args.relation_index).resolve())
    if relation not in old['files']: raise ValueError('RelationIndex is not the exact parent-bound index')
    data=core.FeatureData(old['features'],args.scene);planner=core.EpochPlanner(data,relation,old['seed'])
    plan=planner.make(51,32);expected_steps=len(plan['batches'])*50
    if ck['step']!=expected_steps: raise ValueError('Parent recipient/update budget mismatch')
    if not 1<=ck.get('microbatch',0)<=32: raise ValueError('Invalid inherited microbatch')
    opt=ck.get('optimizer')
    if not opt or not opt['state']: raise ValueError('Missing trained optimizer')
    for group in opt['param_groups']:
        if group['lr']!=3e-4 or group['weight_decay']!=1e-4: raise ValueError('Unexpected optimizer learning rate/decay')
    steps={int(v['step']) for v in opt['state'].values() if 'step' in v}
    if steps!={expected_steps}: raise ValueError('Optimizer update counters differ from source50')
    if ck.get('rng_device_index') is not None and ck['rng_device_index']!=args.source_cuda_index:
        raise ValueError('Caller active CUDA RNG index conflicts with checkpoint metadata')
    chosen_rng=active_rng(ck['rng'],args.source_cuda_index)
    b=dict(version=VERSION,model_version=MODEL_VERSION,scene=args.scene,method=args.method,legacy_method=parent_method,
        parent_checkpoint=str(path),parent_checkpoint_sha256=sha(path),parent_binding_sha256=ck['binding_sha256'],
        parent_binding=old,parent_model_sha256=tree_sha(ck['model']),parent_optimizer_sha256=tree_sha(opt),
        parent_active_rng_sha256=tree_sha(chosen_rng),source_cuda_index=args.source_cuda_index,
        placement_receipt_sha256=sha(args.placement_receipt) if args.placement_receipt else None,
        implementation_sha256=sha(__file__),legacy_runtime_dir=str(Path(args.legacy_runtime_dir).resolve()),
        relation_index=relation,start_epoch=50,end_epoch=150,batch_size=32,microbatch=ck['microbatch'],
        learning_rate=3e-4,weight_decay=1e-4,clip_norm=1.,source_supports=1,queries_per_memory=1,
        route='none' if parent_method=='Base' else 'focal',route_changed=False,
        objective='exact old core.batch_objective; own InfoNCE+SIGReg.2 and optional focal Cross InfoNCE1; no Align',
        sampling='exact old EpochPlanner, public negative groups, focal eligibility and correct donor selection',
        random_policy='no Random condition accepted; never initialize Random from correct relations',
        schedule='same AdamW LR and old BF16 autocast; no scheduler; source validation still diagnostic only',
        divergence_policy='changed dependency/objective/order/microbatch fails; no fallback into v7 objective',
        model_format_preserved=True,coordinate_labels_read=False,test_read=False,
        protocol_sha256=sha(args.protocol) if args.protocol else None)
    b['sha256']=hashlib.sha256(canonical(b).encode()).hexdigest()
    return ck,chosen_rng,data,planner,b


def positive_cohort_audit(args,data,planner,b):
    """Check the exact extra v7 visibility predicate, without decoding RGB.

    If every audited donor is visible in the feature cache, the old/new focal
    Cross cohorts are identical for *all* possible plans. Otherwise exhaust the
    fixed seed0 epoch1..150 plans and report any actual dropped recipients.
    """
    new_dir=Path(args.new_runtime_dir) if args.new_runtime_dir else Path(__file__).resolve().parent.parent/'cpc'
    new_code=new_dir/'train.py';new_hash=sha(new_code) if new_code.exists() else None
    result=dict(version=VERSION,scene=args.scene,method=args.method,binding_sha256=b['sha256'],
       reviewed_new_train_sha256=REVIEWED_V7_TRAIN_SHA,actual_new_train_sha256=new_hash,
       old_rule='recipient_AB_visible AND query_active AND correct_donor_AB_visible',
       new_extra_rule='old_planned_random_donor_AB_visible',test_read=False)
    if args.method=='Structure':
        return dict(result,status='COMPLETE',positive_cohort_equivalent=True,basis='Structure has no donor objective',
                    main_source_reuse='REUSE_SOURCE')
    if new_hash!=REVIEWED_V7_TRAIN_SHA:
        return dict(result,status='NEEDS_REVIEW',positive_cohort_equivalent=None,
                    main_source_reuse='AUXILIARY_ONLY_UNTIL_NEW_PLANNER_REVIEW',reason='New runtime missing or changed')
    seen=np.asarray(data.arrays['train']['presence_ab']).any(1)
    active=np.asarray(data.arrays['train']['presence_cd'][:,:3]).any(1)
    missing=[(r['id'],r['slot']) for r in planner.index.records.values()
             if r['donor'] and not seen[planner.lookup[r['id']],r['slot']]]
    result.update(unseen_audited_donors=len(missing),unseen_examples=missing[:20])
    if not missing:
        return dict(result,status='COMPLETE',positive_cohort_equivalent=True,
            basis='All audited donor=True objects are AB-visible; added predicate is universally true',
            checked_epoch_range=[1,150],main_source_reuse='REUSE_SOURCE')
    rows=np.arange(len(planner.ids));records=[];total_dropped=0
    for epoch in range(1,151):
        p=planner.make(epoch,32);valid=p['focal']>=0;local=rows[valid];f=p['focal'][local]
        old=seen[local,f]&active[local,f]&seen[p['correct'][local],f]
        new=old&seen[p['random'][local],f];dropped=local[old&~new];total_dropped+=len(dropped)
        records.append(dict(epoch=epoch,old_eligible=int(old.sum()),new_eligible=int(new.sum()),
            dropped=len(dropped),dropped_id_examples=[planner.ids[i] for i in dropped[:10]],plan_sha256=p['plan_sha256']))
    equivalent=total_dropped==0
    return dict(result,status='COMPLETE',positive_cohort_equivalent=equivalent,checked_epoch_range=[1,150],
       total_dropped_recipient_epochs=total_dropped,epoch_records=records,
       basis='Exact seed0 registered epoch1..150 original plans evaluated against both predicates',
       main_source_reuse='REUSE_SOURCE' if equivalent else 'AUXILIARY_ONLY_NEW_MATCHED_CROSS_FROM_EPOCH1')


def make_state(core,ck,device):
    model=core.make_model(config=ck['model_config']).to(device);model.load_state_dict(ck['model'],strict=True)
    optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);optim.load_state_dict(ck['optimizer'])
    return model,optim


def anchor50(args,core,ck,state,b):
    folder=Path(args.out);path=folder/'checkpoint_50.pt'
    # Metadata wrapper only: weights and optimizer tensors are bit-identical.
    record=dict(ck,method=args.method,legacy_method=b['legacy_method'],continuation_version=VERSION,
                binding=b,binding_sha256=b['sha256'],parent_binding=ck['binding'],parent_checkpoint_sha256=b['parent_checkpoint_sha256'],
                rng=state,rng_device_index=0,source_initialization_unchanged=True)
    if not path.exists(): core.save(path,record)
    old=torch.load(path,map_location='cpu',weights_only=False)
    if (old.get('binding_sha256')!=b['sha256'] or tree_sha(old['model'])!=b['parent_model_sha256'] or
        tree_sha(old['optimizer'])!=b['parent_optimizer_sha256']): raise ValueError('Source50 anchor differs')
    immutable(folder/'checkpoint_50_complete.json',dict(status='COMPLETE',version=VERSION,method=args.method,scene=args.scene,
        epoch=50,epochs=50,steps=ck['step'],checkpoint=str(path),checkpoint_sha256=sha(path),binding_sha256=b['sha256'],
        reused_parent=True,parent_checkpoint_sha256=b['parent_checkpoint_sha256'],test_read=False))


def prepare(args):
    core=runtime(args);ck,state,data,planner,b=load_parent(args,core);folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    immutable(folder/'legacy_binding.json',b);anchor50(args,core,ck,state,b)
    p=planner.make(51,32)
    summary={k:p[k] for k in ('plan_sha256','query_exposures','paired','randomization_skipped','random_same')}
    immutable(folder/'plan_epoch51.json',summary)
    audit=positive_cohort_audit(args,data,planner,b)
    immutable(folder/'positive_cohort_equivalence.json',audit)
    write(folder/'prepared.json',dict(status='COMPLETE',version=VERSION,method=args.method,scene=args.scene,
        binding_sha256=b['sha256'],parent_steps=ck['step'],plan_epoch51=summary,objective_changed=False,
        optimizer_restored=True,active_rng_index=args.source_cuda_index,positive_cohort_equivalence=audit['positive_cohort_equivalent'],
        main_source_reuse=audit['main_source_reuse'],test_read=False))
    emit('PREPARED',method=args.method,scene=args.scene,parent_steps=ck['step'],plan=summary,
         positive_cohort_equivalence=audit['positive_cohort_equivalent'],main_source_reuse=audit['main_source_reuse'])


def smoke(args):
    core=runtime(args);ck,state,data,planner,b=load_parent(args,core)
    if read(Path(args.out)/'legacy_binding.json')!=b: raise ValueError('Prepare exact continuation first')
    if read(Path(args.out)/'positive_cohort_equivalence.json').get('positive_cohort_equivalent') is not True:
        raise ValueError('Legacy Cross is not certified to match the new positive cohort; use new matched Cross from epoch1')
    device=torch.device(args.device);model,optim=make_state(core,ck,device)
    p=planner.make(51,32);ix=p['batches'][0];records=[];gradients=[]
    before=(tree_sha(model.state_dict()),tree_sha(optim.state_dict()))
    # Repeated exact first continuation batch after restoring the same RNG;
    # no optimizer step and nothing is carried into the real train invocation.
    for repeat in range(2):
        restore(state,device);model.train();optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss,metrics,detail=core.batch_objective(model,data,p,ix,b['legacy_method'],ck['microbatch'])
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss) or norm<=0: raise ValueError('Invalid restored source batch')
        if args.method=='Structure' and (metrics['cross']!=0 or metrics['align']!=0 or detail['donor_p'] is not None):
            raise ValueError('Structure accidentally consumes donor gradient')
        if args.method=='Cross' and metrics['align']!=0: raise ValueError('Cross unexpectedly includes Align')
        records.append(metrics);gradients.append(float(norm))
    if not np.isclose(records[0]['loss'],records[1]['loss'],rtol=1e-6,atol=1e-7): raise ValueError('Restored same-batch loss differs')
    if before!=(tree_sha(model.state_dict()),tree_sha(optim.state_dict())): raise ValueError('Smoke changed weights/optimizer')
    result=dict(status='PASS',version=VERSION,binding_sha256=b['sha256'],method=args.method,scene=args.scene,
       rows=len(ix),plan_sha256=p['plan_sha256'],repeated_metrics=records,gradient_norms=gradients,
       optimizer_updates=0,weights_discarded=True,source_recipe_changed=False,test_read=False)
    write(Path(args.out)/'smoke.json',result);emit('SMOKE_PASS',**result)


def train(args):
    core=runtime(args);parent,parent_rng,data,planner,b=load_parent(args,core)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    if read(folder/'legacy_binding.json')!=b: raise ValueError('Changed continuation binding')
    if read(folder/'positive_cohort_equivalence.json').get('positive_cohort_equivalent') is not True:
        raise ValueError('Do not spend main-matrix compute on an unqualified legacy Cross continuation')
    if read(folder/'smoke.json').get('binding_sha256')!=b['sha256'] or read(folder/'smoke.json').get('status')!='PASS':
        raise ValueError('Matching restored real-batch smoke required')
    with open(folder/'owner.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=torch.device(args.device);model,optimizer=make_state(core,parent,device)
        epoch,next_batch,step,history,running=51,0,parent['step'],list(parent['history']),{}
        previous_seconds=parent.get('seconds',0.);restore(parent_rng,device)
        available=[]
        for name in ('latest.pt','checkpoint_100.pt','checkpoint_150.pt'):
            path=folder/name
            if path.exists():
                ck=torch.load(path,map_location='cpu',weights_only=False)
                if ck.get('binding_sha256')!=b['sha256'] or ck.get('continuation_version')!=VERSION: raise ValueError('Other continuation exists')
                available.append(ck)
        if available:
            if not args.resume: raise ValueError('Explicit --resume required')
            ck=max(available,key=lambda x:x['step']);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            epoch,next_batch,step=ck['next_epoch'],ck['next_batch'],ck['step'];history,running=ck['history'],ck['running']
            previous_seconds=ck['seconds'];restore(ck['rng'],device)
        started=time.monotonic()
        def record(e,nb):
            return dict(version=MODEL_VERSION,continuation_version=VERSION,method=args.method,legacy_method=b['legacy_method'],
                family='CPC',scene=args.scene,model=model.state_dict(),model_config=model.artifact_config(),optimizer=optimizer.state_dict(),
                rng=capture(device),rng_device_index=0,epoch=e-1 if nb==0 else e,next_epoch=e,next_batch=nb,step=step,
                history=history,running=running,microbatch=parent['microbatch'],binding=b,binding_sha256=b['sha256'],
                parent_checkpoint_sha256=b['parent_checkpoint_sha256'],parent_binding=parent['binding'],
                initialization_sha256=parent['initialization_sha256'],seconds=previous_seconds+time.monotonic()-started,
                objective_changed=False,route_changed=False,test_read=False)
        def publish(n,ck):
            path=folder/f'checkpoint_{n}.pt'
            if not path.exists():core.save(path,ck)
            old=torch.load(path,map_location='cpu',weights_only=False)
            if old['binding_sha256']!=b['sha256'] or old['next_epoch']!=n+1 or old['next_batch']!=0: raise ValueError('Bad fixed source checkpoint')
            immutable(folder/f'checkpoint_{n}_complete.json',dict(status='COMPLETE',version=VERSION,method=args.method,scene=args.scene,
                epoch=n,epochs=n,steps=old['step'],checkpoint=str(path),checkpoint_sha256=sha(path),binding_sha256=b['sha256'],test_read=False))
        for n in (100,150):
            if (folder/f'checkpoint_{n}.pt').exists():publish(n,torch.load(folder/f'checkpoint_{n}.pt',map_location='cpu',weights_only=False))
        write(folder/'worker.json',dict(status='RUNNING',method=args.method,pid=os.getpid(),host=socket.gethostname(),device=str(device),test_read=False))
        try:
            if args.max_steps and step>=args.max_steps:
                write(folder/'worker.json',dict(status='PAUSED',reason='max_steps_already_reached',step=step,pid=os.getpid(),test_read=False));return
            while epoch<=args.stop_epoch:
                p=planner.make(epoch,32)
                if next_batch==0:running=dict(weighted={},examples=0,steps=0,started_at=time.time(),plan_sha256=p['plan_sha256'])
                elif running['plan_sha256']!=p['plan_sha256']:raise ValueError('Resumed exact legacy plan differs')
                model.train()
                for bi in range(next_batch,len(p['batches'])):
                    ix=p['batches'][bi];optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        loss,metrics,detail=core.batch_objective(model,data,p,ix,b['legacy_method'],parent['microbatch'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite legacy CPC loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    step+=1;next_batch=bi+1;running['examples']+=len(ix);running['steps']+=1
                    for key,value in metrics.items():
                        if isinstance(value,(int,float)):running['weighted'][key]=running['weighted'].get(key,0.)+value*len(ix)
                    del loss,detail
                    if step%50==0 or next_batch==len(p['batches']):
                        core.save(folder/'latest.pt',record(epoch,next_batch))
                        write(folder/'progress.json',dict(status='RUNNING',epoch=epoch,batch=next_batch,batches=len(p['batches']),step=step,
                            history=history,latest=metrics,grad_norm=float(norm),plan_sha256=p['plan_sha256'],
                            seconds=previous_seconds+time.monotonic()-started,test_read=False))
                    if args.max_steps and step>=args.max_steps:
                        core.save(folder/'latest.pt',record(epoch,next_batch))
                        write(folder/'worker.json',dict(status='PAUSED',reason='max_steps',step=step,epoch=epoch,pid=os.getpid(),test_read=False));return
                # Keep the original per-epoch source diagnostic path; it does not
                # select a checkpoint or read pose/test labels.
                validation=core.evaluate(model,data,min(parent['microbatch'],32))
                if running['examples']!=len(data.ids['train']):raise ValueError('Incomplete recipient epoch')
                row=dict(epoch=epoch,mse=validation['mse'],train={k:v/running['examples'] for k,v in running['weighted'].items()},
                    seconds=time.time()-running['started_at'],plan_sha256=p['plan_sha256'],query_exposures=running['examples'],
                    paired=p['paired'],randomization_skipped=p['randomization_skipped'],random_same=p['random_same'])
                history.append(row);done=epoch;epoch+=1;next_batch=0;running={};ck=record(epoch,0)
                if done in (100,150):publish(done,ck);write(folder/f'validation_{done}.json',dict(epoch=done,**validation))
                core.save(folder/'latest.pt',ck);write(folder/'progress.json',dict(status='RUNNING',epoch=done,step=step,history=history,seconds=ck['seconds'],test_read=False))
                emit('EPOCH_COMPLETE',method=args.method,scene=args.scene,**row)
            completed=epoch-1;status='COMPLETE' if completed>=150 else 'PAUSED';path=folder/f'checkpoint_{completed}.pt'
            final=dict(status=status,version=VERSION,model_version=MODEL_VERSION,method=args.method,legacy_method=b['legacy_method'],
                scene=args.scene,epochs=completed,planned_epochs=150,steps=step,checkpoint=str(path),checkpoint_sha256=sha(path),
                binding_sha256=b['sha256'],parent_checkpoint_sha256=b['parent_checkpoint_sha256'],
                seconds=previous_seconds+time.monotonic()-started,additional_seconds=time.monotonic()-started,
                objective_changed=False,route_changed=False,source_initialization_unchanged=True,test_read=False)
            write(folder/'stage_complete.json',final)
            if status=='COMPLETE':write(folder/'complete.json',final)
            write(folder/'worker.json',dict(final,pid=os.getpid(),exit_code=0));emit('TRAIN_'+status,**final)
        except Exception as error:
            result=dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),epoch=epoch,step=step,test_read=False)
            write(folder/'failure.json',result);write(folder/'worker.json',dict(result,pid=os.getpid()));raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--scene',choices=('balls','collision','blocktower'),required=True)
    p.add_argument('--method',choices=('Structure','Cross'),required=True)
    p.add_argument('--legacy-runtime-dir',required=True);p.add_argument('--parent-checkpoint',required=True)
    p.add_argument('--relation-index',required=True);p.add_argument('--out',required=True)
    p.add_argument('--source-cuda-index',type=int,required=True);p.add_argument('--placement-receipt')
    p.add_argument('--new-runtime-dir',help='Reviewed v7 CPC directory for positive-cohort equivalence audit; defaults sibling cpc')
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--stop-epoch',type=int,choices=(50,100,150),default=100)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--protocol')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.epochs!=150 or args.source_cuda_index<0 or args.max_steps<0:raise ValueError('Fixed source150 continuation and verified active RNG index required')
    {'prepare':prepare,'smoke':smoke,'train':train}[args.command](args)
