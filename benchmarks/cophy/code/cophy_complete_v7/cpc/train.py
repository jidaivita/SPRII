"""CPC v7: fixed Native/Structure/Cross/Random-Cross trajectories to epoch150.

Only visual feature784 and audited pairing metadata enter source training.
The CLI never reads pose targets or sealed test. Native has no donor forward.
"""
import argparse
from collections import defaultdict
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

# Unique package imports: the common readout may load several model families.
HERE = Path(__file__).resolve().parent
PACKAGE = '_cophy_complete_cpc_v7'
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(PACKAGE, HERE/'__init__.py',
                                                 submodule_search_locations=[str(HERE)])
    package = importlib.util.module_from_spec(spec); sys.modules[PACKAGE] = package
    spec.loader.exec_module(package)
from importlib import import_module
models = import_module(PACKAGE+'.models')
data_api = import_module(PACKAGE+'.data')
VERSION = models.VERSION
METHODS = ('Native','Structure','Cross','Random-Cross')
SPECS = data_api.SPECS
read, digest = data_api.read, data_api.digest


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp,path)


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    torch.save(value,tmp); os.replace(tmp,path)


def immutable(path, value):
    if Path(path).exists() and read(path)!=value: raise ValueError('Changed binding: '+str(path))
    write(path,value)


def emit(event, **kw): print(json.dumps(dict(event=event,time=time.time(),**kw),allow_nan=False),flush=True)


def seeded(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()


def rng_state(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(device)] if device.type=='cuda' else [],rng_device_index=0)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if device.type=='cuda':
        if len(state['cuda'])!=1: raise ValueError('Expected one explicitly stored active CUDA RNG')
        torch.cuda.set_rng_state(state['cuda'][0],device)
    elif state['cuda']: raise ValueError('Cannot silently migrate a CUDA run to CPU')


class Planner:
    """Common recipient/focal plans; Collision externalizes every active P.

    Random conditionally permutes the correct donor multiset. Incidental same
    physics remains possible and is counted; Random is not a Wrong assay.
    """
    def __init__(self, data, index_path, seed):
        self.data,self.seed=data,seed
        self.old=data_api.EpochPlanner(data,index_path,seed)
        self.index,self.ids,self.lookup=self.old.index,self.old.ids,self.old.lookup
        self.active=np.asarray(data.arrays['train']['presence_cd'][:,:3]).any(1)
        self.seen=np.asarray(data.arrays['train']['presence_ab']).any(1)
        self.route='all' if data.scene=='collision' else 'focal'
        self.pools={}; self.covered=np.ones(len(self.ids),bool); counts=defaultdict(lambda:dict(active=0,eligible=0))
        reasons=defaultdict(int)
        for i,ident in enumerate(self.ids):
            for slot in np.flatnonzero(self.active[i]):
                slot=int(slot); rec=self.index.records.get((ident,slot))
                group=canonical([slot,rec['stratum'] if rec else None]); counts[group]['active']+=1
                reason=None
                if rec is None: reason='missing_audited_record'
                elif not rec['recipient'] or not self.seen[i,slot]: reason='recipient_not_eligible_or_AB_absent'
                else:
                    key=(rec['group'],rec['physical_key'])
                    if key not in self.pools:
                        self.pools[key]=np.asarray(sorted({self.lookup[r['id']] for r in self.index.by_relation[key]
                            if self.seen[self.lookup[r['id']],slot]}),np.int64)
                    if len(self.pools[key])-int(i in self.pools[key])<1: reason='no_independent_visible_correct_donor'
                if reason: self.covered[i]=False; reasons[reason]+=1
                else: counts[group]['eligible']+=1
        self.coverage=dict(route=self.route,total_recipients=len(self.ids),
            all_object_eligible=int(self.covered.sum()),by_slot_public_stratum=dict(counts),reasons=dict(reasons),
            policy='own loss retains all recipients; Cross and Random share their legal cross cohort; no own-P fallback')

    def make(self, epoch, batch_size):
        p=self.old.make(epoch,batch_size); n=len(self.ids); k=self.data.slots
        common=p['focal']>=0
        if self.route=='all': common &= self.covered
        correct=np.full((n,k),-1,np.int64); randomized=np.full_like(correct,-1)
        for i in np.flatnonzero(common):
            f=int(p['focal'][i]); c=int(p['correct'][i]); r=int(p['random'][i])
            if not (self.active[i,f] and self.seen[i,f] and self.seen[c,f] and self.seen[r,f]):
                common[i]=False; continue
            correct[i,f]=c; randomized[i,f]=r
            if self.route=='all':
                for slot in np.flatnonzero(self.active[i]):
                    if slot==f: continue
                    rec=self.index.records[(self.ids[i],int(slot))]
                    pool=self.pools[(rec['group'],rec['physical_key'])]
                    seed=int.from_bytes(hashlib.sha256(f'{VERSION}:extra:{self.seed}:{epoch}:{self.ids[i]}:{slot}'.encode()).digest()[:8],'little')
                    rng=np.random.default_rng(seed); loc=int(np.searchsorted(pool,i))
                    own=loc<len(pool) and pool[loc]==i; rank=int(rng.integers(len(pool)-int(own)))
                    correct[i,slot]=pool[rank+int(own and rank>=loc)]
        if common.any():
            groups=defaultdict(list)
            for i in np.flatnonzero(common):
                slots=np.flatnonzero(self.active[i]) if self.route=='all' else [int(p['focal'][i])]
                for slot in slots:
                    rec=self.index.records[(self.ids[i],int(slot))]
                    groups[rec['group']].append((int(i),int(slot),int(correct[i,slot])))
            for group,entries in groups.items():
                seed=int.from_bytes(hashlib.sha256(canonical([VERSION,'random-'+self.route,self.seed,epoch,group]).encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                for attempt in range(256):
                    order=rng.permutation(len(entries))
                    if all(entries[int(j)][2]!=entries[ii][0] for ii,j in enumerate(order)): break
                else: raise ValueError('Conditional all-object random permutation failed; no silent own donor fallback')
                for ii,j in enumerate(order):
                    i,slot,_=entries[ii]; randomized[i,slot]=entries[int(j)][2]
                if sorted(x[2] for x in entries)!=sorted(int(randomized[i,s]) for i,s,_ in entries):
                    raise AssertionError('Random changed the donor marginal')
        same=objects=0
        for i in np.flatnonzero(common):
            slots=np.flatnonzero(self.active[i]) if self.route=='all' else [int(p['focal'][i])]
            for slot in slots:
                slot=int(slot); rec=self.index.records[(self.ids[i],slot)]
                for arm,table in (('correct',correct),('random',randomized)):
                    j=int(table[i,slot]); donor=self.index.records[(self.ids[j],slot)]
                    if j==i or not donor['donor'] or not self.seen[j,slot] or donor['group']!=rec['group']:
                        raise ValueError('Illegal independent donor')
                    if arm=='correct' and donor['physical_key']!=rec['physical_key']: raise ValueError('Incorrect correct donor')
                    if arm=='random': same+=int(donor['physical_key']==rec['physical_key'])
                objects+=1
        if not common.any(): raise ValueError('Zero common Cross/Random cohort')
        h=hashlib.sha256()
        for a in (np.concatenate(p['batches']),p['focal'],common,correct,randomized): h.update(a.tobytes())
        p.update(common=common,external_correct=correct,external_random=randomized,
                 original_plan_sha256=p['plan_sha256'],plan_sha256=h.hexdigest(),
                 common_cross_recipients=int(common.sum()),external_objects=objects,random_same_objects=same,route=self.route)
        return p


def binding(args,data):
    files=dict(data.files)
    for f in sorted(HERE.glob('*.py')): files[str(f)]=digest(f)
    files[str(Path(args.relation_index).resolve())]=digest(args.relation_index)
    if args.protocol: files[str(Path(args.protocol).resolve())]=digest(args.protocol)
    b=dict(version=VERSION,scene=args.scene,features=str(Path(args.features).resolve()),files=files,
       seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,microbatch=args.microbatch,
       lr=args.lr,weight_decay=1e-4,clip_norm=1.,temperature=.1,sigreg_weight=.2,lambda_cross=1.,lambda_align=0.,
       source_supports=1,queries_per_memory=1,query_frames=3,
       route='all' if args.scene=='collision' else 'focal',route_constant_from_epoch1=True,
       source_selection='fixed 50/100/150, no latent-loss selection',coordinate_labels_read=False,test_read=False,
       target='live shared motion projection of fixed supervised RGB feature784; future InfoNCE, not pose MSE',
       random='conditional donor permutation within slot/public-type/gravity, preserves marginal; accidental same physics counted')
    b['sha256']=hashlib.sha256(canonical(b).encode()).hexdigest(); return b


def setup(args):
    torch.set_num_threads(args.threads)
    data=data_api.FeatureData(args.features,args.scene)
    expected={'balls':(7000,2000),'collision':(14000,4000),'blocktower':(28310,8088)}
    if tuple(len(data.ids[s]) for s in ('train','val'))!=expected[args.scene]: raise ValueError('Expected complete bound scene')
    b=binding(args,data)
    if read(Path(args.out)/'binding.json')!=b: raise ValueError('Run prepare with the same source binding')
    planner=Planner(data,args.relation_index,args.seed)
    seeded(args.seed); model=models.make_model(args.method).to(args.device)
    return data,b,planner,model


def prepare(args):
    torch.set_num_threads(args.threads); data=data_api.FeatureData(args.features,args.scene)
    b=binding(args,data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    immutable(out/'binding.json',b)
    planner=Planner(data,args.relation_index,args.seed); p=planner.make(1,args.batch_size)
    immutable(out/'coverage.json',planner.coverage)
    initials={}
    for method in METHODS:
        seeded(args.seed); m=models.make_model(method)
        initials[method]=dict(sha256=state_sha(m),parameters=sum(v.numel() for v in m.parameters()),model_config=m.artifact_config())
    if len({initials[m]['sha256'] for m in METHODS[1:]})!=1: raise AssertionError('Split methods have different starts')
    immutable(out/'initialization.json',initials)
    summary={k:p[k] for k in ('plan_sha256','original_plan_sha256','query_exposures','common_cross_recipients','external_objects','random_same_objects','route')}
    immutable(out/'plan_epoch1.json',summary)
    write(out/'prepared.json',dict(status='COMPLETE',version=VERSION,binding_sha256=b['sha256'],initialization=initials,plan=summary,test_read=False))
    emit('PREPARED',scene=args.scene,initialization=initials,plan=summary)


def objective(model,data,p,ix,method,microbatch):
    device=next(model.parameters()).device
    ab,am,c,cm=data.context('train',ix,device); cd,tm=data.target('train',ix,device)
    active=cm.any(1); mask=tm[:,3:] & active[:,None]
    context=model.encode_joint(ab,am,c,cm)
    pred=model.predict_from_context(context,active,data.frames-3); target=model.target(cd)
    groups=torch.as_tensor(p['public_groups'][ix],device=device)
    b,t,k,d=pred.shape
    own,stats=data_api.cpc_loss(pred.permute(0,2,1,3).reshape(b*k,t,d),target,mask,groups,
                                torch.arange(b*k,device=device),model.config.temperature)
    reg=model.regularization(ab,am,cd,tm); cross=context.sum()*0
    donor_p=None; rows=torch.empty(0,device=device,dtype=torch.long)
    if method in ('Cross','Random-Cross'):
        local=np.flatnonzero(p['common'][ix]); rows=torch.as_tensor(local,device=device)
        if len(rows):
            focal=torch.as_tensor(p['focal'][ix[local]],device=device)
            required=p['external_random' if method=='Random-Cross' else 'external_correct'][ix[local]]
            positions=np.argwhere(required>=0)
            donor_ids=required[positions[:,0],positions[:,1]]
            donor_ab,donor_mask=data.history('train',donor_ids,device)
            encoded=data_api.encode_micro(model,donor_ab,donor_mask,microbatch)
            rr=torch.as_tensor(positions[:,0],device=device); ss=torch.as_tensor(positions[:,1],device=device)
            donor_p=encoded[torch.arange(len(encoded),device=device),ss]
            mixed=context[rows].clone(); persistent=mixed[...,:64].clone(); persistent[rr,ss]=donor_p
            mixed=torch.cat((persistent,mixed[...,64:]),-1)
            prediction=model.predict_from_context(mixed,active[rows],data.frames-3)
            cross,cs=data_api.cpc_loss(prediction[torch.arange(len(rows),device=device),:,focal],
               target,mask,groups,rows*k+focal,model.config.temperature)
            stats.update({'cross_'+key:value for key,value in cs.items()})
    loss=own+.2*reg+cross
    metrics=dict(loss=float(loss.detach()),self_loss=float(own.detach()),cross=float(cross.detach()),
                 sigreg=float(reg.detach()),paired=len(rows),donor_objects=0 if donor_p is None else len(donor_p),
                 target_std=float(target.detach().float()[mask].std(0).mean()),**stats)
    return loss,metrics,dict(context=context,target=target,donor_p=donor_p)


def smoke(args):
    data,b,planner,model=setup(args); p=planner.make(1,args.batch_size); ix=p['batches'][0]
    initial=state_sha(model); model.train(); device=next(model.parameters()).device
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        loss,metrics,detail=objective(model,data,p,ix,args.method,args.microbatch)
    detail['context'].retain_grad(); detail['target'].retain_grad()
    if detail['donor_p'] is not None: detail['donor_p'].retain_grad()
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
        raise FloatingPointError('Nonfinite source smoke')
    if metrics['cpc_candidates_mean']<2 or metrics['cpc_anchors']<=0: raise ValueError('InfoNCE has no effective negatives')
    if args.method in ('Native','Structure') and (metrics['cross']!=0 or detail['donor_p'] is not None): raise AssertionError('Baseline donor path')
    gradients={}
    for key in ('context','target','donor_p'):
        value=detail[key]
        if value is None: continue
        if value.grad is None or not torch.isfinite(value.grad).all() or value.grad.norm()<=0:
            raise ValueError('Missing finite nonzero '+key+' gradient')
        gradients[key]=float(value.grad.norm())
    model.eval()
    with torch.no_grad():
        ab,am,c,cm=data.context('train',ix[:2],device)
        direct=model.encode_joint(ab,am,c,cm)
        cached=model.encode_joint_from_cached(model.encode_history_state(ab,am),model.encode_current_tokens(c,cm),cm)
        torch.testing.assert_close(direct,cached,atol=2e-6,rtol=2e-6)
        actual=model.project(ab).float()[am]; flat=model.project(ab,zero_delta=True).float()[am]
        motion=float((actual-flat).square().sum(-1).mean()); spread=float((actual-actual.mean(0)).square().sum(-1).mean())
    if state_sha(model)!=initial: raise AssertionError('Smoke mutated source weights')
    record=dict(status='PASS',version=VERSION,binding_sha256=b['sha256'],method=args.method,rows=len(ix),
       metrics=metrics,gradients=gradients,cached_max_error=float((direct-cached).abs().max()),motion_response=motion,spread=spread,
       optimizer_updates=0,weights_discarded=True,test_read=False)
    write(Path(args.out)/'runs'/args.method/'smoke.json',record); emit('SMOKE_PASS',**record)


@torch.no_grad()
def source_diagnostic(model,data):
    # A fixed train-only set, not a source checkpoint selection rule.
    ix=np.arange(min(32,len(data.ids['train']))); device=next(model.parameters()).device
    ab,am,c,cm=data.context('train',ix,device); cd,tm=data.target('train',ix,device)
    model.eval(); target=model.target(cd); pred=model.predict_from_context(model.encode_joint(ab,am,c,cm),cm.any(1),data.frames-3)
    mask=tm[:,3:] & cm.any(1)[:,None]
    a=model.project(ab)[am]; z=model.project(ab,zero_delta=True)[am]
    return dict(train_latent_mse=float(data_api.masked_mse(pred,target,mask)),
                target_std=float(target[mask].std(0).mean()),motion_response=float((a-z).square().sum(-1).mean()),
                motion_spread=float((a-a.mean(0)).square().sum(-1).mean()),selection=False)


def train(args):
    out=Path(args.out); folder=out/'runs'/args.method; folder.mkdir(parents=True,exist_ok=True)
    with open(folder/'owner.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data,b,planner,model=setup(args); device=next(model.parameters()).device
        expected=read(out/'initialization.json')[args.method]['sha256']
        if state_sha(model)!=expected: raise ValueError('Changed common initialization')
        sm=read(folder/'smoke.json')
        if sm.get('status')!='PASS' or sm.get('binding_sha256')!=b['sha256']: raise ValueError('Matching real source smoke required')
        optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        epoch,batch,step,history,running,prior_seconds=1,0,0,[],{},0.
        candidates=[folder/'latest.pt']+[folder/f'checkpoint_{n}.pt' for n in (50,100,150)]
        available=[]
        for f in candidates:
            if f.exists():
                ck=torch.load(f,map_location='cpu',weights_only=False)
                if ck.get('binding_sha256')!=b['sha256'] or ck.get('method')!=args.method: raise ValueError('Changed saved run')
                available.append((ck['step'],f,ck))
        if available:
            if not args.resume: raise ValueError('Existing run; explicit --resume required')
            _,_,ck=max(available,key=lambda item:item[0])
            model.load_state_dict(ck['model'],strict=True); optim.load_state_dict(ck['optimizer'])
            epoch,batch,step=ck['next_epoch'],ck['next_batch'],ck['step']
            history,running=ck['history'],ck['running']; prior_seconds=ck['seconds']; restore_rng(ck['rng'],device)
        started=time.monotonic()
        def record(e,bb):
            return dict(version=VERSION,method=args.method,family='CPC',scene=args.scene,model_config=model.artifact_config(),
              model=model.state_dict(),optimizer=optim.state_dict(),rng=rng_state(device),rng_device_index=0,
              binding=b,binding_sha256=b['sha256'],initialization_sha256=expected,
              epoch=e-1 if bb==0 else e,next_epoch=e,next_batch=bb,step=step,history=history,running=running,
              microbatch=args.microbatch,seconds=prior_seconds+time.monotonic()-started,test_read=False)
        def snapshot(n,ck):
            path=folder/f'checkpoint_{n}.pt'; marker=folder/f'checkpoint_{n}_complete.json'
            if not path.exists(): save(path,ck)
            old=torch.load(path,map_location='cpu',weights_only=False)
            if old['binding_sha256']!=b['sha256'] or old['next_epoch']!=n+1 or old['next_batch']!=0: raise ValueError('Invalid published source')
            receipt=dict(status='COMPLETE',version=VERSION,scene=args.scene,method=args.method,epoch=n,epochs=n,
                steps=old['step'],checkpoint=str(path),checkpoint_sha256=digest(path),binding_sha256=b['sha256'],test_read=False)
            immutable(marker,receipt)
        # Repair publication after a process stopped between snapshot and marker.
        for n in (50,100,150):
            f=folder/f'checkpoint_{n}.pt'
            if f.exists(): snapshot(n,torch.load(f,map_location='cpu',weights_only=False))
        write(folder/'worker.json',dict(status='RUNNING',pid=os.getpid(),host=socket.gethostname(),device=str(device),test_read=False))
        try:
            if args.max_steps and step>=args.max_steps:
                write(folder/'worker.json',dict(status='PAUSED',reason='max_steps_already_reached',step=step,epoch=epoch,pid=os.getpid(),test_read=False))
                emit('PAUSED',method=args.method,epoch=epoch,step=step,reason='max_steps_already_reached'); return
            while epoch<=args.stop_epoch:
                p=planner.make(epoch,args.batch_size)
                if batch==0: running=dict(examples=0,weighted={},plan_sha256=p['plan_sha256'])
                elif running['plan_sha256']!=p['plan_sha256']: raise ValueError('Resume plan changed')
                model.train(); epoch_started=time.monotonic()
                for bi in range(batch,len(p['batches'])):
                    ix=p['batches'][bi]; optim.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        loss,metrics,_=objective(model,data,p,ix,args.method,args.microbatch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite CPC objective')
                    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optim.step()
                    step+=1; batch=bi+1; running['examples']+=len(ix)
                    for key,value in metrics.items(): running['weighted'][key]=running['weighted'].get(key,0.)+value*len(ix)
                    if step%50==0 or batch==len(p['batches']):
                        save(folder/'latest.pt',record(epoch,batch))
                        write(folder/'progress.json',dict(status='RUNNING',epoch=epoch,batch=batch,batches=len(p['batches']),step=step,
                           history=history,latest=metrics,grad_norm=float(norm),seconds=prior_seconds+time.monotonic()-started,
                           plan_sha256=p['plan_sha256'],test_read=False))
                    if args.max_steps and step>=args.max_steps:
                        save(folder/'latest.pt',record(epoch,batch))
                        write(folder/'worker.json',dict(status='PAUSED',reason='max_steps',step=step,epoch=epoch,pid=os.getpid(),test_read=False))
                        emit('PAUSED',method=args.method,epoch=epoch,step=step); return
                if running['examples']!=len(data.ids['train']): raise ValueError('Incomplete recipient exposure')
                row=dict(epoch=epoch,step=step,train={k:v/running['examples'] for k,v in running['weighted'].items()},
                   query_exposures=running['examples'],common_cross_recipients=p['common_cross_recipients'],
                   external_objects=p['external_objects'],random_same_objects=p['random_same_objects'],
                   route=p['route'],plan_sha256=p['plan_sha256'],seconds=time.monotonic()-epoch_started)
                if epoch in (1,10,25,50,100,150): row['diagnostic']=source_diagnostic(model,data)
                history.append(row); done=epoch; epoch+=1; batch=0; running={}; ck=record(epoch,0)
                if done in (50,100,150): snapshot(done,ck)
                save(folder/'latest.pt',ck)
                write(folder/'progress.json',dict(status='RUNNING',epoch=done,step=step,history=history,seconds=ck['seconds'],test_read=False))
                emit('EPOCH_COMPLETE',method=args.method,scene=args.scene,**row)
            completed=epoch-1; status='COMPLETE' if completed>=args.epochs else 'PAUSED'
            end=dict(status=status,version=VERSION,scene=args.scene,method=args.method,epochs=completed,planned_epochs=args.epochs,
               steps=step,binding_sha256=b['sha256'],seconds=prior_seconds+time.monotonic()-started,test_read=False)
            if completed in (50,100,150):
                path=folder/f'checkpoint_{completed}.pt'; end.update(checkpoint=str(path),checkpoint_sha256=digest(path))
            write(folder/'stage_complete.json',end)
            if status=='COMPLETE': write(folder/'complete.json',end)
            write(folder/'worker.json',dict(end,pid=os.getpid(),exit_code=0)); emit('TRAIN_'+status,**end)
        except Exception as error:
            fail=dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),epoch=epoch,step=step,test_read=False)
            write(folder/'failure.json',fail); write(folder/'worker.json',dict(fail,pid=os.getpid())); raise


def export(args):
    model,ck=models.load_checkpoint(args.checkpoint,args.device)
    description=dict(status='COMPLETE',version=VERSION,checkpoint=str(Path(args.checkpoint).resolve()),
      checkpoint_sha256=digest(args.checkpoint),method=ck['method'],source_epochs=ck['epoch'],representation='complete_U128',
      architecture=model.config.architecture,model_config=model.artifact_config(),
      history_cache='Native: hidden[2,B,O,128]+present[B,O]; split: P64+present',
      current='recipient CD[:3] only; never donor CD',context_api='encode_joint / encode_joint_from_cached',
      no_history='Native zero both cached recurrent layers then actual current; split zero P then actual current T',test_read=False)
    write(Path(args.out)/'export.json',description); emit('EXPORTED',**description)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train','export'))
    p.add_argument('--scene',choices=tuple(SPECS),required=True)
    p.add_argument('--features'); p.add_argument('--relation-index'); p.add_argument('--out',required=True)
    p.add_argument('--method',choices=METHODS,default='Native'); p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--stop-epoch',type=int,choices=(50,100,150),default=150)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--microbatch',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threads',type=int,default=4); p.add_argument('--device',default='cpu')
    p.add_argument('--protocol'); p.add_argument('--resume',action='store_true'); p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--checkpoint'); return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.epochs!=150 or args.batch_size!=32 or not 1<=args.microbatch<=32 or args.seed!=0 or args.lr!=3e-4 or args.max_steps<0:
        raise ValueError('Fixed source plan: seed0,150 total epochs,batch32,AdamW3e-4')
    if args.command!='export' and (not args.features or not args.relation_index): raise ValueError('Missing bound input paths')
    if args.command=='export' and not args.checkpoint: raise ValueError('export requires --checkpoint')
    {'prepare':prepare,'smoke':smoke,'train':train,'export':export}[args.command](args)
