"""Collision v4.6: differentiable, two-support-set reuse adaptation.

Uses audited train/val visual caches; never loads test or future donor targets.
All methods share the supervised path. Only the auxiliary pairing differs.
"""
import argparse
import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from collision_xep import (K, D, HORIZON, TYPES, PilotData, PredictHead,
                           digest, emit, read, write, save_torch, save_npz, scores)

VERSION = 'collision-reuse-ft-v4.6-impl1'
METHODS = {'Native-FT': 0., 'A-weak': .01, 'A-medium': .05, 'Random-weak': .01}


def prepare(out, base):
    out, base = Path(out), Path(base)
    out.mkdir(parents=True, exist_ok=True)
    m = read(base/'manifest.json')
    source = m['source_models']['Native']
    if digest(source['path']) != source['sha256']:
        raise ValueError('Native source checkpoint changed')
    head = base/'runs/Native-U/selected.pt'
    state = torch.load(head, map_location='cpu', weights_only=False)
    bindings = {'version': VERSION, 'base': str(base), 'test_read': False,
                'source': source, 'head': {'path': str(head), 'sha256': digest(head),
                                         'epoch': state['epoch']},
                'base_manifest_sha256': digest(base/'manifest.json'),
                'code_sha256': digest(__file__),
                'dependency_sha256': digest(Path(__file__).with_name('collision_xep.py')),
                'support_sets_train': 2, 'supports_per_set': 3, 'splits': {}}
    for split in ['train', 'val']:
        part = m['splits'][split]
        if digest(part['cache']) != part['cache_sha256']:
            raise ValueError('Visual cache changed')
        need = 6 if split == 'train' else 3
        minimum = min(len(xs) for lists in part['candidates'].values() for xs in lists if xs)
        if minimum < need:
            raise ValueError('Insufficient disjoint supports')
        bindings['splits'][split] = {'queries': len(part['query_ids']), 'minimum_pool': minimum,
            'cache': part['cache'], 'cache_sha256': part['cache_sha256'],
            'input_sha256': digest(base/f'input_{split}.npz'),
            'target_sha256': digest(base/f'target_{split}.npz')}
    path = out/'binding.json'
    if path.exists() and read(path) != bindings:
        raise ValueError('Output directory bound to another implementation or input')
    write(path, bindings)
    emit('prepared', source_epoch=source['epoch'], head_epoch=state['epoch'],
         splits=bindings['splits'], test_read=False)


class HistoryEncoder(nn.Module):
    """Exactly the source GCN/GRU weights; batch the time axis for efficiency."""
    def __init__(self, source_state):
        super().__init__()
        self.mlp_inter = nn.Sequential(nn.Linear(6,32), nn.ReLU(), nn.Linear(32,32),
                                       nn.ReLU(), nn.Linear(32,32), nn.ReLU())
        self.mlp_out = nn.Sequential(nn.Linear(67,32), nn.ReLU(), nn.Linear(32,32))
        self.rnn = nn.GRU(32,32,batch_first=True)
        wanted = {k.removeprefix('backbone.'): v for k,v in source_state.items()
                  if k.startswith(('backbone.mlp_inter.', 'backbone.mlp_out.', 'backbone.rnn.'))}
        self.load_state_dict(wanted, strict=True)

    def forward(self, pose, presence):
        b,t,k,_ = pose.shape
        x = pose.reshape(b*t,k,3)
        active = presence[:,None].expand(b,t,k).reshape(b*t,k)
        # Source orientation is [object j, object i]; do not reverse the pair.
        x1 = x[:,None].expand(-1,k,-1,-1)
        x2 = x[:,:,None].expand(-1,-1,k,-1)
        edges = self.mlp_inter(torch.cat([x1,x2],-1))
        pair = active[:,:,None]*active[:,None,:]
        pair = pair*(1-torch.eye(k,device=x.device)[None])
        e = (edges*pair[...,None]).sum(2)/(.01+pair.sum(2))[...,None]
        total = (e*active[...,None]).sum(1)/(.01+active.sum(1))[:,None]
        local = self.mlp_out(torch.cat([x,e,total[:,None].expand(-1,k,-1)],-1))
        seq = local.reshape(b,t,k,32).permute(0,2,1,3).reshape(b*k,t,32)
        _, h = self.rnn(seq)
        return h[0].reshape(b,k,32)


class ReuseHead(PredictHead):
    def __init__(self, legacy_state):
        super().__init__('Native-U')
        old_input = self.cell.input_size
        self.cell = nn.GRUCell(old_input+64, self.width)
        updated = dict(legacy_state)
        updated['cell.weight_ih'] = F.pad(legacy_state['cell.weight_ih'], (0,64))
        self.load_state_dict(updated, strict=True)

    def forward_memory(self,q,det,mask,memory,horizon=HORIZON):
        b,l,k,_ = q.shape
        x=(q-self.xy_mean)/self.xy_scale
        dx=torch.cat([torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]],1)
        tokens=torch.cat([x,dx,det],-1).permute(0,2,1,3).reshape(b*k,l,2*D+1+len(TYPES))
        _,qh=self.query(tokens); qh=qh[0].reshape(b,k,64)
        available=torch.ones(b,k,1,device=q.device)
        h=self.init(torch.cat([qh,memory,available],-1))
        position=x[:,-1]; v=(x[:,-1]-x[:,0])/(l-1)
        pair=mask[:,:,None]*mask[:,None,:]*(1-torch.eye(k,device=q.device)[None])
        denom=pair.sum(2).clamp_min(1)[...,None]; outputs=[]
        for _ in range(horizon):
            hi=h[:,:,None].expand(b,k,k,self.width); hj=h[:,None,:].expand(b,k,k,self.width)
            rel=position[:,None]-position[:,:,None]; rv=v[:,None]-v[:,:,None]
            messages=self.message(torch.cat([hi,hj,rel,rv],-1))
            agg=(messages*pair[...,None]).sum(2)/denom
            step=torch.cat([agg,position,v,memory],-1)
            h=self.cell(step.reshape(b*k,-1),h.reshape(b*k,-1)).reshape(b,k,-1)
            move=v+self.delta(h); position=position+move; v=move
            outputs.append(position*self.xy_scale+self.xy_mean)
        return torch.stack(outputs,1)


class FineTuneModel(nn.Module):
    def __init__(self, binding):
        super().__init__()
        source=torch.load(binding['source']['path'],map_location='cpu',weights_only=False)
        head=torch.load(binding['head']['path'],map_location='cpu',weights_only=False)
        self.encoder=HistoryEncoder(source['model'])
        self.head=ReuseHead(head['model'])
        self.register_buffer('code_mean',torch.tensor(head['config']['code_mean']))
        self.register_buffer('code_scale',torch.tensor(head['config']['code_scale']))

    def memory(self,data,split,plan):
        # Encode whole donor scenes to preserve source GCN context, then select focal codes.
        indices=torch.as_tensor(plan,device=self.code_mean.device,dtype=torch.long)
        unique,inverse=torch.unique(indices.reshape(-1),sorted=True,return_inverse=True)
        hist=data.hist[split]
        codes=self.encoder(hist['pose'][unique],hist['presence'][unique])
        slots=torch.arange(K,device=indices.device)[None,:,None,None].expand_as(indices)
        selected=codes[inverse,slots.reshape(-1)].reshape(*indices.shape,32)
        standardized=(selected-self.code_mean)/self.code_scale
        # plan B,K,G,S -> memory B,G,K,64
        return self.head.support(standardized).mean(3).permute(0,2,1,3)


class FineTuneData:
    def __init__(self,base,device):
        self.base=Path(base); self.pilot=PilotData(base,'Native-U')
        self.manifest=self.pilot.manifest; self.rows=self.pilot.data
        self.hist={};self.public={};self.physical={}
        for split in ['train','val']:
            part=self.manifest['splits'][split]
            with open(part['cache'],'rb') as f:cache=pickle.load(f)
            ids=part['all_ids']
            self.hist[split]={
                'pose':torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in ids])).float().to(device),
                'presence':torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in ids])).float().to(device)}
            del cache
            qids=self.rows[split]['ids']
            public=np.array([part['known_type'][i] for i in qids]).argmax(-1)
            self.public[split]=public+np.arange(K)[None]*len(TYPES)
            # Used exclusively to report accidental equal-property random pairs, never as model input.
            self.physical[split]=np.array([part['physical'][i] for i in qids])

    def plan(self,split,epoch,groups):
        row=self.rows[split]; part=self.manifest['splits'][split]
        plan=np.zeros((len(row['ids']),K,groups,3),np.int64)
        index={s:i for i,s in enumerate(part['all_ids'])}
        for i,ident in enumerate(row['ids']):
            seed=int.from_bytes(hashlib.sha256(f'xep:20260911:{split}:{ident}:{epoch}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            for k in range(K):
                if row['mask'][i,k]>0:
                    opts=part['candidates'][ident][k]
                    if index[ident] in opts:raise ValueError('Recipient in support pool')
                    plan[i,k]=rng.choice(opts,3*groups,replace=False).reshape(groups,3)
        return plan

    def batch(self,split,ix,device):
        row=self.rows[split]
        tensor=lambda x:torch.from_numpy(np.asarray(x,dtype=np.float32)).to(device)
        return tuple(tensor(row[key][ix]) for key in ['q','det','mask','target'])


def auxiliary(memory,mask,strata,physical,randomized,seed):
    active=mask.reshape(-1)>0
    x=memory[:,0].reshape(-1,64)[active]; y=memory[:,1].reshape(-1,64)[active]
    labels=np.asarray(strata).reshape(-1)[active.detach().cpu().numpy()]
    properties=np.asarray(physical).reshape(-1,3)[active.detach().cpu().numpy()]
    order=np.arange(len(labels));valid=np.zeros(len(labels),bool);movable=0
    rng=np.random.default_rng(seed)
    residual_x=[];residual_y=[]
    for key in np.unique(labels):
        ix=np.flatnonzero(labels==key)
        if len(ix)>1:
            valid[ix]=True;movable+=len(ix)
            shuffled=rng.permutation(ix)
            order[shuffled]=np.roll(shuffled,1)
            tx=torch.as_tensor(ix,device=x.device)
            residual_x.append(x[tx]-x[tx].mean(0,keepdim=True))
            residual_y.append(y[tx]-y[tx].mean(0,keepdim=True))
    paired=y[torch.as_tensor(order,device=y.device)] if randomized else y
    invariance=F.mse_loss(x,paired)
    zero=x.sum()*0
    variance=covariance=zero
    if residual_x:
        rx=torch.cat(residual_x);ry=torch.cat(residual_y)
        variance=(F.relu(1-torch.sqrt(rx.var(0,unbiased=False)+1e-4)).mean()+
                  F.relu(1-torch.sqrt(ry.var(0,unbiased=False)+1e-4)).mean())/2
        def cov_penalty(r):
            cov=r.T@r/max(len(r)-1,1)
            return (cov.square().sum()-cov.diagonal().square().sum())/r.shape[1]
        covariance=(cov_penalty(rx)+cov_penalty(ry))/2
    moved=order!=np.arange(len(order))
    accidental=float(np.mean(np.all(properties[moved]==properties[order[moved]],axis=-1))) if moved.any() else 0.
    metrics={'inv':float(invariance.detach()),'var':float(variance.detach()),
             'cov':float(covariance.detach()),'random_unpermutable':1-movable/max(len(labels),1),
             'random_accidental_same':accidental,'aux_objects':len(labels)}
    return invariance+variance+.04*covariance,metrics


def prediction_pair(model,data,ix,plan,device):
    q,det,mask,y=data.batch('train',ix,device)
    memory=model.memory(data,'train',plan[ix])
    # Both supervised forwards are executed for every method; no branching here.
    p1=model.head.forward_memory(q,det,mask,memory[:,0])
    p2=model.head.forward_memory(q,det,mask,memory[:,1])
    l1=scores(p1,y,mask).mean(); l2=scores(p2,y,mask).mean()
    return (l1+l2)/2,memory,mask,l1,l2


@torch.no_grad()
def evaluate(model,data,plan,device):
    model.eval();mse=[];fde=[];thirds=[]
    for off in range(0,len(data.rows['val']['ids']),64):
        ix=np.arange(off,min(off+64,len(data.rows['val']['ids'])))
        q,det,mask,y=data.batch('val',ix,device)
        memory=model.memory(data,'val',plan[ix])[:,0]
        pred=model.head.forward_memory(q,det,mask,memory)
        mse.extend(scores(pred,y,mask).cpu().tolist())
        last=(pred[:,-1]-y[:,-1]).norm(dim=-1)
        fde.extend(((last*mask).sum(1)/mask.sum(1)).cpu().tolist())
        thirds.extend(torch.stack([scores(pred[:,i:i+4],y[:,i:i+4],mask) for i in [0,4,8]],1).cpu().tolist())
    return {'mse':float(np.mean(mse)),'fde':float(np.mean(fde)),
            'thirds':np.mean(thirds,0).tolist(),'per_recipient_mse':mse,
            'ids':data.rows['val']['ids'],'recipients':len(mse)}


def setup(out,base,device):
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    binding=read(Path(out)/'binding.json')
    if binding['code_sha256']!=digest(__file__) or binding['base_manifest_sha256']!=digest(Path(base)/'manifest.json'):
        raise ValueError('Implementation/input differs from binding')
    if binding['dependency_sha256']!=digest(Path(__file__).with_name('collision_xep.py')):
        raise ValueError('Dependency changed')
    for entry in [binding['source'],binding['head']]:
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Warm start changed')
    data=FineTuneData(base,device)
    model=FineTuneModel(binding).to(device)
    return binding,data,model


def optimizer_for(model):
    return torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
                             {'params':model.head.parameters(),'lr':1e-4}],weight_decay=1e-4)


def smoke(out,base,device):
    binding,data,model=setup(out,base,device)
    plan=data.plan('train',1,2);valplan=data.plan('val',0,1)
    ix=np.arange(4);q,det,mask,y=data.batch('train',ix,device)
    old=PredictHead('Native-U').to(device)
    old.load_state_dict(torch.load(binding['head']['path'],map_location=device,weights_only=False)['model'])
    model.eval();old.eval()
    with torch.no_grad():
        cached=torch.tensor(data.rows['train']['codes'][plan[ix,:,0],np.arange(K)[None,:,None]],device=device)
        old_pred=old(q,det,mask,cached)
        rawmem=model.memory(data,'train',plan[ix])
        new_pred=model.head.forward_memory(q,det,mask,rawmem[:,0])
        delta=float((old_pred-new_pred).abs().max())
        torch.testing.assert_close(old_pred,new_pred,atol=2e-5,rtol=2e-5)
        history=data.hist['train'];u=model.encoder(history['pose'][:16],history['presence'][:16])
        reference=torch.tensor(data.rows['train']['codes'][:16],device=device)
        reference=reference*model.code_scale+model.code_mean
        torch.testing.assert_close(u,reference,atol=2e-5,rtol=2e-5)
        code_delta=float((u-reference).abs().max())
    records=[]
    for method,weight in METHODS.items():
        # Each smoke variant starts anew, and training later reloads anew again.
        candidate=FineTuneModel(binding).to(device).train();opt=optimizer_for(candidate)
        before=candidate.encoder.mlp_inter[0].weight.detach().clone()
        pred,mem,active,l1,l2=prediction_pair(candidate,data,ix,plan,device)
        w=candidate.encoder.mlp_inter[0].weight
        g1=torch.autograd.grad(l1,w,retain_graph=True)[0]
        g2=torch.autograd.grad(l2,w,retain_graph=True)[0]
        if min(float(g1.norm()),float(g2.norm()))<=0:raise ValueError('One history path has no gradient')
        aux,details=auxiliary(mem,active,data.public['train'][ix],data.physical['train'][ix],method=='Random-weak',918)
        loss=pred+weight*.2*aux;loss.backward()
        newgrad=float(candidate.head.cell.weight_ih.grad[:,-64:].norm())
        if newgrad<=0:raise ValueError('New recurrent history connection has no gradient')
        nn.utils.clip_grad_norm_(candidate.parameters(),1.,error_if_nonfinite=True);opt.step()
        update=float((candidate.encoder.mlp_inter[0].weight-before).abs().max())
        if update<=0:raise ValueError('Encoder did not update')
        records.append({'method':method,'mse1':float(l1.detach()),'mse2':float(l2.detach()),
                        'path1_grad':float(g1.norm()),'path2_grad':float(g2.norm()),
                        'new_history_grad':newgrad,'encoder_update':update,**details})
    first=records[0]
    if any(abs(r['mse1']-first['mse1'])>1e-6 or abs(r['mse2']-first['mse2'])>1e-6 for r in records):
        raise ValueError('Supervised initialization differs by method')
    start=time.perf_counter();initial=evaluate(model,data,valplan,device)
    old_metric=read(Path(base)/'runs/Native-U/selected_validation.json')['mse']
    if abs(initial['mse']-old_metric)>2e-5:raise ValueError('Validation warm start did not reproduce Native')
    record={'status':'PASS','version':VERSION,'test_read':False,'initial_prediction_max_error':delta,
            'source_code_max_error':code_delta,'initial_val_mse':initial['mse'],
            'old_native_val_mse':old_metric,'validation_seconds':time.perf_counter()-start,
            'support_plan_sha256':hashlib.sha256(plan.tobytes()).hexdigest(),
            'validation_plan_sha256':hashlib.sha256(valplan.tobytes()).hexdigest(),'methods':records}
    write(Path(out)/'real_batch_smoke.json',record);emit('smoke_pass',**record)


def train(out,base,method,device,epochs):
    out=Path(out);run=out/'runs'/method;run.mkdir(parents=True,exist_ok=True)
    if read(out/'real_batch_smoke.json')['status']!='PASS':raise ValueError('Real batch not verified')
    binding,data,model=setup(out,base,device);opt=optimizer_for(model)
    config={'version':VERSION,'method':method,'lambda_target':METHODS[method],
            'encoder_lr':1e-5,'head_lr':1e-4,'weight_decay':1e-4,'batch':32,
            'warmup_epochs':5,'binding_sha256':digest(out/'binding.json'),
            'code_sha256':digest(__file__),'test_read':False,'seed':0,
            'source_history_frozen':False,'prediction_supports_correct':True,
            'prediction_sets_per_step':2,'supports_per_set':3,'each_step_memory':True}
    write(run/'config.json',config)
    start=0;history=[];best=float('inf');latest=run/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']!=config:raise ValueError('Resume configuration differs')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch'];history=state['history'];best=state['best']
    valplan=data.plan('val',0,1)
    if start==0:
        initial=evaluate(model,data,valplan,device)
        write(run/'initial_validation.json',initial)
        best=initial['mse']
        save_torch(run/'selected.pt',{'model':model.state_dict(),'epoch':0,'config':config})
        write(run/'selected_validation.json',{'method':method,'epoch':0,**initial})
        emit('warm_start',method=method,mse=initial['mse'])
    for epoch in range(start+1,epochs+1):
        began=time.perf_counter();model.train()
        plan=data.plan('train',epoch,2)
        plan_hash=hashlib.sha256(plan.tobytes()).hexdigest()
        order=np.random.default_rng(771+epoch).permutation(len(data.rows['train']['ids']))
        totals={};n=0;first_details=None
        weight=METHODS[method]*min(epoch/5,1)
        for off in range(0,len(order),32):
            ix=order[off:off+32];opt.zero_grad(set_to_none=True)
            pred,mem,mask,l1,l2=prediction_pair(model,data,ix,plan,device)
            aux,metrics=auxiliary(mem,mask,data.public['train'][ix],data.physical['train'][ix],
                                  method=='Random-weak',900000+epoch*10000+off)
            loss=pred+weight*aux
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if off==0:
                encoder_grad=sum(float(p.grad.detach().square().sum()) for p in model.encoder.parameters() if p.grad is not None)**.5
                first_details={'encoder_grad':encoder_grad,
                    'new_history_grad':float(model.head.cell.weight_ih.grad[:,-64:].norm()),
                    'mse1':float(l1.detach()),'mse2':float(l2.detach())}
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            vals={'train_mse':float(pred.detach()),'aux':float(aux.detach()),'weighted_aux':float((weight*aux).detach()),
                  'total_loss':float(loss.detach()),'grad_norm':float(norm),**metrics}
            for key,value in vals.items():totals[key]=totals.get(key,0.)+value*len(ix)
            n+=len(ix)
        metric=evaluate(model,data,valplan,device)
        record={'epoch':epoch,'seconds':time.perf_counter()-began,'lambda':weight,
                'support_plan_sha256':plan_hash,'first_batch':first_details,
                **{k:v/n for k,v in totals.items()},
                **{k:v for k,v in metric.items() if k not in ['per_recipient_mse','ids']}}
        history.append(record)
        if metric['mse']<best:
            best=metric['mse']
            save_torch(run/'selected.pt',{'model':model.state_dict(),'epoch':epoch,'config':config})
            write(run/'selected_validation.json',{'method':method,'epoch':epoch,**metric})
        save_torch(latest,{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,
                           'config':config,'history':history,'best':best})
        write(run/'progress.json',{'method':method,'status':'RUNNING','epoch':epoch,
                                  'best_mse':best,'history':history,'test_read':False})
        emit('epoch',method=method,**record)
    complete={'method':method,'epochs':max(start,epochs),'best_mse':best,'test_read':False,'status':'COMPLETE'}
    write(run/'complete.json',complete)
    write(run/'progress.json',{**complete,'epoch':max(start,epochs),'history':history})
    emit('training_complete',**complete)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','train'])
    p.add_argument('--out',required=True);p.add_argument('--base',required=True)
    p.add_argument('--method',choices=list(METHODS));p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='prepare':prepare(a.out,a.base)
    elif a.command=='smoke':smoke(a.out,a.base,a.device)
    else:train(a.out,a.base,a.method,a.device,a.epochs)
