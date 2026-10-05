"""Prepare a fixed feature scale and check a real RSSM batch; no optimizer CLI.

The functions batch_objective and evaluate are the source-training handoff.
Coordinate targets and physical parameter values never enter either function.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from models import VERSION,ModelConfig,make_model,relation_loss,replace_focal_p
from shared_io import FeatureData,EpochPlanner,read,write,digest,seeded,state_sha,masked_mse

METHODS=('Base','Cross','Align','Both','Random-Both')


def normalize_prepare(data,out):
    cfg=ModelConfig();dest=Path(out)/'normalization.npz';manifest=Path(out)/'normalization.json'
    ids=data.ids['train']
    chosen=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('rssm-normalize/'+ids[i]).encode()).hexdigest())[:cfg.normalization_train_episodes]
    binding={'version':VERSION,'split':'train','ids':[ids[i] for i in chosen],
        'streams':['features_ab','features_cd'],'all_times':True,'visible_objects_only':True,
        'channelwise':784,'scale_floor':cfg.normalization_scale_floor,'test_read':False}
    if manifest.exists():
        previous=read(manifest)
        if previous['definition']!=binding or previous['normalization_sha256']!=digest(dest):
            raise ValueError('Normalization changed')
        return previous
    total=np.zeros(784,np.float64);squares=np.zeros(784,np.float64);n=0
    for first in range(0,len(chosen),8):
        ix=np.asarray(chosen[first:first+8])
        for suffix in ('ab','cd'):
            array=data.arrays['train']['features_'+suffix]
            mask=np.asarray(data.arrays['train']['presence_'+suffix][ix],bool)
            values=np.asarray(array[ix],np.float64)[mask]
            total+=values.sum(0);squares+=np.square(values).sum(0);n+=len(values)
    if n<2:raise ValueError('Insufficient train feature normalization support')
    mean=total/n;std=np.sqrt(np.maximum(squares/n-mean*mean,0))
    scale=std.clip(cfg.normalization_scale_floor)
    np.savez(dest,mean=mean.astype(np.float32),scale=scale.astype(np.float32))
    record={'status':'COMPLETE','definition':binding,'observed_tokens':n,
        'normalization_sha256':digest(dest),'normalization_file':str(dest),
        'target_rule':'(f784 - train_mean784) / train_scale784; mean squared error over channels',
        'scale_min':float(scale.min()),'scale_max':float(scale.max()),'test_read':False}
    write(manifest,record);return record


def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=FeatureData(args.features,args.scene)
    normalizer=normalize_prepare(data,out)
    planner=EpochPlanner(data,args.relation_index,args.seed);plan=planner.make(1,args.batch_size)
    files=dict(data.files)
    for path in [Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('components.py'),
                 Path(__file__).with_name('shared_io.py'),Path(__file__).parents[1]/'cophy_relations.py',
                 Path(args.relation_index),out/'normalization.json']:
        files[str(path.resolve())]=digest(path)
    binding={'version':VERSION,'scene':args.scene,'files':files,'config':asdict(ModelConfig()),
        'seed':args.seed,'batch_size':args.batch_size,'normalization':normalizer,
        'method_names':METHODS,'query_frames':3,'forecast_future_posterior':False,
        'future_posterior_train_only':True,'coordinate_labels_read':False,'test_read':False}
    binding=json.loads(json.dumps(binding))
    binding['sha256']=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    marker=out/'binding.json'
    if marker.exists() and read(marker)!=binding:raise ValueError('Prototype binding changed; use a fresh directory')
    write(marker,binding)
    seeded(args.seed);model=make_model()
    with np.load(out/'normalization.npz') as z:model.set_normalization(z['mean'],z['scale'])
    write(out/'initialization.json',{'version':VERSION,'state_sha256':state_sha(model),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'model_config':model.artifact_config(),'test_read':False})
    print(json.dumps({'event':'PREPARED','scene':args.scene,'binding_sha256':binding['sha256'],
        'normalization_tokens':normalizer['observed_tokens'],'paired':plan['paired'],'optimizer_updates':0}),flush=True)
    return data,planner,binding


def masked_mean(values,mask):return (values.float()*mask).sum()/mask.sum().clamp_min(1)


def batch_objective(model,data,plan,indices,method,microbatch=None):
    device=next(model.parameters()).device
    ab,am,c,cm=data.context('train',indices,device)
    full_cd,mask_cd=data.target('train',indices,device)
    own=model.encode(ab,am)
    prefix,active=model.filter_prefix(c,cm,sample=True)
    mask=mask_cd[:,3:] & active[:,None]
    target=model.target(full_cd)
    # This prior rollout is common to every arm, including Base.
    prior_prediction=model.imagine(own,prefix,active,data.frames-3,sample=True)
    rollout=masked_mse(prior_prediction,target,mask)
    # Full D is only consumed by this explicitly training-only posterior branch.
    reconstructed,kl,raw_kl=model.observe_future(own,prefix,active,full_cd,mask_cd,sample=True)
    reconstruction=masked_mse(reconstructed,target,mask)
    kl_loss=masked_mean(kl,mask)
    cfg=model.config
    common=cfg.rollout_weight*rollout+cfg.reconstruction_weight*reconstruction+cfg.kl_weight*kl_loss
    zero=own.sum()*0;cross=align=zero;donor_p=mixed=None
    rows=torch.empty(0,device=device,dtype=torch.long);focal=rows;align_stats={}
    if method!='Base':
        local=np.flatnonzero(plan['focal'][indices]>=0)
        rows=torch.as_tensor(local,device=device)
        focal=torch.as_tensor(plan['focal'][indices[local]],device=device)
        donors=plan['random' if method=='Random-Both' else 'correct'][indices[local]]
        donor_ab,donor_mask=data.history('train',donors,device)
        donor_all=model.encode(donor_ab,donor_mask)
        donor_p=donor_all[torch.arange(len(rows),device=device),focal]
        legal=(am[rows].any(1)[torch.arange(len(rows),device=device),focal]&active[rows,focal]
               &donor_mask.any(1)[torch.arange(len(rows),device=device),focal])
        rows,focal,donor_p=rows[legal],focal[legal],donor_p[legal]
        if len(rows):
            if method in ('Cross','Both','Random-Both'):
                mixed=replace_focal_p(own,rows,focal,donor_p)
                cross_pred=model.imagine(mixed,{k:v[rows] for k,v in prefix.items()},active[rows],data.frames-3,sample=True)
                cross=masked_mse(cross_pred[torch.arange(len(rows),device=device),:,focal],target[rows,:,focal],mask[rows,:,focal])
            if method in ('Align','Both','Random-Both'):
                align,align_stats=relation_loss(own[rows,focal],donor_p)
    loss=common+cfg.lambda_cross*cross+cfg.lambda_align*align
    metrics={'loss':float(loss.detach()),'self_loss':float(common.detach()),
        'prior_rollout_mse':float(rollout.detach()),'posterior_reconstruction_mse':float(reconstruction.detach()),
        'raw_kl_nats':float(masked_mean(raw_kl,mask).detach()),'free_kl_nats':float(kl_loss.detach()),
        'weighted_kl':float((cfg.kl_weight*kl_loss).detach()),'cross':float(cross.detach()),
        'align':float(align.detach()),'paired':len(rows),**align_stats}
    detail={'own':own,'prefix':prefix,'donor_p':donor_p,'mixed':mixed,'rows':rows,'focal':focal,
        'prior_prediction':prior_prediction,'target':target,
        'terms':{'self':common,'cross':cfg.lambda_cross*cross,'align':cfg.lambda_align*align}}
    return loss,metrics,detail


@torch.no_grad()
def evaluate(model,data,batch_size=32,limit=512):
    model.eval();device=next(model.parameters()).device;values=[];ids=[]
    count=min(limit,len(data.ids['val'])) if limit else len(data.ids['val'])
    for first in range(0,count,batch_size):
        ix=np.arange(first,min(first+batch_size,count))
        ab,am,c,cm=data.context('val',ix,device)
        # Predict before loading future targets. Only the first three frames
        # can enter filtering; the remaining time steps call the prior alone.
        prediction=model.predict(model.encode(ab,am),c,cm,data.frames-3,sample=False)
        full_cd,mask_cd=data.target('val',ix,device)
        mask=mask_cd[:,3:] & cm.any(1)[:,None]
        error=(prediction-model.target(full_cd)).square().mean(-1)
        denom=mask.sum((1,2));score=(error*mask).sum((1,2))/denom.clamp_min(1)
        for i,value,n in zip(ix,score.tolist(),denom.tolist()):
            if n:ids.append(data.ids['val'][i]);values.append(value)
    return {'mse':float(np.mean(values)),'ids':ids,'per_recipient_mse':values,
        'future_posterior_calls':0,'forecast':'deterministic prior means','test_read':False}


def smoke(args):
    began=time.monotonic();data,planner,binding=prepare(args)
    model=make_model().to(args.device)
    with np.load(Path(args.out)/'normalization.npz') as z:model.set_normalization(z['mean'],z['scale'])
    seeded(args.seed)
    # Reset after normalization and before the common seeded initialization.
    model=make_model().to(args.device)
    with np.load(Path(args.out)/'normalization.npz') as z:model.set_normalization(z['mean'],z['scale'])
    if state_sha(model)!=read(Path(args.out)/'initialization.json')['state_sha256']:raise ValueError('Initial RSSM differs')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    plan=planner.make(1,args.batch_size);batch=plan['batches'][0]
    records=[];common=None
    for method in METHODS:
        model.load_state_dict(initial);seeded(args.seed);model.train();model.zero_grad(set_to_none=True)
        loss,metrics,detail=batch_objective(model,data,plan,batch,method)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RSSM real batch')
        if common is None:common=metrics['self_loss']
        elif not np.isclose(common,metrics['self_loss'],rtol=1e-6):raise ValueError('RSSM common objective differs across arms')
        gradients={}
        for name,value in [('recipient_p',detail['own']),('donor_p',detail['donor_p'])]:
            if value is not None and len(value):
                g=torch.autograd.grad(loss,value,retain_graph=True,allow_unused=True)[0]
                if g is None or not torch.isfinite(g).all() or g.norm()<=0:raise ValueError('Missing RSSM '+name+' gradient')
                gradients[name]=float(g.norm())
        if detail['mixed'] is not None:
            nonfocal=~torch.nn.functional.one_hot(detail['focal'],data.slots).bool()
            torch.testing.assert_close(detail['mixed'][nonfocal],detail['own'][detail['rows']][nonfocal],atol=0,rtol=0)
        loss.backward()
        for name in ('prior','posterior','history'):
            norm=sum(float(p.grad.detach().float().square().sum()) for p in getattr(model,name).parameters() if p.grad is not None)**.5
            if not np.isfinite(norm) or norm<=0:raise ValueError('No finite gradient to actual RSSM '+name)
            gradients[name]=norm
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('RSSM gradients nonfinite')
        records.append({'method':method,**metrics,'gradients':gradients})
    model.load_state_dict(initial);model.eval()
    ab,am,c,cm=data.context('train',batch[:4],args.device)
    calls=[]
    handle=model.posterior.register_forward_hook(lambda module,inputs,output:calls.append(tuple(inputs[0].shape)))
    with torch.no_grad():prediction=model.predict(model.encode(ab,am),c,cm,data.frames-3,sample=False)
    handle.remove()
    if len(calls)!=3:raise ValueError('Forecast posterior calls exceed the three input frames')
    if prediction.shape!=(4,data.frames-3,data.slots,784):raise ValueError('RSSM forecast shape changed')
    record={'status':'PASS','version':VERSION,'scene':args.scene,'binding_sha256':binding['sha256'],
        'rows':len(batch),'records':records,'forecast_shape':list(prediction.shape),
        'forecast_posterior_calls':len(calls),'forecast_posterior_frames':[0,1,2],
        'future_posterior_calls':0,'full_future_not_an_argument_to_predict':True,
        'optimizer_updates':0,'weights_discarded_after_smoke':True,'test_read':False,
        'seconds':time.monotonic()-began}
    write(Path(args.out)/'real_batch_smoke.json',record)
    print(json.dumps({'event':'RSSM_SMOKE_PASS',**record}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke'))
    p.add_argument('--scene',required=True,choices=('balls','collision','blocktower'))
    p.add_argument('--features',required=True);p.add_argument('--relation-index',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    {'prepare':prepare,'smoke':smoke}[a.command](a)
