"""Resumable RSSM source training using a private snapshot of the tested runner.

Only the IO/checkpoint loop is reused. Model, ELBO, prior forecasting, binding,
normalization and smoke are the explicit RSSM adapters below. No live JEPA file
is imported or modified. Fixed epoch10/25/50 sources feed frozen P64 readouts.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import shared_io as runner
from models import VERSION,ModelConfig,make_model
from prototype import normalize_prepare,batch_objective as objective,evaluate

METHODS=('Base','Cross','Align','Both','Random-Both')


def checked_binding(args,data):
    folder=Path(args.out)/'RSSM'
    files=dict(data.files)
    for name in ('train.py','models.py','components.py','prototype.py','shared_io.py','PROFILE.json'):
        path=Path(__file__).with_name(name);files[str(path.resolve())]=runner.digest(path)
    for path in (Path(__file__).parents[1]/'cophy_relations.py',Path(args.relation_index),folder/'normalization.json'):
        files[str(path.resolve())]=runner.digest(path)
    if args.protocol:files[str(Path(args.protocol).resolve())]=runner.digest(args.protocol)
    body={'version':VERSION,'scene':args.scene,'features':str(data.path),'files':files,
        'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.lr,
        'weight_decay':1e-4,'clip_norm':1.,'model_config':asdict(ModelConfig()),
        'source_task':'P(full AB) + posterior(CD[:3], P=0) -> prior-only future visual features',
        'normalization':runner.read(folder/'normalization.json'),'source_supports':1,
        'common_objective':'posterior feature reconstruction + prior rollout + 0.01 balanced KL',
        'cross_objective':'replace focal P only and evaluate pure-prior focal forecast against recipient target',
        'source_selection':'fixed epoch50; 10/25 diagnostic snapshots; no latent-MSE model selection',
        'frontend':'shared frozen supervised official CNN784; no coordinate labels in source training',
        'prefix_state_uses_history':False,'future_posterior_only_for_ELBO':True,
        'future_posterior_in_forecast':False,'coordinate_labels_read':False,'test_read':False}
    body['sha256']=hashlib.sha256(runner.canonical(body).encode()).hexdigest()
    return body


def setup(args):
    torch.set_num_threads(args.threads)
    data=runner.FeatureData(args.features,args.scene);binding=checked_binding(args,data)
    folder=Path(args.out)/'RSSM'
    if runner.read(folder/'binding.json')!=binding:raise ValueError('RSSM source binding changed')
    runner.seeded(args.seed);model=make_model().to(args.device)
    with np.load(folder/'normalization.npz') as z:model.set_normalization(z['mean'],z['scale'])
    planner=runner.EpochPlanner(data,args.relation_index,args.seed)
    return data,binding,model,planner


def prepare(args):
    folder=Path(args.out)/'RSSM';folder.mkdir(parents=True,exist_ok=True)
    data=runner.FeatureData(args.features,args.scene)
    normalize_prepare(data,folder);binding=checked_binding(args,data)
    path=folder/'binding.json'
    if path.exists() and runner.read(path)!=binding:raise ValueError('Do not overwrite a changed RSSM profile')
    runner.write(path,binding)
    _,_,model,planner=setup(args)
    init={'version':VERSION,'model_config':model.artifact_config(),'state_sha256':runner.state_sha(model),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'pretrained_dynamics':False,'test_read':False}
    runner.write(folder/'initialization.json',init)
    plan=planner.make(1,args.batch_size)
    runner.write(folder/'plan_epoch1.json',{key:plan[key] for key in ('plan_sha256','query_exposures','paired','randomization_skipped','random_same')})
    runner.emit('PREPARED',family='RSSM',scene=args.scene,binding_sha256=binding['sha256'],initialization=init)


def smoke(args):
    began=time.monotonic();data,binding,model,planner=setup(args)
    plan=planner.make(1,args.batch_size);batch=plan['batches'][0]
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};records=[];common=None
    for method in METHODS:
        model.load_state_dict(initial);runner.seeded(args.seed);model.train();model.zero_grad(set_to_none=True)
        loss,metrics,detail=objective(model,data,plan,batch,method)
        if not torch.isfinite(loss):raise FloatingPointError('RSSM real batch nonfinite')
        if common is None:common=metrics['self_loss']
        elif not np.isclose(common,metrics['self_loss'],rtol=1e-6):raise ValueError('Different common objective across arms')
        gradients={}
        for name,tensor in (('recipient_p',detail['own']),('donor_p',detail['donor_p'])):
            if tensor is None or not len(tensor):continue
            g=torch.autograd.grad(loss,tensor,retain_graph=True,allow_unused=True)[0]
            if g is None or not torch.isfinite(g).all() or g.norm()<=0:raise ValueError('Missing '+name+' gradient')
            gradients[name]=float(g.norm())
        if detail['mixed'] is not None:
            nonfocal=~torch.nn.functional.one_hot(detail['focal'],data.slots).bool()
            torch.testing.assert_close(detail['mixed'][nonfocal],detail['own'][detail['rows']][nonfocal],atol=0,rtol=0)
        loss.backward()
        for name in ('prior','posterior','history'):
            norm=sum(float(p.grad.detach().float().square().sum()) for p in getattr(model,name).parameters() if p.grad is not None)**.5
            if not np.isfinite(norm) or norm<=0:raise ValueError('No '+name+' gradient')
            gradients[name]=norm
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('RSSM gradient nonfinite')
        records.append({'method':method,**metrics,'gradients':gradients})
    model.load_state_dict(initial);model.eval()
    ab,am,c,cm=data.context('train',batch[:4],args.device);calls=[]
    hook=model.posterior.register_forward_hook(lambda module,inputs,output:calls.append(1))
    with torch.no_grad():prediction=model.predict(model.encode(ab,am),c,cm,data.frames-3,sample=False)
    hook.remove()
    if len(calls)!=3 or prediction.shape!=(4,data.frames-3,data.slots,784):raise ValueError('RSSM forecasting context violation')
    result={'status':'PASS','version':VERSION,'scene':args.scene,'family':'RSSM',
        'binding_sha256':binding['sha256'],'rows':len(batch),'records':records,
        'forecast_posterior_frames':[0,1,2],'future_posterior_calls':0,'prefix_state_uses_history':False,
        'optimizer_updates':0,'test_read':False,'seconds':time.monotonic()-began}
    runner.write(Path(args.out)/'RSSM/real_batch_smoke.json',result)
    runner.emit('RSSM_SOURCE_SMOKE_PASS',**result)


def full_batch_objective(model,data,plan,indices,method,microbatch):
    # This initial RSSM release has no approximate per-microbatch VICReg path.
    # Fail clearly on an attempted OOM fallback; do not silently change its loss.
    if microbatch<len(indices):raise ValueError('RSSM v6.3 requires the full effective batch; change memory profile explicitly')
    return objective(model,data,plan,indices,method)


def train(args):
    # Explicit dependency injection into the private, hash-bound runner snapshot.
    runner.setup=setup
    runner.batch_objective=full_batch_objective
    runner.evaluate=evaluate
    runner.train(args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--scene',required=True,choices=('balls','collision','blocktower'))
    p.add_argument('--features',required=True);p.add_argument('--relation-index',required=True)
    p.add_argument('--out',required=True);p.add_argument('--family',choices=('RSSM',),default='RSSM')
    p.add_argument('--method',choices=METHODS,default='Base');p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=32);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--protocol');p.add_argument('--resume',action='store_true');p.add_argument('--max-steps',type=int,default=0)
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if not 1<=a.epochs<=50 or a.batch_size!=32 or a.microbatch!=32 or a.max_steps<0:
        raise ValueError('Initial RSSM release fixes effective/full batch32 and at most50 epochs')
    {'prepare':prepare,'smoke':smoke,'train':train}[a.command](a)
