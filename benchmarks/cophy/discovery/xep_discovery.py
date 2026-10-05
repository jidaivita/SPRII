"""Balls independent-experience prediction pilot. Never reads test data.

Preparation, frozen feature extraction, and downstream learning are separate.
Only query prefix, object presence, and independent support enter the model.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
VERSION = 'xep-discovery-balls-v4.1-impl1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    os.replace(tmp, path)


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(tmp, 'wb') as f: np.savez(f, **values)
    os.replace(tmp, path)


def save_torch(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    torch.save(value,tmp);os.replace(tmp,path)


def emit(event, **items):
    print(json.dumps({'time':time.time(),'event':event,**items},ensure_ascii=False),flush=True)


def artifact(preflight, name):
    item=preflight['artifacts'][name];path=Path(item['path'])
    if digest(path)!=item['sha256']:raise ValueError('Changed source artifact: '+name)
    return path


def prepare_manifest(out):
    out=Path(out);path=out/'manifest.json'
    if path.exists():
        m=read(path)
        if m['version']!=VERSION:raise ValueError('Manifest version differs')
        return m
    prepath=ROOT/'prepared_v3/balls/training_preflight.json';p=read(prepath)
    splits=read(artifact(p,'splits'));m={'version':VERSION,'scene':'balls4','prefix':3,'supports':3,
        'test_read':False,'preflight_sha256':digest(prepath),'data_root':str(Path(p['input_profile']['dataset_dir'])/'4'),
        'derenderer':str(artifact(p,'derenderer')),'source_models':{},'splits':{}}
    for method in ['Native','A']:
        ckpt=ROOT/'runs_seed0/balls'/method/'model_state_dict.pt'
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        if state['run_config']['method']!=method or state['run_config']['data_binding']['preflight_sha256']!=digest(prepath):
            raise ValueError('Source checkpoint binding differs')
        m['source_models'][method]={'path':str(ckpt),'sha256':digest(ckpt),'epoch':state['epoch']}
        del state
    for split in ['train','val']:
        ids=sorted(map(str,splits[split]['ids']));index={v:i for i,v in enumerate(ids)}
        rows=read(artifact(p,'raw_relations_'+split))
        cache_path=artifact(p,'cache_'+split)
        with open(cache_path,'rb') as f:cache=pickle.load(f)
        groups=defaultdict(list);byid=defaultdict(list)
        for r in rows:
            if r['split']!=split or r['id'] not in index:raise ValueError('Split mismatch')
            byid[r['id']].append(r)
            if cache[r['id']]['presence_ab'][r['slot']]>0:
                groups[(r['slot'],tuple(r['physical']))].append(index[r['id']])
        candidates={};mask={};physical={};eligible=[];excluded=[]
        for ident in ids:
            active=[r for r in byid[ident] if r['in_C']]
            lists=[[] for _ in range(9)];pm=[0.]*9;phy=[[0]*3 for _ in range(9)]
            for r in active:
                k=r['slot'];pm[k]=1.;phy[k]=r['physical']
                lists[k]=sorted(v for v in groups[(k,tuple(r['physical']))] if v!=index[ident])
            if active and all(len(lists[r['slot']])>=3 for r in active):
                eligible.append(ident);candidates[ident]=lists;mask[ident]=pm;physical[ident]=phy
            else:excluded.append(ident)
        eligible=sorted(eligible,key=lambda ident:hashlib.sha256(('xep:20260911:'+ident).encode()).digest())
        queries=eligible if split=='train' else eligible[:512]
        m['splits'][split]={'all_ids':ids,'query_ids':queries,'eligible_total':len(eligible),'excluded_ids':excluded,
            'cache':str(cache_path),'cache_sha256':digest(cache_path),'candidates':{q:candidates[q] for q in queries},
            'presence':{q:mask[q] for q in queries},'physical':{q:physical[q] for q in queries}}
        emit('coverage',split=split,total=len(ids),eligible=len(eligible),pilot_queries=len(queries))
    if set(m['splits']['train']['all_ids'])&set(m['splits']['val']['all_ids']):raise ValueError('Cross-split IDs')
    write(path,m);return m


def load_prefix(item):
    ident,folder=item
    from dataloaders.utils import get_rgb
    rgb=get_rgb(str(Path(folder)/ident/'cd'),max_frames=3)
    if rgb.shape!=(3,3,224,224):raise ValueError('Unexpected prefix shape '+str(rgb.shape))
    return ident,rgb


@torch.no_grad()
def prepare_prefix_shard(out,shard,shards):
    from derendering.model import DeRendering
    out=Path(out);m=read(out/'manifest.json');torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.manual_seed(0)
    visual=DeRendering(9).to(device).eval()
    visual.load_state_dict(torch.load(m['derenderer'],map_location='cpu',weights_only=True),strict=True)
    for split in ['train','val']:
        fn=out/'prefix'/f'{split}_{shard}.npz'
        if fn.exists():continue
        ids=m['splits'][split]['query_ids'][shard::shards]
        poses=[];seen=[];done=[];start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for off in range(0,len(ids),24):
                loaded=list(pool.map(load_prefix,[(i,m['data_root']) for i in ids[off:off+24]]))
                rgb=np.stack([a[1] for a in loaded]);B=len(rgb)
                tensor=torch.from_numpy(rgb.reshape(B*3,3,224,224)).contiguous().to(device)
                presence,pose,_=visual(tensor)
                estimate=pose.reshape(B,3,9,3).cpu().numpy()
                detection=(presence.reshape(B,3,9)>0).cpu().numpy().astype(np.float32)
                poses.extend(estimate);seen.extend(detection);done.extend(a[0] for a in loaded)
                if off%240==0:emit('prefix_progress',split=split,shard=shard,done=len(done),total=len(ids),seconds=time.perf_counter()-start)
        save_npz(fn,ids=np.array(done),pose=np.asarray(poses,np.float32),detected=np.asarray(seen,np.float32))
        emit('prefix_complete',split=split,shard=shard,queries=len(done),seconds=time.perf_counter()-start)


@torch.no_grad()
def encode_source(out,method):
    from cophy_adapter import PTCoPhy,ABObservation
    from cf_learning.model import CoPhyNet
    out=Path(out);m=read(out/'manifest.json');entry=m['source_models'][method]
    if digest(entry['path'])!=entry['sha256']:raise ValueError('Source checkpoint changed')
    state=torch.load(entry['path'],map_location='cpu',weights_only=False)
    model=PTCoPhy(CoPhyNet(9),method).to('cuda:0').eval();model.load_state_dict(state['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    del state
    for split in ['train','val']:
        fn=out/'codes'/f'{method}_{split}.npz'
        if fn.exists():continue
        part=m['splits'][split];ids=part['all_ids']
        with open(part['cache'],'rb') as f:cache=pickle.load(f)
        all_u=[];mask=[]
        for off in range(0,len(ids),128):
            batch=ids[off:off+128]
            x=torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in batch])).to('cuda:0')
            p=torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in batch])).to('cuda:0')
            u=model.encode_ab(ABObservation(x,p)).cpu().numpy();all_u.append(u);mask.append(p.cpu().numpy())
        save_npz(fn,ids=np.array(ids),u=np.concatenate(all_u),presence=np.concatenate(mask))
        emit('source_encoded',method=method,split=split,episodes=len(ids))


def merge_inputs(out,shards):
    out=Path(out);m=read(out/'manifest.json');report={'version':VERSION,'test_read':False,'splits':{}}
    for split in ['train','val']:
        data={}
        for shard in range(shards):
            z=np.load(out/'prefix'/f'{split}_{shard}.npz',allow_pickle=False)
            for ident,pose,det in zip(z['ids'],z['pose'],z['detected']):data[str(ident)]=(pose,det)
        part=m['splits'][split];ids=part['query_ids'];gt=[];presence=[];params=[]
        for ident in ids:
            state=np.load(Path(m['data_root'])/ident/'cd/states.npy',allow_pickle=False)
            if state.shape[:2]!=(30,9):raise ValueError('Unexpected target frames')
            actual=(np.abs(state[0,:,:3]).sum(-1)>0).astype(np.float32)
            if not np.array_equal(actual,np.array(part['presence'][ident])):raise ValueError('Initial presence mismatch')
            gt.append(np.array(state[3:,:,:2],np.float32));presence.append(actual)
            labels=np.asarray(part['physical'][ident]);onehot=np.eye(3,dtype=np.float32)[labels].reshape(9,9)
            params.append(onehot*actual[:,None])
        prefix=np.stack([data[i][0] for i in ids]);seen=np.stack([data[i][1] for i in ids]);presence=np.stack(presence)
        if not np.isfinite(prefix).all() or not np.isfinite(gt).all():raise ValueError('Nonfinite data')
        save_npz(out/f'input_{split}.npz',ids=np.array(ids),pose=prefix[:,:,:,:2],detected=seen,presence=presence)
        save_npz(out/f'target_{split}.npz',pose=np.stack(gt))
        save_npz(out/f'parameters_{split}.npz',values=np.stack(params))
        report['splits'][split]={'queries':len(ids),'target_shape':list(np.array(gt).shape),'prefix_indices':[0,1,2]}
    write(out/'data_ready.json',report);emit('data_ready',**report)


def module_seed(name):
    return int.from_bytes(hashlib.sha256(('xep-head0:'+name).encode()).digest()[:4],'little')


class PredictHead(nn.Module):
    def __init__(self,method,width=128):
        super().__init__();self.method=method;self.width=width
        d=16 if method=='A-P' else 32
        if method=='Known-parameters':d=9
        torch.manual_seed(module_seed('support-'+str(d)))
        self.support=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64))
        self.missing=nn.Parameter(torch.zeros(64))
        torch.manual_seed(module_seed('query'))
        self.query=nn.GRU(5,64,batch_first=True)
        torch.manual_seed(module_seed('init'))
        self.init=nn.Sequential(nn.Linear(129,width),nn.ReLU(),nn.Linear(width,width))
        torch.manual_seed(module_seed('message'))
        self.message=nn.Sequential(nn.Linear(width*2+4,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        torch.manual_seed(module_seed('dynamics'))
        self.cell=nn.GRUCell(width+4,width)
        self.delta=nn.Linear(width,2);nn.init.normal_(self.delta.weight,std=.001);nn.init.zeros_(self.delta.bias)
        self.register_buffer('xy_mean',torch.zeros(2));self.register_buffer('xy_scale',torch.ones(2))

    def forward(self,q,det,mask,support,horizon=27):
        B,L,K,_=q.shape;x=(q-self.xy_mean)/self.xy_scale
        dx=torch.cat([torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]],1)
        tokens=torch.cat([x,dx,det[...,None]],-1).permute(0,2,1,3).reshape(B*K,L,5)
        _,qh=self.query(tokens);qh=qh[0].reshape(B,K,64)
        if support is None:
            memory=self.missing.expand(B,K,64);available=torch.zeros(B,K,1,device=q.device)
        else:
            memory=self.support(support).mean(2);available=torch.ones(B,K,1,device=q.device)
        h=self.init(torch.cat([qh,memory,available],-1));position=x[:,-1];v=(x[:,-1]-x[:,0])/(L-1)
        pair=mask[:,:,None]*mask[:,None,:]*(1-torch.eye(K,device=q.device)[None])
        denom=pair.sum(2).clamp_min(1)[...,None];outputs=[]
        for _ in range(horizon):
            hi=h[:,:,None].expand(B,K,K,self.width);hj=h[:,None,:].expand(B,K,K,self.width)
            rel=position[:,None]-position[:,:,None];rv=v[:,None]-v[:,:,None]
            messages=self.message(torch.cat([hi,hj,rel,rv],-1))
            agg=(messages*pair[...,None]).sum(2)/denom
            h=self.cell(torch.cat([agg,position,v],-1).reshape(B*K,-1),h.reshape(B*K,-1)).reshape(B,K,-1)
            move=v+self.delta(h);position=position+move;v=move
            outputs.append(position*self.xy_scale+self.xy_mean)
        return torch.stack(outputs,1)


class PilotData:
    def __init__(self,out,method):
        self.out=Path(out);self.manifest=read(self.out/'manifest.json');self.method=method;self.data={}
        source='Native' if method=='Native-U' else 'A'
        learned=method in ['Native-U','A-U','A-P']
        mean=scale=None
        for split in ['train','val']:
            z=np.load(self.out/f'input_{split}.npz',allow_pickle=False)
            tar=np.load(self.out/f'target_{split}.npz',allow_pickle=False)
            params=np.load(self.out/f'parameters_{split}.npz',allow_pickle=False)
            row={'q':z['pose'].copy(),'det':z['detected'].copy(),'mask':z['presence'].copy(),
                 'ids':list(map(str,z['ids'])),'target':tar['pose'].copy(),'parameters':params['values'].copy()}
            if learned:
                c=np.load(self.out/'codes'/f'{source}_{split}.npz',allow_pickle=False)
                if list(map(str,c['ids']))!=self.manifest['splits'][split]['all_ids']:raise ValueError('Code order differs')
                codes=c['u'].copy();codes=codes[:,:,:16] if method=='A-P' else codes
                if split=='train':
                    observed=codes[c['presence']>0];mean=observed.mean(0);scale=observed.std(0);scale[scale<1e-6]=1.
                row['codes']=(codes-mean)/scale
            self.data[split]=row
        train=self.data['train'];xy=train['q'][np.broadcast_to(train['mask'][:,None,:]>0,train['q'].shape[:-1])]
        self.xy_mean=xy.mean(0);self.xy_scale=xy.std(0).clip(.1)
        self.code_mean=mean;self.code_scale=scale

    def batch(self,split,ix,epoch,device):
        row=self.data[split];support=None
        if self.method=='Known-parameters':support=row['parameters'][ix][:,:,None,:]
        elif self.method in ['Native-U','A-U','A-P']:
            support=np.zeros((len(ix),9,3,row['codes'].shape[-1]),np.float32)
            for b,i in enumerate(ix):
                ident=row['ids'][i];opts=self.manifest['splits'][split]['candidates'][ident]
                seed=int.from_bytes(hashlib.sha256(f'xep:20260911:{split}:{ident}:{epoch}'.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                for k in range(9):
                    if row['mask'][i,k]>0:
                        chosen=rng.choice(opts[k],3,replace=False);support[b,k]=row['codes'][chosen,k]
        tensor=lambda a:torch.from_numpy(np.asarray(a,dtype=np.float32)).to(device)
        return (tensor(row['q'][ix]),tensor(row['det'][ix]),tensor(row['mask'][ix]),
                None if support is None else tensor(support),tensor(row['target'][ix]))


def scores(pred,target,mask):
    sq=(pred-target).square().mean(-1)
    return (sq*mask[:,None]).sum((1,2))/(mask.sum(1)*target.shape[1]).clamp_min(1)


@torch.no_grad()
def evaluate(model,data,device):
    model.eval();mse=[];fde=[];thirds=[];hold=[];cv=[]
    for off in range(0,len(data.data['val']['ids']),128):
        ix=np.arange(off,min(off+128,len(data.data['val']['ids'])))
        q,det,mask,s,y=data.batch('val',ix,0,device);p=model(q,det,mask,s)
        mse.extend(scores(p,y,mask).cpu().tolist())
        last=(p[:,-1]-y[:,-1]).norm(dim=-1)
        fde.extend(((last*mask).sum(1)/mask.sum(1)).cpu().tolist())
        thirds.extend(torch.stack([scores(p[:,i:i+9],y[:,i:i+9],mask) for i in [0,9,18]],1).cpu().tolist())
        frozen=q[:,-1:,].expand_as(y);v=(q[:,-1]-q[:,0])/2
        times=torch.arange(1,28,device=device)[None,:,None,None]
        linear=q[:,-1:,]+times*v[:,None]
        hold.extend(scores(frozen,y,mask).cpu().tolist());cv.extend(scores(linear,y,mask).cpu().tolist())
    return {'mse':float(np.mean(mse)),'fde':float(np.mean(fde)),'thirds':np.mean(thirds,0).tolist(),
        'hold_last':float(np.mean(hold)),'constant_velocity':float(np.mean(cv)),
        'per_recipient_mse':mse,'recipients':len(mse)}


def train(out,method,device,epochs):
    out=Path(out);run=out/'runs'/method;run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8 if device=='cpu' else 4);torch.manual_seed(0);np.random.seed(0)
    data=PilotData(out,method);model=PredictHead(method).to(device)
    model.xy_mean.copy_(torch.tensor(data.xy_mean,device=device));model.xy_scale.copy_(torch.tensor(data.xy_scale,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    latest=run/'latest.pt';start=0;best=float('inf');history=[]
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['version']!=VERSION or state['code_sha256']!=digest(__file__):raise ValueError('Resume code changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch'];best=state['best'];history=state['history']
    config={'version':VERSION,'method':method,'code_sha256':digest(__file__),'manifest_sha256':digest(out/'manifest.json'),
        'device':device,'epochs_target':epochs,'seed':0,'source_models':data.manifest['source_models'],
        'source_history_frozen':True,'head_support_dropout':0.,'test_read':False,'stage':'directional_validation',
        'code_mean':None if data.code_mean is None else data.code_mean.tolist(),
        'code_scale':None if data.code_scale is None else data.code_scale.tolist()}
    write(run/'config.json',config)
    for epoch in range(start+1,epochs+1):
        began=time.perf_counter();model.train();order=np.random.default_rng(771+epoch).permutation(len(data.data['train']['ids']))
        total=0.;n=0
        for off in range(0,len(order),128):
            ix=order[off:off+128];q,det,mask,s,y=data.batch('train',ix,epoch,device)
            optimizer.zero_grad(set_to_none=True);pred=model(q,det,mask,s);loss=scores(pred,y,mask).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ix);n+=len(ix)
        metric=evaluate(model,data,device);record={'epoch':epoch,'train_mse':total/n,'seconds':time.perf_counter()-began,
            **{k:v for k,v in metric.items() if k!='per_recipient_mse'}};history.append(record)
        if metric['mse']<best:
            best=metric['mse'];save_torch(run/'selected.pt',{'model':model.state_dict(),'epoch':epoch,'version':VERSION,'config':config})
            write(run/'selected_validation.json',{'method':method,'epoch':epoch,'ids':data.data['val']['ids'],**metric})
        save_torch(latest,{'version':VERSION,'code_sha256':digest(__file__),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'best':best,'history':history})
        write(run/'progress.json',{'method':method,'epoch':epoch,'best_mse':best,'history':history,'status':'RUNNING','test_read':False})
        emit('epoch',method=method,**record)
    write(run/'complete.json',{'method':method,'epochs':epochs,'best_mse':best,'test_read':False})
    emit('training_complete',method=method,best_mse=best,epochs=epochs)


def smoke(out):
    """One real batch: finite training gradients and causal inputs, not a test suite."""
    torch.set_num_threads(4)
    results=[]
    for method in ['Native-U','A-U','A-P','Known-parameters','Query-only']:
        data=PilotData(out,method);model=PredictHead(method)
        model.xy_mean.copy_(torch.tensor(data.xy_mean));model.xy_scale.copy_(torch.tensor(data.xy_scale))
        q,det,mask,s,y=data.batch('train',np.arange(2),1,'cpu')
        pred=model(q,det,mask,s);loss=scores(pred,y,mask).mean();loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Invalid real batch '+method)
        model.eval()
        with torch.no_grad():
            before=model(q,det,mask,s);y.fill_(9876.);after=model(q,det,mask,s)
            if not torch.equal(before,after):raise ValueError('Targets affect model forward')
        results.append({'method':method,'shape':list(pred.shape),'finite_loss':float(loss.detach())})
    write(Path(out)/'real_batch_smoke.json',{'status':'PASS','results':results,'test_read':False})
    emit('real_batch_smoke_pass')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['manifest','prefix','encode','merge','smoke','train'])
    p.add_argument('--out',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--method');p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args()
    if a.command=='manifest':prepare_manifest(a.out)
    elif a.command=='prefix':prepare_prefix_shard(a.out,a.shard,a.shards)
    elif a.command=='encode':encode_source(a.out,a.method)
    elif a.command=='merge':merge_inputs(a.out,a.shards)
    elif a.command=='smoke':smoke(a.out)
    elif a.command=='train':train(a.out,a.method,a.device,a.epochs)
