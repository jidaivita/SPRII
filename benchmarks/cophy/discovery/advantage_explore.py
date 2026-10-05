"""Bounded continuation and support/context exploration; immutable v4.1 inputs."""
import argparse
import fcntl
import hashlib
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import xep_discovery as base

BASE=base.ROOT/'xep_discovery_balls_v4_1'
OUT=base.ROOT/'xep_advantage_v4_2'
SETTINGS={'standard_s3':3,'standard_s1':1,'standard_s5':5,'far_s3':3}


def describe(pose,mask,slot):
    xy=pose[:3,:,:2];other=np.flatnonzero(mask>0);other=other[other!=slot]
    distances=np.sort(np.linalg.norm(xy[0,other]-xy[0,slot],axis=-1))[:3]
    distances=np.pad(distances,(0,3-len(distances)))
    return np.r_[xy[0,slot],(xy[2,slot]-xy[0,slot])/2,distances,float(mask.sum())]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'far_pools.json').exists():return
    m=base.read(BASE/'manifest.json');features={};query={};training=[]
    for split,part in m['splits'].items():
        with open(part['cache'],'rb') as f:cache=pickle.load(f)
        z=np.load(BASE/f'input_{split}.npz',allow_pickle=False)
        ftr=np.zeros((len(part['all_ids']),9,8),np.float32)
        qtr=np.zeros((len(part['query_ids']),9,8),np.float32)
        for i,ident in enumerate(part['all_ids']):
            mask=(cache[ident]['presence_ab']>0).astype(np.float32)
            for slot in np.flatnonzero(mask):
                ftr[i,slot]=describe(cache[ident]['pose_ab'],mask,slot)
                if split=='train':training.append(ftr[i,slot])
        for i,ident in enumerate(part['query_ids']):
            for slot in np.flatnonzero(z['presence'][i]):
                qtr[i,slot]=describe(z['pose'][i],z['presence'][i],slot)
                if split=='train':training.append(qtr[i,slot])
        features[split]=ftr;query[split]=qtr
    scale=np.asarray(training).std(0).clip(.1)
    result={'source_manifest_sha256':base.digest(BASE/'manifest.json'),'feature_scale':scale.tolist(),'test_read':False,'splits':{}}
    for split,part in m['splits'].items():
        result['splits'][split]={}
        for i,ident in enumerate(part['query_ids']):
            pools=[[] for _ in range(9)]
            for slot,opts in enumerate(part['candidates'][ident]):
                if not opts:continue
                if len(opts)<5:raise ValueError('Insufficient support coverage')
                distances=np.linalg.norm((features[split][opts,slot]-query[split][i,slot])/scale,axis=-1)
                order=np.argsort(-distances,kind='stable');count=max(3,math.ceil(len(opts)*.25))
                pools[slot]=[opts[j] for j in order[:count]]
            result['splits'][split][ident]=pools
    base.write(OUT/'far_pools.json',result)
    base.emit('exploration_prepared',queries={s:len(p['query_ids']) for s,p in m['splits'].items()})


class ExploreData(base.PilotData):
    def __init__(self,method,setting):
        super().__init__(BASE,method);self.setting=setting;self.support_count=SETTINGS[setting]
        self.far=base.read(OUT/'far_pools.json')['splits'] if setting=='far_s3' else None

    def batch(self,split,ix,epoch,device):
        if self.setting=='standard_s3' or self.method in ['Query-only','Known-parameters']:
            return super().batch(split,ix,epoch,device)
        row=self.data[split];S=self.support_count
        support=np.zeros((len(ix),9,S,row['codes'].shape[-1]),np.float32)
        for b,i in enumerate(ix):
            ident=row['ids'][i]
            opts=self.far[split][ident] if self.far is not None else self.manifest['splits'][split]['candidates'][ident]
            seed=int.from_bytes(hashlib.sha256(f'xep:20260911:{split}:{ident}:{epoch}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            for k in np.flatnonzero(row['mask'][i]):
                chosen=rng.choice(opts[k],S,replace=False);support[b,k]=row['codes'][chosen,k]
        tensor=lambda a:torch.from_numpy(np.asarray(a,dtype=np.float32)).to(device)
        return tensor(row['q'][ix]),tensor(row['det'][ix]),tensor(row['mask'][ix]),tensor(support),tensor(row['target'][ix])


def train(setting,method,epochs):
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    device='cuda:0';run=OUT/'runs'/setting/method;run.mkdir(parents=True,exist_ok=True)
    lock=open(run/'run.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=ExploreData(method,setting);model=base.PredictHead(method).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    latest=run/'latest.pt';source=BASE/'runs'/method/'latest.pt'
    config={'method':method,'setting':setting,'support_count':SETTINGS[setting],'epochs_target':epochs,
            'code_sha256':base.digest(__file__),'base_code_sha256':base.digest(base.__file__),
            'source_checkpoint':str(source),'source_sha256':base.digest(source),
            'manifest_sha256':base.digest(BASE/'manifest.json'),'far_pools_sha256':base.digest(OUT/'far_pools.json'),
            'test_read':False,'stage':'validation_advantage_exploration'}
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        for key in ['method','setting','code_sha256','base_code_sha256','source_sha256','manifest_sha256','far_pools_sha256']:
            if state['config'][key]!=config[key]:raise ValueError('Resume binding changed '+key)
        start=state['epoch'];history=state['history'];best=state['best']
    else:
        state=torch.load(source,map_location=device,weights_only=False)
        if state['epoch']!=20:raise ValueError('Expected completed source epoch20')
        start=20;history=[];best=float('inf')
        if setting=='standard_s3':
            history=state['history'].copy();best=state['best']
            shutil.copy2(BASE/'runs'/method/'selected.pt',run/'selected.pt')
            shutil.copy2(BASE/'runs'/method/'selected_validation.json',run/'selected_validation.json')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);del state
    base.write(run/'config.json',config)
    if not history:
        metric=base.evaluate(model,data,device);best=metric['mse']
        history=[{'epoch':20,'phase':'before_setting_adaptation',**{k:v for k,v in metric.items() if k!='per_recipient_mse'}}]
        base.save_torch(run/'selected.pt',{'model':model.state_dict(),'epoch':20,'config':config})
        base.write(run/'selected_validation.json',{'method':method,'setting':setting,'epoch':20,'ids':data.data['val']['ids'],**metric})
    for epoch in range(start+1,epochs+1):
        began=time.perf_counter();model.train();order=np.random.default_rng(771+epoch).permutation(len(data.data['train']['ids']))
        total=0.;n=0
        for off in range(0,len(order),128):
            ix=order[off:off+128];q,det,mask,s,y=data.batch('train',ix,epoch,device)
            optimizer.zero_grad(set_to_none=True);pred=model(q,det,mask,s);loss=base.scores(pred,y,mask).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ix);n+=len(ix)
        metric=base.evaluate(model,data,device)
        record={'epoch':epoch,'train_mse':total/n,'seconds':time.perf_counter()-began,**{k:v for k,v in metric.items() if k!='per_recipient_mse'}}
        history.append(record)
        if metric['mse']<best:
            best=metric['mse'];base.save_torch(run/'selected.pt',{'model':model.state_dict(),'epoch':epoch,'config':config})
            base.write(run/'selected_validation.json',{'method':method,'setting':setting,'epoch':epoch,'ids':data.data['val']['ids'],**metric})
        base.save_torch(latest,{'config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'history':history})
        base.write(run/'progress.json',{'method':method,'setting':setting,'epoch':epoch,'best_mse':best,'history':history,'status':'RUNNING'})
        base.emit('epoch',method=method,setting=setting,**record)
    base.write(run/'complete.json',{'method':method,'setting':setting,'epochs':epochs,'best_mse':best,'test_read':False})
    base.emit('complete',method=method,setting=setting,epochs=epochs,best_mse=best)


def collect():
    report={'test_read':False,'collected_at':time.time(),'settings':{}}
    for setting in SETTINGS:
        rows={}
        for directory in sorted((OUT/'runs'/setting).glob('*')):
            if not (directory/'selected_validation.json').exists():continue
            chosen=base.read(directory/'selected_validation.json');progress=base.read(directory/'progress.json')
            rows[directory.name]={'trained_epoch':progress['epoch'],'complete':(directory/'complete.json').exists(),'selected_epoch':chosen['epoch'],
                                 **{k:chosen[k] for k in ['mse','fde','thirds','per_recipient_mse','ids']}}
        report['settings'][setting]=rows
    base.write(OUT/'result_snapshot.json',report)
    for setting,rows in report['settings'].items():
        n=rows.get('Native-U',{}).get('mse')
        base.emit('setting_result',setting=setting,rows={k:{a:b for a,b in v.items() if a not in ['per_recipient_mse','ids']} for k,v in rows.items()},
                  AP_gain=None if n is None or 'A-P' not in rows else 100*(n-rows['A-P']['mse'])/n)


def dispatch():
    OUT.mkdir(parents=True,exist_ok=True);lock=open(OUT/'dispatch.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
    waves=[[("standard_s3",m) for m in ['Native-U','A-P','A-U','Known-parameters']],
           [(s,m) for s in ['standard_s1','standard_s5'] for m in ['Native-U','A-P']],
           [('far_s3','Native-U'),('far_s3','A-P'),('standard_s3','Query-only')]]
    status={'status':'RUNNING','test_read':False,'waves':[]}
    for wave,tasks in enumerate(waves):
        jobs=[];status['waves'].append(jobs);live=[]
        for gpu,(setting,method) in enumerate(tasks):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
            log=open(logdir/f'{setting}_{method}.log','a',buffering=1)
            process=subprocess.Popen([sys.executable,'-u',__file__,'train','--setting',setting,'--method',method,'--epochs','60'],env=env,stdout=log,stderr=subprocess.STDOUT)
            job={'setting':setting,'method':method,'gpu':gpu,'pid':process.pid,'status':'RUNNING'};jobs.append(job);live.append((process,log,job))
        base.write(OUT/'dispatch_status.json',status);base.emit('wave_started',wave=wave,jobs=jobs)
        while live:
            for p,log,job in live[:]:
                rc=p.poll()
                if rc is not None:
                    job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);log.close();live.remove((p,log,job));base.write(OUT/'dispatch_status.json',status)
            if live:time.sleep(3)
        collect()
        if any(j['exit_code']!=0 for j in jobs):
            status['status']='FAILED';base.write(OUT/'dispatch_status.json',status);raise RuntimeError('Wave failed')
    status['status']='COMPLETE';base.write(OUT/'dispatch_status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['dispatch','train','collect']);p.add_argument('--setting',choices=list(SETTINGS));p.add_argument('--method');p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    if a.command=='dispatch':dispatch()
    elif a.command=='train':train(a.setting,a.method,a.epochs)
    else:collect()
