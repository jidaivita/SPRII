"""Random-relation source, correct support at downstream train and evaluation."""
import argparse
import fcntl
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import xep_discovery as base
from advantage_explore import BASE,OUT,ExploreData,collect as collect_original

RAND=base.ROOT/'xep_random_reference_v4_3'
SETTING='standard_s3'
MAPPING={'Random-U':'A-U','Random-P':'A-P'}


@torch.no_grad()
def prepare():
    from cophy_adapter import PTCoPhy,ABObservation
    from cf_learning.model import CoPhyNet
    RAND.mkdir(parents=True,exist_ok=True)
    if (RAND/'source.json').exists():return
    torch.set_num_threads(4)
    m=base.read(BASE/'manifest.json');path=base.ROOT/'runs_seed0/balls/Random/model_state_dict.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['run_config']['method']!='Random' or state['run_config']['data_binding']['preflight_sha256']!=m['preflight_sha256']:
        raise ValueError('Wrong source checkpoint binding')
    entry={'path':str(path),'sha256':base.digest(path),'epoch':state['epoch']}
    model=PTCoPhy(CoPhyNet(9),'Random').to('cuda:0').eval();model.load_state_dict(state['model'],strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    del state
    for split,part in m['splits'].items():
        with open(part['cache'],'rb') as f:cache=pickle.load(f)
        ids=part['all_ids'];codes=[];presence=[]
        for off in range(0,len(ids),128):
            batch=ids[off:off+128]
            x=torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in batch])).to('cuda:0')
            mask=torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in batch])).to('cuda:0')
            codes.append(model.encode_ab(ABObservation(x,mask)).cpu().numpy());presence.append(mask.cpu().numpy())
        base.save_npz(RAND/'codes'/f'{split}.npz',ids=np.array(ids),u=np.concatenate(codes),presence=np.concatenate(presence))
    for setting in ['standard_s3','standard_s1']:
        base.write(RAND/setting/'manifest.json',{'version':'random-reference-v4.3','setting':setting,'source':entry,
            'base_manifest':str(BASE/'manifest.json'),'base_manifest_sha256':base.digest(BASE/'manifest.json'),
            'wrapper_sha256':base.digest(__file__),'support_relation':'correct','test_read':False,
            'curriculum':'20epochs standard_s3, then40epochs under named setting'})
    base.write(RAND/'source.json',entry)


class RandomData(ExploreData):
    def __init__(self,out,method):
        super().__init__(MAPPING[method],SETTING)
        mean=scale=None
        for split,row in self.data.items():
            z=np.load(RAND/'codes'/f'{split}.npz',allow_pickle=False)
            if list(map(str,z['ids']))!=self.manifest['splits'][split]['all_ids']:raise ValueError('Random code order mismatch')
            codes=z['u'].copy()
            if method=='Random-P':codes=codes[:,:,:16]
            if split=='train':
                observed=codes[z['presence']>0];mean=observed.mean(0);scale=observed.std(0);scale[scale<1e-6]=1.
            row['codes']=(codes-mean)/scale
        self.code_mean=mean;self.code_scale=scale
        self.manifest['source_models']={'Random':base.read(RAND/'source.json')}


def train(setting,method,epochs):
    global SETTING
    SETTING=setting;root=RAND/setting
    root.mkdir(parents=True,exist_ok=True)
    lock=open(root/(method+'.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original_head=base.PredictHead
    base.PilotData=RandomData
    base.PredictHead=lambda name:original_head(MAPPING[name])
    base.train(root,method,'cuda:0',epochs)


def fork_twenty():
    for method in MAPPING:
        destination=RAND/'standard_s1'/'runs'/method;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'latest.pt').exists():continue
        source=RAND/'standard_s3'/'runs'/method/'latest.pt'
        state=torch.load(source,map_location='cpu',weights_only=False)
        if state['epoch']!=20:raise ValueError('Expected20epoch standard source for fork')
        state['best']=float('inf');state['history']=[]
        base.save_torch(destination/'latest.pt',state)
        base.write(destination/'fork.json',{'source':str(source),'source_sha256':base.digest(source),'source_epoch':20,'new_setting':'standard_s1',
            'old_setting_validation_scores_not_reused':True,'test_read':False})


def collect():
    report={'test_read':False,'source':base.read(RAND/'source.json'),'settings':{}}
    for setting in ['standard_s3','standard_s1']:
        rows={}
        for method in MAPPING:
            root=RAND/setting/'runs'/method
            if not (root/'selected_validation.json').exists():continue
            chosen=base.read(root/'selected_validation.json');progress=base.read(root/'progress.json')
            rows[method]={'trained_epoch':progress['epoch'],'complete':(root/'complete.json').exists(),'selected_epoch':chosen['epoch'],
                **{k:chosen[k] for k in ['mse','fde','thirds','per_recipient_mse','ids']}}
        report['settings'][setting]=rows
    base.write(RAND/'result_snapshot.json',report)
    for setting,rows in report['settings'].items():
        base.emit('random_result',setting=setting,rows={m:{k:v for k,v in row.items() if k not in ['per_recipient_mse','ids']} for m,row in rows.items()})


def dispatch():
    RAND.mkdir(parents=True,exist_ok=True);lock=open(RAND/'dispatch.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare();collect_original()
    if not (OUT/'result_60epochs.json').exists():shutil.copy2(OUT/'result_snapshot.json',OUT/'result_60epochs.json')
    status={'status':'RUNNING','test_read':False,'waves':[]};base.write(RAND/'dispatch_status.json',status)
    waves=[[('standard_s3',m,20,g) for m,g in [('Random-U',0),('Random-P',2)]],
           [(s,m,60,g) for g,(s,m) in enumerate([(s,m) for s in ['standard_s3','standard_s1'] for m in MAPPING])]]
    for wave,tasks in enumerate(waves):
        if wave==1:fork_twenty()
        live=[];jobs=[];status['waves'].append(jobs)
        for setting,method,epochs,gpu in tasks:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            logs=RAND/'logs';logs.mkdir(exist_ok=True)
            log=open(logs/f'{setting}_{method}_{epochs}.log','a',buffering=1)
            process=subprocess.Popen([sys.executable,'-u',__file__,'train','--setting',setting,'--method',method,'--epochs',str(epochs)],env=env,stdout=log,stderr=subprocess.STDOUT)
            job={'setting':setting,'method':method,'epochs':epochs,'gpu':gpu,'pid':process.pid,'status':'RUNNING'};jobs.append(job);live.append((process,log,job))
        base.write(RAND/'dispatch_status.json',status);base.emit('random_wave_started',wave=wave,jobs=jobs)
        while live:
            for process,log,job in live[:]:
                rc=process.poll()
                if rc is not None:
                    job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);log.close();live.remove((process,log,job));base.write(RAND/'dispatch_status.json',status)
            if live:time.sleep(3)
        if any(j['exit_code']!=0 for j in jobs):
            status['status']='FAILED';base.write(RAND/'dispatch_status.json',status);raise RuntimeError('Random reference failed')
    collect();status['status']='COMPLETE';base.write(RAND/'dispatch_status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['dispatch','train','collect']);p.add_argument('--setting',choices=['standard_s1','standard_s3']);p.add_argument('--method',choices=list(MAPPING));p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    if a.command=='dispatch':dispatch()
    elif a.command=='train':train(a.setting,a.method,a.epochs)
    else:collect()
