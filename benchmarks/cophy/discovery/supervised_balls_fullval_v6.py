"""Fixed supervised Balls heads on all validation recipients, no new training.

Copies the old 512 inputs exactly, extracts only the missing RGB prefixes with
the original frozen FP32 frontend, then scores old strong/Random and v5.1 heads.
"""

import os
import argparse
from collections import defaultdict
import concurrent.futures
import fcntl
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import xep_discovery as xep

VERSION='supervised-balls-fullval-v6-1'
METHODS=('Native','Cross-only','Align-only','Both-new','Random-Both-new')


def checks(files):
    for path,sha in files.items():
        if xep.digest(path)!=sha: raise ValueError('Changed bound file: '+str(path))


def make_plan(part,ids):
    plan=np.zeros((len(ids),9,8),np.int64);lookup={q:i for i,q in enumerate(part['all_ids'])}
    for i,q in enumerate(ids):
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'xep:20260911:val:{q}:0'.encode()).digest()[:8],'little'))
        for k in range(9):
            if part['presence'][q][k]<=0:continue
            options=part['candidates'][q][k]
            if len(options)<8 or lookup[q] in options:raise ValueError('Invalid independent S8 candidate pool')
            first=rng.choice(options,3,replace=False);rest=[v for v in options if v not in set(first.tolist())]
            seed=int.from_bytes(hashlib.sha256(f'source-formation-readout-v51:val:{q}:{k}:0'.encode()).digest()[:8],'little')
            plan[i,k]=np.concatenate([first,np.random.default_rng(seed).choice(rest,5,replace=False)])
    return plan


def prepare_metadata(args):
    base,out=args.base,args.out;out.mkdir(parents=True,exist_ok=True)
    original=xep.read(base/'manifest.json');old=original['splits']['val'];root=base.parent
    prepath=root/'prepared_v3/balls/training_preflight.json';pre=xep.read(prepath)
    if xep.digest(prepath)!=original['preflight_sha256']:raise ValueError('Preflight changed')
    all_ids=old['all_ids'];lookup={q:i for i,q in enumerate(all_ids)}
    relation=xep.artifact(pre,'raw_relations_val');cachepath=xep.artifact(pre,'cache_val')
    rows=xep.read(relation)
    with open(cachepath,'rb') as stream:cache=pickle.load(stream)
    groups,byid=defaultdict(list),defaultdict(list)
    for r in rows:
        if r['split']!='val' or r['id'] not in lookup:raise ValueError('Invalid validation metadata')
        byid[r['id']].append(r)
        if cache[r['id']]['presence_ab'][r['slot']]>0:groups[(r['slot'],tuple(r['physical']))].append(lookup[r['id']])
    del cache
    part=dict(all_ids=all_ids,candidates={},presence={},physical={});eligible=[];excluded=[]
    for q in all_ids:
        active=[r for r in byid[q] if r['in_C']];p=[0.]*9;c=[[] for _ in range(9)];physical=[[0]*3 for _ in range(9)]
        for r in active:
            k=r['slot'];p[k]=1.;physical[k]=r['physical'];c[k]=sorted(v for v in groups[(k,tuple(r['physical']))] if v!=lookup[q])
        if not active or min(len(c[r['slot']]) for r in active)<3:excluded.append(q);continue
        if min(len(c[r['slot']]) for r in active)<8:raise ValueError('Original eligible query lacks S8: '+q)
        eligible.append(q);part['candidates'][q]=c;part['presence'][q]=p;part['physical'][q]=physical
    if len(eligible)!=old['eligible_total'] or sorted(excluded)!=sorted(old['excluded_ids']):raise ValueError('Eligibility changed')
    for key in ('candidates','presence','physical'):
        if any(part[key][q]!=old[key][q] for q in old['query_ids']):raise ValueError('Original metadata changed')
    oldset=set(old['query_ids']);remaining=sorted((q for q in eligible if q not in oldset),key=lambda q:hashlib.sha256(('xep:20260911:'+q).encode()).digest())
    ids=old['query_ids']+remaining
    if len(ids)!=2000 or len(old['query_ids'])!=512:raise ValueError('Unexpected Balls validation cohort')
    manifest=dict(version=VERSION,scene='balls4',base=str(base),test_read=False,optimizer_steps=0,
        query_ids=ids,original_query_ids=old['query_ids'],remaining_query_ids=remaining,metadata=part,
        data_root=original['data_root'],derenderer=original['derenderer'],selection='unchanged original512 checkpoints',
        file_sha256={str(p):xep.digest(p) for p in [base/'manifest.json',base/'input_val.npz',base/'target_val.npz',prepath,relation,cachepath,Path(original['derenderer'])]})
    if (out/'manifest.json').exists() and xep.read(out/'manifest.json')!=manifest:raise ValueError('Manifest differs')
    xep.write(out/'manifest.json',manifest);plan=make_plan(part,ids)
    xep.save_npz(out/'support_plan.npz',ids=np.asarray(ids),plan=plan)
    # Metadata and support plan are written before future targets are opened.
    return original,old,remaining,part,ids


def prepare_targets(args):
    out=args.out
    if (out/'targets_missing_complete.json').exists():
        receipt=xep.read(out/'targets_missing_complete.json');checks(receipt['file_sha256']);return
    original,old,remaining,part,ids=prepare_metadata(args)
    began=time.time();targets=[];masks=[]
    for ident in remaining:
        state=np.load(Path(original['data_root'])/ident/'cd/states.npy',allow_pickle=False)
        if state.shape[:2]!=(30,9):raise ValueError('Unexpected Balls target shape')
        mask=(np.abs(state[0,:,:3]).sum(-1)>0).astype(np.float32)
        if not np.array_equal(mask,np.asarray(part['presence'][ident])):raise ValueError('Public object mask mismatch')
        masks.append(mask);targets.append(np.asarray(state[3:,:,:2],np.float32))
    xep.save_npz(out/'targets_missing.npz',ids=np.asarray(remaining),pose=np.asarray(targets),presence=np.asarray(masks))
    receipt=dict(status='COMPLETE',version=VERSION,ids=remaining,test_read=False,model_forward_calls=0,gpu_used=False,
        seconds=time.time()-began,file_sha256={str(out/n):xep.digest(out/n) for n in ('manifest.json','support_plan.npz','targets_missing.npz')})
    xep.write(out/'targets_missing_complete.json',receipt);xep.emit('balls_cpu_targets_complete',rows=len(remaining),seconds=receipt['seconds'])


def prepare(args):
    base,out=args.base,args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'data_ready.json').exists():
        ready=xep.read(out/'data_ready.json');checks(ready['file_sha256']);return
    original,old,remaining,part,ids=prepare_metadata(args)
    cached_targets=None
    if (out/'targets_missing_complete.json').exists():
        receipt=xep.read(out/'targets_missing_complete.json');checks(receipt['file_sha256'])
        with np.load(out/'targets_missing.npz',allow_pickle=False) as z:
            if z['ids'].tolist()!=remaining:raise ValueError('Cached future target order differs')
            cached_targets={q:(z['pose'][i].copy(),z['presence'][i].copy()) for i,q in enumerate(remaining)}
    from derendering.model import DeRendering
    torch.set_num_threads(4);torch.manual_seed(0)
    model=DeRendering(9).to(args.device).eval()
    model.load_state_dict(torch.load(original['derenderer'],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
    with np.load(base/'input_val.npz',allow_pickle=False) as z:
        if z['ids'].tolist()!=old['query_ids']:raise ValueError('Original prefix IDs changed')
        old_q,old_det,old_mask=(z[n].copy() for n in ('pose','detected','presence'))
    with np.load(base/'target_val.npz',allow_pickle=False) as z:old_target=z['pose'].copy()
    cached_prefix=None
    if args.prefix_cache is not None:
        cache_ready=xep.read(args.prefix_cache/'complete.json')
        if cache_ready['status']!='COMPLETE' or cache_ready['ids']!=remaining:raise ValueError('CPU prefix cache cohort differs')
        cache_binding=xep.read(args.prefix_cache/'binding.json');checks(cache_binding['file_sha256'])
        if xep.digest(args.prefix_cache/'binding.json')!=cache_ready['binding_sha256']:raise ValueError('CPU prefix cache binding changed')
        if cache_binding['base']!=str(base) or cache_binding['data_root']!=original['data_root']:raise ValueError('CPU prefix cache source differs')
        checks(cache_ready['file_sha256']);cached_prefix={s['ids'][0]:s for s in cache_ready['shards']}
    poses,dets,targets,masks=[],[],[],[];began=time.time()
    with torch.inference_mode(),concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for off in range(0,len(remaining),24):
            batch=remaining[off:off+24]
            if cached_prefix is None:
                loaded=list(pool.map(xep.load_prefix,[(q,original['data_root']) for q in batch]));rgb=np.stack([v[1] for v in loaded])
            else:
                with np.load(cached_prefix[batch[0]]['path'],allow_pickle=False) as z:
                    if z['ids'].tolist()!=batch:raise ValueError('CPU prefix shard order differs')
                    rgb=z['rgb'].copy()
            x=torch.from_numpy(rgb.reshape(-1,3,224,224)).to(args.device)
            presence,pose,_=model(x)
            poses.append(pose.reshape(len(batch),3,9,3)[...,:2].cpu().numpy())
            dets.append((presence.reshape(len(batch),3,9)>0).float().cpu().numpy())
            for q in batch:
                if cached_targets is not None:
                    target,mask=cached_targets[q]
                    if not np.array_equal(mask,np.asarray(part['presence'][q])):raise ValueError('Cached public object mask mismatch')
                    masks.append(mask);targets.append(target);continue
                state=np.load(Path(original['data_root'])/q/'cd/states.npy',allow_pickle=False)
                if state.shape[:2]!=(30,9):raise ValueError('Unexpected Balls target shape')
                mask=(np.abs(state[0,:,:3]).sum(-1)>0).astype(np.float32)
                if not np.array_equal(mask,np.asarray(part['presence'][q])):raise ValueError('Public object mask mismatch')
                masks.append(mask);targets.append(np.asarray(state[3:,:,:2],np.float32))
            if off%240==0:xep.emit('balls_missing_prefix',done=off+len(batch),total=len(remaining),seconds=time.time()-began)
    q=np.concatenate([old_q,np.concatenate(poses)]);det=np.concatenate([old_det,np.concatenate(dets)])
    mask=np.concatenate([old_mask,np.asarray(masks)]);target=np.concatenate([old_target,np.asarray(targets)])
    if not all(np.isfinite(x).all() for x in (q,det,mask,target)):raise ValueError('Nonfinite full validation data')
    xep.save_npz(out/'input_val.npz',ids=np.asarray(ids),pose=q,detected=det,presence=mask)
    xep.save_npz(out/'target_val.npz',pose=target)
    files={str(out/n):xep.digest(out/n) for n in ('manifest.json','support_plan.npz','input_val.npz','target_val.npz')}
    xep.write(out/'data_ready.json',dict(version=VERSION,status='COMPLETE',file_sha256=files,seconds=time.time()-began,
        test_read=False,original_512_copied_exactly=True,only_missing_prefix_extracted=True,inference_precision='original FP32',
        cpu_prefix_cache=None if args.prefix_cache is None else dict(path=str(args.prefix_cache),sha256=xep.digest(args.prefix_cache/'complete.json'))))


class FixedData:
    def __init__(self,row,codes,plan,supports):self.data={'val':row};self.codes=codes;self.plan=plan[...,:supports]
    def batch(self,split,indices,epoch,device):
        if split!='val':raise ValueError('Validation only')
        r=self.data['val'];s=self.codes[self.plan[indices],np.arange(9)[None,:,None]]*r['mask'][indices,:,None,None]
        return tuple(torch.from_numpy(np.asarray(v,np.float32)).to(device) for v in (r['q'][indices],r['det'][indices],r['mask'][indices],s,r['target'][indices]))


def evaluate(args):
    out,root=args.out,args.base.parent;ready=xep.read(out/'data_ready.json');checks(ready['file_sha256'])
    if (out/'summary.json').exists():return
    manifest=xep.read(out/'manifest.json');checks(manifest['file_sha256']);ids=manifest['query_ids'];readout=root/'source_formation_v5_1/balls/readout'
    with np.load(out/'support_plan.npz',allow_pickle=False) as z:plan=z['plan'].copy()
    with np.load(readout/'plans/val_000.npz',allow_pickle=False) as z:
        if z['ids'].tolist()!=ids[:512] or not np.array_equal(plan[:512],z['plan']):raise ValueError('v5.1 original S8 plan differs')
    with np.load(out/'input_val.npz',allow_pickle=False) as z:row=dict(ids=z['ids'].tolist(),q=z['pose'].copy(),det=z['detected'].copy(),mask=z['presence'].copy())
    with np.load(out/'target_val.npz',allow_pickle=False) as z:row['target']=z['pose'].copy()
    jobs=[]
    for method in ('Native-U','A-U','Random-U'):
        source=method.split('-')[0]
        folder=root/'xep_advantage_v4_2/runs/standard_s3'/method if source!='Random' else root/'xep_random_reference_v4_3/standard_s3/runs'/method
        code_train=args.base/'codes'/f'{source}_train.npz' if source!='Random' else root/'xep_random_reference_v4_3/codes/train.npz'
        code_val=args.base/'codes'/f'{source}_val.npz' if source!='Random' else root/'xep_random_reference_v4_3/codes/val.npz'
        jobs.append((f'legacy/{method}',folder,code_train,code_val,3))
    binding=xep.read(readout/'binding.json');checks(binding['file_sha256'])
    for method in METHODS:
        marker=xep.read(readout/'codes'/(method+'_complete.json'));checks(marker['file_sha256'])
        if marker['source_sha256']!=binding['sources'][method]['sha256']:raise ValueError('Frozen source code identity differs')
    for s in (3,5,8):
        for method in METHODS:jobs.append((f'v51/S{s}/{method}',readout/'runs'/f'S{s}'/method,readout/'codes'/f'{method}_train.npz',readout/'codes'/f'{method}_val.npz',s))
    freeze={}
    for key,folder,ct,cv,s in jobs:
        if xep.read(folder/'complete.json')['epochs']!=100:raise ValueError('Head budget not complete')
        state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False);old=xep.read(folder/'selected_validation.json')
        if state['epoch']!=old['epoch'] or old['ids']!=ids[:512]:raise ValueError('Selected head epoch/IDs differ')
        freeze[key]=dict(checkpoint=str(folder/'selected.pt'),sha256=xep.digest(folder/'selected.pt'),epoch=state['epoch'],
            code_sha256={str(p):xep.digest(p) for p in (ct,cv)},receipt_sha256=xep.digest(folder/'selected_validation.json'))
    xep.write(out/'checkpoint_freeze.json',dict(version=VERSION,heads=freeze,code_sha256=xep.digest(__file__),test_read=False,optimizer_steps=0))
    torch.set_num_threads(4);began=time.time();result=dict(version=VERSION,status='COMPLETE',methods={},comparisons={},test_read=False,optimizer_steps=0,
        head_budget=100,cohort_counts=dict(original_selection=512,remaining=1488,all_eligible=2000),selection_rule='unchanged 512 selection',
        checkpoint_freeze_sha256=xep.digest(out/'checkpoint_freeze.json'))
    with torch.inference_mode():
        for key,folder,ct,cv,s in jobs:
            checks(freeze[key]['code_sha256'])
            with np.load(ct,allow_pickle=False) as z:
                observed=z['u'][z['presence']>0];mean=observed.mean(0);scale=observed.std(0);scale[scale<1e-6]=1.
            with np.load(cv,allow_pickle=False) as z:
                if z['ids'].tolist()!=manifest['metadata']['all_ids']:raise ValueError('Source code order differs')
                codes=(z['u'].copy()-mean)/scale
            state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
            model=xep.PredictHead('Native-U').to(args.device).eval();model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
            metric=xep.evaluate(model,FixedData(row,codes,plan,s),args.device);values=np.asarray(metric['per_recipient_mse']);old=xep.read(folder/'selected_validation.json')
            error=float(np.abs(values[:512]-np.asarray(old['per_recipient_mse'])).max())
            if not np.allclose(values[:512],old['per_recipient_mse'],atol=2e-5,rtol=2e-5):raise ValueError(f'{key}: original512 error {error}')
            item=dict(selected_epoch=state['epoch'],original512_max_absolute_error=error,
                cohort_mse=dict(original_selection=float(values[:512].mean()),remaining=float(values[512:].mean()),all_eligible=float(values.mean())),**metric)
            xep.write(out/'scores'/(key+'.json'),dict(ids=ids,**item));result['methods'][key]={k:v for k,v in item.items() if k!='per_recipient_mse'}
            xep.emit('balls_fullval_head',head=key,mse=metric['mse'],old512_error=error);del model
    for prefix in ('legacy','v51/S3','v51/S5','v51/S8'):
        nk=prefix+('/Native-U' if prefix=='legacy' else '/Native');rk=prefix+('/Random-U' if prefix=='legacy' else '/Random-Both-new')
        n=result['methods'][nk]['mse'];r=result['methods'][rk]['mse']
        result['comparisons'][prefix]={key:dict(improvement_over_native_percent=100*(n-v['mse'])/n,
            improvement_over_random_percent=100*(r-v['mse'])/r) for key,v in result['methods'].items() if key.startswith(prefix+'/') and key!=nk}
    result['seconds']=time.time()-began;xep.write(out/'summary.json',result);xep.emit('balls_fullval_complete',seconds=result['seconds'],comparisons=result['comparisons'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('cpu-targets','prepare','evaluate','all'));root=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
    p.add_argument('--base',type=Path,default=root/'xep_discovery_balls_v4_1');p.add_argument('--out',type=Path,default=root/'supervised_tail_v6/balls_fullval')
    p.add_argument('--prefix-cache',type=Path,default=None)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with open(a.out/(a.command+'.lock'),'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='cpu-targets':prepare_targets(a)
        if a.command in ('prepare','all'):prepare(a)
        if a.command in ('evaluate','all'):evaluate(a)
