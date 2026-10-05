"""Full-validation evaluation of fixed v6.2 P64 heads. No training or test reads.

The original512 Matched/Null/Wrong paths are reproduced before expansion.
Wrong's donor metadata domain stays the original readout manifest's domain:
expanding the recipient cohort must not silently change its old Wrong plans.
"""

import os
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import readout as core

VERSION='latent-v6.2-fullval-fixed-P64-readout-1'
ROOT=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
DEFAULT_BASES={'balls':ROOT/'xep_discovery_balls_v4_1','collision':ROOT/'xep_discovery_collision_v4_4','blocktower':ROOT/'xep_discovery_blocktower_v6'}
DEFAULT_FULL={'balls':ROOT/'supervised_tail_v6/balls_fullval','collision':ROOT/'xep_collision_multiquery_v4_9_fullval','blocktower':ROOT/'xep_discovery_blocktower_v6'}


def check_files(files):
    for path, expected in files.items():
        if core.sha(path)!=expected:raise ValueError('Changed bound data: '+str(path))


def npz_write(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.npz');np.savez(tmp,**arrays);tmp.replace(path)


def metadata(binding):
    manifest=core.read(binding['full_manifest'])
    if binding['scene']=='blocktower':part=manifest['splits']['val']
    else:
        part=manifest['metadata']
        if 'all_ids' not in part:part=dict(part,all_ids=manifest['all_history_ids'])
    return part


def prepare(args):
    base,full,out=Path(args.base),Path(args.full_data),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.json').exists():
        saved=core.read(out/'prepared.json')
        if (saved['version']!=VERSION or saved['scene']!=args.scene or
            Path(saved['base']).resolve()!=base.resolve() or Path(saved['full_data']).resolve()!=full.resolve()):
            raise ValueError('Different prepared output')
        check_files(saved['files']);core.emit('fullval_inputs_already_ready',scene=args.scene,out=str(out));return
    original=core.read(base/'manifest.json');manifest=core.read(full/'manifest.json')
    if original.get('test_read') is not False or manifest.get('test_read') is not False:raise ValueError('Only train/validation manifests allowed')
    old=original['splits']['val'];old_ids=old['query_ids'];full_marker=full/'data_ready.json'
    ready=core.read(full_marker)
    if ready.get('test_read') is not False:raise ValueError('Full data is not validation-only')
    check_files(ready.get('file_sha256',{}))
    if args.scene=='blocktower':
        if ready.get('status')!='COMPLETE':raise ValueError('Blocktower pose bridge incomplete')
        part=manifest['splits']['val'];ids=part['full_query_ids'];xp=full/'input_val_full.npz';yp=full/'target_val_full.npz'
        pp=full/'parameters_val_full.npz'
        for path in (xp,yp,pp):
            if core.sha(path)!=ready['files'][path.name]['sha256']:raise ValueError('Changed Blocktower bridge artifact')
    else:
        part=manifest['metadata'];ids=manifest['query_ids'];xp=full/'input_val.npz';yp=full/'target_val.npz';pp=out/'parameters_val_full.npz'
        if 'all_ids' not in part:part=dict(part,all_ids=manifest['all_history_ids'])
    expected={'balls':2000,'collision':4000,'blocktower':8088}[args.scene]
    if len(ids)!=expected or ids[:len(old_ids)]!=old_ids or len(old_ids)!=512:
        raise ValueError('Original selection cohort or full validation cardinality differs')
    if part['all_ids']!=old['all_ids']:raise ValueError('AB history domain changed')
    lookup={q:i for i,q in enumerate(ids)};selection=[lookup[q] for q in old_ids]
    for name in ('presence','physical','known_type','gravity'):
        if name in old and any(part[name][q]!=old[name][q] for q in old_ids):raise ValueError('Old query metadata changed: '+name)
    with np.load(xp,allow_pickle=False) as x,np.load(base/'input_val.npz',allow_pickle=False) as oldx:
        if x['ids'].tolist()!=ids or oldx['ids'].tolist()!=old_ids:raise ValueError('Input order differs')
        for key in ('pose','detected','presence'):
            if not np.array_equal(x[key][selection],oldx[key]):raise ValueError('Old512 prefix changed: '+key)
        schema=dict(pose=list(x['pose'].shape),detected=list(x['detected'].shape),mask=list(x['presence'].shape))
    with np.load(yp,allow_pickle=False) as y,np.load(base/'target_val.npz',allow_pickle=False) as oldy:
        if not np.array_equal(y['pose'][selection],oldy['pose']):raise ValueError('Old512 future targets changed')
        schema['target']=list(y['pose'].shape)
    if args.scene!='blocktower':
        labels=np.asarray([part['physical'][q] for q in ids],np.int64);mask=np.asarray([part['presence'][q] for q in ids],np.float32)
        if labels.shape[-1]!=3 or np.any((labels<0)|(labels>=3)):raise ValueError('Unexpected physical category support')
        params=np.eye(3,dtype=np.float32)[labels].reshape(len(ids),mask.shape[1],9)*mask[...,None]
        npz_write(pp,ids=np.asarray(ids),values=params)
    with np.load(pp,allow_pickle=False) as p,np.load(base/'parameters_val.npz',allow_pickle=False) as oldp:
        if not np.array_equal(p['values'][selection],oldp['values']):raise ValueError('Old512 Known-parameters vectors changed')
    # Full candidate tables remain at their original path, not copied per model.
    files={str(p):core.sha(p) for p in [base/'manifest.json',base/'input_val.npz',base/'target_val.npz',base/'parameters_val.npz',
        full/'manifest.json',full_marker,xp,yp,pp]}
    binding=dict(version=VERSION,status='COMPLETE',scene=args.scene,base=str(base),full_data=str(full),full_manifest=str(full/'manifest.json'),
        input_path=str(xp),target_path=str(yp),parameters_path=str(pp),query_ids=ids,selection_ids=old_ids,selection_indices=selection,
        schema=schema,files=files,test_read=False,optimizer_steps=0,metadata_storage='reference original compact/shared manifest; no candidate table duplication',
        wrong_donor_domain='original base manifest physical keys only; model-independent metadata domain; per-source cached visual presence still applies',
        original512_inputs_targets_parameters_exact=True)
    core.write(out/'prepared.json',binding);core.emit('fullval_inputs_ready',scene=args.scene,queries=len(ids),selection=len(old_ids),schema=schema)


class FullData:
    def __init__(self,args,binding):
        original_args=argparse.Namespace(scene=args.scene,base=binding['base'],out=args.readout,reference=args.reference,supports=args.supports)
        original=core.Data(original_args)
        self.original=original;self.args=original_args;self.learned=original.learned
        for name in ('dims','slots','det_dims','support_dims','horizon','code_mean','code_scale','xy_mean','xy_scale'):
            setattr(self,name,getattr(original,name))
        self.binding=binding;self.selection=np.asarray(binding['selection_indices'],np.int64)
        part=metadata(binding);oldpart=original.manifest['splits']['val'];self.manifest=original.manifest
        with np.load(binding['input_path'],allow_pickle=False) as x:
            row=dict(ids=x['ids'].tolist(),q=x['pose'].copy(),det=x['detected'].copy(),mask=x['presence'].copy())
        with np.load(binding['target_path'],allow_pickle=False) as y:row['target']=y['pose'].copy()
        if row['ids']!=binding['query_ids']:raise ValueError('Expanded input order differs')
        if args.reference=='known':
            with np.load(binding['parameters_path'],allow_pickle=False) as p:row['parameters']=p['values'].copy()
        if self.learned:
            row['codes']=original.data['val']['codes'];row['donor_seen']=original.data['val']['donor_seen']
        self.data={'val':row};self.val_plan=self.wrong_plan=None
        if self.learned:
            self.val_plan,self.wrong_plan=self.make_plans(part,oldpart,row)
            if not np.array_equal(self.val_plan[self.selection],original.val_plan):raise ValueError('Old512 Matched plan changed')
            if not np.array_equal(self.wrong_plan[self.selection],original.wrong_plan):raise ValueError('Old512 Wrong plan changed')

    def make_plans(self,part,domain,row):
        ids=row['ids'];index={q:i for i,q in enumerate(part['all_ids'])};s=self.args.supports
        shared={k:np.asarray(v,np.int64) for k,v in part.get('candidate_groups',{}).items()}
        classes=defaultdict(lambda:defaultdict(list))
        # Exactly v6.2 Data.pools' original-domain Wrong construction. Do not
        # populate these classes from newly added recipients' metadata.
        for ident,physical in domain['physical'].items():
            if ident not in index:continue
            for k in np.flatnonzero(np.asarray(domain['presence'][ident])>0):
                if row['donor_seen'][index[ident],k]<=0:continue
                typ=domain.get('known_type',{}).get(ident,[None]*self.slots)[k];gravity=domain.get('gravity',{}).get(ident)
                public=(int(k),json.dumps(typ,sort_keys=True),json.dumps(gravity,sort_keys=True))
                classes[public][tuple(physical[k])].append(index[ident])
        wrong_shared={(public,label):np.asarray(sorted(v for other,values in group.items() if other!=label for v in values),np.int64)
                      for public,group in classes.items() for label in group}
        plans=[]
        for wrong in (False,True):
            plan=np.zeros((len(ids),self.slots,s),np.int64)
            for i,ident in enumerate(ids):
                rng=np.random.default_rng(core.seedof(f'xep:20260911:val:{ident}:0'))
                for k in np.flatnonzero(row['mask'][i]>0):
                    if wrong:
                        if ident in domain.get('wrong_candidates',{}):options=np.asarray(domain['wrong_candidates'][ident][k],np.int64)
                        else:
                            typ=part.get('known_type',{}).get(ident,[None]*self.slots)[k];gravity=part.get('gravity',{}).get(ident)
                            key=((int(k),json.dumps(typ,sort_keys=True),json.dumps(gravity,sort_keys=True)),tuple(part['physical'][ident][k]))
                            options=wrong_shared.get(key,np.empty(0,np.int64))
                    elif 'candidates' in part:options=np.asarray(part['candidates'][ident][k],np.int64)
                    else:options=shared[part['candidate_keys'][ident][k]]
                    if len(options)>1 and (np.diff(options)<=0).any():raise ValueError('Candidate pool is not sorted unique')
                    at=int(np.searchsorted(options,index[ident]));has_self=at<len(options) and options[at]==index[ident]
                    if len(options)-int(has_self)<s:
                        if wrong:plan[i,k]=-1;continue
                        raise ValueError(f'Insufficient correct supports: {ident}:{k}')
                    first=core.sample_without(rng,options,[index[ident]],min(3,s))
                    if s>3:
                        extra=core.sample_without(np.random.default_rng(core.seedof(f'v6-extra:val:{ident}:{k}:0')),
                            options,[index[ident],*first.tolist()],s-3)
                        first=np.concatenate([first,extra])
                    plan[i,k]=first
            plans.append(plan)
        return plans

    def batch(self,split,ix,epoch,device,arm='matched'):
        if split!='val':raise ValueError('Fixed-head evaluation cannot read a training batch')
        return core.Data.batch(self,split,ix,epoch,device,arm)


def compare_saved(actual,saved,label):
    if actual['ids']!=saved['ids']:raise ValueError('Old512 '+label+' recipient coverage changed')
    a=np.asarray(actual['per_recipient_mse']);b=np.asarray(saved['per_recipient_mse'])
    error=float(np.abs(a-b).max()) if len(a) else 0.
    if not np.allclose(a,b,atol=2e-5,rtol=2e-5):raise ValueError('Old512 '+label+' predictions changed: '+str(error))
    return dict(recipients=len(a),max_absolute_error=error,matched_ids_exact=True)


def evaluate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);prepared=Path(args.prepared)/'prepared.json'
    binding=core.read(prepared)
    if binding['version']!=VERSION or binding['scene']!=args.scene:raise ValueError('Prepared full data differs')
    check_files(binding['files']);folder=Path(args.readout)/f'S{args.supports}'/args.reference
    complete=core.read(folder/'complete.json');config=core.read(folder/'config.json')
    if complete.get('status')!='COMPLETE' or complete.get('epochs')!=100:raise ValueError('Wait for the selected100-epoch head')
    if config['version']!=core.VERSION or config['scene']!=args.scene or config['reference']!=args.reference or config['supports']!=args.supports:
        raise ValueError('Wrong v6.2 readout identity')
    if config['base_sha256']!=core.sha(Path(binding['base'])/'manifest.json'):raise ValueError('Source readout base changed')
    check_files(config['input_sha256'])
    checkpoint=folder/'selected.pt';ck= torch.load(checkpoint,map_location='cpu',weights_only=False)
    old=core.read(folder/'results.json');selected=core.read(folder/'selected_validation.json')
    if ck['config']!=config or ck['epoch']!=selected['epoch'] or ck['epoch']!=old['selected_epoch']:
        raise ValueError('Fixed head and selection receipts differ')
    source_receipt=Path(args.readout)/'codes_complete.json' if args.reference=='learned' else None
    if source_receipt and config['code_sha256']!=core.sha(source_receipt):raise ValueError('P64 source binding changed')
    frozen=dict(version=VERSION,scene=args.scene,reference=args.reference,supports=args.supports,head_budget=100,
        selected_epoch=ck['epoch'],checkpoint=str(checkpoint),checkpoint_sha256=core.sha(checkpoint),
        selected_receipt_sha256=core.sha(folder/'selected_validation.json'),prepared_sha256=core.sha(prepared),
        original_results_sha256=core.sha(folder/'results.json'),head_implementation_sha256=core.sha(core.__file__),
        evaluation_implementation_sha256=core.sha(__file__),source_codes_sha256=core.sha(source_receipt) if source_receipt else None,
        test_read=False,optimizer_steps=0,encoder_frozen=True,head_frozen=True)
    if (out/'complete.json').exists():
        if core.read(out/'checkpoint_freeze.json')!=frozen:raise ValueError('Completed output has a different binding')
        core.emit('fullval_already_complete',scene=args.scene,reference=args.reference,out=str(out));return
    core.immutable(out/'checkpoint_freeze.json',frozen);torch.set_num_threads(4);began=time.monotonic()
    data=FullData(args,binding);model=core.Head(data.dims,data.det_dims,data.support_dims,data.horizon).to(args.device)
    model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False);model.eval()
    reproduction={}
    reproduction['selected']=compare_saved(core.evaluate(model,data,args.device,data.selection),selected,'selected Matched')
    for arm in ('matched','null','wrong') if data.learned else ('matched',):
        reproduction[arm]=compare_saved(core.evaluate(model,data,args.device,data.selection,arm),old[arm],arm)
    # No new-recipient score is computed before the fixed512 checks above pass.
    matched=core.evaluate(model,data,args.device);result=dict(status='COMPLETE',version=VERSION,scene=args.scene,
        reference=args.reference,supports=args.supports,selected_epoch=ck['epoch'],matched=matched,
        reproduction=reproduction,selection_rows=len(data.selection),full_validation_rows=len(data.data['val']['ids']),
        checkpoint_freeze_sha256=core.sha(out/'checkpoint_freeze.json'),test_read=False,optimizer_steps=0,
        scope='full validation at the unchanged selected100-epoch head; no checkpoint reselection',
        wrong_donor_domain=binding['wrong_donor_domain'])
    if data.learned:
        result['null']=core.evaluate(model,data,args.device,arm='null');result['wrong']=core.evaluate(model,data,args.device,arm='wrong')
        eligible=set(result['wrong']['ids']);ix=np.asarray([i for i,q in enumerate(data.data['val']['ids']) if q in eligible],np.int64)
        result['matched_on_wrong_cohort']=core.evaluate(model,data,args.device,ix)
        result['null_on_wrong_cohort']=core.evaluate(model,data,args.device,ix,arm='null')
        result['wrong_coverage']=len(ix)/len(data.data['val']['ids'])
        result['history_gain_percent']=100*(result['null']['mse']-matched['mse'])/result['null']['mse'] if result['null']['mse'] else None
        wm=result['wrong']['mse'];mm=result['matched_on_wrong_cohort']['mse']
        result['correct_vs_wrong_percent']=100*(wm-mm)/wm if wm else None
        npz_write(out/'plans.npz',ids=np.asarray(data.data['val']['ids']),matched=data.val_plan,wrong=data.wrong_plan)
        result['plan_sha256']=core.sha(out/'plans.npz')
    result['seconds']=time.monotonic()-began;core.write(out/'results.json',result)
    core.write(out/'complete.json',dict(status='COMPLETE',version=VERSION,reference=args.reference,supports=args.supports,
        results_sha256=core.sha(out/'results.json'),selected_epoch=ck['epoch'],full_validation_rows=len(data.data['val']['ids']),test_read=False,optimizer_steps=0))
    core.emit('fullval_complete',scene=args.scene,reference=args.reference,supports=args.supports,mse=matched['mse'],rows=len(data.data['val']['ids']),seconds=result['seconds'])


def summarize(args):
    results={}
    for specification in args.result:
        name,path=specification.split('=',1)
        if name in results:raise ValueError('Duplicate result label')
        r=core.read(path)
        if r.get('status')!='COMPLETE' or r.get('version')!=VERSION:raise ValueError('Unfinished fullval result')
        results[name]=r
    if not results:raise ValueError('No completed results')
    domain=None;wrong_sets=[]
    for r in results.values():
        ids=r['matched']['ids']
        if domain is None:domain=ids
        elif domain!=ids:raise ValueError('Cannot compare different full validation cohorts')
        if 'wrong' in r:wrong_sets.append(set(r['wrong']['ids']))
    common=set(domain)
    for ids in wrong_sets:common &= ids
    common_ids=[q for q in domain if q in common]
    table={}
    for name,r in results.items():
        row=dict(reference=r['reference'],supports=r['supports'],selected_epoch=r['selected_epoch'],full_matched_mse=r['matched']['mse'])
        for arm in ('matched','null','wrong'):
            if arm in r:
                byid=dict(zip(r[arm]['ids'],r[arm]['per_recipient_mse']))
                row[arm+'_on_common_wrong_cohort']=float(np.mean([byid[q] for q in common_ids])) if common_ids else None
        table[name]=row
    core.write(Path(args.out)/'summary.json',dict(status='COMPLETE',version=VERSION,full_rows=len(domain),common_wrong_rows=len(common_ids),
        common_wrong_ids=common_ids,methods=table,files={name:core.sha(path.split('=',1)[1]) for name,path in zip(results,args.result)},test_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','evaluate','summary'))
    p.add_argument('--scene',choices=tuple(DEFAULT_BASES));p.add_argument('--base');p.add_argument('--full-data');p.add_argument('--prepared')
    p.add_argument('--readout');p.add_argument('--reference',choices=('learned','query','known'),default='learned')
    p.add_argument('--supports',type=int,choices=(3,8),default=3);p.add_argument('--device',default='cpu');p.add_argument('--out',required=True)
    p.add_argument('--result',action='append',default=[],metavar='NAME=RESULTS_JSON');a=p.parse_args()
    if a.command!='summary' and not a.scene:p.error('--scene is required')
    if a.command=='prepare':a.base=a.base or str(DEFAULT_BASES[a.scene]);a.full_data=a.full_data or str(DEFAULT_FULL[a.scene])
    if a.command=='evaluate' and (not a.prepared or not a.readout):p.error('evaluate needs --prepared and --readout')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with open(out/(a.command+'.lock'),'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'evaluate':evaluate,'summary':summarize}[a.command](a)
