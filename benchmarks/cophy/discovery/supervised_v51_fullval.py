"""Expand fixed v5.1 frozen-head Collision results from 512 to 4000 val queries.

No training, resampling/selection by score, or test access. Reuse existing full
validation visual prefixes, reconstruct the original v5.1 nested support rule,
and require every selected checkpoint to reproduce its old 512 scores first.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import collision_xep as xep

VERSION='supervised-v51-collision-fullval-v6-1'
METHODS=('Native','Cross-only','Align-only','Both-new','Random-Both-new')


def verify_files(entries):
    for name,expected in entries.items():
        if xep.digest(name)!=expected: raise ValueError('Changed bound file: '+name)


def plan8(part, ids):
    plan=np.zeros((len(ids),4,8),np.int64); lookup={q:i for i,q in enumerate(part['all_ids'])}
    for i,ident in enumerate(ids):
        seed=int.from_bytes(hashlib.sha256(f'xep:20260911:val:{ident}:0'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed)
        for slot in range(4):
            if part['presence'][ident][slot]<=0: continue
            options=part['candidates'][ident][slot]
            if len(options)<8 or lookup[ident] in options: raise ValueError('Invalid independent donor pool')
            first=rng.choice(options,3,replace=False); rest=[v for v in options if v not in set(first.tolist())]
            text=f'source-formation-readout-v51:val:{ident}:{slot}:0'
            extra_rng=np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little'))
            plan[i,slot]=np.concatenate([first,extra_rng.choice(rest,5,replace=False)])
    return plan


class FixedData:
    def __init__(self,row,codes,plan,supports):
        self.data={'val':row};self.codes=codes;self.plan=plan[...,:supports]

    def batch(self,split,indices,epoch,device):
        if split!='val': raise ValueError('Validation only')
        row=self.data['val']; slots=np.arange(4)[None,:,None]
        support=self.codes[self.plan[indices],slots]*row['mask'][indices,:,None,None]
        tensor=lambda v:torch.from_numpy(np.asarray(v,np.float32)).to(device)
        return tuple(tensor(v) for v in (row['q'][indices],row['det'][indices],row['mask'][indices],support,row['target'][indices]))


def run(args):
    readout,expanded,out=args.readout,args.expanded,args.out
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'writer.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists():
            prior=xep.read(out/'summary.json')
            if prior['version']!=VERSION: raise ValueError('Different completed output version')
            print(json.dumps({k:v for k,v in prior.items() if k!='methods'}),flush=True); return
        began=time.time()
        while args.wait_complete and not args.wait_complete.exists():
            if time.time()-began>args.wait_seconds: raise TimeoutError('Prior fixed evaluation did not finish')
            time.sleep(5)
        binding=xep.read(readout/'binding.json'); base=Path(binding['base']); original=xep.read(base/'manifest.json')
        if binding['scene']!='collision' or binding['test_read'] or binding['head_budget']!=100:
            raise ValueError('Expected completed Collision v5.1 readout')
        verify_files(binding['file_sha256'])
        em=xep.read(expanded/'manifest.json'); ready=xep.read(expanded/'data_ready.json')
        if ready['manifest_sha256']!=xep.digest(expanded/'manifest.json') or em['test_read']:
            raise ValueError('Changed full validation data binding')
        verify_files(ready['file_sha256'])
        ids=em['query_ids']; old_ids=original['splits']['val']['query_ids']
        if em['original_query_ids']!=old_ids or ids[:512]!=old_ids or len(ids)!=4000:
            raise ValueError('Full validation cohort differs')
        part=dict(em['metadata'],all_ids=em['all_history_ids'])
        if part['all_ids']!=original['splits']['val']['all_ids']: raise ValueError('History IDs differ')
        for name in ('candidates','presence','physical','known_type'):
            if any(part[name][q]!=original['splits']['val'][name][q] for q in old_ids):
                raise ValueError('Old metadata differ: '+name)
        plan=plan8(part,ids)
        with np.load(readout/'plans/val_000.npz',allow_pickle=False) as p:
            if p['ids'].tolist()!=old_ids or not np.array_equal(p['plan'],plan[:512]):
                raise ValueError('Original v5.1 support plan not exactly reproduced')
        xep.save_npz(out/'support_plan.npz',ids=np.asarray(ids),plan=plan)
        with np.load(expanded/'input_val.npz',allow_pickle=False) as x:
            row=dict(ids=x['ids'].tolist(),q=x['pose'].copy(),det=x['detected'].copy(),mask=x['presence'].copy())
        with np.load(expanded/'target_val.npz',allow_pickle=False) as x: row['target']=x['pose'].copy()
        if row['ids']!=ids: raise ValueError('Expanded input IDs differ')
        for name,key in [('pose','q'),('detected','det'),('presence','mask')]:
            with np.load(base/'input_val.npz',allow_pickle=False) as old:
                if not np.array_equal(row[key][:512],old[name]): raise ValueError('Original visual prefix changed')
        with np.load(base/'target_val.npz',allow_pickle=False) as old:
            if not np.array_equal(row['target'][:512],old['pose']): raise ValueError('Original targets changed')
        selected={}
        for support in (3,5,8):
            for method in METHODS:
                folder=readout/'runs'/f'S{support}'/method
                if xep.read(folder/'complete.json')['epochs']!=100: raise ValueError('Readout budget incomplete')
                state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
                receipt=xep.read(folder/'selected_validation.json');config=state['config']
                if (state['epoch']!=receipt['epoch'] or receipt['ids']!=old_ids or config['method']!=method or
                    config['supports']!=support or config['binding_sha256']!=xep.digest(readout/'binding.json')):
                    raise ValueError('Selected head identity differs')
                selected[f'S{support}/{method}']=dict(path=str(folder/'selected.pt'),sha256=xep.digest(folder/'selected.pt'),
                    epoch=state['epoch'],selection_receipt_sha256=xep.digest(folder/'selected_validation.json'))
        xep.write(out/'checkpoint_freeze.json',dict(version=VERSION,selected=selected,test_read=False,optimizer_steps=0,
            plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),base_manifest_sha256=xep.digest(base/'manifest.json'),
            expanded_manifest_sha256=xep.digest(expanded/'manifest.json'),readout_binding_sha256=xep.digest(readout/'binding.json'),
            code_sha256=xep.digest(__file__)))
        torch.set_num_threads(4);torch.manual_seed(0); began=time.time()
        result=dict(version=VERSION,status='COMPLETE',test_read=False,optimizer_steps=0,head_budget=100,
            cohort_counts=dict(original_selection=512,remaining=3488,all_eligible=4000),methods={},comparisons={},
            selection_rule='Original 512 selected checkpoint unchanged; full validation never selects checkpoints',
            checkpoint_freeze_sha256=xep.digest(out/'checkpoint_freeze.json'))
        with torch.inference_mode():
            for method in METHODS:
                cm=xep.read(readout/'codes'/(method+'_complete.json'));verify_files(cm['file_sha256'])
                if cm['source_sha256']!=binding['sources'][method]['sha256']: raise ValueError('Source code identity differs')
                with np.load(readout/'codes'/f'{method}_val.npz',allow_pickle=False) as c:
                    if c['ids'].tolist()!=part['all_ids']: raise ValueError('Code order differs')
                    raw=c['u'].copy()
                for support in (3,5,8):
                    key=f'S{support}/{method}'; selection=selected[key]; folder=readout/'runs'/f'S{support}'/method
                    if xep.digest(selection['path'])!=selection['sha256']: raise ValueError('Selected checkpoint changed')
                    state=torch.load(selection['path'],map_location='cpu',weights_only=False); config=state['config']
                    # Python lists would promote codes to float64; preserve the exact original float32 transform.
                    mean=np.asarray(config['code_mean'],np.float32); scale=np.asarray(config['code_scale'],np.float32)
                    codes=(raw-mean)/scale
                    model=xep.PredictHead('Native-U').to(args.device).eval();model.load_state_dict(state['model'],strict=True)
                    model.requires_grad_(False); data=FixedData(row,codes,plan,support)
                    metric=xep.evaluate(model,data,args.device); values=np.asarray(metric['per_recipient_mse'])
                    old=xep.read(folder/'selected_validation.json'); delta=float(np.abs(values[:512]-np.asarray(old['per_recipient_mse'])).max())
                    if not np.allclose(values[:512],old['per_recipient_mse'],atol=2e-5,rtol=2e-5):
                        raise ValueError(f'{key}: old512 reproduction failed, max error={delta}')
                    item=dict(selected_epoch=state['epoch'],cohort_mse=dict(original_selection=float(values[:512].mean()),
                        remaining=float(values[512:].mean()),all_eligible=float(values.mean())),
                        original512_max_absolute_error=delta,checkpoint_sha256=selection['sha256'],**metric)
                    xep.write(out/'runs'/f'S{support}'/(method+'.json'),dict(ids=ids,**item))
                    result['methods'][key]={k:v for k,v in item.items() if k!='per_recipient_mse'}
                    xep.emit('fullval_fixed_head',method=method,supports=support,mse=metric['mse'],old512_maxerror=delta)
                    del model,data
        for support in (3,5,8):
            native=result['methods'][f'S{support}/Native']['mse']; random=result['methods'][f'S{support}/Random-Both-new']['mse']
            result['comparisons'][str(support)]={method:dict(improvement_over_native_percent=100*(native-result['methods'][f'S{support}/{method}']['mse'])/native,
                improvement_over_random_percent=100*(random-result['methods'][f'S{support}/{method}']['mse'])/random) for method in METHODS if method!='Native'}
        result['seconds']=time.time()-began; xep.write(out/'summary.json',result)
        xep.emit('fullval_complete',seconds=result['seconds'],comparisons=result['comparisons'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();root=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
    parser.add_argument('--readout',type=Path,default=root/'source_formation_v5_1/collision/readout')
    parser.add_argument('--expanded',type=Path,default=root/'xep_collision_multiquery_v4_9_fullval')
    parser.add_argument('--out',type=Path,default=root/'supervised_tail_v6/collision_v51_fullval')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--wait-complete',type=Path)
    parser.add_argument('--wait-seconds',type=float,default=3600)
    run(parser.parse_args())
