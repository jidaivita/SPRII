"""Frozen full-U128 readouts for monolithic and split JEPA source checkpoints.

Mono keeps the complete joint representation of donor AB plus recipient CD[:3].
Split exposes [donor P64, recipient T64]. All use the same supervised fresh head.
No source training, no test access, no P-only/complete-U comparison is pooled.
"""
import argparse
import importlib.util
import fcntl
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = 'monolithic-split-full-U128-readout-v6.6-1'
DIAGNOSTIC_REVISION = 'v6.6.1-identical-batch-cache-check'
ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))


def read(p): return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2)); os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p = Path(p); tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    torch.save(obj, tmp); os.replace(tmp, p)


def seedof(s): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'little')


def emit(event, **kw): print(json.dumps(dict(event=event, time=time.time(), **kw)), flush=True)


def immutable(p, obj):
    if Path(p).exists() and read(p) != obj: raise ValueError('Different frozen binding: ' + str(p))
    write(p, obj)


def sample_without(rng, pool, excluded, count):
    """Sample ranks, skipping a few excluded IDs without copying a large pool."""
    pool = np.asarray(pool, dtype=np.int64)
    positions = []
    for value in set(excluded):
        at = int(np.searchsorted(pool, value))
        if at < len(pool) and pool[at] == value: positions.append(at)
    positions.sort()
    available = len(pool)-len(positions)
    if available < count: raise ValueError('Insufficient independent support pool')
    ranks = rng.choice(available, count, replace=False)
    for at in positions: ranks += ranks >= at
    return pool[ranks]


class Head(nn.Module):
    def __init__(self, dims, detection_dims, support_dims=128, horizon=27, width=128):
        super().__init__(); self.dims, self.horizon, self.width = dims, horizon, width
        # Module-local seeds preserve query/dynamics initialization when only
        # support width changes, including Query-only and parameter references.
        torch.manual_seed(seedof('v6-support-' + str(support_dims)))
        self.support = nn.Sequential(nn.Linear(support_dims, 64), nn.ReLU(), nn.Linear(64, 64))
        self.missing = nn.Parameter(torch.zeros(64))
        torch.manual_seed(seedof('v6-query'))
        self.query = nn.GRU(2*dims+detection_dims, 64, batch_first=True)
        torch.manual_seed(seedof('v6-init'))
        self.init = nn.Sequential(nn.Linear(129, width), nn.ReLU(), nn.Linear(width, width))
        torch.manual_seed(seedof('v6-message'))
        self.message = nn.Sequential(nn.Linear(2*width+2*dims, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        torch.manual_seed(seedof('v6-dynamics'))
        self.cell = nn.GRUCell(width+2*dims, width)
        self.delta = nn.Linear(width, dims)
        nn.init.normal_(self.delta.weight, std=.001); nn.init.zeros_(self.delta.bias)
        self.register_buffer('xy_mean', torch.zeros(dims)); self.register_buffer('xy_scale', torch.ones(dims))

    def forward(self, q, det, mask, support, missing=None):
        b, length, slots, dims = q.shape
        x = (q-self.xy_mean)/self.xy_scale
        dx = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:]-x[:, :-1]), 1)
        if det.ndim == 3: det = det[..., None]
        token = torch.cat((x, dx, det), -1).permute(0, 2, 1, 3).reshape(b*slots, length, -1)
        _, qh = self.query(token); qh = qh[0].reshape(b, slots, 64)
        if not isinstance(support, dict):
            raise ValueError('U128 readout requires matched and current-preserving null contexts')
        memory = self.support(support['u']).mean(2)
        available = support['available']
        if missing is not None:
            null_memory = self.support(support['null_u']).mean(2)
            flag = missing[:, None, None]
            memory = torch.where(flag, null_memory, memory)
            available = available * (~flag)
        h = self.init(torch.cat((qh, memory, available), -1))
        position = x[:, -1]; velocity = (x[:, -1]-x[:, 0])/(length-1)
        pair = mask[:, :, None]*mask[:, None, :]*(1-torch.eye(slots, device=q.device)[None])
        denominator = pair.sum(2).clamp_min(1)[..., None]; result = []
        for _ in range(self.horizon):
            hi = h[:, :, None].expand(b, slots, slots, self.width)
            hj = h[:, None, :].expand(b, slots, slots, self.width)
            rel = position[:, None]-position[:, :, None]; rv = velocity[:, None]-velocity[:, :, None]
            msg = self.message(torch.cat((hi, hj, rel, rv), -1))
            agg = (msg*pair[..., None]).sum(2)/denominator
            h = self.cell(torch.cat((agg, position, velocity), -1).reshape(b*slots, -1), h.reshape(b*slots, -1)).reshape(b, slots, -1)
            velocity = velocity+self.delta(h); position = position+velocity
            result.append(position*self.xy_scale+self.xy_mean)
        return torch.stack(result, 1)


def scores(pred, target, mask):
    per = (pred-target).square().mean(-1)
    return (per*mask[:, None]).sum((1, 2))/(mask.sum(1)*target.shape[1]).clamp_min(1)


class PairPlan:
    def pools(self, split):
        if split in self.pool_cache: return self.pool_cache[split]
        part = self.manifest['splits'][split]; row = self.data[split]
        all_ids = part['all_ids']; index = {s:i for i,s in enumerate(all_ids)}
        shared = {key:np.asarray(values,dtype=np.int64) for key,values in part.get('candidate_groups',{}).items()}
        for pool in shared.values():
            if len(pool) and (np.diff(pool) <= 0).any(): raise ValueError('Compact pool must be sorted and unique')
        correct = []
        for ident in row['ids']:
            slots = []
            for k in range(self.slots):
                if part['presence'][ident][k] <= 0: slots.append(np.empty(0,np.int64)); continue
                if 'candidates' in part:
                    pool = np.asarray(part['candidates'][ident][k],np.int64)
                    if index[ident] in pool: raise ValueError('Self donor in explicit candidate pool')
                else: pool = shared[part['candidate_keys'][ident][k]]
                if len(pool) and (np.diff(pool) <= 0).any(): raise ValueError('Pool must be sorted and unique')
                slots.append(pool)
            correct.append(slots)
        # Construct a wrong pool once per public stratum/physical class, not
        # once per query. Only metadata available in this frozen manifest is
        # used; a donor must also be visibly present in its AB code cache.
        classes = defaultdict(lambda:defaultdict(list)); row_keys = {}
        for ident, physical in part['physical'].items():
            if ident not in index: continue
            for k in np.flatnonzero(np.asarray(part['presence'][ident]) > 0):
                if row['donor_seen'][index[ident], k] <= 0: continue
                typ = part.get('known_type',{}).get(ident,[None]*self.slots)[k]
                gravity = part.get('gravity',{}).get(ident)
                public = (int(k),json.dumps(typ,sort_keys=True),json.dumps(gravity,sort_keys=True))
                label = tuple(physical[k]); classes[public][label].append(index[ident])
                row_keys[(ident,int(k))] = (public,label)
        wrong_shared = {}
        for public, group in classes.items():
            for label in group:
                wrong_shared[(public,label)] = np.asarray(sorted(v for other,values in group.items() if other!=label for v in values),np.int64)
        wrong = []
        for ident in row['ids']:
            slots = []
            for k in range(self.slots):
                if 'wrong_candidates' in part:
                    pool = np.asarray(part['wrong_candidates'][ident][k],np.int64)
                else:
                    typ = part.get('known_type',{}).get(ident,[None]*self.slots)[k]
                    gravity = part.get('gravity',{}).get(ident)
                    key = ((k,json.dumps(typ,sort_keys=True),json.dumps(gravity,sort_keys=True)),tuple(part['physical'][ident][k]))
                    pool = wrong_shared.get(key,np.empty(0,np.int64))
                slots.append(pool)
            wrong.append(slots)
        value = dict(correct=correct,wrong=wrong,index=index)
        self.pool_cache[split] = value
        return value

    def plan(self, split, epoch, wrong=False):
        if not self.learned: return None
        part = self.manifest['splits'][split]; ids = self.data[split]['ids']
        plan = np.zeros((len(ids), self.slots, self.args.supports), dtype=np.int64)
        pools = self.pools(split); index = pools['index']
        for i, ident in enumerate(ids):
            rng = np.random.default_rng(seedof(f'xep:20260911:{split}:{ident}:{epoch}'))
            for slot in np.flatnonzero(self.data[split]['mask'][i] > 0):
                options = pools['wrong' if wrong else 'correct'][i][slot]
                location = int(np.searchsorted(options,index[ident]))
                contains_self = location < len(options) and options[location] == index[ident]
                if len(options)-int(contains_self) < self.args.supports:
                    if wrong: plan[i, slot] = -1; continue
                    raise ValueError(f'Insufficient correct supports {split}:{ident}:{slot}')
                first = sample_without(rng,options,[index[ident]],min(3,self.args.supports))
                if self.args.supports > 3:
                    extra = sample_without(np.random.default_rng(seedof(f'v6-extra:{split}:{ident}:{slot}:{epoch}')),
                                           options,[index[ident],*first.tolist()],self.args.supports-3)
                    first = np.concatenate((first, extra))
                plan[i, slot] = first
        return plan

    def make_full_plans(self,part,domain,row):
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
                rng=np.random.default_rng(seedof(f'xep:20260911:val:{ident}:0'))
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
                    first=sample_without(rng,options,[index[ident]],min(3,s))
                    if s>3:
                        extra=sample_without(np.random.default_rng(seedof(f'v6-extra:val:{ident}:{k}:0')),
                            options,[index[ident],*first.tolist()],s-3)
                        first=np.concatenate([first,extra])
                    plan[i,k]=first
            plans.append(plan)
        return plans


def source_model(path, checkpoint, device):
    path=Path(path).resolve(); name='u128_source_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        model, ck=module.load_checkpoint(checkpoint,device)
    model.eval(); model.requires_grad_(False)
    return model,ck


def npy_write(path, value):
    path=Path(path); tmp=path.with_name(path.stem+'.pending.npy')
    np.save(tmp,value,allow_pickle=False);os.replace(tmp,path)


def encoding_binding(args):
    ckpath=Path(args.checkpoint).resolve(); modelpath=Path(args.model_code).resolve()
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    expected='cophy-monolithic-jepa-v6.6' if args.kind=='mono' else 'cophy-latent-v6.2-sig02'
    if ck.get('version')!=expected or ck.get('epoch')!=args.source_epochs:
        raise ValueError('Source checkpoint family or actual epoch is not the requested source50/source100/source150')
    for config in (ck.get('config',{}),ck.get('binding',{})):
        if config.get('scene') not in (None,args.scene):raise ValueError('Source scene mismatch')
    roots={}
    for split in ('train','val'):
        folder=Path(args.features)/split; ready=read(folder/'COMPLETE.json')
        if ready.get('status')!='COMPLETE' or ready.get('test_read') is not False or ready.get('scene')!=args.scene:
            raise ValueError('Wait for committed train/validation RGB features')
        if sha(folder/'manifest.json')!=ready['manifest_sha256']:raise ValueError('Feature manifest changed')
        roots[split]=dict(path=str(folder.resolve()),complete_sha256=sha(folder/'COMPLETE.json'),ids_sha256=sha(folder/'ids.json'))
    return dict(version=VERSION,scene=args.scene,kind=args.kind,method=args.method,source_epochs=args.source_epochs,
        checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),source_version=ck['version'],
        source_experiment_version=ck.get('experiment_version'),source_route=ck.get('route'),
        model_code=str(modelpath),model_code_sha256=sha(modelpath),readout_code_sha256=sha(__file__),
        base=str(Path(args.base).resolve()),base_sha256=sha(Path(args.base)/'manifest.json'),feature_roots=roots,
        supports=3,head_epochs=100,representation='complete U128; query-conditioned; never AB-only P slice',
        cache_dtype='float32',test_read=False)


@torch.no_grad()
def encode(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    binding=encoding_binding(args);immutable(out/'encoding_config.json',binding)
    if (out/'encoding_complete.json').exists():
        old=read(out/'encoding_complete.json')
        if old['binding']!=binding:raise ValueError('Different completed U128 source')
        for p,h in old['files'].items():
            if sha(p)!=h:raise ValueError('Changed frozen context cache')
        emit('encoding_already_complete',out=str(out))
        fit_normalization(args)
        return
    model,ck=source_model(binding['model_code'],binding['checkpoint'],args.device)
    all_files={}
    for split,root in binding['feature_roots'].items():
        folder=out/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
        marker=folder/'complete.json'
        if marker.exists():
            previous=read(marker)
            if previous['binding_sha256']!=sha(out/'encoding_config.json'):raise ValueError('Different context-cache binding')
            for p,h in previous['files'].items():
                if sha(p)!=h:raise ValueError('Changed completed split cache')
            all_files.update(previous['files']);continue
        fp=Path(root['path']);ids=read(fp/'ids.json');n=len(ids)
        arrays={name:np.load(fp/(name+'.npy'),mmap_mode='r') for name in ('features_ab','features_cd','presence_ab','presence_cd')}
        if any(len(a)!=n for a in arrays.values()):raise ValueError('Feature/ID cardinality differs')
        writers={};paths={};first_started=time.monotonic(); max_diff=0.
        for first in range(0,n,64):
            end=min(first+64,n)
            tensor=lambda x:torch.from_numpy(np.array(x,copy=True)).to(args.device)
            ab=tensor(arrays['features_ab'][first:end]).float();am=tensor(arrays['presence_ab'][first:end]).bool()
            c=tensor(arrays['features_cd'][first:end,:3]).float();cm=tensor(arrays['presence_cd'][first:end,:3]).bool()
            present=am.any(1)
            if args.kind=='mono':
                state=model.encode_history_state(ab,am);tokens=model.encode_current_tokens(c,cm)
                own=model.encode_joint_from_cached(state,tokens,cm)
                zero={'hidden':torch.zeros_like(state['hidden']),'present':torch.zeros_like(state['present'])}
                null=model.encode_joint_from_cached(zero,tokens,cm)
                values=dict(hidden=state['hidden'].permute(1,0,2,3),tokens=tokens,present=present,current_mask=cm,null_u=null,own_u=own)
                if first==0:
                    direct=model.encode_joint(ab,am,c,cm);error=float((direct-own).abs().max())
                    torch.testing.assert_close(direct,own,atol=2e-5,rtol=2e-5);max_diff=max(max_diff,error)
            else:
                p=model.encode(ab,am);t=model.encode_current(c,cm)
                own=torch.cat((p,t),-1);null=torch.cat((torch.zeros_like(p),t),-1)
                values=dict(p=p,t=t,present=present,current_mask=cm,null_u=null,own_u=own)
            for name,value in values.items():
                value=value.detach().cpu().numpy();dtype=np.uint8 if name in ('present','current_mask') else np.float32
                if not np.isfinite(value).all():raise FloatingPointError('Nonfinite context cache '+name)
                if name not in writers:
                    temp=folder/(name+'.pending.npy');paths[name]=folder/(name+'.npy')
                    writers[name]=np.lib.format.open_memmap(temp,mode='w+',dtype=dtype,shape=(n,)+value.shape[1:])
                writers[name][first:end]=value.astype(dtype,copy=False)
            if first==0 or end==n or end%1024==0:emit('context_cache_progress',split=split,episodes=end,total=n)
        for name,writer in writers.items():writer.flush()
        writers.clear()
        for name,path in paths.items():os.replace(folder/(name+'.pending.npy'),path)
        write(folder/'ids.json',ids);paths['ids']=folder/'ids.json'
        files={str(path):sha(path) for path in paths.values()}
        receipt=dict(status='COMPLETE',version=VERSION,split=split,episodes=n,binding_sha256=sha(out/'encoding_config.json'),
            files=files,cache_bytes=sum(p.stat().st_size for p in paths.values()),all_history_layers=True,
            cached_vs_direct_max_difference=max_diff,diagnostic_revision=DIAGNOSTIC_REVISION,
            cached_vs_direct_batch='identical full extraction batch for both paths',seconds=time.monotonic()-first_started,test_read=False)
        write(marker,receipt);all_files.update(files)
    write(out/'encoding_complete.json',dict(status='COMPLETE',version=VERSION,binding=binding,files=all_files,test_read=False))
    fit_normalization(args)


@torch.no_grad()
def fit_normalization(args):
    out=Path(args.out)
    if (out/'prepared.json').exists():
        prepared=read(out/'prepared.json')
        if prepared['encoding_sha256']!=sha(out/'encoding_complete.json') or prepared['normalization_sha256']!=sha(out/'normalization.json'):
            raise ValueError('Changed prepared readout')
        return
    # Model parameters stay frozen; this is a train-input statistic, not head fitting.
    data=Data(args,normalize=False);plan=data.plan('train',0);total=np.zeros(128,np.float64);sq=total.copy();count=0
    for first in range(0,len(data.data['train']['ids']),128):
        ix=np.arange(first,min(first+128,len(data.data['train']['ids'])))
        u,_=data.raw_context('train',ix,plan[ix],args.device)
        active=data.data['train']['mask'][ix]>0;values=u.detach().cpu().numpy()[active].reshape(-1,128).astype(np.float64)
        total+=values.sum(0);sq+=np.square(values).sum(0);count+=len(values)
    if count<2:raise ValueError('No active training support contexts')
    mean=total/count;scale=np.sqrt(np.maximum(sq/count-mean*mean,0)).clip(1e-6)
    write(out/'normalization.json',dict(version=VERSION,mean=mean.tolist(),scale=scale.tolist(),train_support_objects=count,
        plan_epoch=0,plan_sha256=hashlib.sha256(np.ascontiguousarray(plan).tobytes()).hexdigest(),
        encoding_sha256=sha(out/'encoding_complete.json'),validation_used=False,test_read=False))
    write(out/'prepared.json',dict(status='COMPLETE',version=VERSION,encoding_sha256=sha(out/'encoding_complete.json'),
        normalization_sha256=sha(out/'normalization.json'),test_read=False))
    emit('readout_prepared',scene=args.scene,kind=args.kind,source_epochs=args.source_epochs,cache_bytes=sum(Path(p).stat().st_size for p in read(out/'encoding_complete.json')['files']))


class Data(PairPlan):
    def __init__(self,args,normalize=True):
        self.args=args;self.out=Path(args.out);saved=read(self.out/'encoding_complete.json');self.binding=saved['binding']
        if saved.get('status')!='COMPLETE' or saved.get('version')!=VERSION:raise ValueError('Wait for U128 encoding')
        if self.binding['readout_code_sha256']!=sha(__file__):raise ValueError('Frozen readout implementation changed')
        for path,h in saved['files'].items():
            if sha(path)!=h:raise ValueError('Changed frozen context cache '+path)
        self.base=Path(self.binding['base']);self.manifest=read(self.base/'manifest.json');self.learned=True
        if self.manifest.get('test_read') is not False or self.manifest.get('prefix')!=3:raise ValueError('Wrong downstream input contract')
        if sha(self.base/'manifest.json')!=self.binding['base_sha256']:raise ValueError('Changed downstream manifest')
        self.data={};self.cache={};self.cache_lut={};self.history_index={};self.source=None
        for split in ('train','val'):
            folder=self.out/'cache'/split;ids=list(map(str,read(folder/'ids.json')))
            self.cache_lut[split]={q:i for i,q in enumerate(ids)}
            self.cache[split]={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy') if '.pending.' not in p.name}
            with np.load(self.base/f'input_{split}.npz',allow_pickle=False) as x:
                row=dict(ids=list(map(str,x['ids'])),q=x['pose'].copy(),det=x['detected'].copy(),mask=x['presence'].copy())
            with np.load(self.base/f'target_{split}.npz',allow_pickle=False) as y:row['target']=y['pose'].copy()
            part=self.manifest['splits'][split]
            if row['ids']!=part['query_ids']:raise ValueError('Query ordering differs')
            self.history_index[split]=np.asarray([self.cache_lut[split][q] for q in part['all_ids']],np.int64)
            row['donor_seen']=self.cache[split]['present'][self.history_index[split]]
            self.data[split]=row
        tr=self.data['train'];self.dims=tr['q'].shape[-1];self.slots=tr['q'].shape[2];self.horizon=tr['target'].shape[1]
        self.det_dims=1 if tr['det'].ndim==3 else tr['det'].shape[-1];self.support_dims=128
        expected={'balls':(2,9,27,1),'collision':(3,4,12,4),'blocktower':(3,4,27,1)}[self.binding['scene']]
        if (self.dims,self.slots,self.horizon,self.det_dims)!=expected:raise ValueError('Query/target schema changed')
        for split,row in self.data.items():
            n=len(row['ids'])
            if row['q'].shape!=(n,3,self.slots,self.dims) or row['target'].shape!=(n,self.horizon,self.slots,self.dims):raise ValueError('Bad current/future dimensions')
            if not all(np.isfinite(row[k]).all() for k in ('q','det','mask','target')):raise ValueError('Nonfinite pose inputs/targets')
            if self.binding['scene']=='collision':
                expected_type=np.asarray([self.manifest['splits'][split]['known_type'][q] for q in row['ids']],np.float32)
                if not np.array_equal(row['det'][...,1:],np.broadcast_to(expected_type[:,None],row['det'][...,1:].shape)):raise ValueError('Public type changed')
        pos=tr['q'][np.broadcast_to(tr['mask'][:,None]>0,tr['q'].shape[:-1])]
        self.xy_mean=pos.mean(0);self.xy_scale=pos.std(0).clip(.1)
        self.pool_cache={};self.train_plan_epoch=None;self.train_plan=None
        self.val_plan=self.plan('val',0);self.wrong_plan=self.plan('val',0,wrong=True)
        selected=self.manifest['splits']['val'].get('selection_query_ids',self.data['val']['ids'][:512])
        lookup={q:i for i,q in enumerate(self.data['val']['ids'])};self.selection=np.asarray([lookup[q] for q in selected],np.int64)
        if len(self.selection)!=512:raise ValueError('Original512 selection is required')
        self.code_mean=np.zeros(128,np.float32);self.code_scale=np.ones(128,np.float32)
        if normalize:
            norm=read(self.out/'normalization.json')
            if norm['encoding_sha256']!=sha(self.out/'encoding_complete.json'):raise ValueError('U128 normalization binding changed')
            self.code_mean=np.asarray(norm['mean'],np.float32);self.code_scale=np.asarray(norm['scale'],np.float32)
        if getattr(args,'prepared',None):self.expand(args.prepared)

    def frozen_source(self,device):
        if self.source is None:
            b=self.binding
            if sha(b['model_code'])!=b['model_code_sha256'] or sha(b['checkpoint'])!=b['checkpoint_sha256']:
                raise ValueError('Changed source model')
            self.source,_=source_model(b['model_code'],b['checkpoint'],device)
        if str(next(self.source.parameters()).device)!=str(torch.device(device)):raise ValueError('Do not move frozen source mid-run')
        return self.source

    @torch.no_grad()
    def raw_context(self,split,ix,plan,device):
        row=self.data[split];cache=self.cache[split];s=self.args.supports
        qi=np.asarray([self.cache_lut[split][row['ids'][i]] for i in ix],np.int64)
        tensor=lambda x:torch.from_numpy(np.array(x,copy=True)).to(device)
        null=tensor(cache['null_u'][qi]).float()[:,:,None].expand(-1,-1,s,-1)
        if plan is None:return null.clone(),null
        if (plan<0).any():raise ValueError('Restrict unsupported Wrong cohort before context encoding')
        if self.binding['kind']=='split':
            history=self.history_index[split][plan]
            p=tensor(cache['p'][history,np.arange(self.slots)[None,:,None]]).float()
            t=tensor(cache['t'][qi]).float()[:,:,None].expand(-1,-1,s,-1)
            return torch.cat((p,t),-1),null
        model=self.frozen_source(device)
        out=torch.zeros(len(ix),self.slots,s,128,device=device)
        rr,slot,shot=np.where(np.broadcast_to(row['mask'][ix,:,None]>0,plan.shape))
        for first in range(0,len(rr),128):
            r=rr[first:first+128];o=slot[first:first+128];ss=shot[first:first+128]
            hi=self.history_index[split][plan[r,o,ss]];ci=qi[r]
            state={'hidden':tensor(cache['hidden'][hi]).float().permute(1,0,2,3).contiguous(),
                   'present':tensor(cache['present'][hi]).bool()}
            tokens=tensor(cache['tokens'][ci]).float();cm=tensor(cache['current_mask'][ci]).bool()
            u=model.encode_joint_from_cached(state,tokens,cm)
            focal=torch.as_tensor(o,device=device);selected=u[torch.arange(len(o),device=device),focal]
            out[torch.as_tensor(r,device=device),focal,torch.as_tensor(ss,device=device)]=selected
        return out,null

    def batch(self,split,ix,epoch,device,arm='matched'):
        row=self.data[split]
        if split=='train':
            if epoch!=self.train_plan_epoch:self.train_plan=self.plan(split,epoch);self.train_plan_epoch=epoch
            plan=self.train_plan[ix]
        else:plan=(self.wrong_plan if arm=='wrong' else self.val_plan)[ix]
        u,null=self.raw_context(split,ix,None if arm=='null' else plan,device)
        mean=torch.as_tensor(self.code_mean,device=device);scale=torch.as_tensor(self.code_scale,device=device)
        mask=torch.as_tensor(row['mask'][ix],device=device,dtype=torch.float32)
        support=dict(u=((u-mean)/scale)*mask[:,:,None,None],null_u=((null-mean)/scale)*mask[:,:,None,None],
            available=torch.ones(len(ix),self.slots,1,device=device)*(0. if arm=='null' else 1.))
        tensor=lambda v:torch.from_numpy(np.asarray(v,dtype=np.float32)).to(device)
        return tensor(row['q'][ix]),tensor(row['det'][ix]),mask,support,tensor(row['target'][ix])

    def expand(self,prepared):
        path=Path(prepared)/'prepared.json';binding=read(path)
        if binding.get('status')!='COMPLETE' or binding.get('scene')!=self.binding['scene'] or binding.get('test_read') is not False:
            raise ValueError('Full validation input is not complete')
        if Path(binding['base']).resolve()!=self.base.resolve():raise ValueError('Full validation uses another base')
        for p,h in binding['files'].items():
            if sha(p)!=h:raise ValueError('Full validation input changed')
        m=read(binding['full_manifest']);part=m['splits']['val'] if self.binding['scene']=='blocktower' else m['metadata']
        if 'all_ids' not in part:part=dict(part,all_ids=m['all_history_ids'])
        oldpart=self.manifest['splits']['val'];oldval=self.val_plan;oldwrong=self.wrong_plan
        if part['all_ids']!=oldpart['all_ids']:raise ValueError('Full evaluation cannot expand the donor history domain')
        with np.load(binding['input_path'],allow_pickle=False) as x:
            row=dict(ids=list(map(str,x['ids'])),q=x['pose'].copy(),det=x['detected'].copy(),mask=x['presence'].copy())
        with np.load(binding['target_path'],allow_pickle=False) as y:row['target']=y['pose'].copy()
        expected={'balls':2000,'collision':4000,'blocktower':8088}[self.binding['scene']]
        if row['ids']!=binding['query_ids'] or len(row['ids'])!=expected:raise ValueError('Full validation cohort differs')
        selection=np.asarray(binding['selection_indices'],np.int64)
        if [row['ids'][i] for i in selection]!=[self.data['val']['ids'][i] for i in self.selection]:raise ValueError('Selection512 differs')
        for key in ('q','det','mask','target'):
            if not np.array_equal(row[key][selection],self.data['val'][key][self.selection]):raise ValueError('Original512 input changed: '+key)
        row['donor_seen']=self.data['val']['donor_seen'];self.data['val']=row;self.selection=selection
        self.val_plan,self.wrong_plan=self.make_full_plans(part,oldpart,row)
        if not np.array_equal(self.val_plan[selection],oldval) or not np.array_equal(self.wrong_plan[selection],oldwrong):
            raise ValueError('Full validation changed original512 donor plans')
        self.full_binding=binding;self.prepared_sha256=sha(path)
@torch.no_grad()
def score_head(model, data, device, ix=None, arm='matched'):
    model.eval(); ix = np.arange(len(data.data['val']['ids'])) if ix is None else np.asarray(ix)
    if arm == 'wrong':
        valid = (data.wrong_plan[ix] >= 0).all((1, 2)); ix = ix[valid]
    values, thirds = [], []
    for off in range(0, len(ix), 128):
        q, det, mask, support, y = data.batch('val', ix[off:off+128], 0, device, arm)
        pred = model(q, det, mask, support)
        values.extend(scores(pred, y, mask).cpu().tolist())
        slices = np.array_split(np.arange(data.horizon), 3)
        thirds.extend(torch.stack([scores(pred[:, s], y[:, s], mask) for s in slices], 1).cpu().tolist())
    if not values: return dict(mse=None, recipients=0, ids=[], per_recipient_mse=[])
    return dict(mse=float(np.mean(values)), recipients=len(values), ids=[data.data['val']['ids'][i] for i in ix],
                per_recipient_mse=values, thirds=np.mean(thirds, 0).tolist())

def train(args):
    data = Data(args); folder = Path(args.out)/f'S{args.supports}'/args.reference; folder.mkdir(parents=True, exist_ok=True)
    model = Head(data.dims, data.det_dims, data.support_dims, data.horizon).to(args.device)
    model.xy_mean.copy_(torch.tensor(data.xy_mean, device=args.device)); model.xy_scale.copy_(torch.tensor(data.xy_scale, device=args.device))
    conf = dict(version=VERSION, scene=args.scene, reference=args.reference, supports=args.supports, epochs=args.epochs,
                base_sha256=sha(Path(args.base)/'manifest.json'), code_sha256=sha(Path(args.out)/'encoding_complete.json'), source_epochs=data.binding['source_epochs'], source_method=data.binding['method'], source_kind=data.binding['kind'], source_checkpoint_sha256=data.binding['checkpoint_sha256'], representation='complete U128', normalization_sha256=sha(Path(args.out)/'normalization.json'), implementation_sha256=sha(__file__),
                encoder_frozen=True, head_width=128, support_dims=data.support_dims, lr=.0003, batch_size=128,
                current_input='shared pose/detection/public type plus current RGB-derived T64 or joint U; no future features',
                input_sha256={str(Path(args.base)/f'{name}_{split}.npz'): sha(Path(args.base)/f'{name}_{split}.npz')
                              for split in ('train','val') for name in (('input','target','parameters') if args.reference=='known' else ('input','target'))},
                head_history_access='complete U128 support projected then mean; init only; no-history dropout preserves current-only U',
                validation_rows=len(data.data['val']['ids']),selection_rows=len(data.selection),
                null_dropout=.1 if data.learned else 0., null_semantics='zero all-layer history state then query for Mono; [zeroP,T] for Split; same train-only U normalization', seed=0, selection_ids_sha256=hashlib.sha256(data.selection.tobytes()).hexdigest(),
                test_read=False)
    immutable(folder/'config.json', conf)
    if (folder/'complete.json').exists():
        complete=read(folder/'complete.json')
        if complete.get('status')!='COMPLETE' or complete.get('epochs')!=args.epochs:
            raise ValueError('Invalid existing readout completion')
        emit('readout_already_complete',scene=args.scene,reference=args.reference,supports=args.supports)
        return
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    start, best, history = 0, float('inf'), []
    if (folder/'latest.pt').exists():
        old = torch.load(folder/'latest.pt', map_location=args.device, weights_only=False)
        if old['config'] != conf: raise ValueError('Readout resume mismatch')
        model.load_state_dict(old['model']); optim.load_state_dict(old['optimizer'])
        start, best, history = old['epoch'], old['best'], old['history']
    for epoch in range(start+1, args.epochs+1):
        model.train(); total, count = 0., 0; began = time.monotonic()
        torch.manual_seed(991+epoch); order = np.random.default_rng(771+epoch).permutation(len(data.data['train']['ids']))
        for off in range(0, len(order), 128):
            ix = order[off:off+128]; q, det, mask, support, y = data.batch('train', ix, epoch, args.device)
            missing = torch.rand(len(ix), device=args.device) < .1 if data.learned else None
            optim.zero_grad(set_to_none=True); loss = scores(model(q, det, mask, support, missing), y, mask).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Readout loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optim.step()
            total += loss.item()*len(ix); count += len(ix)
        val = score_head(model, data, args.device, data.selection)
        rec = dict(epoch=epoch, train_mse=total/count, validation_mse=val['mse'], seconds=time.monotonic()-began)
        history.append(rec)
        if val['mse'] < best:
            best = val['mse']; save(folder/'selected.pt', dict(model=model.state_dict(), epoch=epoch, config=conf))
            write(folder/'selected_validation.json', dict(epoch=epoch, **val))
        save(folder/'latest.pt', dict(model=model.state_dict(), optimizer=optim.state_dict(), epoch=epoch, best=best, history=history, config=conf))
        write(folder/'progress.json', dict(status='RUNNING', epoch=epoch, best=best, history=history))
        if epoch in (20, 60, 100):
            old = torch.load(folder/'selected.pt', map_location='cpu', weights_only=False); save(folder/f'selected_{epoch}.pt', old)
        emit('readout_epoch', scene=args.scene, reference=args.reference, supports=args.supports, **rec)
    ck = torch.load(folder/'selected.pt', map_location=args.device, weights_only=False); model.load_state_dict(ck['model'])
    results = dict(status='COMPLETE', selected_epoch=ck['epoch'], config=conf, matched=score_head(model, data, args.device),
        selection=score_head(model,data,args.device,data.selection),
        validation_scope='all rows present in the bound base; no automatic loading of val_full files',
        fixed_selection_only=len(data.data['val']['ids'])==len(data.selection),test_read=False)
    if data.learned:
        results['null'] = score_head(model, data, args.device, arm='null')
        results['wrong'] = score_head(model, data, args.device, arm='wrong')
        results['history_gain_percent'] = 100*(results['null']['mse']-results['matched']['mse'])/results['null']['mse'] if results['null']['mse'] else None
        eligible_ids = set(results['wrong']['ids'])
        comparable = [i for i, s in enumerate(data.data['val']['ids']) if s in eligible_ids]
        results['matched_on_wrong_cohort'] = score_head(model, data, args.device, comparable)
        results['null_on_wrong_cohort'] = score_head(model, data, args.device, comparable, arm='null')
    write(folder/'results.json', results)
    write(folder/'complete.json', dict(status='COMPLETE', version=VERSION, scene=args.scene, method=data.binding['method'], source_epochs=data.binding['source_epochs'], epochs=args.epochs, selected_epoch=ck['epoch'], results_sha256=sha(folder/'results.json'), checkpoint_sha256=sha(folder/'selected.pt'), mse=results['matched']['mse'], test_read=False))
    emit('readout_complete', scene=args.scene, supports=args.supports, mse=results['matched']['mse'])



def smoke(args):
    data=Data(args);model=Head(data.dims,data.det_dims,128,data.horizon).to(args.device)
    model.xy_mean.copy_(torch.tensor(data.xy_mean,device=args.device));model.xy_scale.copy_(torch.tensor(data.xy_scale,device=args.device))
    ix=np.arange(min(128,len(data.data['train']['ids'])));began=time.monotonic()
    q,det,mask,support,y=data.batch('train',ix,1,args.device)
    prediction=model(q,det,mask,support);loss=scores(prediction,y,mask).mean();loss.backward()
    grad=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(loss) or float(grad)<=0:raise FloatingPointError('Real U128 head smoke failed')
    if data.source is not None and any(p.grad is not None for p in data.source.parameters()):raise ValueError('Frozen source received a gradient')
    with torch.no_grad():
        _,_,_,null_support,_=data.batch('train',ix,1,args.device,arm='null')
        dropout_prediction=model(q,det,mask,support,torch.ones(len(ix),device=args.device,dtype=torch.bool))
        null_prediction=model(q,det,mask,null_support)
        torch.testing.assert_close(dropout_prediction,null_prediction,atol=2e-5,rtol=2e-5)
    receipt=dict(status='PASS',version=VERSION,scene=args.scene,method=data.binding['method'],source_epochs=data.binding['source_epochs'],
        real_train_examples=len(ix),support_shape=list(support['u'].shape),loss=float(loss.detach()),gradient_norm=float(grad),
        no_history_preserves_current=True,dropout_equals_null_max_difference=float((dropout_prediction-null_prediction).abs().max()),
        source_frozen=True,optimizer_steps=0,weights_discarded_after_smoke=True,seconds=time.monotonic()-began,test_read=False)
    write(Path(args.out)/'S3/learned/smoke.json',receipt);emit('readout_smoke_pass',**receipt)


def compare_rows(actual,saved,label):
    if actual['ids']!=saved['ids']:raise ValueError('Original512 '+label+' IDs changed')
    a=np.asarray(actual['per_recipient_mse']);b=np.asarray(saved['per_recipient_mse'])
    delta=float(np.max(np.abs(a-b))) if len(a) else 0.
    if not np.allclose(a,b,atol=2e-5,rtol=2e-5):raise ValueError('Original512 result mismatch: '+label+' '+str(delta))
    return dict(rows=len(a),max_absolute_difference=delta)


def evaluate(args):
    if not args.prepared:raise ValueError('evaluate requires shared --prepared fullval inputs')
    out=Path(args.out);folder=out/'S3/learned';complete=read(folder/'complete.json');conf=read(folder/'config.json')
    if complete.get('status')!='COMPLETE' or complete.get('epochs')!=100 or conf.get('version')!=VERSION:
        raise ValueError('Wait for this exact U128 head100')
    if conf['code_sha256']!=sha(out/'encoding_complete.json') or conf['normalization_sha256']!=sha(out/'normalization.json'):
        raise ValueError('Selected head source/normalization changed')
    for p,h in conf['input_sha256'].items():
        if sha(p)!=h:raise ValueError('Original head inputs/targets changed')
    ck=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    if ck['config']!=conf:raise ValueError('Selected head config mismatch')
    saved=read(folder/'results.json');selected=read(folder/'selected_validation.json')
    if ck['epoch']!=selected['epoch'] or ck['epoch']!=saved['selected_epoch']:raise ValueError('Checkpoint selection changed')
    data=Data(args);model=Head(data.dims,data.det_dims,128,data.horizon).to(args.device)
    model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False);model.eval()
    fullout=out/'S3/learned/fullval';fullout.mkdir(parents=True,exist_ok=True)
    frozen=dict(version=VERSION,scene=args.scene,method=data.binding['method'],source_epochs=data.binding['source_epochs'],
        source_checkpoint_sha256=data.binding['checkpoint_sha256'],encoding_sha256=sha(out/'encoding_complete.json'),
        head_sha256=sha(folder/'selected.pt'),selected_epoch=ck['epoch'],head_budget=100,
        prepared_sha256=data.prepared_sha256,normalization_sha256=sha(out/'normalization.json'),
        readout_code_sha256=sha(__file__),original_results_sha256=sha(folder/'results.json'),test_read=False)
    immutable(fullout/'checkpoint_freeze.json',frozen)
    if (fullout/'complete.json').exists():
        done=read(fullout/'complete.json')
        if done['results_sha256']!=sha(fullout/'results.json'):raise ValueError('Changed complete fullval result')
        emit('fullval_already_complete',out=str(fullout));return
    started=time.monotonic();reproduction={}
    reproduction['selected']=compare_rows(score_head(model,data,args.device,data.selection),selected,'selected')
    for arm in ('matched','null','wrong'):
        reproduction[arm]=compare_rows(score_head(model,data,args.device,data.selection,arm),saved[arm],arm)
    arms={arm:score_head(model,data,args.device,arm=arm) for arm in ('matched','null','wrong')}
    eligible=set(arms['wrong']['ids']);ix=np.asarray([i for i,q in enumerate(data.data['val']['ids']) if q in eligible],np.int64)
    arms['matched_on_wrong_cohort']=score_head(model,data,args.device,ix)
    arms['null_on_wrong_cohort']=score_head(model,data,args.device,ix,arm='null')
    remaining=np.setdiff1d(np.arange(len(data.data['val']['ids'])),data.selection)
    arms['remaining_matched']=score_head(model,data,args.device,remaining)
    result=dict(status='COMPLETE',version=VERSION,scene=args.scene,method=data.binding['method'],source_epochs=data.binding['source_epochs'],
        representation='complete U128',supports=3,head_budget=100,selected_epoch=ck['epoch'],selection_rows=512,
        full_validation_rows=len(data.data['val']['ids']),remaining_rows=len(remaining),reproduction=reproduction,
        history_gain_percent=100*(arms['null']['mse']-arms['matched']['mse'])/arms['null']['mse'] if arms['null']['mse'] else None,
        null_semantics=conf['null_semantics'],wrong_donor_domain=data.full_binding['wrong_donor_domain'],
        checkpoint_freeze_sha256=sha(fullout/'checkpoint_freeze.json'),seconds=time.monotonic()-started,
        test_read=False,optimizer_steps=0,**arms)
    write(fullout/'results.json',result)
    write(fullout/'complete.json',dict(status='COMPLETE',version=VERSION,scene=args.scene,method=data.binding['method'],
        source_epochs=data.binding['source_epochs'],head_budget=100,full_validation_rows=result['full_validation_rows'],
        results_sha256=sha(fullout/'results.json'),test_read=False,optimizer_steps=0))
    emit('fullval_complete',scene=args.scene,method=data.binding['method'],mse=arms['matched']['mse'],rows=result['full_validation_rows'])


def probe(args):
    # This deliberately probes complete OWN joint context, not a fictitious Mono P.
    out=Path(args.out);record=read(out/'encoding_complete.json');binding=record['binding'];root=Path(args.root)
    prepath=root/'prepared_v3'/args.scene/'training_preflight.json'
    if args.scene=='blocktower':
        profile=read(root/'runtime_profiles.json')['scenes']['blocktower'];prepath=Path(profile['training_preflight']['path'])
        if sha(prepath)!=profile['training_preflight']['sha256']:raise ValueError('Changed Blocktower audit')
    pre=read(prepath);samples={};fields=None
    for split in ('train','val'):
        artifact=pre['artifacts']['raw_relations_'+split]
        if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Changed parameter audit')
        ids=list(map(str,read(out/'cache'/split/'ids.json')));lut={q:i for i,q in enumerate(ids)}
        code=np.load(out/'cache'/split/'own_u.npy',mmap_mode='r');visible=np.load(out/'cache'/split/'current_mask.npy',mmap_mode='r').any(1)
        x=[];values=[];labels=[]
        for row in read(artifact['path']):
            if not row['in_C'] or row['id'] not in lut:continue
            i=lut[row['id']];k=row['slot']
            if not visible[i,k]:continue
            raw=list(row['raw_physical'])
            if args.scene=='blocktower':raw+=list(row['raw_gravity'])
            x.append(code[i,k]);values.append(raw);labels.append(row['physical'])
        samples[split]=dict(x=np.asarray(x,np.float64),y=np.asarray(values,np.float64),labels=np.asarray(labels,np.int64))
    tr=samples['train'];va=samples['val'];mean=tr['x'].mean(0);scale=tr['x'].std(0).clip(1e-8)
    a=(tr['x']-mean)/scale;b=(va['x']-mean)/scale
    classes=[np.unique(tr['labels'][:,j]) for j in range(tr['labels'].shape[1])]
    onehot=np.concatenate([(tr['labels'][:,j,None]==c[None]).astype(float) for j,c in enumerate(classes)],1)
    target=np.concatenate((tr['y'],onehot),1);center=target.mean(0)
    weight=np.linalg.solve(a.T@a+np.eye(128),a.T@(target-center));prediction=b@weight+center
    names=['mass','friction'] if args.scene=='blocktower' else ['mass','friction','restitution']
    if args.scene=='blocktower':names+=['gravity_x','gravity_y']
    if len(names)!=tr['y'].shape[1]:raise ValueError('Audit physical field schema needs explicit update')
    fields={}
    for j,name in enumerate(names):
        mse=float(np.square(va['y'][:,j]-prediction[:,j]).mean());var=float(va['y'][:,j].var())
        fields[name]=dict(mse=mse,r2=1-mse/var if var else None)
    offset=len(names)
    for j,c in enumerate(classes):
        guess=c[prediction[:,offset:offset+len(c)].argmax(1)];actual=va['labels'][:,j]
        if not np.isin(actual,c).all():raise ValueError('Validation parameter category outside training support')
        fields[names[j]].update(accuracy=float((guess==actual).mean()),
            balanced_accuracy=float(np.mean([(guess[actual==v]==v).mean() for v in c if (actual==v).any()])))
        offset+=len(c)
    result=dict(status='COMPLETE',version=VERSION,scene=args.scene,method=binding['method'],source_epochs=binding['source_epochs'],
        representation='complete OWN AB+query3 U128; includes current information; not P-only formation',
        train_objects=len(a),val_objects=len(b),ridge_alpha=1.,normalization='train only',representations={'U':fields},
        preflight_sha256=sha(prepath),encoding_sha256=sha(out/'encoding_complete.json'),test_read=False,source_optimizer_steps=0)
    write(out/'probes.json',result);emit('probe_complete',scene=args.scene,method=binding['method'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','encode','smoke','train','evaluate','probe'))
    p.add_argument('--out',required=True);p.add_argument('--scene',choices=('balls','collision','blocktower'))
    p.add_argument('--base');p.add_argument('--features');p.add_argument('--checkpoint');p.add_argument('--model-code')
    p.add_argument('--kind',choices=('mono','split'));p.add_argument('--method');p.add_argument('--source-epochs',type=int,choices=(50,100,150))
    p.add_argument('--supports',type=int,choices=(3,),default=3);p.add_argument('--epochs',type=int,choices=(100,),default=100)
    p.add_argument('--device',default='cpu');p.add_argument('--prepared');p.add_argument('--root',default=str(ROOT))
    args=p.parse_args();args.reference='learned';Path(args.out).mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if args.command in ('encode','prepare'):
        if not all(getattr(args,k) for k in ('scene','base','features','checkpoint','model_code','kind','method','source_epochs')):
            p.error('prepare/encode requires scene, base, features, checkpoint, model-code, kind, method, source-epochs')
    else:
        b=read(Path(args.out)/'encoding_config.json')
        if args.scene and args.scene!=b['scene']:p.error('Scene differs from prepared output')
        args.scene=b['scene'];args.base=b['base'];args.kind=b['kind'];args.method=b['method'];args.source_epochs=b['source_epochs']
    if args.prepared and args.command!='evaluate':p.error('Full validation can only be opened by fixed-head evaluate')
    with open(Path(args.out)/'worker.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':encode,'encode':encode,'smoke':smoke,'train':train,'evaluate':evaluate,'probe':probe}[args.command](args)


if __name__=='__main__':main()
