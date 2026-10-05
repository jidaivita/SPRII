"""Frozen original supervised CoPhyNet U32 and a shared fresh graph/GRU head.
No FT/MQ source, source gradient, or old task predictor is imported.
This v7 head uses full U32, q3/S3, 100epochs and 0.1 history dropout.
"""
import argparse
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

VERSION = 'cophy-v7-legacy-supervised-frozen-U32-readout'
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
    def __init__(self, dims, detection_dims, support_dims=64, horizon=27, width=128):
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
        if support is None:
            memory = self.missing.expand(b, slots, 64); available = torch.zeros(b, slots, 1, device=q.device)
        else:
            memory = self.support(support).mean(2); available = torch.ones(b, slots, 1, device=q.device)
            if missing is not None:
                flag = missing[:, None, None]
                memory = torch.where(flag, self.missing[None, None, :], memory)
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


# Source encoder implementation follows below.


class Data:
    def __init__(self, args):
        self.args = args; self.base = Path(args.base); self.out = Path(args.out)
        self.manifest = read(self.base/'manifest.json')
        if self.manifest.get('test_read') is not False or self.manifest['prefix'] != 3: raise ValueError('Unexpected downstream task')
        self.data = {}; self.code_mean = self.code_scale = None
        self.learned = args.reference == 'learned'
        if self.learned:
            binding = read(self.out/'codes_complete.json')
            if binding.get('version') != VERSION or binding.get('representation') != 'U32':
                raise ValueError('Readout requires v6.2 frozen U32 codes')
            for file, expected in binding['files'].items():
                if sha(file) != expected: raise ValueError('Encoded P cache changed')
        for split in ('train', 'val'):
            with np.load(self.base/f'input_{split}.npz', allow_pickle=False) as x:
                row = dict(ids=list(map(str, x['ids'])), q=x['pose'].copy(), det=x['detected'].copy(), mask=x['presence'].copy())
            with np.load(self.base/f'target_{split}.npz', allow_pickle=False) as y: row['target'] = y['pose'].copy()
            part = self.manifest['splits'][split]
            if row['ids'] != part['query_ids']: raise ValueError('Input/manifest query ordering differs')
            if self.learned:
                with np.load(self.out/f'codes_{split}.npz', allow_pickle=False) as c:
                    code_ids = list(map(str, c['ids'])); code = c['p'].copy(); seen = c['presence'].copy()
                lookup = {s: i for i, s in enumerate(code_ids)}
                index = [lookup[s] for s in part['all_ids']]
                code = code[index]; seen = seen[index]
                if split == 'train':
                    active = code[seen > 0]
                    self.code_mean = active.mean(0); self.code_scale = active.std(0).clip(1e-6)
                row['codes'] = (code-self.code_mean)/self.code_scale
                row['donor_seen'] = seen
            elif args.reference == 'known':
                with np.load(self.base/f'parameters_{split}.npz', allow_pickle=False) as p: row['parameters'] = p['values'].copy()
            self.data[split] = row
        train = self.data['train']; positions = train['q'][np.broadcast_to(train['mask'][:, None] > 0, train['q'].shape[:-1])]
        self.xy_mean = positions.mean(0); self.xy_scale = positions.std(0).clip(.1)
        self.dims = train['q'].shape[-1]; self.det_dims = 1 if train['det'].ndim == 3 else train['det'].shape[-1]
        self.horizon = train['target'].shape[1]; self.slots = train['q'].shape[2]
        expected = {'balls': (2, 9, 27, 1), 'collision': (3, 4, 12, 4), 'blocktower': (3, 4, 27, 1)}[args.scene]
        if (self.dims, self.slots, self.horizon, self.det_dims) != expected:
            raise ValueError('Public query/target schema differs from the registered task')
        for split, row in self.data.items():
            n = len(row['ids'])
            if row['q'].shape != (n, 3, self.slots, self.dims) or row['mask'].shape != (n, self.slots):
                raise ValueError('Query prefix or object mask shape differs')
            if row['target'].shape != (n, self.horizon, self.slots, self.dims):
                raise ValueError('Target horizon differs')
            if not all(np.isfinite(row[key]).all() for key in ('q','det','mask','target')):
                raise ValueError('Nonfinite public data or readout target')
            if args.scene == 'collision':
                expected_type = np.asarray([self.manifest['splits'][split]['known_type'][i] for i in row['ids']],np.float32)
                if not np.array_equal(row['det'][..., 1:], np.broadcast_to(expected_type[:,None], row['det'][...,1:].shape)):
                    raise ValueError('Collision public object-type input differs from v4.4')
        self.support_dims = train['parameters'].shape[-1] if args.reference == 'known' else 32
        self.pool_cache = {}
        self.train_plan_epoch = None; self.train_plan = None; self.val_plan = self.plan('val', 0)
        self.wrong_plan = self.plan('val', 0, wrong=True) if self.learned else None
        # Keep exactly the historic development cohort for Balls/Collision.
        # Expanded bases carry selection_query_ids. New Blocktower uses a
        # fixed hash-ordered first512 cohort before any v6 results exist.
        selected = self.manifest['splits']['val'].get('selection_query_ids')
        if selected is None: selected = self.data['val']['ids'][:512]
        lut = {s: i for i, s in enumerate(self.data['val']['ids'])}
        self.selection = np.array([lut[s] for s in selected], dtype=np.int64)

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

    def batch(self, split, ix, epoch, device, arm='matched'):
        row = self.data[split]; support = None
        if self.args.reference == 'known': support = row['parameters'][ix, :, None]
        elif self.learned and arm != 'null':
            if split == 'train':
                if epoch != self.train_plan_epoch:
                    self.train_plan = self.plan(split, epoch); self.train_plan_epoch = epoch
                plan = self.train_plan[ix]
            else: plan = (self.wrong_plan if arm == 'wrong' else self.val_plan)[ix]
            if (plan < 0).any(): raise ValueError('Unsupported Wrong query; restrict comparison cohort first')
            support = row['codes'][plan, np.arange(self.slots)[None, :, None]] * row['mask'][ix, :, None, None]
        tensor = lambda v: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
        return tensor(row['q'][ix]), tensor(row['det'][ix]), tensor(row['mask'][ix]), None if support is None else tensor(support), tensor(row['target'][ix])


@torch.no_grad()
def evaluate(model, data, device, ix=None, arm='matched'):
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
                base_sha256=sha(Path(args.base)/'manifest.json'), code_sha256=sha(Path(args.out)/'codes_complete.json') if data.learned else None,
                encoder_frozen=True, head_width=128, support_dims=data.support_dims, lr=.0003, batch_size=128,
                current_input='frozen official pose estimates for CD[0:3], detection and existing public type; no source g/T features',
                input_sha256={str(Path(args.base)/f'{name}_{split}.npz'): sha(Path(args.base)/f'{name}_{split}.npz')
                              for split in ('train','val') for name in (('input','target','parameters') if args.reference=='known' else ('input','target'))},
                head_history_access='support is read only for recurrent-state initialization; original v4.1/v4.4 head',
                validation_rows=len(data.data['val']['ids']),selection_rows=len(data.selection),
                null_dropout=.1 if data.learned else 0., seed=0, selection_ids_sha256=hashlib.sha256(data.selection.tobytes()).hexdigest(),
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
        val = evaluate(model, data, args.device, data.selection)
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
    results = dict(status='COMPLETE', selected_epoch=ck['epoch'], config=conf, matched=evaluate(model, data, args.device),
        selection=evaluate(model,data,args.device,data.selection),
        validation_scope='all rows present in the bound base; no automatic loading of val_full files',
        fixed_selection_only=len(data.data['val']['ids'])==len(data.selection),test_read=False)
    if data.learned:
        results['null'] = evaluate(model, data, args.device, arm='null')
        results['wrong'] = evaluate(model, data, args.device, arm='wrong')
        results['history_gain_percent'] = 100*(results['null']['mse']-results['matched']['mse'])/results['null']['mse']
        eligible_ids = set(results['wrong']['ids'])
        comparable = [i for i, s in enumerate(data.data['val']['ids']) if s in eligible_ids]
        results['matched_on_wrong_cohort'] = evaluate(model, data, args.device, comparable)
        results['null_on_wrong_cohort'] = evaluate(model, data, args.device, comparable, arm='null')
    write(folder/'results.json', results)
    write(folder/'complete.json', dict(status='COMPLETE', epochs=args.epochs, mse=results['matched']['mse'], test_read=False))
    emit('readout_complete', scene=args.scene, supports=args.supports, mse=results['matched']['mse'])


def smoke(args):
    data=Data(args); model=Head(data.dims,data.det_dims,data.support_dims,data.horizon).to(args.device)
    model.xy_mean.copy_(torch.tensor(data.xy_mean,device=args.device));model.xy_scale.copy_(torch.tensor(data.xy_scale,device=args.device))
    ix=np.arange(min(128,len(data.data['train']['ids'])))
    q,det,mask,support,y=data.batch('train',ix,1,args.device)
    pred=model(q,det,mask,support);loss=scores(pred,y,mask).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actual-batch loss')
    loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('No finite useful readout gradient')
    record=dict(status='PASS',version=VERSION,scene=args.scene,reference=args.reference,real_train_examples=len(ix),
        input_pose_shape=list(q.shape),detection_shape=list(det.shape),target_shape=list(y.shape),loss=float(loss.detach()),
        gradient_norm=float(norm),optimizer_steps=0,weights_discarded_after_smoke=True,
        head_history_access='initialization only, unchanged v4.1/v4.4 recurrent readout',
        train_rows=len(data.data['train']['ids']),validation_rows=len(data.data['val']['ids']),selection_rows=len(data.selection),
        base_manifest_sha256=sha(Path(args.base)/'manifest.json'),test_read=False)
    write(Path(args.out)/f'S{args.supports}'/args.reference/'smoke.json',record)
    emit('readout_smoke_pass',**record)


def probe(args):
    prepath=ROOT/'prepared_v3'/args.scene/'training_preflight.json'
    source_root=ROOT/'source'
    if args.scene=='blocktower':
        profile=read(ROOT/'runtime_profiles.json')['scenes']['blocktower']
        prepath=Path(profile['training_preflight']['path'])
        if sha(prepath)!=profile['training_preflight']['sha256']:raise ValueError('Changed Blocktower runtime preflight')
        source_root=Path(profile['source'])
    sys.path.insert(0, str(source_root))
    from cophy_fields import PLANS
    spec = PLANS[args.scene]; pre = read(prepath)
    marker=read(Path(args.out)/'codes_complete.json')
    if marker.get('version')!=VERSION or marker.get('representation')!='U32' or marker.get('status')!='COMPLETE':
        raise ValueError('Probe requires the current frozen U32 encoding')
    for filename,expected in marker['files'].items():
        if sha(filename)!=expected:raise ValueError('Changed probe codes')
    arrays = {}
    for split in ('train', 'val'):
        artifact = pre['artifacts']['raw_relations_'+split]
        if sha(artifact['path']) != artifact['sha256']: raise ValueError('Changed physical audit')
        raw = read(artifact['path'])
        with np.load(Path(args.out)/f'codes_{split}.npz', allow_pickle=False) as z:
            ids = list(map(str, z['ids'])); codes = z['p'].copy(); present = z['presence'].copy()
        lookup = {s: i for i, s in enumerate(ids)}; x, labels, values, types = [], [], [], []
        for r in raw:
            if not r['in_C'] or r['id'] not in lookup: continue
            i, k = lookup[r['id']], r['slot']
            if present[i, k] <= 0: continue
            physical = list(r['raw_physical'])
            if args.scene == 'blocktower': physical += list(r['raw_gravity'])
            x.append(codes[i, k]); labels.append(r['physical']); values.append(physical); types.append((k, r['known_type']))
        arrays[split] = dict(x=np.asarray(x, np.float64), y=np.asarray(values, np.float64), labels=np.asarray(labels), public=types)
    tr, va = arrays['train'], arrays['val']; mean = tr['x'].mean(0); scale = tr['x'].std(0).clip(1e-8)
    x = (tr['x']-mean)/scale; y = (va['x']-mean)/scale
    count = [len(v) for v in spec['support']]
    onehot = np.concatenate([np.eye(n)[tr['labels'][:, j]] for j, n in enumerate(count)], 1)
    target = np.concatenate((tr['y'], onehot), 1); center = target.mean(0)
    names = list(spec['fields']) + (['gravity_x', 'gravity_y'] if args.scene == 'blocktower' else [])
    result = dict(status='COMPLETE', version=VERSION,scene=args.scene, test_read=False, train_objects=len(x), val_objects=len(y),
        fields=names,representations={},preflight_sha256=sha(prepath),source_codes_sha256=sha(Path(args.out)/'codes_complete.json'),
        ridge_alpha=1.,normalization='train only',representation='original supervised U32 from full observed AB; source encoder frozen')
    for view, sl in dict(P=slice(0,16), T=slice(16,32), U=slice(0,32)).items():
        a, b = x[:, sl], y[:, sl]
        weight = np.linalg.solve(a.T@a+np.eye(a.shape[1]), a.T@(target-center)); pred = b@weight+center
        fields = {}
        for j, name in enumerate(names):
            mse = float(np.square(va['y'][:,j]-pred[:,j]).mean()); var = float(va['y'][:,j].var())
            fields[name] = dict(mse=mse, r2=1-mse/var if var > 0 else None)
        start = len(names)
        for j, (name, n) in enumerate(zip(spec['fields'], count)):
            guessed = pred[:, start:start+n].argmax(1); actual = va['labels'][:, j]
            recalls = [float((guessed[actual == c] == c).mean()) for c in range(n) if (actual == c).any()]
            fields[name].update(accuracy=float((guessed == actual).mean()), balanced_accuracy=float(np.mean(recalls)))
            start += n
        result['representations'][view] = fields
    result['source_checkpoint_sha256']=marker['source_sha256']
    result['probe_input']='OWN AB U32, before support aggregation'
    write(Path(args.out)/'probes.json', result)
    emit('probe_complete', scene=args.scene)




def prepare_base(args):
    """Bind existing full validation inputs; preserve original512 selection IDs."""
    import copy
    source=Path(args.base).resolve();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    m=copy.deepcopy(read(source/'manifest.json'))
    if m.get('test_read') is not False or m['prefix']!=3:raise ValueError('Unexpected source task')
    old=m['splits']['val']['query_ids'];full=m['splits']['val'].get('full_query_ids',old)
    m['splits']['val']['selection_query_ids']=old
    m['splits']['val']['query_ids']=full
    m['v7_original_manifest']=dict(path=str(source/'manifest.json'),sha256=sha(source/'manifest.json'))
    files={}
    for split in ('train','val'):
        suffix='val_full' if split=='val' and full!=old else split
        for name in ('input','target','parameters'):
            src=source/f'{name}_{suffix}.npz'
            if not src.exists():raise FileNotFoundError(str(src))
            with np.load(src,allow_pickle=False) as a:
                if a['ids'].tolist()!=m['splits'][split]['query_ids']:raise ValueError('Input ID order differs')
            dst=out/f'{name}_{split}.npz'
            if dst.exists() or dst.is_symlink():
                if dst.resolve()!=src:raise ValueError('Different prepared input alias')
            else:dst.symlink_to(src)
            files[str(src)]=sha(src)
    immutable(out/'manifest.json',m)
    write(out/'prepared.json',dict(status='COMPLETE',version=VERSION,files=files,
        source_manifest_sha256=sha(source/'manifest.json'),validation_rows=len(full),selection_rows=len(old),test_read=False))


@torch.no_grad()
def encode(args):
    import pickle
    import importlib
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    base=Path(args.base);m=read(base/'manifest.json')
    runtime=Path(args.runtime_source or m.get('runtime_source',str(ROOT/'source'))).resolve()
    if args.scene=='blocktower' and not args.runtime_source:
        profile=read(ROOT/'runtime_profiles.json')['scenes']['blocktower'];runtime=Path(profile['source']).resolve()
    sys.path.insert(0,str(runtime))
    adapter=importlib.import_module('cophy_adapter');models=importlib.import_module('cf_learning.model')
    if Path(adapter.__file__).resolve().parent!=runtime:raise ValueError('Wrong supervised runtime import')
    checkpoint=Path(args.checkpoint).resolve();ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    conf=ck.get('run_config',ck.get('config',{}));method=conf.get('method')
    if method!=args.method:raise ValueError(f'Source method mismatch: {method} vs {args.method}')
    if conf.get('phase') not in (None,'source_formation'):raise ValueError('No FT/MQ checkpoints permitted')
    if any(k.startswith(('encoder.','head.')) for k in ck['model']):raise ValueError('FT/newhead checkpoint prohibited')
    slots={'balls':9,'collision':4,'blocktower':4}[args.scene]
    # Container method Native has identical network state; no objective executes.
    model=adapter.PTCoPhy(models.CoPhyNet(slots),'Native').to(args.device).eval()
    model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False)
    source_sha=sha(checkpoint);inputs={}
    for split in ('train','val'):
        part=m['splits'][split];path=Path(part['cache'])
        if sha(path)!=part['cache_sha256']:raise ValueError('Audited AB cache changed')
        inputs[split]=dict(path=str(path),sha256=part['cache_sha256'])
    binding=dict(version=VERSION,status='COMPLETE',representation='U32',source_sha256=source_sha,
        source_checkpoint=str(checkpoint),source_method=method,source_selected_epoch=ck['epoch'],
        source_budget=args.source_budget,source_config=conf,visual_inputs=inputs,runtime_source=str(runtime),
        implementation_sha256=sha(__file__),base_manifest_sha256=sha(base/'manifest.json'),test_read=False)
    marker=out/'codes_complete.json'
    if marker.exists():
        old=read(marker)
        if {k:v for k,v in old.items() if k!='files'}!=binding:raise ValueError('Different frozen encoder binding')
        for p,h in old['files'].items():
            if sha(p)!=h:raise ValueError('Encoded source cache changed')
        return
    files={}
    for split in ('train','val'):
        with open(inputs[split]['path'],'rb') as f:cache=pickle.load(f)
        ids=m['splits'][split]['all_ids'];codes=[];presence=[]
        for first in range(0,len(ids),128):
            chosen=ids[first:first+128]
            for i in chosen:
                if cache[i].get('cache_version')!='ab_c_float32_v2':raise ValueError('Future-bearing cache rejected')
            pose=torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in chosen])).float().to(args.device)
            mask=torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in chosen])).float().to(args.device)
            u=model.encode_ab(adapter.ABObservation(pose,mask))
            if u.shape!=(len(chosen),slots,32) or not torch.isfinite(u).all():raise ValueError('Invalid full U32')
            codes.append(u.cpu().numpy());presence.append(mask.cpu().numpy())
        path=out/f'codes_{split}.npz';tmp=path.with_suffix('.tmp.npz')
        # p is the shared legacy storage key, NOT a claim this is only P16.
        np.savez(tmp,ids=np.asarray(ids),p=np.concatenate(codes),presence=np.concatenate(presence));os.replace(tmp,path)
        files[str(path)]=sha(path);emit('supervised_U32_encoded',scene=args.scene,method=method,split=split,episodes=len(ids))
    write(marker,dict(binding,files=files))



_source_probe = probe


def probe(args):
    """Pair own-source accessibility with actual selected-head S3 memory."""
    _source_probe(args)
    path=Path(args.out);selected=path/f'S{args.supports}'/args.reference/'selected.pt'
    if not selected.exists():
        raise FileNotFoundError('Train selected head before paired source/support probes: '+str(selected))
    data=Data(args);head=Head(data.dims,data.det_dims,32,data.horizon).to(args.device)
    ck=torch.load(selected,map_location=args.device,weights_only=False);head.load_state_dict(ck['model']);head.eval()
    report=read(path/'probes.json');prepath=ROOT/'prepared_v3'/args.scene/'training_preflight.json'
    if args.scene=='blocktower':prepath=Path(read(ROOT/'runtime_profiles.json')['scenes']['blocktower']['training_preflight']['path'])
    pre=read(prepath);parts={}
    from cophy_fields import PLANS
    spec=PLANS[args.scene];counts=[len(x) for x in spec['support']]
    names=list(spec['fields'])+(['gravity_x','gravity_y'] if args.scene=='blocktower' else [])
    for split in ('train','val'):
        item=pre['artifacts']['raw_relations_'+split]
        if sha(item['path'])!=item['sha256']:raise ValueError('Parameter artifact changed')
        metadata={(str(r['id']),int(r['slot'])):r for r in read(item['path'])}
        memories=[];targets=[];labels=[];ids=data.data[split]['ids']
        with torch.no_grad():
            for off in range(0,len(ids),128):
                ix=np.arange(off,min(off+128,len(ids)));q,det,mask,support,target=data.batch(split,ix,0,args.device)
                memory=head.support(support).mean(2).cpu().numpy();active=mask.cpu().numpy()>0
                for local,slot in zip(*np.where(active)):
                    r=metadata[(ids[ix[local]],int(slot))]
                    values=list(r['raw_physical'])+ (list(r['raw_gravity']) if args.scene=='blocktower' else [])
                    memories.append(memory[local,slot]);targets.append(values);labels.append(r['physical'])
        parts[split]=dict(x=np.asarray(memories,np.float64),values=np.asarray(targets,np.float64),labels=np.asarray(labels,np.int64))
    tr,va=parts['train'],parts['val'];mu=tr['x'].mean(0);sc=tr['x'].std(0).clip(1e-8)
    x=(tr['x']-mu)/sc;y=(va['x']-mu)/sc
    onehot=np.concatenate([np.eye(n)[tr['labels'][:,j]] for j,n in enumerate(counts)],1)
    target=np.concatenate((tr['values'],onehot),1);center=target.mean(0)
    w=np.linalg.solve(x.T@x+np.eye(64),x.T@(target-center));pred=y@w+center;fields={}
    for j,name in enumerate(names):
        mse=float(np.square(va['values'][:,j]-pred[:,j]).mean());v=float(va['values'][:,j].var())
        fields[name]=dict(r2=1-mse/v if v>0 else None,mse=mse)
    start=len(names)
    for j,(name,n) in enumerate(zip(spec['fields'],counts)):
        actual=va['labels'][:,j];guess=pred[:,start:start+n].argmax(1);start+=n
        fields[name].update(accuracy=float((guess==actual).mean()),balanced_accuracy=float(np.mean([
            (guess[actual==c]==c).mean() for c in range(n) if (actual==c).any()])))
    report['actual_support_memory64']=dict(fields=fields,train_objects=len(x),val_objects=len(y),
        source_checkpoint_sha256=report['source_checkpoint_sha256'],selected_head_sha256=sha(selected),
        selected_head_epoch=ck['epoch'],supports=args.supports,train_support_plan_epoch=0,
        representation='mean over S3 of selected head support MLP(frozen source U32); correct independent histories',
        probe_training='fixed ridge alpha1/train-only normalization; no source/head optimizer updates')
    write(path/'probes.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare-base','encode','smoke','train','probe'))
    p.add_argument('--scene',required=True,choices=('balls','collision','blocktower'))
    p.add_argument('--root',default=str(ROOT));p.add_argument('--out',required=True);p.add_argument('--base')
    p.add_argument('--checkpoint');p.add_argument('--method');p.add_argument('--runtime-source')
    p.add_argument('--source-budget',type=int,choices=(50,100,150),default=50)
    p.add_argument('--supports',type=int,choices=(3,),default=3);p.add_argument('--epochs',type=int,choices=(100,),default=100)
    p.add_argument('--reference',choices=('learned','query','known'),default='learned');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();ROOT=Path(a.root);Path(a.out).mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    with open(Path(a.out)/f'{a.command}_{a.supports}_{a.reference}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare-base':prepare_base,'encode':encode,'smoke':smoke,'train':train,'probe':probe}[a.command](a)
