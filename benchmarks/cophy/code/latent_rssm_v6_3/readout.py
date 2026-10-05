"""Frozen v6 visual-dynamics representations, shared fresh coordinate readout.

Coordinate labels are used only here, after source training. Test is never read.
The graph/GRU head follows the existing CoPhy independent-experience readout;
the support projection accepts frozen P64 instead of the old U32. No source optimizer
or old prediction head is imported. First512 validation selects the new head;
evaluation covers the bound base, currently 512 for Balls and Collision.
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

VERSION = 'cophy-rssm-v6.3-frozen-P64-pose-prefix-readout'
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


@torch.no_grad()
def encode(args):
    from models import load_checkpoint
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    source = Path(args.checkpoint); digest = sha(source)
    marker = out/'codes_complete.json'
    feature_binding = {}
    for split in ('train', 'val'):
        base = Path(args.features)/split
        complete = read(base/'COMPLETE.json')
        if complete.get('status') != 'COMPLETE' or complete['manifest_sha256'] != sha(base/'manifest.json'):
            raise ValueError('Visual cache incomplete or changed')
        feature_binding[split] = dict(complete_sha256=sha(base/'COMPLETE.json'), ids_sha256=sha(base/'ids.json'))
    if marker.exists():
        old = read(marker)
        if (old.get('status') != 'COMPLETE' or old.get('version') != VERSION or
            old.get('representation') != 'P64' or old['source_sha256'] != digest or
            old.get('feature_binding') != feature_binding):
            raise ValueError('Changed source, representation version, or visual cache; use a new output')
        for filename, expected in old['files'].items():
            if sha(filename) != expected: raise ValueError('Encoded P cache changed')
        return
    loaded = load_checkpoint(source, args.device)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    files = {}
    for split in ('train', 'val'):
        base = Path(args.features)/split
        if read(base/'COMPLETE.json').get('status') != 'COMPLETE': raise ValueError('Visual cache incomplete')
        ids = list(map(str, read(base/'ids.json')))
        x = np.load(base/'features_ab.npy', mmap_mode='r')
        mask = np.load(base/'presence_ab.npy', mmap_mode='r')
        codes, present = [], []
        for i in range(0, len(ids), 128):
            xb = torch.from_numpy(np.array(x[i:i+128], dtype=np.float32)).to(args.device)
            mb = torch.from_numpy(np.array(mask[i:i+128], dtype=np.float32)).to(args.device)
            p = model.encode(xb, mb)
            if not torch.is_tensor(p) or p.shape[1:] != (x.shape[2], 64): raise ValueError('Expected per-object P64')
            codes.append(p.float().cpu().numpy()); present.append(np.asarray(mask[i:i+128]).max(1))
        dest = out/f'codes_{split}.npz'
        tmp = dest.with_suffix('.tmp.npz')
        np.savez(tmp, ids=np.asarray(ids), p=np.concatenate(codes), presence=np.concatenate(present)); os.replace(tmp, dest)
        files[str(dest)] = sha(dest)
        emit('source_encoded', scene=args.scene, split=split, episodes=len(ids))
    write(marker, dict(status='COMPLETE', version=VERSION, representation='P64', source_sha256=digest,
                       feature_root=str(args.features), feature_binding=feature_binding, files=files, test_read=False))


class Data:
    def __init__(self, args):
        self.args = args; self.base = Path(args.base); self.out = Path(args.out)
        self.manifest = read(self.base/'manifest.json')
        if self.manifest.get('test_read') is not False or self.manifest['prefix'] != 3: raise ValueError('Unexpected downstream task')
        self.data = {}; self.code_mean = self.code_scale = None
        self.learned = args.reference == 'learned'
        if self.learned:
            binding = read(self.out/'codes_complete.json')
            if binding.get('version') != VERSION or binding.get('representation') != 'P64':
                raise ValueError('Readout requires v6.2 frozen P64 codes')
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
        self.support_dims = train['parameters'].shape[-1] if args.reference == 'known' else 64
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
    if marker.get('version')!=VERSION or marker.get('representation')!='P64' or marker.get('status')!='COMPLETE':
        raise ValueError('Probe requires the current frozen P64 encoding')
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
        ridge_alpha=1.,normalization='train only',representation='P64; no donor transient input')
    for view, sl in dict(P=slice(0,64)).items():
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
    write(Path(args.out)/'probes.json', result)
    emit('probe_complete', scene=args.scene)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('encode','train','probe','smoke'))
    p.add_argument('--scene', required=True, choices=('balls','collision','blocktower'))
    p.add_argument('--out', required=True); p.add_argument('--base'); p.add_argument('--features'); p.add_argument('--checkpoint')
    p.add_argument('--supports', type=int, choices=(3,8), default=3); p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--reference', choices=('learned','query','known'), default='learned'); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    with open(Path(a.out)/f'{a.command}_{a.supports}_{a.reference}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'encode':encode,'train':train,'probe':probe,'smoke':smoke}[a.command](a)
