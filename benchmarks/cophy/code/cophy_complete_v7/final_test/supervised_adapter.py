"""Legacy CoPhyNet U32 adapter for the separately frozen final-test stage.

No work runs on import. fit() reads TRAIN probe rows only; evaluate() only
applies already frozen source/head/probe weights after the outer freeze gate.
The native source remains the original supervised model, never FT/MQ.
"""
import argparse
import importlib
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from runtime import VERSION, artifact, checked, field_names, load_core, physical_values, read, sha, write

LEGACY_VERSION = 'cophy-v7-legacy-supervised-frozen-U32-readout'
ADAPTER_ABI = 'legacy-supervised-U32-final-test-v1'
SHAPES = {'balls': (30, 9, 2, 27, 1), 'collision': (15, 4, 3, 12, 4),
          'blocktower': (30, 4, 3, 27, 1)}


def _encoding(rd, core_path):
    marker = read(rd/'codes_complete.json')
    if (marker.get('status') != 'COMPLETE' or marker.get('version') != LEGACY_VERSION
            or marker.get('representation') != 'U32' or marker.get('test_read') is not False
            or marker.get('implementation_sha256') != sha(core_path)):
        raise ValueError('Use the original bound legacy U32 encoding implementation')
    if marker.get('source_budget') != 100: raise ValueError('Final test uses the fixed source100 budget')
    conf = marker['source_config']
    if conf.get('phase') not in (None, 'source_formation') or conf.get('seed') != 0:
        raise ValueError('Only original seed0 supervised source models are supported')
    return marker


def _head(rd, marker):
    folder = rd/'S3/learned'; conf = read(folder/'config.json'); complete = read(folder/'complete.json')
    ck = torch.load(folder/'selected.pt', map_location='cpu', weights_only=False)
    selected = read(folder/'selected_validation.json'); results = read(folder/'results.json')
    if (complete.get('status') != 'COMPLETE' or complete.get('epochs') != 100
            or conf.get('epochs') != 100 or conf.get('version') != LEGACY_VERSION
            or conf.get('seed') != 0 or conf.get('supports') != 3
            or (conf.get('support_dims') is not None and conf['support_dims'] != 32)
            or conf.get('test_read') is not False):
        raise ValueError('The final supervised head must be the completed original U32 head100')
    if conf['code_sha256'] != sha(rd/'codes_complete.json') or ck['config'] != conf:
        raise ValueError('Head/encoder binding changed')
    if ck['epoch'] != selected['epoch'] or ck['epoch'] != results['selected_epoch'] or len(selected['ids']) != 512:
        raise ValueError('Selected head changed or was not selected on original validation512')
    if conf.get('encoder_frozen') is not True or conf.get('reference') != 'learned':
        raise ValueError('Only a frozen source with a learned downstream head is supported')
    return conf, ck


def fit(args):
    """Called by fit_probes.py. No validation/test vectors enter any fit."""
    from fit_probes import fit as ridge_fit
    core = load_core(args.core)
    if core.VERSION != LEGACY_VERSION: raise ValueError('Wrong supervised readout core')
    rd = Path(args.readout); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    enc = _encoding(rd, args.core); conf, ck = _head(rd, enc)
    base = Path(args.base) if args.base else Path(next(iter(conf['input_sha256']))).parent
    if sha(base/'manifest.json') != conf['base_sha256'] or sha(base/'manifest.json') != enc['base_manifest_sha256']:
        raise ValueError('Wrong original supervised base')
    manifest = read(base/'manifest.json'); scene = conf['scene']
    raw_path = checked(dict(path=args.raw_train_relations, sha256=args.raw_train_sha256)); records = read(raw_path)
    if any(r['split'] != 'train' for r in records): raise ValueError('Probe fitting accepts TRAIN records only')
    code_path = rd/'codes_train.npz'
    if sha(code_path) != enc['files'][str(code_path)]: raise ValueError('Frozen training U32 cache changed')
    with np.load(code_path, allow_pickle=False) as z:
        code_ids = z['ids'].astype(str).tolist(); codes = z['p'].copy(); seen = z['presence'].copy()
    part = manifest['splits']['train']; all_ids = list(map(str, part['all_ids'])); lut = {q: i for i, q in enumerate(code_ids)}
    order = [lut[q] for q in all_ids]; codes = codes[order]; seen = seen[order]; lut = {q:i for i,q in enumerate(all_ids)}
    if codes.shape != (len(all_ids), SHAPES[scene][1], 32) or not np.isfinite(codes).all():
        raise ValueError('Invalid original full U32 training cache')
    # Exactly the legacy head's TRAIN-only normalization, including dtype.
    active = codes[seen > 0]; mean = active.mean(0); scale = active.std(0).clip(1e-6)
    normalization = dict(status='TRAIN_NORMALIZATION_COMPLETE', version=VERSION, fit_split='train',
        mean=mean.tolist(), scale=scale.tolist(), rows=len(active), encoding_sha256=sha(rd/'codes_complete.json'),
        formula='legacy float32 mean/std over AB-visible TRAIN U32; scale clipped at 1e-6',
        validation_used=False, test_read=False)
    write(out/'normalization.json', normalization, immutable=True)
    input_path = base/'input_train.npz'
    if sha(input_path) != conf['input_sha256'][str(input_path)]: raise ValueError('Frozen training current mask changed')
    with np.load(input_path, allow_pickle=False) as x:
        ids = x['ids'].astype(str).tolist(); mask = x['presence'].copy()
    if ids != part['query_ids']: raise ValueError('Training support query order differs')
    # Reuse only the unchanged legacy sampler. No Data constructor, no val
    # arrays, current pose, or prediction targets are needed to fit these probes.
    data = SimpleNamespace(args=SimpleNamespace(supports=3), manifest=manifest, slots=SHAPES[scene][1],
        learned=True, pool_cache={}, data={'train':dict(ids=ids, mask=mask, donor_seen=seen)})
    data.pools = MethodType(core.Data.pools, data); data.plan = MethodType(core.Data.plan, data)
    plan = data.plan('train', 0)
    head = core.Head(SHAPES[scene][2], SHAPES[scene][4], 32, SHAPES[scene][3]).to(args.device)
    head.load_state_dict(ck['model'], strict=True); head.eval(); head.requires_grad_(False)
    meta = {(str(r['id']), int(r['slot'])): r for r in records}
    datasets = {k: dict(x=[], y=[], labels=[]) for k in ('own', 'support_mean', 'memory')}
    def append(channel, vector, record):
        ds = datasets[channel]; ds['x'].append(vector); ds['y'].append(physical_values(record, scene)); ds['labels'].append(record['physical'])
    for r in records:
        ident, slot = str(r['id']), int(r['slot'])
        if r['in_C'] and ident in lut and seen[lut[ident], slot] > 0: append('own', codes[lut[ident], slot], r)
    with torch.no_grad():
        for first in range(0, len(ids), 128):
            ix = np.arange(first, min(first+128, len(ids))); chosen = plan[ix]
            if (chosen < 0).any(): raise ValueError('Training correct support is incomplete')
            raw = codes[chosen, np.arange(data.slots)[None, :, None]]
            normalized = ((raw-mean)/scale)*mask[ix, :, None, None]
            memory = head.support(torch.as_tensor(normalized, dtype=torch.float32, device=args.device)).mean(2).cpu().numpy()
            average = raw.mean(2)
            for i, slot in zip(*np.where(mask[ix] > 0)):
                r = meta[(ids[ix[i]], int(slot))]
                append('support_mean', average[i, slot], r); append('memory', memory[i, slot], r)
    dependency = dict(version=VERSION, status='TRAIN_FIT_COMPLETE', adapter_abi=ADAPTER_ABI, scene=scene,
        source_epochs=100, source_selected_epoch=enc['source_selected_epoch'], method=enc['source_method'],
        estimator='train-standardized ridge alpha=1; numeric physical values plus category one-hot', fields=field_names(scene),
        fit_split='train', validation_used=False, test_read=False, source_optimizer_steps=0, head_optimizer_steps=0,
        train_plan_epoch=0, train_plan_sha256=__import__('hashlib').sha256(np.ascontiguousarray(plan).tobytes()).hexdigest(),
        readout=artifact(args.core), encoding=artifact(rd/'codes_complete.json'), normalization=artifact(out/'normalization.json'),
        source_checkpoint=dict(path=enc['source_checkpoint'], sha256=enc['source_sha256']),
        head_checkpoint=artifact(rd/'S3/learned/selected.pt'), raw_train_relations=artifact(raw_path),
        implementation=artifact(__file__), runtime=artifact(Path(__file__).with_name('runtime.py')))
    if (out/'probe_fit.json').exists():
        old = read(out/'probe_fit.json')
        if {k:v for k,v in old.items() if k != 'channels'} != dependency: raise ValueError('Different frozen supervised probe fit')
        for channel in old['channels'].values(): checked(channel['artifact'])
        print('SUPERVISED_TRAIN_FIT_ALREADY_COMPLETE'); return
    dependency['channels'] = {k: ridge_fit(v['x'], v['y'], v['labels'], out/(k+'.npz')) for k,v in datasets.items()}
    dependency['channels']['own']['attribution'] = 'original source OWN full AB U32; not P16 and no query encoding'
    dependency['channels']['support_mean']['attribution'] = 'mean of raw independent S3 full AB U32'
    dependency['channels']['memory']['attribution'] = 'after frozen downstream head support projection; not source-only information'
    write(out/'probe_fit.json', dependency, immutable=True); print('SUPERVISED_TRAIN_FIT_COMPLETE')


def freeze_entry(e, bind):
    """Called by freeze_gate before any test producer can run."""
    rd = Path(e['readout']); enc = _encoding(rd, e['core']); conf, head_ck = _head(rd, enc)
    if conf['scene'] != e['scene'] or enc['source_method'] != e['source_method']:
        raise ValueError('Supervised scene/source identity differs')
    base = Path(e.get('base') or Path(next(iter(conf['input_sha256']))).parent)
    if sha(base/'manifest.json') != conf['base_sha256'] or sha(base/'manifest.json') != enc['base_manifest_sha256']:
        raise ValueError('Supervised base binding differs')
    source = bind(enc['source_checkpoint'])
    if source['sha256'] != enc['source_sha256']: raise ValueError('Frozen supervised source changed')
    source_ck = torch.load(source['path'], map_location='cpu', weights_only=False)
    source_conf = source_ck.get('run_config', source_ck.get('config', {}))
    if source_conf != enc['source_config'] or not 0 < source_ck['epoch'] <= 100 or source_ck['epoch'] != enc['source_selected_epoch']:
        raise ValueError('Wrong supervised selected checkpoint')
    if any(k.startswith(('encoder.', 'head.')) for k in source_ck['model']): raise ValueError('FT/MQ source checkpoint prohibited')
    receipt_path = Path(e.get('source_budget_receipt') or Path(source['path']).parent/'budget_100_complete.json')
    receipt = read(receipt_path); source_binding = read(receipt_path.parent/'binding.json')
    if (receipt.get('status') != 'COMPLETE' or receipt.get('epochs') != 100 or receipt.get('test_read') is not False
            or receipt.get('selected_epoch') != source_ck['epoch'] or receipt.get('selected_sha256') != source['sha256']
            or Path(receipt['selected_checkpoint']).resolve() != Path(source['path']).resolve()
            or receipt['binding_sha256'] != source_binding['sha256'] or receipt['scene'] != e['scene']):
        raise ValueError('Source100 budget receipt does not certify this validation-selected checkpoint')
    end = bind(receipt['checkpoint'])
    if end['sha256'] != receipt['checkpoint_sha256']: raise ValueError('Source100 terminal checkpoint changed')
    terminal = torch.load(end['path'], map_location='cpu', weights_only=False)
    if terminal.get('epoch') != 100 or terminal.get('test_read') is not False or terminal.get('extension_binding_sha256') != source_binding['sha256']:
        raise ValueError('Source continuation did not complete exactly budget100')
    del terminal, source_ck
    runtime = Path(enc['runtime_source']); runtime_files = []
    for name in ('cophy_adapter.py','cophy_protocol.py','cophy_training.py','cf_learning/model.py','derendering/model.py',
                 'cf_learning/__init__.py','derendering/__init__.py'):
        path = runtime/name
        if not path.exists() and name.endswith('__init__.py'): continue
        item = bind(path); recorded = source_binding['files'].get(str(path.resolve()))
        if name in ('cophy_adapter.py','cophy_training.py','cf_learning/model.py') and recorded is None:
            raise ValueError('Source continuation did not bind its actual encoder implementation')
        if recorded is not None and item['sha256'] != recorded: raise ValueError('Changed source runtime: '+name)
        runtime_files.append(item)
    for filename, expected in enc['files'].items():
        if bind(filename)['sha256'] != expected: raise ValueError('Changed frozen U32 cache')
    for filename, expected in conf['input_sha256'].items():
        if bind(filename)['sha256'] != expected: raise ValueError('Changed head training/validation data')
    head = bind(rd/'S3/learned/selected.pt'); fit_path = Path(e['probe_fit']); fitted = read(fit_path)
    if (fitted.get('status') != 'TRAIN_FIT_COMPLETE' or fitted.get('adapter_abi') != ADAPTER_ABI
            or fitted.get('fit_split') != 'train' or fitted.get('validation_used') is not False or fitted.get('test_read') is not False
            or fitted.get('source_epochs') != 100 or fitted['head_checkpoint']['sha256'] != head['sha256']
            or fitted['source_checkpoint']['sha256'] != source['sha256']
            or fitted['encoding']['sha256'] != sha(rd/'codes_complete.json')
            or set(fitted['channels']) != {'own','support_mean','memory'}):
        raise ValueError('Complete train-only supervised probe weights are required')
    for channel in fitted['channels'].values(): bind(checked(channel['artifact']))
    for key in ('readout','encoding','normalization','source_checkpoint','head_checkpoint','raw_train_relations','implementation','runtime'):
        bind(checked(fitted[key]))
    for name in ('config.json','complete.json','selected_validation.json','results.json'): bind(rd/'S3/learned'/name)
    bind(rd/'codes_complete.json'); bind(base/'manifest.json'); bind(receipt_path.parent/'binding.json')
    return dict(e, family='supervised', adapter_abi=ADAPTER_ABI, readout=str(rd.resolve()), core=bind(e['core']),
        source=source, model_code=bind(runtime/'cf_learning/model.py'), runtime_source=str(runtime.resolve()),
        runtime_files=runtime_files, head=head, probe_fit=bind(fit_path), normalization=bind(checked(fitted['normalization'])),
        support_dims=32, source_epochs=100, source_selected_epoch=enc['source_selected_epoch'], head_epochs=100,
        selected_head_epoch=head_ck['epoch'], source_budget_receipt=bind(receipt_path), source_terminal=end,
        encoding_binding=enc)


def _source(entry, device):
    runtime = Path(entry['runtime_source']).resolve()
    for item in entry['runtime_files']: checked(item)
    # Separate evaluation workers normally import one runtime. Reject a mixed
    # cached import rather than accidentally using another scene's old model.
    sys.path.insert(0, str(runtime))
    adapter = importlib.import_module('cophy_adapter'); models = importlib.import_module('cf_learning.model')
    if Path(adapter.__file__).resolve().parent != runtime or runtime not in Path(models.__file__).resolve().parents:
        raise ValueError('Mixed supervised runtime imports; use one entry per worker')
    with torch.random.fork_rng(devices=[]):
        model = adapter.PTCoPhy(models.CoPhyNet(SHAPES[entry['scene']][1]), 'Native')
    ck = torch.load(checked(entry['source']), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=True); model.to(device); model.eval(); model.requires_grad_(False)
    return model, adapter


@torch.no_grad()
def _encode(entry, bundle, out, device):
    marker_path = out/'supervised_encoding.json'
    identity = dict(adapter_abi=ADAPTER_ABI, entry=entry['id'], source_sha256=entry['source']['sha256'],
                    bundle_sha256=sha(bundle['_path']), implementation_sha256=sha(__file__))
    if marker_path.exists():
        old = read(marker_path)
        if old['identity'] != identity: raise ValueError('Different existing supervised test encoding')
        for item in old['files']: checked(item)
        return old
    ab = np.load(checked(bundle['inputs']['pose_ab']), mmap_mode='r', allow_pickle=False)
    seen = np.load(checked(bundle['inputs']['pose_presence_ab']), mmap_mode='r', allow_pickle=False)
    frames, slots, _, _, _ = SHAPES[entry['scene']]; n = len(bundle['query_ids'])
    if ab.shape != (n, frames, slots, 3) or ab.dtype != np.float32 or seen.shape != (n, slots):
        raise ValueError('Expected full FP32 visual AB xyz and first-frame AB presence')
    if not np.isfinite(ab).all() or not np.isfinite(seen).all() or not np.isin(seen, (0,1)).all():
        raise ValueError('Nonfinite/bad supervised AB observation')
    model, adapter = _source(entry, device); path = out/'supervised_u32.npy'
    codes = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, slots, 32))
    for first in range(0, n, 128):
        pose = torch.as_tensor(np.array(ab[first:first+128], copy=True), device=device)
        mask = torch.as_tensor(np.asarray(seen[first:first+128], dtype=np.float32), device=device)
        value = model.encode_ab(adapter.ABObservation(pose, mask))
        if value.shape != (len(pose), slots, 32) or not torch.isfinite(value).all(): raise ValueError('Invalid frozen test U32')
        codes[first:first+len(pose)] = value.cpu().numpy()
    codes.flush(); del codes, model
    marker = dict(status='COMPLETE', version=VERSION, identity=identity, codes=artifact(path),
        presence=bundle['inputs']['pose_presence_ab'], files=[artifact(path), bundle['inputs']['pose_presence_ab']],
        source_observations='full AB predicted xyz only; AB frame0 presence, never query future or labels', test_read=True, optimizer_steps=0)
    write(marker_path, marker, immutable=True); return marker


class TestData:
    def __init__(self, entry, bundle, encoding):
        self.support_dims = 32; self.slots = SHAPES[entry['scene']][1]; self.scene = entry['scene']
        self.codes = np.load(checked(encoding['codes']), mmap_mode='r'); self.seen = np.load(checked(encoding['presence']), mmap_mode='r')
        norm = read(checked(entry['normalization']))
        if norm.get('fit_split') != 'train' or norm.get('test_read') is not False: raise ValueError('Normalization must be frozen on TRAIN')
        self.mean = np.asarray(norm['mean'], np.float32); self.scale = np.asarray(norm['scale'], np.float32)
        if self.mean.shape != (32,) or self.scale.shape != (32,) or (self.scale <= 0).any(): raise ValueError('Bad U32 normalization')
        ids = bundle['query_ids']
        with np.load(checked(bundle['inputs']['input']), allow_pickle=False) as x:
            if x['ids'].astype(str).tolist() != ids: raise ValueError('Test query order changed')
            row = dict(ids=ids, q=x['pose'].copy(), det=x['detected'].copy(), mask=x['presence'].copy())
        with np.load(checked(bundle['inputs']['target']), allow_pickle=False) as y:
            if y['ids'].astype(str).tolist() != ids: raise ValueError('Test target order changed')
            row['target'] = y['pose'].copy()
        self.data = {'test': row}; self.dims = row['q'].shape[-1]; self.horizon = row['target'].shape[1]
        self.det_dims = 1 if row['det'].ndim == 3 else row['det'].shape[-1]
        if (self.dims, self.horizon, self.det_dims) != (SHAPES[self.scene][2], SHAPES[self.scene][3], SHAPES[self.scene][4]):
            raise ValueError('Changed original supervised query/target task')
        if (row['q'].shape != (len(ids),3,self.slots,self.dims)
                or row['target'].shape != (len(ids),self.horizon,self.slots,self.dims)
                or row['mask'].shape != (len(ids),self.slots)
                or not all(np.isfinite(row[k]).all() for k in ('q','det','mask','target'))):
            raise ValueError('Invalid fixed supervised test tensors')
        with np.load(checked(bundle['plans']), allow_pickle=False) as z: self.plans = {k:z[k].copy() for k in z.files}
        for arm in ('correct', 'wrong'):
            plan = self.plans[arm]; eligible = self.plans[arm+'_eligible']
            if plan.shape != (len(ids),self.slots,3) or eligible.shape != (len(ids),): raise ValueError('Wrong frozen support shape')
            for i in np.flatnonzero(eligible):
                for slot in np.flatnonzero(row['mask'][i] > 0):
                    chosen = plan[i, slot]
                    if (chosen < 0).any() or (chosen >= len(ids)).any() or i in chosen or len(set(chosen.tolist())) != 3:
                        raise ValueError('Invalid independent test S3 support')
                    if not (self.seen[chosen, slot] > 0).all():
                        raise ValueError('Shared plan includes a donor absent at supervised AB frame0; producer must use common visibility')

    def raw_support(self, ix, arm):
        plan = self.plans['wrong' if arm == 'wrong' else 'correct'][ix]
        if (plan < 0).any(): raise ValueError('Unsupported query must be excluded by fixed eligibility')
        return self.codes[plan, np.arange(self.slots)[None,:,None]]

    def batch(self, ix, arm, device):
        row = self.data['test']; support = None
        if arm != 'null': support = ((self.raw_support(ix,arm)-self.mean)/self.scale)*row['mask'][ix,:,None,None]
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        return t(row['q'][ix]), t(row['det'][ix]), t(row['mask'][ix]), None if support is None else t(support), t(row['target'][ix])


@torch.no_grad()
def _probes(head, data, entry, bundle, device):
    from evaluate import probe_apply
    fit_result = read(checked(entry['probe_fit'])); records = read(checked(bundle['inputs']['raw_relations']))
    if any(r['split'] != 'test' for r in records): raise ValueError('Apply only explicitly audited test labels')
    ids = bundle['query_ids']; lut = {q:i for i,q in enumerate(ids)}; meta = {(str(r['id']),int(r['slot'])):r for r in records}
    datasets = {k: dict(x=[], y=[], labels=[]) for k in ('own','support_mean','memory')}
    def add(channel, vector, r):
        d = datasets[channel]; d['x'].append(vector); d['y'].append(physical_values(r,entry['scene'])); d['labels'].append(r['physical'])
    for r in records:
        ident, slot = str(r['id']), int(r['slot'])
        if r['in_C'] and data.seen[lut[ident],slot] > 0: add('own',data.codes[lut[ident],slot],r)
    valid = np.flatnonzero(data.plans['correct_eligible'])
    for first in range(0,len(valid),128):
        ix = valid[first:first+128]; raw = data.raw_support(ix,'matched'); _,_,mask,support,_ = data.batch(ix,'matched',device)
        memory = head.support(support).mean(2).cpu().numpy(); average = raw.mean(2)
        for i,slot in zip(*np.where(data.data['test']['mask'][ix] > 0)):
            r = meta[(ids[ix[i]],int(slot))]; add('support_mean',average[i,slot],r); add('memory',memory[i,slot],r)
    return {k:dict(probe_apply(fit_result['channels'][k]['artifact'],**v,names=fit_result['fields']),
                   attribution=fit_result['channels'][k]['attribution'],fit_artifact=fit_result['channels'][k]['artifact'])
            for k,v in datasets.items()}


@torch.no_grad()
def _official(entry, bundle, encoding, device):
    """Original source predictor on own AB + one FP32 C frame, CD[1:].

    The source task uses every official test episode, not the new task's
    donor-supported subset. The primary mask is the original visual C mask;
    the existing cophy_evaluate GT-mask diagnostic is reported separately.
    """
    ids = bundle['query_ids']; frames, slots, dims, _, _ = SHAPES[entry['scene']]
    cpath = checked(bundle['inputs']['official_c']); targetpath = checked(bundle['inputs']['target_official'])
    with np.load(cpath, allow_pickle=False) as z:
        if z['ids'].astype(str).tolist() != ids: raise ValueError('Official C ID order changed')
        c = z['pose'].copy(); presence = z['presence'].copy()
    if c.shape != (len(ids),1,slots,3) or c.dtype != np.float32 or presence.shape != (len(ids),slots):
        raise ValueError('Official source task requires one full-xyz FP32 C frame and its visual presence')
    if not np.isfinite(c).all() or not np.isin(presence,(0,1)).all(): raise ValueError('Invalid official visual C')
    with np.load(targetpath, allow_pickle=False) as z:
        if z['ids'].astype(str).tolist() != ids: raise ValueError('Official target ID order changed')
        target = z['pose'].copy()
    if target.shape != (len(ids),frames-1,slots,dims) or not np.isfinite(target).all():
        raise ValueError('Official target must be full CD[1:] with Balls xy / other scenes xyz')
    ab = np.load(checked(bundle['inputs']['pose_ab']),mmap_mode='r',allow_pickle=False)
    seen = np.load(checked(bundle['inputs']['pose_presence_ab']),mmap_mode='r',allow_pickle=False)
    own = np.load(checked(encoding['codes']),mmap_mode='r',allow_pickle=False)
    model, adapter = _source(entry,device)
    training = importlib.import_module('cophy_training')
    if Path(training.__file__).resolve().parent != Path(entry['runtime_source']).resolve():
        raise ValueError('Wrong original official metric implementation')
    metric_binding = next((a for a in entry['runtime_files'] if Path(a['path']).name=='cophy_training.py'),None)
    if metric_binding is None or sha(training.__file__) != metric_binding['sha256']:
        raise ValueError('Official reduction was not frozen with the source runtime')
    records = read(checked(bundle['inputs']['raw_relations'])); lut = {q:i for i,q in enumerate(ids)}
    gtmask = np.zeros((len(ids),slots),np.float32); seen_records=set()
    for r in records:
        ident,slot=str(r['id']),int(r['slot'])
        if r['split']!='test' or ident not in lut or not 0<=slot<slots or (ident,slot) in seen_records:
            raise ValueError('Bad audited test object record for official scoring')
        seen_records.add((ident,slot));gtmask[lut[ident],slot]=float(bool(r['in_C']))
    rows=[]
    for first in range(0,len(ids),64):
        end=min(first+64,len(ids));t=lambda v:torch.as_tensor(np.array(v,copy=True),dtype=torch.float32,device=device)
        visual=adapter.VisualInput(adapter.ABObservation(t(ab[first:end]),t(seen[first:end])),
                                   t(c[first:end]),t(presence[first:end]))
        # code_for_task is the same own encode_ab for Native/A/Random. The
        # frozen cache already contains that exact original U32 computation.
        prediction,visual_mask,_=model.predict_code(t(own[first:end]),visual)
        if prediction.shape!=(end-first,frames-1,slots,3) or not torch.isfinite(prediction).all():
            raise ValueError('Original source prediction shape/nonfinite output')
        if not torch.equal(visual_mask,visual.presence_c): raise ValueError('Official prediction changed visual coverage')
        truth=t(target[first:end]);gm=t(gtmask[first:end]);copy=visual.c.expand(-1,frames-1,-1,-1)
        columns={}
        for label,value in (('model',prediction),('CopyC',copy)):
            # Original helper checks equal shape before slicing dimensions;
            # target_official has already removed Balls z, so truncate both.
            value=value[...,:dims]
            columns[label]=training.mse_per_recipient(value,truth,visual_mask,dims,allow_uncovered=True).cpu().tolist()
            columns[label+'_gtmask']=training.mse_per_recipient(value,truth,gm,dims,allow_uncovered=True).cpu().tolist()
        for i,ident in enumerate(ids[first:end]):
            rows.append(dict(id=ident,**{name:float(values[i]) if np.isfinite(values[i]) else None for name,values in columns.items()}))
    official={}
    for key in ('model','CopyC','model_gtmask','CopyC_gtmask'):
        values=[r[key] for r in rows if r[key] is not None];official[key]=float(np.mean(values)) if values else None
    covered=sum(r['model'] is not None for r in rows)
    return dict(status='COMPLETE',task='original supervised CoPhy counterfactual source task',official=official,
        official_rows=rows,official_test_count=len(ids),scored_recipients=covered,visual_coverage=covered/len(ids),
        uncovered_ids=[r['id'] for r in rows if r['model'] is None],horizon=frames-1,
        score_frames=f'CD[1:{frames}]',metric_dims=dims,
        metric='mean xyz/xy squared error over time and visually present C objects per recipient, then mean over visually covered recipients',
        primary_mask='same frozen FP32 derenderer presence at C frame0',
        auxiliary_gt_mask='audited raw_relations.in_C; scoring only, absent from source inputs',
        source_checkpoint=entry['source'],source_budget=100,source_selected_epoch=entry['source_selected_epoch'],
        source_predictor='original bound PTCoPhy.predict_code / CoPhyNet.pred_D, including original scene stability gating',
        CopyC='D_hat[t] := predicted visual C0 for every original future frame; no training',
        official_input=bundle['inputs']['official_c'],official_target=bundle['inputs']['target_official'],
        metric_implementation=metric_binding,parameter_probe_channel='probes.own (same frozen source U32)',
        no_new_head_used=True,not_the_cross_context_readout=True,test_read=True,optimizer_steps=0)


@torch.no_grad()
def evaluate(entry, bundle, out, device):
    """Only called by evaluate.run after its global freeze/bundle verification."""
    from evaluate import score
    if (entry.get('adapter_abi') != ADAPTER_ABI or entry.get('source_epochs') != 100 or entry.get('head_epochs') != 100
            or bundle.get('split') != 'test' or bundle.get('status') != 'PREPARED' or bundle['scene'] != entry['scene']):
        raise ValueError('Unqualified supervised final-test entry or explicit test bundle')
    core = load_core(checked(entry['core']))
    if core.VERSION != LEGACY_VERSION: raise ValueError('Wrong frozen supervised core')
    encoding = _encode(entry,bundle,out,device); data = TestData(entry,bundle,encoding)
    head = core.Head(data.dims,data.det_dims,32,data.horizon).to(device)
    ck = torch.load(checked(entry['head']),map_location='cpu',weights_only=False)
    head.load_state_dict(ck['model'],strict=True);head.eval();head.requires_grad_(False)
    if ck['epoch'] != entry['selected_head_epoch']: raise ValueError('Frozen head selection changed')
    good=np.flatnonzero(data.plans['correct_eligible']);wrong=np.flatnonzero(data.plans['wrong_eligible'])
    arms={arm:score(core,head,data,wrong if arm=='wrong' else good,arm,device) for arm in ('matched','null','wrong')}
    arms['matched_on_wrong_cohort']=score(core,head,data,wrong,'matched',device)
    arms['null_on_wrong_cohort']=score(core,head,data,wrong,'null',device)
    return dict(arms=arms,probes=_probes(head,data,entry,bundle,device),
                official_task=_official(entry,bundle,encoding,device),selected_head_epoch=ck['epoch'],
                source_selected_epoch=entry['source_selected_epoch'],correct_eligible_count=len(good),wrong_eligible_count=len(wrong),
                unsupported_query_ids=bundle['unsupported_query_ids'],adapter_abi=ADAPTER_ABI,
                source_observation='full AB predicted xyz with AB frame0 presence; no GT/source parameter labels',
                null_semantics='original legacy head receives support=None; its public query3 remains unchanged')
