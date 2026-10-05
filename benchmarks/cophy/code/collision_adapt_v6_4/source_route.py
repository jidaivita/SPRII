"""Paired continuation of one Collision Cross50 source: focal versus all P.

This is a new experiment binding. Original files/checkpoints are read-only.
The model format remains the original sig02 format for frozen readout loading.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

VERSION = 'collision-cross-source-routing-v6.4-1'
CORE_VERSION = 'cophy-latent-v6.2-sig02'
ROUTES = ('focal', 'all')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, p)


def immutable(p, value):
    if Path(p).exists() and read(p) != value:
        raise ValueError('Different frozen artifact: ' + str(p))
    write(p, value)


def emit(event, **kw):
    print(json.dumps(dict(event=event, time=time.time(), **kw), allow_nan=False), flush=True)


def runtime(args):
    global np, torch
    import numpy as np
    import torch
    root = Path(args.runtime_dir).resolve()
    if 'models' in sys.modules and Path(sys.modules['models'].__file__).resolve() != root / 'models.py':
        raise RuntimeError('A different models module is already imported')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_collision_source_route_parent_train', root / 'train.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    if core.VERSION != CORE_VERSION:
        raise ValueError('Expected the original sig02 runtime')
    torch.set_num_threads(args.threads)
    return core


def load_parent(args, core):
    source = Path(args.source_checkpoint).resolve()
    complete = read(source.parent / 'complete.json')
    if (complete.get('status'), complete.get('scene'), complete.get('method'), complete.get('epochs')) != ('COMPLETE', 'collision', 'Cross', 50):
        raise ValueError('Parent must be a completed Collision Cross50 source')
    if complete.get('checkpoint_sha256') != sha(source):
        raise ValueError('Parent checkpoint/complete hash mismatch')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    if (ck.get('version'), ck.get('scene'), ck.get('method'), ck.get('family'), ck.get('epoch'), ck.get('next_epoch'), ck.get('next_batch')) != (CORE_VERSION, 'collision', 'Cross', 'JEPA', 50, 51, 0):
        raise ValueError('Expected an exact end-of-epoch50 model/optimizer/RNG checkpoint')
    if len(ck.get('history', [])) != 50 or ck.get('test_read') is not False:
        raise ValueError('Incomplete parent history or unexpected test status')
    old = ck['binding']
    body = {k: v for k, v in old.items() if k != 'sha256'}
    if hashlib.sha256(canonical(body).encode()).hexdigest() != ck['binding_sha256'] or old['sha256'] != ck['binding_sha256']:
        raise ValueError('Parent binding digest differs')
    if old['epochs'] != 50 or old['batch_size'] != 32 or old['learning_rate'] != .0003:
        raise ValueError('Unexpected parent budget/optimizer')
    for name, expected in old['files'].items():
        if sha(name) != expected:
            raise ValueError('Changed parent dependency: ' + name)
    for name in ('train.py', 'models.py'):
        candidates = [value for key, value in old['files'].items() if Path(key).name == name]
        if candidates != [sha(Path(args.runtime_dir) / name)]:
            raise ValueError('Runtime is not hash-identical to the parent: ' + name)
    if not ck['rng'].get('cuda') or args.source_cuda_index >= len(ck['rng']['cuda']):
        raise ValueError('Parent CUDA RNG index is unavailable; do not guess a state')
    if not ck.get('optimizer') or ck['optimizer']['param_groups'][0]['lr'] != .0003:
        raise ValueError('Parent optimizer state missing or unexpected')
    relation = [p for p in old['files'] if Path(p).suffix == '.json' and 'relation' in Path(p).name.lower()]
    if args.relation_index:
        relation_path = str(Path(args.relation_index).resolve())
        if relation_path not in old['files']:
            raise ValueError('Relation index is not in original source binding')
    else:
        relation = [p for p in relation if read(p).get('version') == 'cophy-relation-index-v3']
        if len(relation) != 1:
            raise ValueError('Pass the original bound --relation-index explicitly')
        relation_path = relation[0]
    data = core.FeatureData(old['features'], 'collision')
    planner = core.EpochPlanner(data, relation_path, old['seed'])
    return ck, data, planner, relation_path


class RoutePlanner:
    """Old recipient/focal/donor plan, plus independent nonfocal donor draws."""
    def __init__(self, data, original, seed):
        self.data, self.original, self.seed = data, original, seed
        self.active = np.asarray(data.arrays['train']['presence_cd'][:, :3]).any(1)
        self.seen = np.asarray(data.arrays['train']['presence_ab']).any(1)
        self.lookup, self.ids = original.lookup, original.ids
        self.index = original.index
        self.pools = {}
        self.covered = np.ones(len(self.ids), dtype=bool)
        self.reason_by_row = [[] for _ in self.ids]
        counts = {}
        for i, ident in enumerate(self.ids):
            for slot in np.flatnonzero(self.active[i]):
                record = self.index.records.get((ident, int(slot)))
                public = canonical([int(slot), record['stratum'] if record else 'MISSING'])
                stat = counts.setdefault(public, dict(active_objects=0, eligible_objects=0, reasons={}))
                stat['active_objects'] += 1
                reason = None
                if record is None:
                    reason = 'active_object_missing_audited_record'
                elif not record['recipient']:
                    reason = 'audit_not_recipient_eligible'
                elif not self.seen[i, slot]:
                    reason = 'recipient_AB_not_visually_present'
                else:
                    key = (record['group'], record['physical_key'])
                    if key not in self.pools:
                        self.pools[key] = np.asarray(sorted({self.lookup[r['id']] for r in self.index.by_relation[key]
                            if self.seen[self.lookup[r['id']], slot]}), dtype=np.int64)
                    pool = self.pools[key]
                    available = len(pool) - int(i in pool)
                    if available <= 0:
                        reason = 'no_independent_visible_correct_donor'
                if reason:
                    self.covered[i] = False; self.reason_by_row[i].append(dict(slot=int(slot), reason=reason))
                    stat['reasons'][reason] = stat['reasons'].get(reason, 0) + 1
                else:
                    stat['eligible_objects'] += 1
        self.coverage = dict(total_recipients=len(self.ids), full_object_eligible_recipients=int(self.covered.sum()),
            active_objects=int(self.active.sum()), by_slot_public_type=counts,
            exclusions=[dict(id=self.ids[i], reasons=r) for i, r in enumerate(self.reason_by_row) if r],
            policy='both routes use identical common cross cohort; own loss keeps every recipient; no fallback to own P')

    def make(self, epoch, batch_size):
        plan = self.original.make(epoch, batch_size)
        focal = plan['focal']; common = self.covered & (focal >= 0)
        external = np.full((len(self.ids), self.data.slots), -1, dtype=np.int64)
        rejected = {}
        for i in np.flatnonzero(common):
            f = int(focal[i]); donor = int(plan['correct'][i])
            if not self.active[i, f] or not self.seen[i, f] or not self.seen[donor, f]:
                common[i] = False; rejected[self.ids[i]] = 'original_focal_visual_legality'; continue
            external[i, f] = donor
            for slot in np.flatnonzero(self.active[i]):
                if slot == f:
                    continue
                row = self.index.records[(self.ids[i], int(slot))]
                pool = self.pools[(row['group'], row['physical_key'])]
                # Dedicated, row/slot/epoch seed: original sampler and model RNG untouched.
                seed = int.from_bytes(hashlib.sha256(f'{VERSION}:extra:{self.seed}:{epoch}:{self.ids[i]}:{slot}'.encode()).digest()[:8], 'little')
                rng = np.random.default_rng(seed)
                loc = int(np.searchsorted(pool, i)); contains = loc < len(pool) and pool[loc] == i
                rank = int(rng.integers(len(pool) - int(contains)))
                if contains and rank >= loc:
                    rank += 1
                external[i, slot] = pool[rank]
            for slot in np.flatnonzero(self.active[i]):
                j = int(external[i, slot]); r = self.index.records[(self.ids[i], int(slot))]
                d = self.index.records[(self.ids[j], int(slot))]
                if j == i or r['group'] != d['group'] or r['physical_key'] != d['physical_key'] or not d['donor']:
                    raise ValueError('Illegal all-object donor')
        if not common.any():
            raise ValueError('Zero common eligible cross recipients')
        order = np.concatenate(plan['batches'])
        h = hashlib.sha256()
        for array in (order, focal, plan['correct'], plan['random'], external, common):
            h.update(array.tobytes())
        plan.update(external=external, common=common, original_plan_sha256=plan['plan_sha256'],
                    plan_sha256=h.hexdigest(), extra_visual_rejections=rejected,
                    common_cross_recipients=int(common.sum()),
                    external_donor_objects=int(self.active[common].sum()),
                    additional_nonfocal_objects=int(self.active[common].sum()-common.sum()))
        return plan


def setup(args, core):
    ck, data, oldplanner, relation_path = load_parent(args, core)
    routing = RoutePlanner(data, oldplanner, ck['binding']['seed'])
    body = dict(version=VERSION, model_version=CORE_VERSION, scene='collision', parent_method='Cross',
        source_checkpoint=str(Path(args.source_checkpoint).resolve()), source_checkpoint_sha256=sha(args.source_checkpoint),
        parent_binding_sha256=ck['binding_sha256'], source_cuda_index=args.source_cuda_index,
        start_epoch=50, end_epoch=args.epochs, additional_epochs=args.epochs-50,
        parent_steps=ck['step'], parent_initialization_sha256=ck['initialization_sha256'],
        model_at_fork_sha256=state_digest(ck['model']),
        optimizer_at_fork_sha256=tree_digest(ck['optimizer']), rng_at_fork_sha256=tree_digest(ck['rng']),
        parent_dependencies=ck['binding']['files'], runtime_dir=str(Path(args.runtime_dir).resolve()),
        implementation_sha256=sha(__file__), relation_index=relation_path,
        protocol_sha256=sha(args.protocol) if args.protocol else None,
        batch_size=ck['binding']['batch_size'], microbatch=ck['microbatch'],
        learning_rate=ck['binding']['learning_rate'], weight_decay=1e-4, clip_norm=1.,
        model_config=ck['model_config'], query_frames=3, history_frames=15,
        loss='original self + SIGReg + lambda_cross * focal Cross; no Align because parent method is Cross',
        cross_normalization='mean over same valid focal future tokens/dimensions, never divide by object count',
        extra_sampler='independent deterministic row/slot/epoch RNG, focal donor and recipient order retained',
        paired_compute='both routes encode focal and additional donor sets; unused extra vectors have zero contribution in focal route; SIGReg recipient-only',
        current_input='recipient CD[:3] only', donor_input='donor AB only',
        source_selection='fixed epoch100; epoch75 diagnostic only', test_read=False, coordinate_labels_read=False)
    body['sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return ck, data, routing, body


def tree_digest(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x: add(y)
        else:
            h.update(repr(x).encode())
    add(value); return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()


def restore_rng(record, device, parent_cuda_index):
    random.setstate(record['python']); np.random.set_state(record['numpy']); torch.set_rng_state(record['torch'])
    if device.type == 'cuda':
        if not record['cuda'] or parent_cuda_index >= len(record['cuda']):
            raise ValueError('CUDA RNG mapping unavailable')
        torch.cuda.set_rng_state(record['cuda'][parent_cuda_index], device=device)


def branch_rng(device):
    # Do not initialize contexts on other GPUs just to save irrelevant RNGs.
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(device)] if device.type=='cuda' else [])


def save_plan(path, plan):
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, order=np.concatenate(plan['batches']), focal=plan['focal'], correct=plan['correct'],
             random=plan['random'], external=plan['external'], common=plan['common'])
    os.replace(tmp, path)


def prepare(args, core):
    parent, data, planner, binding = setup(args, core)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if out == Path(args.source_checkpoint).resolve().parent or out in Path(args.source_checkpoint).resolve().parents:
        raise ValueError('Output may not be a parent/source run directory')
    immutable(out/'binding.json', binding)
    immutable(out/'coverage_static.json',dict(version=VERSION,binding_sha256=binding['sha256'],**planner.coverage))
    plan_dir = out/'plans'; plan_dir.mkdir(exist_ok=True)
    epochs = []
    for epoch in range(51, args.epochs+1):
        plan = planner.make(epoch, binding['batch_size'])
        path = plan_dir/f'epoch_{epoch:03d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for name in ('focal','correct','random','external','common'):
                    if not np.array_equal(z[name], plan[name]): raise ValueError('Changed saved source plan')
                if not np.array_equal(z['order'], np.concatenate(plan['batches'])): raise ValueError('Changed recipient order')
        else:
            save_plan(path, plan)
        epochs.append(dict(epoch=epoch, original_plan_sha256=plan['original_plan_sha256'],
            plan_sha256=plan['plan_sha256'], file_sha256=sha(path), recipients=len(data.ids['train']),
            original_paired=plan['paired'], common_cross_recipients=plan['common_cross_recipients'],
            additional_nonfocal_objects=plan['additional_nonfocal_objects'],
            original_randomization_skipped=plan['randomization_skipped'],
            extra_visual_rejections=plan['extra_visual_rejections']))
    coverage = dict(status='COMPLETE', version=VERSION, binding_sha256=binding['sha256'],
                    static=planner.coverage, epochs=epochs, test_read=False)
    immutable(out/'coverage.json', coverage)
    marker = dict(status='COMPLETE', version=VERSION, binding_sha256=binding['sha256'], coverage_sha256=sha(out/'coverage.json'),
                  routes=list(ROUTES), plans=len(epochs), source_model_unchanged=True, optimizer_steps=0, test_read=False)
    immutable(out/'prepared.json', marker); emit('PREPARED', **marker)


def checked_setup(args, core):
    parent, data, planner, binding = setup(args, core)
    out = Path(args.out)
    if read(out/'binding.json') != binding or read(out/'prepared.json')['binding_sha256'] != binding['sha256']:
        raise ValueError('Prepare exact source continuation binding first')
    if read(out/'prepared.json')['coverage_sha256'] != sha(out/'coverage.json'):
        raise ValueError('Changed coverage receipt')
    return parent, data, planner, binding


def checked_plan(args, planner, binding, epoch):
    plan = planner.make(epoch, binding['batch_size'])
    record = read(Path(args.out)/'coverage.json')['epochs'][epoch-51]
    if record['plan_sha256'] != plan['plan_sha256'] or record['file_sha256'] != sha(Path(args.out)/'plans'/f'epoch_{epoch:03d}.npz'):
        raise ValueError('Paired routing plan differs from preparation')
    return plan


def focal_plan(plan):
    value = dict(plan); value['focal'] = np.where(plan['common'], plan['focal'], -1)
    return value


def objective(core, model, data, plan, indices, route, microbatch):
    if route not in ROUTES: raise ValueError('Unknown source route')
    device = next(model.parameters()).device
    ab, am, current, cm = data.context('train', indices, device)
    cd, mask_cd = data.target('train', indices, device)
    active = cm.any(1); target_mask = mask_cd[:, 3:] & active[:, None]
    own = core.encode_micro(model, ab, am, microbatch)
    current_t = model.encode_current(current, cm); target = model.target(cd)
    prediction = core.predict_micro(model, own, current_t, active, data.frames-3, microbatch)
    self_loss = core.masked_mse(prediction, target, target_mask)
    reg = model.regularization(ab, am, cd, mask_cd)
    cross = own.sum()*0; extra_zero = own.sum()*0
    rows = torch.as_tensor(np.flatnonzero(plan['common'][indices]), device=device)
    mixed = None; extra = 0
    if len(rows):
        host_rows = np.flatnonzero(plan['common'][indices]); original_rows = indices[host_rows]
        focal = torch.as_tensor(plan['focal'][original_rows], device=device)
        # Preserve the original focal donor call, then add only nonfocal AB encodes.
        da, dm = data.history('train', plan['correct'][original_rows], device)
        donor = core.encode_micro(model, da, dm, microbatch)
        focal_p = donor[torch.arange(len(rows), device=device), focal]
        mixed = core.replace_focal_p(own, rows, focal, focal_p)
        additions, local_rows, local_slots = [], [], []
        for local, original in enumerate(original_rows):
            for slot in np.flatnonzero(plan['external'][original] >= 0):
                if slot != plan['focal'][original]:
                    additions.append(int(plan['external'][original, slot])); local_rows.append(local); local_slots.append(int(slot))
        if additions:
            xa, xm = data.history('train', np.asarray(additions), device)
            xp = core.encode_micro(model, xa, xm, microbatch)
            rr = torch.as_tensor(local_rows, device=device); ss = torch.as_tensor(local_slots, device=device)
            additional_p = xp[torch.arange(len(additions), device=device), ss]
            if route == 'all':
                mixed = mixed.clone(); mixed[rr, ss] = additional_p
            # Equal additional encoder graph in both arms, no extra supervision.
            extra_zero = additional_p.sum()*0
            extra = len(additions)
        # Every active P must have a legal external donor; inactive nodes are masked.
        legal = torch.as_tensor(plan['external'][original_rows] >= 0, device=device)
        if not torch.equal(legal, active[rows]): raise ValueError('An active P would silently remain recipient-owned')
        prediction_cross = core.predict_micro(model, mixed, current_t[rows], active[rows], data.frames-3, microbatch)
        cross = core.masked_mse(prediction_cross[torch.arange(len(rows), device=device), :, focal],
            target[rows, :, focal], target_mask[rows, :, focal])
    loss = self_loss + model.config.sigreg_weight*reg + model.config.lambda_cross*cross + extra_zero
    metrics = dict(loss=float(loss.detach()), self_loss=float(self_loss.detach()), latent_mse=float(self_loss.detach()),
        cross=float(cross.detach()), align=0., sigreg=float(reg.detach()),
        p_std_all=float(own.detach().float().flatten(0,1).std(0).mean()),
        target_std=float(target.detach().float().flatten(0,2).std(0).mean()), paired=len(rows),
        additional_donor_encodes=extra, additional_donor_used=extra if route=='all' else 0)
    return loss, metrics, dict(own=own, mixed=mixed, rows=rows,
        terms={'self':self_loss, 'cross':model.config.lambda_cross*cross, 'align':own.sum()*0})


def smoke(args, core):
    parent, data, planner, binding = checked_setup(args, core)
    device = torch.device(args.device)
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    model = core.make_model(config=parent['model_config']).to(device); model.load_state_dict(parent['model']); model.train()
    plan = checked_plan(args, planner, binding, 51)
    indices = next(x for x in plan['batches'] if plan['common'][x].any())
    output = {}
    for name in ('original_common', 'focal', 'all'):
        restore_rng(parent['rng'], device, args.source_cuda_index); model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            if name == 'original_common':
                loss, metrics, detail = core.batch_objective(model, data, focal_plan(plan), indices, 'Cross', parent['microbatch'])
            else:
                loss, metrics, detail = objective(core, model, data, plan, indices, name, parent['microbatch'])
        if not torch.isfinite(loss): raise FloatingPointError('Smoke loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        output[name] = dict(metrics=metrics, gradient_norm=float(norm))
        del loss, detail
    for name in ('loss','self_loss','cross','sigreg'):
        if output['original_common']['metrics'][name] != output['focal']['metrics'][name]:
            raise ValueError('Focal continuation differs from original core objective')
    for name in ('self_loss','sigreg'):
        if output['original_common']['metrics'][name] != output['all']['metrics'][name]:
            raise ValueError('All route changed self/SIGReg path')
    if state_digest(model.state_dict()) != binding['model_at_fork_sha256']:
        raise ValueError('Smoke changed parent model state')
    result = dict(status='PASS', version=VERSION, binding_sha256=binding['sha256'], measurements=output,
        focal_loss_matches_parent_exactly=True, self_and_sigreg_equal_between_routes=True,
        original_plan_sha256=plan['original_plan_sha256'], shared_plan_sha256=plan['plan_sha256'],
        common_cross_count=plan['common_cross_recipients'], source_model_unchanged=True,
        optimizer_steps=0, coordinate_labels_read=False, test_read=False)
    write(Path(args.out)/'smoke.json', result); emit('SMOKE_PASS', **result)


def train(args, core):
    if args.route not in ROUTES: raise ValueError('train requires --route focal or all')
    import fcntl
    out = Path(args.out); folder = out/'runs'/args.route; folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'owner.lock', 'a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Route already running') from e
        parent, data, planner, binding = checked_setup(args, core)
        if read(out/'smoke.json').get('status') != 'PASS' or read(out/'smoke.json')['binding_sha256'] != binding['sha256']:
            raise ValueError('Matching real smoke required')
        if (folder/'complete.json').exists():
            done = read(folder/'complete.json')
            if done.get('binding_sha256') != binding['sha256'] or done.get('route') != args.route: raise ValueError('Conflicting completed route')
            emit('ALREADY_COMPLETE', **done); return
        device = torch.device(args.device)
        if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
        model = core.make_model(config=parent['model_config']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=binding['learning_rate'], weight_decay=1e-4)
        latest = folder/'latest.pt'; old = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else parent
        if latest.exists() and (old.get('route_binding_sha256') != binding['sha256'] or old.get('route') != args.route):
            raise ValueError('Different route continuation checkpoint')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        source_rng_index = old.get('rng_device_index', args.source_cuda_index)
        restore_rng(old['rng'], device, source_rng_index)
        epoch, next_batch, step = old['next_epoch'], old['next_batch'], old['step']
        history = list(old.get('history', [])); running = dict(old.get('running', {})); microbatch = old['microbatch']
        prior_seconds = old.get('extension_seconds', 0.); started = time.time()
        if not latest.exists():
            if state_digest(model.state_dict()) != binding['model_at_fork_sha256'] or tree_digest(optimizer.state_dict()) != binding['optimizer_at_fork_sha256']:
                raise ValueError('Model/optimizer did not restore identically at fork')
            write(folder/'fork.json', dict(status='COMPLETE', route=args.route, binding_sha256=binding['sha256'],
                model_sha256=binding['model_at_fork_sha256'], optimizer_sha256=binding['optimizer_at_fork_sha256'],
                rng_sha256=binding['rng_at_fork_sha256'], source_cuda_index=args.source_cuda_index,
                branch_device=str(device), epoch=50, step=step, test_read=False))
        def record(ne, nb):
            return dict(version=CORE_VERSION, experiment_version=VERSION, route=args.route,
                model=model.state_dict(), model_config=model.artifact_config(), optimizer=optimizer.state_dict(),
                rng=branch_rng(device), rng_device_index=0 if device.type=='cuda' else None,
                method='Cross', family='JEPA', scene='collision', epoch=ne-1 if nb==0 else ne,
                next_epoch=ne, next_batch=nb, step=step, history=history, running=running, microbatch=microbatch,
                binding=binding, binding_sha256=binding['sha256'], route_binding_sha256=binding['sha256'],
                parent_checkpoint=binding['source_checkpoint'], parent_checkpoint_sha256=binding['source_checkpoint_sha256'],
                initialization_sha256=parent['initialization_sha256'],
                extension_seconds=prior_seconds+time.time()-started, test_read=False)
        write(folder/'worker.json', dict(status='RUNNING', pid=os.getpid(), host=socket.gethostname(), device=str(device),
            route=args.route, binding_sha256=binding['sha256'], started_at=started))
        try:
            while epoch <= args.epochs:
                plan = checked_plan(args, planner, binding, epoch)
                if next_batch == 0:
                    running = dict(weighted={}, examples=0, steps=0, started_at=time.time(), plan_sha256=plan['plan_sha256'])
                elif running.get('plan_sha256') != plan['plan_sha256']:
                    raise ValueError('Resumed epoch plan mismatch')
                model.train()
                for batch_number in range(next_batch, len(plan['batches'])):
                    indices = plan['batches'][batch_number]; optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        loss, metrics, detail = objective(core, model, data, plan, indices, args.route, microbatch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite source routing objective')
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); del loss, detail
                    step += 1; next_batch = batch_number+1
                    running['examples'] += len(indices); running['steps'] += 1
                    for key, value in metrics.items(): running['weighted'][key] = running['weighted'].get(key,0.)+value*len(indices)
                    if step%50==0 or next_batch==len(plan['batches']):
                        core.save(latest, record(epoch,next_batch))
                        write(folder/'progress.json', dict(status='RUNNING', route=args.route, epoch=epoch, batch=next_batch,
                            step=step, additional_steps=step-parent['step'], history=history, latest=metrics,
                            grad_norm=float(norm), plan_sha256=plan['plan_sha256'], test_read=False))
                        emit('PROGRESS', route=args.route, epoch=epoch, step=step, **metrics)
                    if args.max_additional_steps and step-parent['step'] >= args.max_additional_steps:
                        core.save(latest,record(epoch,next_batch))
                        write(folder/'paused.json',dict(status='PAUSED',step=step,next_epoch=epoch,next_batch=next_batch,
                            binding_sha256=binding['sha256'],test_read=False))
                        emit('PAUSED',route=args.route,step=step); return
                val=core.evaluate(model,data,min(microbatch,32))
                if running['examples'] != len(data.ids['train']): raise ValueError('Incomplete own-loss recipient exposure')
                row=dict(epoch=epoch,mse=val['mse'],train={k:v/running['examples'] for k,v in running['weighted'].items()},
                    seconds=time.time()-running['started_at'],plan_sha256=plan['plan_sha256'],original_plan_sha256=plan['original_plan_sha256'],
                    query_exposures=running['examples'],paired=plan['common_cross_recipients'],original_paired=plan['paired'],
                    additional_nonfocal_encoded=plan['additional_nonfocal_objects'],
                    additional_nonfocal_used=plan['additional_nonfocal_objects'] if args.route=='all' else 0,
                    route=args.route,metric='live latent diagnostic only, not selection')
                history.append(row); completed=epoch;epoch+=1;next_batch=0;running={}
                core.save(latest,record(epoch,0))
                if completed in (75,100):
                    core.save(folder/f'checkpoint_{completed}.pt',record(epoch,0))
                    write(folder/f'validation_{completed}.json',dict(epoch=completed,**val))
                write(folder/'progress.json',dict(status='RUNNING',route=args.route,epoch=completed,step=step,history=history,test_read=False))
                emit('EPOCH_COMPLETE',**row)
            checkpoint=folder/f'checkpoint_{args.epochs}.pt'
            result=dict(status='COMPLETE',version=VERSION,model_version=CORE_VERSION,route=args.route,method='Cross',
                scene='collision',epochs=args.epochs,additional_epochs=args.epochs-50,steps=step,additional_steps=step-parent['step'],
                binding_sha256=binding['sha256'],source_checkpoint=binding['source_checkpoint'],
                source_checkpoint_sha256=binding['source_checkpoint_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                selected_epoch=args.epochs,selection='fixed common source100 budget; epoch75 diagnostic only',
                original_50_history_preserved=history[:50]==parent['history'],
                extension_seconds=prior_seconds+time.time()-started,test_read=False,coordinate_labels_read=False)
            write(folder/'complete.json',result);write(folder/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('COMPLETE',**result)
        except Exception as error:
            failure=dict(status='FAILED',route=args.route,error=repr(error),traceback=traceback.format_exc(),
                epoch=epoch,next_batch=next_batch,step=step,binding_sha256=binding['sha256'],test_read=False)
            write(folder/'failure.json',failure);emit('FAILED',**failure);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--runtime-dir',required=True);p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--source-cuda-index',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--relation-index');p.add_argument('--protocol')
    p.add_argument('--route',choices=ROUTES);p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,choices=(100,),default=100);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-additional-steps',type=int,default=0)
    args=p.parse_args()
    if args.source_cuda_index<0 or args.threads<1 or args.max_additional_steps<0: p.error('Invalid numeric argument')
    core=runtime(args);globals()[args.command](args,core)


if __name__=='__main__':
    main()
