"""Continue Collision JEPA Base50 to a fixed source100 budget.

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

VERSION = 'collision-base-source-continuation-v6.5-1'
CORE_VERSION = 'cophy-latent-v6.2-sig02'


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
    spec = importlib.util.spec_from_file_location('_collision_base100_parent_train', root / 'train.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    if core.VERSION != CORE_VERSION:
        raise ValueError('Expected the original sig02 runtime')
    torch.set_num_threads(args.threads)
    return core


def load_parent(args, core):
    source = Path(args.source_checkpoint).resolve()
    complete = read(source.parent / 'complete.json')
    if (complete.get('status'), complete.get('scene'), complete.get('method'), complete.get('epochs')) != ('COMPLETE', 'collision', 'Base', 50):
        raise ValueError('Parent must be a completed Collision Base50 source')
    if complete.get('checkpoint_sha256') != sha(source):
        raise ValueError('Parent checkpoint/complete hash mismatch')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    if (ck.get('version'), ck.get('scene'), ck.get('method'), ck.get('family'), ck.get('epoch'), ck.get('next_epoch'), ck.get('next_batch')) != (CORE_VERSION, 'collision', 'Base', 'JEPA', 50, 51, 0):
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


def setup(args, core):
    ck, data, planner, relation_path = load_parent(args, core)
    if ck['step'] != 21900 or len(data.ids['train']) != 14000:
        raise ValueError('Expected original Collision50 budget: 14,000 queries and 21,900 steps')
    if ck['model_config']['family'] != 'JEPA' or ck['model_config']['sigreg_weight'] != .2:
        raise ValueError('Expected the original JEPA SIG02 model configuration')
    body = dict(version=VERSION, model_version=CORE_VERSION, scene='collision', method='Base',
        source_checkpoint=str(Path(args.source_checkpoint).resolve()), source_checkpoint_sha256=sha(args.source_checkpoint),
        parent_binding_sha256=ck['binding_sha256'], source_cuda_index=args.source_cuda_index,
        start_epoch=50, end_epoch=100, additional_epochs=50, expected_final_steps=43800,
        parent_steps=ck['step'], parent_initialization_sha256=ck['initialization_sha256'],
        model_at_fork_sha256=state_digest(ck['model']),
        optimizer_at_fork_sha256=tree_digest(ck['optimizer']), rng_at_fork_sha256=tree_digest(ck['rng']),
        parent_dependencies=ck['binding']['files'], runtime_dir=str(Path(args.runtime_dir).resolve()),
        implementation_sha256=sha(__file__), relation_index=relation_path,
        protocol_sha256=sha(args.protocol) if args.protocol else None,
        batch_size=ck['binding']['batch_size'], microbatch=ck['microbatch'],
        learning_rate=ck['binding']['learning_rate'], weight_decay=1e-4, clip_norm=1.,
        model_config=ck['model_config'], query_frames=3, history_frames=15,
        loss='original core.batch_objective(method=Base): self + SIGReg only; Cross=Align=0',
        donor_policy='no donor observations encoded; no donor training gradients',
        sampler='original EpochPlanner recipient ordering/public groups; pairing metadata is unused by Base loss',
        current_input='recipient CD[:3] only', history_input='recipient AB only',
        source_selection='fixed epoch100; epoch75 diagnostic only', test_read=False, coordinate_labels_read=False)
    body['sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return ck, data, planner, body


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
    np.savez(tmp, order=np.concatenate(plan['batches']))
    os.replace(tmp, path)


def prepare(args, core):
    parent, data, planner, binding = setup(args, core)
    out = Path(args.out).resolve()
    if out == Path(args.source_checkpoint).resolve().parent or out in Path(args.source_checkpoint).resolve().parents:
        raise ValueError('Output may not be a parent/source run directory')
    out.mkdir(parents=True, exist_ok=True)
    immutable(out/'binding.json', binding)
    plan_dir = out/'plans'; plan_dir.mkdir(exist_ok=True)
    epochs = []
    for epoch in range(51, 101):
        plan = planner.make(epoch, binding['batch_size'])
        order = np.concatenate(plan['batches'])
        if not np.array_equal(np.sort(order), np.arange(len(data.ids['train']))) or len(plan['batches']) != 438:
            raise ValueError('Continuation must expose every original query exactly once per epoch in 438 steps')
        path = plan_dir/f'epoch_{epoch:03d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z['order'], order): raise ValueError('Changed recipient order')
        else:
            save_plan(path, plan)
        epochs.append(dict(epoch=epoch, plan_sha256=plan['plan_sha256'], file_sha256=sha(path),
            recipients=len(order), optimizer_steps=len(plan['batches']), donor_encodes=0))
    plans = dict(status='COMPLETE', version=VERSION, binding_sha256=binding['sha256'], epochs=epochs, test_read=False)
    immutable(out/'plans.json', plans)
    marker = dict(status='COMPLETE', version=VERSION, binding_sha256=binding['sha256'], plans_sha256=sha(out/'plans.json'),
                  plans=len(epochs), method='Base', source_model_unchanged=True, optimizer_steps=0, test_read=False)
    immutable(out/'prepared.json', marker); emit('PREPARED', **marker)


def checked_setup(args, core):
    parent, data, planner, binding = setup(args, core)
    out = Path(args.out)
    if read(out/'binding.json') != binding or read(out/'prepared.json')['binding_sha256'] != binding['sha256']:
        raise ValueError('Prepare exact Base continuation binding first')
    if read(out/'prepared.json')['plans_sha256'] != sha(out/'plans.json'):
        raise ValueError('Changed plan receipt')
    return parent, data, planner, binding


def checked_plan(args, planner, binding, epoch):
    plan = planner.make(epoch, binding['batch_size'])
    record = read(Path(args.out)/'plans.json')['epochs'][epoch-51]
    if record['plan_sha256'] != plan['plan_sha256'] or record['file_sha256'] != sha(Path(args.out)/'plans'/f'epoch_{epoch:03d}.npz'):
        raise ValueError('Original recipient plan differs from preparation')
    return plan


def base_objective(core, model, data, plan, indices, microbatch):
    # Do not reimplement or modify the source objective. The Base branch never
    # reads donor observations, even though original plan metadata is retained.
    loss, metrics, detail = core.batch_objective(model, data, plan, indices, 'Base', microbatch)
    if (metrics['cross'] != 0. or metrics['align'] != 0. or metrics['paired'] != 0
            or detail['donor_p'] is not None or detail['mixed'] is not None or len(detail['rows'])):
        raise ValueError('Base unexpectedly activated a relation/donor path')
    return loss, metrics, detail


def smoke(args, core):
    parent, data, planner, binding = checked_setup(args, core)
    device = torch.device(args.device)
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    model = core.make_model(config=parent['model_config']).to(device); model.load_state_dict(parent['model']); model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=binding['learning_rate'], weight_decay=1e-4)
    optimizer.load_state_dict(parent['optimizer'])
    if tree_digest(optimizer.state_dict()) != binding['optimizer_at_fork_sha256']:
        raise ValueError('Smoke optimizer did not restore identically')
    plan = checked_plan(args, planner, binding, 51)
    indices = plan['batches'][0]
    output = {}; gradients = {}
    for name in ('original_base', 'continuation_base'):
        restore_rng(parent['rng'], device, args.source_cuda_index); model.zero_grad(set_to_none=True)
        calls = []
        original_history = data.history
        def recipient_history(split, rows, dev):
            if split != 'train' or not np.array_equal(rows, indices):
                raise ValueError('Unexpected nonrecipient history in Base smoke')
            calls.append(len(rows)); return original_history(split, rows, dev)
        data.history = recipient_history
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                if name == 'original_base':
                    loss, metrics, detail = core.batch_objective(model, data, plan, indices, 'Base', parent['microbatch'])
                else:
                    loss, metrics, detail = base_objective(core, model, data, plan, indices, parent['microbatch'])
        finally:
            data.history = original_history
        if not torch.isfinite(loss): raise FloatingPointError('Smoke loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        gradients[name] = state_digest({n: p.grad for n,p in model.named_parameters() if p.grad is not None})
        output[name] = dict(metrics=metrics, gradient_norm=float(norm), history_calls=calls)
        del loss, detail
    if output['original_base'] != output['continuation_base'] or gradients['original_base'] != gradients['continuation_base']:
        raise ValueError('Base continuation differs from exact original objective/gradient')
    if state_digest(model.state_dict()) != binding['model_at_fork_sha256']:
        raise ValueError('Smoke changed parent model state')
    if tree_digest(optimizer.state_dict()) != binding['optimizer_at_fork_sha256']:
        raise ValueError('Smoke changed optimizer state')
    result = dict(status='PASS', version=VERSION, binding_sha256=binding['sha256'], measurements=output,
        exact_original_loss_and_gradient=True, gradient_sha256=gradients['original_base'],
        plan_sha256=plan['plan_sha256'], source_model_unchanged=True, optimizer_unchanged=True,
        optimizer_steps=0, donor_encodes=0, coordinate_labels_read=False, test_read=False)
    write(Path(args.out)/'smoke.json', result); emit('SMOKE_PASS', **result)


def train(args, core):
    import fcntl
    out = Path(args.out); folder = out; folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'owner.lock', 'a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Base continuation already running') from e
        parent, data, planner, binding = checked_setup(args, core)
        if read(out/'smoke.json').get('status') != 'PASS' or read(out/'smoke.json')['binding_sha256'] != binding['sha256']:
            raise ValueError('Matching real smoke required')
        if (folder/'complete.json').exists():
            done = read(folder/'complete.json')
            if done.get('binding_sha256') != binding['sha256'] or done.get('method') != 'Base': raise ValueError('Conflicting completed Base continuation')
            emit('ALREADY_COMPLETE', **done); return
        device = torch.device(args.device)
        if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
        model = core.make_model(config=parent['model_config']).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=binding['learning_rate'], weight_decay=1e-4)
        latest = folder/'latest.pt'; old = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else parent
        if latest.exists() and (old.get('continuation_binding_sha256') != binding['sha256'] or old.get('method') != 'Base'):
            raise ValueError('Different Base continuation checkpoint')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        source_rng_index = old.get('rng_device_index', args.source_cuda_index)
        restore_rng(old['rng'], device, source_rng_index)
        epoch, next_batch, step = old['next_epoch'], old['next_batch'], old['step']
        history = list(old.get('history', [])); running = dict(old.get('running', {})); microbatch = old['microbatch']
        prior_seconds = old.get('extension_seconds', 0.); started = time.time()
        if not latest.exists():
            if state_digest(model.state_dict()) != binding['model_at_fork_sha256'] or tree_digest(optimizer.state_dict()) != binding['optimizer_at_fork_sha256']:
                raise ValueError('Model/optimizer did not restore identically at fork')
            write(folder/'fork.json', dict(status='COMPLETE', method='Base', binding_sha256=binding['sha256'],
                model_sha256=binding['model_at_fork_sha256'], optimizer_sha256=binding['optimizer_at_fork_sha256'],
                rng_sha256=binding['rng_at_fork_sha256'], source_cuda_index=args.source_cuda_index,
                branch_device=str(device), epoch=50, step=step, test_read=False))
        def record(ne, nb):
            return dict(version=CORE_VERSION, experiment_version=VERSION, method='Base',
                model=model.state_dict(), model_config=model.artifact_config(), optimizer=optimizer.state_dict(),
                rng=branch_rng(device), rng_device_index=0 if device.type=='cuda' else None,
                family='JEPA', scene='collision', epoch=ne-1 if nb==0 else ne,
                next_epoch=ne, next_batch=nb, step=step, history=history, running=running, microbatch=microbatch,
                binding=binding, binding_sha256=binding['sha256'], continuation_binding_sha256=binding['sha256'],
                parent_checkpoint=binding['source_checkpoint'], parent_checkpoint_sha256=binding['source_checkpoint_sha256'],
                initialization_sha256=parent['initialization_sha256'],
                extension_seconds=prior_seconds+time.time()-started, test_read=False)
        write(folder/'worker.json', dict(status='RUNNING', pid=os.getpid(), host=socket.gethostname(), device=str(device),
            method='Base', binding_sha256=binding['sha256'], started_at=started))
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
                        loss, metrics, detail = base_objective(core, model, data, plan, indices, microbatch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Base continuation objective')
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); del loss, detail
                    step += 1; next_batch = batch_number+1
                    running['examples'] += len(indices); running['steps'] += 1
                    for key, value in metrics.items(): running['weighted'][key] = running['weighted'].get(key,0.)+value*len(indices)
                    if step%50==0 or next_batch==len(plan['batches']):
                        core.save(latest, record(epoch,next_batch))
                        write(folder/'progress.json', dict(status='RUNNING', method='Base', epoch=epoch, batch=next_batch,
                            step=step, additional_steps=step-parent['step'], history=history, latest=metrics,
                            grad_norm=float(norm), plan_sha256=plan['plan_sha256'], test_read=False))
                        emit('PROGRESS', method='Base', epoch=epoch, step=step, **metrics)
                    if args.max_additional_steps and step-parent['step'] >= args.max_additional_steps:
                        core.save(latest,record(epoch,next_batch))
                        write(folder/'paused.json',dict(status='PAUSED',step=step,next_epoch=epoch,next_batch=next_batch,
                            binding_sha256=binding['sha256'],test_read=False))
                        emit('PAUSED',method='Base',step=step); return
                val=core.evaluate(model,data,min(microbatch,32))
                if running['examples'] != len(data.ids['train']): raise ValueError('Incomplete own-loss recipient exposure')
                row=dict(epoch=epoch,mse=val['mse'],train={k:v/running['examples'] for k,v in running['weighted'].items()},
                    seconds=time.time()-running['started_at'],plan_sha256=plan['plan_sha256'],
                    query_exposures=running['examples'],paired=0,donor_encodes=0,
                    method='Base',metric='live latent diagnostic only, not selection')
                history.append(row); completed=epoch;epoch+=1;next_batch=0;running={}
                core.save(latest,record(epoch,0))
                if completed in (75,100):
                    core.save(folder/f'checkpoint_{completed}.pt',record(epoch,0))
                    write(folder/f'validation_{completed}.json',dict(epoch=completed,**val))
                write(folder/'progress.json',dict(status='RUNNING',method='Base',epoch=completed,step=step,history=history,test_read=False))
                emit('EPOCH_COMPLETE',**row)
            checkpoint=folder/f'checkpoint_{args.epochs}.pt'
            if step != binding['expected_final_steps'] or len(history) != 100:
                raise ValueError('Final Base continuation budget differs from source100')
            result=dict(status='COMPLETE',version=VERSION,model_version=CORE_VERSION,method='Base',
                scene='collision',epochs=args.epochs,additional_epochs=args.epochs-50,steps=step,additional_steps=step-parent['step'],
                binding_sha256=binding['sha256'],source_checkpoint=binding['source_checkpoint'],
                source_checkpoint_sha256=binding['source_checkpoint_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                selected_epoch=args.epochs,selection='fixed common source100 budget; epoch75 diagnostic only',
                original_50_history_preserved=history[:50]==parent['history'],
                extension_seconds=prior_seconds+time.time()-started,test_read=False,coordinate_labels_read=False)
            write(folder/'complete.json',result);write(folder/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('COMPLETE',**result)
        except Exception as error:
            failure=dict(status='FAILED',method='Base',error=repr(error),traceback=traceback.format_exc(),
                epoch=epoch,next_batch=next_batch,step=step,binding_sha256=binding['sha256'],test_read=False)
            write(folder/'failure.json',failure);emit('FAILED',**failure);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--runtime-dir',required=True);p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--source-cuda-index',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--relation-index');p.add_argument('--protocol')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,choices=(100,),default=100);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-additional-steps',type=int,default=0)
    args=p.parse_args()
    if args.source_cuda_index<0 or args.threads<1 or args.max_additional_steps<0: p.error('Invalid numeric argument')
    core=runtime(args);globals()[args.command](args,core)


if __name__=='__main__':
    main()
