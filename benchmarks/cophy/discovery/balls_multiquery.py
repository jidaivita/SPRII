"""Balls S5 focal-memory reuse across three independent queries.

Post-training from a common Native checkpoint. Physical metadata enter only
the sampler and diagnostics; prediction receives observations and memory.
"""
import argparse
import time
import resource
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import balls_multiquery_components as core
from collision_multiquery_sampler import MultiQuerySampler

VERSION = 'balls-multiquery-v5.0-1'
METHODS = ('Native-MQ', 'A-MQ', 'Random-MQ', 'A-MQ-Reg')
BATCH_GROUPS = 10
QUERY_GROUP = 3
SUPPORTS = 5


def prepare(out, base):
    out, base = Path(out), Path(base)
    out.mkdir(parents=True, exist_ok=True)
    legacy = core.prepare_binding(base)
    data = core.Data(base, 'cpu')
    plan = MultiQuerySampler(data, SUPPORTS, QUERY_GROUP).make_epoch(1)
    valplan = data.plan('val', 0, 1, supports=SUPPORTS)
    val_path = out / 'validation_supports_s5.npz'
    if val_path.exists():
        with np.load(val_path, allow_pickle=False) as saved:
            if not np.array_equal(saved['plan'], valplan):
                raise ValueError('Existing S5 plan differs')
    else:
        core.save_npz(val_path, plan=valplan, query_ids=np.asarray(data.rows['val']['ids']))
    files = {**legacy['file_sha256'], str(Path(__file__)): core.digest(__file__),
        str(val_path): core.digest(val_path),
        str(Path(__file__).with_name('collision_multiquery_sampler.py')):
            core.digest(Path(__file__).with_name('collision_multiquery_sampler.py'))}
    binding = {'version': VERSION, 'base': str(base), 'legacy_binding': legacy,
        'warm_start': legacy['head']['path'], 'warm_start_epoch': legacy['head']['epoch'],
        'validation_plan': str(val_path), 'test_read': False,
        'base_manifest_sha256': core.digest(base / 'manifest.json'), 'file_sha256': files}
    target = out / 'binding.json'
    if target.exists() and core.read(target) != binding:
        raise ValueError('Output already bound to different inputs or code')
    core.write(target, binding)
    core.write(out / 'plan_epoch1_summary.json',
               {**plan['summary'], 'plan_sha256': plan['plan_sha256']})
    validate_plan_ids(binding, data)
    core.emit('prepared', query_exposures=len(plan['query_indices']),
              groups=plan['summary']['groups'], padding=plan['summary']['padding_occurrences'])


def validate_plan_ids(binding, data):
    with np.load(binding['validation_plan'], allow_pickle=False) as saved:
        ids, plan = saved['query_ids'].tolist(), saved['plan'].copy()
    if ids != data.rows['val']['ids'] or plan.shape != (len(ids), core.K, 1, SUPPORTS):
        raise ValueError('S5 validation plan differs from the existing query cohort')
    return plan


def setup(out, base, device):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    binding = core.read(Path(out) / 'binding.json')
    if binding['version'] != VERSION or binding['base_manifest_sha256'] != core.digest(Path(base) / 'manifest.json'):
        raise ValueError('Binding version or manifest changed')
    for path, expected in binding['file_sha256'].items():
        if core.digest(path) != expected:
            raise ValueError('Bound dependency changed: ' + path)
    data = core.FineTuneData(base, device)
    model = core.FineTuneModel(binding['legacy_binding']).to(device)
    return binding, data, model, validate_plan_ids(binding, data)


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-5}], weight_decay=1e-4)


def forward_batch(model, data, plan, group_start, group_end, method, device, epoch):
    lo, hi = group_start * QUERY_GROUP, group_end * QUERY_GROUP
    indices, focal = plan['query_indices'][lo:hi], plan['focal'][lo:hi]
    supports = plan['independent'][lo:hi].copy()
    use_shared = method != 'Native-MQ'
    if use_shared:
        pool = 'shared_wrong' if method == 'Random-MQ' else 'shared_correct'
        shared_ids = np.repeat(plan[pool][group_start:group_end], QUERY_GROUP, axis=0)
        supports[np.arange(len(indices)), focal, 1] = shared_ids
    q, det, mask, target = data.batch('train', indices, device)
    memory = model.memory(data, 'train', supports)
    rows = torch.arange(len(indices), device=device)
    focal_tensor = torch.as_tensor(focal, device=device)
    shared = None
    second = memory[:, 1]
    if use_shared:
        anchors = torch.arange(0, len(indices), QUERY_GROUP, device=device)
        shared = memory[anchors, 1, focal_tensor[anchors]]
        expanded = shared.repeat_interleave(QUERY_GROUP, dim=0)
        selector = F.one_hot(focal_tensor, core.K).to(memory.dtype)[..., None]
        # All three predictions use one graph node, not three detached copies.
        second = second * (1 - selector) + expanded[:, None, :] * selector
    pred1 = model.head.forward_memory(q, det, mask, memory[:, 0])
    pred2 = model.head.forward_memory(q, det, mask, second)
    losses1 = core.scores(pred1, target, mask)
    losses2 = core.scores(pred2, target, mask)
    prediction = (losses1.mean() + losses2.mean()) / 2
    aux = prediction * 0
    metrics = {'inv': 0., 'var': 0., 'cov': 0.}
    weight = 0.
    if method == 'A-MQ-Reg':
        own = memory[rows, 0, focal_tensor]
        pair = torch.stack([own, expanded], 1)[:, :, None, :]
        aux, stats = core.auxiliary(pair, torch.ones(len(indices), 1, device=device),
            data.public['train'][indices, focal][:, None],
            data.physical['train'][indices, focal][:, None, :], False, 1000 + epoch)
        metrics = {key: stats[key] for key in metrics}
        weight = .05 * min(epoch / 5, 1.)
    return prediction + weight * aux, {
        'prediction': prediction, 'losses1': losses1, 'losses2': losses2,
        'memory': memory, 'second': second, 'shared': shared,
        'focal': focal_tensor, 'indices': indices, 'aux': aux,
        'weight': weight, 'metrics': metrics}


def smoke(out, base, device):
    binding, data, model, valplan = setup(out, base, device)
    plan = MultiQuerySampler(data, SUPPORTS, QUERY_GROUP).make_epoch(1)
    initial = core.evaluate(model, data, valplan, device)
    equivalence = core.warm_start_equivalence(model, data, binding['legacy_binding'], valplan, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    records = []
    reference = None
    for method in METHODS:
        model.load_state_dict(original)
        model.train()
        optimizer = optimizer_for(model)
        optimizer.zero_grad(set_to_none=True)
        before = model.encoder.mlp_inter[0].weight.detach().clone()
        loss, detail = forward_batch(model, data, plan, 0, BATCH_GROUPS, method, device, 1)
        first = float(detail['losses1'].mean().detach())
        if reference is None:
            reference = first
        if not np.isclose(first, reference, atol=2e-6, rtol=2e-5):
            raise ValueError('Main correct-history path changed between methods')
        w = model.encoder.mlp_inter[0].weight
        grads = [float(torch.autograd.grad(detail[key].mean(), w, retain_graph=True)[0].norm())
                 for key in ('losses1', 'losses2')]
        shared_grads = []
        if detail['shared'] is not None:
            for j in range(QUERY_GROUP):
                g = torch.autograd.grad(detail['losses2'][j], detail['shared'], retain_graph=True)[0]
                shared_grads.append(float(g[0].norm()))
            other = ~F.one_hot(detail['focal'], core.K).bool()
            torch.testing.assert_close(detail['second'][other], detail['memory'][:, 1][other], rtol=0, atol=0)
            focal_memory = detail['second'][torch.arange(len(detail['focal']), device=device), detail['focal']]
            torch.testing.assert_close(focal_memory, detail['shared'].repeat_interleave(QUERY_GROUP, 0), rtol=0, atol=0)
        if min(grads + shared_grads) <= 0:
            raise ValueError('A history/recipient path has no gradient')
        loss.backward()
        recurrent_gradient = float(model.head.cell.weight_ih.grad[:, -64:].norm())
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        change = float((model.encoder.mlp_inter[0].weight.detach() - before).abs().max())
        if change <= 0 or recurrent_gradient <= 0:
            raise ValueError('Encoder or recurrent history input failed to update')
        records.append({'method': method, 'main_mse': first,
            'second_mse': float(detail['losses2'].mean().detach()),
            'encoder_path_gradients': grads, 'three_query_shared_gradients': shared_grads,
            'recurrent_history_gradient': recurrent_gradient, 'encoder_max_update': change})
    host_peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if host_peak_gib * 4 > 24:
        raise RuntimeError('Four measured worker RSS budgets exceed 24GiB reserve')
    result = {'status': 'PASS', 'version': VERSION, 'test_read': False,
        'warm_start_equivalence': equivalence, 'host_peak_gib': host_peak_gib,
        'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
        'initial_mse': initial['mse'], 'queries': len(initial['ids']),
        'plan_sha256': plan['plan_sha256'], 'methods': records,
        'warm_start_reloaded_after_smoke': True, 'shared_node_used_by_all_three_queries': True}
    core.write(Path(out) / 'real_batch_smoke.json', result)
    core.emit('smoke_pass', **result)


def train(out, base, method, device, epochs):
    out = Path(out)
    if core.read(out / 'real_batch_smoke.json')['status'] != 'PASS':
        raise ValueError('Real batch has not passed')
    binding, data, model, valplan = setup(out, base, device)
    sampler = MultiQuerySampler(data, SUPPORTS, QUERY_GROUP)
    optimizer = optimizer_for(model)
    run = out / 'runs' / method
    run.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'method': method, 'test_read': False, 'seed': 0,
        'binding_sha256': core.digest(out / 'binding.json'),
        'encoder_lr': 1e-5, 'head_lr': 5e-5, 'weight_decay': 1e-4,
        'supports': SUPPORTS, 'queries_per_group': QUERY_GROUP, 'batch_groups': BATCH_GROUPS,
        'supervised_predictions_per_query': 2, 'history_encoder_trainable': True,
        'visual_frontend_frozen': True, 'reg_target_weight': .05 if method == 'A-MQ-Reg' else 0.,
        'reg_warmup_epochs': 5, 'validation_supports': 'correct independent S5'}
    config_path = run / 'config.json'
    if config_path.exists() and core.read(config_path) != config:
        raise ValueError('Run configuration differs')
    core.write(config_path, config)
    start, best, history = 0, float('inf'), []
    latest = run / 'latest.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume checkpoint differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, history = checkpoint['epoch'], checkpoint['best'], checkpoint['history']
    if start == 0:
        initial = core.evaluate(model, data, valplan, device)
        best = initial['mse']
        core.write(run / 'initial_validation.json', initial)
        core.save_torch(run / 'selected.pt', {'model': model.state_dict(), 'epoch': 0, 'config': config})
        core.write(run / 'selected_validation.json', {'method': method, 'epoch': 0, **initial})
        core.emit('warm_start', method=method, mse=best)
    for epoch in range(start + 1, epochs + 1):
        began = time.perf_counter()
        model.train()
        plan = sampler.make_epoch(epoch)
        core.write(run / f'plan_epoch{epoch}_summary.json',
                   {**plan['summary'], 'plan_sha256': plan['plan_sha256']})
        total, seen, first_batch = {}, 0, None
        for offset in range(0, len(plan['shared_correct']), BATCH_GROUPS):
            end = min(offset + BATCH_GROUPS, len(plan['shared_correct']))
            optimizer.zero_grad(set_to_none=True)
            loss, detail = forward_batch(model, data, plan, offset, end, method, device, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if offset == 0:
                first_batch = {'encoder_gradient': sum(float(p.grad.detach().square().sum())
                    for p in model.encoder.parameters() if p.grad is not None) ** .5,
                    'recurrent_history_gradient': float(model.head.cell.weight_ih.grad[:, -64:].norm())}
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            values = {'train_mse': float(detail['prediction'].detach()),
                'main_mse': float(detail['losses1'].mean().detach()),
                'second_mse': float(detail['losses2'].mean().detach()),
                'weighted_aux': float((detail['weight'] * detail['aux']).detach()),
                'total_loss': float(loss.detach()), 'grad_norm': float(norm), **detail['metrics']}
            n = len(detail['indices'])
            seen += n
            for key, value in values.items():
                total[key] = total.get(key, 0.) + value * n
        if seen != len(plan['query_indices']):
            raise ValueError('Training did not consume the complete epoch query plan')
        metric = core.evaluate(model, data, valplan, device)
        row = {'epoch': epoch, 'seconds': time.perf_counter() - began,
            'plan_sha256': plan['plan_sha256'], 'query_exposures': seen,
            'prediction_exposures': 2 * seen, 'padding': plan['summary']['padding_occurrences'],
            'first_batch': first_batch, **{k: v / seen for k, v in total.items()},
            **{k: v for k, v in metric.items() if k not in ('ids', 'per_recipient_mse')}}
        history.append(row)
        if metric['mse'] < best:
            best = metric['mse']
            core.save_torch(run / 'selected.pt', {'model': model.state_dict(), 'epoch': epoch, 'config': config})
            core.write(run / 'selected_validation.json', {'method': method, 'epoch': epoch, **metric})
        core.save_torch(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'best': best, 'history': history, 'config': config})
        core.write(run / 'progress.json', {'method': method, 'status': 'RUNNING',
            'epoch': epoch, 'best_mse': best, 'history': history, 'test_read': False})
        core.emit('epoch', method=method, **row)
    complete = {'method': method, 'status': 'COMPLETE', 'epochs': max(start, epochs),
                'best_mse': best, 'test_read': False}
    core.write(run / 'complete.json', complete)
    core.write(run / 'progress.json', {**complete, 'epoch': max(start, epochs), 'history': history})
    core.emit('training_complete', **complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'smoke', 'train'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.base)
    elif args.command == 'smoke':
        smoke(args.out, args.base, args.device)
    else:
        train(args.out, args.base, args.method, args.device, args.epochs)
