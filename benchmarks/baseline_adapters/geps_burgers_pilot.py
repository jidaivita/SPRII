#!/usr/bin/env python3
"""Finite GEPS-on-released-NOD Burgers development pilot, not a paper result.

Uses published GEPS neural modules and their actual Euler integration. An
external initialization repair zeros the released Swish context weights that
otherwise contain uninitialized memory. Official files are never modified.
Only released train cases 0:40 and development cases 40:45 are loaded. The NOD
loader's cache_mode='none' is deliberate: cond_init caches *all* 50 cases.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tensor_digest(tensor):
    import numpy as np
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def parameter_digest(model, *, exclude_codes=False):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_codes and name.endswith('codes'):
            continue
        h.update(name.encode())
        h.update(tensor_digest(tensor).encode())
    return h.hexdigest()


def load_released_data(args):
    import numpy as np
    import torch
    sys.path.insert(0, str(args.nod_source.resolve()))
    from ngs.utils import BurgersPairedDataset

    expected = {'train': (0, 39), 'eval': (40, 44), 'test': (45, 49)}
    if BurgersPairedDataset.SPLIT_CASES != expected:
        raise RuntimeError('Unknown released split contract; inspect instead of guessing.')
    output = {}
    metadata = {}
    for split in ('train', 'eval'):
        ds = BurgersPairedDataset(
            data_root=str(args.data_root.resolve()), split=split,
            space_stride=1, include_output_add_train=False,
            prediction_horizon=101, cache_mode='none', seed=args.seed,
        )
        try:
            if ds.expected_cond_x != 401 or ds.expected_cond_t != 101:
                raise RuntimeError(f'Unknown Burgers grid: {ds.describe()}')
            curves, envs, cases, per_case = [], [], [], []
            # Loading only explicit allowed cases avoids the original eager
            # cache's incidental loading of held-out case values.
            for shard_idx, case_idx in ds.sample_index:
                shard = ds.shards[shard_idx]
                if case_idx not in range(expected[split][0], expected[split][1] + 1):
                    raise RuntimeError('A held-out case entered the development loader.')
                arr = ds._load_trunk_ds(shard_idx, case_idx)
                if arr.shape != (101, 401) or not np.isfinite(arr).all():
                    raise RuntimeError(f'Invalid trajectory at {shard_idx}/{case_idx}')
                curve = torch.from_numpy(np.ascontiguousarray(arr.T)).unsqueeze(0)
                curves.append(curve)
                envs.append(int(shard['nu_id']))
                cases.append(int(case_idx))
                per_case.append({
                    'path': str(Path(shard['path']).resolve()),
                    'shard_index': shard_idx, 'case': case_idx,
                    'nu_id': int(shard['nu_id']),
                    'trajectory_sha256': tensor_digest(curve),
                })
            temporal = torch.from_numpy(ds.temporal_idx.astype('float32'))
            denom = float(ds.n_t - 1)
            times = temporal / denom  # Exact clean NOD normalized time contract.
            if not torch.all(times[1:] > times[:-1]) or times[0] != 0:
                raise RuntimeError('Nonmonotonic timeline.')
            output[split] = {
                'curves': torch.stack(curves),
                'envs': torch.tensor(envs, dtype=torch.long),
                'cases': torch.tensor(cases, dtype=torch.long), 't': times,
            }
            semantic = json.dumps(per_case, sort_keys=True, separators=(',', ':')).encode()
            metadata[split] = {
                'description': ds.describe(), 'shape': list(output[split]['curves'].shape),
                'allowed_cases': list(range(expected[split][0], expected[split][1] + 1)),
                'temporal_indices': ds.temporal_idx.tolist(), 'time_normalization': denom,
                'data_sha256': hashlib.sha256(semantic).hexdigest(),
                'data_hash_definition': 'ordered allowed-case trajectory tensor hashes and source identities; no test values read',
                'cases': per_case,
            }
        finally:
            ds.close()
    if not torch.equal(output['train']['t'], output['eval']['t']):
        raise RuntimeError('Training and development timelines differ.')
    return output, metadata


def build_model(args, n_env, device):
    import torch
    sys.path.insert(0, str(args.geps_source.resolve()))
    from geps.model.forecasters import Forecaster
    from geps.model.activations import Swish
    from geps.utils import init_weights

    model = Forecaster('burgers', 1, 64, args.code_dim, 1, n_env,
                       True, '', 'euler', None).to(device)
    init_weights(model, init_config={
        'A': {'type': 'orthogonal', 'gain': 1},
        'B': {'type': 'orthogonal', 'gain': 1},
        'weight': {'type': 'orthogonal', 'gain': 1},
    })
    repaired = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Swish):
                module.weight.zero_()
                repaired.append(name + '.weight')
        # This tensor is also created with torch.empty in the release; the
        # physics branch is unused, but deterministic checkpoints still need it.
        model.derivative.model_phy.weight.zero_()
        repaired.append('derivative.model_phy.weight (inactive physics branch)')
    if model.method != 'euler' or len(repaired) != 4:
        raise RuntimeError('Published model structure changed; re-audit required.')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite initialization.')
    return model, repaired


def relative_l2(pred, truth):
    import torch
    num = (pred - truth).flatten(1).norm(dim=1)
    den = truth.flatten(1).norm(dim=1)
    if torch.any(den <= 0):
        raise RuntimeError('Zero-norm target, cannot apply published relative L2.')
    return (num / den).mean()


def smoke(model, curves, env, times):
    import torch
    model.zero_grad(set_to_none=True)
    pred = model(curves, times, env, epsilon=0)
    loss = relative_l2(pred, curves)
    if pred.shape != curves.shape or not torch.isfinite(loss):
        raise RuntimeError('Forward shape/finite smoke failed.')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Backward finite smoke failed.')
    code_grad = model.derivative.codes.grad
    if code_grad is None or float(code_grad.abs().sum()) == 0:
        raise RuntimeError('Context gradient is absent.')
    with torch.no_grad():
        changed = curves.clone()
        changed[..., 1:] += 3.0
        alternate = model(changed, times, env, epsilon=0)
        no_target_read = torch.equal(pred, alternate)
        if not no_target_read:
            raise RuntimeError('Forecast consumed future target values.')
    out = {'loss': float(loss.detach()), 'shape': list(pred.shape),
           'finite_backward': True, 'code_gradient_l1': float(code_grad.abs().sum()),
           'future_target_perturbation_prediction_unchanged': no_target_read,
           'optimizer_updates': 0}
    model.zero_grad(set_to_none=True)
    return out


def development(model, data, args, device, out):
    import numpy as np
    import torch
    from torch import nn
    bank = data['eval']
    n = len(bank['curves'])
    # One independent same-system donor per recipient, cyclic within each of
    # the five released development cases. No test/extra trajectory is loaded.
    pairs = []
    for env in torch.unique(bank['envs']).tolist():
        idx = torch.where(bank['envs'] == env)[0].tolist()
        for pos, recipient in enumerate(idx):
            donor = idx[(pos + 1) % len(idx)]
            pairs.append((recipient, donor))
    if len(pairs) != n:
        raise RuntimeError('Development recipient coverage mismatch.')
    initial_code = model.derivative.codes.detach().mean(dim=0).clone()
    frozen_sha = parameter_digest(model, exclude_codes=True)
    records, code_records = [], []
    times = bank['t'].to(device)
    for start in range(0, n, args.eval_batch_size):
        batch_pairs = pairs[start:start + args.eval_batch_size]
        r = [p[0] for p in batch_pairs]
        d = [p[1] for p in batch_pairs]
        support = bank['curves'][d].to(device)
        target = bank['curves'][r].to(device)
        adapted = copy.deepcopy(model)
        codes = nn.Parameter(initial_code[None].repeat(len(r), 1))
        adapted.derivative.codes = codes
        adapted.derivative.model_aug.codes = codes
        for p in adapted.parameters():
            p.requires_grad_(False)
        codes.requires_grad_(True)
        optimizer = torch.optim.Adam([codes], lr=args.adapt_lr, betas=(0.9, 0.999))
        env = torch.arange(len(r), device=device)
        with torch.no_grad():
            initial_support_loss = float((adapted(support, times, env) - support).square().mean())
        for _ in range(args.adapt_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = adapted(support, times, env, epsilon=0)
            loss = (prediction - support).square().mean()  # Released adaptation objective.
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite support adaptation loss.')
            loss.backward()
            if codes.grad is None or not torch.isfinite(codes.grad).all():
                raise RuntimeError('Nonfinite/missing adaptation gradient.')
            optimizer.step()
        if parameter_digest(adapted, exclude_codes=True) != frozen_sha:
            raise RuntimeError('Adaptation changed a shared predictor parameter.')
        with torch.no_grad():
            prediction = adapted(target, times, env, epsilon=0)
            final_support_loss = float((adapted(support, times, env) - support).square().mean())
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Nonfinite development predictions.')
            for j, (recipient, donor) in enumerate(batch_pairs):
                error = (prediction[j] - target[j]).square()
                records.append({
                    'nu_id': int(bank['envs'][recipient]),
                    'recipient_case': int(bank['cases'][recipient]),
                    'donor_case': int(bank['cases'][donor]),
                    'mse_including_initial': float(error.mean()),
                    'mse_future': float(error[..., 1:].mean()),
                    'horizon_mse': {str(h): float(error[..., h].mean()) for h in (1, 5, 50, 100)},
                    'support_initial_mse_batch': initial_support_loss,
                    'support_final_mse_batch': final_support_loss,
                    'context_steps': args.adapt_steps,
                })
                code_records.append(codes[j].detach().cpu().tolist())
        del adapted, optimizer
    write(out / 'DEVELOPMENT_PAIRS.json', records)
    write(out / 'DEVELOPMENT_CODES.json', {
        'initial_code': initial_code.cpu().tolist(), 'initial_code_sha256': tensor_digest(initial_code),
        'initialization': 'mean of training environment codes; reset independently for each recipient/donor pair',
        'adapted_codes': code_records,
    })
    return {
        'split': 'released eval, cases 40..44', 'n_query_trajectories': n,
        'support_trajectories_per_query': 1, 'support_frames': 101,
        'query_future_points_per_trajectory': 100 * 401,
        'query_points_including_initial_per_trajectory': 101 * 401,
        'direct_report_horizons': [1, 5, 50, 100],
        'forecast_type': 'Euler autoregressive numerical integration, not NOD direct-horizon decoding',
        'mse_future': float(np.mean([r['mse_future'] for r in records])),
        'mse_including_initial': float(np.mean([r['mse_including_initial'] for r in records])),
        'horizon_mse': {str(h): float(np.mean([r['horizon_mse'][str(h)] for r in records])) for h in (1, 5, 50, 100)},
        'predictor_sha256_before_after': frozen_sha,
        'support_steps_per_pair': args.adapt_steps, 'test_read': False, 'ood_read': False,
        'formal_result': False,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('inspect', 'smoke', 'pilot'), default='inspect')
    p.add_argument('--nod-source', type=Path, required=True)
    p.add_argument('--geps-source', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=4)
    p.add_argument('--code-dim', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--adapt-lr', type=float, default=0.01)
    p.add_argument('--adapt-steps', type=int, default=50)
    p.add_argument('--max-seconds', type=float, default=3600)
    return p


def main(args):
    import numpy as np
    import torch
    if not 1 <= args.updates <= 1000 or not 1 <= args.adapt_steps <= 500:
        raise ValueError('Finite pilot cap: <=1000 train updates and <=500 support steps.')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {args.output}')
    args.output.mkdir(parents=True)
    if args.output.resolve().is_relative_to(args.geps_source.resolve()):
        raise ValueError('Output may not be inside immutable source.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_hashes = {str(p.relative_to(args.geps_source)): digest(p)
                     for p in sorted(args.geps_source.rglob('*.py'))}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({
        'script_sha256': digest(__file__), 'geps_source_sha256': source_hashes,
        'nod_loader_sha256': digest(args.nod_source / 'ngs/utils.py'),
        'test_read': False, 'ood_read': False, 'formal_result': False,
        'protocol': 'independent GEPS development pilot on clean released NOD train/eval split',
        'official_original_modified': False, 'training_target': 'one complete training trajectory per item, learned per-system context',
        'observation_budget_note': 'same available training cases; updates/query-point budget not claimed equivalent to NOD',
    })
    write(args.output / 'CONFIG.json', config)
    data, meta = load_released_data(args)
    write(args.output / 'DATA.json', meta)
    if args.mode == 'inspect':
        write(args.output / 'COMPLETE.json', {'status': 'INSPECTED', 'test_read': False, 'ood_read': False})
        return
    device = torch.device(args.device)
    n_env = int(data['train']['envs'].max()) + 1
    model, repaired = build_model(args, n_env, device)
    write(args.output / 'INITIALIZATION.json', {
        'zero_initialized_release_empty_parameters': repaired,
        'formula_unchanged': True, 'actual_solver': model.method,
        'initial_model_sha256': parameter_digest(model),
        'initial_train_codes_sha256': tensor_digest(model.derivative.codes),
        'initial_train_codes': model.derivative.codes.detach().cpu().tolist(),
    })
    train = data['train']
    batch = min(args.batch_size, len(train['curves']))
    times = train['t'].to(device)
    write(args.output / 'SMOKE.json', smoke(
        model, train['curves'][:batch].to(device), train['envs'][:batch].to(device), times,
    ))
    if args.mode == 'smoke':
        write(args.output / 'COMPLETE.json', {'status': 'SMOKE_PASS', 'training_updates': 0})
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=350, threshold=0.01, min_lr=1e-5,
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train['curves']), generator=generator)
    cursor = 0
    started = time.monotonic()
    losses = []
    for step in range(1, args.updates + 1):
        if cursor + batch > len(order):
            order = torch.randperm(len(train['curves']), generator=generator)
            cursor = 0
        selected = order[cursor:cursor + batch]
        cursor += batch
        curves = train['curves'][selected].to(device)
        env = train['envs'][selected].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(curves, times, env, epsilon=0)
        loss = relative_l2(pred, curves)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite train loss at update {step}')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Nonfinite train gradient at update {step}')
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        # Published scheduler consumes a training epoch loss. Use the mean of
        # each complete pass, not a new per-step schedule masquerading as it.
        pass_size = len(train['curves']) // batch
        if step % pass_size == 0:
            scheduler.step(float(np.mean(losses[-pass_size:])))
        if step == 1 or step % 10 == 0 or step == args.updates:
            row = {'update': step, 'relative_l2': value, 'elapsed_seconds': time.monotonic() - started,
                   'lr': optimizer.param_groups[0]['lr']}
            with (args.output / 'TRAIN.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
        if time.monotonic() - started >= args.max_seconds:
            break
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'updates': step, 'config': config}, args.output / 'latest.pt')
    result = development(model, data, args, device, args.output)
    result.update({'actual_training_updates': step, 'requested_training_updates': args.updates,
                   'bounded_time_stop': step < args.updates,
                   'training_relative_l2_first20': float(np.mean(losses[:20])),
                   'training_relative_l2_last20': float(np.mean(losses[-20:])),
                   'train_points_per_update_including_initial': batch * 101 * 401,
                   'train_trajectory_observations': step * batch,
                   'checkpoint_sha256': digest(args.output / 'latest.pt')})
    final_hashes = {str(p.relative_to(args.geps_source)): digest(p)
                    for p in sorted(args.geps_source.rglob('*.py'))}
    if final_hashes != source_hashes:
        raise RuntimeError('Official source changed during run.')
    write(args.output / 'SUMMARY.json', result)
    write(args.output / 'COMPLETE.json', {'status': 'PILOT_COMPLETE', 'formal_result': False,
          'summary_sha256': digest(args.output / 'SUMMARY.json'), 'updates': step})


if __name__ == '__main__':
    arguments = parser().parse_args()
    existed = arguments.output.exists()
    code = 1
    try:
        main(arguments)
        code = 0
    except Exception as exc:
        if arguments.output.is_dir() and not existed:
            write(arguments.output / 'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        if arguments.output.is_dir() and not existed:
            write(arguments.output / 'EXIT.json', {'exit_code': code, 'time': time.time()})
    sys.exit(code)
