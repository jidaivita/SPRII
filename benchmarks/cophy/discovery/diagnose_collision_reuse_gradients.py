"""One TRAIN-batch gradient calibration at the common v4.6 epoch-zero start.

No optimizer is constructed, no weights are updated, no validation arrays or
targets are loaded, and no training files are changed. Weight suggestions are
component gradient scales, not predicted performance improvements.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import collision_reuse_ft as core


class TrainOnlyData(core.FineTuneData):
    """Reuse core plan/batch methods without PilotData's validation reads."""
    def __init__(self, base, device, binding):
        self.base = Path(base)
        self.manifest = core.read(self.base / 'manifest.json')
        part = self.manifest['splits']['train']
        checks = binding['splits']['train']
        for path, expected in ((part['cache'], checks['cache_sha256']),
                               (self.base / 'input_train.npz', checks['input_sha256']),
                               (self.base / 'target_train.npz', checks['target_sha256'])):
            if core.digest(path) != expected:
                raise ValueError('Train input differs from binding: ' + str(path))
        with np.load(self.base / 'input_train.npz', allow_pickle=False) as inp:
            row = {'q': inp['pose'].copy(), 'det': inp['detected'].copy(),
                   'mask': inp['presence'].copy(), 'ids': list(map(str, inp['ids']))}
        with np.load(self.base / 'target_train.npz', allow_pickle=False) as target:
            row['target'] = target['pose'].copy()
        if row['ids'] != part['query_ids']:
            raise ValueError('Train query order differs from frozen manifest')
        self.rows = {'train': row}
        with open(part['cache'], 'rb') as stream:
            cache = pickle.load(stream)
        ids = part['all_ids']
        self.hist = {'train': {
            'pose': torch.from_numpy(np.stack([cache[i]['pose_ab'] for i in ids])).float().to(device),
            'presence': torch.from_numpy(np.stack([cache[i]['presence_ab'] for i in ids])).float().to(device)}}
        del cache
        public = np.asarray([part['known_type'][i] for i in row['ids']]).argmax(-1)
        self.public = {'train': public + np.arange(core.K)[None] * len(core.TYPES)}
        self.physical = {'train': np.asarray([part['physical'][i] for i in row['ids']])}


def setup_train_only(out, base, device):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    binding = core.read(Path(out) / 'binding.json')
    if binding['code_sha256'] != core.digest(core.__file__):
        raise ValueError('Core implementation differs from epoch-zero binding')
    if binding['dependency_sha256'] != core.digest(Path(core.__file__).with_name('collision_xep.py')):
        raise ValueError('Head/data dependency differs from binding')
    if binding['base_manifest_sha256'] != core.digest(Path(base) / 'manifest.json'):
        raise ValueError('Base manifest differs from binding')
    for entry in (binding['source'], binding['head']):
        if core.digest(entry['path']) != entry['sha256']:
            raise ValueError('Common initialization changed')
    data = TrainOnlyData(base, device, binding)
    model = core.FineTuneModel(binding).to(device).train()
    return binding, data, model


def components(memory, mask, strata, physical, seed):
    """Same scalar formulas and public-stratum permutation as core.auxiliary."""
    active = mask.reshape(-1) > 0
    active_np = active.detach().cpu().numpy()
    x = memory[:, 0].reshape(-1, 64)[active]
    y = memory[:, 1].reshape(-1, 64)[active]
    labels = np.asarray(strata).reshape(-1)[active_np]
    properties = np.asarray(physical).reshape(-1, 3)[active_np]
    order = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    residual_x, residual_y = [], []
    counts = {}
    for key in np.unique(labels):
        ix = np.flatnonzero(labels == key)
        counts[str(key)] = len(ix)
        if len(ix) > 1:
            shuffled = rng.permutation(ix)
            order[shuffled] = np.roll(shuffled, 1)
            tx = torch.as_tensor(ix, device=x.device)
            residual_x.append(x[tx] - x[tx].mean(0, keepdim=True))
            residual_y.append(y[tx] - y[tx].mean(0, keepdim=True))
    invariance_correct = F.mse_loss(x, y)
    invariance_random = F.mse_loss(x, y[torch.as_tensor(order, device=y.device)])
    zero = x.sum() * 0
    variance = covariance = zero
    if residual_x:
        rx, ry = torch.cat(residual_x), torch.cat(residual_y)
        variance = (F.relu(1 - torch.sqrt(rx.var(0, unbiased=False) + 1e-4)).mean() +
                    F.relu(1 - torch.sqrt(ry.var(0, unbiased=False) + 1e-4)).mean()) / 2
        def cov_penalty(value):
            cov = value.T @ value / max(len(value) - 1, 1)
            return (cov.square().sum() - cov.diagonal().square().sum()) / value.shape[1]
        covariance = (cov_penalty(rx) + cov_penalty(ry)) / 2
    moved = order != np.arange(len(order))
    accidental = (float(np.mean(np.all(properties[moved] == properties[order[moved]], axis=-1)))
                  if moved.any() else 0.0)
    if not np.array_equal(labels, labels[order]):
        raise ValueError('Random pairing crossed a public stratum')
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError('Random pairing changed side marginals')
    return {'invariance_correct': invariance_correct, 'invariance_random': invariance_random,
            'variance': variance, 'covariance': covariance}, {
                'active_objects': len(labels), 'public_stratum_counts': counts,
                'random_unpermutable_fraction': float(np.mean(~moved)),
                'random_accidental_same_fraction_among_moved': accidental,
                'same_public_strata': True, 'same_vector_marginals': True,
                'random_permutation_sha256': hashlib.sha256(order.tobytes()).hexdigest()}


def grad_vector(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    pieces = [(torch.zeros_like(parameter) if grad is None else grad).detach().reshape(-1).cpu().double()
              for parameter, grad in zip(parameters, grads)]
    value = torch.cat(pieces)
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite component gradient')
    return value


def describe(vector, task):
    norm, task_norm = float(vector.norm()), float(task.norm())
    return {'norm': norm, 'norm_ratio_to_prediction': norm / task_norm if task_norm > 0 else None,
            'cosine_with_prediction': float(torch.dot(vector, task) / (norm * task_norm))
            if norm > 0 and task_norm > 0 else None}


def run(out, base, device, batch_size):
    began = time.perf_counter()
    binding, data, model = setup_train_only(out, base, device)
    epoch = 1
    plan = data.plan('train', epoch, 2)
    # Match the first actual epoch-one minibatch and its random relation seed.
    ix = np.random.default_rng(771 + epoch).permutation(len(data.rows['train']['ids']))[:batch_size]
    pred, memory, mask, l1, l2 = core.prediction_pair(model, data, ix, plan, device)
    losses, pairing = components(memory, mask, data.public['train'][ix],
                                 data.physical['train'][ix], 900000 + epoch * 10000)
    losses = {'prediction': pred, **losses}
    # Compare repeated formulas against the actual core on this same train batch.
    with torch.no_grad():
        for randomized, name in ((False, 'invariance_correct'), (True, 'invariance_random')):
            combined, _ = core.auxiliary(memory, mask, data.public['train'][ix],
                                         data.physical['train'][ix], randomized,
                                         900000 + epoch * 10000)
            expected = losses[name] + losses['variance'] + .04 * losses['covariance']
            torch.testing.assert_close(combined, expected, rtol=1e-6, atol=1e-7)
    encoder = tuple(model.encoder.parameters())
    support = tuple(model.head.support.parameters())
    parameters = encoder + support
    split = sum(parameter.numel() for parameter in encoder)
    vectors = {name: grad_vector(loss, parameters) for name, loss in losses.items()}
    vectors['auxiliary_correct'] = (vectors['invariance_correct'] + vectors['variance'] +
                                    .04 * vectors['covariance'])
    vectors['auxiliary_random'] = (vectors['invariance_random'] + vectors['variance'] +
                                   .04 * vectors['covariance'])
    vectors['correct_minus_random_invariance'] = vectors['invariance_correct'] - vectors['invariance_random']
    task = vectors['prediction']
    metrics = {name: {'encoder_plus_support': describe(vector, task),
                     'history_encoder': describe(vector[:split], task[:split]),
                     'support_mlp': describe(vector[split:], task[split:])}
               for name, vector in vectors.items()}
    suggested = {}
    for group, sl in (('encoder_plus_support', slice(None)),
                      ('history_encoder', slice(None, split)), ('support_mlp', slice(split, None))):
        pnorm = float(task[sl].norm())
        invnorm = float(vectors['invariance_correct'][sl].norm())
        suggested[group] = {str(fraction): fraction * pnorm / invnorm if invnorm > 0 else None
                            for fraction in (.05, .1)}
    scalar_losses = {name: float(loss.detach()) for name, loss in losses.items()}
    current = {}
    for weight in (.01, .05):
        current[str(weight)] = {
            name: describe(weight * vectors[name], task)
            for name in ('invariance_correct', 'invariance_random', 'auxiliary_correct', 'auxiliary_random')}
    result = {'stage': 'one_train_batch_common_epoch0_gradient_diagnostic',
              'test_read': False, 'validation_arrays_or_targets_read': False,
              'optimizer_created': False, 'optimizer_steps': 0,
              'initialization': {'source_sha256': binding['source']['sha256'],
                                 'head_sha256': binding['head']['sha256'],
                                 'head_epoch': binding['head']['epoch']},
              'core_code_sha256': core.digest(core.__file__),
              'diagnostic_code_sha256': core.digest(__file__),
              'batch_size': len(ix), 'query_ids': [data.rows['train']['ids'][i] for i in ix],
              'epoch1_train_plan_sha256': hashlib.sha256(plan.tobytes()).hexdigest(),
              'batch_plan_sha256': hashlib.sha256(plan[ix].tobytes()).hexdigest(),
              'prediction_mse_set1': float(l1.detach()), 'prediction_mse_set2': float(l2.detach()),
              'pairing': pairing, 'losses': scalar_losses, 'gradient_metrics': metrics,
              'current_full_weight_gradient_ratios': current,
              'suggested_invariance_weights_for_task_gradient_fractions': suggested,
              'interpretation': 'Unclipped raw gradients at one common initial train batch; '
                  'weights scale the invariance component alone. They are not optimal weights, '
                  'do not predict validation gains, and do not include AdamW or clipping effects.',
              'seconds': time.perf_counter() - began}
    core.write(Path(out) / 'diagnostic_gradient_initial.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error('--batch-size must be at least 2')
    print(json.dumps(run(args.out, args.base, args.device, args.batch_size),
                     ensure_ascii=False, allow_nan=False))
