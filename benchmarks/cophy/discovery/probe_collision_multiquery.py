"""Fixed linear probes of the actual S5 memory used by Collision v4.9.

No task training or checkpoint selection. The train split fits the probe and
its normalization; the original 512 validation recipients only score it.
Memory64 means the head.support(U32) aggregation, not the original P16 branch.
"""

import os
import argparse
import fcntl
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import collision_reuse_ft as core
import collision_xep as xep

VERSION = 'collision-multiquery-memory64-probe-v1'
METHODS = ('Warm-Native', 'Native-MQ', 'A-MQ', 'Random-MQ', 'A-MQ-Reg')
ALPHA = 1.
SUPPORTS = 5


def immutable_write(path, value):
    path = Path(path)
    if path.exists() and core.read(path) != value:
        raise ValueError('Output already bound to a different probe: ' + str(path))
    core.write(path, value)


def training_plan(data):
    part, row = data.manifest['splits']['train'], data.rows['train']
    lookup = {ident: i for i, ident in enumerate(part['all_ids'])}
    plan = np.zeros((len(row['ids']), core.K, 1, SUPPORTS), np.int64)
    for i, ident in enumerate(row['ids']):
        seed = int.from_bytes(hashlib.sha256(f'xep:20260911:train:{ident}:0'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        for slot in range(core.K):
            if row['mask'][i, slot] <= 0:
                continue
            options = part['candidates'][ident][slot]
            if len(options) < SUPPORTS or lookup[ident] in options:
                raise ValueError('Insufficient or non-independent train probe support')
            plan[i, slot, 0] = rng.choice(options, SUPPORTS, replace=False)
    return plan


def labels_from_audit(base, data):
    """Verify semantic order and actual raw values; never regress on class IDs."""
    from cophy_fields import PLANS
    import cophy_fields
    base = Path(base)
    prepath = base.parent / 'prepared_v3/collision/training_preflight.json'
    if core.digest(prepath) != data.manifest['preflight_sha256']:
        raise ValueError('Probe metadata preflight differs')
    preflight = core.read(prepath)
    raw_audit_path = xep.artifact(preflight, 'raw_audit')
    raw_audit = core.read(raw_audit_path)
    spec = PLANS['collision']
    fields, levels = spec['fields'], spec['support']
    if fields != ['mass', 'friction', 'restitution'] or raw_audit['fields'] != fields:
        raise ValueError('Collision physical field order was not confirmed')
    source_file = Path(cophy_fields.__file__)
    recorded = preflight.get('code_sha256', {}).get('cophy_fields.py')
    if recorded is not None and core.digest(source_file) != recorded:
        raise ValueError('Audited field interpreter changed')
    documents, tables, raw_available = {}, {}, True
    provenance = {'fields': fields, 'nominal_levels': levels, 'preflight_sha256': core.digest(prepath),
        'raw_audit_sha256': core.digest(raw_audit_path), 'field_interpreter_sha256': core.digest(source_file),
        'category_mapping_tolerance': .01, 'raw_relation_sha256': {}, 'numeric_unavailable_reasons': []}
    for split in ('train', 'val'):
        path = xep.artifact(preflight, 'raw_relations_' + split)
        provenance['raw_relation_sha256'][split] = core.digest(path)
        documents[split] = core.read(path)
        table = {(r['id'], r['slot']): r for r in documents[split]}
        labels, numeric, public, object_ids, slots = [], [], [], [], []
        row = data.rows[split]
        for i, ident in enumerate(row['ids']):
            for slot in np.flatnonzero(row['mask'][i] > 0):
                r = table[(ident, int(slot))]
                if r['split'] != split or not r['in_C'] or not np.array_equal(r['physical'], data.physical[split][i, slot]):
                    raise ValueError('Query label disagrees with audited raw relation')
                if core.TYPES.index(r['known_type']) + slot * len(core.TYPES) != data.public[split][i, slot]:
                    raise ValueError('Public slot/type disagrees with audited raw relation')
                values = r.get('raw_physical')
                if values is None or len(values) != len(fields) or not np.isfinite(values).all():
                    raw_available = False
                    values = [0.] * len(fields)
                    provenance['numeric_unavailable_reasons'].append(f'{split}:{ident}:{slot}:missing_raw_values')
                else:
                    for j, (value, label) in enumerate(zip(values, r['physical'])):
                        matches = [k for k, level in enumerate(levels[j]) if abs(float(value) - float(level)) < .01]
                        if matches != [label]:
                            raise ValueError('Raw physical value/category mapping is inconsistent')
                labels.append(r['physical']); numeric.append(values)
                public.append(data.public[split][i, slot]); object_ids.append(ident); slots.append(int(slot))
        tables[split] = {'labels': np.asarray(labels, np.int64), 'numeric': np.asarray(numeric, np.float64),
            'public': np.asarray(public, np.int64), 'query_ids': np.asarray(object_ids), 'slots': np.asarray(slots)}
    if set(tables['train']['query_ids']) & set(tables['val']['query_ids']):
        raise ValueError('Probe train/validation experiments overlap')
    provenance['numeric_r2_available'] = raw_available
    provenance['numeric_target'] = 'actual raw_physical values, not category identifiers' if raw_available else None
    return tables, provenance


@torch.no_grad()
def memory_features(model, data, split, plan, device):
    """Encode each full AB scene once, then aggregate its selected object codes."""
    model.eval()
    hist = data.hist[split]
    codes = torch.cat([model.encoder(hist['pose'][off:off + 256], hist['presence'][off:off + 256])
                       for off in range(0, len(hist['pose']), 256)])
    features = []
    for off in range(0, len(plan), 256):
        indices = torch.as_tensor(plan[off:off + 256, :, 0], device=device, dtype=torch.long)
        slots = torch.arange(core.K, device=device)[None, :, None].expand_as(indices)
        selected = codes[indices, slots]
        standardized = (selected - model.code_mean) / model.code_scale
        memory = model.head.support(standardized).mean(2)
        active = data.rows[split]['mask'][off:off + len(indices)] > 0
        features.append(memory.cpu().numpy()[active])
    answer = np.concatenate(features)
    # One small direct-path comparison verifies this cache optimization, not model quality.
    direct = model.memory(data, split, plan[:4])[:, 0].cpu().numpy()
    active = data.rows[split]['mask'][:4] > 0
    delta = float(np.abs(answer[:int(active.sum())] - direct[active]).max())
    if not np.allclose(answer[:int(active.sum())], direct[active], atol=2e-5, rtol=2e-5):
        raise ValueError('Cached memory extraction differs from the prediction path')
    return answer.astype(np.float64), delta


def ridge_fit_predict(x, target, evaluation):
    """Unpenalized intercept, train-only standardization; sum squared error + alpha*||W||²."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    z, e = (x - mean) / scale, (evaluation - mean) / scale
    center = target.mean(0)
    weight = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (target - center))
    prediction = e @ weight + center
    return prediction, {'mean': mean, 'scale': scale, 'weight': weight, 'intercept': center}


def group_means(values, train_groups, evaluation_groups):
    overall = values.mean(0)
    means = {int(group): values[train_groups == group].mean(0) for group in np.unique(train_groups)}
    train = np.stack([means[int(group)] for group in train_groups])
    evaluation = np.stack([means.get(int(group), overall) for group in evaluation_groups])
    return train, evaluation


def classification(truth, predicted, classes):
    confusion = np.zeros((classes, classes), np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(1)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(classes), where=support > 0)
    return {'accuracy': float((truth == predicted).mean()),
        'balanced_accuracy': float(recall[support > 0].mean()),
        'class_support': support.tolist(), 'confusion': confusion.tolist(), 'examples': len(truth)}


def regression(truth, predicted):
    residual = float(np.square(truth - predicted).sum())
    total = float(np.square(truth - truth.mean()).sum())
    return {'r2': 1 - residual / total if total > 0 else None,
            'mse': residual / len(truth), 'target_variance': total / len(truth), 'examples': len(truth)}


def fit_probes(train_x, val_x, labels, metadata):
    tr, va = labels['train'], labels['val']
    classes = [len(levels) for levels in metadata['nominal_levels']]
    onehot = np.concatenate([np.eye(n)[tr['labels'][:, j]] for j, n in enumerate(classes)], 1)
    target = np.concatenate([onehot, tr['numeric']], 1) if metadata['numeric_r2_available'] else onehot
    primary, fit = ridge_fit_predict(train_x, target, val_x)
    train_prior, val_prior = group_means(target, tr['public'], va['public'])
    train_xprior, val_xprior = group_means(train_x, tr['public'], va['public'])
    residual, conditioned_fit = ridge_fit_predict(train_x - train_xprior, target - train_prior, val_x - val_xprior)
    conditional = val_prior + residual
    global_prior = np.broadcast_to(target.mean(0), (len(val_x), target.shape[1]))
    record, predictions = {'fields': {}}, {}
    start = 0
    for j, (name, count) in enumerate(zip(metadata['fields'], classes)):
        truth = va['labels'][:, j]
        entry = {'train_class_support': np.bincount(tr['labels'][:, j], minlength=count).tolist(),
                 'validation_class_support': np.bincount(truth, minlength=count).tolist(),
                 'unseen_validation_classes': sorted(set(truth.tolist()) - set(tr['labels'][:, j].tolist()))}
        for view, value in [('memory64', primary), ('memory64_given_public', conditional),
                            ('public_slot_type_prior', val_prior), ('global_majority', global_prior)]:
            prediction = value[:, start:start + count].argmax(1)
            entry[view] = classification(truth, prediction, count)
            predictions[name + '_' + view] = prediction
        entry['beyond_public_balanced_accuracy_delta'] = (entry['memory64_given_public']['balanced_accuracy']
                                                        - entry['public_slot_type_prior']['balanced_accuracy'])
        if metadata['numeric_r2_available']:
            numeric_column = sum(classes) + j
            entry['numeric_regression'] = {view: regression(va['numeric'][:, j], value[:, numeric_column])
                for view, value in [('memory64', primary), ('memory64_given_public', conditional),
                                    ('public_slot_type_mean', val_prior), ('global_train_mean', global_prior)]}
            predictions[name + '_actual_numeric_prediction'] = primary[:, numeric_column]
        record['fields'][name] = entry
        start += count
    fit_arrays = {**{'primary_' + key: value for key, value in fit.items()},
                  **{'conditioned_' + key: value for key, value in conditioned_fit.items()}}
    return record, predictions, fit_arrays


def run(base, source_run, out, device):
    base, source_run, out = map(Path, (base, source_run, out))
    if out.resolve() in (base.resolve(), source_run.resolve()):
        raise ValueError('Use an independent probe output directory')
    out.mkdir(parents=True, exist_ok=True)
    status = core.read(source_run / 'controller_status.json')
    if status.get('status') != 'COMPLETE':
        raise ValueError('Probe only after v4.9 completes checkpoint selection')
    binding = core.read(source_run / 'binding.json')
    if binding['base_manifest_sha256'] != core.digest(base / 'manifest.json'):
        raise ValueError('Task manifest binding differs')
    for path, sha in binding['file_sha256'].items():
        if core.digest(path) != sha:
            raise ValueError('v4.9 dependency changed: ' + path)
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    data = core.FineTuneData(base, device)
    if len(data.rows['train']['ids']) != 14000 or len(data.rows['val']['ids']) != 512:
        raise ValueError('Expected original 14000/512 new-task probe cohorts')
    plans = {'train': training_plan(data)}
    with np.load(binding['validation_plan'], allow_pickle=False) as saved:
        plans['val'] = saved['plan'].copy()
        if saved['query_ids'].tolist() != data.rows['val']['ids'] or plans['val'].shape != (512, core.K, 1, SUPPORTS):
            raise ValueError('Original S5 validation support plan differs')
    labels, metadata = labels_from_audit(base, data)
    sources = {}
    for method in METHODS:
        path = Path(binding['warm_start']) if method == 'Warm-Native' else source_run / 'runs' / method / 'selected.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if method == 'Warm-Native':
            if checkpoint['epoch'] != 12 or checkpoint['config']['method'] != 'Native-FT':
                raise ValueError('Expected common v4.6 Native epoch12 warm start')
        else:
            metric = core.read(path.parent / 'selected_validation.json')
            if checkpoint['epoch'] != metric['epoch'] or checkpoint['config']['method'] != method or metric['ids'] != data.rows['val']['ids']:
                raise ValueError('Selected v4.9 checkpoint differs from receipt')
            if checkpoint['config']['binding_sha256'] != core.digest(source_run / 'binding.json'):
                raise ValueError('Selected model configuration binding differs')
        sources[method] = {'path': str(path), 'sha256': core.digest(path), 'selected_epoch': checkpoint['epoch']}
    config = {'version': VERSION, 'test_read': False, 'inference_only': True, 'task_optimizer_steps': 0,
        'probe': 'fixed ridge to one-hot classes and audited raw values; alpha=1; no validation tuning',
        'alpha': ALPHA, 'supports': SUPPORTS, 'representation': 'actual head.support(U32), mean over S5, dimension64',
        'not_original_P16': True, 'normalization': 'train objects only; unpenalized intercept',
        'public_control': 'slot/type target prior plus ridge of within-public-group feature and target residuals; train group means only',
        'methods': sources, 'metadata': metadata, 'train_recipients': len(data.rows['train']['ids']),
        'validation_recipients': len(data.rows['val']['ids']), 'train_objects': len(labels['train']['labels']),
        'validation_objects': len(labels['val']['labels']), 'script_sha256': core.digest(__file__),
        'binding_sha256': core.digest(source_run / 'binding.json'),
        'plan_hashes': {split: hashlib.sha256(plan.tobytes()).hexdigest() for split, plan in plans.items()}}
    immutable_write(out / 'config.json', config)
    for split in ('train', 'val'):
        core.save_npz(out / ('labels_' + split + '.npz'), **labels[split])
        core.save_npz(out / ('support_plan_' + split + '.npz'), plan=plans[split], query_ids=np.asarray(data.rows[split]['ids']))
    results = {'config': config, 'methods': {}, 'comparisons': {}, 'test_read': False,
        'interpretation': 'linear accessibility at fixed memory budget; neither full parameter recovery nor causal task use is established by probe alone'}
    for method, source in sources.items():
        if core.digest(source['path']) != source['sha256']:
            raise ValueError('Probe checkpoint changed after freeze')
        model = core.FineTuneModel(binding['legacy_binding']).to(device)
        checkpoint = torch.load(source['path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        features, errors = {}, {}
        for split in ('train', 'val'):
            features[split], errors[split] = memory_features(model, data, split, plans[split], device)
            if len(features[split]) != len(labels[split]['labels']):
                raise ValueError('Memory/active-label row count differs')
        record, predictions, fits = fit_probes(features['train'], features['val'], labels, metadata)
        record['direct_memory_path_max_errors'] = errors
        results['methods'][method] = record
        core.save_npz(out / (method + '_probe_fit.npz'), **fits)
        core.save_npz(out / (method + '_validation_predictions.npz'), **predictions)
        core.write(out / 'result_snapshot.json', results)
        core.emit('memory_probe_complete', method=method,
            balanced_accuracy={field: value['memory64']['balanced_accuracy'] for field, value in record['fields'].items()},
            r2={field: value.get('numeric_regression', {}).get('memory64', {}).get('r2') for field, value in record['fields'].items()})
        del model, checkpoint, features
    for candidate in ('A-MQ', 'A-MQ-Reg'):
        for reference in ('Warm-Native', 'Native-MQ', 'Random-MQ'):
            result = {}
            for field in metadata['fields']:
                ca, re = [results['methods'][name]['fields'][field] for name in (candidate, reference)]
                result[field] = {'balanced_accuracy_delta': ca['memory64']['balanced_accuracy'] - re['memory64']['balanced_accuracy'],
                    'accuracy_delta': ca['memory64']['accuracy'] - re['memory64']['accuracy'],
                    'conditional_balanced_accuracy_delta': ca['memory64_given_public']['balanced_accuracy'] - re['memory64_given_public']['balanced_accuracy']}
                if metadata['numeric_r2_available']:
                    result[field]['numeric_r2_delta'] = ca['numeric_regression']['memory64']['r2'] - re['numeric_regression']['memory64']['r2']
            results['comparisons'][candidate + '_vs_' + reference] = result
    results['status'] = 'COMPLETE'
    core.write(out / 'result_snapshot.json', results)
    core.emit('memory_probe_all_complete', out=str(out), test_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_discovery_collision_v4_4'))
    parser.add_argument('--run', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_collision_multiquery_v4_9'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / 'probe.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.base, args.run, args.out, args.device)
