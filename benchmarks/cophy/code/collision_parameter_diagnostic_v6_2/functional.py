"""Fixed-head, focal Wrong-1 assay for Collision v6.2 P64 representations.

Only validation is scored. All sampling uses audited metadata and frozen visual
presence, never target trajectories, predictions, or a parameter probe. The
original Correct S3 plan is preserved. No optimizer or checkpoint selection.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

VERSION = 'collision-v6.2-focal-wrong1-functional-v1'
FIELDS = ('mass', 'friction', 'restitution')
ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Frozen diagnostic binding changed: ' + str(path))
    else:
        write(path, value)


def npz_write(path, **arrays):
    path = Path(path); tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(tmp, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def modules(args):
    # The existing full-validation module imports its head as `readout`.
    # A fresh process binds that name to the explicitly selected source file.
    core = load_module(args.readout_module, 'readout')
    full = load_module(args.fullval_module, 'collision_functional_fullval')
    if core.VERSION != 'latent-relation-v6.2-frozen-P64-pose-prefix-readout':
        raise ValueError('Requires the existing v6.2 P64 head')
    return core, full


def specs(args):
    result = {}
    for specification in args.method:
        name, path = specification.split('=', 1)
        if not name or '/' in name or name in result:
            raise ValueError('Method names must be unique simple labels')
        result[name] = Path(path)
    if len(result) < 2:
        raise ValueError('Supply at least two methods for a shared-plan comparison')
    return result


def ready(args, methods):
    missing = []
    for name, folder in methods.items():
        for path in (folder/'codes_complete.json', folder/'S3/learned/complete.json'):
            if not path.is_file() or read(path).get('status') != 'COMPLETE':
                missing.append(dict(method=name, path=str(path)))
    prepared = Path(args.prepared)/'prepared.json'
    if not prepared.is_file():
        missing.append(dict(method=None, path=str(prepared)))
    if missing:
        write(Path(args.out)/'waiting.json', dict(status='WAITING', version=VERSION,
              missing=missing, time=time.time(), optimizer_steps=0, test_read=False))
        print(json.dumps(dict(status='WAITING', missing=missing)), flush=True)
        return False
    return True


def field_provenance(args, base_manifest):
    prepath = Path(args.preflight); pre = read(prepath)
    if sha(prepath) != base_manifest['preflight_sha256']:
        raise ValueError('Parameter metadata preflight differs from the trained task')
    item = pre['artifacts']['raw_audit']; audit = Path(item['path'])
    if sha(audit) != item['sha256'] or tuple(read(audit)['fields']) != FIELDS:
        raise ValueError('Audited Collision field order is not mass/friction/restitution')
    return dict(preflight=str(prepath), preflight_sha256=sha(prepath),
                raw_audit=str(audit), raw_audit_sha256=sha(audit), fields=list(FIELDS),
                category_ids_used_only_for_equality=True)


def identity(core, full, folder, binding):
    head = folder/'S3/learned'; complete = read(head/'complete.json')
    config = read(head/'config.json'); selected = read(head/'selected_validation.json')
    result = read(head/'results.json')
    if complete.get('epochs') != 100 or config['scene'] != 'collision':
        raise ValueError('Wait for the fixed100-epoch Collision head')
    if config['version'] != core.VERSION or config['reference'] != 'learned' or config['supports'] != 3:
        raise ValueError('Wrong head identity')
    if config.get('test_read') is not False or result.get('test_read') is not False:
        raise ValueError('Expected validation-only head provenance')
    if config['base_sha256'] != sha(Path(binding['base'])/'manifest.json'):
        raise ValueError('Base manifest changed')
    full.check_files(config['input_sha256'])
    if config['code_sha256'] != sha(folder/'codes_complete.json'):
        raise ValueError('Frozen P64 cache changed')
    ckpath = head/'selected.pt'; ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    if ck['config'] != config or ck['epoch'] != selected['epoch'] or ck['epoch'] != result['selected_epoch']:
        raise ValueError('Selected checkpoint/receipts disagree')
    return ck, selected, dict(readout=str(folder), checkpoint=str(ckpath),
        checkpoint_sha256=sha(ckpath), selected_epoch=ck['epoch'], head_training_epochs=100,
        source_codes_sha256=sha(folder/'codes_complete.json'),
        source_checkpoint_sha256=read(folder/'codes_complete.json')['source_sha256'],
        selected_receipt_sha256=sha(head/'selected_validation.json'))


def public_key(part, ident, slot):
    # No frame, position, target, or parameter values are passed to a model.
    return (int(slot), json.dumps(part['known_type'][ident][slot], sort_keys=True),
            json.dumps(part.get('gravity', {}).get(ident), sort_keys=True))


def build_plan(core, part, ids, mask, correct, seen_by_method, seed):
    history_ids = list(map(str, part['all_ids'])); index = {q:i for i,q in enumerate(history_ids)}
    shared_seen = np.logical_and.reduce([x > 0 for x in seen_by_method.values()])
    groups = defaultdict(lambda: defaultdict(list)); metadata_missing = []
    for ident in history_ids:
        if ident not in part['physical'] or ident not in part['presence'] or ident not in part['known_type']:
            metadata_missing.append(ident); continue
        for slot in np.flatnonzero(np.asarray(part['presence'][ident]) > 0):
            if not shared_seen[index[ident], slot]:
                continue
            label = tuple(part['physical'][ident][slot])
            if len(label) != 3:
                raise ValueError('Unexpected number of physical factors')
            groups[public_key(part, ident, slot)][label].append(index[ident])
    for bylabel in groups.values():
        for label, pool in bylabel.items():
            bylabel[label] = np.asarray(sorted(pool), dtype=np.int64)
    n, slots = mask.shape
    wrong = np.full((n, slots, 3, 3), -1, np.int64)
    eligible = np.zeros((n, slots, 3), bool)
    alternatives = np.zeros((n, slots, 3), np.int64)
    candidates = np.zeros((n, slots, 3), np.int64)
    selected_label = np.full((n, slots, 3, 3), -1, np.int64)
    for i, ident in enumerate(ids):
        if ident not in index:
            raise ValueError('Recipient outside the frozen history-ID domain')
        for slot in np.flatnonzero(mask[i] > 0):
            truth = tuple(part['physical'][ident][slot]); public = public_key(part, ident, slot)
            supports = correct[i, slot]
            if len(set(supports.tolist())) != 3 or index[ident] in supports:
                raise ValueError('Original Correct plan is not independent S3')
            for d in supports:
                donor = history_ids[d]
                if public_key(part, donor, slot) != public or tuple(part['physical'][donor][slot]) != truth:
                    raise ValueError('Original Correct support metadata is inconsistent')
            for factor in range(3):
                options = []
                for other, pool in groups.get(public, {}).items():
                    differences = [j for j in range(3) if other[j] != truth[j]]
                    if differences != [factor]:
                        continue
                    valid = pool[pool != index[ident]]
                    candidates[i, slot, factor] += len(valid)
                    if len(valid) >= 3:
                        options.append((other, valid))
                options.sort(key=lambda x: x[0]); alternatives[i, slot, factor] = len(options)
                if not options:
                    continue
                rng = np.random.default_rng(core.seedof(f'{VERSION}:{seed}:{ident}:{slot}:{factor}'))
                other, pool = options[int(rng.integers(len(options)))]
                chosen = rng.choice(pool, 3, replace=False)
                wrong[i, slot, factor] = chosen; selected_label[i, slot, factor] = other
                eligible[i, slot, factor] = True
    return dict(ids=np.asarray(ids), history_ids=np.asarray(history_ids), correct=correct,
        wrong=wrong, eligible=eligible, alternative_counts=alternatives,
        candidate_counts=candidates, selected_wrong_labels=selected_label,
        common_visible=shared_seen, metadata_missing_ids=np.asarray(metadata_missing, dtype=str))


@torch.inference_mode()
def per_object(model, data, device, indices, plan, batch_size):
    row = data.data['val']; values = []
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start+batch_size]; donor = plan[start:start+batch_size]
        tensor = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
        supports = row['codes'][donor, np.arange(data.slots)[None, :, None]]
        supports = supports*row['mask'][ix, :, None, None]
        predicted = model(tensor(row['q'][ix]), tensor(row['det'][ix]), tensor(row['mask'][ix]), tensor(supports))
        errors = (predicted-tensor(row['target'][ix])).square().mean((1, 3))
        if not torch.isfinite(errors).all():
            raise ValueError('Nonfinite fixed-head prediction errors')
        values.append(errors.cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, data.slots), np.float32)


def event_metrics(errors, mask, focal):
    rows = np.arange(len(focal)); focal_mask = np.eye(mask.shape[1])[focal]
    others = mask*(1-focal_mask); count = others.sum(1)
    return dict(focal=errors[rows, focal], scene=(errors*mask).sum(1)/mask.sum(1),
                partner=np.divide((errors*others).sum(1), count,
                    out=np.full(len(count), np.nan), where=count > 0))


def aggregate(correct, wrong, recipient_indices):
    valid = np.isfinite(correct) & np.isfinite(wrong)
    correct, wrong, ix = correct[valid], wrong[valid], recipient_indices[valid]
    if not len(ix):
        return dict(recipient_count=0, focal_count=0, correct_mse=None, wrong_mse=None, paired_delta=None)
    recipients = np.unique(ix)
    # Each recipient contributes once, after averaging its eligible focal objects.
    c = np.asarray([correct[ix == i].mean() for i in recipients])
    w = np.asarray([wrong[ix == i].mean() for i in recipients])
    return dict(recipient_count=len(recipients), focal_count=len(ix),
        correct_mse=float(c.mean()), wrong_mse=float(w.mean()),
        paired_delta=float((w-c).mean()), delta_sign='Wrong minus Correct; positive means correct parameter helps',
        relative_error_increase_percent=float(100*(w.mean()-c.mean())/c.mean()) if c.mean() > 0 else None,
        focal_weighted_delta=float((wrong-correct).mean()))


def evaluate_method(args, core, full, name, folder, binding, shared, method_binding):
    out = Path(args.out)/name; out.mkdir(parents=True, exist_ok=True)
    freeze = dict(version=VERSION, method=name, **method_binding,
        manifest_sha256=sha(Path(args.out)/'manifest.json'), plan_sha256=sha(Path(args.out)/'plan.npz'),
        head_frozen=True, encoder_frozen=True, optimizer_steps=0, test_read=False)
    immutable(out/'checkpoint_freeze.json', freeze)
    if (out/'complete.json').exists():
        marker = read(out/'complete.json')
        if marker['checkpoint_freeze_sha256'] != sha(out/'checkpoint_freeze.json') or marker['results_sha256'] != sha(out/'results.json'):
            raise ValueError('Completed diagnostic changed')
        return read(out/'results.json')
    a = argparse.Namespace(scene='collision', readout=str(folder), reference='learned', supports=3)
    data = full.FullData(a, binding); row = data.data['val']
    if not np.array_equal(data.val_plan, shared['correct']):
        raise ValueError('All methods must receive exactly the same Correct donor plan')
    ck, selected, actual_binding = identity(core, full, folder, binding)
    if actual_binding != method_binding:
        raise ValueError('Checkpoint changed while preparing diagnostic')
    model = core.Head(data.dims, data.det_dims, data.support_dims, data.horizon).to(args.device)
    model.load_state_dict(ck['model'], strict=True); model.requires_grad_(False); model.eval()
    correct = per_object(model, data, args.device, np.arange(len(row['ids'])), shared['correct'], args.batch_size)
    scene = (correct*row['mask']).sum(1)/row['mask'].sum(1)
    actual = dict(ids=[row['ids'][i] for i in data.selection], per_recipient_mse=scene[data.selection].tolist())
    reproduction = full.compare_saved(actual, selected, 'selected Correct')
    events = np.argwhere(shared['eligible']); qi, focal, factor = events.T if len(events) else (np.array([], int),)*3
    altered = shared['correct'][qi].copy()
    altered[np.arange(len(events)), focal] = shared['wrong'][qi, focal, factor]
    wrong = per_object(model, data, args.device, qi, altered, args.batch_size)
    cm = event_metrics(correct[qi], row['mask'][qi], focal)
    wm = event_metrics(wrong, row['mask'][qi], focal)
    selected_mask = np.zeros(len(row['ids']), bool); selected_mask[data.selection] = True
    cohorts = {'all4000':np.ones(len(events), bool), 'selection512':selected_mask[qi], 'remaining3488':~selected_mask[qi]}
    table = {}
    for j, field in enumerate(FIELDS):
        table[field] = {}
        for cohort, valid in cohorts.items():
            take = valid & (factor == j)
            table[field][cohort] = {metric:aggregate(cm[metric][take], wm[metric][take], qi[take]) for metric in cm}
    payload = dict(ids=np.asarray(row['ids']), correct_per_object=correct, recipient_index=qi,
        focal_slot=focal, parameter_index=factor, wrong_per_object=wrong,
        mask=row['mask'], selection_indices=data.selection)
    for metric in cm:
        payload[f'correct_{metric}'] = cm[metric]; payload[f'wrong_{metric}'] = wm[metric]
        payload[f'delta_{metric}'] = wm[metric]-cm[metric]
    npz_write(out/'per_recipient.npz', **payload)
    result = dict(status='COMPLETE', version=VERSION, method=name, table=table,
        correct_full_mse=float(scene.mean()), reproduction=reproduction, events=len(events), recipients=len(row['ids']),
        checkpoint_freeze_sha256=sha(out/'checkpoint_freeze.json'),
        per_recipient_sha256=sha(out/'per_recipient.npz'), optimizer_steps=0, test_read=False,
        interpretation='Paired functional reliance on one mismatched parameter in a focal external-history channel; not a simulator parameter intervention.')
    write(out/'results.json', result)
    write(out/'complete.json', dict(status='COMPLETE', results_sha256=sha(out/'results.json'),
        checkpoint_freeze_sha256=sha(out/'checkpoint_freeze.json'), optimizer_steps=0, test_read=False))
    return result


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    methods = specs(args)
    if not ready(args, methods):
        return 75
    core, full = modules(args); prepared = Path(args.prepared)/'prepared.json'; binding = read(prepared)
    if binding['scene'] != 'collision' or binding.get('test_read') is not False or len(binding['query_ids']) != 4000:
        raise ValueError('Requires the existing full4000 Collision validation preparation')
    full.check_files(binding['files'])
    metadata = full.metadata(binding); base_manifest = read(Path(binding['base'])/'manifest.json')
    provenance = field_provenance(args, base_manifest)
    method_bindings = {}; seen = {}; correct = None; selection = None
    for name, folder in methods.items():
        ck, selected, method_bindings[name] = identity(core, full, folder, binding); del ck, selected
        data = full.FullData(argparse.Namespace(scene='collision', readout=str(folder), reference='learned', supports=3), binding)
        row = data.data['val']; seen[name] = row['donor_seen'].copy()
        if correct is None:
            correct = data.val_plan.copy(); ids = row['ids']; mask = row['mask'].copy(); selection = data.selection.copy()
        elif not np.array_equal(correct, data.val_plan) or ids != row['ids'] or not np.array_equal(mask, row['mask']):
            raise ValueError('Methods do not share recipient inputs and Correct support plans')
        del data
    shared = build_plan(core, metadata, ids, mask, correct, seen, args.seed)
    count = mask.astype(bool).sum(); visibility = {}
    for name, visible in seen.items():
        correct_visible = visible[correct, np.arange(mask.shape[1])[None, :, None]] > 0
        visibility[name] = dict(presence_sha256=array_sha(visible),
            visible_history_objects=int((visible > 0).sum()),
            active_correct_slots_with_any_unseen_support=int(((~correct_visible).any(-1) & (mask > 0)).sum()))
    manifest = dict(status='FROZEN', version=VERSION, seed=args.seed, supports=3,
        prepared_sha256=sha(prepared), parameter_provenance=provenance,
        code_sha256={str(p):sha(p) for p in (Path(__file__), Path(args.readout_module), Path(args.fullval_module))},
        query_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        correct_plan_sha256=array_sha(correct), wrong_plan_sha256=array_sha(shared['wrong']),
        all_validation_recipients=len(ids), active_focal_objects=int(count),
        model_visibility=visibility, visibility_identical=all(np.array_equal(next(iter(seen.values())), x) for x in seen.values()),
        wrong_pool='intersection of visibly present cached AB objects across specified models; audited metadata only',
        correct_visibility_policy='Original Correct plan unchanged; unseen supports counted, not silently resampled',
        support_rule='one uniformly sampled alternate parameter class with at least3 independent donors; then3 without replacement; all other physical factors, slot, publictype and recorded gravity unchanged',
        metadata_missing_history_ids=shared['metadata_missing_ids'].tolist(),
        coverage={field:dict(eligible_focal=int(shared['eligible'][:,:,j].sum()), total_focal=int(count),
            fraction=float(shared['eligible'][:,:,j].sum()/count),
            eligible_recipients=int(shared['eligible'][:,:,j].any(1).sum())) for j,field in enumerate(FIELDS)},
        targets_used_for_sampling=False, checkpoint_selection=False, optimizer_steps=0, test_read=False)
    immutable(out/'manifest.json', manifest)
    if not (out/'plan.npz').exists():
        npz_write(out/'plan.npz', **shared, selection_indices=selection)
    else:
        with np.load(out/'plan.npz', allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k], v) for k,v in shared.items()):
                raise ValueError('Frozen Wrong-1 plan differs')
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED', coverage=manifest['coverage'])), flush=True); return 0
    torch.set_num_threads(args.threads)
    results = {name:evaluate_method(args,core,full,name,folder,binding,shared,method_bindings[name]) for name,folder in methods.items()}
    summary = dict(status='COMPLETE', version=VERSION, methods=results,
        coverage=manifest['coverage'], manifest_sha256=sha(out/'manifest.json'),
        same_plan_for_all_models=True, primary_aggregation='average eligible focal objects within recipient, then average recipients',
        primary_delta='Wrong minus Correct; positive means correct parameter helps',
        partner_definition='all other active objects, averaged; absent partner yields no partner statistic',
        uncertainty='This diagnostic gives paired point estimates; it does not add seeds or significance claims.',
        optimizer_steps=0, test_read=False)
    write(out/'summary.json', summary)
    print(json.dumps(dict(status='COMPLETE', methods=list(results), coverage=manifest['coverage'])), flush=True)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', action='append', required=True, metavar='NAME=READOUT_DIRECTORY')
    p.add_argument('--prepared', required=True, help='Directory containing the existing Collision fullval prepared.json')
    p.add_argument('--readout-module', default=str(ROOT/'source/latent_v6_2_sig02/readout.py'))
    p.add_argument('--fullval-module', default=str(ROOT/'source/latent_v6/fullval_readout.py'))
    p.add_argument('--preflight', default=str(ROOT/'prepared_v3/collision/training_preflight.json'))
    p.add_argument('--out', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=20260912); p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with (Path(args.out)/'functional.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main(args))
