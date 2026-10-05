"""Collect the fixed Collision source100/head100 comparison; CPU arithmetic only.

Reads completed validation artifacts, never models, raw data, optimizers or test.
Outputs compact JSON and gzip, with no per-recipient arrays copied into them.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

VERSION = 'collision-base100-v6.5-fixed-comparison-1'
CORE = 'latent-relation-v6.2-frozen-P64-pose-prefix-readout'
FULLVAL = 'latent-v6.2-fullval-fixed-P64-readout-1'
NAMES = ('Base100', 'Cross100-focal', 'Cross100-all')
PAIRS = ((2, 0), (1, 0), (2, 1))  # treatment, control
FIELDS = ('mass', 'friction', 'restitution')
REPLICATES = 2000
SEED = 20260912


class Waiting(Exception):
    pass


def read(path):
    path = Path(path)
    if not path.is_file():
        raise Waiting(str(path))
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def done(value, label):
    if value.get('status') != 'COMPLETE':
        raise Waiting(label + ' is not COMPLETE')
    check(value.get('test_read') is False, label + ' must be validation-only')


def array_digest(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_bytes(content)
    os.replace(temporary, path)


def artifact(spec, prepared, files):
    """Validate completed source, fixed head, selected512 reproduction and probe."""
    full, source, readout = spec['full'], spec['source'], spec['readout']
    paths = dict(result=full/'results.json', source=source/'complete.json',
        head=readout/'S3/learned/complete.json', config=readout/'S3/learned/config.json',
        original=readout/'S3/learned/results.json', selected=readout/'S3/learned/selected_validation.json',
        codes=readout/'codes_complete.json', probe=readout/'probes.json',
        freeze=full/'checkpoint_freeze.json', complete=full/'complete.json')
    d = {key: read(path) for key, path in paths.items()}
    for key in ('result', 'source', 'head', 'codes', 'probe', 'complete'):
        done(d[key], str(paths[key]))
    r, src, cfg, codes, freeze = (d[k] for k in ('result', 'source', 'config', 'codes', 'freeze'))
    check(src['epochs'] == 100 and src['selected_epoch'] == 100, 'Source must be fixed epoch100')
    check(src['steps'] == 43800, 'Expected 100 x 438 source updates')
    check(src.get('coordinate_labels_read') is False, 'Source must not read coordinate labels')
    check(src['method'] == spec['method'], 'Wrong source method')
    if 'route' in spec:
        check(src.get('route') == spec['route'], 'Wrong Cross100 route')
    check(d['head']['epochs'] == cfg['epochs'] == freeze['head_budget'] == 100, 'Unequal head budget')
    check(cfg['version'] == CORE and cfg['scene'] == 'collision' and cfg['reference'] == 'learned'
          and cfg['supports'] == 3 and cfg['encoder_frozen'] is True, 'Wrong head/input protocol')
    check(cfg.get('memory_mode', 'init-only') == 'init-only', 'Do not mix in per-step heads')
    check('initialization' in cfg['head_history_access'], 'Expected original init-only head')
    check(codes['version'] == CORE and codes['representation'] == 'P64', 'Wrong representation')
    check(src['checkpoint_sha256'] == codes['source_sha256'], 'Source/code binding differs')
    check(cfg['code_sha256'] == sha(paths['codes']) == freeze['source_codes_sha256'], 'Changed P cache receipt')
    check(d['probe']['source_codes_sha256'] == sha(paths['codes']), 'Probe used another source cache')
    check(r['version'] == FULLVAL and r['scene'] == 'collision' and r['supports'] == 3,
          'Wrong full-validation identity')
    check(r['full_validation_rows'] == 4000 and r['selection_rows'] == 512, 'Wrong cohort sizes')
    check(r['matched']['ids'] == prepared['query_ids'], 'Full4000 recipient order differs')
    check(d['selected']['ids'] == prepared['selection_ids'], 'Original512 selection IDs differ')
    check(r['optimizer_steps'] == 0 and freeze['encoder_frozen'] and freeze['head_frozen'],
          'Evaluation must be frozen and read-only')
    check(r['selected_epoch'] == freeze['selected_epoch'] == d['original']['selected_epoch']
          == d['selected']['epoch'] == d['complete']['selected_epoch'], 'Selected head differs')
    check(d['complete']['results_sha256'] == sha(paths['result']), 'Changed fullval results')
    check(r['checkpoint_freeze_sha256'] == sha(paths['freeze']), 'Changed checkpoint freeze')
    check(freeze['selected_receipt_sha256'] == sha(paths['selected'])
          and freeze['original_results_sha256'] == sha(paths['original']), 'Changed selection receipts')
    check(freeze['prepared_sha256'] == prepared['_sha256'], 'Changed prepared4000 inputs')
    for arm in ('selected', 'matched', 'null', 'wrong'):
        rep = r['reproduction'][arm]
        check(rep['matched_ids_exact'] is True, 'Original512 reproduction IDs differ: ' + arm)
        expected = d['selected'] if arm == 'selected' else d['original'][arm]
        current = r['matched' if arm == 'selected' else arm]
        lookup = dict(zip(current['ids'], current['per_recipient_mse']))
        check(len(expected['ids']) == rep['recipients'], 'Reproduction row count differs')
        check(len(expected['ids']) == len(set(expected['ids'])), 'Repeated original recipient')
        actual = np.asarray([lookup[q] for q in expected['ids']], np.float64)
        saved = np.asarray(expected['per_recipient_mse'], np.float64)
        check(np.allclose(actual, saved, atol=2e-5, rtol=2e-5), 'Original512 numerical mismatch: ' + arm)
        check(np.isfinite(rep['max_absolute_error']), 'Missing reproduction error')
    arm_data = {}
    for arm in ('matched', 'null', 'wrong'):
        row = r[arm]; values = np.asarray(row['per_recipient_mse'], np.float64)
        check(len(row['ids']) == len(values) == len(set(row['ids'])), 'Invalid recipient scores')
        check(np.isfinite(values).all() and (values >= 0).all(), 'Nonfinite/negative MSE')
        check(set(row['ids']).issubset(prepared['query_ids']), 'Foreign validation recipient')
        check(np.isclose(values.mean(), row['mse'], atol=1e-8, rtol=1e-6), 'MSE does not match recipient mean')
        arm_data[arm] = dict(zip(row['ids'], values))
    check(r['null']['ids'] == prepared['query_ids'], 'Null must use all4000')
    check(np.isclose(r['wrong_coverage'], len(arm_data['wrong']) / 4000), 'Wrong coverage differs')
    p = d['probe']
    check(p['scene'] == 'collision' and p['ridge_alpha'] == 1 and p['normalization'] == 'train only',
          'Different probe protocol')
    check(p['fields'] == list(FIELDS) and 'P' in p['representations'], 'Wrong physical fields')
    for field in FIELDS:
        for metric in ('r2', 'mse', 'accuracy', 'balanced_accuracy'):
            check(np.isfinite(p['representations']['P'][field][metric]), 'Invalid probe statistic')
    files.update({str(path): sha(path) for path in paths.values()})
    plan_path = full/'plans.npz'
    if not plan_path.is_file():
        raise Waiting(str(plan_path))
    check(r['plan_sha256'] == sha(plan_path), 'Changed frozen support plans')
    with np.load(plan_path, allow_pickle=False) as z:
        check(z['ids'].tolist() == prepared['query_ids'], 'Plan IDs differ')
        plans = {arm: z[arm].copy() for arm in ('matched', 'wrong')}
    files[str(plan_path)] = sha(plan_path)
    info = dict(source_epochs=src['epochs'], source_steps=src['steps'], head_epochs=100,
        selected_head_epoch=r['selected_epoch'], source_method=src['method'], route=src.get('route'),
        source_checkpoint_sha256=src['checkpoint_sha256'], source_seconds=src.get('extension_seconds'),
        original512_reproduction=r['reproduction'], wrong_coverage=r['wrong_coverage'],
        probe=dict(train_objects=p['train_objects'], val_objects=p['val_objects'],
            fields=p['representations']['P'], representation=p['representation'],
            preflight_sha256=p['preflight_sha256'], ridge_alpha=p['ridge_alpha'], normalization=p['normalization']))
    return dict(info=info, scores=arm_data, plans=plans, config=cfg, freeze=freeze, probe=p)


def paired_summary(values, arms, seed):
    """One shared recipient resample per replicate for every method and arm."""
    # Shape: method x recipient x arm. Bootstrap ratios use means, not mean ratios.
    n = values.shape[1]
    if n == 0:
        return dict(rows=0, methods={}, comparisons={}, reason='No common eligible recipients')
    mean = values.mean(1)
    rng = np.random.default_rng(seed)
    boot = np.empty((REPLICATES, len(NAMES), len(arms)), np.float64)
    for start in range(0, REPLICATES, 50):
        size = min(50, REPLICATES-start)
        indices = rng.integers(0, n, size=(size, n))
        boot[start:start+size] = values[:, indices, :].mean(2).transpose(1, 0, 2)

    def interval(point, draws):
        return dict(estimate=float(point), ci95_percentile=[float(x) for x in np.quantile(draws, [.025, .975])])

    matched, null = arms.index('matched'), arms.index('null')
    methods = {}
    for i, name in enumerate(NAMES):
        row = {arm + '_mse': float(mean[i, j]) for j, arm in enumerate(arms)}
        row['history_gain_percent'] = float(100*(mean[i, null]-mean[i, matched])/mean[i, null])
        if 'wrong' in arms:
            wrong = arms.index('wrong')
            row['correct_vs_wrong_percent'] = float(100*(mean[i, wrong]-mean[i, matched])/mean[i, wrong])
        methods[name] = row
    comparisons = {}
    for treatment, control in PAIRS:
        row = {}
        for j, arm in enumerate(arms):
            absolute = mean[control, j]-mean[treatment, j]
            draws = boot[:, control, j]-boot[:, treatment, j]
            row[arm] = dict(mse_reduction=interval(absolute, draws),
                mse_reduction_percent=interval(100*absolute/mean[control, j], 100*draws/boot[:, control, j]))
        gp = (mean[control, null]-mean[control, matched])/mean[control, null]
        gt = (mean[treatment, null]-mean[treatment, matched])/mean[treatment, null]
        bp = (boot[:, control, null]-boot[:, control, matched])/boot[:, control, null]
        bt = (boot[:, treatment, null]-boot[:, treatment, matched])/boot[:, treatment, null]
        row['history_gain_difference_percentage_points'] = interval(100*(gt-gp), 100*(bt-bp))
        if 'wrong' in arms:
            j = arms.index('wrong')
            # Positive = the treatment's Wrong-Correct MSE gap is larger.
            dg = (mean[treatment, j]-mean[treatment, matched])-(mean[control, j]-mean[control, matched])
            db = (boot[:, treatment, j]-boot[:, treatment, matched])-(boot[:, control, j]-boot[:, control, matched])
            row['wrong_any_specificity_difference_in_differences'] = interval(dg, db)
        comparisons[NAMES[treatment] + '_vs_' + NAMES[control]] = row
    return dict(rows=n, methods=methods, comparisons=comparisons)


def collect(args):
    root = args.root
    base = args.base_root or root/'collision_base100_v6_5'
    adapt = args.adapt_root or root/'collision_adapt_v6_4'
    prep_path = args.prepared or root/'latent_v6_2/fullval_inputs/collision/prepared.json'
    prepared = read(prep_path)
    done(prepared, str(prep_path))
    check(prepared['scene'] == 'collision' and prepared['version'] == FULLVAL, 'Wrong prepared scene')
    ids, selected = prepared['query_ids'], prepared['selection_ids']
    check(len(ids) == len(set(ids)) == 4000 and len(selected) == len(set(selected)) == 512,
          'Wrong validation domain')
    check(set(selected).issubset(ids), 'Selection outside validation')
    prepared['_sha256'] = sha(prep_path)
    files = {str(prep_path): prepared['_sha256'], str(Path(__file__).resolve()): sha(__file__)}
    specs = {
        'Base100': dict(full=base/'fullval', source=base/'source', readout=base/'readout', method='Base'),
        **{f'Cross100-{route}': dict(full=adapt/'fullval'/f'Cross100-{route}',
            source=adapt/'routing/runs'/route, readout=adapt/'readouts'/f'Cross100-{route}',
            method='Cross', route=route) for route in ('focal', 'all')},
    }
    results = {name: artifact(specs[name], prepared, files) for name in NAMES}
    reference = results['Base100']
    for name, current in results.items():
        for field in ('head_implementation_sha256', 'evaluation_implementation_sha256', 'prepared_sha256'):
            check(current['freeze'][field] == reference['freeze'][field], 'Unequal evaluator/head: ' + name)
        for field in ('lr', 'batch_size', 'head_width', 'support_dims', 'current_input', 'head_history_access',
                      'null_dropout', 'seed', 'selection_rows', 'selection_ids_sha256', 'input_sha256', 'base_sha256'):
            check(current['config'][field] == reference['config'][field], 'Unequal readout protocol: ' + field)
        for arm in ('matched', 'wrong'):
            check(np.array_equal(current['plans'][arm], reference['plans'][arm]),
                  'Support plans differ across methods: ' + name + '/' + arm)
        for field in ('preflight_sha256', 'train_objects', 'val_objects', 'fields'):
            check(current['probe'][field] == reference['probe'][field], 'Probe audit/domain differs: ' + field)
    common_wrong = set(ids)
    for r in results.values():
        common_wrong &= set(r['scores']['wrong'])
    selected_set = set(selected)
    cohorts = dict(full4000=ids, original512=selected,
                   remaining3488=[q for q in ids if q not in selected_set])
    output = {}
    for ci, (cohort, group) in enumerate(cohorts.items()):
        common = [q for q in group if q in common_wrong]

        def table(domain, arms):
            return np.asarray([[[results[name]['scores'][arm][q] for arm in arms] for q in domain]
                               for name in NAMES], np.float64).reshape(len(NAMES), len(domain), len(arms))

        output[cohort] = dict(ids_sha256=array_digest(group),
            all_recipients=paired_summary(table(group, ('matched', 'null')), ('matched', 'null'), SEED+10*ci),
            common_wrong_coverage=len(common)/len(group), common_wrong_ids_sha256=array_digest(common),
            common_wrong_recipients=paired_summary(table(common, ('matched', 'null', 'wrong')),
                ('matched', 'null', 'wrong'), SEED+10*ci+1))
    probes = {name: results[name]['info']['probe'] for name in NAMES}
    probe_differences = {}
    for treatment, control in PAIRS:
        t, c = NAMES[treatment], NAMES[control]
        probe_differences[t + '_vs_' + c] = {
            field: dict(r2_difference=probes[t]['fields'][field]['r2']-probes[c]['fields'][field]['r2'],
                balanced_accuracy_difference_percentage_points=100*(probes[t]['fields'][field]['balanced_accuracy']
                    -probes[c]['fields'][field]['balanced_accuracy'])) for field in FIELDS}
    return dict(status='COMPLETE', version=VERSION, collected_at=time.time(), scene='collision',
        source_budget_epochs=100, source_updates=43800, head_budget_epochs=100, supports=3,
        representation='Frozen P64; original init-only head; current3 -> future12 xyz',
        metric='Equal recipient mean of xyz MSE averaged over12 future frames and active objects',
        methods={name: results[name]['info'] for name in NAMES}, cohorts=output,
        probe_differences=probe_differences,
        bootstrap=dict(unit='recipient', replicates=REPLICATES, seed=SEED, interval='95% percentile',
            paired_across='all methods and arms use the same bootstrap recipient draw within each cohort',
            limitation='Conditional on these fixed checkpoints; not variability across training seeds; no multiplicity correction'),
        interpretation=dict(primary='Cross100-all versus Base100: same source epochs/updates and head protocol',
            route_control='Cross100-all versus Cross100-focal: shared Cross50 parent and matched continuation budget',
            budget='Equal epochs/recipient updates, not equal FLOPs; Cross has an additional prediction objective',
            signs='Positive MSE reduction means treatment is better; percentages use control MSE as denominator',
            development='Validation development result, including original512 selection and a separate remaining3488 report',
            probe='Single AB episode per object; R2 uses audited real parameter values; not S3 pooled memory',
            attribution='This tests Cross-only and its routing adaptation, not Both; probe changes do not prove mediation'),
        checks=dict(full4000_ids_equal=True, original512_reproduced=True, source100_head100_equal=True,
                    original_init_only_head_equal=True, support_plans_equal=True, probe_audit_counts_equal=True),
        files_sha256=files, test_read=False, optimizer_steps=0, copied_per_recipient_arrays=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))))
    p.add_argument('--base-root', type=Path)
    p.add_argument('--adapt-root', type=Path)
    p.add_argument('--prepared', type=Path, help='prepared.json file, not directory')
    p.add_argument('--out', type=Path, help='Output directory; defaults BASE_ROOT/comparison')
    args = p.parse_args()
    out = args.out or (args.base_root or args.root/'collision_base100_v6_5')/'comparison'
    try:
        result = collect(args)
    except Waiting as error:
        value = dict(status='WAITING', version=VERSION, dependency=str(error), test_read=False, optimizer_steps=0)
        atomic(out/'waiting.json', (json.dumps(value, indent=2)+'\n').encode())
        print(json.dumps(value), flush=True)
        return 75
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    atomic(out/'results.json', encoded)
    atomic(out/'results.json.gz', gzip.compress(encoded, mtime=0))
    marker = dict(status='COMPLETE', version=VERSION, results_sha256=sha(out/'results.json'),
        gzip_sha256=sha(out/'results.json.gz'), test_read=False, optimizer_steps=0)
    atomic(out/'complete.json', (json.dumps(marker, indent=2)+'\n').encode())
    waiting = out/'waiting.json'
    if waiting.exists():
        waiting.unlink()
    print(json.dumps(dict(marker, out=str(out),
        primary=result['cohorts']['full4000']['all_recipients']['comparisons']['Cross100-all_vs_Base100']['matched'])), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
