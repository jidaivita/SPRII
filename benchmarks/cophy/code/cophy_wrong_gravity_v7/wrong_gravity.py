"""Fixed Blocktower wrong-global-gravity assay; no training or test access.

prepare/spec/collect use only the standard library. evaluate imports the exact
bound readout implementation and changes its in-memory validation support plan.
Existing source/head files and all existing evaluation files are read-only.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

VERSION = 'cophy-v7-blocktower-wrong-gravity-1'
SUPPORTED = {'cophy-complete-v7-full-state-readout',
             'monolithic-split-full-U128-readout-v6.6-1',
             'cophy-v7-legacy-supervised-frozen-U32-readout'}


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def checked(path):
    p = Path(path)
    if any(re.search(r'(^|[_.-])test([_.-]|$)', s.lower()) for s in p.parts):
        raise ValueError('Test artifact path prohibited: ' + str(p))
    return p


def read(path):
    x = json.loads(checked(path).read_text())
    if isinstance(x, dict) and (x.get('test_read') is True or x.get('split') == 'test'):
        raise ValueError('Test artifact prohibited')
    return x


def sha(path):
    h = hashlib.sha256()
    with checked(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def write(path, x):
    p = checked(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.pending.' + str(os.getpid()))
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); os.replace(tmp, p)


def immutable(path, x):
    p = checked(path)
    if p.exists() and read(p) != x: raise ValueError('Different existing frozen artifact: ' + str(p))
    write(p, x)


def seeded(*parts):
    return random.Random(int(digest(list(parts)), 16))


def decode_key(key):
    fields = json.loads(key)
    if len(fields) != 6 or not isinstance(fields[0], int):
        raise ValueError('Expected audited slot/type/mass/friction/gravity_x/gravity_y key')
    gravity = tuple(float(v) for v in fields[-2:])
    if not all(math.isfinite(v) for v in gravity): raise ValueError('Nonfinite gravity')
    return canonical(fields[:4]), gravity


def metadata(path):
    m = read(path)
    if m.get('scene') not in ('blocktower', 'blocktower-normal') or m.get('test_read') is not False:
        raise ValueError('Expected audited Blocktower validation metadata')
    if m.get('prefix') != 3 or m.get('slots') != 4:
        raise ValueError('Blocktower prefix/slots changed')
    part = m['splits']['val']; all_ids = list(map(str, part['all_ids']))
    ids = list(map(str, part.get('full_query_ids', part['query_ids'])))
    selection = list(map(str, part.get('selection_query_ids', part['query_ids'][:512])))
    if len(ids) != 8088 or len(selection) != 512 or len(set(ids)) != 8088:
        raise ValueError('Expected full8088 and original512 validation cohorts')
    if not set(ids) <= set(all_ids) or not set(selection) <= set(ids): raise ValueError('Invalid IDs')
    pools = {}; groups = {}
    for key, indices in part['candidate_groups'].items():
        base, gravity = decode_key(key)
        if (base, gravity) in pools: raise ValueError('Duplicate semantic candidate key')
        if indices != sorted(set(indices)) or any(i < 0 or i >= len(all_ids) for i in indices):
            raise ValueError('Candidate indices must be sorted, unique and in the validation donor domain')
        pools[(base, gravity)] = tuple(indices); groups[(base, gravity)] = key
        # This confirms the global field whenever the donor is an eligible query.
        for i in indices:
            donor = all_ids[i]
            if donor in part['gravity'] and tuple(map(float, part['gravity'][donor])) != gravity:
                raise ValueError('Candidate group contradicts audited global gravity')
    return m, part, all_ids, ids, selection, pools, groups


def prepare(args):
    m, part, all_ids, ids, selection, pools, groups = metadata(args.metadata_manifest)
    lookup = {q: i for i, q in enumerate(all_ids)}
    gravities = sorted({g for _, g in pools})
    if len(gravities) < 2: raise ValueError('No varying global gravity to evaluate')
    rows = []
    for ident in ids:
        active = [i for i, flag in enumerate(part['presence'][ident]) if flag > 0]
        truth = tuple(map(float, part['gravity'][ident])); bases = {}
        for slot in active:
            key = part['candidate_keys'][ident][slot]; base, gravity = decode_key(key)
            fields = json.loads(key)
            if fields[0] != slot or gravity != truth or list(fields[2:4]) != list(part['physical'][ident][slot]):
                raise ValueError('Recipient key does not match audited physical/global fields')
            bases[slot] = base
        valid = []
        for gravity in gravities:
            if gravity == truth: continue
            if active and all(len(pools.get((bases[s], gravity), ())) -
                              int(lookup[ident] in pools.get((bases[s], gravity), ())) >= 3 for s in active):
                valid.append(gravity)
        row = dict(id=ident, active_slots=active, true_gravity=list(truth), eligible_gravities=len(valid))
        if not valid:
            rows.append(dict(row, covered=False, reason='NO_SINGLE_WRONG_GRAVITY_WITH_S3_ALL_ACTIVE_SLOTS'))
            continue
        gravity = seeded(VERSION, args.seed, ident, 'gravity').choice(valid)
        donors, keys = [None]*4, [None]*4
        for slot in active:
            key = (bases[slot], gravity)
            available = [i for i in pools[key] if i != lookup[ident]]
            chosen = seeded(VERSION, args.seed, ident, slot, 'supports').sample(available, 3)
            donors[slot] = [all_ids[i] for i in chosen]; keys[slot] = groups[key]
        rows.append(dict(row, covered=True, wrong_gravity=list(gravity), donor_ids=donors, donor_keys=keys))
    out = dict(version=VERSION, status='COMPLETE', scene='blocktower', split='val', seed=args.seed,
               supports=3, slots=4, query_frames=3, full_validation_rows=8088, test_read=False,
               metadata_manifest=str(Path(args.metadata_manifest).resolve()), metadata_sha256=sha(args.metadata_manifest),
               preflight_sha256=m.get('preflight_sha256'), all_ids=all_ids, query_ids=ids,
               selection_query_ids=selection, gravity_values=[list(g) for g in gravities],
               metadata_covered=sum(r['covered'] for r in rows),
               policy='one wrong gravity per recipient; same gravity for every active slot and all S3 supports; preserve slot/public type/mass/friction; independent donors; no outcome-dependent selection',
               visibility='audited candidate_groups already require AB visibility; each evaluated checkpoint additionally checks its own frozen cache, reports missing rows, and never resamples',
               rows=rows)
    out['plan_sha256'] = digest(out)
    dest = Path(args.out); immutable(dest/'manifest.json', out)
    immutable(dest/'complete.json', dict(status='COMPLETE', version=VERSION, manifest_sha256=sha(dest/'manifest.json'),
                                       metadata_covered=out['metadata_covered'], full_validation_rows=8088, test_read=False))
    print(json.dumps({k: out[k] for k in ('status', 'metadata_covered', 'full_validation_rows', 'plan_sha256')}))


def load_plan(path):
    p = read(path); unbound = dict(p); expected = unbound.pop('plan_sha256')
    if p.get('version') != VERSION or p.get('status') != 'COMPLETE' or digest(unbound) != expected:
        raise ValueError('Wrong-Gravity plan binding changed')
    if sha(p['metadata_manifest']) != p['metadata_sha256']: raise ValueError('Audited relation metadata changed')
    return p


def module(path):
    name = '_wrong_gravity_readout_' + sha(path)[:12]
    spec = importlib.util.spec_from_file_location(name, checked(path)); result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result; spec.loader.exec_module(result)
    if result.VERSION not in SUPPORTED: raise ValueError('Unreviewed readout interface: ' + result.VERSION)
    return result


def paired(ids, matched, wrong, selection):
    def cohort(chosen):
        if not chosen: return dict(rows=0, matched_mse=None, wrong_gravity_mse=None, delta_wrong_minus_matched=None)
        c = math.fsum(matched[q] for q in chosen)/len(chosen); w = math.fsum(wrong[q] for q in chosen)/len(chosen)
        return dict(rows=len(chosen), matched_mse=c, wrong_gravity_mse=w,
                    delta_wrong_minus_matched=w-c, correct_gain_vs_wrong_percent=100*(w-c)/w if w > 0 else None,
                    wrong_increase_vs_correct_percent=100*(w-c)/c if c > 0 else None)
    selected = set(selection)
    return dict(all=cohort(ids), original512=cohort([q for q in ids if q in selected]),
                remaining7576=cohort([q for q in ids if q not in selected]))


def evaluate(args):
    import numpy as np
    import torch
    plan = load_plan(args.plan); core = module(args.readout_code)
    readout = checked(args.readout_out); head_dir = readout/'S3/learned'
    done = read(head_dir/'complete.json'); conf = read(head_dir/'config.json')
    if done.get('status') != 'COMPLETE' or done.get('epochs') != 100 or conf.get('version') != core.VERSION:
        raise ValueError('Wait for this fixed head100')
    supervised = args.kind == 'supervised'
    enc_path = readout/('codes_complete.json' if supervised else 'encoding_complete.json'); enc = read(enc_path)
    if conf.get('code_sha256') != sha(enc_path): raise ValueError('Frozen head encoding binding mismatch')
    bound = enc if supervised else enc['binding']
    code_binding = bound.get('implementation_sha256') if supervised else conf.get('implementation_sha256')
    if code_binding != sha(args.readout_code): raise ValueError('Wrong bound readout implementation')
    if supervised and not args.base: raise ValueError('Supervised readout requires its full --base')
    base = Path(args.base if supervised else bound['base'])
    if conf['base_sha256'] != sha(base/'manifest.json'): raise ValueError('Frozen base manifest changed')
    budget = bound['source_budget'] if supervised else bound['source_epochs']
    if budget != args.source_budget: raise ValueError('Wrong source budget')
    for path, h in conf.get('input_sha256', {}).items():
        if sha(path) != h: raise ValueError('Bound head input changed')
    if not supervised and conf['normalization_sha256'] != sha(readout/'normalization.json'):
        raise ValueError('Frozen normalization changed')
    old = read(args.original_results)
    if old.get('status') != 'COMPLETE' or old.get('test_read') is not False: raise ValueError('Need completed validation result')
    saved_ids = list(map(str, old['matched']['ids']))
    if len(saved_ids) != 8088 or set(saved_ids) != set(plan['query_ids']): raise ValueError('Full8088 Matched results required')
    ck = torch.load(head_dir/'selected.pt', map_location='cpu', weights_only=False)
    selected = read(head_dir/'selected_validation.json')
    if ck['config'] != conf or ck['epoch'] != selected['epoch'] or ck['epoch'] != old['selected_epoch']:
        raise ValueError('Fixed selected checkpoint changed')
    if done.get('checkpoint_sha256') and done['checkpoint_sha256'] != sha(head_dir/'selected.pt'):
        raise ValueError('Completed checkpoint digest changed')
    probe_path = checked(args.probe or readout/'probes.json'); probe = read(probe_path)
    if probe.get('status') != 'COMPLETE': raise ValueError('Wait for this checkpoint parameter probe')
    probe_encoding = probe.get('source_codes_sha256') if supervised else probe.get('encoding_sha256')
    if probe_encoding != sha(enc_path): raise ValueError('Parameter probe belongs to another frozen encoding')
    # Keep exactly the original module's Data and scoring implementation.
    dataargs = argparse.Namespace(out=str(readout), scene='blocktower', supports=3, reference='learned',
                                  base=args.base, prepared=args.prepared, device=args.device)
    data = core.Data(dataargs)
    if not supervised and not args.prepared: raise ValueError('Self readout requires shared fullval --prepared')
    row = data.data['val']; ids = list(map(str, row['ids'])); part = data.manifest['splits']['val']
    if len(ids) != 8088 or set(ids) != set(plan['query_ids']): raise ValueError('Data is not full validation')
    all_ids = list(map(str, part['all_ids'])); index = {q: i for i, q in enumerate(all_ids)}
    if all_ids != plan['all_ids']: raise ValueError('History domain/order differs')
    selection_ids = [ids[i] for i in data.selection]
    if selection_ids != plan['selection_query_ids']: raise ValueError('Original512 selection differs')
    if (data.dims, data.slots, data.horizon) != (3, 4, 27): raise ValueError('Wrong Blocktower prediction task')
    model = core.Head(data.dims, data.det_dims, data.support_dims, data.horizon).to(args.device)
    model.load_state_dict(ck['model'], strict=True); model.requires_grad_(False); model.eval()
    score = core.evaluate if supervised else core.score_head
    frozen = dict(version=VERSION, plan_sha256=plan['plan_sha256'], row_id=args.row_id, source_budget=budget,
                  readout_out=str(readout), readout_version=core.VERSION, readout_code=str(args.readout_code),
                  readout_code_sha256=sha(args.readout_code), wrapper_sha256=sha(__file__),
                  encoding_sha256=sha(enc_path), head_sha256=sha(head_dir/'selected.pt'), selected_epoch=ck['epoch'],
                  head_budget=100, supports=3, parameter_probe=str(probe_path), parameter_probe_sha256=sha(probe_path),
                  original_results=str(args.original_results), original_results_sha256=sha(args.original_results),
                  source_checkpoint=bound.get('source_checkpoint', bound.get('checkpoint')),
                  source_checkpoint_sha256=bound.get('source_sha256', bound.get('checkpoint_sha256')),
                  test_read=False, optimizer_steps=0)
    immutable(Path(args.out)/'binding.json', frozen)
    finish = Path(args.out)/'complete.json'
    if finish.exists():
        if read(finish)['results_sha256'] != sha(Path(args.out)/'results.json'): raise ValueError('Changed final result')
        print(json.dumps(dict(status='COMPLETE', already_complete=True, row_id=args.row_id))); return
    started = time.monotonic()
    actual = score(model, data, args.device, arm='matched')
    prior = dict(zip(saved_ids, map(float, old['matched']['per_recipient_mse'])))
    now = dict(zip(actual['ids'], actual['per_recipient_mse']))
    if set(now) != set(prior): raise ValueError('Matched reproduction IDs changed')
    a = np.asarray([now[q] for q in ids]); b = np.asarray([prior[q] for q in ids])
    if not np.allclose(a, b, atol=2e-5, rtol=2e-5): raise ValueError('Frozen Matched reproduction failed')
    planrows = {r['id']: r for r in plan['rows']}; wrong_plan = np.zeros_like(data.val_plan)
    original_plan = data.val_plan; supported = []; missing = []
    for i, ident in enumerate(ids):
        r = planrows[ident]; active = list(map(int, np.flatnonzero(row['mask'][i] > 0)))
        if active != r['active_slots']: raise ValueError('Public recipient active objects differ')
        if not r['covered']:
            missing.append(dict(id=ident, reason=r['reason'])); continue
        valid = True
        for slot in active:
            donors = r['donor_ids'][slot]
            if len(donors) != 3 or len(set(donors)) != 3 or ident in donors: raise ValueError('Invalid independent S3')
            chosen = [index[q] for q in donors]; wrong_plan[i, slot] = chosen
            if not (np.asarray(row['donor_seen'])[chosen, slot] > 0).all(): valid = False
            if not (np.asarray(row['donor_seen'])[original_plan[i, slot], slot] > 0).all(): valid = False
        if valid: supported.append(i)
        else: missing.append(dict(id=ident, reason='SELECTED_MATCHED_OR_WRONG_DONOR_NOT_VISIBLE_IN_THIS_FROZEN_CACHE'))
    try:
        data.val_plan = wrong_plan
        result = score(model, data, args.device, np.asarray(supported, np.int64), arm='matched')
    finally:
        data.val_plan = original_plan
    wrong = dict(zip(result['ids'], map(float, result['per_recipient_mse'])))
    matched = {q: prior[q] for q in result['ids']}
    current_hash = digest({k: hashlib.sha256(np.ascontiguousarray(row[k]).tobytes()).hexdigest()
                           for k in ('q', 'det', 'mask', 'target')})
    output = dict(status='COMPLETE', version=VERSION, scene='blocktower', row_id=args.row_id,
                  binding_sha256=sha(Path(args.out)/'binding.json'), plan_sha256=plan['plan_sha256'],
                  source_budget=budget, head_budget=100, selected_epoch=ck['epoch'], input_target_sha256=current_hash,
                  current_input_order_sha256=digest(ids), support_dims=data.support_dims,
                  full_validation_rows=8088, metadata_covered=plan['metadata_covered'], cache_covered=len(supported),
                  missing=missing, selection_query_ids=selection_ids,
                  matched_reproduction=dict(rows=len(ids), max_absolute_difference=float(np.max(np.abs(a-b))),
                                            original512_rows=len(selection_ids)),
                  cohorts=paired(result['ids'], matched, wrong, selection_ids),
                  per_recipient=[dict(id=q, matched_mse=matched[q], wrong_gravity_mse=wrong[q],
                                      delta_wrong_minus_matched=wrong[q]-matched[q]) for q in result['ids']],
                  matched_plan_sha256=digest([[all_ids[j] for j in original_plan[i, s]]
                                              for i in range(len(ids)) for s in range(data.slots) if row['mask'][i, s] > 0]),
                  parameter_probe_reused=frozen['parameter_probe'], parameter_probe_sha256=frozen['parameter_probe_sha256'],
                  seconds=time.monotonic()-started, test_read=False, optimizer_steps=0)
    write(Path(args.out)/'results.json', output)
    write(finish, dict(status='COMPLETE', version=VERSION, results_sha256=sha(Path(args.out)/'results.json'),
                       plan_sha256=plan['plan_sha256'], row_id=args.row_id, cache_covered=len(supported), test_read=False, optimizer_steps=0))
    print(json.dumps(dict(status='COMPLETE', row_id=args.row_id, cohorts=output['cohorts'])))


def collect(args):
    spec = read(args.spec); plan = load_plan(spec['plan']); runs = {}; waiting = []
    for entry in spec['evaluations']:
        path = Path(entry['out'])/'results.json'; marker = path.with_name('complete.json')
        if not marker.exists(): waiting.append(entry['id']); continue
        if read(marker).get('results_sha256') != sha(path): raise ValueError('Changed complete result')
        r = read(path)
        if r.get('status') != 'COMPLETE' or r['plan_sha256'] != plan['plan_sha256'] or r['row_id'] != entry['id']:
            raise ValueError('Wrong evaluation binding')
        runs[entry['id']] = (entry, r)
    groups = defaultdict(list)
    for ident, (entry, r) in runs.items(): groups[(entry['family'], entry['source_budget'])].append(ident)
    reports = []
    for (family, budget), names in sorted(groups.items()):
        expected = [e['id'] for e in spec['evaluations'] if e['family'] == family and e['source_budget'] == budget]
        if set(names) != set(expected): continue
        values = [runs[n][1] for n in names]
        # Across different families input representation widths may differ;
        # within a family, public input/target and existing Matched plan must match.
        for key in ('input_target_sha256', 'current_input_order_sha256', 'matched_plan_sha256'):
            if len({r[key] for r in values}) != 1: raise ValueError('Within-family comparison mismatch: ' + key)
        common = set.intersection(*[{p['id'] for p in r['per_recipient']} for r in values])
        common_ids = [q for q in plan['query_ids'] if q in common]
        report = dict(family=family, source_budget=budget, common_rows=len(common_ids), full_validation_rows=8088,
                      common_ids_sha256=digest(common_ids), methods={})
        for name in names:
            r = runs[name][1]; rows = {x['id']: x for x in r['per_recipient']}
            report['methods'][runs[name][0]['role']] = dict(row_id=name, own_cache_covered=r['cache_covered'],
                cohorts=paired(common_ids, {q: rows[q]['matched_mse'] for q in common_ids},
                               {q: rows[q]['wrong_gravity_mse'] for q in common_ids}, plan['selection_query_ids']),
                parameter_probe=r['parameter_probe_reused'], parameter_probe_sha256=r['parameter_probe_sha256'])
        reports.append(report)
    # Also provide a single intersection across every completed run. It is
    # labelled provisional until all scheduled rows exist; no cross-family ranking.
    all_common = set.intersection(*[{x['id'] for x in r['per_recipient']} for _, r in runs.values()]) if runs else set()
    all_ids = [q for q in plan['query_ids'] if q in all_common]
    global_results = {}
    for name, (_, r) in runs.items():
        rows = {x['id']: x for x in r['per_recipient']}
        global_results[name] = paired(all_ids, {q: rows[q]['matched_mse'] for q in all_ids},
                                      {q: rows[q]['wrong_gravity_mse'] for q in all_ids}, plan['selection_query_ids'])
    output = dict(version=VERSION, status='WAITING' if waiting else 'COMPLETE', expected=len(spec['evaluations']),
                  completed=len(runs), waiting=waiting, plan_sha256=plan['plan_sha256'], per_family_budget=reports,
                  across_all_runs_common_rows=len(all_ids), across_all_runs_common_is_provisional=bool(waiting),
                  across_all_runs_common_ids_sha256=digest(all_ids), across_all_runs=global_results,
                  claim='Functional sensitivity to a coherent wrong global gravity at inference; not a Random-global source-training attribution control.',
                  test_read=False, optimizer_steps=0)
    write(Path(args.out)/'summary.json', output)
    if not waiting: write(Path(args.out)/'complete.json', dict(status='COMPLETE', summary_sha256=sha(Path(args.out)/'summary.json'), test_read=False))
    print(json.dumps(dict(status=output['status'], expected=output['expected'], completed=output['completed'])))


def specs(args):
    source = read(args.presentation); root = Path(args.root); code = root/'source/cophy_wrong_gravity_v7/wrong_gravity.py'
    outroot = root/'cophy_complete_v7/wrong_gravity'; plan = outroot/'prepared/manifest.json'
    entries = []
    for row in source['rows']:
        if row['scene'] != 'blocktower' or row['role'] not in ('Native','Structure','A','Random'): continue
        readout = Path(row['readout_out']); supervised = row['family'].lower() in ('supervised', 'cophynet')
        if supervised: core = root/'source/cophy_complete_v7/legacy/supervised_readout.py'
        elif '/monolithic_jepa_v6_6_1/' in str(readout): core = root/'source/monolithic_jepa_v6_6_1/readout.py'
        elif '/monolithic_jepa_v6_6/' in str(readout): core = root/'source/monolithic_jepa_v6_6/readout.py'
        else: core = root/'source/cophy_complete_v7/common/readout.py'
        original = row.get('fullval_results', str(readout/'S3/learned/fullval/results.json'))
        marker = row.get('fullval_marker', str(Path(original).with_name('complete.json')))
        out = outroot/'evaluations'/row['id']; probe = row.get('probe_path', str(readout/'probes.json'))
        argv = [args.python, str(code), 'evaluate', '--plan', str(plan), '--readout-code', str(core),
                '--readout-out', str(readout), '--kind', 'supervised' if supervised else 'self',
                '--original-results', original, '--probe', probe, '--row-id', row['id'],
                '--source-budget', str(row['source_budget']), '--out', str(out), '--device', 'cuda:{gpu}']
        if supervised: argv += ['--base', str(root/'cophy_complete_v7/prepared/blocktower_supervised_full')]
        else: argv += ['--prepared', str(root/'latent_v6_2/fullval_inputs/blocktower')]
        entries.append(dict(id=row['id'], family=row['family'], role=row['role'], source_budget=row['source_budget'],
                            out=str(out), argv=argv, resource='gpu-evaluation-only',
                            depends_on=[str(plan.with_name('complete.json')), marker, probe],
                            marker=str(out/'complete.json'), optimizer_steps=0))
    if len(entries) != 45: raise ValueError('Expected 36 self + 9 supervised Blocktower readouts, got ' + str(len(entries)))
    spec = dict(version=VERSION, status='DRAFT_NOT_SUBMITTED', source_presentation_sha256=sha(args.presentation),
                code_sha256=sha(__file__), root=str(root), plan=str(plan), expected_evaluations=len(entries),
                prepare=dict(id='blocktower-wrong-gravity-prepare', resource='cpu',
                    argv=[args.python, str(code), 'prepare', '--metadata-manifest', str(root/'xep_discovery_blocktower_v6/manifest.json'),
                          '--out', str(plan.parent), '--seed', '20260912'], marker=str(plan.with_name('complete.json'))),
                evaluations=entries, collect=dict(argv=[args.python, str(code), 'collect', '--spec',
                    str(root/'source/cophy_wrong_gravity_v7/task_spec.json'), '--out', str(outroot/'comparison')]),
                no_new_training=True, test_read=False)
    write(args.out, spec)
    print(json.dumps(dict(status=spec['status'], evaluations=len(entries))))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--metadata-manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20260912); p.set_defaults(func=prepare)
    p = sub.add_parser('evaluate')
    for name in ('plan','readout-code','readout-out','original-results','row-id','out'): p.add_argument('--'+name, required=True)
    p.add_argument('--source-budget', type=int, required=True); p.add_argument('--kind', choices=['self','supervised'], required=True)
    p.add_argument('--base'); p.add_argument('--prepared'); p.add_argument('--probe'); p.add_argument('--device', default='cuda:0')
    p.set_defaults(func=evaluate)
    p = sub.add_parser('collect'); p.add_argument('--spec', required=True); p.add_argument('--out', required=True); p.set_defaults(func=collect)
    p = sub.add_parser('spec'); p.add_argument('--presentation', required=True); p.add_argument('--out', required=True)
    p.add_argument('--root', default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))); p.add_argument('--python', default='python3'); p.set_defaults(func=specs)
    args = parser.parse_args()
    if args.command in ('prepare','evaluate','collect'):
        folder = checked(args.out); folder.mkdir(parents=True, exist_ok=True)
        with (folder/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); args.func(args)
    else: args.func(args)


if __name__ == '__main__': main()
