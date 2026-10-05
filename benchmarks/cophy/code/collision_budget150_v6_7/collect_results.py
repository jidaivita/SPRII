"""Read-only aggregation of three Collision source150 complete-U128 heads (no torch/GPU).

Reads the readout manifest and committed result files. Does not train, evaluate,
select checkpoints, or import old P64-only metrics. Missing results are PENDING.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

VERSION = 'collision-source150-u128-collector-v6.7-1'
READOUT_VERSION = 'monolithic-split-full-U128-readout-v6.6-1'
FULL_ROWS = {'balls': 2000, 'collision': 4000, 'blocktower': 8088}
SOURCE_UPDATES = {'balls': {50: 10950, 100: 21900}, 'collision': {50: 21900, 100: 43800, 150: 65700},
                  'blocktower': {50: 44250, 100: 88500}}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(2**20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def ids_sha(ids):
    return hashlib.sha256(json.dumps(sorted(map(str, ids)), separators=(',', ':')).encode()).hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def reduction(control, treatment):
    return 100*(control-treatment)/control if control is not None and treatment is not None and control > 0 else None


def flag(argv, name):
    if name in argv:
        i = argv.index(name)
        return str(argv[i+1]) if i+1 < len(argv) else None
    for value in argv:
        if str(value).startswith(name+'='): return str(value).split('=', 1)[1]
    return None


def canonical_method(method, kind=None, route=None):
    label = re.sub(r'[^a-z0-9]', '', str(method).lower())
    if kind == 'mono' or 'monolithic' in label or label == 'mono': return 'Monolithic'
    if 'base' in label: return 'Split-Base'
    if 'cross' in label:
        if route == 'all' or 'all' in label: return 'Cross-all'
        if route == 'focal' or 'focal' in label: return 'Cross-focal'
        return 'Cross'
    if 'both' in label: return 'Both'
    return str(method)


def manifest_runs(path):
    manifest = read(path); runs = {}
    def merge(value):
        output = value.get('out') or value.get('path') or value.get('readout_out')
        if not output: raise ValueError('Readout manifest entry has no out/path')
        output = str(Path(output))
        target = runs.setdefault(output, {'out': output})
        for key in ('scene', 'method', 'kind', 'source_epochs', 'checkpoint'):
            raw = value.get(key)
            if raw is None: continue
            raw = int(raw) if key == 'source_epochs' else str(raw)
            if key in target and target[key] != raw: raise ValueError('Conflicting manifest readout '+output+' field '+key)
            target[key] = raw
    for entry in manifest.get('readout_runs', []): merge(entry)
    for task in manifest.get('tasks', []):
        commands = task.get('commands', [])
        if task.get('argv'): commands = [task['argv']] + commands
        for argv in commands:
            if not isinstance(argv, list): continue
            if not any(Path(str(x)).name == 'readout.py' for x in argv): continue
            output = flag(argv, '--out')
            if not output: continue
            values = {'out': output}
            for key in ('scene', 'method', 'kind', 'source_epochs', 'checkpoint'):
                v = flag(argv, '--'+key.replace('_', '-'))
                if v is not None: values[key] = v
            merge(values)
    if not runs: raise ValueError('Manifest contains no readout.py --out commands or readout_runs')
    return list(runs.values())


def checked_result(folder, budget_key, budget):
    marker_path = folder/'complete.json'; result_path = folder/'results.json'
    marker = read(marker_path)
    if marker.get('status') != 'COMPLETE': raise ValueError('Non-COMPLETE result marker: '+str(marker_path))
    if marker.get('version') != READOUT_VERSION or marker.get(budget_key) != budget or marker.get('test_read') is not False:
        raise ValueError('Wrong U128 version/budget/test status: '+str(marker_path))
    if marker.get('results_sha256') != sha(result_path): raise ValueError('Committed result hash differs: '+str(result_path))
    result = read(result_path)
    if result.get('status') != 'COMPLETE' or result.get('test_read') is not False:
        raise ValueError('Result is incomplete or reads test: '+str(result_path))
    return result, marker


def arm_map(arm, label):
    ids = list(map(str, arm.get('ids', []))); values = arm.get('per_recipient_mse', [])
    if len(ids) != len(values) or len(set(ids)) != len(ids): raise ValueError('Invalid per-recipient arm: '+label)
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in values): raise ValueError('Invalid MSE: '+label)
    if arm.get('recipients', len(ids)) != len(ids): raise ValueError('Arm count mismatch: '+label)
    if values:
        declared = arm.get('mse'); actual = mean(map(float, values))
        if declared is None or abs(actual-float(declared)) > 1e-8*(1+abs(actual)):
            raise ValueError('Declared/per-recipient MSE mismatch: '+label)
    return dict(zip(ids, map(float, values)))


def cohort_row(meta, cohort, desired, maps):
    wanted = set(map(str, desired)); matched = maps['matched']; null = maps['null']; wrong = maps['wrong']
    if not wanted <= matched.keys() or not wanted <= null.keys(): raise ValueError('Matched/Null missing requested cohort')
    common = wanted & wrong.keys()
    mm = mean(matched[i] for i in wanted); nn = mean(null[i] for i in wanted)
    cm = mean(matched[i] for i in common); cn = mean(null[i] for i in common); ww = mean(wrong[i] for i in common)
    row = {k:meta.get(k) for k in ('scene','source_method','method','kind','source_epoch','source_updates',
        'source_checkpoint_sha256','head_budget_epochs','selected_head_epoch','head_sha256','head_implementation_sha256')}
    row.update(cohort=cohort,rows=len(wanted),query_ids_sha256=ids_sha(wanted),matched_mse=mm,
        query_preserving_null_mse=nn,history_gain_percent=reduction(nn,mm),wrong_mse=ww,
        wrong_rows=len(common),wrong_coverage=len(common)/len(wanted) if wanted else 0.,
        matched_on_common_wrong=cm,null_on_common_wrong=cn,correct_vs_wrong_reduction_percent=reduction(ww,cm),
        wrong_cohort_ids_sha256=ids_sha(common),representation='complete U128',supports=3,status='COMPLETE',test_read=False)
    return row


def collect_run(entry):
    out = Path(entry['out']); head = out/'S3/learned'; full = head/'fullval'
    meta = dict(entry, status='PENDING', pending_stage='encoding', head_complete=False, fullval_complete=False)
    maps = {}; rows = []; cohorts = {}
    try:
        encpath = out/'encoding_config.json'
        if encpath.exists():
            enc = read(encpath)
            if enc.get('version') != READOUT_VERSION or enc.get('test_read') is not False:
                raise ValueError('Expected new complete-U128 encoding binding')
            for key in ('scene','method','kind','source_epochs'):
                if entry.get(key) is not None and enc.get(key) != entry[key]: raise ValueError('Manifest/encoding mismatch: '+key)
            meta.update(scene=enc['scene'],method=enc['method'],kind=enc['kind'],source_epoch=enc['source_epochs'],
                        source_checkpoint_sha256=enc['checkpoint_sha256'],source_route=enc.get('source_route'),
                        encoding_config_sha256=sha(encpath))
        else:
            meta['source_epoch'] = entry.get('source_epochs')
        meta['source_method'] = canonical_method(meta.get('method'),meta.get('kind'),meta.get('source_route'))
        if meta.get('scene') not in FULL_ROWS or meta.get('source_epoch') not in SOURCE_UPDATES.get(meta.get('scene'), {}):
            raise ValueError('Missing or invalid scene/source budget metadata')
        meta['source_updates'] = SOURCE_UPDATES[meta['scene']][meta['source_epoch']]
        meta['head_budget_epochs'] = 100
        progress = head/'progress.json'
        if progress.exists():
            pr = read(progress)
            meta['head_progress'] = {k:pr.get(k) for k in ('status','epoch','best')}
        if not (out/'encoding_complete.json').exists(): return meta,rows,maps,cohorts
        meta['pending_stage'] = 'head100'
        if not (head/'complete.json').exists(): return meta,rows,maps,cohorts
        result, done = checked_result(head,'epochs',100)
        conf = read(head/'config.json')
        if (conf.get('version') != READOUT_VERSION or conf.get('representation') != 'complete U128'
                or conf.get('supports') != 3 or conf.get('epochs') != 100 or conf.get('encoder_frozen') is not True):
            raise ValueError('Not the new frozen complete-U128 S3 head100')
        if (conf.get('source_epochs') != meta['source_epoch'] or conf.get('scene') != meta['scene']
                or conf.get('source_checkpoint_sha256') != meta.get('source_checkpoint_sha256')):
            raise ValueError('Source binding differs between head/encoding')
        if conf.get('code_sha256') != sha(out/'encoding_complete.json') or result.get('config') != conf:
            raise ValueError('Head config/encoding/result binding differs')
        if done.get('checkpoint_sha256') != sha(head/'selected.pt'): raise ValueError('Selected head changed')
        selected = read(head/'selected_validation.json')
        epoch = int(selected['epoch'])
        if epoch != done.get('selected_epoch') or epoch != result.get('selected_epoch') or not 1 <= epoch <= 100:
            raise ValueError('Selected head epoch differs')
        selection_ids = list(map(str,selected['ids']))
        if len(selection_ids) != 512 or len(set(selection_ids)) != 512: raise ValueError('Expected original 512 selection recipients')
        meta.update(head_complete=True,pending_stage='full_validation',selected_head_epoch=epoch,
            head_sha256=done['checkpoint_sha256'],head_implementation_sha256=conf['implementation_sha256'],
            head_results_path=str(head/'results.json'),head_results_sha256=sha(head/'results.json'),
            null_semantics=conf.get('null_semantics'),head_config_sha256=sha(head/'config.json'))
        # Original base may contain 512 or more rows; always select explicit IDs.
        original_maps = {arm:arm_map(result[arm],'head/'+arm) for arm in ('matched','null','wrong')}
        rows = [cohort_row(meta,'selection512',selection_ids,original_maps)]
        maps,cohorts = original_maps,{'selection512':selection_ids}
        if not (full/'complete.json').exists(): return meta,rows,maps,cohorts
        expanded, committed = checked_result(full,'head_budget',100)
        freeze = read(full/'checkpoint_freeze.json')
        if expanded.get('checkpoint_freeze_sha256') != sha(full/'checkpoint_freeze.json'):
            raise ValueError('Fullval freeze hash differs')
        if (freeze.get('head_sha256') != meta['head_sha256'] or freeze.get('source_checkpoint_sha256') != meta['source_checkpoint_sha256']
                or freeze.get('original_results_sha256') != meta['head_results_sha256']):
            raise ValueError('Fullval selected-source/head binding differs')
        if (expanded.get('scene') != meta['scene'] or expanded.get('source_epochs') != meta['source_epoch']
                or expanded.get('representation') != 'complete U128' or expanded.get('supports') != 3
                or expanded.get('head_budget') != 100 or expanded.get('selected_epoch') != epoch
                or expanded.get('optimizer_steps') != 0):
            raise ValueError('Wrong fullval representation/budget/source or extra optimization')
        maps = {arm:arm_map(expanded[arm],'full/'+arm) for arm in ('matched','null','wrong')}
        full_ids = list(maps['matched'])
        if len(full_ids) != FULL_ROWS[meta['scene']] or expanded.get('full_validation_rows') != len(full_ids):
            raise ValueError('Incomplete full validation cohort')
        if set(maps['null']) != set(full_ids) or not set(maps['wrong']) <= set(full_ids):
            raise ValueError('Fullval arm cohorts differ unexpectedly')
        # Reuse the evaluator's original tolerance, with explicit IDs and no new selection.
        reproduction = {}
        for arm in ('matched','null','wrong'):
            selected_arm = set(selection_ids) & original_maps[arm].keys()
            if (set(selection_ids)&maps[arm].keys()) != selected_arm:
                raise ValueError('Original selection arm coverage changed: '+arm)
            deltas = [abs(maps[arm][i]-original_maps[arm][i]) for i in selected_arm]
            if any(abs(maps[arm][i]-original_maps[arm][i]) > 2e-5+2e-5*abs(original_maps[arm][i]) for i in selected_arm):
                raise ValueError('Original512 per-recipient scores changed: '+arm)
            reproduction[arm] = dict(rows=len(selected_arm),max_absolute_difference=max(deltas,default=0.))
        remaining = [i for i in full_ids if i not in set(selection_ids)]
        cohorts = {'selection512':selection_ids,'remaining':remaining,'full_validation':full_ids}
        rows = [cohort_row(meta,label,ids,maps) for label,ids in cohorts.items()]
        meta.update(status='COMPLETE',pending_stage=None,fullval_complete=True,
            fullval_results_path=str(full/'results.json'),fullval_results_sha256=sha(full/'results.json'),
            reproduction512=reproduction,wrong_donor_domain=expanded.get('wrong_donor_domain'),
            fullval_seconds=expanded.get('seconds'),fullval_checkpoint_freeze_sha256=sha(full/'checkpoint_freeze.json'))
        return meta,rows,maps,cohorts
    except Exception as exc:
        meta.update(status='INVALID',error=str(exc))
        return meta,[],{},{}


def compare(treatment, control, cohort):
    tm,tmap,tc=treatment; cm,cmap,cc=control
    if cohort not in tc or cohort not in cc: return None
    common = set(tc[cohort]) & set(cc[cohort])
    if not common: return None
    a=mean(tmap['matched'][i] for i in common);b=mean(cmap['matched'][i] for i in common)
    return dict(scene=tm['scene'],source_epoch=tm['source_epoch'],cohort=cohort,
        treatment=tm['source_method'],control=cm['source_method'],treatment_source_epoch=tm['source_epoch'],
        control_source_epoch=cm['source_epoch'],head_budget_epochs=100,rows=len(common),
        treatment_cohort_rows=len(tc[cohort]),control_cohort_rows=len(cc[cohort]),
        identical_recipient_cohort=set(tc[cohort])==set(cc[cohort]),query_ids_sha256=ids_sha(common),
        treatment_matched_mse=a,control_matched_mse=b,mse_difference=a-b,
        relative_reduction_percent=reduction(b,a),status='COMPLETE',test_read=False)


def collect(manifest, expected_heads, reference_manifest=None):
    entries=manifest_runs(manifest);results=[];rows=[];internal=[]
    for entry in entries:
        meta,rr,maps,cohorts=collect_run(entry);results.append(meta);rows.extend(rr)
        if meta['status']!='INVALID' and maps: internal.append((meta,maps,cohorts))
    comparisons=[]
    for item in internal:
        meta=item[0]
        for baseline in ('Monolithic','Split-Base'):
            if meta['source_method']==baseline: continue
            matches=[x for x in internal if x[0]['scene']==meta['scene'] and x[0]['source_epoch']==meta['source_epoch']
                     and x[0]['source_method']==baseline]
            if len(matches)>1: raise ValueError('Ambiguous duplicate control for '+str((meta['scene'],meta['source_epoch'],baseline)))
            if matches:
                for cohort in ('selection512','remaining','full_validation'):
                    row=compare(item,matches[0],cohort)
                    if row:comparisons.append(row)
    curves=[]; references=[]
    if reference_manifest:
        for entry in manifest_runs(reference_manifest):
            identity=dict(entry)
            encpath=Path(entry['out'])/'encoding_config.json'
            if encpath.exists(): identity.update(read(encpath))
            method=canonical_method(identity.get('method'),identity.get('kind'),identity.get('source_route'))
            if identity.get('scene')!='collision' or identity.get('source_epochs')!=100 or method not in ('Monolithic','Split-Base','Cross-all'):
                continue
            meta,rr,maps,cohorts=collect_run(entry);references.append(meta)
            if meta['status']=='INVALID' or not maps:continue
            matches=[x for x in internal if x[0]['scene']=='collision' and x[0]['source_epoch']==150 and x[0]['source_method']==method]
            if len(matches)>1:raise ValueError('Duplicate source150 budget-curve treatment: '+method)
            if matches:
                for cohort in ('selection512','remaining','full_validation'):
                    row=compare(matches[0],(meta,maps,cohorts),cohort)
                    if row:
                        row['comparison_type']='within-method source100-to150 budget curve; not a matched-budget method comparison'
                        row['control_source_updates']=SOURCE_UPDATES['collision'][100]
                        row['treatment_source_updates']=SOURCE_UPDATES['collision'][150]
                        curves.append(row)
        canonical=[r['source_method'] for r in references]
        if len(set(canonical))!=len(canonical):raise ValueError('Ambiguous duplicate source100 reference rows')
    by_status={state:sum(x['status']==state for x in results) for state in ('COMPLETE','PENDING','INVALID')}
    issues=[]
    if len(entries)!=expected_heads: issues.append(f'Manifest has {len(entries)} unique heads; expected {expected_heads}')
    # Only three new Collision source150 cells belong to this execution.
    matrix={(r.get('scene'),r.get('source_epoch'),r.get('source_method')) for r in results}
    expected={('collision',150,m) for m in ('Monolithic','Split-Base','Cross-all')}
    missing=sorted(expected-matrix);extra=sorted(matrix-expected,key=str)
    if missing:issues.append('Missing frozen matrix cells: '+str(missing))
    if extra:issues.append('Unexpected matrix cells: '+str(extra))
    if len(matrix)!=len(entries):issues.append('Duplicate scene/source-budget/canonical-method cells')
    status='INVALID' if by_status['INVALID'] or issues else ('COMPLETE' if by_status['COMPLETE']==expected_heads else 'PENDING')
    return dict(status=status,version=VERSION,readout_version=READOUT_VERSION,generated_at=time.time(),
        manifest=str(Path(manifest).resolve()),manifest_sha256=sha(manifest),expected_heads=expected_heads,
        discovered_heads=len(entries),head100_complete=sum(r.get('head_complete',False) for r in results),
        fullval_complete=by_status['COMPLETE'],counts=by_status,issues=issues,runs=results,rows=rows,
        comparisons=comparisons,source_budget_curves=curves,source100_references=references,
        source_budget_curve_status=('NOT_REQUESTED' if not reference_manifest else 'INVALID' if any(r['status']=='INVALID' for r in references) else 'COMPLETE' if len(references)==3 and all(r['status']=='COMPLETE' for r in references) and len(curves)==9 else 'PENDING'),
        reference_manifest=str(Path(reference_manifest).resolve()) if reference_manifest else None,
        reference_manifest_sha256=sha(reference_manifest) if reference_manifest else None,
        scope='New complete-U128 S3, frozen source, fresh100-epoch heads; validation only; old P64 scores excluded',
        comparison_boundary='Collision source150 matched-budget comparison (65700 source updates each); optional source100-to150 curves reuse only complete-U128 S3/head100 results and remain within each method.',
        seed_note='Single training seed; no cross-seed claim',test_read=False,optimizer_steps=0)


def number(x, places=5):
    return '—' if x is None else f'{x:.{places}f}'


def markdown(result):
    lines=['# Collision source150 complete-U128 comparison','',
        f"Status: **{result['status']}**. Head100 complete {result['head100_complete']}/{result['expected_heads']}; full validation {result['fullval_complete']}/{result['expected_heads']}.",
        '',result['scope']+'.','',result['comparison_boundary'],'',
        'Source100-to150 budget curves: **'+result['source_budget_curve_status']+'**.','']
    if result['issues']:lines += ['Protocol/receipt issues: '+ '; '.join(result['issues']),'']
    for cohort in ('full_validation','selection512','remaining'):
        lines += ['## '+{'full_validation':'Full validation','selection512':'Original 512 selection recipients','remaining':'Additional validation recipients'}[cohort], '',
            '| Scene | Source epochs | Method | Rows | Matched | Null | Wrong | Wrong coverage | Gain vs Mono | Gain vs Split-Base |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        rr=[r for r in result['rows'] if r['cohort']==cohort]
        for row in sorted(rr,key=lambda r:(r['scene'],r['source_epoch'],r['source_method'])):
            gains={c['control']:c['relative_reduction_percent'] for c in result['comparisons'] if c['scene']==row['scene']
                   and c['source_epoch']==row['source_epoch'] and c['treatment']==row['source_method'] and c['cohort']==cohort}
            lines.append(f"| {row['scene']} | {row['source_epoch']} | {row['source_method']} | {row['rows']} | {number(row['matched_mse'])} | {number(row['query_preserving_null_mse'])} | {number(row['wrong_mse'])} | {100*row['wrong_coverage']:.1f}% | {number(gains.get('Monolithic'),2)} | {number(gains.get('Split-Base'),2)} |")
        if not rr:lines.append('| PENDING | | | | | | | | | |')
        lines+=['','Gains are percentage reductions in Matched MSE against a source-budget-matched control. Wrong can cover fewer rows; its effect uses Matched on the same Wrong cohort (recorded in JSON).','']
    if result['source_budget_curves']:
        lines+=['## Source100 to source150 budget curves','',
            '| Method | Cohort | Rows | Source100 MSE | Source150 MSE | Reduction (%) |',
            '|---|---|---:|---:|---:|---:|']
        for row in result['source_budget_curves']:
            lines.append(f"| {row['treatment']} | {row['cohort']} | {row['rows']} | {number(row['control_matched_mse'])} | {number(row['treatment_matched_mse'])} | {number(row['relative_reduction_percent'],2)} |")
        lines+=['','These are within-method budget curves. Both sides use separately trained 100-epoch U128 heads and fixed original512 selection.','']
    lines+=['## Pending or invalid runs','']
    pending=[r for r in result['runs'] if r['status']!='COMPLETE']
    for r in pending:
        lines.append(f"- {r.get('scene')} / source{r.get('source_epoch')} / {r.get('source_method')}: {r['status']} — {r.get('error') or r.get('pending_stage')}")
    if not pending:lines.append('None.')
    lines+=['','All checkpoints and head selections remain fixed. This collector does not run inference or training, change cohorts, or reuse the previous P64-only scores.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--expected-heads',type=int,default=3);p.add_argument('--require-complete',action='store_true')
    p.add_argument('--reference-manifest',help='Optional existing U128 manifest supplying only Collision source100 references')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result=collect(args.manifest,args.expected_heads,args.reference_manifest)
    write(out/'summary.json',result)
    md=out/'summary.md';tmp=md.with_name(md.name+'.tmp.'+str(os.getpid()));tmp.write_text(markdown(result));os.replace(tmp,md)
    status={k:result[k] for k in ('status','version','generated_at','manifest','manifest_sha256','expected_heads','discovered_heads',
        'head100_complete','fullval_complete','counts','issues','test_read','optimizer_steps')}
    status.update(summary_path=str(out/'summary.json'),summary_sha256=sha(out/'summary.json'),markdown_sha256=sha(md))
    write(out/'status.json',status)
    if result['status']=='COMPLETE':write(out/'complete.json',status)
    elif (out/'complete.json').exists():
        # An old success marker must never conceal a now-invalid binding/read.
        write(out/'stale_previous_complete.json',read(out/'complete.json'));(out/'complete.json').unlink()
    print(json.dumps(status,ensure_ascii=False,allow_nan=False))
    if args.require_complete and result['status']!='COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
