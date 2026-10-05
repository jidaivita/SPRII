"""Read-only validation presentation collector; standard library only.

CLI: python collect.py --manifest presentation_manifest.json --out SUMMARY_DIR

Required row keys: family, scene, role, source_budget, source_checkpoint,
readout_out, profile, status. status is a scheduling hint, never result proof.
Relative paths resolve against the manifest directory. Optional artifact keys:
source_marker, head_dir, fullval_results, fullval_marker, probe_path,
encoding_marker, checkpoint_freeze, selected_validation, source_checkpoint_sha256.
Reference rows (Known/Query-only) need no fictional source/probe. Set required=false
for exploratory rows. No test files, model weights, training or goal APIs are read.

Comparisons require the same family/scene/source budget/profile/head budget/S,
recipient cohort and actual readout version. A profile object may explicitly list
compatible_readout_versions to authorize a reviewed compatibility bridge. It
does not silently merge different experimental recipes. Source version/routes
are retained separately. No Oracle-space percentage or best-result picking.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from collections import Counter

VERSION = 'cophy-v7-presentation-collector-1'
FULL_ROWS = {'balls': 2000, 'collision': 4000, 'blocktower': 8088}
GOOD = {'COMPLETE', 'PASS', 'RELEASED'}
METRICS = ('r2', 'mse', 'accuracy', 'balanced_accuracy')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def check_path(path):
    path = Path(path)
    if any(re.search(r'(^|[_.-])test([_.-]|$)', p.lower()) for p in path.parts):
        raise ValueError('Refusing test artifact path: ' + str(path))
    return path


def load(path):
    path = check_path(path)
    with path.open() as stream:
        value = json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    if isinstance(value, dict):
        if value.get('test_read') is True or value.get('split') == 'test':
            raise ValueError('Artifact declares test access: '+str(path))
    return value


def atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(text); os.replace(temporary, path)


def write(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reduction(control, treatment):
    return 100*(control-treatment)/control if number(control) and number(treatment) and control > 0 else None


def role_name(role):
    key = re.sub(r'[^a-z0-9]', '', str(role).lower())
    if key in ('native', 'monolithic', 'mono'): return 'Native'
    if key in ('structure', 'splitbase', 'base', 'structural'): return 'Structure'
    if key in ('known', 'knownparameters', 'gtparameters', 'oracle'): return 'Known'
    if key in ('query', 'queryonly', 'qonly'): return 'Query-only'
    return str(role)


def existing(paths):
    return next((p for p in paths if p is not None and Path(p).exists()), None)


def source_marker_candidates(checkpoint, budget):
    if checkpoint is None: return []
    parent = checkpoint.parent
    return [parent/f'checkpoint_{budget}_complete.json', parent/f'complete_{budget}.json',
            parent/f'budget_{budget}_complete.json', parent/'complete.json']


def marker(path, issues, name, result_path=None):
    if path is None or not path.exists():
        issues.append(name+'缺少完成收据'); return None
    value = load(path)
    if not isinstance(value, dict) or value.get('status') not in GOOD:
        issues.append(name+'收据尚未完成'); return None
    if result_path is not None:
        declared = value.get('results_sha256')
        if declared and declared != sha(result_path):
            raise ValueError(name+'结果文件与完成哈希不符')
    return value


def arm_map(arm, label):
    if not isinstance(arm, dict): raise ValueError('Missing arm '+label)
    ids = list(map(str, arm.get('ids', []))); values = arm.get('per_recipient_mse', [])
    if len(ids) != len(values) or len(ids) != len(set(ids)):
        raise ValueError('Invalid paired IDs: '+label)
    if any(not number(x) or x < 0 for x in values): raise ValueError('Invalid MSE: '+label)
    if arm.get('recipients', len(ids)) != len(ids): raise ValueError('Wrong arm count: '+label)
    observed = mean(values)
    if observed is not None and (not number(arm.get('mse')) or abs(observed-arm['mse']) > 1e-8*(1+abs(observed))):
        raise ValueError('Mean/per-recipient MSE mismatch: '+label)
    if not ids and arm.get('mse') is not None:
        raise ValueError('MSE has no recipient evidence: '+label)
    return dict(zip(ids, values))


def cohort(maps, ids, name):
    ids = sorted(map(str, ids)); desired = set(ids)
    matched = maps['matched']; null = maps.get('null', {}); wrong = maps.get('wrong', {})
    if not desired <= matched.keys(): raise ValueError('Missing matched cohort '+name)
    null_ids = desired & null.keys(); wrong_ids = desired & wrong.keys()
    if null and null_ids != desired: raise ValueError('Null cohort differs from Correct')
    mm = mean(matched[i] for i in ids)
    nn = mean(null[i] for i in ids) if null else None
    wc = mean(matched[i] for i in sorted(wrong_ids)); ww = mean(wrong[i] for i in sorted(wrong_ids))
    return dict(cohort=name, recipients=len(ids), ids_sha256=digest(ids), matched_mse=mm,
                null_mse=nn, history_gain_percent=reduction(nn, mm),
                wrong_mse=ww, correct_on_wrong_cohort_mse=wc,
                wrong_minus_correct=ww-wc if ww is not None and wc is not None else None,
                correct_vs_wrong_reduction_percent=reduction(ww, wc),
                wrong_recipients=len(wrong_ids), wrong_coverage=len(wrong_ids)/len(ids) if ids else None,
                wrong_ids_sha256=digest(sorted(wrong_ids)))


def probe_rows(value, row):
    result = []
    def channel(name, fields, meta):
        for field, scores in fields.items():
            if not isinstance(scores, dict) or not any(k in scores for k in METRICS): continue
            item = {k: row.get(k) for k in ('id', 'family', 'scene', 'role', 'source_budget', 'profile')}
            item.update(channel=name, physical_quantity=field,
                        representation=meta.get('representation', value.get('representation', value.get('probe_input'))),
                        train_objects=meta.get('train_objects', value.get('train_objects')),
                        validation_objects=meta.get('val_objects', value.get('val_objects')),
                        attribution=meta.get('attribution'), supports=meta.get('supports'),
                        normalization=value.get('normalization'), ridge_alpha=value.get('ridge_alpha'),
                        head_sha256=meta.get('head_sha256', meta.get('selected_head_sha256')))
            for metric in METRICS:
                item[metric] = scores.get(metric)
                if item[metric] is not None and not number(item[metric]):
                    raise ValueError('Invalid probe metric '+name+':'+field+':'+metric)
            result.append(item)
    for name, fields in value.get('representations', {}).items(): channel(name, fields, value)
    for name in ('independent_support_mean', 'actual_readout_memory', 'actual_support_memory64'):
        if isinstance(value.get(name), dict): channel(name, value[name].get('fields', {}), value[name])
    # Some legacy probes have one explicit representation with a fields mapping.
    if not result and isinstance(value.get('fields'), dict):
        channel(value.get('channel', 'unspecified'), value['fields'], value)
    return result


def collect_row(entry, index, root, manifest, verify_checkpoints=False):
    row = {key: entry.get(key) for key in ('family', 'scene', 'role', 'source_method', 'source_route', 'source_budget', 'source_checkpoint',
                                         'readout_out', 'profile', 'status')}
    row['id'] = entry.get('id') or '/'.join(str(row[k]) for k in ('family', 'scene', 'role', 'source_budget'))
    row['declared_status'] = row.pop('status'); row['status'] = 'PENDING'
    row['required'] = entry.get('required', True); row['issues'] = []; row['warnings'] = []
    row['canonical_role'] = role_name(row['role']); reference = row['canonical_role'] in ('Known', 'Query-only')
    row['reference_only'] = reference; row['cohorts'] = []; row['probes'] = []; row['artifacts'] = {}
    row['source_route_evidence'] = 'manifest' if row.get('source_route') is not None else None
    row['source_complete'] = row['head_complete'] = row['fullval_complete'] = row['probe_complete'] = False
    row['comparison_notes'] = []
    def path(value):
        if value is None: return None
        value = Path(value); return check_path(value if value.is_absolute() else root/value)
    def getpath(key, default=None): return path(entry.get(key)) if entry.get(key) is not None else default
    def artifact(name, location):
        if location is not None:
            row['artifacts'][name] = dict(path=str(location), exists=location.exists())
            if location.exists() and location.suffix == '.json': row['artifacts'][name]['sha256'] = sha(location)
    try:
        if not all(row[k] is not None for k in ('family', 'scene', 'role', 'readout_out', 'profile')):
            raise ValueError('Missing required manifest row metadata')
        if not reference and row['source_budget'] not in (50, 100, 150):
            raise ValueError('Invalid source budget; expected 50/100/150')
        out = path(row['readout_out']); cp = path(row['source_checkpoint'])
        head = getpath('head_dir', out/('S3/known' if row['canonical_role']=='Known' else 'S3/query' if reference else 'S3/learned'))
        resultpath = getpath('fullval_results', head/'fullval/results.json')
        fullmarkerpath = getpath('fullval_marker', resultpath.parent/'complete.json')
        headmarkerpath = getpath('head_marker', head/'complete.json')
        selectedpath = getpath('selected_validation', head/'selected_validation.json')
        configpath = getpath('head_config', head/'config.json')
        probepath = getpath('probe_path', out/'probes.json')
        encpath = getpath('encoding_marker', existing([out/'encoding_complete.json', out/'codes_complete.json']))
        freezepath = getpath('checkpoint_freeze', resultpath.parent/'checkpoint_freeze.json')
        srcmarkerpath = getpath('source_marker', existing(source_marker_candidates(cp, row['source_budget'])))
        for name, p in (('source_checkpoint', cp), ('source_marker', srcmarkerpath), ('head_marker', headmarkerpath),
                        ('fullval_results', resultpath), ('fullval_marker', fullmarkerpath), ('probe', probepath),
                        ('encoding', encpath), ('checkpoint_freeze', freezepath), ('head_config', configpath)):
            artifact(name, p)
        source_hash = entry.get('source_checkpoint_sha256')
        if not reference:
            if cp is None or not cp.exists(): row['issues'].append('源检查点尚未落盘')
            src = marker(srcmarkerpath, row['issues'], '源')
            if src is not None:
                budget = src.get('epochs', src.get('epoch', src.get('source_epochs', src.get('source_budget', src.get('budget')))))
                if budget != row['source_budget']: raise ValueError('Source marker budget differs from manifest')
                for key in ('scene', 'family'):
                    if src.get(key) is not None and src[key] != row[key]:
                        raise ValueError('Source marker differs: '+key)
                source_hash = (src.get('selected_sha256') if cp and src.get('selected_checkpoint') and
                               path(src['selected_checkpoint']) == cp else src.get('checkpoint_sha256', source_hash))
                if entry.get('source_checkpoint_sha256') and source_hash != entry['source_checkpoint_sha256']:
                    raise ValueError('Source checkpoint hash differs from manifest')
                if cp and cp.exists() and verify_checkpoints:
                    actual = sha(cp)
                    if source_hash and source_hash != actual: raise ValueError('Source checkpoint content hash mismatch')
                    source_hash = actual
                row['source_complete'] = cp is not None and cp.exists()
                row['source_steps'] = src.get('steps', src.get('step'))
                row['source_selected_epoch'] = src.get('selected_epoch', src.get('epoch'))
            row['source_checkpoint_sha256'] = source_hash
        else:
            row['source_complete'] = True; row['probe_complete'] = True
            row['warnings'].append('任务参照；不计算 Oracle 空间回收比例；无源参数探针要求')

        encoding = load(encpath) if encpath is not None and encpath.exists() else {}
        binding = encoding.get('binding', encoding)
        if not reference and encoding:
            boundhash = binding.get('checkpoint_sha256', binding.get('source_sha256'))
            if source_hash and boundhash and boundhash != source_hash:
                raise ValueError('Encoded model differs from manifest source checkpoint')
            if boundhash: row['source_checkpoint_sha256'] = boundhash
            budget = binding.get('source_epochs', binding.get('source_budget'))
            if budget is not None and budget != row['source_budget']:
                raise ValueError('Encoding source budget differs')
            row['source_version'] = binding.get('source_version', binding.get('version'))
            observed_route = binding.get('source_route') or binding.get('experiment_binding', {}).get('route') or binding.get('source_config', {}).get('route')
            if observed_route is not None:
                if row.get('source_route') is not None and row['source_route'] != observed_route:
                    raise ValueError('Manifest and encoded source routes differ')
                row['source_route_evidence'] = 'manifest+encoding' if row.get('source_route') is not None else 'encoding'
                row['source_route'] = observed_route
            row['representation'] = binding.get('representation')
            row['support_dims'] = binding.get('support_dims')

        config = load(configpath) if configpath.exists() else {}
        headmarker = marker(headmarkerpath, row['issues'], '新头')
        if headmarker is not None:
            if headmarker.get('source_epochs', row['source_budget']) != row['source_budget']:
                raise ValueError('Head marker source budget differs')
            row['head_complete'] = True
            row['head_budget'] = headmarker.get('epochs', headmarker.get('head_budget', config.get('epochs')))
            row['selected_head_epoch'] = headmarker.get('selected_epoch')
            row['head_sha256'] = headmarker.get('checkpoint_sha256')
        row['head_progress'] = None
        if not row['head_complete'] and (head/'progress.json').exists():
            p = load(head/'progress.json'); history = p.get('history', [])
            row['head_progress'] = dict(epoch=p.get('epoch'), best=p.get('best'), last=history[-1] if history else None)
        full = None
        if resultpath.exists():
            fullmarker = marker(fullmarkerpath, row['issues'], '完整验证', resultpath)
            if fullmarker is not None:
                if fullmarker.get('source_epochs', row['source_budget']) != row['source_budget']:
                    raise ValueError('Full validation marker source budget differs')
                full = load(resultpath)
                if full.get('status') not in GOOD: raise ValueError('Full validation result not COMPLETE')
                if full.get('test_read') is not False:
                    raise ValueError('Full validation must explicitly declare test_read=false')
                if full.get('scene') is not None and full['scene'] != row['scene']:
                    raise ValueError('Full validation scene differs')
                if full.get('source_epochs', row['source_budget']) != row['source_budget']:
                    raise ValueError('Full validation source budget differs')
                row['readout_version'] = full.get('version', config.get('version'))
                row['head_budget'] = full.get('head_budget', row.get('head_budget', config.get('epochs')))
                row['supports'] = full.get('supports', config.get('supports', 3))
                row['selected_head_epoch'] = full.get('selected_epoch', row.get('selected_head_epoch'))
                row['representation'] = full.get('representation', row.get('representation'))
                row['null_semantics'] = full.get('null_semantics', config.get('null_semantics'))
                row['wrong_donor_domain'] = full.get('wrong_donor_domain')
                if row.get('head_budget') != entry.get('head_budget', 100):
                    raise ValueError('Head budget differs from registered budget')
                maps = {'matched': arm_map(full.get('matched'), 'Correct')}
                if not reference:
                    maps.update({k: arm_map(full.get(k), k) for k in ('null', 'wrong')})
                    if not maps['wrong'].keys() <= maps['matched'].keys():
                        raise ValueError('Wrong includes recipients outside Correct')
                expected = entry.get('full_validation_rows', manifest.get('full_validation_rows', FULL_ROWS).get(row['scene']))
                if expected is None: raise ValueError('Unknown complete validation count')
                if len(maps['matched']) != expected or full.get('full_validation_rows', expected) != expected:
                    raise ValueError('Result is not the complete registered validation cohort')
                row['cohorts'].append(cohort(maps, maps['matched'], 'full_validation'))
                if selectedpath.exists():
                    selected = load(selectedpath); selected_ids = list(map(str, selected.get('ids', [])))
                    if not selected_ids: row['warnings'].append('选点收据缺逐样本 ID；不猜原 512 子集')
                    else:
                        row['cohorts'].append(cohort(maps, selected_ids, 'selection'))
                        row['cohorts'].append(cohort(maps, set(maps['matched'])-set(selected_ids), 'remaining'))
                else: row['warnings'].append('缺选点逐样本收据；仅报告全验证集')
                if freezepath.exists():
                    frozen = load(freezepath)
                    if row.get('source_checkpoint_sha256') and frozen.get('source_checkpoint_sha256') and frozen['source_checkpoint_sha256'] != row['source_checkpoint_sha256']:
                        raise ValueError('Full validation uses a different source checkpoint')
                    if row.get('head_sha256') and frozen.get('head_sha256') and row['head_sha256'] != frozen['head_sha256']:
                        raise ValueError('Full validation uses a different selected head')
                    row['head_sha256'] = frozen.get('head_sha256', row.get('head_sha256'))
                    row['fullval_input_sha256'] = frozen.get('prepared_sha256')
                row['reproduction'] = full.get('reproduction')
                row['fullval_complete'] = True
        else: row['issues'].append('完整验证集结果尚未落盘')

        if not reference:
            if probepath.exists():
                probe = load(probepath)
                if probe.get('status') not in GOOD: row['issues'].append('探针尚未完成')
                else:
                    if probe.get('test_read') is not False: raise ValueError('Probe must declare test_read=false')
                    if probe.get('scene') is not None and probe['scene'] != row['scene']:
                        raise ValueError('Probe scene differs')
                    if probe.get('source_epochs', row['source_budget']) != row['source_budget']:
                        raise ValueError('Probe source budget differs')
                    if probe.get('source_checkpoint_sha256') and row.get('source_checkpoint_sha256') and probe['source_checkpoint_sha256'] != row['source_checkpoint_sha256']:
                        raise ValueError('Probe source differs')
                    encoding_hash = probe.get('encoding_sha256', probe.get('source_codes_sha256'))
                    if encoding_hash and encpath is not None and encoding_hash != sha(encpath):
                        raise ValueError('Probe encoding binding differs')
                    row['probes'] = probe_rows(probe, row)
                    for p in row['probes']:
                        if p.get('head_sha256') and row.get('head_sha256') and p['head_sha256'] != row['head_sha256']:
                            raise ValueError('Actual-memory probe uses another head')
                    if not row['probes']: row['issues'].append('探针无真实物理量数值')
                    else: row['probe_complete'] = True
                    for channel in entry.get('required_probe_channels', []):
                        if not any(p['channel'] == channel for p in row['probes']):
                            row['probe_complete'] = False; row['issues'].append('缺指定探针通道 '+channel)
            else: row['issues'].append('探针尚未落盘')
        row['status'] = 'COMPLETE' if all(row[k] for k in ('source_complete','head_complete','fullval_complete','probe_complete')) else 'PENDING'
    except (OSError, ValueError, KeyError, TypeError) as error:
        row['status'] = 'ERROR'; row['issues'].append(str(error))
    return row


def compare_signature(row, c):
    profile = row['profile']; version = row.get('readout_version')
    if isinstance(profile, dict) and version in profile.get('compatible_readout_versions', []):
        version = 'explicit-profile-compatible'
    return (row['family'], row['scene'], row['source_budget'], canonical(profile), version,
            row.get('head_budget'), row.get('supports'), c['cohort'], c['ids_sha256'])


def add_comparisons(rows):
    controls = {}
    for row in rows:
        if row['status'] == 'ERROR' or not row['fullval_complete'] or row['canonical_role'] not in ('Native', 'Structure'): continue
        for c in row['cohorts']:
            key = (compare_signature(row, c), row['canonical_role'])
            controls.setdefault(key, []).append((row, c))
    for row in rows:
        for c in row['cohorts']:
            for name, prefix in (('Native', 'native'), ('Structure', 'structure')):
                c[prefix+'_row_id'] = c[prefix+'_mse'] = c[prefix+'_minus_method_mse'] = c[prefix+'_reduction_percent'] = None
                if row['reference_only'] or row['status']=='ERROR': continue
                pool = controls.get((compare_signature(row, c), name), [])
                if len(pool) != 1:
                    note = ('缺同协议 '+name+'，不跨版本/预算补比' if not pool else '同组 '+name+' 不唯一，不挑最优')
                    if note not in row['comparison_notes']: row['comparison_notes'].append(note)
                    continue
                control, score = pool[0]
                # Fullval prepared inputs must agree when both provide bindings.
                if row.get('fullval_input_sha256') and control.get('fullval_input_sha256') and row['fullval_input_sha256'] != control['fullval_input_sha256']:
                    row['comparison_notes'].append(name+' 完整输入绑定不同，未计算差值'); continue
                c[prefix+'_row_id'] = control['id']; c[prefix+'_mse'] = score['matched_mse']
                if number(score['matched_mse']) and number(c['matched_mse']):
                    c[prefix+'_minus_method_mse'] = score['matched_mse']-c['matched_mse']
                    c[prefix+'_reduction_percent'] = reduction(score['matched_mse'], c['matched_mse'])


def csv_text(rows, columns):
    stream = io.StringIO(); writer = csv.DictWriter(stream, columns, extrasaction='ignore'); writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k,v in row.items() if k in columns})
    return stream.getvalue()


def fmt(value, digits=5, suffix=''):
    return '—' if value is None else f'{value:.{digits}f}'+suffix


def esc(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown(summary):
    count = summary['counts']
    text = [f"# CoPhy 单种子验证结果（{summary['status']}）", '',
            f"必需条目完成 {count['required_complete']}/{count['required']}；全量评价完成 {count['fullval_complete']}；探针完成 {count['learned_probe_complete']}。",
            '', '以下均为 validation；源预算、评价版本和样本范围分开。误差越小越好，收益百分比为误差相对下降。缺项用 —，不按 0 处理。', '',
            '| 骨架 | 场景 | 源轮数 | 条件 | 状态 | Correct MSE↓ | 对 Native 收益 | 对 Structure 收益 | Null MSE | Wrong MSE | Wrong−Correct | Wrong 覆盖 |',
            '|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['rows']:
        c = next((x for x in r['cohorts'] if x['cohort']=='full_validation'), {})
        label = r['role'] + (' ('+r['source_method']+')' if r.get('source_method') and r['source_method'] != r['role'] else '')
        values = [r['family'], r['scene'], r['source_budget'], label, r['status'], fmt(c.get('matched_mse')),
                  fmt(c.get('native_reduction_percent'),2,'%'), fmt(c.get('structure_reduction_percent'),2,'%'),
                  fmt(c.get('null_mse')), fmt(c.get('wrong_mse')), fmt(c.get('wrong_minus_correct')),
                  fmt(100*c['wrong_coverage'],1,'%') if c.get('wrong_coverage') is not None else '—']
        text.append('| '+' | '.join(esc(v if v is not None else '—') for v in values)+' |')
    text += ['', 'Wrong−Correct 只在 Wrong 可配对的同一 recipient 集上计算；它与全体 Correct 均值可能不是同一分母。Known 仅为已知参数任务参考，不当作理论上界，也不自动计算“Oracle 空间回收”。', '',
             '## 原选点与其余验证样本', '', '| 条目 | 子集 | 样本数 | Correct MSE↓ | 对 Native 收益 | 对 Structure 收益 |', '|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        for c in r['cohorts']:
            if c['cohort']=='full_validation': continue
            text.append('| '+' | '.join(map(esc,[r['id'],c['cohort'],c['recipients'],fmt(c['matched_mse']),fmt(c.get('native_reduction_percent'),2,'%'),fmt(c.get('structure_reduction_percent'),2,'%')]))+' |')
    text += ['', '## 真实物理量探针', '', '完整状态、独立历史聚合和头内记忆分别列出。头内记忆包含监督读出的作用，不冒称纯预训练表示；R² 缺失或负值原样保留，不等同信息不存在。', '',
             '| 条目 | 通道 | 物理量 | R² | 分类准确率 | 平衡准确率 | 验证对象数 |', '|---|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        for p in r['probes']:
            text.append('| '+' | '.join(map(esc,[r['id'],p['channel'],p['physical_quantity'],fmt(p.get('r2'),4),fmt(100*p['accuracy'],2,'%') if p.get('accuracy') is not None else '—',fmt(100*p['balanced_accuracy'],2,'%') if p.get('balanced_accuracy') is not None else '—',p.get('validation_objects') if p.get('validation_objects') is not None else '—']))+' |')
    text += ['', '## 绑定、缺项与解释边界', '']
    for r in summary['rows']:
        text.append('- **'+esc(r['id'])+'**：profile='+esc(canonical(r['profile']))+'；readout='+esc(r.get('readout_version','待落盘'))+'；源路由='+esc(r.get('source_route') or '未登记/不适用')+'；表示='+esc(r.get('representation','待落盘'))+'。')
        notes = r['issues']+r['warnings']+r['comparison_notes']
        if notes: text.append('  '+ '；'.join(esc(x) for x in dict.fromkeys(notes))+'。')
    text += ['', '本汇总器只读取已提交结果，不选新检查点、不训练、不读取封闭 test、不自动完成研究 goal。COMPLETE 仅表示本清单的必需产物齐全，不代表所有科学假设成立。', '']
    return '\n'.join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--verify-checkpoints', action='store_true', help='Also stream SHA256 of source weights; never deserialize weights')
    args = p.parse_args(); manifestpath = check_path(Path(args.manifest).resolve()); manifest = load(manifestpath)
    entries = manifest.get('rows', [])
    if not entries: p.error('Manifest must contain nonempty rows')
    rows = [collect_row(r, i, manifestpath.parent, manifest, args.verify_checkpoints) for i,r in enumerate(entries)]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)): p.error('Duplicate row IDs; give explicit unique id for distinct profiles')
    add_comparisons(rows); required = [r for r in rows if r['required']]
    counts = dict(rows=len(rows), required=len(required), required_complete=sum(r['status']=='COMPLETE' for r in required),
                  fullval_complete=sum(r['fullval_complete'] for r in rows),
                  learned_probe_complete=sum(r['probe_complete'] and not r['reference_only'] for r in rows),
                  states=dict(Counter(r['status'] for r in rows)))
    status = 'COMPLETE' if required and all(r['status']=='COMPLETE' for r in required) else 'PARTIAL'
    summary = dict(version=VERSION,status=status,created_unix=time.time(),manifest=str(manifestpath),
                   manifest_sha256=sha(manifestpath),counts=counts,rows=rows,
                   checkpoint_contents_verified=args.verify_checkpoints,test_read=False,goal_updated=False,
                   comparisons='same family, scene, source budget, declared profile, reviewed readout version, head budget, supports and recipient IDs',
                   known_parameters='reference only; no Oracle recovery fraction',scientific_success_required=False)
    out = Path(args.out).resolve()
    # Output is an independent presentation directory, not a source/head directory.
    for r in rows:
        artifactroot = Path(r['readout_out'])
        if not artifactroot.is_absolute(): artifactroot = manifestpath.parent/artifactroot
        if out == artifactroot.resolve() or artifactroot.resolve() in out.parents:
            p.error('--out must not be a readout directory or its child')
    out.mkdir(parents=True, exist_ok=True); write(out/'summary.json', summary)
    flattened = []
    for r in rows:
        base = {k:r.get(k) for k in ('id','family','scene','role','source_method','source_route','source_route_evidence','source_budget','profile','status','readout_version','head_budget','selected_head_epoch','representation','source_checkpoint_sha256')}
        for c in r['cohorts'] or [{}]: flattened.append(dict(base, **c, issues=r['issues'], comparison_notes=r['comparison_notes']))
    cols = ['id','family','scene','role','source_method','source_route','source_route_evidence','source_budget','profile','status','readout_version','head_budget','selected_head_epoch','representation','source_checkpoint_sha256',
            'cohort','recipients','ids_sha256','matched_mse','native_row_id','native_mse','native_minus_method_mse','native_reduction_percent',
            'structure_row_id','structure_mse','structure_minus_method_mse','structure_reduction_percent','null_mse','history_gain_percent',
            'wrong_mse','correct_on_wrong_cohort_mse','wrong_minus_correct','correct_vs_wrong_reduction_percent','wrong_recipients','wrong_coverage','issues','comparison_notes']
    atomic(out/'results.csv', csv_text(flattened, cols))
    probes = [probe for r in rows for probe in r['probes']]
    atomic(out/'probes.csv', csv_text(probes, ['id','family','scene','role','source_budget','profile','channel','physical_quantity','representation',
             'r2','mse','accuracy','balanced_accuracy','train_objects','validation_objects','attribution','supports','normalization','ridge_alpha','head_sha256']))
    atomic(out/'summary.md', markdown(summary))
    receipt = dict(version=VERSION,status=status,counts=counts,manifest_sha256=summary['manifest_sha256'],
                   summary_sha256=sha(out/'summary.json'),test_read=False,goal_updated=False)
    write(out/'status.json', receipt)
    if status=='COMPLETE': write(out/'complete.json', receipt)
    elif (out/'complete.json').exists(): (out/'complete.json').unlink()
    print(json.dumps(dict(status=status,counts=counts,out=str(out),test_read=False,goal_updated=False),ensure_ascii=False))


if __name__ == '__main__': main()
