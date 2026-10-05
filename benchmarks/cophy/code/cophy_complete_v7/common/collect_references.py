"""Read-only shared-reference and original supervised-task supplement.

python collect_references.py --manifest presentation_manifest.json --out DIR
Optional --allow-archives permits hash-bound downloaded receipts when the remote
artifact is not mounted; these rows remain explicitly ARCHIVED_COMPLETE.
Never trains, loads model tensors, reads test, substitutes a different source,
or interprets Known as a paired/theoretical Oracle.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location('_cophy_v7_presentation_io', Path(__file__).with_name('collect.py'))
io = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(io)
VERSION = 'cophy-v7-reference-official-collector-1'


def resolve(value, root):
    if value is None: return None
    p = Path(value)
    return io.check_path(p if p.is_absolute() else root/p)


def read_present(path):
    return io.load(path) if path is not None and path.exists() else None


def checkpoint_hash(row, root):
    cp = resolve(row['source_checkpoint'], root)
    names = [resolve(row.get('source_marker'), root),
             cp.parent/f"budget_{row['source_budget']}_complete.json",
             cp.parent/f"checkpoint_{row['source_budget']}_complete.json"]
    for p in names:
        value = read_present(p)
        if not value: continue
        if value.get('status')!='COMPLETE': continue
        budget = value.get('source_budget', value.get('epochs', value.get('epoch')))
        if budget != row['source_budget']: raise ValueError('Wrong source budget receipt')
        selected = resolve(value.get('selected_checkpoint'), root)
        h = value.get('selected_sha256') if selected == cp else value.get('checkpoint_sha256')
        if h: return h, value, str(p)
    return row.get('source_checkpoint_sha256'), None, None


def collect_reference(entry, asset, root, allow_archives):
    row = dict(scene=entry['scene'], reference=entry['reference'], status='PENDING',
               profile='shared-pose-q3-head100-task-reference', source_budget=None,
               reference_only=True, probe_required=False, issues=[], test_read=False)
    rp = resolve(entry['results'], root); row['results'] = str(rp)
    expected = asset['compatibility']['counts'][row['scene']]
    bound = next((a for a in asset['references'] if a['scene']==row['scene'] and a['reference']==row['reference']), None)
    try:
        if not bound or bound['results'] != str(rp):
            raise ValueError('Reference path has no reviewed task-profile binding')
        row['expected_results_sha256'] = bound['result_sha256']
        row['known_inputs'] = asset['compatibility']['known_inputs'][row['scene']] if row['reference']=='known' else []
        if rp.exists():
            if io.sha(rp) != bound['result_sha256']: raise ValueError('Frozen reference result changed')
            result = io.load(rp); marker = io.load(rp.parent/'complete.json'); frozen = io.load(rp.parent/'checkpoint_freeze.json')
            if marker.get('status')!='COMPLETE' or marker.get('results_sha256')!=io.sha(rp):
                raise ValueError('Uncommitted full reference evaluation')
            if result.get('status')!='COMPLETE' or result.get('test_read') is not False or result.get('reference')!=row['reference']:
                raise ValueError('Wrong reference/test scope')
            if result.get('full_validation_rows') != expected or frozen.get('head_budget')!=100 or result.get('supports')!=3:
                raise ValueError('Reference task/budget differs')
            mapping = io.arm_map(result.get('matched'), 'shared reference')
            if len(mapping)!=expected: raise ValueError('Reference incomplete recipient coverage')
            selected_cp = resolve(frozen['checkpoint'], root)
            config = io.load(selected_cp.parent/'config.json')
            if config.get('epochs')!=100 or config.get('supports')!=3 or config.get('reference')!=row['reference'] or config.get('test_read') is not False:
                raise ValueError('Reference head is not the registered S3/head100 profile')
            if config.get('seed')!=0: raise ValueError('Reference seed differs')
            row.update(status='COMPLETE', mse=result['matched']['mse'], recipients=expected,
                       selected_head_epoch=result['selected_epoch'], head_budget=100,
                       ids_sha256=io.digest(sorted(mapping)), checkpoint=str(selected_cp),
                       checkpoint_sha256=frozen.get('checkpoint_sha256'),
                       prepared_sha256=frozen.get('prepared_sha256'), head_config_sha256=io.sha(selected_cp.parent/'config.json'),
                       reproduction=result.get('reproduction'), provenance='read committed fullval JSON and original head config')
        elif allow_archives:
            x = bound['archived_result']
            if x['full_rows']!=expected: raise ValueError('Archive is not the full validation cohort')
            row.update(status='ARCHIVED_COMPLETE',mse=x['matched_mse'],recipients=x['full_rows'],
                       selected_head_epoch=x['selected_epoch'],head_budget=100,
                       original512_max_error=x['original512_max_error'],archive=bound['archive'],
                       provenance='previously downloaded frozen receipt; not a current remote filesystem check')
        else: row['issues'].append('Result path not mounted; use --allow-archives only for a labelled offline report')
        row['interpretation'] = bound['interpretation']
        row['retraining_required_for_source_budget_axis'] = False
    except (OSError, ValueError, KeyError, TypeError) as e:
        row['status']='ERROR';row['issues'].append(str(e))
    return row


def own_probe(report):
    # Only own frozen source representations belong to the original task table.
    # Head-projected support memory is a separate cross-experience readout.
    if report is None: return None
    if report.get('representations'):
        return dict(representation=report.get('representation',report.get('probe_input')),
                    representations=report['representations'],train_objects=report.get('train_objects'),
                    val_objects=report.get('val_objects'),normalization=report.get('normalization'))
    return None


def collect_official(entry, assets, root, allow_archives):
    row = {k:entry.get(k) for k in ('id','scene','role','source_method','source_budget','source_checkpoint','source_route','readout_out')}
    row.update(status='PENDING',family='CoPhyNet',task='official AB plus one C frame predicts CD[1:]',
               metric_profile='corrected official coordinate MSE; shared visual-presence mask; balls xy/others xyz',
               issues=[],test_read=False,source_probe=None)
    archived = next((a for a in assets['official_source50'] if a['scene']==row['scene'] and a['role']==row['role'] and a['source_checkpoint']==row['source_checkpoint'] and row['source_budget']==50), None)
    try:
        expected_hash, budget, budget_path = checkpoint_hash(entry, root)
        if archived and expected_hash and expected_hash!=archived['source_checkpoint_sha256']:
            raise ValueError('Official source50 archive is a different selected checkpoint')
        if archived and expected_hash is None: expected_hash=archived['source_checkpoint_sha256']
        row['source_checkpoint_sha256']=expected_hash
        rp = resolve(entry.get('official_results') or (archived or {}).get('official_results'), root)
        selection = resolve(entry.get('official_selected_validation') or (archived or {}).get('selected_validation'), root)
        result = read_present(rp)
        if result is not None:
            if archived and archived.get('official_results_sha256') and io.sha(rp)!=archived['official_results_sha256']:
                raise ValueError('Official source50 evaluation hash differs')
            if result.get('test_read') is not False or result.get('split','val')!='val': raise ValueError('Official task is not validation')
            if result.get('checkpoint_sha256')!=expected_hash: raise ValueError('Official result checkpoint differs')
            row.update(status='COMPLETE', official=result['official'],selected_source_epoch=result['epoch'],
                       visual_coverage=result.get('visual_coverage'),official_results=str(rp),
                       official_results_sha256=io.sha(rp),provenance='fixed selected source evaluation',
                       preflight_sha256=result.get('preflight_sha256'))
        elif budget is not None and io.number(budget.get('best_mse')):
            cp=resolve(entry['source_checkpoint'],root)
            if resolve(budget.get('selected_checkpoint'),root)!=cp or budget.get('selected_sha256')!=expected_hash:
                raise ValueError('Budget MSE does not bind the requested selected checkpoint')
            row.update(status='COMPLETE',official={'model':budget['best_mse']},selected_source_epoch=budget['selected_epoch'],
                       official_results=budget_path,provenance='budget-frozen selected full-validation MSE; not last-epoch MSE',
                       source_training_profile=budget.get('profile'))
        elif selection is not None and selection.exists():
            value=io.load(selection)
            if not archived: raise ValueError('Selected metric needs explicit checkpoint/selection binding')
            if io.sha(selection)!=archived['selected_validation_sha256'] or value.get('epoch')!=archived['selected_epoch']:
                raise ValueError('Source-generated selected metric changed')
            if not io.number(value.get('mse')): raise ValueError('Missing selected official MSE')
            row.update(status='COMPLETE',official={'model':value['mse']},selected_source_epoch=value['epoch'],
                       official_results=str(selection),provenance='source-generated fixed selected-validation receipt')
        elif archived and allow_archives and archived.get('official') is not None:
            row.update(status='ARCHIVED_COMPLETE',official=archived['official'],selected_source_epoch=archived['selected_epoch'],
                       visual_coverage=archived.get('visual_coverage'),official_results=archived.get('official_results'),
                       archive=archived['archive'],preflight_sha256=archived.get('preflight_sha256'),
                       provenance='downloaded selected-source official evaluation; not current remote check')
        else:
            row['issues'].append('Selected checkpoint official MSE not locally available; do not substitute previous recipe')
            if selection: row['pending_selected_validation']=str(selection)
        probe_path=resolve(entry.get('probe_path') or str(Path(entry['readout_out'])/'probes.json'),root)
        report=read_present(probe_path)
        if report is not None:
            if report.get('status')!='COMPLETE' or report.get('test_read') is not False: raise ValueError('Uncommitted source probe')
            if report.get('source_checkpoint_sha256')!=expected_hash: raise ValueError('Probe belongs to another source')
            row['source_probe']=own_probe(report);row['probe_path']=str(probe_path)
        elif archived and archived.get('source_probe_balanced_accuracy'):
            row['source_probe']=dict(representation='OWN AB P16/T16/U32; original source before support aggregation',
                                     balanced_accuracy=archived['source_probe_balanced_accuracy'],archive=archived['archive'])
        row['probe_status']='COMPLETE' if row['source_probe'] else 'PENDING'
    except (OSError, ValueError, KeyError, TypeError) as e:
        row['status']='ERROR';row['issues'].append(str(e))
    return row


def report_text(summary):
    lines=['# CoPhy 共享参照与监督原任务', '',
           '这是主跨情境结果表的补充。已知参数是任务参考，不是理论 Oracle，也不计算收益空间回收率。ARCHIVED_COMPLETE 表示沿用有哈希的已下载收据，没有重新检查远端。', '',
           '## 六个共享任务参照', '', '| 场景 | 参照 | 状态 | 完整 validation MSE↓ | 样本数 | 头预算 / 选中轮 |', '|---|---|---|---:|---:|---:|']
    for r in summary['references']:
        lines.append('| '+' | '.join(map(io.esc,[r['scene'],r['reference'],r['status'],io.fmt(r.get('mse')),r.get('recipients','—'),str(r.get('head_budget','—'))+' / '+str(r.get('selected_head_epoch','—'))]))+' |')
    lines += ['', '这六项共享相同 pose/detection/公开类型输入、q=3、预测区间和100轮头协议；Known 另给 GT 参数。自监督模型的当前完整潜状态不送进这些参照，因此不能把它当作“同模型完整输入再额外给参数”的成对 Oracle。源50/100/150不改变任务参照。', '',
              '## 官方监督任务：同一源模型的另一项评价', '', '| 场景 | 条件 / 源实现 | 源预算 / 选中轮 | 状态 | 官方 MSE↓ | 官方 Copy C | 源参数 probe |', '|---|---|---:|---|---:|---:|---|']
    for r in summary['official']:
        m=r.get('official',{})
        lines.append('| '+' | '.join(map(io.esc,[r['scene'],str(r['role'])+' / '+str(r['source_method']),str(r['source_budget'])+' / '+str(r.get('selected_source_epoch','—')),r['status'],io.fmt(m.get('model')),io.fmt(m.get('CopyC')),r.get('probe_status','PENDING')]))+' |')
    lines += ['', '官方任务输入是完整 AB 加一帧 C，目标是 CD[1:]；新任务输入是独立历史加当前三帧，目标从第4帧开始。两项 MSE 不互相替代。100/150 的 best_mse 来自预算内已选源模型，源选中轮与训练预算分别保留。', '',
              'Copy C 定义为未来始终等于原任务的 C（一帧）；不是新任务的 Copy-last，也不是 Correct donor。已有 Copy C 作为每场景零训练参考单列，不借同名把不同评分区间混合。', '',
              '| 场景 | 官方 Copy C MSE | 目标 |', '|---|---:|---|']
    for r in summary['official_copy_c']:
        lines.append('| '+r['scene']+' | '+io.fmt(r['mse'])+' | CD[1:]；共享视觉 presence mask |')
    lines += ['',
              '## 缺项与下一步', '']
    for r in summary['references']+summary['official']:
        for issue in r.get('issues',[]):lines.append('- '+io.esc(r.get('id',r['scene']+'/'+r.get('reference','')))+'：'+io.esc(issue))
    lines += ['', '参考头无需因源预算变化重训。若后续改动 query 输入、目标区间、训练归一化/选点协议，最小工作仅为该场景重新训练 Query 与 Known 两个头；不需要重复源训练。',
              '目前缺失的是尚未就绪的预算源结果或未回收的小型选择收据，不是六个参考训练。', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--assets',default=str(Path(__file__).with_name('reference_official_assets.json')))
    p.add_argument('--allow-archives',action='store_true');args=p.parse_args()
    manifest_path=Path(args.manifest).resolve();manifest=io.load(manifest_path);assets=io.load(args.assets);root=manifest_path.parent
    references=[collect_reference(r,assets,root,args.allow_archives) for r in manifest.get('shared_references',[])]
    if len(references)!=6 or len({(r['scene'],r['reference']) for r in references})!=6:
        p.error('Expected exactly six explicit shared_references entries')
    official=[collect_official(r,assets,root,args.allow_archives) for r in manifest['rows'] if r['family']=='CoPhyNet']
    copy_c=[]
    for scene in assets['compatibility']['counts']:
        baseline=next(a for a in assets['official_source50'] if a['scene']==scene and a['role']=='Native')
        copy_c.append(dict(scene=scene,mse=baseline['official']['CopyC'],
                           gtmask_mse=baseline['official'].get('CopyC_gtmask'),
                           scope='official CD[1:]; constant visual C; visual presence mask',
                           preflight_sha256=baseline['preflight_sha256'],archive=baseline['archive'],test_read=False))
    complete=lambda r:r['status'] in ('COMPLETE','ARCHIVED_COMPLETE')
    counts=dict(references=len(references),reference_complete=sum(map(complete,references)),official=len(official),
                official_mse_complete=sum(map(complete,official)),official_probe_complete=sum(r.get('probe_status')=='COMPLETE' for r in official))
    references_status='COMPLETE' if all(map(complete,references)) else 'PARTIAL'
    official_status='COMPLETE' if all(map(complete,official)) and all(r.get('probe_status')=='COMPLETE' for r in official) else 'PARTIAL'
    status='COMPLETE' if references_status==official_status=='COMPLETE' else 'PARTIAL'
    summary=dict(version=VERSION,status=status,created_unix=time.time(),counts=counts,references=references,official=official,official_copy_c=copy_c,
                 references_status=references_status,official_status=official_status,
                 manifest_sha256=io.sha(manifest_path),asset_sha256=io.sha(args.assets),allow_archives=args.allow_archives,
                 test_read=False,goal_updated=False,minimum_reference_retraining=0,
                 compatibility=assets['compatibility'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    io.write(out/'references_official.json',summary);io.atomic(out/'references_official.md',report_text(summary))
    io.atomic(out/'references.csv',io.csv_text(references,['scene','reference','status','mse','recipients','head_budget','selected_head_epoch','profile','results','expected_results_sha256','provenance','issues']))
    flat=[dict(r,official_mse=r.get('official',{}).get('model'),copy_c=r.get('official',{}).get('CopyC')) for r in official]
    io.atomic(out/'official.csv',io.csv_text(flat,['id','scene','role','source_method','source_route','source_budget','selected_source_epoch','status','official_mse','copy_c','visual_coverage','source_checkpoint','source_checkpoint_sha256','official_results','probe_status','provenance','issues']))
    receipt=dict(version=VERSION,status=status,counts=counts,summary_sha256=io.sha(out/'references_official.json'),test_read=False,goal_updated=False)
    io.write(out/'references_official_status.json',receipt)
    reference_receipt=dict(receipt,status=references_status,scope='six shared task references only')
    io.write(out/'shared_references_status.json',reference_receipt)
    if references_status=='COMPLETE':io.write(out/'shared_references_complete.json',reference_receipt)
    elif (out/'shared_references_complete.json').exists():(out/'shared_references_complete.json').unlink()
    if status=='COMPLETE':io.write(out/'references_official_complete.json',receipt)
    elif (out/'references_official_complete.json').exists():(out/'references_official_complete.json').unlink()
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
