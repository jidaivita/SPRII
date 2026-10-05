"""Read completed frozen test predictions/probes; never run or select a model."""
import argparse
import csv
import io
from pathlib import Path
from statistics import mean
from runtime import VERSION, artifact, checked, read, sha, verify_freeze, write


def arm(value):
    if not value: return {}
    ids=list(map(str,value.get('ids',[]))); scores=value.get('per_recipient_mse',[])
    if len(ids)!=len(scores) or len(ids)!=len(set(ids)):raise ValueError('Invalid per-recipient predictions')
    return dict(zip(ids,map(float,scores)))


def paired(first,second):
    keys=sorted(set(first)&set(second))
    if not keys:return dict(recipients=0,first=None,second=None,difference=None,reduction_percent=None)
    a=mean(first[k] for k in keys);b=mean(second[k] for k in keys)
    return dict(recipients=len(keys),first=a,second=b,difference=a-b,
                reduction_percent=100*(a-b)/a if a else None,
                same_complete_cohort=set(first)==set(second),ids_sha256=__import__('hashlib').sha256('\n'.join(keys).encode()).hexdigest())


def table(path,rows,columns):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    path=Path(path);temporary=path.with_name(path.name+'.pending');temporary.write_text(stream.getvalue());temporary.replace(path)


def collect(args):
    freeze=verify_freeze(args.freeze,verify_all=False);results=Path(args.results);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    learned={e['id']:e for e in freeze['learned_entries']};refs={e['id']:e for e in freeze['references']}
    rows=[];values={};files=[];probe_rows=[]
    for ident in freeze['expected_ids']:
        entry=learned.get(ident,refs.get(ident));row=dict(id=ident,scene=entry['scene'],family=entry.get('family','reference'),
            role=entry.get('role',entry.get('reference')),source_budget=100 if ident in learned else None,head_budget=100,seed=0,status='PENDING',issues=[])
        try:
            path=results/ident/'results.json';done=read(path.with_name('complete.json'));item=done['results'];checked(item)
            if Path(item['path']).resolve()!=path.resolve():raise ValueError('Completion points to a different result')
            r=read(path)
            if done.get('status')!='COMPLETE' or r.get('status')!='COMPLETE' or r.get('split')!='test' or r.get('freeze_sha256')!=sha(args.freeze):
                raise ValueError('Wrong test/freeze/completion binding')
            if r.get('entry')!=ident or r.get('scene')!=entry['scene']:raise ValueError('Result identity mismatch')
            for key in ('source_optimizer_steps','head_optimizer_steps','test_probe_fit_steps'):
                if r.get(key,0)!=0:raise ValueError('Unexpected test fitting')
            a={name:arm(v) for name,v in r['arms'].items()};values[ident]=a
            if not a.get('matched'):raise ValueError('No qualified matched test recipient')
            row.update(status='COMPLETE',matched_mse=mean(a['matched'].values()),recipients=len(a['matched']),
                official_test_count=r.get('official_test_count'),correct_eligible_count=r.get('correct_eligible_count'),
                selected_head_epoch=r.get('selected_head_epoch'),source_selected_epoch=r.get('source_selected_epoch'),
                source_sha256=r.get('source_sha256'),head_sha256=r.get('head_sha256'))
            if ident in learned:
                if r.get('source_epochs')!=100 or r.get('head_budget')!=100:raise ValueError('Test source/head budget differs')
                if set(r.get('probes',{}))!={'own','support_mean','memory'}:raise ValueError('Missing paired physical probes')
                row['null_vs_matched']=paired(a.get('null',{}),a['matched'])
                row['wrong_vs_matched']=paired(a.get('wrong',{}),a.get('matched_on_wrong_cohort',a['matched']))
                row['history_gain_percent']=row['null_vs_matched']['reduction_percent']
                row['wrong_minus_correct']=row['wrong_vs_matched']['difference']
                if row['family']=='supervised':
                    official_task=r.get('official_task',{});official=official_task.get('official')
                    if not official or official.get('model') is None or official.get('CopyC') is None:
                        raise ValueError('Missing original supervised task and Copy C evaluation')
                    row['official']=dict(official,visual_coverage=official_task.get('visual_coverage'),official_rows=official_task.get('official_rows'))
                for channel,report in r['probes'].items():
                    for name,v in report.get('fields',{}).items():
                        probe_rows.append(dict(id=ident,scene=entry['scene'],family=row['family'],role=row['role'],channel=channel,
                            parameter=name,rows=report.get('rows'),attribution=report.get('attribution'),fit_split='train',**v))
            files.extend((artifact(path),artifact(path.with_name('complete.json'))))
        except (OSError,KeyError,ValueError,TypeError) as exc:
            row['status']='PENDING' if isinstance(exc,FileNotFoundError) else 'ERROR';row['issues'].append(str(exc))
        rows.append(row)
    for row in rows:
        if row['status']!='COMPLETE' or row['id'] not in learned:continue
        for role in ('Native','Structure','Random'):
            reference=next((r for r in rows if (r['scene'],r['family'],r['role'])==(row['scene'],row['family'],role) and r['status']=='COMPLETE'),None)
            if reference:
                comp=paired(values[reference['id']]['matched'],values[row['id']]['matched'])
                row[role.lower()+'_comparison']=dict(reference_id=reference['id'],**comp)
                row[role.lower()+'_reduction_percent']=comp['reduction_percent']
    status='COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL'
    result=dict(version=VERSION,status=status,freeze_sha256=sha(args.freeze),expected=51,complete=sum(r['status']=='COMPLETE' for r in rows),
        rows=rows,files=files,test_read=True,model_inference=False,model_selection=False,probe_refit=False,
        scope='fixed source100 and seed0; source50/150 remain validation learning curves',
        claim_boundary='single-seed evidence; parameter readability, history utility and method improvement are reported separately; Known is a task reference, not a mathematical upper bound')
    write(out/'summary.json',result)
    cols=['id','scene','family','role','source_budget','head_budget','status','recipients','matched_mse','native_reduction_percent',
        'structure_reduction_percent','random_reduction_percent','history_gain_percent','wrong_minus_correct','source_selected_epoch','selected_head_epoch','source_sha256','head_sha256','issues']
    table(out/'results.csv',rows,cols);table(out/'probes.csv',probe_rows,['id','scene','family','role','channel','parameter','rows','r2','mse','accuracy','balanced_accuracy','train_majority_accuracy','unseen_test_label_count','attribution','fit_split'])
    official_rows=[dict(id=r['id'],scene=r['scene'],role=r['role'],status=r['status'],source_budget=100,
        source_selected_epoch=r.get('source_selected_epoch'),**r.get('official',{})) for r in rows if r['family']=='supervised']
    table(out/'official_supervised.csv',official_rows,['id','scene','role','status','source_budget','source_selected_epoch','model','CopyC','visual_coverage'])
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['# CoPhy 最终测试结果','',f'完成 {result["complete"]}/51；固定单种子、源100轮、下游头100轮。','',
        '| 场景 | 方法族 | 条件 | 状态 | MSE↓ | 相对Native降低% | 历史收益% |','|---|---|---|---|---:|---:|---:|']
    for r in rows:lines.append('| '+' | '.join(str(x) for x in [r['scene'],r['family'],r['role'],r['status'],fmt(r.get('matched_mse')),fmt(r.get('native_reduction_percent')),fmt(r.get('history_gain_percent'))])+' |')
    lines+=['','参数探针见 probes.csv；own 是冻结源表示，support_mean 是历史聚合，memory 包含已训练下游头的投影。三者不混称。',
        '监督源模型的官方 AB+C→CD[1:] 测试和 Copy C 另列 official_supervised.csv，不与上表跨情境 q3→CD[3:] 的误差混合。',
        '不同方法的比较使用同场景、同任务和相同recipient；如覆盖不全，配对比较的覆盖量单独存入summary.json。',
        '已知参数是相同任务的参考，不计算“恢复Oracle空间百分比”。负收益保留，测试结果不用于再选预算或超参数。','']
    (out/'summary.md').write_text('\n'.join(lines))
    marker=out/'complete.json'
    if status=='COMPLETE':write(marker,dict(status='COMPLETE',summary=artifact(out/'summary.json'),test_read=True,expected=51))
    elif marker.exists():marker.unlink()
    print(f'{status}: {result["complete"]}/51')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','results','out'):parser.add_argument('--'+name,required=True)
    collect(parser.parse_args())
