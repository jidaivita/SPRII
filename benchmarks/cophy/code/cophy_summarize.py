"""Paired fixed-manifest summaries; never select a seed, donor or checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from cophy_prepare_artifacts import frozen_write
from cophy_protocol import digest


def intervals(values, groups, repeats=10000, seed=20260911):
    """Shared group draws across all columns, with recipient-weighted estimates."""
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    if len(groups)!=len(values) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite aligned recipient-level values')
    labels=list(dict.fromkeys(groups));lookup={g:i for i,g in enumerate(labels)}
    idx=np.array([lookup[g] for g in groups])
    sums=np.zeros((len(labels),values.shape[1]));np.add.at(sums,idx,values)
    counts=np.bincount(idx,minlength=len(labels))
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,repeats,64):
        draws=rng.integers(len(labels),size=(min(64,repeats-start),len(labels)))
        samples.append(sums[draws].sum(1)/counts[draws].sum(1)[:,None])
    ci=np.quantile(np.concatenate(samples),[.025,.975],axis=0)
    return [{'estimate':float(values[:,j].mean()),'ci95':ci[:,j].tolist()} for j in range(values.shape[1])]


def recipient_errors(rows,metric='focal_mse'):
    by_id=defaultdict(list)
    for row in rows:by_id[row['recipient']].append(row['errors'])
    return {ident:{arm:float(np.mean([r[arm][metric] for r in records])) for arm in records[0]}
            for ident,records in by_id.items()}


def summarize(reports):
    if not reports:raise ValueError('No evaluations')
    identities={(r['scene'],r['split'],r['seed'],r['preflight_sha256']) for r in reports}
    if len(identities)!=1:raise ValueError('Compare one scene/split/seed and qualified dataset at a time')
    models={r['method']:r for r in reports}
    if len(models)!=len(reports) or not {'Native','A','Random'}<=models.keys():
        raise ValueError('Need the paired Native, A and Random, without duplicate variants')
    official={name:{r['id']:r for r in report['official_rows']} for name,report in models.items()}
    ids=list(official['Native'])
    if any(set(rows)!=set(ids) for rows in official.values()):raise ValueError('Official recipient IDs differ')
    for ident in ids:
        reference=official['Native'][ident]
        if any(row[ident]['CopyC']!=reference['CopyC'] or row[ident]['CopyC_gtmask']!=reference['CopyC_gtmask']
               for row in official.values()):raise ValueError('Shared frontend/Copy C discrepancy')
        if any((row[ident]['model'] is None)!=(reference['model'] is None) for row in official.values()):
            raise ValueError('Model-dependent evaluation coverage')
    scored=[i for i in ids if official['Native'][i]['model'] is not None]
    result={'scene':reports[0]['scene'],'split':reports[0]['split'],'seed':reports[0]['seed'],
        'official':{'CopyC':float(np.mean([official['Native'][i]['CopyC'] for i in scored])),
                    **{m:r['official']['model'] for m,r in models.items()}},
        'visual_coverage':len(scored)/len(ids),'comparison_is_same_seed':True,
        'intervals_condition_on_fixed_checkpoints_and_donors':True,
        'multiple_scene_significance_not_claimed':True,'assays':{}}
    columns=['Native_minus_A','Random_minus_A']
    values=[[official['Native'][i]['model']-official['A'][i]['model'],
             official['Random'][i]['model']-official['A'][i]['model']] for i in scored]
    result['official_paired']=dict(zip(columns,intervals(values,scored)))
    for kind in ['primary','wrong1']:
        manifests={models[m]['assays'][kind]['manifest_sha256'] for m in ['Native','A','Random']}
        if len(manifests)!=1:raise ValueError('Models do not use the same frozen manifest')
        rows={m:models[m]['assays'][kind]['rows'] for m in ['Native','A','Random']}
        keys=lambda rr:[(r['recipient'],r['focal'],r.get('factor')) for r in rr]
        if any(keys(rr)!=keys(rows['Native']) for rr in rows.values()):raise ValueError('Focal queues differ')
        if not rows['Native']:
            result['assays'][kind]={'status':'NO_OPPORTUNITIES','coverage':models['A']['assays'][kind]['coverage']}
            continue
        factors=sorted({r['factor'] for r in rows['Native']}) if kind=='wrong1' else [None]
        summary={}
        for factor in factors:
            rr={m:[r for r in data if factor is None or r['factor']==factor] for m,data in rows.items()}
            for metric in ['focal_mse','scene_mse']:
                reduced={m:recipient_errors(data,metric) for m,data in rr.items()}
                ids=list(reduced['Native']);n,a,r=[reduced[m] for m in ['Native','A','Random']]
                names=['ReuseGain','Use_A','DoD_Wrong_any','A_vs_Random_Correct',
                       'Use_Native','Use_Random','Specificity_any_A','Specificity_any_Native']
                vals=[[n[i]['Correct']-a[i]['Correct'],a[i]['Null']-a[i]['Correct'],
                    (a[i]['Wrong-any']-a[i]['Correct'])-(n[i]['Wrong-any']-n[i]['Correct']),
                    r[i]['Correct']-a[i]['Correct'],n[i]['Null']-n[i]['Correct'],r[i]['Null']-r[i]['Correct'],
                    a[i]['Wrong-any']-a[i]['Correct'],n[i]['Wrong-any']-n[i]['Correct']] for i in ids]
                if factor is not None:
                    names+=['Specificity_1_A','DoD_Wrong_1_exploratory']
                    for value,i in zip(vals,ids):value.extend([a[i]['Wrong-1']-a[i]['Correct'],
                        (a[i]['Wrong-1']-a[i]['Correct'])-(n[i]['Wrong-1']-n[i]['Correct'])])
                key=metric if factor is None else factor+'/'+metric
                summary[key]={'recipients':len(ids),'opportunities':len(rr['Native']),
                    'paired':dict(zip(names,intervals(vals,ids))),
                    'means':{m:{arm:float(np.mean([value[arm] for value in data.values()]))
                        for arm in next(iter(data.values()))} for m,data in reduced.items()}}
        result['assays'][kind]={'coverage':models['A']['assays'][kind]['coverage'],'results':summary}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',action='append',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();reports=[json.loads(Path(p).read_text()) for p in args.input]
    report=summarize(reports);report['inputs']=[{'path':str(Path(p).resolve()),'sha256':digest(p)} for p in args.input]
    frozen_write(args.output,report)
    print(json.dumps({k:report[k] for k in ['scene','split','seed','official','official_paired']},indent=2))
