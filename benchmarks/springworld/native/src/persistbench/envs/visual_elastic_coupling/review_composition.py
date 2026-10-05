"""System-level development composition summaries; keep all observed failures."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(report):
    systems=report['systems'];conditions=tuple(c['condition'] for c in systems[0]['conditions'])
    metric_names=tuple(systems[0]['conditions'][0]['prediction_errors']['16'])
    unique=sorted(set(s['system_index'] for s in systems));groups={i:[s for s in systems if s['system_index']==i] for i in unique}
    results={};contrasts=[];rng=np.random.default_rng(982753)
    bootstrap=rng.integers(0,len(unique),(4096,len(unique)))
    for h in ('16','32'):
        results[h]={}
        for metric in metric_names:
            values={}
            for condition in conditions:
                values[condition]=np.array([np.mean([next(c for c in s['conditions'] if c['condition']==condition)['prediction_errors'][h][metric] for s in groups[i]]) for i in unique])
            results[h][metric]={condition:dict(mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(v.max())) for condition,v in values.items()}
            for reference in ('independent_MM','independent_FF','repeated_M','repeated_F','null'):
                diff=values[reference]-values['mixed_MF'];ci=np.quantile(diff[bootstrap].mean(1),[.025,.975])
                contrasts.append(dict(horizon=int(h),metric=metric,reference=reference,comparison='reference minus mixed_MF',
                    mean_difference=float(diff.mean()),system_bootstrap_95=ci.tolist(),positive_systems=int((diff>0).sum()),systems=len(diff),
                    interpretation='development interval; not a sealed confirmatory p-value'))
    repeated_equal=True;query_equal=True;paired_budget_equal=True
    for system in systems:
        rows={c['condition']:c for c in system['conditions']}
        repeated_equal &= all(rows[c]['prediction_errors']==rows['repeated_'+c]['prediction_errors'] for c in ('M','F'))
        q={c['query_sha256'] for c in system['conditions'] if 'query_sha256' in c};query_equal &= len(q)==1
        paired_budget_equal &= len({rows[c]['processed_frames'] for c in ('repeated_M','repeated_F','independent_MM','independent_FF','mixed_MF')})==1
    estimates={}
    for condition in conditions:
        if 'posterior_mean' not in systems[0]['conditions'][conditions.index(condition)]:continue
        rows=[next(c for c in s['conditions'] if c['condition']==condition) for s in systems]
        truth=np.asarray([s['theta'] for s in systems]);pred=np.asarray([c['posterior_mean'] for c in rows]);interval=np.asarray([c['posterior_interval_80'] for c in rows])
        rel=np.abs(pred/truth-1);covered=(truth>=interval[:,0])&(truth<=interval[:,1])
        estimates[condition]=dict(relative_error_median=np.median(rel,axis=0).tolist(),relative_error_p90=np.quantile(rel,.9,axis=0).tolist(),
            relative_error_max=rel.max(0).tolist(),empirical_80_interval_coverage=covered.mean(0).tolist(),
            note='all system-replicate cases retained; local Gaussian uncertainty approximation')
    key=[c for c in contrasts if c['metric']=='position_mse_m2' and c['reference'] in ('independent_MM','independent_FF')]
    return dict(schema='vec.composition-review.v1.1',split='development',config=report['config'],systems=len(unique),cases=len(systems),
        execution_errors=report['errors'],checks=dict(repeated_observation_not_double_counted=repeated_equal,identical_query_bytes=query_equal,paired_processed_frame_budget_equal=paired_budget_equal),
        means=results,contrasts=contrasts,parameter_estimates=estimates,
        development_position_complementarity_both_horizons=not report['errors'] and all(c['system_bootstrap_95'][0]>0 for c in key),
        limitations=['M is weakly informative about stiffness, not structurally stiffness-free; F has exact mass/stiffness scale ambiguity.',
            'Joint raw-history fit is a measurement reference; compressed state and learned composition require separate measurements.',
            'Physical effort differs by evidence type; paired processed-frame budgets and unique-transition counts are explicit.',
            'Calibration systems and intervals are development evidence, not formal test results.'],test_read=False,formal_training=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('use a new report')
    a.output.write_text(json.dumps(summarize(json.loads(a.report.read_text())),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
