"""Paired inference respecting donor reuse and fixed training artifacts.

One call contains exactly one stratum/query/horizon/metric profile. All planned
cases and all registered training seeds must be present. Missing or failed
cases cannot be silently dropped. Seeds are averaged as fixed artifacts within
independent evaluation blocks; seed spread is reported separately, never used
to multiply the number of independent systems. Intervals are approximate.
"""
import math
import numpy as np
from scipy.stats import t
from .evaluation_blocks import validate_plan


def paired_block_summary(records,plan,*,condition,reference,metric,profile_key,seeds,
                         alpha=.05,family_size=1,bootstrap_draws=4096,bootstrap_seed=991054,min_blocks=16):
    validate_plan(plan);seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('explicit distinct fixed training seeds required')
    if condition==reference or not profile_key or not 0<alpha<1:raise ValueError('invalid comparison profile')
    if not isinstance(family_size,int) or family_size<1 or bootstrap_draws<100 or min_blocks<2:raise ValueError('invalid registered inference budget')
    cases={c['case_id']:c for c in plan['cases']};selected={}
    for row in records:
        if row['profile_key']!=profile_key:raise ValueError('mixed query/horizon/population profiles')
        if row['case_id'] not in cases or row['model_seed'] not in seeds:raise ValueError('unplanned case or training seed')
        if row['condition'] not in (condition,reference):continue
        identity=(row['case_id'],row['model_seed'],row['condition'])
        if identity in selected:raise ValueError('duplicate condition outcome')
        if row.get('status')!='OK' or not np.isfinite(row['metrics'][metric]):raise ValueError('failed/nonfinite outcome requires its prespecified failure policy')
        selected[identity]=row
    if len(selected)!=len(cases)*len(seeds)*2:raise ValueError('incomplete paired outcomes; no successful-case-only analysis')
    blocks=sorted({c['block_id'] for c in cases.values()});by_block={b:[] for b in blocks};by_seed={s:[] for s in seeds}
    by_system={}
    for case_id,case in cases.items():
        differences=[];commitments=set()
        for seed in seeds:
            left=selected[(case_id,seed,condition)];right=selected[(case_id,seed,reference)]
            for row in (left,right):
                if not row.get('query_fingerprint') or not row.get('target_fingerprint'):raise ValueError('missing paired query/target commitment')
                commitments.add((row['query_fingerprint'],row['target_fingerprint']))
            delta=float(left['metrics'][metric])-float(right['metrics'][metric]);differences.append(delta);by_seed[seed].append(delta)
        if len(commitments)!=1:raise ValueError('query or target changed across conditions/seeds')
        by_system.setdefault((case['block_id'],case['system_key']),[]).append(float(np.mean(differences)))
    for (block,system),values in by_system.items():by_block[block].append(float(np.mean(values)))
    if len({len(v) for v in by_block.values()})!=1:raise ValueError('unequal independent-block physical-system counts')
    values=np.asarray([np.mean(by_block[b]) for b in blocks]);mean=float(values.mean());count=len(values)
    standard_error=float(values.std(ddof=1)/np.sqrt(count)) if count>1 else None
    result=dict(contrast=condition+' minus '+reference,profile_key=profile_key,metric=metric,mean=mean,
        independent_blocks=count,physical_systems=len(plan['system_generation_order']),episode_replicates=plan['replicates'],
        training_seeds=seeds,seed_means={str(s):float(np.mean(v)) for s,v in by_seed.items()},
        seed_policy='fixed-artifact average within each block; seeds are not independent test-system replicates',
        independent_unit=plan['independent_unit'],plan_sha256=plan['plan_sha256'],block_means=values.tolist(),
        standard_error=standard_error,alpha=alpha,registered_family_size=family_size,
        precision_status='ESTIMATED' if count>=min_blocks else 'INSUFFICIENT_INDEPENDENT_BLOCKS',
        interval_assumptions='independent preallocated blocks; approximate cluster-mean t intervals; no exact finite-sample or power guarantee')
    if count>=min_blocks:
        critical=float(t.ppf(1-alpha/2,count-1));family_critical=float(t.ppf(1-alpha/(2*family_size),count-1))
        rng=np.random.default_rng(bootstrap_seed);boot=[]
        for start in range(0,bootstrap_draws,256):
            indices=rng.integers(count,size=(min(256,bootstrap_draws-start),count));boot.extend(values[indices].mean(1))
        result.update(interval=[mean-critical*standard_error,mean+critical*standard_error],
            bonferroni_family_interval=[mean-family_critical*standard_error,mean+family_critical*standard_error],
            bootstrap_sensitivity_interval=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
            bootstrap_draws=bootstrap_draws,bootstrap_seed=bootstrap_seed,
            reference_halfwidth=critical*standard_error,
            relative_halfwidth=None if mean==0 else critical*standard_error/abs(mean))
    return result
