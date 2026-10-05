"""Evaluator-owned candidate train/validation/transfer populations.

The factorial split leaves every individual factor value and every factor pair
represented in training. It deliberately withholds triple combinations. This
introduces a declared third-order training dependence; it is not a full-product
training prior. The primary continuous evaluation population is kept separate.
No population ID, seed, split, or theta belongs in a method input packet.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np

VERSION='vec.population-candidate.v1.1'
LEVELS=([.5,.8,1.1,1.4,1.7,2.],[.25,.5,.75,1.,1.25,1.5],[4.,8.2,12.4,16.6,20.8,25.])


def _key(theta):return hashlib.sha256(json.dumps([float(x) for x in theta],separators=(',',':')).encode()).hexdigest()[:24]


def factorial_partition():
    rows=[]
    for indices in itertools.product(range(6),repeat=3):
        slot=sum(indices)%6
        split='train' if slot<4 else ('validation' if slot==4 else 'test')
        theta=[LEVELS[i][index] for i,index in enumerate(indices)]
        rows.append(dict(system_key=_key(theta),theta=theta,factor_indices=list(indices),split=split,
            stratum='seen_factorial_training' if split=='train' else 'heldout_factorial_combinations'))
    return rows


def iid_population(count,seed,split,stratum,low=(.5,.25,4.),high=(2.,1.5,25.)):
    rng=np.random.default_rng(seed);theta=rng.uniform(low,high,(count,3))
    return [dict(system_key=_key(t),theta=t.tolist(),split=split,stratum=stratum) for t in theta]


def candidate_design():
    rows=factorial_partition()
    rows+=iid_population(64,984201,'validation','continuous_new_systems')
    rows+=iid_population(256,984301,'test','continuous_new_systems')
    # Exactly one novel interior factor value; other two are observed levels.
    for split,seed,repeats in (('validation',984211,2),('test',984311,4)):
        rng=np.random.default_rng(seed)
        for axis in range(3):
            for interval in range(5):
                for repeat in range(repeats):
                    t=[float(rng.choice(values)) for values in LEVELS]
                    fraction=float(rng.uniform(.2,.8));t[axis]=LEVELS[axis][interval]+fraction*(LEVELS[axis][interval+1]-LEVELS[axis][interval])
                    rows.append(dict(system_key=_key(t),theta=t,split=split,stratum='new_factor_value_interpolation',
                        changed_factor=('m','gamma','k')[axis],interval=interval))
    # These are proposed stress ranges, not yet admitted evaluation ranges.
    extrapolation=((.4,2.2),(.2,1.8),(3.5,28.))
    for split,seed,repeats in (('validation',984221,8),('test',984321,24)):
        rng=np.random.default_rng(seed)
        for axis in range(3):
            for side in range(2):
                for repeat in range(repeats):
                    # Other factors are continuous interior draws. Separate
                    # matched in-range counterparts are required by the assay.
                    t=rng.uniform([.65,.4,6.],[1.85,1.35,23.]).tolist()
                    t[axis]=extrapolation[axis][side]
                    counterpart=list(t);counterpart[axis]=LEVELS[axis][0 if side==0 else -1]
                    rows.append(dict(system_key=_key(t),theta=t,split=split,stratum='single_factor_range_extrapolation',
                        changed_factor=('m','gamma','k')[axis],side=('low','high')[side],
                        in_range_counterpart_theta=counterpart,admission='PENDING_PHYSICS_VISION_CALIBRATION'))
    config=dict(schema=VERSION,status='CANDIDATE_NOT_FROZEN',systems=rows,
        training_prior='uniform over144 registered factorial triples; balanced univariate and pairwise marginals, declared triple-combination holes',
        primary_test_prior='256 iid continuous systems, uniform independent physical m/gamma/k on original full support',
        nuisance_transfer='new independent episodes and nuisance seeds of training systems; IDs duplicated only under explicit same-system-nuisance assay',
        certificate='continuous free-motion scale pairs required separately; finite-bank recoverability and continuous structural identifiability are distinct',
        evaluation_strata='never pool interpolation, combinations, continuous ID, extrapolation and nuisance shift as one unnamed OOD score',
        random_donor_policy='fresh balanced assignment per training epoch/replicate, not a fixed invertible parameter map',
        test_access='only specification generated; no test rollouts, target labels, method results or sealed reads',
        formal_training=False,test_read=False)
    validate_design(config);return config


def validate_design(design):
    rows=design['systems'];keys=[r['system_key'] for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate physical system across population splits')
    grid=[r for r in rows if 'factor_indices' in r]
    if len(grid)!=216:raise ValueError('factorial candidate lost supported triples')
    counts={s:sum(r['split']==s for r in grid) for s in ('train','validation','test')}
    if counts!={'train':144,'validation':36,'test':36}:raise ValueError('factorial partition counts')
    for split in counts:
        subset=[r for r in grid if r['split']==split]
        for a,b in ((0,1),(0,2),(1,2)):
            marginal=np.zeros((6,6),int)
            for r in subset:marginal[r['factor_indices'][a],r['factor_indices'][b]]+=1
            if not np.all(marginal==len(subset)//36):raise ValueError('imbalanced factor-pair support')
    train=[r for r in grid if r['split']=='train']
    for recipient in train:
        t=recipient['theta']
        for relation,shared,varied in (('G1',(0,),(1,2)),('G2',(0,1),(2,))):
            candidates=[r for r in train if all(r['theta'][a]==t[a] for a in shared) and all(r['theta'][a]!=t[a] for a in varied)]
            if not candidates:raise ValueError('training relation has no legal donor: '+relation)
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('population attempt already exists')
    result=candidate_design();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(a.output),systems=len(result['systems']),status=result['status'],test_read=False)))


if __name__=='__main__':main()
