"""Private sampling plans with explicit independent units for wrong donors.

This module plans identities and episode slots; it never generates or opens
test data. The caller supplies systems in their original generation order.
Within an iid block every system is used once as a wrong donor per replicate.
Different blocks share neither systems nor episodes. A fixed factorial census
instead uses independent whole-census episode replicates, conditional on the
fixed physical grid. Those are different estimands and are never pooled.
"""
import hashlib,json
import numpy as np

SCHEMA='vec.private-evaluation-blocks.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def episode_slots(replicate,replicates,include_certificate=False):
    if not 0<=replicate<replicates:raise ValueError('episode replicate outside registered support')
    # cold index 3*r selects the existing pulse_050 template. Control uses a
    # separate initial episode, after all scored cold-query episode indices.
    result=dict(forced=[replicate],mass=[2*replicate,2*replicate+1],free=[2*replicate,2*replicate+1],
        cold=[3*replicate],moving=[replicate],control_cold=[3*replicates+3*replicate],
        interference_moving=list(range(replicates+4*replicate,replicates+4*replicate+4)))
    if include_certificate:result.update(glide=[replicate],static=[replicate])
    return result


def _derangement(size,rng):
    while True:
        order=rng.permutation(size)
        if np.all(order!=np.arange(size)):return order.tolist()


def build_plan(system_keys,*,stratum,mode,replicates,seed,block_size=None,include_certificate=False):
    keys=list(system_keys)
    if len(keys)<2 or any(not isinstance(k,str) or not k for k in keys) or len(set(keys))!=len(keys):
        raise ValueError('distinct opaque system identities are required')
    if not isinstance(seed,int) or seed<0 or not isinstance(replicates,int) or replicates<1:raise ValueError('invalid plan randomness or replicates')
    if not isinstance(stratum,str) or not stratum:raise ValueError('one explicit population stratum is required')
    if mode=='iid_system_blocks':
        if not isinstance(block_size,int) or block_size<2 or len(keys)%block_size:raise ValueError('complete equal-size iid blocks required')
        groups=[keys[i:i+block_size] for i in range(0,len(keys),block_size)]
        independent_unit='independent physical-system block; all donor permutations remain within block'
    elif mode=='fixed_census_episode_blocks':
        if block_size not in (None,len(keys)):raise ValueError('fixed census cannot be split into iid physical populations')
        block_size=len(keys);groups=[keys]*replicates
        independent_unit='whole fixed-physical-grid episode-and-assignment replicate, conditional on that grid'
    else:raise ValueError('unregistered independent-unit mode')
    cases=[]
    for bi,group in enumerate(groups):
        reps=range(replicates) if mode=='iid_system_blocks' else (bi,)
        for rep in reps:
            order=_derangement(len(group),np.random.default_rng(np.random.SeedSequence([seed,bi,rep])))
            for si,key in enumerate(group):
                cases.append(dict(case_id=f'b{bi:04d}/s{si:04d}/r{rep:03d}',block_id=f'b{bi:04d}',system_key=key,
                    generation_index=keys.index(key),replicate=rep,wrong_system_key=group[order[si]],
                    episode_slots=episode_slots(rep,replicates,include_certificate)))
    plan=dict(schema=SCHEMA,stratum=stratum,mode=mode,replicates=replicates,seed=seed,block_size=block_size,
        system_generation_order=keys,independent_unit=independent_unit,cases=cases,
        test_generated=False,selection='generation order only; no sorting or grouping by physical values, image eligibility or method outcomes')
    if include_certificate:plan['include_certificate']=True
    plan['plan_sha256']=digest(plan);validate_plan(plan)
    return plan


def validate_plan(plan):
    payload={k:v for k,v in plan.items() if k!='plan_sha256'}
    if plan.get('schema')!=SCHEMA or plan.get('plan_sha256')!=digest(payload):raise ValueError('evaluation plan content differs')
    # Reconstruct the structural expectations, not the random draw. This also
    # rejects a consistently rehashed plan with cross-block donor leakage.
    keys=plan['system_generation_order'];mode=plan['mode'];size=plan['block_size'];reps=plan['replicates']
    if len(set(keys))!=len(keys) or size<2 or len(keys)%size or reps<1:raise ValueError('invalid evaluation plan population')
    if mode not in ('iid_system_blocks','fixed_census_episode_blocks'):raise ValueError('unknown block mode')
    if mode=='fixed_census_episode_blocks' and size!=len(keys):raise ValueError('partial fixed census')
    if len(plan['cases'])!=len(keys)*reps:raise ValueError('incomplete evaluation cases')
    observed={};seen=set()
    for case in plan['cases']:
        key=case['system_key'];rep=case['replicate'];gi=case['generation_index']
        if gi not in range(len(keys)) or keys[gi]!=key or rep not in range(reps):raise ValueError('system or replicate identity differs')
        bi=gi//size if mode=='iid_system_blocks' else rep
        if case['block_id']!=f'b{bi:04d}' or case['case_id']!=f'b{bi:04d}/s{gi%size:04d}/r{rep:03d}':raise ValueError('case block identity differs')
        identity=(key,rep)
        if identity in seen:raise ValueError('duplicate system replicate')
        seen.add(identity);group=keys[bi*size:(bi+1)*size] if mode=='iid_system_blocks' else keys
        if case['wrong_system_key'] not in group or case['wrong_system_key']==key:raise ValueError('wrong donor leaves block or matches recipient')
        if case['episode_slots']!=episode_slots(rep,reps,plan.get('include_certificate',False)):raise ValueError('episode slots differ from disjoint allocation')
        observed.setdefault((bi,rep),[]).append(case['wrong_system_key'])
    if any(len(donors)!=size or len(set(donors))!=size for donors in observed.values()):raise ValueError('unbalanced wrong-donor usage')
    return plan
