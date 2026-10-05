"""Deterministic, pre-training donor manifests from audited metadata and AB+C caches."""
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np

VERSION = 'cophy-donor-manifest-v3'


def qualify_objects(raw_rows, cache, *, scene, split, field_indices, gravity_branch):
    """No outcome fields are consulted. Metadata remains sampler-only data."""
    if split not in {'train', 'val', 'test'} or not field_indices:
        raise ValueError('Need a known split and audited physical field selection')
    if scene == 'blocktower' and gravity_branch not in {'constant_verified', 'varying_verified'}:
        raise ValueError('Blocktower relations require a qualified gravity branch')
    records, seen = [], set()
    for row in raw_rows:
        ident, slot = str(row['id']), int(row['slot'])
        if row['split'] != split or (ident, slot) in seen:
            raise ValueError('Cross-split or duplicate metadata row')
        seen.add((ident, slot))
        item = cache[ident]
        if item.get('cache_version') != 'ab_c_float32_v2':
            raise ValueError('Need an audited AB+C cache')
        pa, pc = np.asarray(item['presence_ab']), np.asarray(item['presence_c'])
        if pa.shape != pc.shape or not 0 <= slot < len(pa):
            raise ValueError('Metadata/cache slots disagree')
        stratum = [row['known_type']]
        if scene == 'blocktower' and gravity_branch == 'varying_verified':
            gravity = row.get('raw_gravity')
            if gravity is None or len(gravity) != 2 or not np.isfinite(gravity).all():
                raise ValueError('Reliable global gravity missing from object stratum')
            stratum.append(list(gravity))
        visible_ab, visible_c = bool(pa[slot] > 0), bool(pc[slot] > 0)
        records.append({'id': ident, 'slot': slot, 'stratum': stratum, 'scene': scene, 'split': split,
            'physical': [row['physical'][i] for i in field_indices],
            'recipient': bool(row['in_C'] and visible_ab and visible_c),
            'recipient_candidate': bool(row['in_C']), 'donor': visible_ab,
            'known_type': row['known_type']})
    return sorted(records, key=lambda r: (r['id'], r['slot']))


def _group(row):
    return (row['slot'], json.dumps(row['stratum'], sort_keys=True, separators=(',', ':')))


def _choose(candidates, seed, scene, split, recipient, condition):
    if not candidates:
        return None
    token = json.dumps([VERSION, seed, scene, split, recipient['id'], recipient['slot'], condition], separators=(',', ':'))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little'))
    row = candidates[int(rng.integers(len(candidates)))]
    return {'id': row['id'], 'slot': row['slot']}


def build_donor_manifests(records, *, scene, split, physical_fields, seed=20260911):
    """Donors come from the SAME held-out split; all training seeds share manifests.

    The joint primary queue needs both Correct and Wrong-any. Wrong-1 keeps a
    separate per-factor subset so it cannot remove opportunities from Use.
    """
    if split not in {'val', 'test'} or len(set(physical_fields)) != len(physical_fields):
        raise ValueError('Evaluation manifest requires a held-out split and unique fields')
    groups = defaultdict(list)
    denominator = sum(r['recipient_candidate'] for r in records)
    for row in records:
        if row.get('scene') != scene or row.get('split') != split:
            raise ValueError('Donor pool is not from the declared scene and held-out split')
        if len(row['physical']) != len(physical_fields):
            raise ValueError('Physical field schema mismatch')
        if row['donor']:
            groups[_group(row)].append(row)
    for pool in groups.values():
        pool.sort(key=lambda r: (r['id'], r['slot']))
    primary, wrong1 = [], []
    correct_n = wrong_n = visible_n = 0
    for row in sorted(records, key=lambda r: (r['id'], r['slot'])):
        if not row['recipient']:
            continue
        visible_n += 1
        candidates = [d for d in groups[_group(row)] if d['id'] != row['id']]
        correct = [d for d in candidates if d['physical'] == row['physical']]
        wrong = [d for d in candidates if d['physical'] != row['physical']]
        correct_n += bool(correct); wrong_n += bool(wrong)
        if not correct or not wrong:
            continue
        item = {'recipient': row['id'], 'focal': row['slot'], 'known_type': row['known_type'],
                'Correct': _choose(correct, seed, scene, split, row, 'Correct'),
                'Wrong-any': _choose(wrong, seed, scene, split, row, 'Wrong-any')}
        primary.append(item)
        for factor, name in enumerate(physical_fields):
            candidates_one = [d for d in wrong if
                [i for i,(a,b) in enumerate(zip(d['physical'],row['physical'])) if a != b] == [factor]]
            donor = _choose(candidates_one, seed, scene, split, row, 'Wrong-1:'+name)
            if donor is not None:
                wrong1.append(dict(item, factor=name, **{'Wrong-1':donor}))
    frac = lambda n,d: n/d if d else 0.
    per_factor=Counter(r['factor'] for r in wrong1)
    reused=Counter((arm,r[arm]['id'],r[arm]['slot']) for r in primary for arm in ['Correct','Wrong-any'])
    coverage = {'candidate_opportunities': denominator, 'visually_eligible': visible_n,
        'correct_coverage':frac(correct_n,denominator), 'wrong_any_coverage':frac(wrong_n,denominator),
        'primary_joint_coverage':frac(len(primary),denominator), 'primary_opportunities':len(primary),
        'recipient_count':len({r['recipient'] for r in primary}),
        'wrong1_factor_coverage':{f:frac(per_factor[f],len(primary)) for f in physical_fields},
        'max_donor_reuse':max(reused.values(),default=0),
        'assay_status':'QUALIFIED' if denominator and min(correct_n,wrong_n)/denominator>=.8 else 'EXPLORATORY_COVERAGE'}
    common={'version':VERSION,'scene':scene,'split':split,'donor_split':split,'seed':seed,
            'physical_fields':list(physical_fields),'coverage':coverage}
    return (dict(common,kind='primary',rows=primary), dict(common,kind='wrong1',rows=wrong1))


def parameter_documents(raw_by_split, *, scene, slots, object_fields, include_gravity=False):
    """Train-only standardization; rows have already been mapped to color slots."""
    names=list(object_fields)+(['gravity_x','gravity_y'] if include_gravity else [])
    train=[]; examples={}
    for split,rows in raw_by_split.items():
        if split not in {'train','val','test'}:
            raise ValueError('Unknown parameter split')
        storage={}
        for row in rows:
            if row['split'] != split:
                raise ValueError('Parameter records cross split boundary')
            values=list(row['raw_physical'])+(list(row['raw_gravity']) if include_gravity else [])
            if len(values)!=len(names) or not np.isfinite(values).all():
                raise ValueError('Parameter fields incomplete or nonfinite')
            ident=str(row['id'])
            storage.setdefault(ident,np.zeros((slots,len(names)),np.float32))[row['slot']]=values
            if split=='train': train.append(values)
        examples[split]={ident:array.tolist() for ident,array in storage.items()}
    if not train:
        raise ValueError('Parameter normalization requires active train objects')
    train=np.asarray(train,dtype=float); mean=train.mean(0); scale=train.std(0); scale[scale<1e-8]=1.
    fields=[{'name':name,'train_mean':float(mean[i]),'train_scale':float(scale[i])} for i,name in enumerate(names)]
    return {split:{'version':'cophy-parameters-v3','scene':scene,'split':split,
             'all_varying_parameters_included':True,'fields':fields,'examples':rows}
            for split,rows in examples.items()}
