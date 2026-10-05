"""Physical-system pairing, complete grids, and development-only pilot gates."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import read, sha, digest
from .engine import jobs,job_name


def load_rows(run_dir):
    root=Path(run_dir);complete=read(root/'COMPLETE.json');run=read(root/'RUN.json')
    if complete.get('smoke') or run.get('smoke'):raise ValueError('smoke outputs cannot enter scientific aggregates')
    if complete['run_sha256']!=sha(root/'RUN.json') or complete['checkpoint_sha256']!=sha(root/'head.pt'):
        raise ValueError('reader artifact changed')
    for filename,key in (('PROBE.json','probe_sha256'),('probe_vectors.npz','probe_vectors_sha256')):
        if sha(root/filename)!=complete[key]:raise ValueError('probe artifact changed')
    result_path=root/'evaluation/RESULT.json'
    if complete['evaluation_sha256']!=sha(result_path):raise ValueError('evaluation result changed')
    result=read(result_path);rows=[]
    for shard in result['shards']:
        for kind in ('vectors','rows'):
            if sha(result_path.parent/shard[kind])!=shard[kind+'_sha256']:raise ValueError('row/vector artifact changed')
        values=read(result_path.parent/shard['rows'])
        if len(values)!=shard['count']:raise ValueError('shard row count changed')
        rows.extend(values)
    if len(rows)!=result['all_cases']:raise ValueError('incomplete row denominator')
    return rows,complete,run


def paired_key(row):return row['system_id'],row['query_id']


def assert_pairing(a,b):
    if len(a)!=len(b) or [paired_key(r) for r in a]!=[paired_key(r) for r in b]:
        raise ValueError('recipient identities/order differ')
    fields=('donor_id','donor_system_id','horizon','mass','drag','stiffness','query_episode','donor_episode')
    for ra,rb in zip(a,b):
        if any(ra[k]!=rb[k] for k in fields):raise ValueError('donor/query/physics pairing differs')


def bootstrap_mean(values,draws=10000,seed=20260919):
    v=np.asarray(values,np.float64)
    if v.ndim!=1 or len(v)<2:raise ValueError('at least two physical systems required')
    rng=np.random.default_rng(seed)
    samples=np.concatenate([v[rng.integers(len(v),size=(min(256,draws-i),len(v)))].mean(1) for i in range(0,draws,256)])
    return dict(mean=float(v.mean()),interval95=np.quantile(samples,[.025,.975]).tolist(),systems=len(v),
        unit='physical_system',scope='conditional on fitted sources/readers and development population')


def spring_summary(root):
    expected=jobs('springworld');data={};recipes=set();pairings=set();normalizations=set();secondary=defaultdict(list);horizons=defaultdict(lambda:defaultdict(list));probes=[];global_ref=None
    for job in expected:
        rows,c,r=load_rows(Path(root)/job_name(job));recipes.add(r.get('recipe_sha256',r['protocol_sha256']));normalizations.add(digest(r['normalization']))
        pairings.add(c.get('evaluation_pairing_sha256','legacy_without_vector_pairing_digest'))
        if global_ref is None:global_ref=rows
        else:assert_pairing(global_ref,rows)
        if c['job']!=job:raise ValueError('job identity mismatch')
        data[job['source_seed'],job['reader_seed'],job['arm']]=(rows,c)
        denominator=sum(v['split_weight'] for v in rows)
        if not np.isclose(denominator,1.):raise ValueError('full mixture weights differ')
        secondary[job['arm']].append(float(sum(v['aggregate_error']*v['split_weight'] for v in rows)))
        for h in (1,2,4,8,16):
            sel=[v for v in rows if v['horizon']==h]
            horizons[job['arm']][h].append(float(sum(v['aggregate_error']*v['split_weight'] for v in sel)/sum(v['split_weight'] for v in sel)))
        if job['arm']=='persistent' and job['reader_seed']==0:
            probes.append(dict(source_seed=job['source_seed'],**read(Path(root)/job_name(job)/'PROBE.json')))
    if len(recipes)!=1 or len(normalizations)!=1 or len(pairings)!=1:raise ValueError('recipe/normalization/target population differs across grid')
    deltas=defaultdict(lambda:defaultdict(list));means=defaultdict(list);cells=[]
    for s in range(3):
        for r in range(3):
            ref,rc=data[s,r,'persistent'];ref=[v for v in ref if v['primary']]
            for arm in ('null','persistent','decode','oracle'):
                rows,c=data[s,r,arm];rows=[v for v in rows if v['primary']]
                assert_pairing(ref,rows)
                if c['training_sequence_sha256']!=rc['training_sequence_sha256']:raise ValueError('unpaired training batches')
                groups=defaultdict(list)
                for a,b in zip(rows,ref):
                    groups[a['system_id']].append(a['aggregate_error'])
                    deltas[arm][a['system_id']].append(a['aggregate_error']-b['aggregate_error'])
                means[arm].append(float(np.mean([np.mean(v) for v in groups.values()])))
                cells.append(dict(source_seed=s,reader_seed=r,arm=arm,mse=means[arm][-1]))
    return dict(complete_readers=36,test_read=False,stage='development',recipe_sha256=next(iter(recipes)),
        evaluation_pairing_sha256=next(iter(pairings)),
        mean_mse={k:float(np.mean(v)) for k,v in means.items()},
        secondary_full_mixture_mse={k:float(np.mean(v)) for k,v in secondary.items()},
        secondary_horizon_mse={k:{h:float(np.mean(v)) for h,v in d.items()} for k,d in horizons.items()},
        frozen_p_probe_by_source=probes,
        primary_decode_minus_persistent=bootstrap_mean([np.mean(v) for v in deltas['decode'].values()]),
        secondary_vs_persistent={k:bootstrap_mean([np.mean(v) for v in deltas[k].values()]) for k in ('null','oracle')},
        interval_policy='primary nominal 95%; secondary descriptive, no familywise claim',
        source_reader_cell_mse=cells,aggregation='pair queries first, equal readers then sources, physical-system bootstrap')


def baseline_summary(root):
    data={};probes=[];recipes=set();norms=set()
    for method,stage in (('Both','development'),('RelInfoNCE','baseline')):
        for source in range(3):
            for reader in range(3):
                for arm in ('null','persistent'):
                    job=dict(environment='springworld',stage=stage,method=method,source_seed=source,reader_seed=reader,arm=arm)
                    path=Path(root)/job_name(job);rows,complete,run=load_rows(path)
                    if complete['job']!=job:raise ValueError('baseline grid job differs')
                    recipes.add(digest(run['reader']));norms.add(digest(run['normalization']))
                    data[method,source,reader,arm]=(rows,complete)
                    if reader==0 and arm=='persistent':probes.append(dict(method=method,source_seed=source,**read(path/'PROBE.json')))
    if len(recipes)!=1 or len(norms)!=1:raise ValueError('reader recipe or target normalization differs across methods')
    effects=defaultdict(list);values=defaultdict(lambda:defaultdict(list));cells=[]
    for s in range(3):
        for r in range(3):
            reference,rc=data['Both',s,r,'persistent']
            for method in ('Both','RelInfoNCE'):
                for arm in ('null','persistent'):
                    rows,c=data[method,s,r,arm];assert_pairing(reference,rows)
                    if rc['training_sequence_sha256']!=c['training_sequence_sha256']:raise ValueError('baseline training sample schedule differs')
                    groups=defaultdict(list)
                    for row in rows:
                        if row['primary']:groups[row['system_id']].append(row['aggregate_error'])
                    for sid,v in groups.items():values[method+':'+arm][sid].append(float(np.mean(v)))
                    cells.append(dict(method=method,source_seed=s,reader_seed=r,arm=arm,mse=float(np.mean([np.mean(v) for v in groups.values()]))))
    ids=sorted(values['Both:persistent'])
    a=np.asarray([np.mean(values['Both:persistent'][i]) for i in ids]);b=np.asarray([np.mean(values['RelInfoNCE:persistent'][i]) for i in ids])
    if np.any(b<=0):raise ValueError('baseline risk must be positive for relative gain')
    rng=np.random.default_rng(20260919);gains=[]
    for _ in range(10000):
        ix=rng.integers(len(ids),size=len(ids));gains.append(float(1-a[ix].mean()/b[ix].mean()))
    return dict(stage='development',test_read=False,comparison='Both vs fixed-recipe FCRL-style temporal RelInfoNCE adaptation; no result-dependent selection',
        mean_mse={k:float(np.mean([np.mean(v) for v in groups.values()])) for k,groups in values.items()},
        closest_minus_both=bootstrap_mean(b-a),relative_gain=dict(value=float(1-a.mean()/b.mean()),interval95=np.quantile(gains,[.025,.975]).tolist()),
        cells=cells,probes=probes,statistical_unit='physical_system',interval_scope='conditional on the fitted source/reader grid')


def pilot_summary(root,formal=False):
    readers=range(3) if formal else [0]
    pooled=[];per_seed=[];artifacts={};normalizations=set();protocols=set();global_ref=None;probes=[]
    for s in range(3):
        reader_effect=[];ref=None;dominance=None
        for reader in readers:
            store={}
            for method in ('G1','G2'):
                for arm in ('null','matched'):
                    job=dict(environment='pokeworld',stage='pilot' if reader==0 else 'formal',method=method,source_seed=s,reader_seed=reader,arm=arm)
                    path=Path(root)/job_name(job)
                    rows,c,r=load_rows(path)
                    if c['job']!=job:raise ValueError('wrong pilot/formal cell')
                    rows=[v for v in rows if v['primary']]
                    for artifact in ('COMPLETE.json','RUN.json','head.pt','evaluation/RESULT.json','PROBE.json','probe_vectors.npz'):
                        artifacts[str((path/artifact).resolve())]=sha(path/artifact)
                    normalizations.add(digest(r['normalization']));protocols.add(r['protocol_sha256'])
                    if reader==0 and arm=='matched':probes.append(dict(method=method,source_seed=s,**read(path/'PROBE.json')))
                    if ref is None:ref=rows
                    else:assert_pairing(ref,rows)
                    if global_ref is None:global_ref=rows
                    else:
                        assert_pairing(global_ref,rows)
                        if [x['sensitivity_dominance'] for x in global_ref]!=[x['sensitivity_dominance'] for x in rows]:
                            raise ValueError('moderator differs across frozen source recipes')
                    store[method,arm]=(np.array([v['aggregate_error'] for v in rows]),c['training_sequence_sha256'])
            if len({v[1] for v in store.values()})!=1:raise ValueError('pilot batch schedules differ')
            reader_effect.append((store['G1','null'][0]-store['G1','matched'][0])-(store['G2','null'][0]-store['G2','matched'][0]))
        d=np.asarray([v['sensitivity_dominance'] for v in ref],np.float64)
        u=np.mean(reader_effect,axis=0)
        if not np.isfinite(d).all() or d.std()<1e-12:raise ValueError('sensitivity moderator is uninformative')
        counts={sid:sum(r['system_id']==sid for r in ref) for sid in {r['system_id'] for r in ref}}
        w=np.asarray([1/counts[r['system_id']] for r in ref])
        design=np.column_stack((np.ones(len(d)),d))
        coef=np.linalg.lstsq(design*np.sqrt(w[:,None]),u*np.sqrt(w),rcond=None)[0]
        # Bins display trends only. Continuous moderator is the primary diagnostic.
        edges=np.quantile(d,np.linspace(0,1,6));bins=[]
        for i in range(5):
            mask=(d>=edges[i]) & ((d<=edges[i+1]) if i==4 else (d<edges[i+1]))
            selected_ids=np.asarray([r['system_id'] for r in ref])[mask]
            bin_counts={sid:sum(selected_ids==sid) for sid in set(selected_ids)}
            bin_weights=np.asarray([1/bin_counts[sid] for sid in selected_ids])
            bins.append(dict(left=float(edges[i]),right=float(edges[i+1]),queries=int(mask.sum()),systems=len(bin_counts),
                dominance=None if not mask.any() else float(np.average(d[mask],weights=bin_weights)),
                delta_utility=None if not mask.any() else float(np.average(u[mask],weights=bin_weights))))
        per_seed.append(dict(source_seed=s,slope=float(coef[1]),bins=bins))
        pooled.append(u)
    if len(normalizations)!=1 or len(protocols)!=1:raise ValueError('G1/G2 normalization/protocol differs')
    slopes=[x['slope'] for x in per_seed]
    result=dict(stage='formal' if formal else 'pilot',test_read=False,per_seed=per_seed,
        positive_slopes=sum(v>0 for v in slopes),negative_slopes=sum(v<0 for v in slopes),
        p_value_used_for_gate=False,automatic_go=False,
        gate='review continuous slopes and binned structure; at least 2/3 same directions; record reversed trends; visibly flat stops expansion',
        artifacts=artifacts,protocol_sha256=next(iter(protocols)),
        probes=probes,
        interpretation='development grid including pilot reader; not independent confirmation' if formal else 'development gate only')
    if formal:
        # Cluster bootstrap of the continuous slope, after fixed seed averaging.
        u=np.mean(pooled,axis=0);ids=np.asarray([r['system_id'] for r in ref]);unique=np.unique(ids)
        rng=np.random.default_rng(20260919);bs=[]
        for _ in range(4000):
            sampled=rng.choice(unique,len(unique));indices=np.concatenate([np.flatnonzero(ids==v) for v in sampled])
            a=design[indices];weights=w[indices];yy=u[indices]
            bs.append(float(np.linalg.lstsq(a*np.sqrt(weights[:,None]),yy*np.sqrt(weights),rcond=None)[0][1]))
        result['continuous_slope_interval95']=np.quantile(bs,[.025,.975]).tolist()
        result['statistical_unit']='physical_system; fixed source and reader grid'
    return result


def verify_gate(path,protocol_sha256=None):
    gate=read(path)
    if gate.get('decision')!='go' or not gate.get('trend_review') or gate.get('p_value_used',True) or gate.get('binned_curve_nonflat') is not True:
        raise PermissionError('documented non-p-value pilot go decision required')
    summary=read(gate['summary'])
    if sha(gate['summary'])!=gate['summary_sha256'] or summary['stage']!='pilot':raise ValueError('pilot summary changed')
    if protocol_sha256 is not None and summary['protocol_sha256']!=protocol_sha256:raise ValueError('pilot belongs to another protocol')
    for p,h in summary['artifacts'].items():
        if sha(p)!=h:raise ValueError('pilot artifact changed')
    if max(summary['positive_slopes'],summary.get('negative_slopes',0))<2:raise PermissionError('pilot lacks 2/3 consistent directions')
    return gate
