"""Read-only data admission checks independent of learned method scores."""
import argparse,collections,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .target_scaling import HORIZONS


def checked_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('manifest asset escapes bank')
    return path


def inspect_episode(job):
    root,row=job;issues=[];targets={};cold_identical={}
    for res,asset in row['assets'].items():
        path=checked_path(root,asset['path']);blob=path.read_bytes()
        if len(blob)!=asset['bytes'] or hashlib.sha256(blob).hexdigest()!=asset['sha256']:issues.append('visible_asset_digest_'+res)
        with np.load(path,allow_pickle=False) as data:
            images=data['images'];actions=data['actions'];times=data['timestamps']
            n=row['raw_frames']
            if images.dtype!=np.uint8 or images.shape!=(n,int(res),int(res)):issues.append('image_shape_dtype_'+res)
            if actions.shape!=(n-1,2) or not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-6):issues.append('actions_'+res)
            if times.shape!=(n,) or not np.allclose(times,np.arange(n)*.05,atol=1e-12,rtol=0):issues.append('time_index_'+res)
            if row['kind']=='cold':
                cold_identical[res]=bool(n>=2 and np.array_equal(images[0],images[1]) and not actions[0].any())
                if not cold_identical[res]:issues.append('cold_rest_contract_'+res)
            if row['kind']=='moving' and np.any(actions[:min(95,len(actions))]):issues.append('passive_prefix_force_'+res)
    # Truth is evaluator-owned here. Only train labels feed normalization checks.
    with np.load(checked_path(root,row['private_state_path']),allow_pickle=False) as data:state=data['state']
    if state.shape!=(row['raw_frames'],8) or not np.isfinite(state).all():issues.append('private_state_shape')
    if row['split']=='train' and row['anchor'] is not None:
        for h in HORIZONS:
            if row['anchor']+h<len(state):targets[h]=(state[row['anchor']+h]-state[row['anchor']]).tolist()
    return dict(episode_key=row['episode_key'],issues=issues,train_targets=targets)


def audit(root,workers=16):
    root=Path(root);start=time.monotonic();config=json.loads((root/'BANK_CONFIG.json').read_text())
    generation=json.loads((root/'BANK_REPORT.json').read_text());manifest_path=root/'MANIFEST.private.json'
    manifest=json.loads(manifest_path.read_text());rows=manifest['episodes'];problems=[]
    if config.get('test_generated') or config.get('test_read') or generation.get('test_read'):raise ValueError('test access in training bank')
    if any(r['split'] not in ('train','validation') for r in rows):raise ValueError('test/private split excluded from this audit')
    if generation['execution_errors']:problems.append('generation_execution_errors')
    if len(rows)!=config['job_count'] or generation['episodes']!=len(rows):problems.append('episode_count')
    for key in ('episode_key','seed'):
        if len({r[key] for r in rows})!=len(rows):problems.append('duplicate_'+key)
    systems={};groups=collections.defaultdict(list)
    for row in rows:
        system=(row['split'],row['system_key']);systems[system]=row['theta']
        groups[(row['split'],row['stratum'],row['kind'])].append(row)
        if row['system']['split']!=row['split'] or row['system']['system_key']!=row['system_key'] or row['system']['theta']!=row['theta']:
            problems.append('nested_system_metadata_mismatch')
    if len({tuple(t) for t in systems.values()})!=len(systems):problems.append('same_physics_in_distinct_population_splits')
    counts=collections.Counter((r['split'],r['system_key'],r['kind']) for r in rows)
    for (split,key),theta in systems.items():
        for kind,expected in config['counts'][split].items():
            wanted=min(expected,3) if config['smoke'] else expected
            if counts[(split,key,kind)]!=wanted:problems.append('per_system_episode_count')
    audits=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(inspect_episode,[(root,r) for r in rows]):audits.append(row)
    asset_failures=[r for r in audits if r['issues']]
    if asset_failures:problems.append('asset_or_causal_contract')
    statistics=json.loads((root/'TRAIN_TARGET_STATISTICS.json').read_text())
    if statistics['source_split']!='train':problems.append('normalization_source_split')
    statistics_errors={}
    for h in HORIZONS:
        values=np.asarray([r['train_targets'][h] for r in audits if h in r['train_targets']],float)
        mean=values.mean(0);scale=np.maximum(values.std(0),np.array([1e-5]*4+[1e-4]*4))
        delta_mean=float(np.max(np.abs(mean-np.asarray(statistics['mean'][str(h)]))))
        delta_scale=float(np.max(np.abs(scale-np.asarray(statistics['scale'][str(h)]))))
        statistics_errors[h]=dict(count=len(values),mean_max_abs_difference=delta_mean,scale_max_abs_difference=delta_scale)
        if len(values)!=statistics['count'][str(h)] or max(delta_mean,delta_scale)>1e-12:problems.append('independent_train_normalization_check')
    summaries={}
    for key,rs in sorted(groups.items()):
        history=key[2] not in ('cold','moving')
        eligible={str(n):sum(r['raw_frames']>=n for r in rs) for n in (24,48,96,193)} if history else {
            str(h):sum(r['raw_frames']>r['anchor']+h for r in rs) for h in HORIZONS}
        summaries['/'.join(key)]=dict(episodes=len(rs),full_valid=sum(r['full_valid'] for r in rs),support_eligible=eligible,
            max_tracking_rmse_m={res:max(r['tracking_rmse_m'][res] for r in rs) for res in ('64','128')})
        # All registered primary training/selection systems require their full
        # query target support. Late glide failure does not discard valid history.
        if history and any(r['raw_frames']<96 for r in rs):problems.append('missing_registered_history_support')
        if not history and any(r['raw_frames']<=r['anchor']+32 for r in rs):problems.append('missing_registered_query_support')
    return dict(schema='vec.data-admission.v1.1',status='PASS' if not problems else 'FAIL',problems=sorted(set(problems)),
        episode_count=len(rows),systems_by_split=dict(collections.Counter(s[0] for s in systems)),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        target_statistics_sha256=hashlib.sha256((root/'TRAIN_TARGET_STATISTICS.json').read_bytes()).hexdigest(),
        bank_config_sha256=hashlib.sha256((root/'BANK_CONFIG.json').read_bytes()).hexdigest(),
        support_groups=summaries,independent_train_statistics_check=statistics_errors,asset_failures=asset_failures,
        physical_failures_preserved=generation['physical_failures'],seconds=time.monotonic()-start,
        scope='data integrity, causal support and training normalization admission; not benchmark scientific completion',
        formal_training=False,test_read=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    if a.output.exists():raise ValueError('audit output already exists')
    result=audit(a.bank,a.workers);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','episode_count','systems_by_split','problems','seconds')}))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
