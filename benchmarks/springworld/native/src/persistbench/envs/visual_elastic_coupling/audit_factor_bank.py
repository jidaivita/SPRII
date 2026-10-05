"""Independent file and pairing admission for controlled donor substitutions."""
import argparse,collections,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .audit_bank import checked_path


def inspect(job):
    root,base,row,original=job;issues=[];key=row['episode_key']
    factors={'m_only':(0,),'gamma_only':(1,),'k_only':(2,),'surface_all':(0,1,2)}[row['variant']]
    truth=np.asarray(original['theta']);other=np.asarray(row['donor_theta']);expected=np.array([.75,.625,10.5])
    if row['query_theta']!=original['theta'] or row['query_system_key']!=original['system_key']:issues.append('source_system_pairing')
    for i in range(3):
        if i in factors and not np.isclose(abs(other[i]-truth[i]),expected[i],atol=1e-12,rtol=0):issues.append('factor_magnitude')
        if i not in factors and other[i]!=truth[i]:issues.append('unchanged_factor')
    if np.any(other<np.array([.5,.25,4.])) or np.any(other>np.array([2.,1.5,25.])):issues.append('factor_support')
    if row['failure'] or row['raw_frames']!=96:issues.append('actual_window_incomplete')
    private_path=checked_path(root,row['private_state_path']);blob=private_path.read_bytes()
    with np.load(private_path,allow_pickle=False) as data:states=data['state']
    with np.load(checked_path(base,original['private_state_path']),allow_pickle=False) as data:baseline=data['state'][:96]
    if states.shape!=(96,8) or not np.isfinite(states).all():issues.append('private_state_support')
    if not np.array_equal(states[0],baseline[0]):issues.append('changed_initial_state')
    center_max=float(np.max(np.abs((states[:,:2]+states[:,2:4]-baseline[:,:2]-baseline[:,2:4])/2))) if len(states)==96 else None
    if row['variant']=='k_only' and (center_max is None or center_max>1e-8):issues.append('center_k_invariant')
    for res,asset in row['assets'].items():
        path=checked_path(root,asset['path']);raw=path.read_bytes()
        if len(raw)!=asset['bytes'] or hashlib.sha256(raw).hexdigest()!=asset['sha256']:issues.append('asset_digest_'+res)
        with np.load(path,allow_pickle=False) as data:images=data['images'];actions=data['actions'];times=data['timestamps']
        with np.load(checked_path(base,original['assets'][res]['path']),allow_pickle=False) as data:
            initial_image=data['images'][0];original_actions=data['actions'][:95]
        if images.dtype!=np.uint8 or images.shape!=(96,int(res),int(res)):issues.append('image_shape_'+res)
        if not np.array_equal(images[0],initial_image):issues.append('changed_initial_pixels_'+res)
        if not np.array_equal(actions,original_actions):issues.append('changed_force_sequence_'+res)
        if times.shape!=(96,) or not np.allclose(times,np.arange(96)*.05,atol=1e-12,rtol=0):issues.append('time_support_'+res)
        if hashlib.sha256(actions.tobytes()).hexdigest()!=row['action_sha256']:issues.append('action_digest_'+res)
    return dict(episode_key=key,issues=issues,private_state_sha256=hashlib.sha256(blob).hexdigest(),center_difference_m=center_max,variant=row['variant'])


def audit(root,bank,workers=16):
    root=Path(root);manifest=root/'MANIFEST.private.json';data=json.loads(manifest.read_text());config=data['config'];rows=data['episodes']
    if config['split']!='validation' or config['test_read'] or any(r['split']!='validation' for r in rows):raise ValueError('test assets excluded from development admission')
    if config['bank_manifest_sha256']!=bank.manifest_sha256:raise ValueError('base bank binding differs')
    problems=[];begin=time.monotonic()
    if len(rows)!=config['rows'] or len({r['episode_key'] for r in rows})!=len(rows):problems.append('row_identity_or_count')
    pairs=collections.defaultdict(list)
    for row in rows:pairs[(row['query_system_key'],row['replicate'])].append(row)
    if len(pairs)!=config['systems']*config['replicates']:problems.append('pair_count')
    for group in pairs.values():
        if len(group)!=4 or {r['variant'] for r in group}!={'m_only','gamma_only','k_only','surface_all'}:problems.append('variant_family')
        if len({r['source_episode_key'] for r in group})!=1:problems.append('unpaired_sources')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases=list(pool.map(inspect,[(root,bank.root,r,bank.rows[r['source_episode_key']]) for r in rows]))
    failures=[r for r in cases if r['issues']]
    if failures:problems.append('asset_or_intervention_contract')
    digests={r['episode_key']:r['private_state_sha256'] for r in cases}
    return dict(schema='vec.factor-donor-admission.v1.1',status='PASS' if not problems else 'FAIL',problems=sorted(set(problems)),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),base_manifest_sha256=bank.manifest_sha256,
        episodes=len(rows),systems=config['systems'],replicates=config['replicates'],failures=failures,private_state_sha256=digests,
        irrelevant_k_center_max_m=max(r['center_difference_m'] for r in cases if r['variant']=='k_only'),
        scope='asset integrity, full input support and controlled factor substitution; does not establish any learned specificity result',
        seconds=time.monotonic()-begin,test_read=False,formal_results=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--factor-bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    from .pixel_training import TrainingBank
    if a.output.exists():raise ValueError('factor admission exists')
    result=audit(a.factor_bank,TrainingBank(a.bank),a.workers)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','episodes','problems','seconds')}))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
