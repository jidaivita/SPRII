"""Paired visual donor interventions with fixed initial state and actions.

Only donor physics changes. Query bytes/targets are supplied separately by the
assay. All identities and factor labels remain in this private evaluator bank.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import numpy as np
from .schema import Config,Parameters

LOW=np.array([.5,.25,4.]);HIGH=np.array([2.,1.5,25.])
VARIANTS={'k_only':(2,),'m_only':(0,),'gamma_only':(1,),'surface_all':(0,1,2)}


class FactorDonorBank:
    """Read-only evaluator adapter, bound to the exact original development bank."""
    def __init__(self,root,bank):
        self.root=Path(root);manifest=self.root/'MANIFEST.private.json';report=json.loads((self.root/'SPECIFICITY_REPORT.json').read_text())
        self.manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if report['manifest_sha256']!=self.manifest_sha256 or report['errors'] or report['test_read']:raise ValueError('factor donor bank is not admitted')
        admission=json.loads((self.root/'FACTOR_ADMISSION.json').read_text())
        if admission['status']!='PASS' or admission['manifest_sha256']!=self.manifest_sha256 or admission['base_manifest_sha256']!=bank.manifest_sha256:
            raise ValueError('independent factor donor admission is absent or stale')
        data=json.loads(manifest.read_text());self.config=data['config']
        if self.config['bank_manifest_sha256']!=bank.manifest_sha256 or self.config['split']!='validation':raise ValueError('factor bank provenance differs')
        self.cases={}
        for row in data['episodes']:
            key=(row['query_system_key'],row['replicate']);self.cases.setdefault(key,{})[row['variant']]=row
        for variants in self.cases.values():
            if set(variants)!=set(VARIANTS) or len({r['source_episode_key'] for r in variants.values()})!=1:raise ValueError('incomplete paired factor family')

    def histories(self,key,replicate,resolution):
        from .schema import Episode,history_payload
        if (key[1],replicate) not in self.cases:return None,{},{}
        rows=self.cases[(key[1],replicate)];histories={};metadata={}
        for variant,row in rows.items():
            if row['failure'] or row['raw_frames']<96:raise ValueError('registered factor donor lacks complete support; retain case failure')
            asset=row['assets'][str(resolution)];path=(self.root/asset['path']).resolve()
            if not path.is_relative_to(self.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('factor donor asset provenance differs')
            with np.load(path,allow_pickle=False) as data:images=data['images'];actions=data['actions']
            ep=Episode(images,actions,np.arange(len(images))*.05,np.zeros((len(images),8)),{})
            name='factor_'+variant;histories[name]=[history_payload(ep,0,95)]
            metadata[name]=dict(donor_episode_keys=[row['episode_key']],processed_frames=96,unique_frames=96,unique_transitions=95,episodes=1,
                effort_n2_s=float(np.sum(actions[:95]**2)*.05))
        return next(iter(rows.values()))['source_episode_key'],histories,metadata


def changed_parameters(theta,variant):
    theta=np.asarray(theta,float)
    if theta.shape!=(3,) or not np.isfinite(theta).all() or np.any(theta<LOW-1e-12) or np.any(theta>HIGH+1e-12):
        raise ValueError('factor intervention is registered only within the main support')
    if variant not in VARIANTS:raise ValueError('unregistered factor intervention')
    result=theta.copy();mid=(LOW+HIGH)/2;half=(HIGH-LOW)/2
    for i in VARIANTS[variant]:result[i]+=half[i] if theta[i]<=mid[i] else -half[i]
    return result


def generate_case(job):
    from .physics import trajectory,InvalidTrajectory
    from .rendering import render_states
    from .calibration import track
    root,output,row,replicate,variant=job;root=Path(root);output=Path(output)
    if row['split'] not in ('validation','test'):raise ValueError('factor generation requires an evaluation source')
    original=np.load(root/row['private_state_path'],allow_pickle=False)['state'][:96]
    base=np.load(root/row['assets']['128']['path'],allow_pickle=False);actions=base['actions'][:95]
    theta=changed_parameters(row['theta'],variant)
    key=hashlib.sha256((row['episode_key']+':'+variant+':factor-intervention-v1').encode()).hexdigest()[:32]
    destination=output/'episodes'/key;destination.mkdir(parents=True,exist_ok=False)
    failure=None;diagnostics={};began=time.monotonic();cfg=Config()
    try:states=trajectory(Parameters(*theta),original[0],actions,cfg,diagnostics=diagnostics)
    except InvalidTrajectory as exc:
        failure=dict(reason=exc.reason,time_s=exc.time);states=diagnostics.get('accepted_observed_states',original[:1])
    private=destination/'private.npz';np.savez_compressed(private,state=states)
    assets={};tracking={};initial_equal={}
    for resolution in (64,128):
        config=replace(cfg,resolution=resolution);images=render_states(states,config)
        baseline=np.load(root/row['assets'][str(resolution)]['path'],allow_pickle=False)
        initial_equal[str(resolution)]=bool(np.array_equal(images[0],baseline['images'][0]))
        if not initial_equal[str(resolution)]:raise ValueError('same donor initial state changed rendering')
        path=destination/f'visible_r{resolution}.npz'
        np.savez_compressed(path,images=images,actions=actions[:len(states)-1],timestamps=np.arange(len(states))*.05)
        assets[str(resolution)]=dict(path=str(path.relative_to(output)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        tracking[str(resolution)]=float(np.sqrt(np.mean((track(images,config)-states[:,:4])**2)))
    original_center=(original[:len(states),:2]+original[:len(states),2:4])/2
    changed_center=(states[:,:2]+states[:,2:4])/2
    center_difference=float(np.max(np.abs(original_center-changed_center)))
    if variant=='k_only' and center_difference>1e-8:raise ValueError('fixed-action irrelevant-k center invariant failed')
    result=dict(episode_key=key,source_episode_key=row['episode_key'],query_system_key=row['system_key'],split=row['split'],stratum=row['stratum'],
        replicate=replicate,variant=variant,query_theta=row['theta'],donor_theta=theta.tolist(),changed_factors=list(VARIANTS[variant]),
        raw_frames=len(states),requested_frames=96,failure=failure,assets=assets,private_state_path=str(private.relative_to(output)),
        tracking_rmse_m=tracking,identical_initial_pixels=initial_equal,identical_initial_state=bool(np.array_equal(original[0],states[0])),
        action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),max_donor_center_change_m=center_difference,seconds=time.monotonic()-began)
    (destination/'private.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--replicates',type=int,default=2);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    from .pixel_training import TrainingBank
    from .training_protocol import source_fingerprint
    if a.replicates not in (1,2,3,4) or a.workers<1:raise ValueError('invalid registered generation budget')
    if a.output.exists():raise ValueError('specificity bank attempt exists')
    bank=TrainingBank(a.bank);keys=bank.selection_keys
    if a.smoke:keys=[next(k for k in keys if bank.systems[k]['stratum']==stratum) for stratum in ('continuous_new_systems','heldout_factorial_combinations')]
    jobs=[]
    for key in keys:
        rows=bank.eligible(key,'forced',96)
        if len(rows)<a.replicates:raise ValueError('independent matched source support unavailable')
        for rep,row in enumerate(rows[:a.replicates]):
            for variant in VARIANTS:jobs.append((str(a.bank),str(a.output),row,rep,variant))
    a.output.mkdir(parents=True);config=dict(schema='vec.factor-donor-bank.v1.1',bank_manifest_sha256=bank.manifest_sha256,
        source_fingerprint=source_fingerprint(),split='validation',systems=len(keys),replicates=a.replicates,variants=VARIANTS,
        frames=96,rows=len(jobs),test_read=False,formal_results=False,smoke=a.smoke,
        intervention='Each changed coordinate moves by exactly half its original physical range, toward the opposite half. Unchanged coordinates, source donor initial state, applied force sequence, renderer and query remain fixed.',
        center_semantics='k_only is relevant-matched for Center under identical open-loop donor actions; m_only/gamma_only are relevant-wrong; relative/joint do not inherit the irrelevant-k claim.',
        surface_semantics='surface_all shares exact donor initial pixels and action sequence with matched but changes all physical factors; independent wrong donors are separately assigned by the evaluator.',
        selection='all main validation continuous/heldout-combination systems, no model score or recovery filter; all rejected physics retained')
    (a.output/'SPECIFICITY_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(generate_case,j):(j[2]['episode_key'],j[3],j[4]) for j in jobs}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:errors.append(dict(case=futures[future],error=repr(exc)))
            if (len(rows)+len(errors))%32==0 or a.smoke:print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors))),flush=True)
    rows.sort(key=lambda r:r['episode_key']);manifest=a.output/'MANIFEST.private.json'
    manifest.write_text(json.dumps(dict(config=config,episodes=rows),indent=2)+'\n')
    report=dict(status='GENERATED_UNREVIEWED' if not errors else 'FAILURES_RETAINED',episodes=len(rows),errors=errors,
        physical_failures=[dict(episode_key=r['episode_key'],failure=r['failure'],frames=r['raw_frames']) for r in rows if r['failure']],
        all_initial_pixels_equal=all(all(r['identical_initial_pixels'].values()) for r in rows),
        irrelevant_k_center_max_m=max((r['max_donor_center_change_m'] for r in rows if r['variant']=='k_only'),default=None),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),test_read=False,formal_results=False)
    (a.output/'SPECIFICITY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
