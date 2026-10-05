"""Versioned train/validation episodes for the calibrated visual platform.

Private manifests drive sampling; methods receive only white-listed arrays.
This generator deliberately has no test mode. A sealed evaluator is a separate
entry point, activated only after protocol and model selection are frozen.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .schema import Config,Parameters
from .data import action_library
from .populations import candidate_design
from .target_scaling import TargetStatistics,HORIZONS

BANK_VERSION='vec.research-bank.candidate1'
COUNTS={
    'train':dict(forced=24,mass=8,free=8,glide=4,static=4,cold=48,moving=24),
    'validation':dict(forced=4,mass=2,free=2,glide=1,static=1,cold=6,moving=3)}


def episode_seed(namespace,split,system_key,kind,replicate):
    text=json.dumps([namespace,split,system_key,kind,replicate],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')


def episode_design(seed,kind,replicate,config=Config()):
    streams=np.random.SeedSequence(seed).spawn(2);rng=np.random.default_rng(streams[0])
    angle=rng.uniform(0,2*np.pi);n=np.array([np.cos(angle),np.sin(angle)])
    center=rng.uniform(-.06 if kind=='cold' else -.08,.06 if kind=='cold' else .08,2)
    strain=0.;velocity=np.zeros(2);anchor=None;template=None
    if kind=='forced':strain=float(rng.choice([-.08,-.04,.04,.08]))
    if kind in ('free','moving'):strain=float(rng.choice([-.08,.08]))
    if kind in ('glide','moving'):
        speed=rng.uniform(.1,.17) if kind=='glide' else rng.uniform(.01,.04)
        velocity=speed*np.array([np.cos(angle+.5),np.sin(angle+.5)])
    state=np.r_[center-(config.ell0+strain)*n/2,center+(config.ell0+strain)*n/2,velocity,velocity]
    if kind=='forced':actions=action_library(np.random.default_rng(streams[1]),192,'forced')*(2/3)
    elif kind in ('glide','static'):actions=np.zeros((192,2),np.float32)
    elif kind in ('mass','free'):
        actions=np.zeros((96,2),np.float32)
        if kind=='mass':actions[:48]=.08*np.array([-n[1],n[0]]);actions[48:]=-actions[0]
    elif kind=='cold':
        anchor=1;template=('pulse_050','pulse_025','rest')[replicate%3]
        actions=np.zeros((33,2),np.float32)
        if template!='rest':actions[1:17,0]=.5 if template=='pulse_050' else .25
    elif kind=='moving':
        anchor=95;template='passive_prefix95';actions=np.zeros((127,2),np.float32)
        actions[95:111]=.5*np.array([np.cos(angle+.8),np.sin(angle+.8)])
    else:raise ValueError('unregistered episode kind')
    return state,actions,anchor,template


def jobs_for_design(design,namespace,smoke=False):
    rows=[r for r in design['systems'] if r['split'] in COUNTS]
    if smoke:
        rows=[next(r for r in rows if r['split']=='train'),next(r for r in rows if r['split']=='validation' and r['stratum']=='heldout_factorial_combinations')]
    jobs=[]
    for system in rows:
        for kind,count in COUNTS[system['split']].items():
            for replicate in range(min(count,3) if smoke else count):
                seed=episode_seed(namespace,system['split'],system['system_key'],kind,replicate)
                key=hashlib.sha256(f'{namespace}:{seed}:{kind}'.encode()).hexdigest()[:32]
                jobs.append(dict(episode_key=key,seed=seed,kind=kind,replicate=replicate,system=system))
    if len({r['episode_key'] for r in jobs})!=len(jobs):raise ValueError('episode identity collision')
    return jobs


def generate(job):
    from .physics import trajectory,InvalidTrajectory
    from .rendering import render_states
    from .calibration import track
    spec,out=job;system=spec['system'];cfg=Config();root=Path(out)/'episodes'/spec['episode_key'];root.mkdir(parents=True)
    initial,actions,anchor,template=episode_design(spec['seed'],spec['kind'],spec['replicate'],cfg)
    diagnostics={};failure=None;began=time.monotonic()
    try:states=trajectory(Parameters(*system['theta']),initial,actions,cfg,diagnostics=diagnostics)
    except InvalidTrajectory as exc:
        failure=dict(reason=exc.reason,time_s=exc.time);states=diagnostics.get('accepted_observed_states',initial[None])
    np.savez_compressed(root/'private.npz',state=states)
    assets={};tracking={}
    for resolution in (64,128):
        c=replace(cfg,resolution=resolution);images=render_states(states,c)
        measured=track(images,c)
        path=root/f'visible_r{resolution}.npz'
        np.savez_compressed(path,images=images,actions=actions[:len(states)-1],timestamps=np.arange(len(states))*.05)
        assets[str(resolution)]=dict(path=str(path.relative_to(out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        tracking[str(resolution)]=float(np.sqrt(np.mean((measured-states[:,:4])**2)))
    row=dict(**spec,split=system['split'],system_key=system['system_key'],theta=system['theta'],stratum=system['stratum'],
        anchor=anchor,template=template,raw_frames=len(states),requested_frames=len(actions)+1,
        failure=failure,full_valid=failure is None,assets=assets,tracking_rmse_m=tracking,
        observed_support_eligible={str(n):len(states)>=n for n in (24,48,96,193)},
        physics_config=asdict(cfg),private_state_path=str((root/'private.npz').relative_to(out)),seconds=time.monotonic()-began)
    (root/'private.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def finalize_statistics(rows,out):
    stats=TargetStatistics()
    for row in sorted(rows,key=lambda r:r['episode_key']):
        if row['split']!='train' or row['anchor'] is None:continue
        # Eligibility of a horizon uses its actual causal target support, not
        # success of a later trajectory segment or any model recovery score.
        state=np.load(Path(out)/row['private_state_path'],allow_pickle=False)['state'];anchor=row['anchor']
        for horizon in HORIZONS:
            if anchor+horizon<len(state):
                stats.add(split=row['split'],episode_key=row['episode_key'],horizon=horizon,targets=state[anchor+horizon]-state[anchor])
    return stats.export()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--namespace',required=True);parser.add_argument('--workers',type=int,default=48)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if a.output.exists():raise ValueError('bank attempt exists; use a new immutable attempt')
    a.output.mkdir(parents=True);design=candidate_design();jobs=jobs_for_design(design,a.namespace,a.smoke)
    config=dict(schema=BANK_VERSION,namespace=a.namespace,split_policy='train and validation only',test_read=False,test_generated=False,
        counts=COUNTS,physics=asdict(Config()),resolutions=[64,128],population=design,job_count=len(jobs),smoke=a.smoke,
        state='GENERATING_UNREVIEWED',sampler_permissions='IDs/theta/seeds/file paths stay evaluator-side; physical train labels enter losses only',
        episode_randomness='independent across system, split, episode kind and replicate; fixed public nuisance distributions',
        train_target_statistics='all train cold/moving labels once at each eligible horizon; validation is excluded before label access')
    (a.output/'BANK_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[];errors=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(generate,(job,str(a.output))):job['episode_key'] for job in jobs}
        with (a.output/'progress.jsonl').open('w') as log:
            for future in as_completed(futures):
                try:row=future.result();rows.append(row);event=dict(episode_key=row['episode_key'],failure=row['failure'])
                except Exception as exc:event=dict(episode_key=futures[future],error=repr(exc));errors.append(event)
                log.write(json.dumps(event)+'\n');log.flush()
                if (len(rows)+len(errors))%128==0 or a.smoke:print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),execution_errors=len(errors))),flush=True)
    rows.sort(key=lambda r:r['episode_key'])
    (a.output/'MANIFEST.private.json').write_text(json.dumps(dict(schema=BANK_VERSION,namespace=a.namespace,episodes=rows),indent=2)+'\n')
    if not errors:
        (a.output/'TRAIN_TARGET_STATISTICS.json').write_text(json.dumps(finalize_statistics(rows,a.output),indent=2)+'\n')
    report=dict(status='GENERATED_UNREVIEWED' if not errors else 'FAILURES_RETAINED',episodes=len(rows),execution_errors=errors,
        physical_failures=[dict(episode_key=r['episode_key'],split=r['split'],kind=r['kind'],theta=r['theta'],failure=r['failure'],raw_frames=r['raw_frames']) for r in rows if r['failure']],
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        seconds=time.monotonic()-start,smoke=a.smoke,formal_training=False,test_read=False,test_generated=False)
    (a.output/'BANK_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
