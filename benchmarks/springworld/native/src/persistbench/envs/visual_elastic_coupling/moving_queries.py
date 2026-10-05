"""Fixed-anchor moving-query budget scan with a declared passive prefix.

The prefix contains no applied force, which is a public episode-design prior;
past action arrays are nevertheless absent from the default query interface.
Different query budgets end at the same physical anchor and share targets/U.
Long passive queries can identify gamma and k/m, not the absolute mass scale.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .schema import Config,Parameters,Episode,query_packet

ANCHOR=95
BUDGETS=(0,1,3,7,15,31,63,95)
HORIZONS=(1,4,16,32)


def prototype(replicate,config=Config()):
    rng=np.random.default_rng(983600+replicate);angle=rng.uniform(0,2*np.pi)
    direction=np.array([np.cos(angle),np.sin(angle)]);center=rng.uniform(-.08,.08,2)
    length=config.ell0+float(rng.choice([-.08,.08]));speed=rng.uniform(.01,.04)
    velocity=speed*np.array([np.cos(angle+.5),np.sin(angle+.5)])
    state=np.r_[center-length*direction/2,center+length*direction/2,velocity,velocity]
    actions=np.zeros((ANCHOR+32,2),np.float32)
    actions[ANCHOR:ANCHOR+16]=.5*np.array([np.cos(angle+.8),np.sin(angle+.8)])
    return state,actions


def generate(job):
    from .physics import trajectory
    from .rendering import render_states
    from .calibration import track
    index,theta,replicate,output=job;cfg=Config();initial,actions=prototype(replicate,cfg)
    root=Path(output)/'queries'/f'{3*index+replicate:05d}';root.mkdir(parents=True)
    states=trajectory(Parameters(*theta),initial,actions,cfg);rows=[]
    np.savez_compressed(root/'private.npz',state=states)
    for resolution in (64,128):
        c=replace(cfg,resolution=resolution);images=render_states(states,c)
        ep=Episode(images,actions,np.arange(len(images))*cfg.control_dt,states,{})
        np.savez_compressed(root/f'visible_r{resolution}.npz',images=images,actions=actions)
        tracked=track(images,c);np.savez_compressed(root/f'tracking_r{resolution}.npz',positions=tracked)
        fingerprints={}
        for horizon in HORIZONS:
            fingerprints[horizon]=[]
            for q in BUDGETS:
                packet=query_packet(ep,ANCHOR,q,horizon)
                target=states[ANCHOR+horizon]-states[ANCHOR]
                key=hashlib.sha256(packet['observations'][-1,0].tobytes()+packet['future_actions'].tobytes()+target.tobytes()).hexdigest()
                fingerprints[horizon].append(key)
                assert 'past_actions' not in packet and not packet['observations'][0,1].any()
            assert len(set(fingerprints[horizon]))==1
        rows.append(dict(resolution=resolution,position_tracking_rmse_m=float(np.sqrt(np.mean((tracked-states[:,:4])**2))),
            fixed_anchor_future_target_sha256={h:fingerprints[h][0] for h in HORIZONS}))
    row=dict(system_index=index,theta=theta,replicate=replicate,split='development',anchor=ANCHOR,
        config=asdict(cfg),physics_valid=True,profiles=rows)
    (root/'private.json').write_text(json.dumps(row,indent=2)+'\n');return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if a.output.exists():raise ValueError('use a new query attempt')
    bank=json.loads((a.bank/'BANK_CONFIG.json').read_text())
    if bank['split']!='development':raise ValueError('development-only generator')
    a.output.mkdir(parents=True)
    config=dict(schema='vec.moving-query-calibration.v1.1',split='development',parameter_grid=bank['parameter_grid'],
        anchor=ANCHOR,budgets=BUDGETS,horizons=HORIZONS,past_actions_exposed=False,
        public_prefix_prior='zero applied force; moving freely from an independently drawn strained initial state',
        query_initial_prior='common orientation/center distributions; common translational speed U[.01,.04] m/s; no relative initial velocity',
        force_profile='parameter-independent .5 N pulse for16 steps then zero16 steps; fixed-anchor family',
        claim='query budget is not assumed to remove donor value; free mass/stiffness scale ambiguity remains',
        test_read=False,formal_training=False,smoke=a.smoke)
    (a.output/'QUERY_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    jobs=[(i,t,r,str(a.output)) for i,t in enumerate(bank['parameter_grid']) for r in range(3)]
    if a.smoke:jobs=[j for j in jobs if j[0] in (0,15,32,63) and j[2]==0]
    rows=[];errors=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(generate,j):(j[0],j[2]) for j in jobs}
        with (a.output/'progress.jsonl').open('w') as log:
            for future in as_completed(futures):
                try:row=future.result();rows.append(row);event=dict(system=row['system_index'],replicate=row['replicate'])
                except Exception as exc:event=dict(system=futures[future],error=repr(exc));errors.append(event)
                log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),last=event)),flush=True)
    (a.output/'QUERY_REPORT.json').write_text(json.dumps(dict(config=config,queries=rows,errors=errors,elapsed_seconds=time.monotonic()-began),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
