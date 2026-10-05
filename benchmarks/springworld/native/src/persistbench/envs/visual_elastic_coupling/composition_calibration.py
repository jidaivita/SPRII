"""Test complementary episode evidence at a fixed processed-frame budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .schema import Config,Parameters
from .calibration import track,component_errors
from .identification import reference_rollout
from .joint_identification import fit_histories,information_factor
from .persistent_reference import posterior_samples
from .cold_opportunity import visual_rest_state


def generate_donor(theta,seed,kind,config):
    from .physics import trajectory
    from .rendering import render_states
    rng=np.random.default_rng(seed);center=rng.uniform(-.08,.08,2);angle=rng.uniform(0,2*np.pi)
    n=np.array([np.cos(angle),np.sin(angle)])
    length=config.ell0+(float(rng.choice([-.08,.08])) if kind=='free' else 0.)
    initial=np.r_[center-length*n/2,center+length*n/2,np.zeros(4)]
    actions=np.zeros((96,2),np.float32)
    if kind=='mass':
        force=.08*np.array([-n[1],n[0]])
        actions[:48]=force;actions[48:]=-force
    states=trajectory(Parameters(*theta),initial,actions,config)
    images=render_states(states,config)
    return images,actions,states


def run_system(job):
    index,theta,replicate,query_root,output,frames,samples=job
    cfg=replace(Config(),resolution=128);root=Path(output)/'systems'/f'{index:04d}_{replicate}'
    root.mkdir(parents=True);began=time.monotonic();histories={};budgets={}
    for di,(name,kind) in enumerate((('M0','mass'),('M1','mass'),('F0','free'),('F1','free'))):
        seed=982000+1000*index+10*replicate+di
        images,actions,states=generate_donor(theta,seed,kind,cfg)
        path=root/name;path.mkdir()
        np.savez_compressed(path/'visible.npz',images=images,actions=actions)
        np.savez_compressed(path/'private.npz',state=states)
        images=images[:frames];actions=actions[:frames-1]
        digest=hashlib.sha256(images.tobytes()+actions.tobytes()).hexdigest()
        histories[name]=dict(positions=track(images,cfg),actions=actions,fingerprint=digest)
        budgets[name]=dict(frames=frames,transitions=frames-1,effort_n2_s=float(np.sum(actions**2)*cfg.control_dt),fingerprint=digest)
    conditions=dict(M=['M0'],F=['F0'],repeated_M=['M0','M0'],repeated_F=['F0','F0'],
                    independent_MM=['M0','M1'],independent_FF=['F0','F1'],mixed_MF=['M0','F0'])
    query_index=index*9+replicate*3+2
    query_path=Path(query_root)/'queries'/f'{query_index:05d}'
    v=np.load(query_path/'visible_r128.npz',allow_pickle=False);truth=np.load(query_path/'private.npz',allow_pickle=False)['state']
    initial=visual_rest_state(track(v['images'][:2],cfg)[-1],cfg);future=v['actions'][1:33]
    rows=[];fits={}
    for name,members in conditions.items():
        if name.startswith('repeated_'):
            fit=dict(fits[name[-1]],processed_histories=2,deduplicated=1)
        else:fit=fit_histories([histories[m] for m in members],cfg)
        fits[name]=fit
        H,b=information_factor(fit,cfg)
        parameters=posterior_samples(H,b,samples=samples,seed=71973)
        trajectories=np.asarray([reference_rollout(p,initial,future,cfg)[[16,32]]-initial for p in parameters])
        unique=set(budgets[m]['fingerprint'] for m in members)
        row=dict(condition=name,fit=fit,processed_frames=len(members)*frames,
            unique_frames=len(unique)*frames,unique_transitions=len(unique)*(frames-1),episodes=len(members),
            effort_n2_s=sum(budgets[m]['effort_n2_s'] for m in members),
            posterior_mean=parameters.mean(0).tolist(),posterior_interval_80=np.quantile(parameters,[.1,.9],axis=0).tolist(),
            prediction_errors={},query_sha256=hashlib.sha256(v['images'][:2].tobytes()+future.tobytes()).hexdigest(),
            information_H=H.tolist(),information_b=b.tolist())
        for hi,h in enumerate((16,32)):
            row['prediction_errors'][str(h)]=component_errors(trajectories[:,hi].mean(0),truth[1+h]-truth[1])
        rows.append(row)
    for name,parameters in (('null',posterior_samples(np.zeros((3,3)),np.zeros(3),samples=samples,seed=71973)),
                            ('visual_theta',np.asarray(theta)[None])):
        predictions=np.mean([reference_rollout(p,initial,future,cfg)[[16,32]]-initial for p in parameters],axis=0)
        rows.append(dict(condition=name,prediction_errors={str(h):component_errors(predictions[hi],truth[1+h]-truth[1]) for hi,h in enumerate((16,32))}))
    result=dict(system_index=index,theta=theta,replicate=replicate,conditions=rows,
        seconds=time.monotonic()-began,config=asdict(cfg),split='development')
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--queries',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--frames',type=int,default=48)
    parser.add_argument('--posterior-samples',type=int,default=256);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('use a new attempt')
    args.output.mkdir(parents=True)
    bank=json.loads((args.bank/'BANK_CONFIG.json').read_text());queries=json.loads((args.queries/'QUERY_CONFIG.json').read_text())
    if bank['split']!='development' or bank['parameter_grid']!=queries['parameter_grid']:raise ValueError('development bank/query mismatch')
    jobs=[(i,theta,r,str(args.queries),str(args.output),args.frames,args.posterior_samples) for i,theta in enumerate(bank['parameter_grid']) for r in range(3)]
    if args.smoke:jobs=[j for j in jobs if j[0] in (0,15,32,63) and j[2]==0]
    config=dict(schema='vec.composition-calibration.v1.1',bank=str(args.bank),queries=str(args.queries),
        candidate='natural-rest perpendicular weak sustained force versus free strained radial motion',
        per_episode_frames=args.frames,paired_processed_frames=2*args.frames,posterior_samples=args.posterior_samples,
        budget_note='single-source diagnostics use half the paired budget; paired conditions have equal processed frames, and report unique transitions/effort separately',
        likelihood='full joint trajectory fit with separate initial states; local Gaussian posterior approximation with uniform physical prior',
        seed_policy='independent initial/action streams per theta and episode; no response-based selection',
        scientific_claim='complementarity is a hypothesis until comparisons against BOTH independent same-kind conditions pass',
        jobs=len(jobs),test_read=False,formal_training=False,smoke=args.smoke)
    (args.output/'COMPOSITION_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[];errors=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_system,j):(j[0],j[2]) for j in jobs}
        with (args.output/'progress.jsonl').open('w') as log:
            for future in as_completed(futures):
                try:
                    row=future.result();rows.append(row);event=dict(system=row['system_index'],replicate=row['replicate'],seconds=row['seconds'])
                except Exception as exc:event=dict(system=futures[future],error=repr(exc));errors.append(event)
                log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),last=event)),flush=True)
    (args.output/'COMPOSITION_REPORT.json').write_text(json.dumps(dict(config=config,systems=rows,errors=errors,
        elapsed_seconds=time.monotonic()-began,qualification='PENDING_ANALYSIS'),indent=2,allow_nan=False)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
