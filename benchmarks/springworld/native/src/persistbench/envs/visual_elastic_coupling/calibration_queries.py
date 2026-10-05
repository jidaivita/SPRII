"""Theta-independent cold-query interventions for development calibration.

The public profile starts at natural length and rest, with one observed zero
action interval. Future forces are fixed before seeing theta. No query or
trajectory is selected by the performance of a learned method.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .schema import Config, Parameters


def query_initial(replicate, config=Config()):
    rng=np.random.default_rng(950120+replicate)
    center=rng.uniform(-.06,.06,2)
    angle=rng.uniform(0,2*np.pi)
    direction=np.array([np.cos(angle),np.sin(angle)])
    return np.r_[center-config.ell0*direction/2, center+config.ell0*direction/2,np.zeros(4)]


def query_actions(template):
    actions=np.zeros((33,2),np.float32)
    if template=="rest": return actions
    if template not in ("pulse_025","pulse_050"): raise ValueError(template)
    # A fixed global direction, unrelated to hidden parameters or initial angle.
    actions[1:17,0]=.25 if template=="pulse_025" else .5
    return actions


def generate_query(job):
    from .physics import trajectory, InvalidTrajectory
    from .rendering import render_states
    from .calibration import track
    index,theta,replicate,template,out=job
    root=Path(out)/"queries"/f"{index:05d}"
    root.mkdir(parents=True)
    cfg=Config(); initial=query_initial(replicate,cfg); actions=query_actions(template)
    row=dict(index=index,theta=dict(zip(("m","gamma","k"),theta)),replicate=replicate,
             template=template,episode_key=f"vec_dev_query_{index:05d}",split="development",
             config=asdict(cfg),anchor=1,query_frames=2,query_past_actions=0,
             requested_horizons=[1,4,16,32],profiles=[],status="EXECUTED")
    diag={}
    try: states=trajectory(Parameters(*theta),initial,actions,cfg,diagnostics=diag)
    except InvalidTrajectory as exc:
        row.update(status="PHYSICS_REJECTED",failure=dict(reason=exc.reason,time_s=exc.time))
        states=diag.get("accepted_observed_states",initial[None])
    row["completed_transitions"]=len(states)-1
    np.savez_compressed(root/"private.npz",state=states,initial_state=initial)
    np.savez_compressed(root/"actions.npz",actions=actions)
    for res in (64,128):
        c=replace(cfg,resolution=res)
        images=render_states(states,c)
        measured=track(images,c)
        np.savez_compressed(root/f"visible_r{res}.npz",images=images,actions=actions[:len(states)-1],
                            timestamps=np.arange(len(states))*c.control_dt)
        np.savez_compressed(root/f"coverage_r{res}.npz",positions=measured)
        row["profiles"].append(dict(resolution=res,query_sha256=hashlib.sha256(images[:2].tobytes()).hexdigest(),
            actions_sha256=hashlib.sha256(actions[1:].tobytes()).hexdigest(),
            position_rmse_m=float(np.sqrt(np.mean((measured-states[:,:4])**2)))))
    (root/"private.json").write_text(json.dumps(row,indent=2)+"\n")
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("use a new attempt directory")
    args.output.mkdir(parents=True)
    bank=json.loads((args.bank/"BANK_CONFIG.json").read_text())
    if bank["split"]!="development": raise ValueError("development only")
    jobs=[]
    for theta in bank["parameter_grid"]:
        for replicate in range(3):
            for template in ("rest","pulse_025","pulse_050"):
                jobs.append((len(jobs),theta,replicate,template,str(args.output)))
    config=dict(schema="vec.cold-query-calibration.v1.1",source_bank=str(args.bank),split="development",
                parameter_grid=bank["parameter_grid"],jobs=len(jobs),test_read=False,
                prior="natural length, zero velocity; orientation and center drawn independently of theta",
                future_policy="zero control at transition 0; fixed global x force .25 or .5 for 16 transitions then 16 zero; rest negative control",
                q_past_actions=0,query_frames=2,horizons=[1,4,16,32],replicates=3)
    (args.output/"QUERY_CONFIG.json").write_text(json.dumps(config,indent=2)+"\n")
    rows=[]; errors=[]; began=time.monotonic()
    with (args.output/"progress.jsonl").open("w") as f:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(generate_query,j):j[0] for j in jobs}
            for future in as_completed(futures):
                try: rows.append(future.result())
                except Exception as exc: errors.append(dict(index=futures[future],error=repr(exc)))
                if (len(rows)+len(errors))%32==0:
                    event=dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors),seconds=time.monotonic()-began)
                    f.write(json.dumps(event)+"\n");f.flush();print(json.dumps(event),flush=True)
    equivalent=True
    for replicate in range(3):
        for template in ("rest","pulse_025","pulse_050"):
            group=[r for r in rows if r["replicate"]==replicate and r["template"]==template]
            for res in (64,128):
                hashes={(p["query_sha256"],p["actions_sha256"]) for r in group for p in r["profiles"] if p["resolution"]==res}
                equivalent &= len(hashes)==1 and len(group)==len(bank["parameter_grid"])
    result=dict(config=config,queries=sorted(rows,key=lambda r:r["index"]),errors=errors,
                query_and_future_actions_theta_independent=bool(equivalent),
                elapsed_seconds=time.monotonic()-began,scientific_qualification="PENDING_ANALYSIS")
    (args.output/"QUERY_REPORT.json").write_text(json.dumps(result,indent=2)+"\n")
    if errors or not equivalent: raise SystemExit(1)


if __name__=="__main__": main()
