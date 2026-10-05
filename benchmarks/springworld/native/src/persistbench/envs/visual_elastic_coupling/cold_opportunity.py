"""Four-reference cold-query audit after independent visible donor fitting.

The no-donor reference integrates a declared continuous prior at a constrained
visual estimate of the known-rest initial state. It is a numerical reference,
not a claim of an exact posterior over all possible visual states.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from scipy.stats import qmc
from .schema import Config
from .identification import reference_rollout
from .calibration import component_errors


def visual_rest_state(positions, config):
    """Use only measured centers and the public natural-length/rest profile."""
    p=np.asarray(positions,float).reshape(2,2)
    center=p.mean(0); relative=p[1]-p[0]
    norm=np.linalg.norm(relative)
    if norm<1e-6: raise ValueError("degenerate visual object separation")
    relative*=config.ell0/norm
    return np.r_[center-relative/2,center+relative/2,np.zeros(4)]


def prior_predictions(job):
    key,initial,actions,resolution,points=job
    cfg=replace(Config(),resolution=resolution)
    parameters=qmc.scale(qmc.Sobol(3,scramble=True,seed=961709).random_base2(points.bit_length()-1),
                         [.5,.25,4.],[2.,1.5,25.])
    horizons=[1,4,16,32]
    predictions=np.asarray([reference_rollout(p,initial,actions,cfg)[horizons]-initial for p in parameters])
    average=predictions.mean(0)
    convergence=[]
    for n in (64,128,256,512):
        if n>=points: continue
        for hi,h in enumerate(horizons):
            convergence.append(dict(points=n,horizon=h,error_to_full=component_errors(predictions[:n,hi].mean(0),average[hi])))
    variances=[]
    for hi,h in enumerate(horizons):
        # Component MSE accepts rows and averages over parameter draws.
        variances.append(dict(horizon=h,conditional_parameter_variance=component_errors(predictions[:,hi],average[hi])))
    return key,dict(means=average.tolist(),convergence=convergence,variances=variances)


def evaluate_query(job):
    query,root,fit_lookup,nulls,theta_index,thetas=job
    root=Path(root)/"queries"/f"{query['index']:05d}"
    states=np.load(root/"private.npz",allow_pickle=False)["state"]
    if query["status"]!="EXECUTED" or len(states)<34:
        return [dict(query_index=query["index"],status="QUERY_REJECTED",failure=query.get("failure"))]
    if np.max(np.abs(states[:2,4:]))>1e-11:
        raise ValueError("declared cold rest prior not satisfied")
    theta=[query["theta"][k] for k in ("m","gamma","k")]
    si=theta_index[tuple(theta)]; wrong=si^21 if len(thetas)==64 else (si+1)%len(thetas)
    rows=[];horizons=[1,4,16,32]
    for resolution in (64,128):
        cfg=replace(Config(),resolution=resolution)
        v=np.load(root/f"visible_r{resolution}.npz",allow_pickle=False)
        x=np.load(root/f"coverage_r{resolution}.npz",allow_pickle=False)["positions"]
        visual=visual_rest_state(x[1],cfg); actions=v["actions"][1:33]
        key=f"{resolution}:{query['replicate']}:{query['template']}"
        null=np.asarray(nulls[key]["means"])
        predictions={"query_only_continuous_prior":null,
            "same_visual_true_theta":reference_rollout(theta,visual,actions,cfg)[horizons]-visual,
            "privileged_state_theta":reference_rollout(theta,states[1],actions,cfg)[horizons]-states[1]}
        provenance={label:dict(status="EXECUTED") for label in predictions}
        for frames in (24,48,96):
            for condition,donor_system in (("matched",si),("wrong",wrong)):
                label=f"{condition}_{frames}"
                donor_index=donor_system*12+9+query["replicate"]
                fit=fit_lookup.get((donor_index,resolution,frames))
                usable=fit is not None and fit["status"]=="EXECUTED" and all(fit["fit"][k] is not None for k in ("m","gamma","k"))
                if usable:
                    p=[fit["fit"][k] for k in ("m","gamma","k")]
                    predictions[label]=reference_rollout(p,visual,actions,cfg)[horizons]-visual
                    provenance[label]=dict(status="EXECUTED",donor_index=donor_index,
                        donor_fit_status=fit["fit"]["status"],whole_episode_valid=fit["whole_episode_valid"])
                else:
                    predictions[label]=null.copy()
                    provenance[label]=dict(status="DONOR_UNUSABLE_NULL_FALLBACK",donor_index=donor_index,
                                           reason=None if fit is None else fit["status"])
        for hi,h in enumerate(horizons):
            target=states[1+h]-states[1]
            common=dict(query_index=query["index"],system_index=si,replicate=query["replicate"],
                template=query["template"],resolution=resolution,horizon=h,
                query_sha256=hashlib.sha256(v["images"][:2].tobytes()+actions[:h].tobytes()).hexdigest(),
                target_sha256=hashlib.sha256(target.tobytes()).hexdigest())
            for label,p in predictions.items():
                rows.append(dict(**common,condition=label,**provenance[label],
                    error=component_errors(p[hi],target)))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--queries",type=Path,required=True)
    parser.add_argument("--fits",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=32)
    parser.add_argument("--prior-points",type=int,default=1024)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("use a new attempt directory")
    if args.prior_points<128 or args.prior_points&(args.prior_points-1): raise ValueError("prior sample count must be power of two >=128")
    args.output.mkdir(parents=True)
    qr=json.loads((args.queries/"QUERY_REPORT.json").read_text())
    fits=json.loads((args.fits/"AUDIT_REPORT.json").read_text())
    if qr["config"]["split"]!="development" or not qr["query_and_future_actions_theta_independent"]:
        raise ValueError("cold query equivalence required")
    thetas=qr["config"]["parameter_grid"]; theta_index={tuple(t):i for i,t in enumerate(thetas)}
    lookup={(r["index"],r["resolution"],r["frames"]):r for r in fits["results"]}
    representatives={}
    for query in qr["queries"]:
        if query["status"]!="EXECUTED": continue
        for res in (64,128):
            key=f"{res}:{query['replicate']}:{query['template']}"
            if key in representatives: continue
            path=args.queries/"queries"/f"{query['index']:05d}"
            x=np.load(path/f"coverage_r{res}.npz",allow_pickle=False)["positions"]
            v=np.load(path/f"visible_r{res}.npz",allow_pickle=False)
            representatives[key]=(key,visual_rest_state(x[1],replace(Config(),resolution=res)),v["actions"][1:33],res,args.prior_points)
    nulls={};rows=[];errors=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(prior_predictions,j) for j in representatives.values()]
        for future in as_completed(futures):
            key,value=future.result();nulls[key]=value
            print(json.dumps(dict(prior_complete=key,completed=len(nulls),total=len(futures))),flush=True)
        (args.output/"PRIOR_REPORT.json").write_text(json.dumps(nulls,indent=2)+"\n")
        # Send only one system's donor fits per task, avoiding repeated huge
        # dictionary serialization in every query worker.
        futures={}
        for query in qr["queries"]:
            si=theta_index[tuple(query["theta"][k] for k in ("m","gamma","k"))]
            wi=si^21 if len(thetas)==64 else (si+1)%len(thetas)
            indices={s*12+9+query["replicate"] for s in (si,wi)}
            needed={k:v for k,v in lookup.items() if k[0] in indices}
            futures[pool.submit(evaluate_query,(query,str(args.queries),needed,nulls,theta_index,thetas))]=query["index"]
        with (args.output/"results.jsonl").open("w") as f:
            completed=0
            for future in as_completed(futures):
                try:
                    batch=future.result();rows.extend(batch)
                    for row in batch:f.write(json.dumps(row,allow_nan=False)+"\n")
                except Exception as exc: errors.append(dict(query=futures[future],error=repr(exc)))
                completed+=1;f.flush()
                if completed%64==0:print(json.dumps(dict(queries_complete=completed,total=len(futures))),flush=True)
    summary=[]
    for template in ("rest","pulse_025","pulse_050"):
        for res in (64,128):
            for h in (1,4,16,32):
                for condition in sorted({r.get("condition") for r in rows if "condition" in r}):
                    selected=[r for r in rows if r.get("condition")==condition and r["template"]==template and r["resolution"]==res and r["horizon"]==h]
                    if not selected:continue
                    system_errors=[]
                    for si in sorted({r["system_index"] for r in selected}):
                        group=[r for r in selected if r["system_index"]==si]
                        system_errors.append({k:float(np.mean([r["error"][k] for r in group])) for k in group[0]["error"]})
                    summary.append(dict(template=template,resolution=res,horizon=h,condition=condition,systems=len(system_errors),
                        fallback_cases=sum(r["status"]!="EXECUTED" for r in selected),
                        mean_error={k:float(np.mean([r[k] for r in system_errors])) for k in system_errors[0]}))
    report=dict(schema="vec.cold-opportunity.v1.1",query_source=str(args.queries),fit_source=str(args.fits),
        sources_sha256={"queries":hashlib.sha256((args.queries/"QUERY_REPORT.json").read_bytes()).hexdigest(),
                        "fits":hashlib.sha256((args.fits/"AUDIT_REPORT.json").read_bytes()).hexdigest()},
        prior_points=args.prior_points,prior="independent continuous uniform m [.5,2], gamma [.25,1.5], k [4,25]",
        visual_state_prior="known rest and natural length projected using measured image centers; plug-in angle and center",
        null_qualification="numerical parameter integration at visual state point estimate; not exact general visual Bayes optimal",
        donor_failure_policy="keep cases and fall back to the same query-only reference; also report failures separately",
        aggregation="average query replicates within system, then average systems; development only",
        test_read=False,formal_training=False,results=rows,summary=summary,errors=errors,elapsed_seconds=time.monotonic()-began)
    (args.output/"OPPORTUNITY_REPORT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if errors:raise SystemExit(1)


if __name__=="__main__":main()
