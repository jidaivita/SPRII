"""Parallel pixel-derived identification audit; labels are evaluator-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .schema import Config


def fit_visible_case(job):
    # This worker receives no parameter labels or simulator states.
    from .identification import fit_positions
    root, field, resolution, frames, starts = job
    root=Path(root)
    x=np.load(root/f"coverage_f{field:g}_r{resolution}.npz",allow_pickle=False)["positions"]
    a=np.load(root/f"visible_f{field:g}_r{resolution}.npz",allow_pickle=False)["actions"]
    if len(x)<frames:
        return dict(status="INSUFFICIENT_VISIBLE_PREFIX",available_frames=len(x))
    cfg=replace(Config(),field_width=field,resolution=resolution)
    began=time.monotonic()
    result=fit_positions(x[:frames],a[:frames-1],cfg,starts=starts)
    return dict(status="EXECUTED",fit=result,elapsed_seconds=time.monotonic()-began,
                observation_source="positions deterministically derived from uint8 images",uses_private_state=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--starts",type=int,default=3)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("use a new attempt directory")
    args.output.mkdir(parents=True)
    bank=json.loads((args.bank/"BANK_REPORT.json").read_text())
    if bank["config"]["split"]!="development": raise ValueError("development-only")
    cases=[]
    for ep in bank["episodes"]:
        if ep["kind"]=="forced":
            settings=[(res,n) for res in (64,128) for n in (24,48,96)]
        elif ep["kind"]=="free": settings=[(128,n) for n in (24,96)]
        elif ep["replicate"]==0: settings=[(128,n) for n in (24,96)]
        else: settings=[]
        for resolution,frames in settings:
            profile=next(p for p in ep["profiles"] if p["resolution"]==resolution and p["field_width"]==2.)
            envelope=dict(index=ep["index"],kind=ep["kind"],theta=ep["theta"],resolution=resolution,
                          frames=frames,duration_s=(frames-1)*.05,valid_prefix_frames=profile["valid_prefix_frames"],
                          whole_episode_valid=ep["full_physics_valid"],split="development")
            cases.append((ep,envelope))
    config=dict(schema="vec.pixel-id-audit.v1.1",bank=str(args.bank),cases=len(cases),starts=args.starts,
                bank_config_sha256=hashlib.sha256((args.bank/"BANK_CONFIG.json").read_bytes()).hexdigest(),
                claim="measured recovery on development grid, not general identifiability",test_read=False)
    (args.output/"AUDIT_CONFIG.json").write_text(json.dumps(config,indent=2)+"\n")
    rows=[];began=time.monotonic()
    with (args.output/"results.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={}
            for ep,envelope in cases:
                root=args.bank/"episodes"/f"{ep['index']:05d}"
                if envelope["valid_prefix_frames"]<envelope["frames"]:
                    row=dict(**envelope,status="INSUFFICIENT_VISIBLE_PREFIX")
                    rows.append(row);log.write(json.dumps(row)+"\n")
                else:
                    job=(str(root),2.,envelope["resolution"],envelope["frames"],args.starts)
                    futures[pool.submit(fit_visible_case,job)]=envelope
            for future in as_completed(futures):
                envelope=futures[future]
                try:
                    row=dict(**envelope,**future.result())
                    if row["status"]=="EXECUTED":
                        t=row["theta"]; f=row["fit"]
                        truth=dict(**t,k_over_m=t["k"]/t["m"])
                        row["relative_errors"]={k:None if f[k] is None else abs(f[k]-v)/v for k,v in truth.items()}
                except Exception as exc: row=dict(**envelope,status="EXECUTION_FAILED",error=repr(exc))
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
                if len(rows)%32==0 or len(rows)==len(cases):
                    print(json.dumps(dict(completed=len(rows),total=len(cases),seconds=time.monotonic()-began)),flush=True)
    (args.output/"AUDIT_REPORT.json").write_text(json.dumps(dict(config=config,results=rows,elapsed_seconds=time.monotonic()-began),indent=2,allow_nan=False)+"\n")
    if any(r["status"]=="EXECUTION_FAILED" for r in rows): raise SystemExit(1)


if __name__=="__main__": main()
