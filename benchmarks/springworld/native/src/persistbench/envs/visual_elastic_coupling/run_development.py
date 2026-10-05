import os
"""Run bounded CPU development calibration; never opens sealed data."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .schema import Config, Parameters, Episode, VERSION, history_payload, query_packet
from .data import generate_episode,save_episode,fixed_query_cases,action_library,relation_edges
from .calibration import track,identify,component_errors
from .physics import trajectory,InvalidTrajectory,run_checks
from .rendering import render_states,BACKEND

def run(output):
    start=time.time(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    gate=run_checks(output/"physics.json")
    if any(c["status"]!="PASS" for c in gate["checks"].values()):
        raise RuntimeError("physics gate failed; development generation halted; see physics.json")
    config=Config(); records=[]; bank={}; failures=[]
    # Explicit shared levels for relations and equal k/m scaling pair.
    parameters=[Parameters(.8,.5,8),Parameters(1.6,.5,16),Parameters(.8,.5,16),Parameters(.8,1.1,8)]
    for i,p in enumerate(parameters):
        for j,kind in enumerate(("static","glide","free","forced")):
            seed=910000+100*i+j
            try:
                episode=generate_episode(p,seed,kind,96,config)
                bank[i,kind]=episode; save_episode(episode,output/"episodes"/str(seed))
                for resolution in (64,128):
                    cfg=replace(config,resolution=resolution)
                    images=episode.images if resolution==64 else render_states(episode.private_state,cfg)
                    measured=track(images,cfg)
                    entry=dict(system=i,kind=kind,resolution=resolution,
                        position_rmse_m=float(np.sqrt(np.mean((measured-episode.private_state[:,:4])**2))),
                        id_by_frames={})
                    for n in (24,48,97):
                        entry["id_by_frames"][str(n)]=identify(images[:n],episode.actions[:n-1],cfg)
                    records.append(entry)
            except Exception as exc:
                failures.append(dict(system=i,kind=kind,seed=seed,error=repr(exc)))
    case_manifest=[]; prediction_records=[]
    free_initial=np.array([-.21,0,.21,0,.03,.01,-.03,-.01])
    free_actions=np.zeros((32,2))
    equivalent_a=trajectory(parameters[0],free_initial,free_actions,config)
    equivalent_b=trajectory(parameters[1],free_initial,free_actions,config)
    equivalence=dict(state_max_difference=float(np.max(np.abs(equivalent_a-equivalent_b))),
        images_exactly_equal=bool(np.array_equal(render_states(equivalent_a,config),render_states(equivalent_b,config))),
        theta_a=asdict(parameters[0]),theta_b=asdict(parameters[1]),
        interpretation="Development free-motion observation-equivalent pair; mass/stiffness cannot be separately determined under this shared initial state and zero force.")
    for i,p in enumerate(parameters):
        # Fresh common cold query, independent of donor RNG, same future force script.
        initial=np.array([-.175,0,.175,0,0,0,0,0],float)
        actions=action_library(np.random.default_rng(921000),32,"forced"); actions[0]=0
        states=trajectory(p,initial,actions,config)
        query=Episode(render_states(states,config),actions,np.arange(33)*config.control_dt,states,
                      dict(episode_key=f"cold_query_{i}",split="development",theta=asdict(p),seed=921000,
                           shared_query_action_template="paired_physical_development_intervention"))
        save_episode(query,output/"queries"/str(i))
        donor=bank.get((i,"forced")); wrong=bank.get(((i+1)%len(parameters),"forced"))
        if donor is None or wrong is None: continue
        cases=fixed_query_cases(query,{"query_only":None,"matched":donor,"wrong":wrong},anchor=1,q=1,horizon=16)
        case_manifest.extend(c["private"] for c in cases)
        packet=query_packet(query,1,1,16,False)
        positions=track(query.images[:2],config)
        visual_initial=np.r_[positions[-1],(positions[-1]-positions[-2])/config.control_dt]
        # Comparison prior is fixed development midpoint, explicitly not Bayes-optimal.
        candidates={"fixed_prior":Parameters(1.25,.875,14.5),"same_visual_true_theta":p}
        for label,ep in (("matched_id",donor),("wrong_id",wrong)):
            ident=identify(ep.images[:24],ep.actions[:23],config)
            if ident["m"] is not None and ident["k"] is not None and ident["k"]>0 and ident["gamma"]>=0:
                candidates[label]=Parameters(ident["m"],ident["gamma"],ident["k"])
            else:
                prediction_records.append(dict(system=i,condition=label,status="INVALID_ID",identification=ident))
        target=states[17]-states[1]
        for label,p_est in candidates.items():
            try:
                prediction=trajectory(p_est,visual_initial,packet["future_actions"],config)[-1]-visual_initial
                prediction_records.append(dict(system=i,condition=label,status="executed",metrics=component_errors(prediction,target)))
            except Exception as exc:
                prediction_records.append(dict(system=i,condition=label,status="INVALID_PREDICTION",error=repr(exc)))
    cases_ok=all(len({c[k] for c in case_manifest[n:n+3]})==1 for n in range(0,len(case_manifest),3)
                 for k in ("query_hash","target_hash")) and len(case_manifest)>0
    metadata=[e.private_metadata for e in bank.values()]
    relations={g:relation_edges(metadata,g) for g in ("G1","G2","G3")}
    report=dict(status="executed_development_diagnostics",research_supported=False,
        physical_gate_reference="physics.json",visual_and_id=records,generation_failures=failures,free_observation_equivalence=equivalence,
        predictions=prediction_records,fixed_query_hash_check=cases_ok,cases=case_manifest,
        relation_edge_counts={g:len(v) for g,v in relations.items()},
        scientific_gates=dict(history_sufficient_high_demand="NOT_VALIDATED",history_sufficient_low_demand="NOT_VALIDATED",
            history_insufficient_high_demand="physical_equivalence_check_only",fair_fresh_heads="NOT_RUN",
            composition="NOT_RUN",delayed_value="NOT_RUN",transfer="NOT_RUN",control="NOT_RUN"),
        limitations=["Four development systems; no inferential statistical or generalization claim.",
          "Pixel-integral ID is approximate and unconstrained; invalid parameter estimates are preserved, not clipped.",
          "Fixed-prior predictor is not a posterior/Bayes baseline; differences do not certify optimal donor headroom.",
          "No model training, assay freeze, sealed data access, normalized metrics or multi-seed research.",
          "1/2/4-worker throughput sweep not run; single worker only."])
    (output/"CALIBRATION_REPORT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (output/"relation_manifest.json").write_text(json.dumps(dict(split="development",edges=relations),indent=2)+"\n")
    source=Path(__file__).parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    manifest=dict(environment_version=VERSION,config=asdict(config),python=platform.python_version(),
        numpy=np.__version__,mujoco="3.3.5",render_backend=BACKEND,platform=platform.platform(),
        source_sha256=hashes,elapsed_seconds=time.time()-start,workers=1,split="development",test_read=False,
        generated_episodes=len(bank),generation_attempts=16,seed_policy="910000+100*system+kind; paired common query action template 921000",
        source_plan_sha256=hashlib.sha256(Path(os.environ['SPRII_SOURCE_PLAN']).read_bytes()).hexdigest(),
        budget=dict(strict_history_frames=24,strict_history_transitions=23,query_frames=2,query_past_actions=0),
        data_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*.npz")})
    (output/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(episodes=len(bank),failures=len(failures),cases=len(case_manifest),fixed_query_hash_check=cases_ok,seconds=manifest["elapsed_seconds"])))

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
