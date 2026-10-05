import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc, rankdata

from paper_c.coupled_sled.learner import PersistentJEPA, RawHistoryPredictor, _predictions_and_z
from paper_c.coupled_sled.learner_data import LearnerArrays

from .model import SwimmerModel, prior_log_scale
from .s0_screen import _landmarks, _response_and_jacobian
from .waveforms import banks


_BASE = _S2 = _MODEL = _HISTORY = _QUERY = _LANDMARKS = _SCALE = None

GRID_AXES = {
    "baseline": ("system", "realization", "history", "query"),
    "candidate": ("system", "realization", "history", "candidate", "query"),
}


def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _systems(count, seed, prior):
    points = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    lo = np.asarray([prior["mass_scale"][0]]*3 + [prior["damping_scale"][0]]*2)
    hi = np.asarray([prior["mass_scale"][1]]*3 + [prior["damping_scale"][1]]*2)
    return np.log(lo + points*(hi-lo))


def _init_worker(base, s2, horizon):
    global _BASE, _S2, _MODEL, _HISTORY, _QUERY, _LANDMARKS, _SCALE
    _BASE, _S2 = base, s2
    _MODEL = SwimmerModel(base["model"]); _SCALE = prior_log_scale(base["persistent_prior"])
    _HISTORY, _QUERY = banks(horizon, base["model"]["timestep_s"])
    _LANDMARKS = _landmarks(len(next(iter(_HISTORY.values()))), base["observation"]["landmark_count"])


def _action_landmarks(actions):
    return actions[np.clip(_LANDMARKS-1, 0, len(actions)-1)].reshape(-1)


def _bank(theta, initial, bank):
    means, fishers = [], []
    for action in bank.values():
        mean, jac = _response_and_jacobian(_MODEL, theta, initial, action, _LANDMARKS, _SCALE, _BASE["s0"]["finite_difference_log_step"])
        means.append(mean); fishers.append(jac.T @ jac / _BASE["observation"]["sensor_std"]**2)
    return np.asarray(means), np.asarray(fishers)


def _worker(task):
    split_seed, system_index, theta, realizations = task
    baseline_history=[]; baseline_mask=[]; baseline_query=[]; baseline_target=[]
    candidate_history=[]; candidate_mask=[]; candidate_query=[]; candidate_target=[]
    baseline_meta=[]; candidate_meta=[]; physical=[]
    history_names, query_names = list(_HISTORY), list(_QUERY)
    actions = np.asarray([value.ravel() for value in _HISTORY.values()])
    for realization in range(realizations):
        rng=np.random.default_rng(np.random.SeedSequence([split_seed, system_index, realization]))
        ih=_MODEL.sample_initial_state(rng,_BASE["transient_initial_state"])
        ie=_MODEL.sample_initial_state(rng,_BASE["transient_initial_state"])
        iq=_MODEL.sample_initial_state(rng,_BASE["transient_initial_state"])
        mh,fh=_bank(theta,ih,_HISTORY); me,fe=_bank(theta,ie,_HISTORY); mq,fq=_bank(theta,iq,_QUERY)
        oh=np.concatenate([ih.qpos[2:],ih.qvel]); oe=np.concatenate([ie.qpos[2:],ie.qvel]); oq=np.concatenate([iq.qpos[2:],iq.qvel])
        seg_h=np.asarray([np.concatenate([oh,mh[i]+rng.normal(0,_BASE["observation"]["sensor_std"],mh[i].shape),_action_landmarks(_HISTORY[name])]) for i,name in enumerate(history_names)])
        seg_e=np.asarray([np.concatenate([oe,me[i]+rng.normal(0,_BASE["observation"]["sensor_std"],me[i].shape),_action_landmarks(_HISTORY[name])]) for i,name in enumerate(history_names)])
        targets=np.asarray([mq[i]+rng.normal(0,_BASE["observation"]["sensor_std"],mq[i].shape) for i in range(6)])
        qinputs=np.asarray([np.concatenate([oq,_action_landmarks(_QUERY[name])]) for name in query_names])
        response_e=me.reshape(6,-1,8)-oe[None,None,:]
        eye=np.eye(5)
        for h in range(6):
            cov_h=np.linalg.inv(eye+fh[h])
            for q in range(6):
                baseline_history.append(np.stack([seg_h[h],np.zeros(48)])); baseline_mask.append([1,0]); baseline_query.append(qinputs[q]); baseline_target.append(targets[q]); baseline_meta.append((system_index,realization,h,q))
                remaining=float(np.trace(fq[q]@cov_h))
                for e in range(6):
                    cov_e=np.linalg.inv(eye+fe[e]); cov_he=np.linalg.inv(eye+fh[h]+fe[e])
                    candidate_history.append(np.stack([seg_h[h],seg_e[e]])); candidate_mask.append([1,1]); candidate_query.append(qinputs[q]); candidate_target.append(targets[q]); candidate_meta.append((system_index,realization,h,e,q))
                    physical.append((float(np.trace(fq[q]@(cov_h-cov_he))),float(np.trace(fq[q]@(eye-cov_e))),float(np.mean((response_e[h]-response_e[e])**2)),float(np.mean((actions[h]-actions[e])**2)),remaining))
    def arrays(history,mask,query,target,meta,candidate):
        meta=np.asarray(meta,dtype=np.int64)
        return LearnerArrays(np.asarray(history,np.float32),np.asarray(mask,np.float32),np.asarray(query,np.float32),np.asarray(target,np.float32),
            np.full(len(meta),2 if candidate else 1,np.int64),meta[:,0],meta[:,2],meta[:,4] if candidate else meta[:,3],np.repeat(theta[None,:],len(meta),axis=0).astype(np.float32))
    return arrays(baseline_history,baseline_mask,baseline_query,baseline_target,baseline_meta,False), arrays(candidate_history,candidate_mask,candidate_query,candidate_target,candidate_meta,True), np.asarray(baseline_meta,np.int64), np.asarray(candidate_meta,np.int64), np.asarray(physical,np.float64)


def _concat_arrays(items):
    return LearnerArrays(**{field:np.concatenate([getattr(item,field) for item in items]) for field in LearnerArrays.__dataclass_fields__})


def _grid_from_metadata(values, metadata, axis_sizes, candidate):
    """Place row-aligned values into a semantic grid without relying on row order."""
    values = np.asarray(values)
    metadata = np.asarray(metadata, dtype=np.int64)
    axes = GRID_AXES["candidate" if candidate else "baseline"]
    expected_columns = len(axes)
    if metadata.ndim != 2 or metadata.shape[1] != expected_columns:
        raise ValueError(f"{axes} metadata must have {expected_columns} columns, got {metadata.shape}")
    if len(values) != len(metadata):
        raise ValueError(f"row/value count mismatch: {len(metadata)} metadata rows vs {len(values)} values")
    shape = tuple(int(axis_sizes[name]) for name in axes)
    if any(size <= 0 for size in shape):
        raise ValueError(f"all axis sizes must be positive, got {dict(zip(axes, shape))}")
    if len(metadata) != int(np.prod(shape)):
        raise ValueError(f"incomplete grid: expected {np.prod(shape)} rows for {dict(zip(axes, shape))}, got {len(metadata)}")
    for column, (name, size) in enumerate(zip(axes, shape)):
        if np.any(metadata[:, column] < 0) or np.any(metadata[:, column] >= size):
            raise ValueError(f"{name} index outside [0, {size})")
    flat = np.ravel_multi_index(tuple(metadata[:, i] for i in range(expected_columns)), shape)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("duplicate semantic evaluation rows")
    grid = np.empty(shape + values.shape[1:], dtype=values.dtype)
    grid[tuple(metadata[:, i] for i in range(expected_columns))] = values
    return grid


def _validate_shared_query_targets(baseline_target, candidate_target, baseline_meta, candidate_meta, axis_sizes):
    baseline = _grid_from_metadata(baseline_target, baseline_meta, axis_sizes, candidate=False)
    candidate = _grid_from_metadata(candidate_target, candidate_meta, axis_sizes, candidate=True)
    expected = np.broadcast_to(baseline[:, :, :, None, :, :], candidate.shape)
    if not np.array_equal(candidate, expected):
        mismatch = np.argwhere(np.any(candidate != expected, axis=-1))[0]
        raise ValueError(f"candidate and baseline do not share the same query target at semantic index {tuple(mismatch)}")


def _cluster_boot(values, seed, reps):
    rng=np.random.default_rng(seed); return values[rng.integers(0,len(values),size=(reps,len(values)))].mean(axis=1)


def _alignment(score,gain,seed,reps):
    def corr(aggregate):
        # Semantic layout is (history, candidate, query); rank candidates.
        rs=rankdata(np.round(aggregate,12),axis=1); rg=rankdata(np.round(gain_aggregate,12),axis=1)
        a=rs-rs.mean(axis=1,keepdims=True); b=rg-rg.mean(axis=1,keepdims=True)
        denominator=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
        if np.any(denominator == 0):
            raise ValueError("candidate-ranking alignment is undefined for a tied (history, query) cell")
        return float(np.mean(np.sum(a*b,axis=1)/denominator))
    if score.shape != gain.shape or score.ndim != 4:
        raise ValueError(f"alignment expects matching (system, history, candidate, query) arrays, got {score.shape} and {gain.shape}")
    gain_aggregate=gain.mean(axis=0); point=corr(score.mean(axis=0))
    rng=np.random.default_rng(seed); boot=[]
    for _ in range(reps):
        idx=rng.integers(0,len(score),size=len(score)); gain_aggregate=gain[idx].mean(axis=0); boot.append(corr(score[idx].mean(axis=0)))
    return {"mean":point,"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))}


def _selection(score_arrays,gain,seed,reps):
    if gain.ndim != 4:
        raise ValueError(f"selection expects (system, history, candidate, query), got {gain.shape}")
    rows=[]; regrets={}
    for name,score in score_arrays.items():
        if score.shape != gain.shape:
            raise ValueError(f"selector {name} shape {score.shape} does not match gain shape {gain.shape}")
        choice=score.argmax(axis=2); chosen=np.take_along_axis(gain,choice[:,:,None,:],axis=2).squeeze(2)
        regrets[name]=(gain.max(axis=2)-chosen).mean(axis=(1,2))
        boot=_cluster_boot(regrets[name],seed+len(rows),reps)
        rows.append({"selector":name,"mean_regret":float(regrets[name].mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))})
    comparisons=[]; base=regrets["conditional_physical_value"]
    for name in ("standalone_query_value","trajectory_diversity","action_diversity"):
        diff=regrets[name]-base; boot=_cluster_boot(diff,seed+100+len(comparisons),reps)
        comparisons.append({"comparison":f"{name}_minus_conditional","mean":float(diff.mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))})
    return rows,comparisons


def _prediction_replay_check(model, arrays, expected, norms, s2, device, persistent, atol=1e-7, rtol=1e-5):
    replay, _ = _predictions_and_z(model, arrays, norms, s2, device, persistent)
    difference = np.abs(replay - expected)
    if not np.allclose(replay, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"deterministic prediction replay mismatch: max_abs={difference.max()}")
    return {"max_abs":float(difference.max(initial=0.0)),"atol":atol,"rtol":rtol,"allclose":True}


def run(root,config_path,s0_root,split,output_root,workers,device_name,verify_predictions=False):
    root,config_path,s0_root,output_root=map(Path,(root,config_path,s0_root,output_root)); cfg=json.loads(config_path.read_text())
    base=json.loads((root/cfg["base_config"]).read_text()); s2=json.loads((root/cfg["s2_config"]).read_text()); s0=json.loads((s0_root/"s0_receipt.json").read_text())
    train_receipt=json.loads((root/cfg["s2_models"]/"s2_training_receipt.json").read_text())
    if train_receipt["status"]!="S2_LEARNER_SUFFICIENCY_GO": raise RuntimeError("S3 requires S2 GO")
    if any(cfg["access"].values()): raise RuntimeError("S3 cannot access sealed, A/B, or NAD")
    history_bank,query_bank=banks(s0["chosen_horizon_s"],base["model"]["timestep_s"])
    split_cfg=cfg[split]; systems=_systems(split_cfg["systems"],split_cfg["seed"],base["persistent_prior"])
    tasks=[(split_cfg["seed"],i,systems[i],cfg["nuisance_realizations"]) for i in range(len(systems))]
    ctx=mp.get_context("spawn")
    with ctx.Pool(max(1,workers),initializer=_init_worker,initargs=(base,s2,s0["chosen_horizon_s"])) as pool: chunks=list(pool.imap(_worker,tasks,chunksize=1))
    baseline=_concat_arrays([c[0] for c in chunks]); candidate=_concat_arrays([c[1] for c in chunks]); bmeta=np.concatenate([c[2] for c in chunks]); cmeta=np.concatenate([c[3] for c in chunks]); physical=np.concatenate([c[4] for c in chunks])
    if device_name=="auto": device_name="mps" if torch.backends.mps.is_available() else "cpu"
    device=torch.device(device_name); norms={k:v for k,v in np.load(root/cfg["s2_models"]/"train_only_normalization.npz").items()}
    raw=RawHistoryPredictor(48,16,s2["hidden_dim"],32).to(device); jepa=PersistentJEPA(48,16,32,s2).to(device)
    raw.load_state_dict(torch.load(root/cfg["s2_models"]/"raw_frozen.pt",map_location=device)); jepa.load_state_dict(torch.load(root/cfg["s2_models"]/"persistent_jepa_frozen.pt",map_location=device)); raw.eval(); jepa.eval()
    raw_b,_=_predictions_and_z(raw,baseline,norms,s2,device,False); raw_c,_=_predictions_and_z(raw,candidate,norms,s2,device,False); jepa_b,_=_predictions_and_z(jepa,baseline,norms,s2,device,True); jepa_c,_=_predictions_and_z(jepa,candidate,norms,s2,device,True)
    prediction_replay_checks={}
    if verify_predictions:
        prediction_replay_checks={
            "raw_baseline":_prediction_replay_check(raw,baseline,raw_b,norms,s2,device,False),
            "raw_candidate":_prediction_replay_check(raw,candidate,raw_c,norms,s2,device,False),
            "jepa_baseline":_prediction_replay_check(jepa,baseline,jepa_b,norms,s2,device,True),
            "jepa_candidate":_prediction_replay_check(jepa,candidate,jepa_c,norms,s2,device,True),
        }
    tb=(baseline.target-norms["target_mean"])/norms["target_std"]; tc=(candidate.target-norms["target_mean"])/norms["target_std"]
    losses={"raw_b":np.mean((raw_b-tb)**2,axis=1),"raw_c":np.mean((raw_c-tc)**2,axis=1),"jepa_b":np.mean((jepa_b-tb)**2,axis=1),"jepa_c":np.mean((jepa_c-tc)**2,axis=1)}
    n=len(systems); r=cfg["nuisance_realizations"]
    axis_sizes={"system":n,"realization":r,"history":len(history_bank),"candidate":len(history_bank),"query":len(query_bank)}
    _validate_shared_query_targets(baseline.target,candidate.target,bmeta,cmeta,axis_sizes)
    score_names=("conditional_physical_value","standalone_query_value","trajectory_diversity","action_diversity","oracle_reducible_value")
    scores={name:_grid_from_metadata(physical[:,i],cmeta,axis_sizes,candidate=True).mean(axis=1) for i,name in enumerate(score_names)}
    results={}
    diagnostic_values={**{f"score_{key}":value for key,value in scores.items()}}
    for model in ("raw","jepa"):
        bl=_grid_from_metadata(losses[f"{model}_b"],bmeta,axis_sizes,candidate=False).mean(axis=1)
        cl=_grid_from_metadata(losses[f"{model}_c"],cmeta,axis_sizes,candidate=True).mean(axis=1)
        gain=bl[:,:,None,:]-cl
        diagnostic_values[f"{model}_baseline_loss"]=bl; diagnostic_values[f"{model}_candidate_loss"]=cl; diagnostic_values[f"{model}_gain"]=gain
        selection,comparisons=_selection({k:scores[k] for k in ("conditional_physical_value","standalone_query_value","trajectory_diversity","action_diversity")},gain,split_cfg["seed"]+(0 if model=="raw" else 1000),cfg["bootstrap_replicates"])
        alignment=_alignment(scores["conditional_physical_value"],gain,split_cfg["seed"]+2000+(0 if model=="raw" else 1000),cfg["bootstrap_replicates"])
        results[model]={"selection_regret":selection,"comparisons":comparisons,"physical_gain_alignment":alignment}
    primary_pass=all(row["ci_low"]>0 for row in results["jepa"]["comparisons"]); alignment_pass=results["jepa"]["physical_gain_alignment"]["ci_low"]>0
    status=f"{split.upper()}_GO" if primary_pass and alignment_pass else f"{split.upper()}_NO_GO"
    output_root.mkdir(parents=True,exist_ok=True); pd.DataFrame(cmeta,columns=["system_index","realization","anchor_index","candidate_index","query_index"]).assign(**{name:physical[:,i] for i,name in enumerate(score_names)}).to_csv(output_root/f"{split}_physical_manifest.csv.gz",index=False,compression="gzip")
    np.savez_compressed(output_root/f"{split}_diagnostic_system_values.npz",**diagnostic_values)
    row_level_path=output_root/f"{split}_row_level_replay.npz"
    np.savez_compressed(row_level_path,
        systems=systems,baseline_meta=bmeta,candidate_meta=cmeta,
        baseline_target=baseline.target,candidate_target=candidate.target,physical=physical,
        raw_baseline_prediction=raw_b,raw_candidate_prediction=raw_c,
        jepa_baseline_prediction=jepa_b,jepa_candidate_prediction=jepa_c)
    receipt={"status":status,"split":split,"systems":n,"nuisance_realizations":r,"candidate_rows":len(cmeta),"baseline_rows":len(bmeta),"axis_semantics":{"scores_and_gain":["system","history","candidate","query"],"candidate_axis":2,"alignment_candidate_axis_after_system_aggregation":1},"results":results,"primary_pass":primary_pass,"alignment_pass":alignment_pass,"device":device_name,"models_retrained":False,"checkpoint_hashes":train_receipt["checkpoint_hashes"],"prediction_replay_checks":prediction_replay_checks,"row_level_replay_sha256":_sha256(row_level_path),"sealed_accessed":False,"protected_scope_2_touched":False,"config_sha256":_sha256(config_path),"evaluator_sha256":_sha256(Path(__file__))}
    (output_root/f"{split}_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); return receipt


def main():
    p=argparse.ArgumentParser(); p.add_argument("root");p.add_argument("config");p.add_argument("s0_root");p.add_argument("split",choices=("discovery","validation"));p.add_argument("output_root");p.add_argument("--workers",type=int,default=8);p.add_argument("--device",default="auto",choices=("auto","cpu","mps"));p.add_argument("--verify-predictions",action="store_true");a=p.parse_args()
    print(json.dumps(run(a.root,a.config,a.s0_root,a.split,a.output_root,a.workers,a.device,a.verify_predictions),sort_keys=True))


if __name__=="__main__": main()
