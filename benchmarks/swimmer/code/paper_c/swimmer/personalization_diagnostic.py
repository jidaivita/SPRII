import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, rankdata

from paper_c.coupled_sled.learner import PersistentJEPA, RawHistoryPredictor, _predictions_and_z
from paper_c.coupled_sled.learner_data import LearnerArrays

from .s3_evaluate import (
    _concat_arrays,
    _grid_from_metadata,
    _init_worker,
    _sha256,
    _validate_shared_query_targets,
    _worker,
)
from .waveforms import banks


SCORE_NAMES = (
    "conditional_physical_value",
    "standalone_query_value",
    "trajectory_diversity",
    "action_diversity",
    "oracle_reducible_value",
)


def _canonical_system_bytes(theta):
    return np.ascontiguousarray(np.asarray(theta, dtype="<f8")).tobytes()


def _domain_hash(salt, source_system_index, theta):
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(int(source_system_index).to_bytes(8, "little", signed=False))
    digest.update(_canonical_system_bytes(theta))
    return digest.hexdigest()


def build_system_manifest(systems, system_count, selection_salt, crossfit_salt):
    rows = []
    for source_index, theta in enumerate(np.asarray(systems)):
        rows.append({
            "source_system_index":int(source_index),
            "theta":np.asarray(theta, dtype=np.float64).tolist(),
            "selection_hash":_domain_hash(selection_salt, source_index, theta),
            "crossfit_hash":_domain_hash(crossfit_salt, source_index, theta),
        })
    selected = sorted(rows, key=lambda row:row["selection_hash"])[:system_count]
    crossfit_order = sorted(range(system_count), key=lambda index:selected[index]["crossfit_hash"])
    half_by_selected_index = {index:("A" if rank < system_count // 2 else "B") for rank,index in enumerate(crossfit_order)}
    for local_index, row in enumerate(selected):
        row["local_system_index"] = local_index
        row["system_crossfit_half"] = half_by_selected_index[local_index]
    return selected


def _remap_arrays(arrays, local_system_index):
    values = {field.name:getattr(arrays, field.name) for field in fields(LearnerArrays)}
    values["system_index"] = np.full(len(arrays.system_index), local_system_index, dtype=np.int64)
    return LearnerArrays(**values)


def _diagnostic_worker(task):
    split_seed, source_system_index, local_system_index, theta, realizations = task
    baseline,candidate,bmeta,cmeta,physical = _worker((split_seed,source_system_index,theta,realizations))
    bmeta = bmeta.copy(); cmeta = cmeta.copy()
    bmeta[:,0] = local_system_index; cmeta[:,0] = local_system_index
    return _remap_arrays(baseline,local_system_index),_remap_arrays(candidate,local_system_index),bmeta,cmeta,physical


def _arrays_dict(prefix, arrays):
    return {f"{prefix}_{field.name}":getattr(arrays,field.name) for field in fields(LearnerArrays)}


def _arrays_from_archive(archive, prefix):
    return LearnerArrays(**{field.name:archive[f"{prefix}_{field.name}"] for field in fields(LearnerArrays)})


def _axis_sizes(system_count, realizations, history_count=6, candidate_count=6, query_count=6):
    return {"system":system_count,"realization":realizations,"history":history_count,"candidate":candidate_count,"query":query_count}


def _local_grid(values, metadata, start, shard_systems, realizations, candidate):
    local_meta = metadata.copy(); local_meta[:,0] -= start
    sizes = _axis_sizes(shard_systems,realizations)
    return _grid_from_metadata(values,local_meta,sizes,candidate)


def _allclose_record(new, old, atol, rtol, exact=False):
    difference = np.abs(np.asarray(new)-np.asarray(old))
    passed = np.array_equal(new,old) if exact else np.allclose(new,old,atol=atol,rtol=rtol)
    return {"passed":bool(passed),"max_abs":float(difference.max(initial=0.0)),"exact":exact,"atol":0.0 if exact else atol,"rtol":0.0 if exact else rtol}


def _load_context(root, config_path):
    root,config_path = Path(root),Path(config_path)
    cfg = json.loads(config_path.read_text())
    if any(cfg["access"].values()): raise RuntimeError("diagnostic cannot access sealed, A/B, or NAD")
    source_cfg = json.loads((root/cfg["source_evaluation_config"]).read_text())
    base = json.loads((root/source_cfg["base_config"]).read_text())
    s2 = json.loads((root/source_cfg["s2_config"]).read_text())
    s0_root = root/cfg["s0_root"]
    s0 = json.loads((s0_root/"s0_receipt.json").read_text())
    output_root = root/cfg["output_root"]
    return root,config_path,cfg,source_cfg,base,s2,s0,output_root


def generate(root, config_path):
    root,config_path,cfg,source_cfg,base,s2,s0,output_root = _load_context(root,config_path)
    output_root.mkdir(parents=True,exist_ok=True); shard_root=output_root/"shards"; shard_root.mkdir(exist_ok=True)
    source_path=root/cfg["source_discovery_replay"]
    with np.load(source_path) as source:
        source_systems=source["systems"].copy()
        old_bmeta=source["baseline_meta"].copy();old_cmeta=source["candidate_meta"].copy()
        old_btarget=source["baseline_target"].copy();old_ctarget=source["candidate_target"].copy();old_physical=source["physical"].copy()
    manifest=build_system_manifest(source_systems,cfg["system_count"],cfg["system_selection_hash_salt"],cfg["system_crossfit_hash_salt"])
    manifest_path=output_root/"system_manifest.json"
    manifest_payload={"status":"DIAGNOSTIC_ONLY","source_replay_sha256":_sha256(source_path),"systems":manifest}
    manifest_path.write_text(json.dumps(manifest_payload,indent=2,sort_keys=True)+"\n")
    split_seed=source_cfg["discovery"]["seed"]; realizations=cfg["nuisance_realizations"]
    tasks=[(split_seed,row["source_system_index"],row["local_system_index"],np.asarray(row["theta"]),realizations) for row in manifest]
    os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
    ctx=mp.get_context("spawn"); chunks=[]; shard_paths=[]
    with ctx.Pool(cfg["cpu_workers"],initializer=_init_worker,initargs=(base,s2,s0["chosen_horizon_s"])) as pool:
        for chunk in pool.imap(_diagnostic_worker,tasks,chunksize=1):
            chunks.append(chunk)
            if len(chunks)==cfg["shard_systems"]:
                shard_index=len(shard_paths); start=shard_index*cfg["shard_systems"]
                baseline=_concat_arrays([item[0] for item in chunks]);candidate=_concat_arrays([item[1] for item in chunks])
                bmeta=np.concatenate([item[2] for item in chunks]);cmeta=np.concatenate([item[3] for item in chunks]);physical=np.concatenate([item[4] for item in chunks])
                shard_path=shard_root/f"shard_{shard_index:02d}_inputs.npz"
                np.savez_compressed(shard_path,**_arrays_dict("baseline",baseline),**_arrays_dict("candidate",candidate),baseline_meta=bmeta,candidate_meta=cmeta,physical=physical,local_start=np.asarray(start),local_stop=np.asarray(start+len(chunks)))
                shard_paths.append(shard_path);chunks=[]
    if chunks: raise RuntimeError("system_count must be divisible by shard_systems")
    selected_source=np.asarray([row["source_system_index"] for row in manifest],dtype=np.int64)
    old_sizes=_axis_sizes(len(source_systems),source_cfg["nuisance_realizations"])
    old_btarget_grid=_grid_from_metadata(old_btarget,old_bmeta,old_sizes,False)[selected_source]
    old_ctarget_grid=_grid_from_metadata(old_ctarget,old_cmeta,old_sizes,True)[selected_source]
    old_physical_grid=_grid_from_metadata(old_physical,old_cmeta,old_sizes,True)[selected_source]
    new_btarget=[];new_ctarget=[];new_physical=[]
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            start=int(shard["local_start"]);stop=int(shard["local_stop"]);count=stop-start
            new_btarget.append(_local_grid(shard["baseline_target"],shard["baseline_meta"],start,count,realizations,False))
            new_ctarget.append(_local_grid(shard["candidate_target"],shard["candidate_meta"],start,count,realizations,True))
            new_physical.append(_local_grid(shard["physical"],shard["candidate_meta"],start,count,realizations,True))
    new_btarget=np.concatenate(new_btarget);new_ctarget=np.concatenate(new_ctarget);new_physical=np.concatenate(new_physical)
    atol=cfg["comparison_tolerance"]["atol"];rtol=cfg["comparison_tolerance"]["rtol"]
    replay_gate={
        "systems":_allclose_record(np.asarray([row["theta"] for row in manifest]),source_systems[selected_source],atol,rtol,exact=True),
        "baseline_target":_allclose_record(new_btarget[:,:4],old_btarget_grid,atol,rtol),
        "candidate_target":_allclose_record(new_ctarget[:,:4],old_ctarget_grid,atol,rtol),
        "physical":_allclose_record(new_physical[:,:4],old_physical_grid,atol,rtol),
    }
    if not all(item["passed"] for item in replay_gate.values()): raise RuntimeError(f"first-four generation replay gate failed: {replay_gate}")
    receipt={"status":"GENERATION_REPLAY_GATE_GO","diagnostic_only":True,"systems":cfg["system_count"],"realizations":realizations,"source_local_index_separated":True,"system_crossfit_half_counts":{"A":sum(row["system_crossfit_half"]=="A" for row in manifest),"B":sum(row["system_crossfit_half"]=="B" for row in manifest)},"nuisance_reliability_half":cfg["nuisance_reliability_half"],"nested_realization_counts":cfg["nested_realization_counts"],"first_four_replay_gate":replay_gate,"manifest_sha256":_sha256(manifest_path),"shard_hashes":{path.name:_sha256(path) for path in shard_paths},"source_replay_sha256":_sha256(source_path),"models_retrained":False,"sealed_accessed":False}
    (output_root/"generation_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def infer(root, config_path, device_name="auto"):
    root,config_path,cfg,source_cfg,base,s2,s0,output_root = _load_context(root,config_path)
    generation=json.loads((output_root/"generation_receipt.json").read_text())
    if generation["status"]!="GENERATION_REPLAY_GATE_GO": raise RuntimeError("inference requires generation replay GO")
    manifest=json.loads((output_root/"system_manifest.json").read_text())["systems"]
    selected_source=np.asarray([row["source_system_index"] for row in manifest],dtype=np.int64)
    if device_name=="auto": device_name="mps" if torch.backends.mps.is_available() else "cpu"
    device=torch.device(device_name);models_root=root/source_cfg["s2_models"]
    norms={key:value for key,value in np.load(models_root/"train_only_normalization.npz").items()}
    raw=RawHistoryPredictor(48,16,s2["hidden_dim"],32).to(device);jepa=PersistentJEPA(48,16,32,s2).to(device)
    raw.load_state_dict(torch.load(models_root/"raw_frozen.pt",map_location=device));jepa.load_state_dict(torch.load(models_root/"persistent_jepa_frozen.pt",map_location=device));raw.eval();jepa.eval()
    source_path=root/cfg["source_discovery_replay"]
    with np.load(source_path) as source:
        old_bmeta=source["baseline_meta"].copy();old_cmeta=source["candidate_meta"].copy()
        old_predictions={key:source[key].copy() for key in ("raw_baseline_prediction","raw_candidate_prediction","jepa_baseline_prediction","jepa_candidate_prediction")}
        old_system_count=len(source["systems"])
    old_sizes=_axis_sizes(old_system_count,source_cfg["nuisance_realizations"])
    old_grids={
        "raw_baseline":_grid_from_metadata(old_predictions["raw_baseline_prediction"],old_bmeta,old_sizes,False)[selected_source],
        "raw_candidate":_grid_from_metadata(old_predictions["raw_candidate_prediction"],old_cmeta,old_sizes,True)[selected_source],
        "jepa_baseline":_grid_from_metadata(old_predictions["jepa_baseline_prediction"],old_bmeta,old_sizes,False)[selected_source],
        "jepa_candidate":_grid_from_metadata(old_predictions["jepa_candidate_prediction"],old_cmeta,old_sizes,True)[selected_source],
    }
    n=cfg["system_count"];r=cfg["nuisance_realizations"]
    score_tensors={name:np.empty((n,r,6,6,6),dtype=np.float64) for name in SCORE_NAMES}
    losses={model:{"baseline":np.empty((n,r,6,6),dtype=np.float32),"candidate":np.empty((n,r,6,6,6),dtype=np.float32)} for model in ("raw","jepa")}
    prediction_gate={name:{"passed":True,"max_abs":0.0,"atol":cfg["comparison_tolerance"]["atol"],"rtol":cfg["comparison_tolerance"]["rtol"]} for name in old_grids}
    for shard_path in sorted((output_root/"shards").glob("shard_*_inputs.npz")):
        with np.load(shard_path) as shard:
            baseline=_arrays_from_archive(shard,"baseline");candidate=_arrays_from_archive(shard,"candidate")
            bmeta=shard["baseline_meta"].copy();cmeta=shard["candidate_meta"].copy();physical=shard["physical"].copy();start=int(shard["local_start"]);stop=int(shard["local_stop"]);count=stop-start
        raw_b,_=_predictions_and_z(raw,baseline,norms,s2,device,False);raw_c,_=_predictions_and_z(raw,candidate,norms,s2,device,False)
        jepa_b,_=_predictions_and_z(jepa,baseline,norms,s2,device,True);jepa_c,_=_predictions_and_z(jepa,candidate,norms,s2,device,True)
        for name,pred,meta,candidate_flag in (("raw_baseline",raw_b,bmeta,False),("raw_candidate",raw_c,cmeta,True),("jepa_baseline",jepa_b,bmeta,False),("jepa_candidate",jepa_c,cmeta,True)):
            grid=_local_grid(pred,meta,start,count,r,candidate_flag)
            record=_allclose_record(grid[:,:4],old_grids[name][start:stop],cfg["comparison_tolerance"]["atol"],cfg["comparison_tolerance"]["rtol"])
            prediction_gate[name]["passed"] &= record["passed"];prediction_gate[name]["max_abs"]=max(prediction_gate[name]["max_abs"],record["max_abs"])
        tb=(baseline.target-norms["target_mean"])/norms["target_std"];tc=(candidate.target-norms["target_mean"])/norms["target_std"]
        for model,pb,pc in (("raw",raw_b,raw_c),("jepa",jepa_b,jepa_c)):
            bl=np.mean((pb-tb)**2,axis=1);cl=np.mean((pc-tc)**2,axis=1)
            losses[model]["baseline"][start:stop]=_local_grid(bl,bmeta,start,count,r,False)
            losses[model]["candidate"][start:stop]=_local_grid(cl,cmeta,start,count,r,True)
        physical_grid=_local_grid(physical,cmeta,start,count,r,True)
        for column,name in enumerate(SCORE_NAMES): score_tensors[name][start:stop]=physical_grid[...,column]
    if not all(item["passed"] for item in prediction_gate.values()): raise RuntimeError(f"first-four prediction replay gate failed: {prediction_gate}")
    compact={f"score_{name}":value for name,value in score_tensors.items()}
    for model in ("raw","jepa"):
        compact[f"{model}_baseline_loss"]=losses[model]["baseline"];compact[f"{model}_candidate_loss"]=losses[model]["candidate"]
        compact[f"{model}_gain"]=losses[model]["baseline"][:,:,:,None,:]-losses[model]["candidate"]
    compact_path=output_root/"nuisance_level_semantic_tensors.npz";np.savez_compressed(compact_path,**compact)
    receipt={"status":"INFERENCE_REPLAY_GATE_GO","diagnostic_only":True,"device":device_name,"axis_semantics":["system","realization","history","candidate","query"],"first_four_prediction_replay_gate":prediction_gate,"compact_tensor_sha256":_sha256(compact_path),"checkpoint_hashes":{"raw":_sha256(models_root/"raw_frozen.pt"),"jepa":_sha256(models_root/"persistent_jepa_frozen.pt"),"normalization":_sha256(models_root/"train_only_normalization.npz")},"models_retrained":False,"sealed_accessed":False}
    (output_root/"inference_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def _candidate_spearman(a,b):
    ra=rankdata(np.round(a,12),axis=2);rb=rankdata(np.round(b,12),axis=2)
    da=ra-ra.mean(axis=2,keepdims=True);db=rb-rb.mean(axis=2,keepdims=True)
    den=np.sqrt(np.sum(da*da,axis=2)*np.sum(db*db,axis=2));num=np.sum(da*db,axis=2)
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=np.float64),where=den>0)


def _rank_metrics(first,second):
    spearman=_candidate_spearman(first,second)
    kendall=[]
    for system in range(first.shape[0]):
        values=[]
        for history in range(first.shape[1]):
            for query in range(first.shape[3]): values.append(kendalltau(first[system,history,:,query],second[system,history,:,query]).statistic)
        kendall.append(np.nanmean(values))
    top1=np.mean(np.argmax(first,axis=2)==np.argmax(second,axis=2),axis=(1,2))
    top2_first=np.argpartition(first,-2,axis=2)[:,:,-2:,:];top2_second=np.argpartition(second,-2,axis=2)[:,:,-2:,:]
    overlap=[]
    for system in range(first.shape[0]):
        total=[]
        for history in range(first.shape[1]):
            for query in range(first.shape[3]): total.append(len(set(top2_first[system,history,:,query])&set(top2_second[system,history,:,query]))/2)
        overlap.append(np.mean(total))
    per_system=np.nanmean(spearman,axis=(1,2));mean=float(np.nanmean(per_system));sb=float(2*mean/(1+mean)) if mean>-1 else float("nan")
    return {"spearman":mean,"spearman_brown":sb,"kendall":float(np.nanmean(kendall)),"top1":float(np.mean(top1)),"top2_overlap":float(np.mean(overlap)),"per_system_spearman":per_system}


def _bootstrap_interval(per_system,indices):
    boot=np.asarray(per_system)[indices].mean(axis=1)
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]


def _take_candidate(values,choices):
    indices=np.broadcast_to(choices[:,None,:,None,:],(values.shape[0],values.shape[1],values.shape[2],1,values.shape[4]))
    return np.take_along_axis(values,indices,axis=3).squeeze(3)


def _take_candidate_by_realization(values,choices):
    """Take candidate values when the frozen choice may differ by realization fold."""
    indices=choices[:,:,:,None,:]
    return np.take_along_axis(values,indices,axis=3).squeeze(3)


def _crossfit_choices(score_mean,halves):
    choices=np.empty((len(score_mean),score_mean.shape[1],score_mean.shape[3]),dtype=np.int64)
    for target,source in (("A","B"),("B","A")):
        target_mask=halves==target;source_mask=halves==source
        population=score_mean[source_mask].mean(axis=0).argmax(axis=1)
        choices[target_mask]=population
    return choices


def _crossfit_center(values,halves):
    """Remove population structure using only the opposite system half."""
    centered=np.empty_like(values,dtype=np.float64)
    for target,source in (("A","B"),("B","A")):
        target_mask=halves==target;source_mask=halves==source
        centered[target_mask]=values[target_mask]-values[source_mask].mean(0,keepdims=True)
    return centered


def _double_crossfit_physical_choices(score,halves):
    """Choose on one nuisance half and evaluate on the other.

    Personalized choices use the same system's selection half. Population
    choices additionally use only systems in the opposite system cross-fit
    half. Returned choices are indexed by evaluation realization, so no cell is
    evaluated on nuisance realizations that helped choose its candidate.
    """
    n,r,h,e,q=score.shape
    even=np.arange(r)%2==0;odd=~even
    personal=np.empty((n,r,h,q),dtype=np.int64)
    population=np.empty((n,r,h,q),dtype=np.int64)
    for selection_mask,evaluation_mask in ((even,odd),(odd,even)):
        selection_mean=score[:,selection_mask].mean(1)
        personal_choice=selection_mean.argmax(2)
        personal[:,evaluation_mask]=personal_choice[:,None]
        for target,source in (("A","B"),("B","A")):
            target_mask=halves==target;source_mask=halves==source
            population_choice=selection_mean[source_mask].mean(0).argmax(1)
            target_indices=np.flatnonzero(target_mask);evaluation_indices=np.flatnonzero(evaluation_mask)
            population[np.ix_(target_indices,evaluation_indices)]=population_choice[None,None]
    return personal,population


def _crossfit_range(values):
    n,r,h,e,q=values.shape;first=np.arange(r)%2==0;second=~first
    mean_first=values[:,first].mean(1);mean_second=values[:,second].mean(1)
    max_first=mean_first.argmax(2);min_first=mean_first.argmin(2);max_second=mean_second.argmax(2);min_second=mean_second.argmin(2)
    differences=np.empty((n,r,h,q),dtype=np.float64)
    for mask,max_choice,min_choice in ((second,max_first,min_first),(first,max_second,min_second)):
        chosen_max=_take_candidate(values[:,mask],max_choice);chosen_min=_take_candidate(values[:,mask],min_choice)
        differences[:,mask]=chosen_max-chosen_min
    mean=differences.mean(1);se=differences.std(1,ddof=1)/np.sqrt(r);lower=mean-1.96*se
    return mean,se,lower,lower>0


def _opportunity(score,gain,halves):
    score_mean=score.mean(1);gain_mean=gain.mean(1)
    descriptive_personal=score_mean.argmax(2);descriptive_population=_crossfit_choices(score_mean,halves)
    personal,population=_double_crossfit_physical_choices(score,halves)
    personal_score=_take_candidate_by_realization(score,personal).mean(1);population_score=_take_candidate_by_realization(score,population).mean(1)
    personal_gain=_take_candidate_by_realization(gain,personal).mean(1);population_gain=_take_candidate_by_realization(gain,population).mean(1)
    o_phys=personal_score-population_score;o_learner=personal_gain-population_gain
    # A common full-sample learner oracle is only a non-negative descriptive
    # reference for regret. The scientific comparison between the two frozen
    # physical selectors remains their paired, double-cross-fitted gain gap.
    best_gain=gain_mean.max(2)
    personal_regret=best_gain-personal_gain;population_regret=best_gain-population_gain
    range_phys,_,lower_phys,effective_phys=_crossfit_range(score);range_gain,_,lower_gain,effective_gain=_crossfit_range(gain)
    normalized_phys=float(o_phys[effective_phys].sum()/range_phys[effective_phys].sum()) if np.any(effective_phys) else float("nan")
    normalized_learner=float(o_learner[effective_gain].sum()/range_gain[effective_gain].sum()) if np.any(effective_gain) else float("nan")
    descriptive_personal_score=_take_candidate(score,descriptive_personal).mean(1)
    descriptive_population_score=_take_candidate(score,descriptive_population).mean(1)
    return {"choice_disagreement_per_system":np.mean(personal!=population,axis=(1,2,3)),"o_phys_per_system":o_phys.mean(axis=(1,2)),"o_learner_per_system":o_learner.mean(axis=(1,2)),"personal_regret_per_system":personal_regret.mean(axis=(1,2)),"population_regret_per_system":population_regret.mean(axis=(1,2)),"normalized_o_phys":normalized_phys,"normalized_o_learner":normalized_learner,"effective_phys_fraction":float(np.mean(effective_phys)),"effective_learner_fraction":float(np.mean(effective_gain)),"physical_range_lower_mean":float(lower_phys.mean()),"learner_range_lower_mean":float(lower_gain.mean()),"descriptive_fullsample_choice_disagreement":float(np.mean(descriptive_personal!=descriptive_population)),"descriptive_fullsample_o_phys":float(np.mean(descriptive_personal_score-descriptive_population_score)),"nuisance_choice_evaluation_crossfit":True,"system_population_crossfit":True,"regret_reference":"full-sample descriptive learner oracle; selector choices double-crossfit"}


def analyze(root,config_path):
    root,config_path,cfg,source_cfg,base,s2,s0,output_root=_load_context(root,config_path)
    inference=json.loads((output_root/"inference_receipt.json").read_text())
    if inference["status"]!="INFERENCE_REPLAY_GATE_GO": raise RuntimeError("analysis requires inference replay GO")
    manifest=json.loads((output_root/"system_manifest.json").read_text())["systems"]
    halves=np.asarray([row["system_crossfit_half"] for row in manifest])
    with np.load(output_root/"nuisance_level_semantic_tensors.npz") as archive: values={name:archive[name].copy() for name in archive.files}
    score=values["score_conditional_physical_value"];n=len(score)
    rng=np.random.default_rng(77431);boot_indices=rng.integers(0,n,size=(cfg["bootstrap_replicates"],n))
    convergence=[];per_r={}
    for r in cfg["nested_realization_counts"]:
        even=np.arange(r)%2==0;odd=~even
        s_first=score[:,:r][:,even].mean(1);s_second=score[:,:r][:,odd].mean(1)
        s_rel=_rank_metrics(s_first,s_second)
        s_centered_rel=_rank_metrics(_crossfit_center(s_first,halves),_crossfit_center(s_second,halves))
        row={"R":r,"rel_s":s_rel["spearman_brown"],"rel_s_split":s_rel["spearman"],"rel_s_kendall":s_rel["kendall"],"rel_s_top1":s_rel["top1"],"rel_s_top2":s_rel["top2_overlap"],"rel_s_centered":s_centered_rel["spearman_brown"],"rel_s_centered_split":s_centered_rel["spearman"]}
        per_r[r]={"score_reliability":s_rel}
        s_mean=score[:,:r].mean(1)
        for model in ("raw","jepa"):
            gain=values[f"{model}_gain"][:,:r]
            g_first=gain[:,even].mean(1);g_second=gain[:,odd].mean(1);g_rel=_rank_metrics(g_first,g_second)
            g_centered_rel=_rank_metrics(_crossfit_center(g_first,halves),_crossfit_center(g_second,halves))
            g_mean=gain.mean(1)
            population=float(np.nanmean(_candidate_spearman(s_mean.mean(0,keepdims=True),g_mean.mean(0,keepdims=True))))
            per_system_values=np.nanmean(_candidate_spearman(s_mean,g_mean),axis=(1,2));per_system=float(np.nanmean(per_system_values))
            s_residual=_crossfit_center(s_mean,halves);g_residual=_crossfit_center(g_mean,halves)
            centered_values=np.nanmean(_candidate_spearman(s_residual,g_residual),axis=(1,2));centered=float(np.nanmean(centered_values))
            total_ceiling=float(np.sqrt(s_rel["spearman_brown"]*g_rel["spearman_brown"])) if s_rel["spearman_brown"]>0 and g_rel["spearman_brown"]>0 else float("nan")
            centered_ceiling=float(np.sqrt(s_centered_rel["spearman_brown"]*g_centered_rel["spearman_brown"])) if s_centered_rel["spearman_brown"]>0 and g_centered_rel["spearman_brown"]>0 else float("nan")
            row.update({f"rel_g_{model}":g_rel["spearman_brown"],f"rel_g_{model}_split":g_rel["spearman"],f"rel_g_{model}_kendall":g_rel["kendall"],f"rel_g_{model}_top1":g_rel["top1"],f"rel_g_{model}_top2":g_rel["top2_overlap"],f"rel_g_centered_{model}":g_centered_rel["spearman_brown"],f"rel_g_centered_split_{model}":g_centered_rel["spearman"],f"ceiling_total_{model}":total_ceiling,f"ceiling_{model}":centered_ceiling,f"population_alignment_{model}":population,f"per_system_alignment_{model}":per_system,f"centered_alignment_{model}":centered,f"centered_ci_low_{model}":_bootstrap_interval(centered_values,boot_indices)[0],f"centered_ci_high_{model}":_bootstrap_interval(centered_values,boot_indices)[1],f"centered_to_ceiling_{model}":centered/centered_ceiling if centered_ceiling>0 else float("nan")})
            per_r[r][model]={"gain_reliability":g_rel,"centered_gain_reliability":g_centered_rel,"population_alignment":population,"per_system_values":per_system_values,"centered_values":centered_values,"total_ceiling":total_ceiling,"centered_ceiling":centered_ceiling}
        convergence.append(row)
    convergence_frame=pd.DataFrame(convergence);convergence_frame.to_csv(output_root/"reliability_alignment_convergence.csv",index=False)
    opportunity={}
    r=max(cfg["nested_realization_counts"])
    for model in ("raw","jepa"):
        metrics=_opportunity(score[:,:r],values[f"{model}_gain"][:,:r],halves)
        result={key:value for key,value in metrics.items() if not key.endswith("_per_system")}
        for key,value in metrics.items():
            if key.endswith("_per_system"):
                result[key.replace("_per_system","")]=float(np.mean(value));ci=_bootstrap_interval(value,boot_indices);result[key.replace("_per_system","")+"_ci"] = ci
        opportunity[model]=result
    (output_root/"opportunity_realization.json").write_text(json.dumps(opportunity,indent=2,sort_keys=True)+"\n")
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(convergence_frame.R,convergence_frame.rel_s,marker="o",label="Physical S (total)")
    ax.plot(convergence_frame.R,convergence_frame.rel_s_centered,marker="o",label="Physical S (centered)",linestyle="--");ax.plot(convergence_frame.R,convergence_frame.rel_g_jepa,marker="o",label="JEPA gain (total)");ax.plot(convergence_frame.R,convergence_frame.rel_g_centered_jepa,marker="o",label="JEPA gain (centered)",linestyle="--");ax.set(xlabel="Nuisance realizations R",ylabel="Spearman-Brown reliability",ylim=(-.05,1.05),title="Total and instance-residual reliability");ax.legend();fig.tight_layout();fig.savefig(output_root/"01_measurement_reliability.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(convergence_frame.R,convergence_frame.population_alignment_jepa,marker="o",label="Population")
    ax.plot(convergence_frame.R,convergence_frame.per_system_alignment_jepa,marker="o",label="Per-system");ax.plot(convergence_frame.R,convergence_frame.centered_alignment_jepa,marker="o",label="Centered");ax.plot(convergence_frame.R,convergence_frame.ceiling_jepa,marker="o",label="Empirical ceiling",linestyle="--");ax.set(xlabel="Nuisance realizations R",ylabel="Candidate-rank Spearman",ylim=(-.05,1.05),title="JEPA alignment convergence");ax.legend();fig.tight_layout();fig.savefig(output_root/"02_alignment_vs_ceiling.png",dpi=180);plt.close(fig)
    j=opportunity["jepa"]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4));ax1.bar(["Full-sample\nchoice gap","Cross-fit\nchoice gap","Effective\ncontexts"],[j["descriptive_fullsample_choice_disagreement"],j["choice_disagreement"],j["effective_phys_fraction"]]);ax1.set(ylim=(0,1),ylabel="Fraction",title="Candidate-choice structure")
    ax2.bar(["Full-sample\n(descriptive)","Held-out\n(double cross-fit)"],[j["descriptive_fullsample_o_phys"],j["o_phys"]]);ax2.axhline(0,color="black",linewidth=.8);ax2.set(ylabel="Physical value advantage",title="Personalized minus population");fig.tight_layout();fig.savefig(output_root/"03_physical_opportunity.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(["Personalized physical","Population physical"],[j["personal_regret"],j["population_regret"]]);ax.set(ylabel="JEPA regret",title="Personalized vs population physical choice");fig.tight_layout();fig.savefig(output_root/"04_learner_realization.png",dpi=180);plt.close(fig)
    last=convergence[-1];receipt={"status":"DIAGNOSTIC_ANALYSIS_COMPLETE","diagnostic_only":True,"R_values":cfg["nested_realization_counts"],"R32":{"reliability_s_total":last["rel_s"],"reliability_s_centered":last["rel_s_centered"],"reliability_g_jepa_total":last["rel_g_jepa"],"reliability_g_jepa_centered":last["rel_g_centered_jepa"],"empirical_centered_ceiling_jepa":last["ceiling_jepa"],"empirical_total_ceiling_jepa":last["ceiling_total_jepa"],"population_alignment_jepa":last["population_alignment_jepa"],"per_system_alignment_jepa":last["per_system_alignment_jepa"],"centered_alignment_jepa":last["centered_alignment_jepa"],"centered_alignment_jepa_ci":[last["centered_ci_low_jepa"],last["centered_ci_high_jepa"]],"alignment_relative_to_empirical_centered_ceiling":last["centered_to_ceiling_jepa"]},"opportunity":opportunity,"measurement_ceiling_is_diagnostic_not_corrected_truth":True,"centered_ceiling_uses_centered_reliabilities":True,"ratio_above_one_not_clamped":True,"coupled_sled_unlocked":False,"z_analysis_unlocked":False,"models_retrained":False,"sealed_accessed":False,"artifact_hashes":{"manifest":_sha256(output_root/"system_manifest.json"),"generation_receipt":_sha256(output_root/"generation_receipt.json"),"inference_receipt":_sha256(output_root/"inference_receipt.json"),"compact_tensors":_sha256(output_root/"nuisance_level_semantic_tensors.npz"),"convergence":_sha256(output_root/"reliability_alignment_convergence.csv"),"opportunity":_sha256(output_root/"opportunity_realization.json")}}
    (output_root/"diagnostic_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def run_all(root,config_path,device_name="auto"):
    return {"generation":generate(root,config_path),"inference":infer(root,config_path,device_name),"analysis":analyze(root,config_path)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("root");parser.add_argument("config");parser.add_argument("command",choices=("generate","infer","analyze","all"));parser.add_argument("--device",default="auto",choices=("auto","cpu","mps"));args=parser.parse_args()
    if args.command=="generate":result=generate(args.root,args.config)
    elif args.command=="infer":result=infer(args.root,args.config,args.device)
    elif args.command=="analyze":result=analyze(args.root,args.config)
    else:result=run_all(args.root,args.config,args.device)
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
