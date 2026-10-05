"""Bind and evaluate the one-seed native Overcooked comparison.

Only new external receipts are written. Original AD, paired models, training,
environment and evaluator sources remain unchanged. The reserved manifest
keeps its actual development/heldout split; a temporary in-memory TaskEntry
uses the old evaluator's train-only interface, never a training data writer.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from source_isolation import digest, read, require, rows, source_row

MODES = ("upstream_ad", "none", "VC", "I+VC")
UPDATES, SEED, EPISODES, BATCH = 20000, 4200, 20, 1024
ROLE_SPLITS = {"development", "heldout", "held_out", "validation"}
SOURCE_FILES = ("native_a/train.py", "native_a/model.py", "native_a/sampler.py", "native_a/losses.py",
                "native_a/evaluate.py", "native_a/original_checkpoint.py",
                "benchmarks/baselines/ad/train.py", "benchmarks/baselines/ad/model.py",
                "benchmarks/baselines/ad/buffer.py", "runners/history_adapter.py",
                "eval_icrl.py", "envs/__init__.py", "envs/overcooked_v2/overcooked.py",
                "native_a/large_batch.py", "native_a/joint_batch.py", "native_a/mapped_history.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite value in measurement or training log")
    elif isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def verify_ref(ref):
    require(digest(ref["path"]) == ref["sha256"], f"Bound input changed: {ref['path']}")


def runtime(args):
    require(platform.system() == "Linux", "Real inference belongs on the Linux experiment runtime")
    require(1 <= args.threads <= 24, "CPU evaluator thread budget must be 1..24")
    os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.threads])
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import jax
    require(jax.default_backend() == "cpu", "CPU inference backend required")


def adapted_task(raw, *, unseen):
    """Return an ephemeral interface object, preserving raw and its file hash."""
    from benchmarks.manifest_schema import TaskEntry
    require(raw["split"] in ROLE_SPLITS if unseen else raw["split"] == "train", "Wrong external source role")
    adapted = deepcopy(raw)
    if unseen:
        adapted["split"] = "train"
    task = TaskEntry.from_json(adapted)
    require(task.task_id == raw["task_id"], "Interface adapter changed partner task identity")
    return task


def audit_inputs(args):
    """Reuse the source preflight and additionally audit the actual HDF5."""
    from eval_icrl import create_env_for_task, create_teammate
    from native_a.train import tensor_sha
    from native_a.original_checkpoint import parameter_receipt
    import h5py
    plan, root, repo = read(args.plan), Path(args.run_root).resolve(), Path(args.repo).resolve()
    require(plan["model_seeds"] == [SEED] and plan["methods"] == list(MODES)
            and plan["model_fit_count"] == 4, "Single-seed four-method plan changed")
    require(plan["training"]["updates_per_method"] == UPDATES
            and plan["training"]["batch_size"] == BATCH, "Formal training budget changed")
    require(plan["evaluation"]["episodes_per_fixed_partner"] == EPISODES
            and plan["evaluation"]["episode_horizon"] == 100, "Evaluation budget changed")
    for name, sha in plan["source_hashes"].items():
        require(digest(repo / name) == sha, f"Plan source differs: {name}")
    train, unseen = rows(args.train_manifest), rows(args.unseen_manifest)
    require(len(train) == 20 and len(unseen) == 2, "Twenty training and two reserved fixed partners required")
    require(all(t["split"] == "train" for t in train), "Training manifest role changed")
    require(all(t["split"] in ROLE_SPLITS for t in unseen),
            "Reserved manifest must really be labeled development/heldout, not train")
    official = repo / "benchmarks/overcooked_icrl"
    require(not Path(args.unseen_manifest).resolve().is_relative_to(official), "Official benchmark test is outside this pass")
    identifiers = [t["task_id"] for t in train + unseen]
    require(len(set(identifiers)) == 22, "Duplicate task identities across source roles")
    source_receipts = []
    for reserved, tasks in ((False, train), (True, unseen)):
        role = "unseen_development" if reserved else "training"
        for raw in tasks:
            row = source_row(raw, role)
            task = adapted_task(raw, unseen=reserved)
            teammate = create_teammate(task, create_env_for_task(task, max_steps=100))
            require(teammate._params is not None, "Missing actual RL partner tensors")
            parameter_receipt(teammate._params)
            row.update(external_split=raw["split"], teammate_params_sha256=tensor_sha(teammate._params))
            source_receipts.append(row)
    qualified = read(root / "dataset_inputs/qualification.json")
    require(qualified["status"] == "PASS" and len(qualified["selected"]) == 20, "Frozen twenty-policy qualification missing")
    require(train == [r["task"] for r in qualified["selected"]]
            and unseen == [r["task"] for r in qualified["development"]], "Selected policy order changed")
    expected = {(r["source_seed"], r["checkpoint_idx"]) for r in qualified["selected"] + qualified["development"]}
    expected_hashes = {r["task"]["task_id"]: r["parameter_sha256"] for r in qualified["selected"] + qualified["development"]}
    require(all(r["teammate_params_sha256"] == expected_hashes[r["task_id"]] for r in source_receipts), "Qualified policy weights changed")
    require({(r["source_seed"], r["checkpoint_idx"]) for r in source_receipts} == expected, "Source cohort changed")
    require(len({r["teammate_params_sha256"] for r in source_receipts}) == 22, "Source populations share actual policy tensors")
    paths = {"h5": root / "dataset/histories.h5", "index": root / "dataset/histories_index.jsonl",
             "task_manifest": Path(args.train_manifest), "episode_index": root / "dataset/native_episode_index.jsonl"}
    data = {key: file_ref(path) for key, path in paths.items()}
    packed = read(root / "dataset/dataset_receipt.json")
    require(packed["status"] == "PASS" and packed["official_test_read"] is False
            and packed["expert_relabel"] is False
            and packed["true_episode_boundaries_not_inferred_from_packed_dones"] is True,
            "Dataset is not a passing original-history artifact")
    for key, field in (("h5", "h5_sha256"), ("index", "index_sha256"), ("episode_index", "episode_index_sha256")):
        require(data[key]["sha256"] == packed[field], "Actual data differs from packing receipt")
    index, allowed = rows(paths["index"]), {t["task_id"] for t in train}
    require(len(index) == 2560 and sum(r["T"] for r in index) == 37376000
            and {r["task_id"] for r in index} == allowed
            and all(r["split"] == "train" for r in index), "Reserved source is present in actual training index")
    by_history = {r["history_id"]: r for r in index}
    require(len(by_history) == len(index), "Duplicate packed history identity")
    with h5py.File(paths["h5"], "r") as store:
        require(set(store.keys()) == {r["h5_group"].lstrip("/") for r in index}, "HDF5 has unindexed histories")
        for row in index:
            g = store[row["h5_group"]]
            require(g.attrs["task_id"] == row["task_id"] and g.attrs["split"] == "train",
                    "Actual HDF5 group provenance differs from its index")
    for row in rows(paths["episode_index"]):
        expected_row = by_history.get(row["history_id"])
        require(expected_row and row["task_id"] == expected_row["task_id"]
                and row["task_id"] in allowed, "Episode sidecar contains another source")
    return {"status": "SOURCE_AND_DATA_ISOLATION_PASS", "plan": file_ref(args.plan),
            "train_manifest": file_ref(args.train_manifest), "unseen_manifest": file_ref(args.unseen_manifest),
            "dataset_receipt": file_ref(root / "dataset/dataset_receipt.json"), "data": data,
            "sources": source_receipts, "source_sha256": {name: digest(repo / name) for name in SOURCE_FILES},
            "official_benchmark_test_read": False, "gradient_training_on_reserved_sources": False,
            "split_interface_adapter": "copy in memory only for old TaskEntry/evaluation interface; never write into gradient data",
            "reserved_external_splits": sorted({t["split"] for t in unseen}),
            "statistical_scope": "one model seed and one reserved population; its checkpoints are not independent seeds"}


def bind(args):
    root = Path(args.run_root).resolve()
    for mode in MODES:
        require(not (root / "models" / mode.replace("+", "_") / "config.json").exists(),
                "Cannot retrospectively create a before-training binding for an already started fit")
    isolation = audit_inputs(args)
    plan = read(args.plan)
    for reference in plan["executable_tools"].values():
        verify_ref(reference)
    precision = read(plan["precision_gate"])
    require(precision["status"] == "PASS", "Joint1024 precision gate must pass before model fitting")
    isolation["precision_gate"] = file_ref(plan["precision_gate"])
    isolation["probe_contracts"] = {key: file_ref(value) for key, value in plan["probe_contracts"].items()}
    isolation["execution_tools"] = plan["executable_tools"]
    verify_ref(plan["mapped_history"])
    isolation["mapped_history"] = plan["mapped_history"]
    verify_ref(plan["checkpoint_gate"])
    isolation["checkpoint_gate"] = plan["checkpoint_gate"]
    write(root / "source_isolation.json", isolation)
    for mode in MODES:
        binding = {"status": "BOUND_BEFORE_TRAINING", "mode": mode, "seed": SEED,
                   "planned_updates": UPDATES, "batch_size": BATCH, "created_at_unix": time.time(),
                   "isolation": file_ref(root / "source_isolation.json"), **isolation}
        binding["status"] = "BOUND_BEFORE_TRAINING"
        write(root / "training_bindings" / (mode.replace("+", "_") + ".json"), binding)
    return {"status": "BOUND_BEFORE_TRAINING", "model_fits_started": 0, "model_seeds": [SEED]}


def model_binding(args, isolation):
    root, mode = Path(args.run_root).resolve(), args.mode
    path = root / "training_bindings" / (mode.replace("+", "_") + ".json")
    binding = read(path)
    require(binding["status"] == "BOUND_BEFORE_TRAINING" and binding["mode"] == mode
            and binding["seed"] == SEED and binding["planned_updates"] == UPDATES,
            "Missing formal before-training input binding")
    verify_ref(binding["isolation"])
    for key in ("plan", "data", "source_sha256", "sources", "train_manifest", "unseen_manifest"):
        require(binding[key] == isolation[key], f"Evaluation no longer matches frozen training input: {key}")
    execution = read(root / "execution_bindings" / (mode.replace("+", "_") + ".json"))
    require(execution["precision"] == "highest" and execution["effective_batch"] == 1024
            and execution["microbatch"] == 128 and execution["seed"] == SEED, "Execution numerics changed")
    verify_ref(execution["precision_gate"])
    for reference in execution["tools"].values():
        verify_ref(reference)
    model_root = root / "models" / mode.replace("+", "_")
    cfg = read(model_root / "config.json")
    require(cfg["seed"] == SEED and cfg["num_steps"] == UPDATES and cfg["batch_size"] == BATCH,
            "Formal fit seed or update budget changed")
    require(binding["created_at_unix"] <= (model_root / "config.json").stat().st_mtime,
            "Training input binding postdates the model configuration")
    architecture = cfg if mode == "upstream_ad" else cfg["ad"]
    for key, value in {"seq_len": 500, "embedding_dim": 64, "hidden_dim": 256,
                       "num_layers": 4, "num_heads": 4, "use_teammate_actions": False}.items():
        require(architecture[key] == value, f"Architecture/permission changed: {key}")
    return model_root, cfg, file_ref(path)


def load_model(args, isolation):
    from native_a.original_checkpoint import load_original_ad_checkpoint, parameter_receipt
    from native_a.train import load_native_checkpoint
    repo = Path(args.repo).resolve()
    root, cfg, binding = model_binding(args, isolation)
    if args.mode == "upstream_ad":
        for key in ("h5", "index"):
            require(Path(cfg[key + "_path"]).resolve() == Path(isolation["data"][key]["path"]),
                    "Original AD trained against another dataset path")
        require(cfg["learning_rate"] == 0.001 and cfg["weight_decay"] == 0
                and cfg["label_smoothing"] == 0 and cfg["ignore_loss_after_done"] is False,
                "Original AD objective/default optimizer changed")
        metrics = read(root / "metrics.json")
        require([r["step"] for r in metrics] == list(range(0, UPDATES, cfg["log_every"])),
                "Original AD logged update sequence incomplete")
        finite_json(metrics)
        model, params, model_cfg, receipt = load_original_ad_checkpoint(
            root / "checkpoints" / f"checkpoint_{UPDATES}", expected_step=UPDATES)
        receipt.update(training_binding=binding, training_metrics=file_ref(root / "metrics.json"))
    else:
        require(cfg["mode"] == args.mode and cfg["persistent_dim"] == 32
                and cfg["lambda_p"] == (0 if args.mode == "none" else 0.001)
                and cfg["cross_weight"] == 0, "Paired model or mechanism weight changed")
        require(cfg["data"] == isolation["data"], "Native checkpoint data fingerprint changed")
        for key, value in {"query_len": 300, "support_count": 2, "support_len": 100,
                           "total_history_tokens": 500, "split": "train", "query_loss_only": True,
                           "distinct_partners_per_batch": False}.items():
            require(cfg["sampler"][key] == value, "Paired sample or loss permissions changed")
        require(cfg["optimizer"]["learning_rate"] == 0.0003 and cfg["optimizer"]["weight_decay"] == 0,
                "Paired optimizer changed")
        done, terminal = read(root / "completion.json"), read(root / "terminal.json")
        require(done["status"] == "PASS" and done["actual_updates"] == done["planned_updates"] == UPDATES
                and done["terminal"] == terminal == read(root / "latest.json"), "Native fit is not terminal")
        model, params, loaded, receipt = load_native_checkpoint(root / "terminal.json")
        require(loaded == cfg and receipt["step"] == UPDATES, "Native optimizer checkpoint step/config differs")
        for name, sha in receipt["code_sha256"].items():
            require(digest(repo / name) == sha, f"Native checkpoint implementation changed: {name}")
        parameter_receipt(params)  # Reject NaN/Inf, not only hash mismatches.
        committed, cursor = [], 0
        for attempt in sorted((root / "attempts").glob("attempt_*")):
            start, finish = read(attempt / "started.json"), read(attempt / "completion.json")
            require(start["start_step"] == cursor, "Training resume lacks a contiguous committed prefix")
            endpoint = finish.get("actual_updates", finish.get("latest_checkpoint", {}).get("step"))
            require(endpoint is not None and cursor <= endpoint <= UPDATES, "Unverifiable committed update range")
            records = [r for r in rows(attempt / "metrics.jsonl") if cursor < r["step"] <= endpoint]
            require([r["step"] for r in records] == list(range(cursor + 1, endpoint + 1)), "Missing optimizer records")
            for record in records:
                finite_json(record)
                require(record["optimizer_update_completed"] is True and record["gradients_finite"] == 1,
                        "Nonfinite or incomplete native optimizer step")
            committed.extend(records)
            cursor = endpoint
        require(cursor == UPDATES and len(committed) == UPDATES, "Native optimizer history incomplete")
        initialization = read(root / "initialization.json")
        for sibling_mode in ("none", "VC", "I_VC"):
            sibling_path = root.parent / sibling_mode / "initialization.json"
            if sibling_path.exists():
                sibling = read(sibling_path)
                for key in ("params_sha256", "common_backbone_sha256", "first_plan_sha256"):
                    require(sibling[key] == initialization[key], "Paired methods have different initial weights or first batch")
        batch_plan_hash = hashlib.sha256("\n".join(
            canonical(record["batch_plan"]) for record in committed).encode()).hexdigest()
        model_cfg = model.config.ad
        receipt.update(training_binding=binding, verified_optimizer_updates=len(committed),
                       initialization=file_ref(root / "initialization.json"),
                       ordered_training_batch_plan_sha256=batch_plan_hash)
    return model, params, model_cfg, receipt


@contextmanager
def verified_partner_factory(expected):
    """Check weights at actual native rollout creation, without policy changes."""
    import eval_icrl
    from native_a.train import tensor_sha
    original = eval_icrl.create_teammate
    def checked(task, env):
        mate = original(task, env)
        require(task.task_id in expected and mate._params is not None
                and tensor_sha(mate._params) == expected[task.task_id], "Actual rollout partner is not the bound reserved policy")
        return mate
    eval_icrl.create_teammate = checked
    try:
        yield
    finally:
        eval_icrl.create_teammate = original


def summarize(episodes):
    require(len(episodes) == EPISODES, "Incomplete native evaluation stream")
    values = [float(r["return"]) for r in episodes]
    require(all(math.isfinite(x) for x in values), "Nonfinite native team return")
    return {"mean_return_all20": sum(values) / EPISODES, "auc_all20": sum(values),
            "mean_return_episodes6to20": sum(values[5:]) / 15,
            "per_episode_returns": values, "episodes": EPISODES}


def evaluate(args):
    import jax
    import numpy as np
    import eval_icrl as ev
    from native_a.evaluate import StaticADInference, check_static_equivalence, evaluate_native_task
    isolation = audit_inputs(args)
    model, params, model_cfg, checkpoint = load_model(args, isolation)
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing to overwrite or silently repeat an evaluation")
    output.mkdir(parents=True)
    sources = {r["task_id"]: r for r in isolation["sources"] if r["role"] == "unseen_development"}
    raw_tasks = rows(args.unseen_manifest)
    conditions = ["native_matched_history"] if args.mode == "upstream_ad" else ["matched", "null_support"]
    receipt = {"status": "RUNNING", "mode": args.mode, "model_seed": SEED, "evaluation_seed": SEED,
               "evaluation_role": "unseen_development", "official_benchmark_test_read": False,
               "optimizer_updates": 0, "episodes_per_task_condition": EPISODES, "conditions": conditions,
               "checkpoint": checkpoint, "isolation": isolation, "completed_episode_files": [], "results": [],
               "upstream_reference_context": "original rolling500-token full history; original action CE/sampler/inference",
               "paired_context": "300 recent query tokens plus two older100-token episodes; null removes only support slots",
               "comparison_boundary": "I+VC versus none/VC is matched; original AD is a separate native-method reference",
               "rng_boundary": "none/VC/I+VC share fold-in reset/noise streams; original AD retains its original sequential RNG schedule",
               "new_cross_seed_repeats": False}
    write(output / "evaluation.json", receipt)
    started = time.monotonic()
    try:
        if args.mode == "upstream_ad":
            receipt["static_inference_equivalence"] = check_static_equivalence(model, params, model_cfg.obs_shape)
            class FiniteStatic(StaticADInference):
                def apply(self, *pos, **kw):
                    logits = super().apply(*pos, **kw)
                    require(np.isfinite(logits).all(), "Nonfinite original AD logits")
                    return logits
            model = FiniteStatic(model, 500)
        with verified_partner_factory({key: r["teammate_params_sha256"] for key, r in sources.items()}):
            for raw in raw_tasks:
                task = adapted_task(raw, unseen=True)
                task_dir = output / "tasks" / hashlib.sha256(task.task_id.encode()).hexdigest()[:16]
                write(task_dir / "external_task.json", raw)
                for condition in conditions:
                    directory = task_dir / condition
                    committed = []
                    def on_episode(row):
                        episode = int(row.get("episode", row.get("episode_id", -1)))
                        require(episode == len(committed), "Episode sequence incomplete or duplicated")
                        entry = {**row, "episode": episode, "task_id": task.task_id, "external_split": raw["split"],
                                 "evaluation_role": "unseen_development", "mode": args.mode,
                                 "history_condition": condition, "teammate_params_sha256": sources[task.task_id]["teammate_params_sha256"],
                                 "optimizer_updates": 0, "official_benchmark_test_read": False}
                        finite_json(entry)
                        path = directory / f"episode_{episode:04d}.json"
                        write(path, entry)
                        committed.append(entry)
                        receipt["completed_episode_files"].append(file_ref(path))
                        receipt["elapsed_seconds"] = time.monotonic() - started
                        write(output / "evaluation.json", receipt, replace=True)
                    if args.mode == "upstream_ad":
                        config = ev.EvalConfig(algos=["ad"], tracks=["teammate"],
                            checkpoints={"ad": checkpoint["checkpoint"]}, num_episodes=EPISODES,
                            max_steps=100, context_len=500, seed=SEED, greedy=True, gpu="-1")
                        config.validate()
                        result = dataclasses.asdict(ev.evaluate_ad_task(model, params, task, config, model_cfg,
                                                                        jax.random.PRNGKey(SEED)))
                        for episode in result["per_episode"]:
                            on_episode(episode)
                    else:
                        result = evaluate_native_task(model, params, task, mode=args.mode, episodes=EPISODES,
                            seed=SEED, history_condition="null" if condition == "null_support" else "matched",
                            episode_callback=on_episode,
                            expected_teammate_params_sha256=sources[task.task_id]["teammate_params_sha256"])
                    finite_json(result)
                    summary = summarize(committed)
                    result.update(external_split=raw["split"], evaluation_role="unseen_development", summary=summary,
                                  split_interface_adapter="in memory only; no gradient update or manifest relabeling")
                    write(directory / "result.json", result)
                    receipt["results"].append({"task_id": task.task_id, "history_condition": condition,
                                               **summary, "artifact": file_ref(directory / "result.json")})
        require(len(receipt["completed_episode_files"]) == len(conditions) * 2 * EPISODES, "Evaluation episode total incomplete")
        for ref in isolation["data"].values():
            verify_ref(ref)
        for key in ("plan", "train_manifest", "unseen_manifest", "dataset_receipt"):
            verify_ref(isolation[key])
        for name, sha in isolation["source_sha256"].items():
            require(digest(Path(args.repo) / name) == sha, "Frozen source changed during inference")
        if args.mode == "upstream_ad":
            from native_a.original_checkpoint import inventory
            require(inventory(checkpoint["checkpoint"]) == checkpoint["checkpoint_files"],
                    "Original AD checkpoint changed during evaluation")
        else:
            verify_ref({"path": checkpoint["path"], "sha256": checkpoint["sha256"]})
        receipt["condition_aggregates"] = {
            condition: {"mean_return_episodes6to20": sum(r["mean_return_episodes6to20"] for r in receipt["results"]
                                                            if r["history_condition"] == condition) / 2,
                        "mean_return_all20": sum(r["mean_return_all20"] for r in receipt["results"]
                                                 if r["history_condition"] == condition) / 2,
                        "weighting": "equal fixed-partner weights; both checkpoints share one source population"}
            for condition in conditions}
        receipt["status"] = "PASS_EXECUTION"
        receipt["claim"] = "One-seed native cooperation measurements; completion does not require a positive A effect"
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write(output / "evaluation.json", receipt, replace=True)
    return {"status": receipt["status"], "mode": args.mode,
            "actual_episodes": len(receipt["completed_episode_files"]), "receipt": str(output / "evaluation.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("bind", "evaluate"))
    for name in ("repo", "run-root", "plan", "train-manifest", "unseen-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.operation == "evaluate":
        require(args.mode is not None and args.output, "Evaluation needs one method and a new output directory")
    runtime(args)
    print(json.dumps(bind(args) if args.operation == "bind" else evaluate(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
