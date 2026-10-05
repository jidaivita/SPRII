"""One bounded native-HDF5 engineering check; never a method-effect experiment.

Runs baseline/none/VC/I+VC sequentially for exactly eight updates each, then
native matched/null evaluation.  Only I+VC has one deliberate wall-budget
pause and one resume.  There are no retries, hyperparameter searches, extra
updates, or official-test inputs.  This driver itself uses only the stdlib.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "none", "VC", "I+VC")
SEED, UPDATES, BATCH_SIZE, EPISODES = 4200, 8, 2, 6


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite numeric result")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def inside(root, name):
    path = Path(name)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink(),
            f"Missing file or path outside owned run: {path}")
    return resolved


def pipeline_health(root):
    summary_path = root / "pipeline_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    require(summary.get("status") not in ("STOPPED", "FAIL", "FAILED", "TECHNICAL_FAILURE"),
            f"Pipeline stopped: {summary.get('error', summary.get('status'))}")
    require(summary.get("official_test_read", False) is False, "Pipeline reports official test access")
    for stage in sorted((root / "stages").glob("*")):
        attempts = sorted(stage.glob("attempt_*"))
        if attempts and (attempts[-1] / "completion.json").exists():
            receipt = read_json(attempts[-1] / "completion.json")
            require(receipt.get("status") == "PASS", f"Failed pipeline stage: {stage.name}")
    return summary


def wait_for_dataset(root, out):
    started = time.monotonic()
    while True:
        summary = pipeline_health(root)
        attempts = sorted((root / "stages/build_dataset").glob("attempt_*"))
        stage = attempts[-1] / "completion.json" if attempts else None
        if stage is not None and stage.is_file():
            completion = read_json(stage)
            require(completion.get("status") == "PASS" and completion.get("exit_code") == 0
                    and completion.get("stage") == "build_dataset" and completion.get("phase") == "dataset"
                    and completion.get("official_test_read") is False, "Dataset stage has not passed execution QA")
            return validate_dataset(root, stage, time.monotonic() - started)
        require(summary.get("status") != "PASS", "Pipeline finished before a passing build_dataset stage")
        write_json(out / "progress.json", {"status": "WAITING_FOR_DATASET", "at": now(),
                   "pipeline_root": str(root), "wait_seconds": time.monotonic() - started,
                   "training_updates_executed": 0, "official_test_read": False})
        print(canonical({"event": "waiting_for_dataset", "at": now(), "poll_seconds": 30}), flush=True)
        time.sleep(30)


def validate_dataset(root, stage_path, wait_seconds):
    completion = read_json(stage_path)
    receipt_path = root / "dataset/dataset_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "PASS" and receipt.get("num_histories", 0) > 0
            and receipt.get("official_test_read") is False and receipt.get("expert_relabel") is False
            and receipt.get("true_episode_boundaries_not_inferred_from_packed_dones") is True,
            "Dataset receipt is not a native train-only PASS")
    paths = {"h5": root / "dataset/histories.h5", "index": root / "dataset/histories_index.jsonl",
             "episode_index": root / "dataset/native_episode_index.jsonl",
             "task_manifest": root / "dataset_inputs/train_manifest.jsonl"}
    for key, field in (("h5", "h5_sha256"), ("index", "index_sha256"), ("episode_index", "episode_index_sha256")):
        require(digest(paths[key]) == receipt[field], f"Dataset receipt hash mismatch: {key}")
    inventory = {str(Path(name).resolve()): value for name, value in completion["output_inventory"].items()}
    for path in (paths["h5"], paths["index"], paths["episode_index"], receipt_path):
        recorded = inventory.get(str(path))
        require(recorded and recorded["sha256"] == digest(path) and recorded["bytes"] == path.stat().st_size,
                f"Passing stage inventory no longer matches {path}")
    tasks = rows(paths["task_manifest"])
    require(tasks and all(t.get("split") == "train" and t.get("teammate", {}).get("kind") == "rl" for t in tasks),
            "Only custom RL train tasks are permitted")
    ids = [t["task_id"] for t in tasks]
    require(len(ids) == len(set(ids)), "Duplicate engineering task IDs")
    identities = {canonical({k: v for k, v in t["teammate"].items() if k not in ("seed", "base_seed")}) for t in tasks}
    require(len(identities) >= BATCH_SIZE, "At least two distinct fixed partners are required; random seeds alone do not count")
    manifest_receipt_path = root / "dataset_inputs/manifest_receipt.json"
    manifest_receipt = read_json(manifest_receipt_path)
    require(manifest_receipt.get("official_test_read") is False and manifest_receipt.get("role") == "training_RL_partners"
            and manifest_receipt.get("num_tasks") == len(tasks), "Merged manifest receipt mismatch")
    return {"paths": {k: str(v) for k, v in paths.items()}, "files": {k: ref(v) for k, v in paths.items()},
            "dataset_receipt": ref(receipt_path), "dataset_stage": ref(stage_path),
            "manifest_receipt": ref(manifest_receipt_path), "task_ids": ids,
            "distinct_partner_specs": len(identities), "wait_seconds": wait_seconds, "official_test_read": False}


def train_command(data, mode, out, threads):
    return [sys.executable, "-m", "native_a.train", "--h5-path", data["paths"]["h5"],
            "--index-path", data["paths"]["index"], "--task-manifest", data["paths"]["task_manifest"],
            "--episode-index", data["paths"]["episode_index"], "--out-dir", str(out), "--mode", mode,
            "--lambda-p", "0.001" if mode in ("VC", "I+VC") else "0", "--cross-weight", "0",
            "--num-steps", str(UPDATES), "--batch-size", str(BATCH_SIZE), "--seed", str(SEED),
            "--seq-len", "500", "--query-len", "300", "--support-count", "2", "--support-len", "100",
            "--learning-rate", "0.0003", "--save-every", "1", "--log-every", "1", "--threads", str(threads)]


def checked_pointer(run, pointer):
    path = inside(run, pointer["path"])
    require(digest(path) == pointer["sha256"], "Checkpoint pointer hash mismatch")
    return path


def verify_pause(run):
    attempts = sorted((run / "attempts").glob("attempt_*"))
    require(len(attempts) == 1, "The deliberate pause must be the first and only attempt")
    receipt = read_json(attempts[0] / "completion.json")
    latest = read_json(run / "latest.json")
    require(receipt.get("status") == "INCOMPLETE_BUDGET" and receipt["latest_checkpoint"] == latest,
            "I+VC first attempt did not stop through the intended budget boundary")
    require(1 <= latest["step"] < UPDATES and not (run / "completion.json").exists(),
            "Pause requires a real committed update before the unchanged terminal budget")
    checked_pointer(run, latest)
    return {"status": "PAUSED_AFTER_COMMITTED_UPDATE", "step": latest["step"],
            "checkpoint": latest, "receipt": ref(attempts[0] / "completion.json")}


def committed_training(run, mode, data):
    config, completion = read_json(run / "config.json"), read_json(run / "completion.json")
    require(completion.get("status") == "PASS" and completion.get("actual_updates") == UPDATES
            and completion.get("planned_updates") == UPDATES and completion.get("mode") == mode,
            f"Incomplete fixed engineering budget: {mode}")
    require(config["mode"] == mode and config["seed"] == SEED and config["num_steps"] == UPDATES
            and config["batch_size"] == BATCH_SIZE and config["cross_weight"] == 0
            and config["lambda_p"] == (0.001 if mode in ("VC", "I+VC") else 0), "Training configuration changed")
    require(all(config["ad"].get(k) == v for k, v in {"embedding_dim": 64, "hidden_dim": 256,
            "num_layers": 4, "num_heads": 4, "seq_len": 500, "use_teammate_actions": False}.items()), "Original AD architecture changed")
    require(config["optimizer"]["learning_rate"] == 3e-4 and config["optimizer"]["weight_decay"] == 0,
            "Engineering optimizer changed")
    require(all(config["sampler"].get(k) == v for k, v in {"query_len": 300, "support_count": 2,
            "support_len": 100, "total_history_tokens": 500, "split": "train", "query_loss_only": True,
            "distinct_partners_per_batch": True}.items()), "Paired history/label permissions changed")
    require(all(config["data"][k]["sha256"] == f["sha256"] for k, f in data["files"].items()), "Training data fingerprint mismatch")
    terminal = read_json(run / "terminal.json")
    require(terminal == read_json(run / "latest.json") == completion["terminal"] and terminal["step"] == UPDATES,
            "Terminal/latest/completion pointers differ")
    checked_pointer(run, terminal)
    committed, attempt_refs, cursor = [], [], 0
    attempts = sorted((run / "attempts").glob("attempt_*"))
    require(len(attempts) == (2 if mode == "I+VC" else 1), "Unexpected extra/retried training attempt")
    for i, attempt in enumerate(attempts):
        start, done = read_json(attempt / "started.json"), read_json(attempt / "completion.json")
        require(start["start_step"] == cursor and start["resume"] == (i > 0), "Resume skipped or repeated a committed prefix")
        if done["status"] == "INCOMPLETE_BUDGET":
            require(mode == "I+VC" and i == 0, "Unexpected budget failure")
            end = done["latest_checkpoint"]["step"]
            checked_pointer(run, done["latest_checkpoint"])
        else:
            require(done["status"] == "PASS", "Failed attempt cannot be selected away")
            end = done["actual_updates"]
            require(digest(attempt / "metrics.jsonl") == done["metrics_sha256"], "Committed metrics hash mismatch")
        all_rows = rows(attempt / "metrics.jsonl")
        selected = [row for row in all_rows if cursor < row["step"] <= end]
        require([row["step"] for row in selected] == list(range(cursor + 1, end + 1)), "Committed metric sequence is incomplete or duplicated")
        require(len(selected) == len(all_rows), "Unexpected uncommitted metrics in this controlled save-every-1 check")
        for row in selected:
            finite(row)
            require(row["optimizer_update_completed"] is True and row["device"] == "cpu"
                    and row["gradients_finite"] == 1, "Invalid real CPU optimizer receipt")
            plan = row["batch_plan"]
            require(plan["batch_plan_sha256"] == hashlib.sha256(canonical(plan["rows"]).encode()).hexdigest(), "Batch plan content hash mismatch")
            require(len(plan["rows"]) == BATCH_SIZE and len({x["partner_identity"] for x in plan["rows"]}) == BATCH_SIZE
                    and plan["total_tokens_per_example"] == 500 and plan["target"] == "recorded_ego_actions",
                    "Batch does not have two distinct partners under the paired budget")
            require((run / "checkpoints" / f"step_{row['step']:08d}.msgpack").is_file(), "Committed step checkpoint is missing")
        committed.extend(selected)
        cursor = end
        attempt_refs.append({"started": ref(attempt / "started.json"), "completion": ref(attempt / "completion.json"),
                             "metrics": ref(attempt / "metrics.jsonl"), "committed_start": start["start_step"], "committed_end": end})
    require(cursor == UPDATES and len(committed) == UPDATES, "The four-mode comparison must contain exactly eight updates per mode")
    return {"mode": mode, "actual_updates": UPDATES, "completion": ref(run / "completion.json"),
            "config": ref(run / "config.json"), "initialization": read_json(run / "initialization.json"),
            "terminal": terminal, "attempts": attempt_refs, "batch_plans": [x["batch_plan"] for x in committed],
            "per_update": [{k: x[k] for k in ("step", "ego_ce", "loss", "gradient_norm", "step_seconds")} for x in committed]}


def verify_evaluation(out, mode, training, data):
    receipt = read_json(out / "evaluation.json")
    require(receipt.get("status") == "PASS_EXECUTION" and receipt["mode"] == mode and receipt["seed"] == SEED
            and receipt["task_ids"] == data["task_ids"] and receipt["conditions"] == ["matched", "null"]
            and receipt["episodes_per_task_condition"] == EPISODES and receipt["max_steps"] == 100
            and receipt["test_read"] is False and receipt["optimizer_updates"] == 0
            and receipt["checkpoint_step"] == UPDATES and receipt["checkpoint_is_terminal"] is True,
            "Native online engineering evaluation did not complete the fixed contract")
    require(receipt["checkpoint"]["sha256"] == training["terminal"]["sha256"]
            and receipt["manifest"]["sha256"] == data["files"]["task_manifest"]["sha256"], "Evaluation used another checkpoint or manifest")
    expected_pairs = {(tid, condition) for tid in data["task_ids"] for condition in ("matched", "null")}
    require({(r["task_id"], r["history_condition"]) for r in receipt["task_results"]} == expected_pairs
            and len(receipt["task_results"]) == len(expected_pairs), "Missing/duplicated task-condition result")
    require(len(receipt["completed_episode_files"]) == len(expected_pairs) * EPISODES, "Online episode count mismatch")
    for item in receipt["completed_episode_files"]:
        require(digest(inside(out, item["path"])) == item["sha256"], "Committed episode file hash mismatch")
    results, teammate_hashes = [], {}
    for item in receipt["task_results"]:
        path = inside(out, item["path"])
        require(digest(path) == item["sha256"], "Native result hash mismatch")
        result = read_json(path)
        finite(result)
        condition, task = item["history_condition"], item["task_id"]
        require(result["task_id"] == task and result["mode"] == mode and result["history_condition"] == condition
                and result["seed"] == SEED and result["context_budget"] == 500 and result["max_steps"] == 100
                and result["test_read"] is False and result["development_only"] is True,
                "Result file disagrees with its task/condition contract")
        require([e["episode"] for e in result["episodes"]] == list(range(EPISODES)), "Incomplete episode sequence")
        require(all(e["steps"] == 100 and e["max_observed_tokens"] <= 500 for e in result["episodes"]), "The requested six 100-step episodes were not completed")
        last = result["episodes"][-1]
        require(last["available_support_episodes"] == 2 and last["consumed_support_episodes"] == (2 if condition == "matched" else 0),
                "Episode six did not exercise two available old supports and the requested intervention")
        require(all(eid < last["final_query_first_episode"] for eid in last["final_support_episode_ids"]), "Support reaches into the query")
        require(all(e["consumed_support_episodes"] == 0 for e in result["episodes"]) if condition == "null" else True,
                "Null condition consumed old supports")
        old = teammate_hashes.setdefault(task, result["teammate_params_sha256"])
        require(old == result["teammate_params_sha256"], "Matched/null used different teammate parameters")
        values = [e["return"] for e in result["episodes"]]
        require(math.isclose(sum(values), result["auc"], rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(sum(values) / EPISODES, result["mean_return"], rel_tol=1e-9, abs_tol=1e-9), "Episode return aggregation mismatch")
        results.append({"task_id": task, "condition": condition, "returns": values,
                        "mean_return": result["mean_return"], "last_episode_support_count": last["consumed_support_episodes"]})
    return {"receipt": ref(out / "evaluation.json"), "results": results, "teammate_params_sha256": teammate_hashes,
            "episodes": len(expected_pairs) * EPISODES, "checkpoint_reload_validated_by_native_evaluator": True}


def execute(args):
    require(platform.system() == "Linux", "Execution belongs on the CPU Linux host; local use is limited to static/interface checks")
    require(args.threads >= 1 and hasattr(os, "sched_getaffinity"), "Invalid CPU runtime")
    allowed = sorted(os.sched_getaffinity(0))
    require(len(allowed) >= args.threads, "Insufficient explicitly available affinity CPUs")
    cpus, eval_threads = allowed[:args.threads], min(args.threads, 24)
    taskset = shutil.which("taskset")
    require(taskset, "taskset is required to apply the recorded child CPU slice")
    pipeline, out = args.pipeline_root.resolve(), args.out_dir.resolve()
    require(not out.exists(), "Use a new output directory; this check never retries or overwrites runs")
    require(not out.is_relative_to(ROOT / "benchmarks"), "Output cannot be placed in official benchmark inputs")
    out.mkdir(parents=True)
    sources = {name: digest(ROOT / name) for name in ("native_a/paired_smoke.py", "native_a/train.py", "native_a/sampler.py",
               "native_a/model.py", "native_a/losses.py", "native_a/evaluate.py", "benchmarks/baselines/ad/model.py",
               "runners/history_adapter.py", "eval_icrl.py")}
    receipt = {"schema": "native-paired-smoke/1", "status": "RUNNING", "engineering_only": True,
               "claim": "Execution/restore/history-control check; not evidence of method effectiveness",
               "official_test_read": False, "pipeline_root": str(pipeline), "seed": SEED,
               "planned_training_updates": 32, "updates_per_mode": UPDATES, "batch_size": BATCH_SIZE,
               "history_tokens": {"support": [100, 100], "query": 300, "total": 500},
               "engineering_lambda_p": 0.001, "training_threads": args.threads, "evaluation_threads": eval_threads,
               "training_affinity": cpus, "evaluation_affinity": cpus[:eval_threads],
               "source_sha256": sources, "processes": [], "training": {}, "evaluations": {}, "started_at": now()}
    write_json(out / "receipt.json", receipt)
    def launch(name, argv, thread_count):
        pipeline_health(pipeline)
        require(all(digest(ROOT / name) == sha for name, sha in sources.items()), "Source changed during engineering check")
        log = out / "logs" / (name + ".log")
        log.parent.mkdir(exist_ok=True)
        command = [taskset, "-c", ",".join(map(str, cpus[:thread_count]))] + argv
        before = time.monotonic()
        entry = {"name": name, "command": command, "cwd": str(ROOT), "log_path": str(log),
                 "started_at": now(), "threads": thread_count, "affinity": cpus[:thread_count], "returncode": None}
        receipt["processes"].append(entry)
        write_json(out / "receipt.json", receipt)
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        rc = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pipeline_health(pipeline)
                        print(canonical({"event": "child_running", "name": name, "elapsed_seconds": time.monotonic() - before}), flush=True)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            finally:
                entry.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - before, finished_at=now())
                stream.flush()
                os.fsync(stream.fileno())
                write_json(out / "receipt.json", receipt)
        entry["log_sha256"] = digest(log)
        print(canonical({"event": "child_finished", "name": name, "returncode": rc,
                         "elapsed_seconds": entry["elapsed_seconds"]}), flush=True)
        return rc
    try:
        data = wait_for_dataset(pipeline, out)
        receipt["data"] = data
        write_json(out / "receipt.json", receipt)
        for mode in MODES:
            label = "I_VC" if mode == "I+VC" else mode
            run = out / "training" / label
            command = train_command(data, mode, run, args.threads)
            if mode == "I+VC":
                rc = launch(label + "_pause", command + ["--wall-seconds", "1"], args.threads)
                require(rc != 0, "Expected deliberate wall-budget pause was not observed")
                receipt["resume_check"] = verify_pause(run)
                require(launch(label + "_resume", command + ["--resume"], args.threads) == 0, "I+VC resume failed; no retry allowed")
            else:
                require(launch(label + "_train", command, args.threads) == 0, f"{mode} training failed; no retry allowed")
            training = committed_training(run, mode, data)
            receipt["training"][mode] = training
            if mode != "baseline":
                reference = receipt["training"]["baseline"]
                require(training["initialization"]["common_backbone_sha256"] == reference["initialization"]["common_backbone_sha256"], "Common backbone initialization differs")
                require(training["batch_plans"] == reference["batch_plans"], "Committed batches differ across modes/resume")
            if mode in ("VC", "I+VC"):
                require(training["initialization"]["params_sha256"] == receipt["training"]["none"]["initialization"]["params_sha256"], "Persistent-mode initial parameters differ")
            write_json(out / "receipt.json", receipt)
        receipt["resume_check"]["status"] = "PASS_RESUMED_TO_SAME_EIGHT_UPDATE_BUDGET"
        receipt["actual_training_updates"] = sum(r["actual_updates"] for r in receipt["training"].values())
        require(receipt["actual_training_updates"] == 32, "Unexpected extra real optimizer updates")
        for mode in MODES:
            label = "I_VC" if mode == "I+VC" else mode
            eval_out = out / "evaluation" / label
            command = [sys.executable, "-m", "native_a.evaluate", "--checkpoint", str(out / "training" / label / "terminal.json"),
                       "--manifest", data["paths"]["task_manifest"], "--out-dir", str(eval_out), "--episodes", str(EPISODES),
                       "--seed", str(SEED), "--history-condition", "both", "--threads", str(eval_threads)]
            require(launch(label + "_evaluation", command, eval_threads) == 0, f"{mode} native evaluation failed; no retry allowed")
            evaluation = verify_evaluation(eval_out, mode, receipt["training"][mode], data)
            if mode != "baseline":
                require(evaluation["teammate_params_sha256"] == receipt["evaluations"]["baseline"]["teammate_params_sha256"], "Modes did not use the same frozen RL partners")
            receipt["evaluations"][mode] = evaluation
            write_json(out / "receipt.json", receipt)
        pipeline_health(pipeline)
        require(all(digest(ROOT / name) == sha for name, sha in sources.items()), "Source changed during engineering check")
        require(all(digest(f["path"]) == f["sha256"] for f in data["files"].values()), "Engineering dataset changed")
        finite(receipt)
        receipt.update(status="PASS_ENGINEERING", finished_at=now(), matched_batch_plans=True,
                       common_backbone_equal=True, persistent_initialization_equal=True,
                       real_optimizer_updates=32, checkpoint_reload_all_modes=True,
                       total_native_eval_episodes=sum(x["episodes"] for x in receipt["evaluations"].values()))
    except BaseException as error:
        receipt.update(status="STOPPED", error=f"{type(error).__name__}: {error}", finished_at=now(),
                       automatic_retry_performed=False)
        write_json(out / "receipt.json", receipt)
        raise
    write_json(out / "receipt.json", receipt)
    write_json(out / "completion.json", {"status": "PASS_ENGINEERING", "real_optimizer_updates": 32,
               "resume_completed": True, "all_four_native_evaluations_passed": True,
               "official_test_read": False, "engineering_only": True, "receipt": ref(out / "receipt.json")})
    print(canonical({"status": receipt["status"], "receipt": str(out / "receipt.json"),
                     "real_optimizer_updates": 32, "scientific_effect_claim": False}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
