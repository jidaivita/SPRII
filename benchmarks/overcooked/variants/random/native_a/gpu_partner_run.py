"""Resume a complete native CPU IPPO checkpoint on one CUDA GPU.

Only the execution backend changes. The native PPO implementation, complete
optimizer/environment state and original total update schedule are preserved.
The frozen CPU driver and its strict CPU resume guard are never modified.
"""
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

from native_a.partner_run import (
    SCHEMA, STATE_KEYS, atomic_bytes, atomic_json, canonical, chunk_boundaries,
    commit_state, cpu_receipt, exclusive_run, file_ref, make_budget,
    parser as cpu_parser, publish_native, require, restore_state,
    runtime_signature, sha256_file, source_hashes, summarize_metrics,
    tree_tensor_sha, utc_now, validate_state, verify_ref,
)


def configure_gpu(threads):
    require(threads > 0, "Threads must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    require(visible not in ("", "-1") and len(visible.split(",")) == 1,
            "Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU before starting")
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for key, value in {"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": 1,
                       "MKL_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads}.items():
        os.environ[key] = str(value)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        require(len(allowed) >= threads, "Requested host threads exceed available CPU affinity")
        os.sched_setaffinity(0, set(allowed[:threads]))


def gpu_signature(threads, jax):
    result = runtime_signature(threads, jax)
    plugins = {name: importlib.metadata.version(name) for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt")}
    require(set(plugins.values()) == {"0.5.3"}, "CUDA plugins must match JAX 0.5.3")
    result.update(cuda_plugins=plugins,
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                  preallocate=os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
                  x64_enabled=bool(jax.config.x64_enabled),
                  default_prng_impl=str(jax.config.jax_default_prng_impl),
                  matmul_precision=str(jax.config.jax_default_matmul_precision))
    return result


def inspect_cpu_snapshot(source, binding, serialization, jax):
    require(source.is_dir(), "CPU snapshot directory is missing")
    # Pointers are read once. Copying their exact hashes below rejects concurrent
    # mutation of a live CPU directory; supply an immutable committed snapshot.
    refs = [file_ref(source, source / name) for name in ("binding.json", "latest.json")]
    old = json.loads(verify_ref(source, refs[0]).read_text())
    pointer = json.loads(verify_ref(source, refs[1]).read_text())
    require(old["schema"] == SCHEMA and old["config"] == binding["config"]
            and old["budget"] == binding["budget"], "CPU native configuration or total schedule changed")
    require(old["source_sha256"] == binding["source_sha256"], "Frozen CPU/native source hashes changed")
    prior, current = old["runtime_signature"], binding["runtime_signature"]
    require(prior["backend"] == "cpu" and current["backend"] == "gpu", "Migration must be CPU to GPU")
    require(prior["packages"] == current["packages"], "Core package versions differ from the CPU run")
    require(current["packages"]["jax"] == current["packages"]["jaxlib"] == "0.5.3",
            "This migration requires the pinned JAX and jaxlib 0.5.3")
    require(prior["python"].split(".")[:2] == current["python"].split(".")[:2]
            and prior["machine"] == platform.machine(), "Python major/minor or architecture changed")
    old_x64 = str(prior["thread_environment"].get("JAX_ENABLE_X64") or "false").lower() in ("1", "true", "yes", "on")
    require(old_x64 == bool(jax.config.x64_enabled), "Effective JAX x64 dtype policy changed")
    old_hash = hashlib.sha256(canonical(old)).hexdigest()
    require(pointer["binding_sha256"] == old_hash, "CPU pointer binding is corrupt")
    payload = serialization.msgpack_restore(verify_ref(source, pointer["state"]).read_bytes())
    require(payload["schema"] == SCHEMA and payload["binding_sha256"] == old_hash
            and payload["update"] == pointer["actual_updates"]
            and payload["state_tensor_sha256"] == pointer["state_tensor_sha256"],
            "CPU state, pointer and binding do not agree")
    refs.append(pointer["state"])
    previous = 0
    for item in payload["metric_ledger"]:
        require(item["start_update"] == previous and item["end_update"] > previous,
                "CPU metric ledger is not contiguous")
        previous = item["end_update"]
        verify_ref(source, item["metrics"])
        refs.append(item["metrics"])
    require(previous == payload["update"], "CPU metrics omit committed updates")
    require(len({r["path"] for r in refs}) == len(refs), "Duplicate CPU snapshot file references")
    lineage = {"binding_sha256": old_hash, "state_sha256": pointer["state"]["sha256"],
               "state_tensor_sha256": pointer["state_tensor_sha256"], "actual_updates": payload["update"],
               "immutable_files": refs}
    return old, refs, lineage


def copy_verified_snapshot(source, destination, refs):
    for ref in refs:
        data = verify_ref(source, ref).read_bytes()
        require(len(data) == ref["size"] and hashlib.sha256(data).hexdigest() == ref["sha256"],
                "CPU snapshot changed during copy")
        atomic_bytes(destination / ref["path"], data)
        verify_ref(destination, ref)
    atomic_json(destination / "snapshot_receipt.json", {"copied_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source), "immutable_files": refs})


# Same native one-update loop and durable receipts as partner_run.run, with the
# initial state imported from CPU and a separately bound GPU runtime.
def run(args):
    require(args.max_new_updates >= 0, "max-new-updates must be nonnegative")
    require(0 <= args.seed < 2 ** 32, "Seed must be a uint32")
    budget = make_budget(args.total_timesteps)
    configure_gpu(args.threads)
    import jax
    import numpy as np
    from flax import serialization
    from envs import make_env
    from envs.log_wrapper import LogWrapper
    from marl.ippo import make_train_chunked
    from teammate_generation.train_ippo_overcooked_v2 import get_default_config, compute_checkpoint_returns
    from common.save_load_utils import save_separated_checkpoints

    require(jax.default_backend() == "gpu" and len(jax.devices()) == 1
            and jax.devices()[0].platform == "gpu", "Exactly one CUDA GPU is required; CPU fallback is forbidden")
    config = get_default_config(args.layout)
    config.update(TOTAL_TIMESTEPS=int(args.total_timesteps), TRAIN_SEED=args.seed,
                  NUM_CHUNKS=budget["planned_updates"])
    require(config["NUM_ENVS"] == config["ROLLOUT_LENGTH"] == 256 and config["UPDATE_EPOCHS"] == 4
            and config["NUM_MINIBATCHES"] == 64 and config["FC_DIM_SIZE"] == config["GRU_HIDDEN_DIM"] == 128
            and config["NUM_CHECKPOINTS"] == 5 and config["NUM_SEEDS"] == 1, "Native defaults changed")
    env = LogWrapper(make_env(config["ENV_NAME"], config["ENV_KWARGS"]))
    init_fn, native_chunk_fn, config = make_train_chunked(config, env)
    require(type(config["NUM_UPDATES"]) is int and config["NUM_UPDATES"] == budget["planned_updates"],
            "Official update count is not the planned integer")
    require(config["NUM_UPDATES"] // config["NUM_CHUNKS"] == 1, "Native calls must each execute exactly one update")
    binding = {"schema": SCHEMA, "config": config, "budget": budget, "source_sha256": source_hashes(),
               "runtime_signature": gpu_signature(args.threads, jax),
               "gpu_adapter_sha256": sha256_file(Path(__file__).resolve()),
               "resume_rule": "full native state; optimizer count and RNG/env/GRU carry restored; unchanged total schedule"}
    out = args.out_dir.expanduser().resolve()
    require(bool(args.resume) != bool(args.import_cpu_run),
            "Use --import-cpu-run for first migration or --resume for an existing GPU run")
    source = out / "imported_cpu" if args.resume else args.import_cpu_run.expanduser().resolve()
    if not args.resume:
        require(not source.is_relative_to(out) and not out.is_relative_to(source),
                "CPU snapshot and new GPU run must be separate directory trees")
    old_binding, files, lineage = inspect_cpu_snapshot(source, binding, serialization, jax)
    binding["cpu_import"] = lineage
    if args.resume:
        require(out.is_dir(), "Resume directory is missing")
    else:
        require(not out.exists() or not any(out.iterdir()), "New run requires a new or empty directory; use --resume")
        out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out):
        if args.resume:
            require(json.loads((out / "binding.json").read_text()) == binding,
                    "Resume refused: config, source, CPU lineage, package version, or GPU runtime changed")
        else:
            atomic_json(out / "binding.json", binding)
            copy_verified_snapshot(source, out / "imported_cpu", files)
        attempt = out / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True)
        started = time.monotonic()
        initial_cpu = cpu_receipt()["cpu_seconds_self"]
        atomic_json(attempt / "started.json", {"status": "RUNNING", "at": utc_now(), "cpu": cpu_receipt(), "backend": jax.default_backend(),
                    "argv": sys.argv, "resume": args.resume, "updates_per_durable_chunk": args.updates_per_chunk,
                    "max_new_updates": args.max_new_updates, "binding_sha256": hashlib.sha256(canonical(binding)).hexdigest()})
        stop = {"signal": None}
        old_handlers = {}
        def request_stop(signum, _frame):
            stop["signal"] = signum
            print(json.dumps({"event": "stop_requested", "signal": signum,
                              "action": "finish the current native update, synchronize, checkpoint, then stop"}), flush=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, request_stop)
        pointer = None
        restored_sha = None
        current = 0
        ledger = []
        try:
            initialization_started = time.monotonic()
            init_jit = jax.jit(jax.vmap(init_fn))
            native_chunk_jit = jax.jit(jax.vmap(native_chunk_fn))
            template = jax.device_get(init_jit(jax.random.split(jax.random.PRNGKey(args.seed), 1)))
            initialization_seconds = time.monotonic() - initialization_started
            if args.resume:
                state, ledger, pointer = restore_state(out, template, binding, serialization, np, jax)
                restored_sha = pointer["state"]["sha256"]
                current = pointer["actual_updates"]
                host_state = jax.device_get(state)
            else:
                # The template supplies only static functions/tree structure. Every dynamic
                # value, including Adam state, PRNG and environment/GRU carry, comes from CPU.
                state, ledger, cpu_pointer = restore_state(out / "imported_cpu", template, old_binding,
                                                          serialization, np, jax)
                restored_sha = cpu_pointer["state"]["sha256"]
                current = cpu_pointer["actual_updates"]
                host_state = jax.device_get(state)
                require(tree_tensor_sha(host_state, np, jax) == cpu_pointer["state_tensor_sha256"],
                        "Device transfer changed the imported complete state")
                validate_state(host_state, current, budget, config, np, jax)
                for entry in ledger:
                    ref = entry["metrics"]
                    atomic_bytes(out / ref["path"], verify_ref(out / "imported_cpu", ref).read_bytes())
                    verify_ref(out, ref)
                pointer = commit_state(out, host_state, ledger, binding, serialization, np, jax,
                                       resume_from=restored_sha)
                state, ledger, pointer = restore_state(out, template, binding, serialization, np, jax)
                host_state = jax.device_get(state)
                require(tree_tensor_sha(host_state, np, jax) == cpu_pointer["state_tensor_sha256"],
                        "GPU checkpoint round trip changed the complete CPU state")
                atomic_json(out / "migration.json", {
                    "status": "PASS", "at": utc_now(), "cpu_source": lineage,
                    "gpu_state": pointer, "actual_updates": current,
                    "optimizer_steps": int(np.asarray(host_state["train_state"].step)[0]),
                    "full_state_value_hash_verified_on_gpu": True,
                    "old_runtime": old_binding["runtime_signature"], "new_runtime": binding["runtime_signature"],
                    "runtime_change": "CPU to CUDA; subsequent floating-point trajectories need not be bitwise identical",
                    "native_config_and_full_schedule_unchanged": True,
                    "inherited_metric_ledger_records": len(ledger)})
            del template
            validate_state(host_state, current, budget, config, np, jax)
            start_update = current
            atomic_json(attempt / "restoration.json", {"resume": args.resume, "actual_updates": current,
                        "restored_state_sha256": restored_sha, "state_tensor_sha256": pointer["state_tensor_sha256"],
                        "initialization_seconds_synchronized": initialization_seconds,
                        "optimizer_steps": int(np.asarray(host_state["train_state"].step)[0]),
                        "all_native_state_fields": sorted(STATE_KEYS), "full_state_value_hash_verified": True})
            print(json.dumps({"event": "ready", "actual_updates": current, "budget": budget,
                              "resume": args.resume, "cpu": cpu_receipt(), "backend": jax.default_backend()}), flush=True)
            for begin, planned_end in chunk_boundaries(current, config["NUM_UPDATES"], args.updates_per_chunk, args.max_new_updates):
                if stop["signal"] is not None:
                    break
                chunk_started = time.monotonic()
                chunk_started_at = utc_now()
                update_metrics = []
                for _ in range(begin, planned_end):
                    state, metrics = native_chunk_jit(state)
                    update_metrics.append(metrics)
                    current += 1
                    if stop["signal"] is not None:
                        break
                # This is the timing barrier. No dispatch-only timing is reported as training time.
                host_state, host_parts = jax.device_get((state, update_metrics))
                chunk_seconds = time.monotonic() - chunk_started
                synchronized_at = utc_now()
                validate_state(host_state, current, budget, config, np, jax)
                metrics = jax.tree.map(lambda *parts: np.concatenate(parts, axis=1), *host_parts)
                require(np.asarray(metrics["update_steps"]).tolist() == [list(range(begin, current))],
                        "Native metrics update indices are not the committed contiguous interval")
                summary = summarize_metrics(metrics, np)
                metric_path = out / "metrics" / f"updates_{begin + 1:06d}_{current:06d}_{uuid.uuid4().hex}.msgpack.gz"
                persistence_started = time.monotonic()
                atomic_bytes(metric_path, gzip.compress(serialization.msgpack_serialize(metrics), compresslevel=1, mtime=0))
                record = {"start_update": begin, "end_update": current, "metrics": file_ref(out, metric_path),
                          "started_at": chunk_started_at, "synchronized_at": synchronized_at,
                          "synchronized_training_seconds": chunk_seconds,
                          "new_environment_transitions": (current - begin) * budget["timesteps_per_update"],
                          "returns": summary, "attempt": str(attempt.relative_to(out))}
                ledger = ledger + [record]
                pointer = commit_state(out, host_state, ledger, binding, serialization, np, jax, resume_from=restored_sha)
                persistence_seconds = time.monotonic() - persistence_started
                row = {"event": "chunk_committed", "status": "RUNNING", "actual_updates": current,
                       "planned_updates": config["NUM_UPDATES"], "actual_timesteps": current * budget["timesteps_per_update"],
                       "chunk": record, "persistence_seconds": persistence_seconds, "state": pointer,
                       "cumulative_synchronized_training_seconds": sum(x["synchronized_training_seconds"] for x in ledger),
                       "cpu": cpu_receipt(), "backend": jax.default_backend()}
                atomic_json(attempt / f"chunk_{current:06d}.json", row)
                atomic_json(out / "progress.json", row, replace=True)
                print(json.dumps(row, sort_keys=True), flush=True)
                del update_metrics, host_parts, metrics
            complete = current == config["NUM_UPDATES"]
            finalization_started = time.monotonic()
            native_receipt = None
            if complete:
                native_receipt = publish_native(out, host_state, ledger, config, pointer, serialization, np, jax,
                                               compute_checkpoint_returns, save_separated_checkpoints)
                atomic_json(out / "terminal.json", pointer, replace=True)
            receipt = {"schema": SCHEMA, "status": "PASS" if complete else "PARTIAL",
                       "complete": complete, "start_update_this_attempt": start_update, "actual_updates": current,
                       "planned_updates": config["NUM_UPDATES"], "actual_timesteps": current * budget["timesteps_per_update"],
                       "requested_timesteps": args.total_timesteps, "state": pointer,
                       "native_run": str(out / "ippo_train_run") if complete else None,
                       "native_publish_sha256": sha256_file(out / "ippo_train_run/native_publish.json") if complete else None,
                       "synchronized_training_seconds_committed": sum(x["synchronized_training_seconds"] for x in ledger),
                       "finalization_seconds": time.monotonic() - finalization_started,
                       "attempt_wall_seconds": time.monotonic() - started,
                       "attempt_cpu_seconds": cpu_receipt()["cpu_seconds_self"] - initial_cpu,
                       "stopping_signal": stop["signal"], "finished_at": utc_now(), "cpu": cpu_receipt(), "backend": jax.default_backend(),
                       "partial_exit_zero_means": "successful durable boundary only; full budget not completed",
                       "binding_sha256": hashlib.sha256(canonical(binding)).hexdigest()}
            atomic_json(attempt / "completion.json", receipt)
            atomic_json(out / "progress.json", receipt, replace=True)
            if complete:
                if (out / "completion.json").exists():
                    old = json.loads((out / "completion.json").read_text())
                    require(old["state"]["state"]["sha256"] == pointer["state"]["sha256"] and old["complete"],
                            "Existing completion is inconsistent with terminal state")
                else:
                    atomic_json(out / "completion.json", receipt)
            print(json.dumps(receipt, sort_keys=True), flush=True)
            return 128 + stop["signal"] if stop["signal"] is not None and not complete else 0
        except BaseException as error:
            failure = {"schema": SCHEMA, "status": "NUMERICAL_FAILURE" if "NUMERICAL_FAILURE" in str(error) else "TECHNICAL_FAILURE",
                       "error": f"{type(error).__name__}: {error}", "complete": False,
                       "latest_committed_updates": pointer["actual_updates"] if pointer else None,
                       "last_dispatched_updates": current, "attempt_wall_seconds": time.monotonic() - started,
                       "finished_at": utc_now(), "cpu": cpu_receipt(), "backend": jax.default_backend()}
            atomic_json(attempt / "failure.json", failure, replace=True)
            atomic_json(out / "progress.json", failure, replace=True)
            print(json.dumps(failure, sort_keys=True), flush=True)
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main():
    p = cpu_parser()
    p.description = __doc__
    p.set_defaults(threads=8)
    p.add_argument("--import-cpu-run", type=Path,
                   help="Immutable full CPU snapshot; required for the first GPU run")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
