"""Durable CPU driver for the *unmodified* native chunked IPPO implementation.

Native chunks always contain exactly one PPO update.  ``updates-per-chunk``
only groups these calls between durable saves; neither PPO nor its schedule is
reimplemented here.  A fresh run with ``--max-new-updates 1`` followed by the
same command with ``--resume`` (and without that limit) continues the same run.
``completion.json`` is published only after all planned updates and the five
official checkpoints have been saved.  A successful partial save exits zero.

This module imports JAX only after argument parsing and CPU configuration, so
``--help`` and the pure planning helpers do not require the training packages.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import tempfile
import time
import uuid


SCHEMA = "native-partner-run/1"
REPO = Path(__file__).resolve().parents[1]
STATE_KEYS = {
    "train_state", "env_state", "obs", "done", "hstate", "rng",
    "cumulative_shaped", "returned_shaped", "update_steps",
    "checkpoint_array", "ckpt_idx",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"Refusing symlink: {path}")
    require(replace or not path.exists(), f"Refusing to overwrite immutable file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value, *, replace=False):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
                 replace=replace)


def positive_integer(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value() or number <= 0:
            raise ValueError
        return int(number)
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError("Expected a positive integer (scientific notation such as 3e7 is allowed)")


def make_budget(total_timesteps, rollout_length=256, num_envs=256, num_checkpoints=5):
    """Plan exactly the native cadence, refusing its underfilled/overflow cases."""
    require(isinstance(total_timesteps, int) and total_timesteps > 0, "Timesteps must be a positive integer")
    per_update = rollout_length * num_envs
    updates = total_timesteps // per_update
    interval = updates // max(1, num_checkpoints - 1)
    require(interval > 0, "Budget is too short for the original five-checkpoint cadence")
    indices = [u for u in range(1, updates + 1) if (u - 1) % interval == 0 or u == updates]
    require(len(indices) == num_checkpoints,
            f"Unmodified native checkpoint cadence would store {len(indices)} checkpoints into "
            f"{num_checkpoints} slots for {updates} updates. Use a valid budget; 30000000 gives "
            "457 updates and exactly five checkpoints. Budget is never silently adjusted.")
    return {
        "requested_timesteps": total_timesteps, "planned_updates": updates,
        "timesteps_per_update": per_update, "actual_budget_timesteps": updates * per_update,
        "discarded_incomplete_update_timesteps": total_timesteps - updates * per_update,
        "native_updates_per_call": 1, "native_num_chunks": updates,
        "checkpoint_update_indices": indices, "num_checkpoints": num_checkpoints,
        "timestep_unit": "environment transitions (not multiplied by the two actors)",
    }


def chunk_boundaries(start, total, updates_per_chunk, max_new_updates=0):
    require(0 <= start <= total and updates_per_chunk > 0 and max_new_updates >= 0, "Invalid update boundary")
    end = min(total, start + max_new_updates) if max_new_updates else total
    while start < end:
        stop = min(start + updates_per_chunk, end)
        yield start, stop
        start = stop


def source_hashes():
    paths = {Path(__file__).resolve(), REPO / "teammate_generation/train_ippo_overcooked_v2.py"}
    for subtree in ("agents", "common", "envs", "marl"):
        paths.update((REPO / subtree).rglob("*.py"))
    return {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths)}


def configure_cpu(threads):
    require(threads > 0, "Threads must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # Match the measured CPU run: affinity is independent from BLAS/OpenMP.
    for key, value in {"OMP_NUM_THREADS": 8, "OPENBLAS_NUM_THREADS": 1,
                       "MKL_NUM_THREADS": 8, "NUMEXPR_NUM_THREADS": 8}.items():
        os.environ[key] = str(value)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        require(len(allowed) >= threads, f"Only {len(allowed)} affinity CPUs available, but {threads} requested")
        # External taskset chooses the exact slice; this only narrows an overly broad slice.
        os.sched_setaffinity(0, set(allowed[:threads]))


def runtime_signature(threads, jax):
    packages = {}
    for name in ("jax", "jaxlib", "flax", "optax", "numpy", "orbax-checkpoint", "distrax", "chex", "jaxmarl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "threads": threads, "python": platform.python_version(), "machine": platform.machine(),
        "packages": packages, "backend": jax.default_backend(),
        "device_kinds": [d.device_kind for d in jax.devices()],
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "XLA_FLAGS", "JAX_ENABLE_X64")},
    }


def cpu_receipt():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "pid": os.getpid(), "hostname": platform.node(), "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_seconds_self": usage.ru_utime + usage.ru_stime,
        "max_rss_native_units": usage.ru_maxrss, "max_rss_unit": "KiB on Linux; bytes on macOS",
    }


@contextmanager
def exclusive_run(out):
    import fcntl
    with (out / ".runner.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another driver already holds {out}")
        yield


def local_file(out, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Receipt path must stay inside this run")
    target = out / path
    require(target.resolve().is_relative_to(out.resolve()), "Receipt path escaped the run directory")
    require(target.is_file() and not target.is_symlink(), f"Missing regular file: {target}")
    return target


def file_ref(out, path):
    path = Path(path)
    return {"path": str(path.relative_to(out)), "size": path.stat().st_size, "sha256": sha256_file(path)}


def verify_ref(out, ref):
    path = local_file(out, ref["path"])
    require(path.stat().st_size == ref["size"] and sha256_file(path) == ref["sha256"], f"File hash/size mismatch: {path}")
    return path


def validate_state(state, update, budget, config, np, jax):
    require(0 <= update <= budget["planned_updates"], "Native state exceeds the registered update budget")
    require(set(state) == STATE_KEYS, "Native state schema changed: refusing partial-state checkpoint")
    require(np.asarray(state["update_steps"]).tolist() == [update], "Native update counter mismatch")
    expected_opt = update * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    require(np.asarray(state["train_state"].step).tolist() == [expected_opt], "Optimizer step counter mismatch")
    expected_ckpts = sum(u <= update for u in budget["checkpoint_update_indices"])
    require(np.asarray(state["ckpt_idx"]).tolist() == [expected_ckpts], "Native checkpoint counter mismatch")
    for group in (state["train_state"].params, state["train_state"].opt_state, state["hstate"]):
        for value in jax.tree_util.tree_leaves(group):
            arr = np.asarray(value)
            require(not np.issubdtype(arr.dtype, np.inexact) or bool(np.isfinite(arr).all()),
                    "NUMERICAL_FAILURE: non-finite model, optimizer, or recurrent state")


def encode_native_state(tree, np, jax):
    """All dynamic leaves; no pickle or assumption that Chex is Flax-serializable.

    The native environment contains Chex dataclasses, which are valid JAX
    pytrees but are not necessarily registered with flax.serialization.  A
    freshly constructed, hash-bound native template supplies the treedef and
    static optimizer/apply functions on restore; every dynamic leaf is saved.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for path, value in leaves:
        array = np.asarray(value)
        require(array.dtype.kind != "O", "Object leaf in native training state")
        records.append({"path": jax.tree_util.keystr(path), "dtype": array.dtype.str,
                        "shape": list(array.shape), "value": array})
    return {"state_keys": sorted(tree), "leaves": records}


def decode_native_state(template, encoded, np, jax):
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    require(encoded["state_keys"] == sorted(template), "Native template state fields changed")
    require(len(paths) == len(encoded["leaves"]), "Native template leaf count changed")
    restored = []
    for (path, example), leaf in zip(paths, encoded["leaves"]):
        array = np.asarray(leaf["value"])
        target = np.asarray(example)
        require(leaf["path"] == jax.tree_util.keystr(path), "Native template leaf path changed")
        require(leaf["dtype"] == target.dtype.str == array.dtype.str
                and leaf["shape"] == list(target.shape) == list(array.shape),
                f"Native template leaf shape/dtype changed: {leaf['path']}")
        restored.append(array)
    return jax.tree_util.tree_unflatten(treedef, restored)


def tree_tensor_sha(tree, np, jax):
    """Value hash independent of Python container identity / Flax static fields."""
    digest = hashlib.sha256()
    encoded = encode_native_state(tree, np, jax)
    digest.update(canonical(encoded["state_keys"]))
    for leaf in encoded["leaves"]:
        digest.update(canonical([leaf["path"], leaf["dtype"], leaf["shape"]]))
        digest.update(leaf["value"].tobytes(order="C"))
    return digest.hexdigest()


def commit_state(out, state, ledger, binding, serialization, np, jax, *, resume_from=None):
    update = int(np.asarray(state["update_steps"])[0])
    name = f"states/update_{update:06d}_{uuid.uuid4().hex}.msgpack"
    payload = {
        "schema": SCHEMA, "binding_sha256": hashlib.sha256(canonical(binding)).hexdigest(),
        "update": update, "state": encode_native_state(state, np, jax), "metric_ledger": ledger,
        "state_tensor_sha256": tree_tensor_sha(state, np, jax),
    }
    path = out / name
    atomic_bytes(path, serialization.msgpack_serialize(payload))
    pointer = {"schema": SCHEMA, "actual_updates": update, "state": file_ref(out, path),
               "state_tensor_sha256": payload["state_tensor_sha256"],
               "binding_sha256": payload["binding_sha256"], "committed_at": utc_now(),
               "resumed_from_state_sha256": resume_from}
    # The state and all its referenced metric chunks are immutable and durable
    # before the pointer is replaced.  An unreferenced interrupted write is not resumed.
    atomic_json(out / "latest.json", pointer, replace=True)
    return pointer


def restore_state(out, template, binding, serialization, np, jax):
    pointer = json.loads((out / "latest.json").read_text())
    payload = serialization.msgpack_restore(verify_ref(out, pointer["state"]).read_bytes())
    bind_sha = hashlib.sha256(canonical(binding)).hexdigest()
    require(pointer["schema"] == payload["schema"] == SCHEMA, "Unknown state schema")
    require(pointer["binding_sha256"] == payload["binding_sha256"] == bind_sha, "State binding changed")
    require(pointer["actual_updates"] == payload["update"], "State pointer counter mismatch")
    state = decode_native_state(template, payload["state"], np, jax)
    require(tree_tensor_sha(state, np, jax) == payload["state_tensor_sha256"] == pointer["state_tensor_sha256"],
            "Full state failed restoration value-hash check")
    previous = 0
    for chunk in payload["metric_ledger"]:
        require(chunk["start_update"] == previous and chunk["end_update"] > previous, "Non-contiguous metric ledger")
        verify_ref(out, chunk["metrics"])
        previous = chunk["end_update"]
    require(previous == payload["update"], "Metrics do not cover all committed training updates")
    return jax.device_put(state), payload["metric_ledger"], pointer


def summarize_metrics(metrics, np):
    mask = np.asarray(metrics["returned_episode"]) > 0 if "returned_episode" in metrics else None
    result = {"completed_actor_episode_events": int(mask.sum()) if mask is not None else None,
              "return_unit": "native actor-episode return; each environment episode normally has two actor events"}
    for name in ("returned_episode_returns", "returned_episode_shaped_returns"):
        values = np.asarray(metrics[name])[mask] if name in metrics and mask is not None else np.asarray([])
        require(bool(np.isfinite(values).all()), f"NUMERICAL_FAILURE: non-finite {name}")
        result[name] = {"mean": float(values.mean()) if values.size else None,
                        "std": float(values.std()) if values.size else None,
                        "count": int(values.size)}
    return result


def publish_native(out, state, ledger, config, pointer, serialization, np, jax,
                   compute_checkpoint_returns, save_separated_checkpoints):
    target = out / "ippo_train_run"
    if target.exists():
        # Recovery from a crash between the atomic directory publish and completion.json.
        receipt = json.loads((target / "native_publish.json").read_text())
        require(receipt["terminal_state_sha256"] == pointer["state"]["sha256"], "Existing native export has a different terminal state")
        for item in receipt["files"]:
            verify_ref(out, item)
        return receipt
    chunks = []
    for record in ledger:
        chunks.append(serialization.msgpack_restore(gzip.decompress(verify_ref(out, record["metrics"]).read_bytes())))
    metrics = jax.tree.map(lambda *arrays: np.concatenate(arrays, axis=1), *chunks)
    require(all(np.asarray(x).shape[1] == config["NUM_UPDATES"] for x in jax.tree_util.tree_leaves(metrics)),
            "Final metric arrays do not contain every planned update")
    returns = compute_checkpoint_returns(metrics, config)
    staging = out / (".native-publish-" + uuid.uuid4().hex)
    staging.mkdir()
    native = Path(save_separated_checkpoints(
        {"checkpoints": state["checkpoint_array"], "metrics": metrics}, config, str(staging),
        savename="ippo_train_run", pop_size_key=None, checkpoint_returns=returns))
    require(all((native / "pi_0" / f"ckpt_{i}").is_dir() for i in range(config["NUM_CHECKPOINTS"])),
            "Official saver did not produce all five checkpoints")
    files = []
    for path in sorted(native.rglob("*")):
        require(not path.is_symlink(), "Symlink in native checkpoint output")
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files.append({"path": str(Path("ippo_train_run") / path.relative_to(native)),
                          "size": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {"schema": SCHEMA, "actual_updates": config["NUM_UPDATES"],
               "terminal_state_sha256": pointer["state"]["sha256"],
               "num_checkpoints": config["NUM_CHECKPOINTS"], "files": files, "published_at": utc_now()}
    atomic_json(native / "native_publish.json", receipt)
    os.rename(native, target)
    fsync_dir(out)
    staging.rmdir()
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=positive_integer, default=30000000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layout", default="grounded_coord_simple")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--updates-per-chunk", type=positive_integer, default=1,
                   help="Number of unchanged one-update native calls between durable saves")
    p.add_argument("--max-new-updates", type=int, default=0,
                   help="Successful partial checkpoint boundary; 0 means continue to the full original budget")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=positive_integer, default=64)
    return p


def run(args):
    require(args.max_new_updates >= 0, "max-new-updates must be nonnegative")
    require(0 <= args.seed < 2 ** 32, "Seed must be a uint32")
    budget = make_budget(args.total_timesteps)
    configure_cpu(args.threads)
    import jax
    import numpy as np
    from flax import serialization
    from envs import make_env
    from envs.log_wrapper import LogWrapper
    from marl.ippo import make_train_chunked
    from teammate_generation.train_ippo_overcooked_v2 import get_default_config, compute_checkpoint_returns
    from common.save_load_utils import save_separated_checkpoints

    require(jax.default_backend() == "cpu" and all(d.platform == "cpu" for d in jax.devices()), "CPU-only run required")
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
               "runtime_signature": runtime_signature(args.threads, jax),
               "resume_rule": "full native state; optimizer count and RNG/env/GRU carry restored; unchanged total schedule"}
    out = args.out_dir.expanduser().resolve()
    if args.resume:
        require(out.is_dir(), "Resume directory is missing")
    else:
        require(not out.exists() or not any(out.iterdir()), "New run requires a new or empty directory; use --resume")
        out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out):
        if args.resume:
            require(json.loads((out / "binding.json").read_text()) == binding,
                    "Resume refused: config, budget, source hash, package version, or CPU runtime settings changed")
        else:
            atomic_json(out / "binding.json", binding)
        attempt = out / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True)
        started = time.monotonic()
        initial_cpu = cpu_receipt()["cpu_seconds_self"]
        atomic_json(attempt / "started.json", {"status": "RUNNING", "at": utc_now(), "cpu": cpu_receipt(),
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
                host_state = template
                state = jax.device_put(template)
                validate_state(host_state, 0, budget, config, np, jax)
                pointer = commit_state(out, host_state, [], binding, serialization, np, jax)
                # Verify actual serializer / full state reconstruction before any PPO update.
                state, ledger, pointer = restore_state(out, template, binding, serialization, np, jax)
                host_state = jax.device_get(state)
            del template
            validate_state(host_state, current, budget, config, np, jax)
            start_update = current
            atomic_json(attempt / "restoration.json", {"resume": args.resume, "actual_updates": current,
                        "restored_state_sha256": restored_sha, "state_tensor_sha256": pointer["state_tensor_sha256"],
                        "initialization_seconds_synchronized": initialization_seconds,
                        "optimizer_steps": int(np.asarray(host_state["train_state"].step)[0]),
                        "all_native_state_fields": sorted(STATE_KEYS), "full_state_value_hash_verified": True})
            print(json.dumps({"event": "ready", "actual_updates": current, "budget": budget,
                              "resume": args.resume, "cpu": cpu_receipt()}), flush=True)
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
                       "cpu": cpu_receipt()}
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
                       "stopping_signal": stop["signal"], "finished_at": utc_now(), "cpu": cpu_receipt(),
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
                       "finished_at": utc_now(), "cpu": cpu_receipt()}
            atomic_json(attempt / "failure.json", failure, replace=True)
            atomic_json(out / "progress.json", failure, replace=True)
            print(json.dumps(failure, sort_keys=True), flush=True)
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main():
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
