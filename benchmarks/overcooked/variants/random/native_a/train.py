"""Paired-history AD / persistent AD training; upstream AD stays unchanged.

Modes baseline, none, VC and I+VC consume identical sampled token windows and
query labels. `baseline` is the unmodified AD architecture on concatenated
windows; it is distinct from the upstream original-sampler reference run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time

# Select the CLI backend before importing JAX. Importing this module as a
# checkpoint loader leaves the caller's device selection untouched, including
# the CPU-only native evaluator. GPU runs must be assigned one external card.
if __name__ == "__main__":
    _backend_parser = argparse.ArgumentParser(add_help=False)
    _backend_parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    _early_backend = _backend_parser.parse_known_args()[0].backend
    if _early_backend == "gpu":
        _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(_visible) != 1 or not _visible[0].strip() or _visible[0].strip() == "-1":
            raise ValueError("GPU training requires externally assigned CUDA_VISIBLE_DEVICES for exactly one card")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = _early_backend
    os.environ["JAX_PLATFORM_NAME"] = _early_backend
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# BLAS reads these during import, so configure before NumPy/JAX are imported.
_early_threads = 4
for _index, _argument in enumerate(sys.argv):
    if _argument == "--threads" and _index + 1 < len(sys.argv):
        _early_threads = int(sys.argv[_index + 1])
    elif _argument.startswith("--threads="):
        _early_threads = int(_argument.split("=", 1)[1])
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = str(_early_threads)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.core import freeze, unfreeze
from flax.training import train_state

from benchmarks.baselines.ad.model import ADConfig, ADModel
from native_a.model import PersistentADConfig, PersistentADModel, apply_batch, batch_model_kwargs, validate_model_batch
from native_a.losses import LossConfig, loss_from_batch, masked_action_ce
from native_a.sampler import baseline_batch, canonical, require, sha256_file
from native_a.large_batch import make_replacement_sampler_class, SAMPLER_POLICY
from native_a.joint_batch import make_native_step
from native_a.history_subset import make_restricted_sampler_class
NativeHistorySampler = make_restricted_sampler_class(make_replacement_sampler_class())


FORMAT = "native-ad-a/1"


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def cpu_receipt():
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
            "pid": os.getpid(), "jax_version": jax.__version__, "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kinds": [device.device_kind for device in jax.devices()],
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def tensor_sha(params):
    result = hashlib.sha256()
    # State-dict paths are independent of FrozenDict versus plain dict, so the
    # inference loader and optimizer restore produce the same tensor identity.
    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [str(key)])
        else:
            array = np.asarray(value)
            result.update(canonical(path).encode())
            result.update(str(array.dtype).encode())
            result.update(canonical(list(array.shape)).encode())
            result.update(array.tobytes())
    visit(serialization.to_state_dict(params), [])
    return result.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("native_a/train.py", "native_a/sampler.py", "native_a/model.py", "native_a/losses.py",
             "benchmarks/baselines/ad/model.py", "runners/history_adapter.py",
             "native_a/large_batch.py", "native_a/joint_batch.py", "native_a/mapped_history.py", "native_a/history_subset.py")
    return {name: sha256_file(root / name) for name in names}


def build_model(config):
    ad = dict(config["ad"])
    ad["obs_shape"] = tuple(ad["obs_shape"])
    ad = ADConfig(**ad)
    if config["mode"] == "baseline":
        return ADModel(ad)
    return PersistentADModel(PersistentADConfig(ad=ad, persistent_dim=config["persistent_dim"], return_cross_logits=False))


def _checkpoint_payload(path):
    path = Path(path).resolve()
    pointer = None
    if path.is_dir():
        path = path / "latest.json"
    if path.suffix == ".json":
        pointer = json.loads(path.read_text())
        target = Path(pointer["path"])
        target = target if target.is_absolute() else path.parent / target
        require(sha256_file(target) == pointer["sha256"], "Checkpoint pointer SHA mismatch")
        path = target
    payload = serialization.msgpack_restore(path.read_bytes())
    require(payload.get("format") == FORMAT, "Unsupported native checkpoint format")
    config = json.loads(payload["config_json"])
    require(payload["config_sha256"] == hashlib.sha256(canonical(config).encode()).hexdigest(), "Checkpoint configuration corrupted")
    require(int(payload["step"]) == int(payload["train_state"]["step"]), "Checkpoint optimizer step mismatch")
    if pointer is not None:
        require(int(pointer["step"]) == int(payload["step"]), "Checkpoint pointer step mismatch")
    return payload, config, path


def load_native_checkpoint(path):
    """Return (model, raw_params, config, metadata), without any optimizer step.

    Accept a checkpoint file, latest.json, terminal.json, or a run directory.
    Caller applies `model.apply({'params': params}, ...)` or `apply_batch`.
    Serialized parameter arrays are backend-neutral; the caller selects CPU or
    GPU before importing JAX. Loading never creates or updates an optimizer.
    """
    payload, config, checkpoint = _checkpoint_payload(path)
    model = build_model(config)
    params = payload["train_state"]["params"]
    require(tensor_sha(params) == payload["params_sha256"], "Checkpoint parameter tensor hash mismatch")
    return model, params, config, {"step": int(payload["step"]), "path": str(checkpoint),
                                   "sha256": sha256_file(checkpoint), "params_sha256": payload["params_sha256"],
                                   "data": config["data"], "code_sha256": config["code_sha256"]}


def optimizer(config):
    hp = config["optimizer"]
    warmup = hp["warmup_steps"]
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, hp["learning_rate"], warmup),
        optax.cosine_decay_schedule(hp["learning_rate"], max(1, config["num_steps"] - warmup))], [warmup])
    return optax.chain(optax.clip_by_global_norm(hp["max_grad_norm"]),
                       optax.adamw(schedule, weight_decay=hp["weight_decay"]))


def ad_kwargs(batch, train):
    return {"obs": batch["obs"], "prev_actions": batch["prev_actions"], "prev_rewards": batch["prev_rewards"],
            "attention_mask": batch["attention_mask"], "prev_teammate_actions": batch.get("prev_teammate_actions"), "train": train}


def make_step(model, config):
    baseline = config["mode"] == "baseline"
    loss_config = LossConfig("none" if baseline else config["mode"], config["lambda_p"], 0.0)

    @jax.jit
    def step(state, batch, dropout_rng):
        dropout_rng, use_rng = jax.random.split(dropout_rng)

        def objective(params):
            if baseline:
                logits = model.apply({"params": params}, **ad_kwargs(batch, True), rngs={"dropout": use_rng})
                ce, accuracy, count = masked_action_ce(logits, batch["target_actions"], batch["loss_mask"])
                metrics = {"loss": ce, "ego_ce": ce, "accuracy": accuracy, "valid_tokens": count,
                           "persistent_loss": jnp.array(0.0), "persist_inv": jnp.array(0.0),
                           "persist_var": jnp.array(0.0), "persist_cov": jnp.array(0.0), "cross_ce": jnp.array(0.0)}
                return ce, metrics
            outputs = apply_batch(model, params, batch, train=True, dropout_rng=use_rng)
            return loss_from_batch(outputs, batch, loss_config)

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        metrics["gradient_norm"] = optax.global_norm(gradients)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(gradients)]))
        metrics["gradients_finite"] = finite
        next_state = state.apply_gradients(grads=gradients)
        return next_state, metrics, dropout_rng

    return step


def save_checkpoint(out, state, config, rng, relation_rng, dropout_rng, *, metric_path, metric_rows):
    step = int(state.step)
    path = out / "checkpoints" / f"step_{step:08d}.msgpack"
    require(not path.exists(), f"Refusing to overwrite committed checkpoint: {path}")
    payload = {"format": FORMAT, "config_json": canonical(config),
               "config_sha256": hashlib.sha256(canonical(config).encode()).hexdigest(),
               "step": step, "train_state": serialization.to_state_dict(state),
               "numpy_rng_json": json.dumps(rng.bit_generator.state),
               "relation_rng_json": json.dumps(relation_rng.bit_generator.state),
               "dropout_rng": np.asarray(dropout_rng),
               "params_sha256": tensor_sha(state.params), "metrics": {"path": str(metric_path), "rows": metric_rows}}
    atomic_bytes(path, serialization.msgpack_serialize(payload))
    pointer = {"format": FORMAT, "path": str(path.relative_to(out)), "sha256": sha256_file(path), "step": step,
               "params_sha256": payload["params_sha256"]}
    atomic_json(out / "latest.json", pointer)
    return pointer


def relation_derangement(rng, rows):
    """Return a row permutation whose donor partner identity always differs."""
    identities = np.asarray([str(row["partner_identity"]) for row in rows], dtype=object)
    size = len(identities)
    if size < 2 or len(set(identities.tolist())) < 2:
        raise ValueError("A random relation requires at least two partner identities")
    groups = {}
    for index, identity in enumerate(identities):
        groups.setdefault(identity, []).append(index)
    for identity in groups:
        groups[identity] = rng.permutation(groups[identity]).astype(np.int32).tolist()
    order = np.asarray([index for identity in sorted(groups) for index in groups[identity]], dtype=np.int32)
    max_group = max(len(group) for group in groups.values())
    offsets = list(range(max_group, size))
    rng.shuffle(offsets)
    for offset in offsets:
        candidate = np.empty(size, dtype=np.int32)
        for position, row_index in enumerate(order):
            candidate[row_index] = order[(position + offset) % size]
        if np.all(identities != identities[candidate]):
            return candidate
    raise RuntimeError("Could not construct a partner-aware relation derangement")


def train(args):
    require(args.threads >= 1, "Threads must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        requested = [int(x) for x in args.cpu_ids.split(",")] if args.cpu_ids else allowed[:args.threads]
        require(requested and len(set(requested)) == len(requested) and set(requested) <= set(allowed), "Unavailable CPU affinity requested")
        os.sched_setaffinity(0, set(requested))
    require(jax.default_backend() == args.backend and all(x.platform == args.backend for x in jax.devices()),
            f"Requested {args.backend} backend is unavailable; CPU fallback is forbidden")
    if args.backend == "gpu":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        require(len(visible) == 1 and visible[0].strip() and visible[0].strip() != "-1"
                and len(jax.devices()) == 1, "GPU training requires exactly one externally assigned visible card")
        require(os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false",
                "GPU training requires XLA_PYTHON_CLIENT_PREALLOCATE=false before JAX initialization")
    require(args.num_steps >= 1 and args.batch_size >= 1 and args.save_every >= 1, "Invalid training budget")
    require(args.log_every >= 1 and args.wall_seconds >= 0, "Invalid logging or time budget")
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0
            and math.isfinite(args.weight_decay) and args.weight_decay >= 0, "Invalid optimizer configuration")
    require(args.mode in ("baseline", "none") or args.batch_size >= 2, "VC requires at least two real fixed partners")
    require(args.cross_weight == 0, "The first native-A protocol fixes cross_weight=0")
    require(args.lambda_p >= 0 and math.isfinite(args.lambda_p), "Invalid regularization weight")
    if args.mode in ("baseline", "none"):
        require(args.lambda_p == 0, "Unregularized methods require lambda_p=0")
    else:
        require(args.lambda_p > 0, "A VC condition requires explicit positive --lambda-p")
    require(args.query_len + args.support_count * args.support_len == args.seq_len, "Total history budget must equal seq_len")
    out = Path(args.out_dir).resolve()
    require(not out.exists() or args.resume, "Output exists; use --resume only for this exact run")
    sampler = NativeHistorySampler(args.h5_path, args.index_path, args.task_manifest, args.episode_index,
                                   query_len=args.query_len, support_count=args.support_count, support_len=args.support_len,
                                   use_teammate_actions=args.use_teammate_actions, cache_mb=args.hdf5_cache_mb,
                                   history_allowlist=args.history_allowlist)
    try:
        require(args.batch_size == 1024, "This run fixes effective batch1024")
        require(len(sampler.identities) == 20, "This run requires twenty actual training partners")
        require(jax.config.jax_default_matmul_precision == "highest", "All four methods require verified highest float32 precision")
        ad = ADConfig(obs_shape=sampler.obs_shape, num_actions=sampler.num_actions, embedding_dim=64,
                      hidden_dim=256, num_layers=4, num_heads=4, seq_len=args.seq_len,
                      use_teammate_actions=args.use_teammate_actions)
        warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.num_steps * .05))
        require(1 <= warmup <= args.num_steps, "Warmup outside the fixed update budget")
        config = {"format": FORMAT, "mode": args.mode, "seed": args.seed, "num_steps": args.num_steps,
                  "batch_size": args.batch_size, "ad": asdict(ad), "persistent_dim": 32,
                  "lambda_p": args.lambda_p, "cross_weight": 0.0,
                  "sampler": {"query_len": args.query_len, "support_count": args.support_count, "support_len": args.support_len,
                              "total_history_tokens": args.seq_len, "target": "recorded_ego_actions", "split": "train",
                              "support_policy": "independent explicitly annotated episodes before the query's first episode",
                              "query_loss_only": True, "distinct_partners_per_batch": False,
                              "large_batch_policy": SAMPLER_POLICY},
                  "numerics": {"dtype": "float32", "matmul_precision": "highest", "effective_batch": 1024,
                               "physical_microbatch": 128, "statistics": "joint1024", "optimizer_updates_per_batch": 1,
                               "dropout": "same split/fold_in key per microbatch replay", "replay_max_abs_tolerance": 1e-5},
                  "optimizer": {"learning_rate": args.learning_rate, "warmup_steps": warmup,
                                "weight_decay": args.weight_decay, "max_grad_norm": 1.0, "schedule": "warmup_cosine"},
                  "data": sampler.fingerprint(), "code_sha256": source_hashes(),
                  "batch_audit": "full first batch; every update stores ordered row-plan SHA256 and policy/counts; checkpoint sampler RNG enables replay",
                  "comparison": "paired_history_budget; upstream original-sampler AD remains a separate reference",
                  "relation_mode": args.relation_mode}
        config = json.loads(canonical(config))
        rng = np.random.default_rng(args.seed)
        relation_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 20260921, 17]))
        first_batch = sampler.sample(rng, args.batch_size)
        initial_plan = sampler.last_metadata
        model = build_model(config)
        init_key, dropout_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        original = ADModel(ad)
        # Initialize shape-independent weights on one row; keep the real first batch and its RNG.
        dummy = jax.tree_util.tree_map(lambda value: np.asarray(value[:1]).copy(), first_batch)
        flat = baseline_batch(dummy)
        original_params = original.init({"params": init_key}, **ad_kwargs(flat, False))["params"]
        backbone = {key: value for key, value in unfreeze(original_params).items() if key != "action_head"}
        if args.mode == "baseline":
            params = original_params
        else:
            validate_model_batch(first_batch, model.config)
            params = unfreeze(model.init({"params": init_key}, **batch_model_kwargs(dummy, train=False))["params"])
            # Both architectures start with the same upstream CNN/Transformer
            # weights. All three persistent conditions also share extra heads.
            require(set(params["backbone"]) == set(backbone), "Persistent backbone diverges from upstream AD parameters")
            params["backbone"] = backbone
            params = freeze(params)
        initial_sha = tensor_sha(params)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer(config))
        if args.resume:
            require((out / "config.json").is_file() and json.loads((out / "config.json").read_text()) == config,
                    "Resume changed the model, data, source, loss or fixed update schedule")
            payload, saved_config, _ = _checkpoint_payload(out / "latest.json")
            require(saved_config == config, "Checkpoint belongs to another configuration")
            state = serialization.from_state_dict(state, payload["train_state"])
            require(tensor_sha(state.params) == payload["params_sha256"], "Resume tensor hash mismatch")
            rng.bit_generator.state = json.loads(payload["numpy_rng_json"])
            relation_rng.bit_generator.state = json.loads(payload["relation_rng_json"])
            dropout_rng = jnp.asarray(payload["dropout_rng"])
            first_batch = None
        else:
            out.mkdir(parents=True, exist_ok=False)
            atomic_json(out / "config.json", config)
            atomic_json(out / "initialization.json", {"params_sha256": initial_sha,
                        "common_backbone_sha256": tensor_sha(backbone), "cpu": cpu_receipt(),
                        "first_plan_sha256": initial_plan["batch_plan_sha256"]})
            # A durable step-0 state makes interruption before the first periodic
            # checkpoint resumable. Its sampler RNG precedes the cached batch 1.
            save_checkpoint(out, state, config, np.random.default_rng(args.seed),
                            np.random.default_rng(np.random.SeedSequence([args.seed, 20260921, 17])), dropout_rng,
                            metric_path=out / "initialization.json", metric_rows=0)
        if int(state.step) == config["num_steps"]:
            pointer = json.loads((out / "latest.json").read_text())
            if not (out / "terminal.json").exists():
                atomic_json(out / "terminal.json", pointer)
            if not (out / "completion.json").exists():
                metric_path = Path(payload["metrics"]["path"])
                atomic_json(out / "completion.json", {"status": "PASS", "mode": args.mode,
                            "actual_updates": int(state.step), "planned_updates": config["num_steps"],
                            "terminal": pointer, "query_target": "recorded_ego_actions",
                            "total_history_tokens": args.seq_len, "metrics_path": str(metric_path),
                            "metrics_sha256": sha256_file(metric_path), "cpu": cpu_receipt(),
                            "recovered_terminal_receipt_from_complete_checkpoint": True,
                            "offline_training_only": True, "native_return_evaluation_performed": False})
            return {"status": "ALREADY_COMPLETE", "step": int(state.step), "terminal": str(out / "terminal.json")}
        require(int(state.step) < config["num_steps"], "Checkpoint exceeds fixed update budget")
        attempts = out / "attempts"
        attempts.mkdir(exist_ok=True)
        number = 1
        while (attempts / f"attempt_{number:04d}").exists():
            number += 1
        attempt = attempts / f"attempt_{number:04d}"
        attempt.mkdir(exist_ok=False)
        metrics_path = attempt / "metrics.jsonl"
        started = time.monotonic()
        atomic_json(attempt / "started.json", {"status": "RUNNING", "start_step": int(state.step), "resume": args.resume,
                    "cpu": cpu_receipt(), "uncommitted_prior_attempt_metrics_retained": True})
        step_fn = make_native_step(model, config, microbatch_size=128, effective_batch_size=1024)
        rows_written, last_pointer, failure = 0, None, None
        try:
            with metrics_path.open("x") as stream:
                while int(state.step) < config["num_steps"]:
                    if args.wall_seconds and time.monotonic() - started >= args.wall_seconds:
                        raise TimeoutError("Requested active training time cap reached")
                    if first_batch is not None:
                        batch, plan = first_batch, initial_plan
                        first_batch = None
                    else:
                        batch = sampler.sample(rng, args.batch_size)
                        plan = sampler.last_metadata
                    if args.mode != "baseline":
                        validate_model_batch(batch, model.config)
                    if args.mode != "baseline" and args.relation_mode == "random":
                        permutation = relation_derangement(relation_rng, plan["rows"])
                        batch["relation_pair_rows"] = permutation
                        donor_ids = [str(plan["rows"][int(index)]["partner_identity"]) for index in permutation]
                        recipient_ids = [str(row["partner_identity"]) for row in plan["rows"]]
                        plan = dict(plan, relation_mode="random",
                                    relation_pair_rows_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(),
                                    relation_donor_identities_sha256=hashlib.sha256(canonical(donor_ids).encode()).hexdigest(),
                                    relation_no_same_identity=bool(np.all(np.asarray(recipient_ids) != np.asarray(donor_ids))))
                    device_batch = baseline_batch(batch) if args.mode == "baseline" else batch
                    before = time.monotonic()
                    updated, metrics, next_rng = step_fn(state, device_batch, dropout_rng)
                    host_metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
                    require(all(math.isfinite(value) for value in host_metrics.values()) and host_metrics["gradients_finite"] == 1,
                            "Nonfinite loss or gradients; optimizer result was not committed")
                    state, dropout_rng = updated, next_rng
                    del batch, device_batch  # Release the consumed2GB host batch before sampling the next.
                    step = int(state.step)
                    if step == 1:
                        atomic_json(out / "runtime_first_update.json", {"step": step, "finite": True,
                            "effective_batch": 1024, "sampler_sha256": plan["batch_plan_sha256"],
                            "at": time.time()})
                    record = {"step": step, **host_metrics, "step_seconds": time.monotonic() - before,
                              "elapsed_seconds": time.monotonic() - started,
                              "batch_plan": plan if step == 1 else {key: value for key, value in plan.items() if key != "rows"},
                              "optimizer_update_completed": True, "device": jax.default_backend()}
                    stream.write(canonical(record) + "\n")
                    stream.flush()
                    rows_written += 1
                    if step % args.log_every == 0 or step == 1 or step == config["num_steps"]:
                        print(canonical({"event": "update_complete", "step": step, "mode": args.mode,
                                         "ego_ce": record["ego_ce"], "loss": record["loss"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
                    if step % args.save_every == 0 or step == config["num_steps"]:
                        os.fsync(stream.fileno())
                        last_pointer = save_checkpoint(out, state, config, rng, relation_rng, dropout_rng, metric_path=metrics_path, metric_rows=rows_written)
        except BaseException as error:
            failure = error
            # Preserve the last committed optimizer state. A failed forward's
            # sampled RNG is not saved, so continuation starts at a known point.
            atomic_json(attempt / "completion.json", {"status": "INCOMPLETE_BUDGET" if isinstance(error, TimeoutError) else "TECHNICAL_FAILURE",
                        "error": repr(error), "last_in_memory_step": int(state.step),
                        "latest_checkpoint": json.loads((out / "latest.json").read_text()) if (out / "latest.json").exists() else None,
                        "elapsed_seconds": time.monotonic() - started, "cpu": cpu_receipt()})
            raise
        finally:
            if failure is None:
                require(source_hashes() == config["code_sha256"], "Training source changed during the run")
        require(last_pointer is not None and int(state.step) == config["num_steps"], "Fixed update budget incomplete")
        atomic_json(out / "terminal.json", last_pointer)
        receipt = {"status": "PASS", "mode": args.mode, "actual_updates": int(state.step),
                   "planned_updates": config["num_steps"], "terminal": last_pointer,
                   "query_target": "recorded_ego_actions", "total_history_tokens": args.seq_len,
                   "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path),
                   "elapsed_seconds_this_attempt": time.monotonic() - started, "cpu": cpu_receipt(),
                   "offline_training_only": True, "native_return_evaluation_performed": False}
        atomic_json(attempt / "completion.json", receipt)
        atomic_json(out / "completion.json", receipt)
        return receipt
    finally:
        sampler.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("h5-path", "index-path", "task-manifest", "episode-index", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("baseline", "none", "VC", "I+VC"), required=True)
    p.add_argument("--lambda-p", type=float, default=0.0)
    p.add_argument("--cross-weight", type=float, default=0.0)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=500)
    p.add_argument("--query-len", type=int, default=300)
    p.add_argument("--support-count", type=int, default=2)
    p.add_argument("--support-len", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu-ids")
    p.add_argument("--hdf5-cache-mb", type=float, default=64)
    p.add_argument("--history-allowlist")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wall-seconds", type=float, default=0)
    p.add_argument("--use-teammate-actions", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--relation-mode", choices=("matched", "random"), default="matched")
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
