"""Receipt-bound orchestration of the unmodified native benchmark, CPU only.

Planning and syntax checks are safe on macOS; execution requires Linux.  A
smoke plan tests the original IPPO -> PPO history -> AD -> reward path only.
It does not establish partner competence or method effectiveness.  All
scientific budget increases require a new explicit plan and a new run root.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time

UPSTREAM_SHA = "22f6c1783e4fdc2153a84e57e197cbfabd0eace3"
CORE_FILES = [
    "envs/__init__.py", "envs/overcooked_v2/overcooked.py",
    "envs/overcooked_v2/overcooked_v2_wrapper.py", "envs/overcooked_v2/layouts.py",
    "teammate_generation/train_ippo_overcooked_v2.py",
    "teammate_generation/generate_ippo_manifest.py", "marl/ippo.py",
    "runners/task_runner.py", "runners/history_recorder.py",
    "runners/history_adapter.py", "scripts/build_index.py",
    "benchmarks/baselines/ad/train.py", "benchmarks/baselines/ad/model.py",
    "benchmarks/baselines/ad/buffer.py", "common/save_load_utils.py", "eval_icrl.py",
]
NATIVE_DEFAULTS = {"agent_view_size": 2, "random_reset": False,
                   "random_agent_positions": True, "negative_rewards": True,
                   "sample_recipe_on_delivery": True, "flatten_obs": False,
                   "observation_type": "default", "op_ingredient_permutations": None}
PHASES = ["partner", "manifest", "history", "dataset", "ad", "sanity"]
ENGINEERING_FILES = {"native_a/pipeline.py", "native_a/evaluate.py",
                     "native_a/original_checkpoint.py"}
LOADED_PIPELINE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(data)
    else:
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp.write_text(data)
        os.replace(tmp, path)


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numeric value in receipt/artifact")
    if isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, list):
        for v in value:
            finite_tree(v)


def native_factory_contract(repo):
    tree = ast.parse((Path(repo) / "envs/__init__.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                                               t.id == "BENCHMARK_DEFAULTS"
                                               for t in node.targets):
            found = ast.literal_eval(node.value)
    if any(found.get(k) != v for k, v in NATIVE_DEFAULTS.items()):
        raise ValueError(f"Native benchmark factory contract mismatch: {found}")
    return found


def source_inventory(repo):
    repo = Path(repo)
    files = CORE_FILES + ["native_a/pipeline.py"]
    if (repo / "native_a/evaluate.py").exists():
        files.append("native_a/evaluate.py")
    if (repo / "native_a/original_checkpoint.py").exists():
        files.append("native_a/original_checkpoint.py")
    return {p: digest(repo / p) for p in files}


def verify_source_binding(p, planfile):
    """Permit only explicitly listed mechanical repairs bound to this old plan.

    The original plan is never rewritten.  Auto-discovery also covers new worker
    processes launched by a parent that loaded the pre-repair pipeline code.
    """
    repo, plan_sha = Path(p["repo"]).resolve(), digest(planfile)
    if read_json(planfile) != p:
        raise ValueError("Loaded plan differs from the frozen plan file")
    expected = dict(p["source_hashes"])
    if not set(CORE_FILES + ["native_a/pipeline.py", "native_a/evaluate.py"]) <= set(expected):
        raise ValueError("Plan is missing required original source bindings")
    amendment_path = Path(p["run_root"]) / "engineering_amendments.json"
    amendment_binding = None
    if amendment_path.exists():
        amendment = read_json(amendment_path)
        if (set(amendment) != {"schema_version", "plan_sha256", "changes"}
                or amendment["schema_version"] != 1 or amendment["plan_sha256"] != plan_sha):
            raise ValueError("Engineering amendment must bind the exact unchanged plan SHA")
        changes, seen = amendment["changes"], set()
        if not isinstance(changes, list) or not changes:
            raise ValueError("Engineering amendment requires an explicit nonempty file list")
        for change in changes:
            if not isinstance(change, dict) or set(change) != {"file", "old_sha", "new_sha", "reason"}:
                raise ValueError("Engineering amendment accepts only file/old_sha/new_sha/reason")
            name = change["file"]
            if name not in ENGINEERING_FILES or name in seen:
                raise ValueError(f"Unlisted or duplicate engineering repair: {name}")
            seen.add(name)
            if (change["old_sha"] != p["source_hashes"].get(name)
                    or (name not in expected and name != "native_a/original_checkpoint.py")):
                raise ValueError(f"Repair old SHA must match the original plan: {name}")
            new_sha = change["new_sha"]
            if (not isinstance(new_sha, str) or len(new_sha) != 64
                    or any(c not in "0123456789abcdef" for c in new_sha)
                    or new_sha == change["old_sha"]
                    or not isinstance(change["reason"], str) or not change["reason"].strip()):
                raise ValueError(f"Invalid engineering repair SHA/reason: {name}")
            expected[name] = new_sha
        amendment_binding = {"path": str(amendment_path.resolve()), "sha256": digest(amendment_path)}
    if "native_a/original_checkpoint.py" not in expected:
        raise ValueError("New checkpoint loader needs an explicit plan or added-file amendment binding")
    actual = {name: digest(repo / name) for name in expected}
    for name, sha in expected.items():
        if actual[name] != sha:
            raise ValueError(f"Source differs from frozen plan plus listed engineering repair: {name}")
    if ((repo / "native_a/pipeline.py").resolve() != Path(__file__).resolve()
            or actual["native_a/pipeline.py"] != LOADED_PIPELINE_SHA256):
        raise ValueError("Loaded pipeline code does not match the verified repository source")
    return {"plan_sha256": plan_sha, "actual_source_hashes": actual,
            "loaded_pipeline_sha256": LOADED_PIPELINE_SHA256,
            "engineering_amendments": amendment_binding}


def make_plan(args):
    repo = Path(args.repo).resolve()
    run = Path(args.run_root).resolve()
    seeds = [int(x) for x in args.partner_seeds.split(",")]
    layouts = args.layouts.split(",")
    if len(set(seeds)) != len(seeds) or len(set(layouts)) != len(layouts):
        raise ValueError("Duplicate layout or partner seed")
    if any(not x or "/" in x or ".." in x for x in layouts):
        raise ValueError("Invalid layout identifier")
    if args.profile == "custom" and any(x is None for x in
            [args.partner_steps, args.history_steps, args.ad_steps, args.min_partner_return]):
        raise ValueError("Custom plans require explicit partner/history/AD steps and min-partner-return")
    cfg = {
        "profile": args.profile, "technical_only": args.profile == "smoke",
        "layouts": layouts, "partner_seeds": seeds, "ad_seeds": [int(x) for x in args.ad_seeds.split(",")],
        "partner_steps": args.partner_steps or 131072,
        "partner_envs": args.num_envs, "partner_rollout": 256,
        "partner_checkpoints": 2, "history_steps": args.history_steps or 524288,
        "history_envs": args.num_envs, "history_rollout": 256,
        "history_minibatches": 64, "history_update_epochs": 4,
        "history_record_envs": args.num_envs, "history_record_first_steps": 100,
        "history_save_interval": 0, "max_histories_per_task": 128,
        "num_teammates_per_layout": args.teammates_per_layout,
        "min_partner_return": args.min_partner_return if args.min_partner_return is not None else -1e30,
        "manifest_sample_seed": 4400, "history_seed": 4500,
        "ad_steps": args.ad_steps or 8, "ad_batch_size": args.ad_batch_size,
        "ad_seq_len": 500, "ad_embedding_dim": 64, "ad_hidden_dim": 256,
        "ad_layers": 4, "ad_heads": 4, "use_teammate_actions": False,
        "train_episode_horizon": 400, "eval_episode_horizon": 100,
        "eval_episodes": args.eval_episodes, "eval_seed": 4600,
        "eval_scope": "training_RL_partner_sanity", "official_test_read": False,
        "science_matrix_auto_schedule": False,
    }
    if args.num_envs < 64 or args.num_envs % 64:
        raise ValueError("History ego PPO requires NUM_ENVS divisible by 64 minibatches")
    for key in ["partner_steps", "history_steps"]:
        if cfg[key] % (args.num_envs * 256):
            raise ValueError(f"{key} must be divisible by num_envs * 256; no silent budget flooring")
    if cfg["partner_steps"] // (args.num_envs * 256) < 2:
        raise ValueError("At least two PPO updates needed for two checkpoints")
    if cfg["history_steps"] // args.num_envs < 2000:
        raise ValueError("Need >= five 100-step recorded episode prefixes for original AD context500")
    if any(cfg[k] <= 0 for k in ["ad_steps", "ad_batch_size", "eval_episodes", "num_teammates_per_layout"]):
        raise ValueError("Positive budgets required")
    plan = {
        "schema_version": 1, "created_at": now(), "upstream_sha": UPSTREAM_SHA,
        "repo": str(repo), "run_root": str(run), "python": str(Path(args.python).absolute()),
        "source_hashes": source_inventory(repo), "native_factory_defaults": native_factory_contract(repo),
        "config": cfg, "cpu_limit": args.cpu_limit,
        "max_active_seconds": args.max_active_seconds,
        "notes": [
            "All phases call official algorithms and the native make_env factory.",
            "No heuristic test manifest supplies any training data.",
            "Smoke min-return filter is deliberately disabled solely to test artifact plumbing; no competence claim.",
            "Original CNN/GRU and AD widths/layers/context are preserved; budgets/parallelism/AD batch are declared.",
            "HDF5 uses official quality selection; expert-action relabeling is disabled because original AD does not use it.",
            "Episode sidecar comes from original episodes.json, never inferred from artificial done flags.",
            "Sanity returns on training RL partners are not held-out generalization or a method-effect gate.",
            "Independent RL development partners and the 3-seed scientific matrix require a separately frozen plan.",
        ],
    }
    write_json(args.output, plan, exclusive=True)
    print(canonical({"plan": str(Path(args.output).resolve()), "sha256": digest(args.output),
                     "first_stage": partner_stages(plan)[0]}))


def module(plan, name, *args):
    return [plan["python"], "-u", "-m", name, *map(str, args)]


def stage(sid, phase, argv, outputs, validation, inputs=None, **extra):
    return dict(id=sid, phase=phase, argv=argv, output_roots=list(map(str, outputs)),
                validation=validation, inputs=list(map(str, inputs or [])), **extra)


def partner_stages(p):
    c, root = p["config"], Path(p["run_root"])
    result = []
    for layout in c["layouts"]:
        for seed in c["partner_seeds"]:
            out = root / "partners" / f"ippo_{layout}_seed{seed}"
            cmd = module(p, "teammate_generation.train_ippo_overcooked_v2",
                "--layout", layout, "--seed", seed, "--num_seeds", 1,
                "--num_envs", c["partner_envs"], "--num_steps", 256,
                "--total_timesteps", c["partner_steps"], "--num_checkpoints", 2,
                "--num_chunks", 1, "--save_interval", 0, "--max_steps", 400,
                "--output_dir", out, "--gpu", -1)
            result.append(stage(f"partner_{layout}_{seed}", "partner", cmd, [out], "partner"))
    return result


def manifest_paths(p):
    root = Path(p["run_root"])
    return [root / "manifests" / layout / "track_teammate" / layout / "ippo/manifest_train.jsonl"
            for layout in p["config"]["layouts"]]


def manifest_stages(p, planfile):
    c, root = p["config"], Path(p["run_root"])
    result = []
    for layout in c["layouts"]:
        out = root / "manifests" / layout
        cmd = module(p, "teammate_generation.generate_ippo_manifest", "--layout", layout,
            "--model_dir", root / "partners", "--output_dir", out,
            "--num_teammates", c["num_teammates_per_layout"],
            f"--min_return={c['min_partner_return']}", "--sample_seed", c["manifest_sample_seed"],
            "--base_seed", c["history_seed"], "--split", "train", "--track", "teammate")
        result.append(stage(f"manifest_{layout}", "manifest", cmd, [out], "manifest",
                            inputs=[root / "partners" / f"ippo_{layout}_seed{s}" for s in c["partner_seeds"]]))
    out = root / "dataset_inputs"
    result.append(stage("merge_manifests", "manifest", module(p, "native_a.pipeline", "worker",
                        "--plan", planfile, "--operation", "merge"), [out], "merged_manifest",
                        inputs=manifest_paths(p)))
    return result


def history_stages(p):
    c, root = p["config"], Path(p["run_root"])
    manifest = root / "dataset_inputs/train_manifest.jsonl"
    rows = jsonlines(manifest)
    result = []
    for idx, row in enumerate(rows):
        if row["split"] != "train" or row["teammate"]["kind"] != "rl":
            raise ValueError("Training manifest must contain exclusively RL training partners")
        tid = row["task_id"]
        if "/" in tid or ".." in tid:
            raise ValueError("Invalid task id")
        out = root / "histories" / tid
        cmd = module(p, "runners.task_runner", "--manifest", manifest, "--task_idx", idx,
            "--out_dir", root / "histories", "--total_steps", c["history_steps"],
            "--num_envs", c["history_envs"], "--num_minibatches", 64,
            "--rollout_length", 256, "--update_epochs", 4, "--record_envs", c["history_record_envs"],
            "--record_first_steps", 100, "--save_interval", c["history_save_interval"],
            "--max_steps", 400, "--actor_type", "cnn_rnn", "--fc_dim_size", 128,
            "--gru_hidden_dim", 128, "--rew_shaping_horizon", 15000000,
            "--log_every", 1, "--csv_interval", 1, "--gpu", -1)
        result.append(stage(f"history_{idx:04d}", "history", cmd, [out], "history",
                            inputs=[manifest, row["teammate"]["ckpt"]], task_id=tid))
    return result


def tail_stages(p, planfile):
    c, root = p["config"], Path(p["run_root"])
    data = root / "dataset"
    manifest = root / "dataset_inputs/train_manifest.jsonl"
    result = [stage("build_dataset", "dataset", module(p, "native_a.pipeline", "worker",
                    "--plan", planfile, "--operation", "pack"), [data], "dataset",
                    inputs=[root / "histories", manifest])]
    for seed in c["ad_seeds"]:
        out = root / "original_ad" / f"seed{seed}"
        cmd = module(p, "benchmarks.baselines.ad.train", "--h5_path", data / "histories.h5",
            "--index_path", data / "histories_index.jsonl", "--out_dir", out,
            "--num_steps", c["ad_steps"], "--batch_size", c["ad_batch_size"],
            "--seq_len", 500, "--embedding_dim", 64, "--hidden_dim", 256,
            "--num_layers", 4, "--num_heads", 4, "--seed", seed,
            "--eval_every", 0, "--save_every", max(1, c["ad_steps"] // 4),
            "--log_every", 1, "--csv_interval", 1, "--gpu", -1)
        result.append(stage(f"original_ad_{seed}", "ad", cmd, [out], "ad", inputs=[data]))
    for seed in c["ad_seeds"]:
        out = root / "sanity_returns" / f"seed{seed}"
        result.append(stage(f"sanity_returns_{seed}", "sanity", module(p, "native_a.pipeline", "worker",
            "--plan", planfile, "--operation", "sanity", "--ad-seed", seed), [out], "sanity",
            inputs=[manifest, root / "original_ad" / f"seed{seed}"]))
    return result


def file_inventory(roots):
    result = {}
    for root in map(Path, roots):
        if not root.exists():
            raise FileNotFoundError(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"Empty artifact: {root}")
        for path in files:
            result[str(path)] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def validate_ad_training_log(model_root, p):
    """A terminal filename is not evidence that the original AD loop completed."""
    model_root, c = Path(model_root), p["config"]
    conf = read_json(model_root / "config.json")
    for key, expected in {"seq_len": 500, "embedding_dim": 64, "hidden_dim": 256,
            "num_layers": 4, "num_heads": 4, "num_steps": c["ad_steps"],
            "batch_size": c["ad_batch_size"], "log_every": 1,
            "eval_every": 0, "csv_interval": 1, "csv_log": True,
            "use_teammate_actions": False}.items():
        if conf.get(key) != expected:
            raise ValueError(f"Original AD contract mismatch: {key}")
    expected_seed = int(model_root.name.removeprefix("seed"))
    if expected_seed not in c["ad_seeds"] or conf["seed"] != expected_seed:
        raise ValueError("Original AD seed differs from the frozen plan")
    metrics = read_json(model_root / "metrics.json")
    if (not isinstance(metrics, list) or len(metrics) != c["ad_steps"]
            or [x.get("step") for x in metrics] != list(range(c["ad_steps"]))):
        raise ValueError("Original AD metrics must contain every step 0..N-1 exactly once")
    finite_tree(metrics)
    for row in metrics:
        for key in ("loss", "accuracy"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), (float, int)):
                raise ValueError(f"Invalid AD metric: {key}")
    csv_path = model_root / "training_metrics.csv"
    with csv_path.open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if [int(x["step"]) for x in csv_rows] != list(range(c["ad_steps"])):
        raise ValueError("Original AD CSV does not contain all frozen training steps")
    for logged, saved in zip(csv_rows, metrics):
        for key in ("loss", "accuracy"):
            if not math.isfinite(float(logged[key])) or float(logged[key]) != saved[key]:
                raise ValueError("Original AD CSV/JSON metrics disagree or are nonfinite")
    return {"metrics_sha256": digest(model_root / "metrics.json"),
            "training_csv_sha256": digest(csv_path), "logged_steps": len(metrics),
            "first_logged_step": 0, "last_logged_step": c["ad_steps"] - 1,
            "all_planned_steps_logged": True}


def load_completed_original_ad(model_root, p):
    from native_a.original_checkpoint import load_original_ad_checkpoint
    log_receipt = validate_ad_training_log(model_root, p)
    checkpoint = Path(model_root) / "checkpoints" / f"checkpoint_{p['config']['ad_steps']}"
    model, variables, model_config, receipt = load_original_ad_checkpoint(
        checkpoint, expected_step=p["config"]["ad_steps"])
    if (receipt["checkpoint_step"] != p["config"]["ad_steps"]
            or not receipt["checkpoint_is_terminal"]
            or not receipt["raw_vs_typed_parameters_exact"]):
        raise ValueError("Actual original AD TrainState is not the completed terminal state")
    return model, variables, model_config, {**log_receipt,
        "terminal_checkpoint": str(checkpoint), "checkpoint_verification": receipt,
        "parameters_finite": True,
        "checkpoint_loader_sha256": digest(Path(p["repo"]) / "native_a/original_checkpoint.py")}


def validate_stage(s, p):
    roots = list(map(Path, s["output_roots"]))
    kind, c = s["validation"], p["config"]
    details = {"status": "PASS", "kind": kind, "official_test_read": False}
    if kind == "partner":
        run = roots[0] / "ippo_train_run"
        conf = read_json(run / "config.json")
        if conf["ENV_KWARGS"]["max_steps"] != 400 or conf["ACTOR_TYPE"] != "cnn_rnn":
            raise ValueError("Original IPPO model/environment contract mismatch")
        for key, expected in {"NUM_ENVS": c["partner_envs"], "ROLLOUT_LENGTH": 256,
                              "TOTAL_TIMESTEPS": c["partner_steps"], "NUM_CHECKPOINTS": 2,
                              "NUM_SEEDS": 1, "NUM_MINIBATCHES": 64,
                              "UPDATE_EPOCHS": 4, "FC_DIM_SIZE": 128,
                              "GRU_HIDDEN_DIM": 128, "NUM_CHUNKS": 1,
                              "SAVE_INTERVAL": 0, "LR": 0.00025}.items():
            if conf[key] != expected:
                raise ValueError(f"Original IPPO configuration mismatch: {key}")
        expected_seed = int(roots[0].name.rsplit("_seed", 1)[1])
        expected_layout = roots[0].name[len("ippo_"):].rsplit("_seed", 1)[0]
        if conf["TRAIN_SEED"] != expected_seed:
            raise ValueError("Partner seed mismatch")
        if conf["ENV_KWARGS"]["layout"] != expected_layout or conf["ENV_KWARGS"]["flatten_obs"] is not False:
            raise ValueError("Partner layout/observation configuration mismatch")
        returns = read_json(run / "checkpoint_returns.json")
        finite_tree(returns)
        for i in range(2):
            if not any(x.is_file() for x in (run / "pi_0" / f"ckpt_{i}").rglob("*")):
                raise ValueError(f"Missing actual partner checkpoint {i}")
        details.update(checkpoint_returns=returns, num_updates=conf.get("NUM_UPDATES"))
    elif kind in ("manifest", "merged_manifest"):
        paths = list(roots[0].rglob("manifest_train.jsonl")) if kind == "manifest" else [roots[0] / "train_manifest.jsonl"]
        rows = [x for path in paths for x in jsonlines(path)]
        if not rows or len({x["task_id"] for x in rows}) != len(rows):
            raise ValueError("Empty or duplicate task manifest")
        if any(x["split"] != "train" or x["teammate"]["kind"] != "rl" for x in rows):
            raise ValueError("Training contamination: only RL train tasks are accepted")
        details["num_tasks"] = len(rows)
    elif kind == "history":
        from scripts.collect_histories import validate_task_outputs
        validate_task_outputs(roots[0])
        meta = read_json(roots[0] / "metadata.json")
        eps = read_json(roots[0] / "episodes.json")["episodes"]
        if meta.get("record_first_steps") != 100 or not eps:
            raise ValueError("Native episode recording contract mismatch")
        details.update(num_recorded_prefixes=len(eps), updates_completed=meta.get("updates_completed"))
    elif kind == "dataset":
        receipt = read_json(roots[0] / "dataset_receipt.json")
        if receipt["status"] != "PASS" or not receipt["num_histories"]:
            raise ValueError("Dataset QA failed")
        details.update(receipt)
    elif kind == "ad":
        _, _, _, ad_receipt = load_completed_original_ad(roots[0], p)
        details.update(ad_receipt)
    elif kind == "sanity":
        receipt = read_json(roots[0] / "sanity_receipt.json")
        finite_tree(receipt)
        if receipt["status"] != "PASS" or receipt["official_test_read"]:
            raise ValueError("Sanity evaluation scope mismatch")
        details.update(receipt)
    return details


class Runner:
    def __init__(self, plan, planfile, retry_failed=False):
        self.p, self.planfile = plan, str(Path(planfile).resolve())
        self.root = Path(plan["run_root"])
        self.retry_failed = retry_failed
        self.active_seconds = 0.0
        self.completed = []
        self.revalidated = {}
        self.plan_hash = digest(planfile)
        for completion in self.root.glob("stages/*/attempt_*/completion.json"):
            self.active_seconds += float(read_json(completion).get("active_seconds", 0))

    def adopt_partner_smoke(self, s, external_path):
        """Adopt the already running first smoke only after its external job closed."""
        source_binding = verify_source_binding(self.p, self.planfile)
        if not self.p["config"]["technical_only"] or s["phase"] != "partner":
            raise ValueError("This adoption entry is only for the pre-launched partner smoke")
        stage_root = self.root / "stages" / s["id"]
        if list(stage_root.glob("attempt_*")):
            return  # Normal receipt/resume checks below remain authoritative.
        ext = read_json(external_path)
        if not ext.get("job_handle") or not ext.get("command"):
            raise ValueError("External adoption needs actual job handle and launch command")
        evidence = {str(Path(external_path).resolve()): {"sha256": digest(external_path)}}
        if ext.get("jobdone_rc_path"):
            rcpath = Path(ext["jobdone_rc_path"])
            if int(rcpath.read_text().strip()) != 0:
                raise ValueError("External job is not complete with rc=0")
            evidence[str(rcpath)] = {"sha256": digest(rcpath)}
        else:
            if ext.get("exit_code") != 0 or not ext.get("evidence_path"):
                raise ValueError("Need external jobdone.rc=0 or recorded successful tool completion evidence")
            ep = Path(ext["evidence_path"])
            evidence[str(ep)] = {"sha256": digest(ep)}
        if ext.get("stdout_path"):
            evidence[ext["stdout_path"]] = {"sha256": digest(ext["stdout_path"])}
        validation = validate_stage(s, self.p)
        inventory = file_inventory(s["output_roots"])
        attempt = stage_root / "attempt_001"
        attempt.mkdir(parents=True)
        elapsed = float(ext.get("active_seconds", 0.0))
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError("Invalid external elapsed time")
        write_json(attempt / "request.json", {**s, "plan_sha256": self.plan_hash,
                   "validation_source_binding": source_binding,
                   "adopted_external_job": ext, "evidence_inventory": evidence,
                   "adopted_at": now()}, exclusive=True)
        write_json(attempt / "validation.json", validation, exclusive=True)
        self.active_seconds += elapsed
        write_json(attempt / "completion.json", {"stage": s["id"], "phase": "partner", "status": "PASS",
            "plan_sha256": self.plan_hash, "exit_code": 0, "error": None,
            "completed_at": now(), "active_seconds": elapsed,
            "external_elapsed_unavailable": "active_seconds" not in ext,
            "output_inventory": inventory, "official_test_read": False, "technical_only": True,
            "validation_source_binding": source_binding,
            "adopted_external_job_handle": ext["job_handle"]}, exclusive=True)
        print(canonical({"event": "adopted", "stage": s["id"], "external_job": ext["job_handle"]}), flush=True)

    def run(self, s):
        source_binding = verify_source_binding(self.p, self.planfile)
        stage_root = self.root / "stages" / s["id"]
        prior = sorted(stage_root.glob("attempt_*/completion.json"))
        if prior:
            last = read_json(prior[-1])
            if last["plan_sha256"] != self.plan_hash:
                raise ValueError("Plan changed inside an existing run root")
            if last["status"] == "PASS":
                actual = file_inventory(s["output_roots"])
                if actual != last["output_inventory"]:
                    raise ValueError(f"Completed artifacts changed: {s['id']}")
                if s["validation"] == "ad":
                    self.revalidated[s["id"]] = {"source_binding": source_binding,
                                                  "validation": validate_stage(s, self.p)}
                print(canonical({"event": "reused", "stage": s["id"]}), flush=True)
                self.completed.append(s["id"])
                return
            if not self.retry_failed:
                raise RuntimeError(f"Failed stage retained: {s['id']}; explicit --retry-failed required after repair")
        existing = sorted(stage_root.glob("attempt_*"))
        if any(not (x / "completion.json").exists() for x in existing):
            raise RuntimeError(f"Unclosed attempt at {stage_root}; inspect its PID before recovery")
        attempt = stage_root / f"attempt_{len(existing) + 1:03d}"
        attempt.mkdir(parents=True)
        if prior:
            for i, path in enumerate(map(Path, s["output_roots"])):
                if path.exists():
                    archived = prior[-1].parent / "partial_outputs" / str(i)
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(archived)
        elif any(Path(x).exists() for x in s["output_roots"]):
            raise RuntimeError(f"Unreceipted output exists for {s['id']}; refusing overwrite")
        remaining = self.p["max_active_seconds"] - self.active_seconds
        if remaining <= 0:
            raise RuntimeError("Actual active execution budget exhausted")
        spec = {**s, "plan_sha256": self.plan_hash, "created_at": now(),
                "source_binding": source_binding,
                "hostname": platform.node(), "parent_pid": os.getpid(),
                "affinity": sorted(os.sched_getaffinity(0)), "technical_only": self.p["config"]["technical_only"],
                "input_inventory": file_inventory(s["inputs"]) if s["inputs"] else {}}
        write_json(attempt / "request.json", spec, exclusive=True)
        env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                   CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
        print(canonical({"event": "start", "stage": s["id"], "request": str(attempt / "request.json")}), flush=True)
        started, proc, error, rc = time.monotonic(), None, None, None
        try:
            with (attempt / "stdout.log").open("wb") as log:
                proc = subprocess.Popen(s["argv"], cwd=self.p["repo"], env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                write_json(attempt / "process.json", {"pid": proc.pid, "started_at": now()}, exclusive=True)
                sel = selectors.DefaultSelector()
                sel.register(proc.stdout, selectors.EVENT_READ)
                last_progress = time.monotonic()
                while True:
                    if time.monotonic() - started > remaining:
                        raise TimeoutError("Active execution budget exhausted during stage")
                    for key, _ in sel.select(timeout=1):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if chunk:
                            log.write(chunk)
                            log.flush()
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        else:
                            sel.unregister(key.fileobj)
                    if proc.poll() is not None and not sel.get_map():
                        break
                    if time.monotonic() - last_progress >= 30:
                        print(canonical({"event": "running", "stage": s["id"],
                                         "seconds": time.monotonic() - started}), flush=True)
                        last_progress = time.monotonic()
                sel.close()
                rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"Child exit code {rc}")
            if verify_source_binding(self.p, self.planfile) != source_binding:
                raise ValueError("Source binding changed while this stage was running")
            validation = validate_stage(s, self.p)
            write_json(attempt / "validation.json", validation, exclusive=True)
            inventory = file_inventory(s["output_roots"])
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            inventory = {}
            if proc and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            rc = proc.returncode if proc else rc
        elapsed = time.monotonic() - started
        self.active_seconds += elapsed
        result = {"stage": s["id"], "phase": s["phase"], "status": "FAIL" if error else "PASS",
                  "source_binding": source_binding,
                  "plan_sha256": self.plan_hash, "exit_code": rc, "error": error,
                  "completed_at": now(), "active_seconds": elapsed,
                  "cumulative_active_seconds": self.active_seconds,
                  "output_inventory": inventory, "official_test_read": False,
                  "technical_only": self.p["config"]["technical_only"]}
        write_json(attempt / "completion.json", result, exclusive=True)
        print(canonical({"event": "completed", "stage": s["id"], "status": result["status"],
                         "active_seconds": elapsed}), flush=True)
        if error:
            raise RuntimeError(error)
        self.completed.append(s["id"])


def run_pipeline(args):
    if platform.system() != "Linux":
        raise RuntimeError("Scientific execution is Linux remote only; macOS permits make-plan/inspect only")
    os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="-1")
    import fcntl
    p = read_json(args.plan)
    root = Path(p["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = native_factory_contract(p["repo"])
        source_binding = verify_source_binding(p, args.plan)
        cpus = sorted(os.sched_getaffinity(0))[:p["cpu_limit"]]
        os.sched_setaffinity(0, cpus)
        state = {"status": "RUNNING", "plan_sha256": digest(args.plan), "hostname": platform.node(),
                 "source_binding": source_binding,
                 "affinity": cpus, "native_factory_defaults": contract,
                 "official_test_read": False, "technical_only": p["config"]["technical_only"]}
        write_json(root / "pipeline_summary.json", state)
        runner = Runner(p, args.plan, args.retry_failed)
        try:
            if args.adopt_partner_smoke:
                runner.adopt_partner_smoke(partner_stages(p)[0], args.adopt_partner_smoke)
            ceiling = PHASES.index(args.through)
            for s in partner_stages(p) + manifest_stages(p, str(Path(args.plan).resolve())):
                if PHASES.index(s["phase"]) <= ceiling:
                    runner.run(s)
            if ceiling >= PHASES.index("history"):
                for s in history_stages(p):
                    runner.run(s)
            for s in tail_stages(p, str(Path(args.plan).resolve())):
                if PHASES.index(s["phase"]) <= ceiling:
                    runner.run(s)
            state.update(status="PASS", through=args.through,
                         interpretation="Engineering chain completed; no method-effect or environment-failure conclusion")
        except Exception as exc:
            state.update(status="STOPPED", error=f"{type(exc).__name__}: {exc}",
                         interpretation="Engineering/budget/partner-availability stop, not environment ineffectiveness")
            raise
        finally:
            state.update(completed_stages=runner.completed, active_seconds=runner.active_seconds,
                         revalidated_existing_ad=runner.revalidated, updated_at=now())
            write_json(root / "pipeline_summary.json", state)


def merge_worker(p):
    out = Path(p["run_root"]) / "dataset_inputs"
    out.mkdir(parents=True)
    rows = [row for path in manifest_paths(p) for row in jsonlines(path)]
    if not rows or any(x["split"] != "train" or x["teammate"]["kind"] != "rl" for x in rows):
        raise ValueError("RL-only training manifest required")
    if len({x["task_id"] for x in rows}) != len(rows):
        raise ValueError("Duplicate task identifiers")
    (out / "train_manifest.jsonl").write_text("".join(canonical(x) + "\n" for x in rows))
    write_json(out / "manifest_receipt.json", {"num_tasks": len(rows), "official_test_read": False,
                "source_manifests": file_inventory(manifest_paths(p)), "role": "training_RL_partners"})


def pack_worker(p):
    import h5py
    import numpy as np
    from scripts.build_index import BuildConfig, build_dataset
    root, c = Path(p["run_root"]), p["config"]
    out = root / "dataset"
    out.mkdir(parents=True)
    h5, index = out / "histories.h5", out / "histories_index.jsonl"
    build_dataset(BuildConfig(collected_root=str(root / "histories"), out_h5=str(h5),
        out_index=str(index), track="teammate", split="train", relabel=False,
        max_histories_per_task=c["max_histories_per_task"], gpu="-1"))
    rows, sidecar, unannotated = jsonlines(index), [], []
    if not rows:
        raise ValueError("No packed histories")
    episode_inputs, allowed = {}, {x["task_id"] for x in jsonlines(root / "dataset_inputs/train_manifest.jsonl")}
    with h5py.File(h5, "r") as store:
        for row in rows:
            if row["task_id"] not in allowed or row["split"] != "train" or row["teammate_kind"] != "rl":
                raise ValueError("Packed training data does not match the RL training manifest")
            g = store[row["h5_group"]]
            if tuple(g["obs"].shape[1:3]) != (5, 5) or g["obs"].ndim != 4:
                raise ValueError("Packed observations must be native local 5x5 grids")
            for name in ("obs", "actions", "rewards"):
                ds = g[name]
                for start in range(0, len(ds), 256):
                    values = ds[start:start + 256]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite dataset {name}")
                    if name == "actions" and ((values < 0).any() or (values >= 6).any()):
                        raise ValueError("Invalid action labels")
            task_dir = root / "histories" / row["task_id"]
            epfile, metafile = task_dir / "episodes.json", task_dir / "metadata.json"
            episode_inputs[str(epfile)] = digest(epfile)
            episode_inputs[str(metafile)] = digest(metafile)
            meta = read_json(metafile)
            prefix = int(meta["record_first_steps"])
            eps = sorted((e for e in read_json(epfile)["episodes"] if e["env_idx"] == row["env_idx"]),
                         key=lambda e: e["start_idx"])
            cursor, seen = 0, set()
            for e in eps:
                a, b = int(e["start_idx"]), int(e["end_idx"])
                if a != cursor or not a < b <= row["T"] or e["episode_id"] in seen:
                    raise ValueError("Explicit episode metadata is discontinuous/overlapping; do not guess from dones")
                if prefix and b - a != prefix:
                    raise ValueError("Unexpected shortened prefix: inspect recorder/chunk boundary before using for A")
                sidecar.append({"history_id": row["history_id"], "task_id": row["task_id"],
                    "env_idx": row["env_idx"], "episode_id": e["episode_id"], "start": a, "end": b,
                    "complete": prefix == 0, "recorded_prefix": prefix > 0,
                    "source_episode_metadata": str(epfile), "boundary_source": "explicit_episodes_json"})
                cursor = b
                seen.add(e["episode_id"])
            if cursor < row["T"]:
                unannotated.append({"history_id": row["history_id"], "start": cursor, "end": row["T"],
                                    "reason": "No completed prefix record in upstream episodes.json; no inferred episode id"})
            if len(eps) < 5:
                raise ValueError("Insufficient explicitly recorded episodes for 500-token paired assay")
    sidepath = out / "native_episode_index.jsonl"
    sidepath.write_text("".join(canonical(x) + "\n" for x in sidecar))
    write_json(out / "dataset_receipt.json", {"status": "PASS", "num_histories": len(rows),
        "num_annotated_prefixes": len(sidecar), "unannotated_tails": unannotated,
        "episode_metadata_sha256": episode_inputs, "h5_sha256": digest(h5), "index_sha256": digest(index),
        "episode_index_sha256": digest(sidepath), "observation_shape": rows[0]["obs_shape"],
        "official_test_read": False, "expert_relabel": False,
        "technical_only": c["technical_only"], "true_episode_boundaries_not_inferred_from_packed_dones": True})


def sanity_worker(p, seed, source_binding=None):
    import dataclasses
    import jax
    import eval_icrl as ev
    from benchmarks.manifest_schema import load_manifest
    from native_a.evaluate import StaticADInference
    root, c = Path(p["run_root"]), p["config"]
    out = root / "sanity_returns" / f"seed{seed}"
    out.mkdir(parents=True)
    model_root = root / "original_ad" / f"seed{seed}"
    checkpoint = model_root / "checkpoints" / f"checkpoint_{c['ad_steps']}"
    model, params, model_config, ad_receipt = load_completed_original_ad(model_root, p)
    config = ev.EvalConfig(algos=["ad", "random"], tracks=["teammate"],
        checkpoints={"ad": str(checkpoint)}, num_episodes=c["eval_episodes"],
        max_steps=c["eval_episode_horizon"], context_len=500, seed=c["eval_seed"], gpu="-1")
    config.validate()
    model = StaticADInference(model, min(config.context_len, model_config.seq_len))
    tasks = load_manifest(str(root / "dataset_inputs/train_manifest.jsonl"))
    if not tasks:
        raise ValueError("No RL training tasks for the native sanity evaluation")
    results = []
    for idx, task in enumerate(tasks):
        if task.split != "train" or task.teammate.kind != "rl":
            raise ValueError("Engineering evaluation accepts only its RL training tasks")
        rng = jax.random.fold_in(jax.random.PRNGKey(config.seed), idx)
        for algo in ("ad", "random"):
            result = ev.evaluate_ad_task(model, params, task, config, model_config, rng) if algo == "ad" else ev.evaluate_random_task(task, config, rng)
            row = dataclasses.asdict(result)
            finite_tree(row)
            if row["num_episodes"] != c["eval_episodes"] or len(row["per_episode"]) != c["eval_episodes"]:
                raise ValueError("Incomplete native reward evaluation")
            write_json(out / f"{idx:04d}_{algo}.json", row, exclusive=True)
            results.append(row)
    write_json(out / "sanity_receipt.json", {"status": "PASS", "official_test_read": False,
        "evaluation_role": "training_RL_partner_sanity", "technical_only": True,
        "ad_seed": seed, "num_results": len(results), "results": results,
        "original_ad_completion": ad_receipt, "source_binding": source_binding,
        "warning": "Not held-out generalization; no competence/effectiveness criterion is applied to returns."})


def run_worker(args):
    if platform.system() != "Linux":
        raise RuntimeError("Worker execution prohibited on macOS")
    p = read_json(args.plan)
    os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="-1")
    cpus = sorted(os.sched_getaffinity(0))[:p["cpu_limit"]]
    os.sched_setaffinity(0, cpus)
    binding = verify_source_binding(p, args.plan)
    if args.operation == "sanity" and args.ad_seed not in p["config"]["ad_seeds"]:
        raise ValueError("Sanity worker needs an AD seed listed in the frozen plan")
    receipt_path = Path(p["run_root"]) / "worker_receipts" / (
        f"{args.operation}_{args.ad_seed}_{time.time_ns()}.json")
    receipt = {"status": "RUNNING", "operation": args.operation, "ad_seed": args.ad_seed,
        "source_binding": binding, "hostname": platform.node(), "pid": os.getpid(),
        "affinity": cpus, "started_at": now(), "official_test_read": False,
        "technical_only": p["config"]["technical_only"]}
    write_json(receipt_path, receipt, exclusive=True)
    started = time.monotonic()
    try:
        if args.operation == "merge":
            merge_worker(p)
        elif args.operation == "pack":
            pack_worker(p)
        else:
            sanity_worker(p, args.ad_seed, binding)
        if verify_source_binding(p, args.plan) != binding:
            raise ValueError("Source binding changed while this worker was running")
        receipt.update(status="PASS")
    except BaseException as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(completed_at=now(), active_seconds=time.monotonic() - started)
        write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-plan")
    make.add_argument("--repo", required=True)
    make.add_argument("--run-root", required=True)
    make.add_argument("--python", default=sys.executable)
    make.add_argument("--output", required=True)
    make.add_argument("--profile", choices=["smoke", "custom"], default="smoke")
    make.add_argument("--layouts", default="grounded_coord_simple")
    make.add_argument("--partner-seeds", default="4100")
    make.add_argument("--ad-seeds", default="4200")
    make.add_argument("--num-envs", type=int, default=256)
    make.add_argument("--partner-steps", type=int)
    make.add_argument("--history-steps", type=int)
    make.add_argument("--ad-steps", type=int)
    make.add_argument("--ad-batch-size", type=int, default=8)
    make.add_argument("--teammates-per-layout", type=int, default=1)
    make.add_argument("--min-partner-return", type=float)
    make.add_argument("--eval-episodes", type=int, default=2)
    make.add_argument("--cpu-limit", type=int, default=64)
    make.add_argument("--max-active-seconds", type=float, default=28800)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--through", choices=PHASES, default="sanity")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--adopt-partner-smoke", metavar="EXTERNAL_RECEIPT_JSON")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--plan", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--operation", choices=["merge", "pack", "sanity"], required=True)
    worker.add_argument("--ad-seed", type=int)
    args = parser.parse_args()
    if args.command == "make-plan":
        make_plan(args)
    elif args.command == "inspect":
        p = read_json(args.plan)
        print(json.dumps({"plan": p, "partner_stages": partner_stages(p),
                          "manifest_stages": manifest_stages(p, args.plan),
                          "tail_stages": tail_stages(p, args.plan)}, indent=2))
    elif args.command == "run":
        run_pipeline(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
