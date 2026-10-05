#!/usr/bin/env python3
"""Fixed-budget CaDM deterministic PyTorch adaptation for D-Clean.

This is not untouched official reproduction and is not a control experiment.
Uses the audited dclean_external.py train/val loader, 48 systems x two legal
histories, a context shared over ten teacher-forced future transitions, and
the published forward + 0.5 backward prediction objective. No physical
parameter or system identity enters the model. No test data are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

OFFICIAL_COMMIT = "38c11a58d959bfd597f9323e58f28b17f6bf4fd9"
HISTORY_TRANSITIONS = 23
FUTURE_TRANSITIONS = 10
STATE_DIM, ACTION_DIM, CONTEXT_DIM = 4, 2, 10
SOURCE_SYSTEMS_PER_BATCH = 48
BACK_COEFF = 0.5
LR = 0.001
DYNAMICS_DECAYS = (0.000025, 0.00005, 0.000075, 0.000075, 0.0001)
CONTEXT_DECAYS = (0.000025, 0.00005, 0.000075, 0.000075)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_helper(path, data_root=None):
    spec = importlib.util.spec_from_file_location("cadm_dclean_data_helper", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    if data_root is not None:
        helper.DATA = Path(data_root)
    # Do not call helper.contract(): it includes unrelated NOD model paths.
    return helper


def source_evidence(args):
    receipt_path = Path(args.source_manifest)
    receipt = json.loads(receipt_path.read_text())["cadm"]
    receipt["files"] = json.loads((receipt_path.parent / "third_party/cadm_files.sha256.json").read_text())
    assert receipt["commit"] == OFFICIAL_COMMIT
    evidence = {
        "repository": receipt["url"], "commit": OFFICIAL_COMMIT,
        "source_manifest_sha256": sha(receipt_path), "official_file_hashes": receipt["files"],
        "official_files_verified_locally": False,
    }
    if args.official_root:
        root = Path(args.official_root)
        for rel, expected in receipt["files"].items():
            assert sha(root / rel) == expected, rel
        evidence["official_files_verified_locally"] = True
    return evidence


def normalizations(bank):
    """Only train states/actions; all quantities are position-wise in history."""
    x = np.asarray(bank["states"], dtype=np.float64)
    u = np.asarray(bank["actions"], dtype=np.float64)
    delta = x[:, :, 1:] - x[:, :, :-1]
    # Match legal t values of the existing helper's 48-system sampling.
    times = np.arange(23, 48)
    hist_idx = times[:, None] + np.arange(-23, 0)[None]
    history_delta = delta[:, :, hist_idx, :].reshape(-1, 23 * STATE_DIM)
    history_action = u[:, :, hist_idx, :].reshape(-1, 23 * ACTION_DIM)
    current = x[:, :, times, :].reshape(-1, STATE_DIM)
    action = u[:, :, times, :].reshape(-1, ACTION_DIM)
    target = delta[:, :, times, :].reshape(-1, STATE_DIM)
    out = {}
    for name, values in (("state", current), ("action", action),
                         ("delta", target), ("back_delta", -target),
                         ("context_delta", history_delta), ("context_action", history_action)):
        out[name + "_mean"] = values.mean(axis=0).tolist()
        out[name + "_std"] = values.std(axis=0).clip(1e-8).tolist()
    return out


def make_batch(bank, specs, device, future=FUTURE_TRANSITIONS):
    """The model-facing tuple contains only observed histories and prediction data."""
    specs = np.asarray(specs, dtype=np.int64)
    assert specs.ndim == 2 and specs.shape[1] == 5
    assert np.all(specs[:, 1] != specs[:, 3])
    parts = []
    for rollout_col, time_col in ((1, 2), (3, 4)):
        s, r, t = specs[:, 0, None], specs[:, rollout_col, None], specs[:, time_col, None]
        assert np.all(t >= HISTORY_TRANSITIONS)
        assert np.all(t + future < bank["states"].shape[2])
        history_states = bank["states"][s, r, t + np.arange(-23, 1)]
        history_actions = bank["actions"][s, r, t + np.arange(-23, 0)]
        future_states = bank["states"][s, r, t + np.arange(future + 1)]
        future_actions = bank["actions"][s, r, t + np.arange(future)]
        assert np.array_equal(history_states[:, -1], future_states[:, 0])
        parts.append((history_states, history_actions, future_states, future_actions))
    return tuple(torch.as_tensor(np.concatenate([p[i] for p in parts]),
                                 dtype=torch.float32, device=device) for i in range(4))


class Dense(nn.Linear):
    def reset_parameters(self):
        std = 1 / (2 * math.sqrt(self.in_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.bias)


class Context(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [23 * (STATE_DIM + ACTION_DIM), 256, 128, 64, CONTEXT_DIM]
        self.layers = nn.ModuleList([Dense(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def regularizer(self):
        return sum(0.5 * c * layer.weight.square().sum()
                   for c, layer in zip(CONTEXT_DECAYS, self.layers))


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [STATE_DIM + ACTION_DIM + CONTEXT_DIM, 200, 200, 200, 200]
        self.layers = nn.ModuleList([Dense(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.mu = Dense(200, STATE_DIM)
        # Official deterministic implementation retains and regularizes this head,
        # even though only mu is used for its prediction loss/inference.
        self.logvar = Dense(200, STATE_DIM)

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x))
        return self.mu(x)

    def regularizer(self):
        hidden = sum(0.5 * c * layer.weight.square().sum()
                     for c, layer in zip(DYNAMICS_DECAYS[:-1], self.layers))
        return hidden + 0.5 * DYNAMICS_DECAYS[-1] * (
            self.mu.weight.square().sum() + self.logvar.weight.square().sum())


class CaDM(nn.Module):
    def __init__(self, norm):
        super().__init__()
        for name, value in norm.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.context = Context()
        self.forward_model = Dynamics()
        self.backward_model = Dynamics()

    def normalize(self, value, name):
        return (value - getattr(self, name + "_mean")) / (getattr(self, name + "_std") + 1e-10)

    def encode(self, history_states, history_actions):
        assert history_states.shape[1:] == (24, STATE_DIM)
        assert history_actions.shape[1:] == (23, ACTION_DIM)
        delta = (history_states[:, 1:] - history_states[:, :-1]).flatten(1)
        action = history_actions.flatten(1)
        return self.context(torch.cat([self.normalize(delta, "context_delta"),
                                       self.normalize(action, "context_action")], dim=-1))

    def predict_delta(self, state, action, context):
        x = torch.cat([self.normalize(state, "state"), self.normalize(action, "action"), context], -1)
        return self.forward_model(x) * (self.delta_std + 1e-10) + self.delta_mean

    def objective(self, history_states, history_actions, future_states, future_actions):
        assert future_states.shape[1] == FUTURE_TRANSITIONS + 1
        z = self.encode(history_states, history_actions)
        shared_z = z[:, None].expand(-1, FUTURE_TRANSITIONS, -1)
        x, nxt = future_states[:, :-1], future_states[:, 1:]
        action = self.normalize(future_actions, "action")
        forward = self.forward_model(torch.cat([self.normalize(x, "state"), action, shared_z], -1))
        backward = self.backward_model(torch.cat([self.normalize(nxt, "state"), action, shared_z], -1))
        forward_loss = (forward - self.normalize(nxt - x, "delta")).square().mean()
        backward_loss = (backward - self.normalize(x - nxt, "back_delta")).square().mean()
        l2 = self.context.regularizer() + self.forward_model.regularizer() + self.backward_model.regularizer()
        total = forward_loss + BACK_COEFF * backward_loss + l2
        return total, {"forward_normalized_delta_mse": forward_loss,
                       "backward_normalized_delta_mse": backward_loss, "l2": l2,
                       "context_std_mean": z.std(0, unbiased=False).mean()}

    @torch.no_grad()
    def rollout(self, history_states, history_actions, actions):
        z = self.encode(history_states, history_actions)
        x = history_states[:, -1]
        result = []
        for u in actions.unbind(1):
            x = x + self.predict_delta(x, u, z)
            result.append(x)
        return torch.stack(result, 1)


def development_specs(bank, seed):
    ids = list(map(int, bank["system_ids"]))
    ranked = sorted(ids, key=lambda v: hashlib.sha256(
        f"sprii-dclean-select-report-20260916:{v}".encode()).digest())
    selected = set(ranked[:100])
    rng = np.random.default_rng(202609240100 + seed)
    rows = []
    for _ in range(2):
        for index, sid in enumerate(ids):
            if sid not in selected:
                continue
            a = int(rng.integers(8))
            b = (a + int(rng.integers(1, 8))) % 8
            rows.append([index, a, int(rng.integers(23, 48)), b, int(rng.integers(23, 48))])
    return np.asarray(rows, dtype=np.int64), sorted(selected)


@torch.no_grad()
def evaluate(model, bank, specs, device):
    model.eval()
    totals, count, codes = {}, 0, []
    for start in range(0, len(specs), SOURCE_SYSTEMS_PER_BATCH):
        b = make_batch(bank, specs[start:start + SOURCE_SYSTEMS_PER_BATCH], device, future=16)
        hs, ha, fs, fa = b
        size = len(hs)
        _, terms = model.objective(hs, ha, fs[:, :11], fa[:, :10])
        pred = model.rollout(hs, ha, fa)
        for key in ("forward_normalized_delta_mse", "backward_normalized_delta_mse"):
            totals[key] = totals.get(key, 0.0) + float(terms[key]) * size
        for horizon in (1, 4, 16):
            error = pred[:, horizon - 1] - fs[:, horizon]
            for suffix, val in (("raw_state_mse", error.square().mean()),
                                ("normalized_state_mse", (error / model.state_std).square().mean())):
                key = f"h{horizon}_{suffix}"
                totals[key] = totals.get(key, 0.0) + float(val) * size
        codes.append(model.encode(hs, ha).cpu())
        count += size
    result = {key: value / count for key, value in totals.items()}
    z = torch.cat(codes)
    result.update(cases=count, context_std=z.std(0, unbiased=False).tolist(),
                  split="historical_validation_selection_half", test_read=False,
                  closed_loop=False, evaluation="prediction; fixed-context autoregressive horizons")
    assert all(math.isfinite(v) for v in result.values() if isinstance(v, float))
    return result


def smoke(model, bank, specs, device):
    model.train()
    b = make_batch(bank, specs, device)
    assert len(specs) == 48 and len(b[0]) == 96
    assert len(np.unique(np.asarray(specs)[:, 0])) == 48
    loss, terms = model.objective(*b)
    assert torch.isfinite(loss)
    expected = terms["forward_normalized_delta_mse"] + .5 * terms["backward_normalized_delta_mse"] + terms["l2"]
    assert torch.equal(loss, expected)
    loss.backward()
    norms = {}
    for name in ("context", "forward_model", "backward_model"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norms[name] = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert norms[name] > 0, name
    # The model's encoder has no future/metadata argument; future perturbations
    # cannot change context. Also verify frozen evaluation and batch independence.
    model.eval()
    with torch.no_grad():
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        z = model.encode(b[0], b[1])
        assert z.shape == (96, 10)
        assert torch.equal(z, model.encode(b[0], b[1]))
        permutation = torch.arange(95, -1, -1, device=device)
        permuted = model.encode(b[0][permutation], b[1][permutation])
        assert torch.allclose(z[permutation], permuted, atol=1e-6, rtol=1e-5)
        pred = model.rollout(b[0], b[1], b[3])
        assert pred.shape == (96, 10, 4) and torch.isfinite(pred).all()
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    model.zero_grad(set_to_none=True)
    return {"status": "PASS", "loss": float(loss), "gradient_norms": norms,
            "shapes": [list(v.shape) for v in b], "context_shape": list(z.shape),
            "loss_terms": {k: float(v) for k, v in terms.items()},
            "parameters": sum(p.numel() for p in model.parameters()),
            "context_input": "23 completed state differences and 23 observed actions only",
            "physical_labels_input": False, "future_encoder_input": False,
            "frozen_eval": True, "test_read": False, "closed_loop": False}


def inspect_data(args):
    helper = load_helper(args.data_helper, args.data_root)
    train, val = helper.data("train"), helper.data("val")
    assert not set(map(int, train["system_ids"])) & set(map(int, val["system_ids"]))
    assert not set(train["rollout_seeds"].flat) & set(val["rollout_seeds"].flat)
    norm = normalizations(train)
    dev_specs, dev_ids = development_specs(val, args.seed)
    config = {
        "method": "CaDM deterministic PyTorch adaptation", "official_untouched_reproduction": False,
        "official": source_evidence(args), "code_sha256": sha(__file__),
        "data_helper": str(Path(args.data_helper).resolve()), "data_helper_sha256": sha(args.data_helper),
        "data_root": str(helper.DATA),
        "data_hashes": {name: sha(helper.DATA / name) for name in ("manifest.json", "train.npz", "val.npz")},
        "source_seed": args.seed, "steps": args.steps, "source_systems_per_batch": 48,
        "independent_trajectories_per_system": 2, "effective_history_batch": 96,
        "history_states": 24, "history_transitions": 23, "future_transitions": 10,
        "context_dim": 10, "context_hidden": [256, 128, 64], "context_activation": "relu",
        "dynamics_hidden": [200] * 4, "dynamics_activation": "swish",
        "ensemble_size": 1, "deterministic": True, "backward_coefficient": .5,
        "optimizer": "Adam", "learning_rate": LR, "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
        "weight_decay": "explicit 0.5 sum(weight squared) with official per-layer coefficients",
        "context_decays": CONTEXT_DECAYS, "dynamics_decays": DYNAMICS_DECAYS,
        "gradient_clipping": None, "initialization": "official truncated normal +/-2 std, std=1/(2 sqrt fan_in)",
        "normalization": norm, "normalization_data": "train-only all legal sampled-time windows",
        "selection_system_ids": dev_ids, "evaluation_cases": len(dev_specs) * 2,
        "evaluation_every": 250, "primary_checkpoint": "fixed final20000",
        "development_best_checkpoint": "separately retained, minimum dev forward normalized delta MSE",
        "adaptations": ["D-Clean 4D state/2D action", "K=23 for existing 24-state common interface",
                        "48 systems x two independent histories instead of official transition batch256",
                        "fixed offline data bank and 20000 updates instead of online model-based data collection",
                        "train-only fixed statistics instead of growing on-policy statistics"],
        "physical_parameter_input": False, "test_read": False, "closed_loop": False,
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "device": args.device,
    }
    return helper, train, val, dev_specs, config


def synthetic_smoke(args):
    rng = np.random.default_rng(args.seed)
    bank = {"states": rng.normal(size=(1000, 8, 64, 4)).astype("float32"),
            "actions": rng.normal(size=(1000, 8, 64, 2)).astype("float32")}
    helper = load_helper(args.data_helper, args.data_root)
    specs = helper.specs(args.seed, 1)
    seed_all(args.seed)
    model = CaDM(normalizations(bank)).to(args.device)
    result = smoke(model, bank, specs, args.device)
    result.update(data="synthetic engineering tensors only", real_data_read=False,
                  real_data_smoke=False, code_sha256=sha(__file__))
    write(Path(args.output) / "SYNTHETIC_SMOKE.json", result)
    print(json.dumps(result), flush=True)


def save_checkpoint(path, model, optimizer, step, config_sha):
    temp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config_sha256": config_sha, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temp)
    temp.replace(path)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "synthetic-smoke":
        synthetic_smoke(args)
        return
    if args.action == "train":
        assert args.steps == 20000, "Formal source budget is fixed at 20000 updates."
        assert not (out / "RUN.json").exists(), "Existing run: preserve it; do not restart/overwrite."
    helper, train, val, dev_specs, config = inspect_data(args)
    config_path = out / "CONFIG.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(config)), "Config changed; use another output."
    else:
        write(config_path, config)
    write(out / "DEVELOPMENT_SPECS.json", {"specs": dev_specs.tolist(),
          "system_ids": config["selection_system_ids"], "test_read": False})
    write(out / "INSPECT.json", {"status": "PASS", "config_sha256": sha(config_path),
          "train_shape": list(train["states"].shape), "val_shape": list(val["states"].shape),
          "development_spec_sha256": sha(out / "DEVELOPMENT_SPECS.json"), "test_read": False})
    if args.action == "inspect":
        print(json.dumps({"status": "INSPECT_PASS", "output": str(out)}), flush=True)
        return
    seed_all(args.seed)
    model = CaDM(config["normalization"]).to(args.device)
    result = smoke(model, train, helper.specs(args.seed, 1), args.device)
    result.update(real_data_smoke=True, config_sha256=sha(config_path))
    write(out / "SMOKE.json", result)
    if args.action == "smoke":
        print(json.dumps(result), flush=True)
        return
    # Recreate a clean seed/init after smoke; smoke never supplies training updates.
    seed_all(args.seed)
    model = CaDM(config["normalization"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8)
    write(out / "RUN.json", {"pid": os.getpid(), "start_unix": time.time(),
                             "config_sha256": sha(config_path), "parameters": result["parameters"]})
    initial = evaluate(model, val, dev_specs, args.device)
    write(out / "DEVELOPMENT_INITIAL.json", initial)
    best, best_step, best_evaluation = math.inf, None, None
    begin = time.monotonic()
    evaluations = []
    with (out / "train.jsonl").open("x") as log:
        for step in range(1, args.steps + 1):
            model.train()
            batch = make_batch(train, helper.specs(args.seed, step), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.objective(*batch)
            assert torch.isfinite(loss), (step, "loss")
            loss.backward()
            norm2 = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)
            assert torch.isfinite(norm2), (step, "gradient")
            optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"step": step, "loss": float(loss), "gradient_norm": float(norm2.sqrt()),
                       "seconds": time.monotonic() - begin,
                       **{k: float(v) for k, v in terms.items()}}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                write(out / "PROGRESS.json", row)
                print(json.dumps(row), flush=True)
            if step % 250 == 0:
                evaluation = evaluate(model, val, dev_specs, args.device)
                evaluation["step"] = step
                evaluations.append(evaluation)
                write(out / f"DEVELOPMENT_STEP_{step:04d}.json", evaluation)
                score = evaluation["forward_normalized_delta_mse"]
                if score < best:
                    best, best_step, best_evaluation = score, step, evaluation
                    save_checkpoint(out / "best_development.pt", model, optimizer, step, sha(config_path))
                if step == args.steps:
                    save_checkpoint(out / "final.pt", model, optimizer, step, sha(config_path))
    summary = {"method": config["method"], "status": "COMPLETE", "steps": args.steps,
               "seed": args.seed, "config_sha256": sha(config_path),
               "initial_development": initial, "final_development": evaluations[-1],
               "best_development_step": best_step, "best_development": best_evaluation,
               "final_sha256": sha(out / "final.pt"), "best_sha256": sha(out / "best_development.pt"),
               "seconds": time.monotonic() - begin, "test_read": False, "closed_loop": False,
               "interpretation": "Fixed final20000 source for the common-reader comparison; no MPC return."}
    write(out / "SUMMARY.json", summary)
    write(out / "COMPLETE.json", {"status": "COMPLETE", "summary_sha256": sha(out / "SUMMARY.json"),
                                    "final_checkpoint_sha256": summary["final_sha256"], "test_read": False})
    print(json.dumps(summary), flush=True)


def parser():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "smoke", "synthetic-smoke", "train"))
    p.add_argument("--output", required=True)
    p.add_argument("--data-helper", default=str(here.parent / "missing_blocks" / "dclean_external.py"))
    p.add_argument("--data-root")
    p.add_argument("--source-manifest", default=str(here / "sources.json"))
    p.add_argument("--official-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:0")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        run(args)
        write(Path(args.output) / f"EXIT_{args.action}.json",
              {"exit_code": 0, "action": args.action, "code_sha256": sha(__file__),
               "unix_time": time.time(), "test_read": False})
    except Exception as exc:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        write(out / f"FAILED_{args.action}_{int(time.time())}.json",
              {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
               "code_sha256": sha(__file__), "test_read": False})
        raise
