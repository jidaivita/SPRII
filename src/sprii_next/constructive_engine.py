"""Frozen-protocol runner for the constructive formation/use loop.

The historical runner remains untouched.  This runner deliberately exposes a
small interface around any provider implementing ``training_batch`` and
``evaluation_batches``.  Development batches are the only data used for
checkpoint selection; self/wrong interventions are evaluated after selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .constructive import RecipientBase, ResidualReader
from .io import code_hashes, digest, sha, write


def _tensors(batch, device):
    return [
        torch.as_tensor(batch.query, dtype=torch.float32, device=device),
        torch.as_tensor(batch.persistent, dtype=torch.float32, device=device),
        torch.as_tensor(batch.actions, dtype=torch.float32, device=device),
        torch.as_tensor(batch.mask, dtype=torch.float32, device=device),
        torch.as_tensor(batch.horizon_index, dtype=torch.long, device=device),
    ]


def _forward(model, batch, device):
    x = _tensors(batch, device)
    if model.arm == "oracle_route":
        theta = torch.as_tensor(batch.theta, dtype=torch.float32, device=device)
        return model(*x, theta=theta)
    return model(*x)


def _mse(model, provider, device, *, split="validation"):
    # Providers expose development validation only.  This function never
    # receives intervention labels and therefore cannot tune specificity.
    total = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in provider.evaluation_batches():
            pred = _forward(model, batch, device).cpu().numpy().astype(np.float64)
            target = np.asarray(batch.target, np.float64)
            total += float(np.square(pred - target).sum())
            count += int(target.size)
    if count == 0:
        raise ValueError("empty development evaluation")
    return total / count


def _save_state(path, model, step, dev_mse):
    torch.save({"model": model.state_dict(), "step": int(step), "dev_mse": float(dev_mse)}, path)


def train_b0(provider, *, seed, steps, batch_size, learning_rate, device):
    """Train query-only B0 and return state plus a development receipt."""
    torch.manual_seed(seed)
    base = RecipientBase(seed).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=learning_rate)
    best = None
    for step in range(1, steps + 1):
        batch = provider.training_batch(seed, step - 1, batch_size)
        # The base is intentionally trained without persistent input.
        q, _, actions, mask, horizon = _tensors(batch, device)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        opt.zero_grad(set_to_none=True)
        loss = (base(q, actions, mask, horizon) - target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite B0 loss")
        loss.backward(); opt.step()
        if step == steps or step % max(1, steps // 10) == 0:
            dev = _mse_b0(base, provider, device)
            if best is None or dev < best["dev_mse"]:
                best = {"step": step, "dev_mse": dev,
                        "state": {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}}
    if best is None:
        raise RuntimeError("B0 did not produce a checkpoint")
    return best


def _mse_b0(base, provider, device):
    base.eval(); total = 0.0; count = 0
    with torch.inference_mode():
        for batch in provider.evaluation_batches():
            q, _, actions, mask, horizon = _tensors(batch, device)
            pred = base(q, actions, mask, horizon).cpu().numpy().astype(np.float64)
            target = np.asarray(batch.target, np.float64)
            total += float(np.square(pred - target).sum()); count += int(target.size)
    return total / max(count, 1)


def train_route(provider, *, arm, seed, base_state, weights, steps, batch_size,
                learning_rate, device, output):
    """Train one matched arm, selecting only by development MSE."""
    model = ResidualReader(arm, seed, base_state=base_state, w=weights).to(device)
    # B0 must stay frozen in every route arm.
    if any(p.requires_grad for p in model.base.parameters()):
        raise AssertionError("B0 parameters are not frozen")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=learning_rate)
    best = None
    for step in range(1, steps + 1):
        batch = provider.training_batch(seed, step - 1, batch_size)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        opt.zero_grad(set_to_none=True)
        loss = (_forward(model, batch, device) - target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite {arm} loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if step == steps or step % max(1, steps // 10) == 0:
            dev = _mse(model, provider, device)
            if best is None or dev < best["dev_mse"]:
                best = {"step": step, "dev_mse": dev,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if best is None:
        raise RuntimeError(f"{arm} did not produce a checkpoint")
    model.load_state_dict(best["state"], strict=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "selected_step": best["step"],
                "dev_mse": best["dev_mse"], "checkpoint_rule": "development_mse_only"},
               output / "checkpoint.pt")
    receipt = dict(schema="sprii-next.constructive-run.v1", arm=arm, seed=seed,
                   selected_step=best["step"], dev_mse=best["dev_mse"],
                   checkpoint_rule="development_mse_only", test_read=False,
                   architecture=model.architecture(), weights=np.asarray(weights).tolist(),
                   code_sha256=code_hashes())
    write(output / "RUN.json", receipt)
    return model, receipt


def run_arms(provider, *, seed, weights, b0_steps=1000, route_steps=1000,
             batch_size=128, learning_rate=1e-3, device="cuda:0", output="runs"):
    """Run B0 then all constructive arms against one immutable B0 state."""
    device = torch.device(device)
    b0 = train_b0(provider, seed=seed, steps=b0_steps, batch_size=batch_size,
                  learning_rate=learning_rate, device=device)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    torch.save(b0["state"], root / "b0_checkpoint.pt")
    write(root / "B0.json", dict(selected_step=b0["step"], dev_mse=b0["dev_mse"],
                                  checkpoint_rule="development_mse_only", test_read=False,
                                  checkpoint_sha256=sha(root / "b0_checkpoint.pt")))
    results = []
    for arm in ("m1", "m1_phys", "m2", "m2_phys", "oracle_route"):
        _, receipt = train_route(provider, arm=arm, seed=seed, base_state=b0["state"],
                                 weights=weights, steps=route_steps, batch_size=batch_size,
                                 learning_rate=learning_rate, device=device,
                                 output=root / arm)
        results.append(receipt)
    write(root / "SUMMARY.json", dict(schema="sprii-next.constructive-summary.v1",
        b0=dict(step=b0["step"], dev_mse=b0["dev_mse"]), arms=results,
        test_read=False, checkpoint_selection="development_mse_only"))
    return results

