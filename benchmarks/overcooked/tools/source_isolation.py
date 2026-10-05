"""Audit custom RL source isolation before first-pass native evaluation.

This is only the isolation preflight. It does not run evaluation, train a model,
or promote the reserved development population to an official benchmark test.
Run on the existing Linux CPU runtime from the deployed repository directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def source_row(task, role):
    native = Path(task["teammate"]["ckpt"]).resolve()
    root = native.parent
    config = read(native / "config.json")
    binding, done = read(root / "binding.json"), read(root / "completion.json")
    publication = read(native / "native_publish.json")
    seed = config["TRAIN_SEED"]
    allowed = {4600, 4601, *range(4700, 4708)} if role == "training" else {4602}
    require(seed in allowed and binding["config"]["TRAIN_SEED"] == seed, "Wrong source population role")
    require(done["status"] == "PASS" and done["complete"] is True
            and done["actual_updates"] == done["planned_updates"] == 457, "Partner training incomplete")
    require(done["native_publish_sha256"] == digest(native / "native_publish.json"), "Native publication changed")
    require(publication["actual_updates"] == 457 and publication["num_checkpoints"] == 5,
            "Native export budget changed")
    for item in publication["files"]:
        path = (root / item["path"]).resolve()
        require(path.is_relative_to(native) and path.is_file(), "Native file outside committed export")
        require(path.stat().st_size == item["size"] and digest(path) == item["sha256"], "Export hash mismatch")
    spec = task["teammate"]
    require(spec["kind"] == "rl" and spec["family"] == "ippo", "Unexpected teammate policy family")
    extra = spec["extra"]
    require(extra["checkpoint_idx"] in (range(5) if role == "training" else (3, 4)) and extra["seed_idx"] == 0
            and extra["population_idx"] == 0, "First pass uses only the fixed last two checkpoints")
    returns = read(native / "checkpoint_returns.json")
    source_return = returns["base_returns"][0][extra["checkpoint_idx"]]
    require(math.isfinite(source_return) and source_return > 20,
            "Fixed source checkpoint does not meet the original manifest's default return >20 qualification")
    return {"task_id": task["task_id"], "source_seed": seed, "checkpoint_idx": extra["checkpoint_idx"],
            "role": role, "native_root": str(native), "binding_sha256": digest(root / "binding.json"),
            "publication_sha256": digest(native / "native_publish.json"), "native_checkpoint_base_return": source_return}

