"""Compatible, strict loading of terminal checkpoints from upstream AD.

Upstream AD trains with clip_by_global_norm + scheduled AdamW.  Its evaluator
constructs a plain-Adam restore target, whose optimizer pytree differs.  This
adapter constructs the target with the original create_train_state function.
It does not change the policy, optimizer, observations, or saved artifacts.

The CLI only reloads existing checkpoints and compares parameter tensors; it
performs no training, environment rollout, or model prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files or not any(p.stat().st_size for p in files):
        raise ValueError(f"Checkpoint contains no actual payload files: {path}")
    return {str(p.relative_to(path.parent)): {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in files}


def parameter_receipt(params):
    """Stable hashes include tensor path, shape, dtype and exact host bytes."""
    import numpy as np
    from flax import serialization

    state = serialization.to_state_dict(params)
    leaves = []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], path + (str(key),))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, path + (str(i),))
        else:
            a = np.asarray(value)
            if a.dtype.kind not in "biufc" or not np.isfinite(a).all():
                raise ValueError(f"Non-finite or nonnumeric model parameter: {'/'.join(path)}")
            a = np.ascontiguousarray(a)
            leaves.append({"path": list(path), "shape": list(a.shape), "dtype": a.dtype.str,
                           "bytes": a.nbytes, "sha256": hashlib.sha256(a.tobytes()).hexdigest()})

    visit(state, ())
    if not leaves:
        raise ValueError("Empty model parameter tree")
    encoded = json.dumps(leaves, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "leaf_count": len(leaves),
            "tensor_bytes": sum(x["bytes"] for x in leaves), "leaves": leaves}


def _paths(checkpoint_path, config_path, expected_step):
    checkpoint = Path(checkpoint_path).resolve()
    match = re.fullmatch(r"checkpoint_(\d+)", checkpoint.name)
    if not match:
        raise ValueError("Pass an exact upstream checkpoint_<step> artifact, not its parent directory")
    named_step = int(match.group(1))
    config_file = Path(config_path).resolve() if config_path else checkpoint.parent.parent / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config_dict = json.loads(config_file.read_text())
    terminal = int(config_dict["num_steps"])
    expected = terminal if expected_step is None else int(expected_step)
    if expected < 1 or expected != terminal or named_step != expected:
        raise ValueError(f"Terminal checkpoint required: filename={named_step}, expected={expected}, config={terminal}")
    return checkpoint, config_file, config_dict, expected


def load_original_ad_checkpoint(checkpoint_path, config_path=None, *, expected_step=None):
    """Return (model, full_variables, ADConfig, receipt), rejecting silent fallback.

    `full_variables` is the unchanged original TrainState.params tree, including
    its outer `params` collection.  Pass it directly to original model.apply.
    Do not wrap it in an additional {'params': ...} dictionary.
    """
    import dataclasses
    import numpy as np
    from flax.training import checkpoints
    from benchmarks.baselines.ad.model import create_ad_model
    from benchmarks.baselines.ad.train import TrainConfig, create_train_state

    checkpoint, config_file, cfg, expected = _paths(checkpoint_path, config_path, expected_step)
    before = inventory(checkpoint)
    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in cfg.items() if k in valid_fields}
    if "obs_shape" in filtered:
        filtered["obs_shape"] = tuple(filtered["obs_shape"])
    config = TrainConfig(**filtered)
    model_config = config.to_model_config()
    model, initial_variables = create_ad_model(model_config)
    target = create_train_state(config, model, initial_variables)

    # Read without a target as well: a missing artifact must never be mistaken
    # for the initialized target returned by permissive restore APIs.
    raw = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint.parent), target=None, step=expected)
    if not isinstance(raw, dict) or "params" not in raw or "step" not in raw:
        raise ValueError("The requested checkpoint did not deserialize to an AD training state")
    raw_step = int(np.asarray(raw["step"]))
    if raw_step != expected:
        raise ValueError(f"Checkpoint is not a completed terminal run: saved state.step={raw_step}, expected={expected}")

    restored = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint.parent), target=target, step=expected)
    restored_step = int(np.asarray(restored.step))
    if restored_step != expected:
        raise ValueError(f"Restored state.step={restored_step}, expected={expected}; initializer fallback is forbidden")
    raw_params, typed_params = parameter_receipt(raw["params"]), parameter_receipt(restored.params)
    if raw_params != typed_params:
        raise ValueError("Typed reload changed saved parameter tensors")
    after = inventory(checkpoint)
    if before != after:
        raise ValueError("Checkpoint changed while it was being read")
    receipt = {
        "status": "PASS", "checkpoint": str(checkpoint), "checkpoint_step": restored_step,
        "checkpoint_is_terminal": True, "config": str(config_file), "config_sha256": file_sha(config_file),
        "checkpoint_files": after, "parameters": typed_params,
        "raw_vs_typed_parameters_exact": True,
        "restore_target": "upstream_create_train_state_clip_by_global_norm_scheduled_adamw",
        "parameters_are_full_original_variables": True,
        "new_training": False, "model_predictions_read": False, "official_test_read": False,
    }
    return model, restored.params, model_config, receipt


def load_ad_model(checkpoint_path, config_path=None):
    """Drop-in three-value replacement for eval_icrl.load_ad_model."""
    model, variables, model_config, _ = load_original_ad_checkpoint(checkpoint_path, config_path)
    return model, variables, model_config


def check_reload(checkpoint_path, config_path=None, expected_step=None):
    """Independently reload twice and require byte-identical parameter leaves."""
    started = time.monotonic()
    _, params, _, first = load_original_ad_checkpoint(checkpoint_path, config_path, expected_step=expected_step)
    first_hash = parameter_receipt(params)
    del params
    _, params, _, second = load_original_ad_checkpoint(checkpoint_path, config_path, expected_step=expected_step)
    second_hash = parameter_receipt(params)
    if first_hash != second_hash or first["checkpoint_files"] != second["checkpoint_files"]:
        raise ValueError("Independent checkpoint reloads disagree")
    first.update(independent_reload_parameters_exact=True, reload_count=2,
                 elapsed_seconds=time.monotonic() - started)
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Checkpoint initialization/reload is reserved for the Linux experiment machine")
    os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="-1")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint check receipt: {out}")
    result = check_reload(args.checkpoint, args.config, args.expected_step)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "receipt": str(out), "receipt_sha256": file_sha(out),
                      "parameters_sha256": result["parameters"]["sha256"]}))


if __name__ == "__main__":
    main()
