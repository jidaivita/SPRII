#!/usr/bin/env python3
"""CaDM fixed-budget source through the exact existing D-Clean common reader.

The original dclean_external.py/dclean_panel.py files and controls are read-only.
Only the imported panel module's in-process dispatch/output path are adapted.
Source20000 + reader20000; one of the fixed three-source by three-reader cells.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.dont_write_bytecode = True


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(args):
    helper = load_module(args.data_helper, "dclean_external")
    panel = load_module(args.panel_module, "cadm_isolated_common_panel")
    pilot = load_module(args.pilot_module, "cadm_source_model_definition")
    old_root = Path(helper.ROOT).resolve()
    root = Path(args.output).resolve()
    assert root != old_root and old_root not in root.parents, "Use an independent output root."
    root.mkdir(parents=True, exist_ok=True)
    source_run = Path(args.source_run).resolve()
    source_done = helper.read(source_run / "COMPLETE.json")
    source_summary = helper.read(source_run / "SUMMARY.json")
    source_config = helper.read(source_run / "CONFIG.json")
    checkpoint = source_run / "final.pt"
    assert source_done["status"] == "COMPLETE"
    assert helper.sha(source_run / "SUMMARY.json") == source_done["summary_sha256"]
    assert helper.sha(checkpoint) == source_done["final_checkpoint_sha256"]
    assert helper.sha(source_run / "CONFIG.json") == source_summary["config_sha256"]
    assert source_config["code_sha256"] == helper.sha(args.pilot_module)
    assert source_config["data_helper_sha256"] == helper.sha(args.data_helper)
    assert source_config["method"] == "CaDM deterministic PyTorch adaptation"
    assert source_config["source_seed"] == args.seed and args.seed in (0, 1, 2)
    assert source_summary["steps"] == 20000 and source_config["steps"] == 20000
    assert source_config["test_read"] is False and source_config["physical_parameter_input"] is False
    assert source_summary["closed_loop"] is False
    # The original contract function verifies the original immutable data/code hashes.
    # Require its old protocol first: this call must not create a historical artifact.
    assert (old_root / "PROTOCOL.json").is_file()
    original_protocol_sha = helper.sha(old_root / "PROTOCOL.json")
    historical_contract = helper.contract()
    assert helper.sha(old_root / "PROTOCOL.json") == original_protocol_sha
    assert source_config["data_hashes"]["manifest.json"] == historical_contract["data_manifest_sha256"]
    assert source_config["data_hashes"]["train.npz"] == helper.sha(helper.DATA / "train.npz")
    assert source_config["data_hashes"]["val.npz"] == helper.sha(helper.DATA / "val.npz")

    reference = old_root / "results" / f"SPRII_s{args.seed}_r{args.reader}"
    reference_done = helper.read(reference / "COMPLETE.json")
    assert reference_done["status"] == "COMPLETE"
    assert helper.sha(reference / "per_case.npz") == reference_done["arrays_sha256"]
    null_root = old_root / "readers" / "null_shared"
    null_run = null_root / f"r{args.reader}"
    null_done = helper.read(null_run / "COMPLETE.json")
    assert null_done["step"] == 20000
    assert helper.sha(null_run / "final.pt") == null_done["checkpoint_sha256"]

    protocol = dict(historical_contract)
    protocol.update(
        status="FROZEN_CADM_FIXED_BUDGET", method="CaDM PyTorch adaptation, frozen-code common reader",
        source_steps=20000, source_seeds=[args.seed], reader_seeds=[0, 1, 2], reader_steps=20000,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=helper.sha(checkpoint),
        source_config_sha256=helper.sha(source_run / "CONFIG.json"),
        adaptation_code_sha256=helper.sha(__file__), panel_code_sha256=helper.sha(args.panel_module),
        historical_protocol_sha256=original_protocol_sha,
        historical_references={str(r): str(old_root / "results" / f"SPRII_s{args.seed}_r{r}") for r in range(3)},
        reference_arrays_sha256={str(r): helper.read(old_root / "results" / f"SPRII_s{args.seed}_r{r}" / "COMPLETE.json")["arrays_sha256"] for r in range(3)},
        original_control_training="Existing SPRII/NOD/FCRL source20000 reused; CaDM fixed source20000; no claim of equal FLOPs.",
        source_interface="23 normalized completed state differences/actions -> 10D context; no gamma",
        reader_interface="Exact existing Head; training-normalized CaDM 10D code padded with zeros to 64D",
        evaluation="Existing report100 systems x16 recipients =1600 cases; fixed donor mapping",
        closed_loop=False, formal_method_ranking=False,
    )
    protocol_path = root / "PROTOCOL.json"
    if protocol_path.exists():
        assert helper.read(protocol_path) == protocol, "Changed protocol; use a separate output."
    else:
        helper.write(protocol_path, protocol)

    def load_source(method, seed):
        assert method == "CaDM" and seed == args.seed
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert ck["step"] == 20000 and ck["config_sha256"] == source_summary["config_sha256"]
        model = pilot.CaDM(source_config["normalization"])
        model.load_state_dict(ck["model"], strict=True)
        for name, values in source_config["normalization"].items():
            torch.testing.assert_close(getattr(model, name), torch.tensor(values, dtype=torch.float32), rtol=0, atol=0)
        return model.cuda().eval().requires_grad_(False), {
            "checkpoint": str(checkpoint), "sha256": helper.sha(checkpoint),
            "method": "CaDM", "seed": seed, "source_steps": 20000,
            "source_config_sha256": source_summary["config_sha256"],
        }

    def encode(model, method, x, u):
        assert method == "CaDM"
        return model.encode(x, u)

    panel.ROOT = root
    panel.contract = lambda: protocol
    panel.load_source = load_source
    panel.encode = encode
    # Existing source-independent null readout is used only by evaluate(), never fit().
    link = root / "readers" / "null_shared"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        assert link.resolve() == null_root.resolve()
    else:
        assert not link.exists(), "Unexpected local null model: do not silently replace."
        link.symlink_to(null_root, target_is_directory=True)
    context = dict(old_root=old_root, root=root, reference=reference,
                   checkpoint=checkpoint, protocol=protocol, original_protocol_sha=original_protocol_sha,
                   null_checkpoint_sha=null_done["checkpoint_sha256"], seed=args.seed, reader=args.reader)
    helper.write(root / f"INSPECT_r{args.reader}.json", {
        "status": "PASS", "source_checkpoint_sha256": helper.sha(checkpoint),
        "source_steps": 20000, "reader_steps": 20000,
        "historical_protocol_sha256": original_protocol_sha,
        "exact_panel_code_sha256": helper.sha(args.panel_module),
        "controls_reused_without_training": [f"{m}_s{args.seed}_r{args.reader}" for m in ("SPRII", "NOD", "FCRL")] + [f"null_r{args.reader}"],
        "test_read": False, "closed_loop": False})
    return helper, panel, pilot, context


def real_interface_smoke(helper, panel, pilot, c):
    model, _ = panel.load_source("CaDM", c["seed"])
    before = panel.digest(model)
    sp = helper.specs(0, 1, True)
    b = helper.batch("train", sp, panel.H)
    pb = pilot.make_batch(helper.data("train"), sp, "cuda:0", future=10)
    # Checks exact observed windows through two independently implemented batch paths.
    torch.testing.assert_close(b[0], pb[0], rtol=0, atol=0)
    torch.testing.assert_close(b[1], pb[1], rtol=0, atol=0)
    with torch.no_grad():
        z = panel.encode(model, "CaDM", b[0], b[1])
        torch.testing.assert_close(z, model.encode(pb[0], pb[1]), rtol=0, atol=0)
        assert z.shape == (96, 10) and torch.isfinite(z).all()
        # Future prediction inputs are separate from code inputs and may be changed freely.
        changed_future = b[2].clone().add_(123)
        assert not torch.equal(changed_future, b[2])
        torch.testing.assert_close(z, panel.encode(model, "CaDM", b[0], b[1]), rtol=0, atol=0)
    assert panel.digest(model) == before
    helper.seed_all(2026091600 + c["reader"])
    head = panel.Head().cuda()
    initial_sha = panel.digest(head)
    historical = helper.read(c["old_root"] / "readers" / f"SPRII_s{c['seed']}" / f"r{c['reader']}" / "RUN.json")
    assert initial_sha == historical["initial_sha256"], "Head init differs from the corresponding historical reader."
    with torch.no_grad():
        slots = torch.zeros((96, 64), device="cuda:0")
        pred = head(b[0][:, -1], b[2], slots)
        assert pred.shape == (96, 4, 4) and torch.isfinite(pred).all()
        # Only unavailable actions after H1 are perturbed in the actual masking test.
        later = b[2].clone()
        later[:, 1:] += 123
        torch.testing.assert_close(pred[:, 0], head(b[0][:, -1], later, slots)[:, 0], rtol=0, atol=0)
    helper.write(c["root"] / f"INTERFACE_SMOKE_r{c['reader']}.json", {
        "status": "PASS", "source_frozen": True, "same_observed_windows": True,
        "source_normalization_exact_checkpoint": True, "history_states": 24,
        "history_actions": 23, "context_shape": list(z.shape),
        "head_initial_sha256": initial_sha, "historical_reader_initial_match": True,
        "future_action_mask": True, "physical_labels_input": False,
        "code_sha256": helper.sha(__file__), "test_read": False, "closed_loop": False})


def verify_cache(helper, panel, c):
    train, train_info = panel.load_cache("CaDM", c["seed"], "train")
    val, _ = panel.load_cache("CaDM", c["seed"], "val")
    assert train.shape == (1000, 8, 9, 10) and val.shape == (200, 8, 9, 10)
    sp = helper.specs(0, 1, True)
    slot = panel.slot(train, train_info["normalization"], sp)
    assert slot.shape == (96, 64)
    assert torch.count_nonzero(slot[:, 10:]) == 0
    norm = train_info["normalization"]
    expected = np.concatenate([train[sp[:, 0], sp[:, 3], sp[:, 4] - 23],
                               train[sp[:, 0], sp[:, 1], sp[:, 2] - 23]])
    expected = (expected - np.asarray(norm["mean"])) / np.asarray(norm["scale"])
    np.testing.assert_allclose(slot[:, :10].cpu().numpy(), expected, rtol=1e-6, atol=1e-6)
    helper.write(c["root"] / "CACHE_INTERFACE_AUDIT.json", {
        "status": "PASS", "train_shape": list(train.shape), "val_shape": list(val.shape),
        "same_system_independent_donor_swapped": True, "zero_padding_exact": True,
        "normalization": "training code statistics", "test_read": False})


def final_audit(helper, panel, c):
    result = c["root"] / "results" / f"CaDM_s{c['seed']}_r{c['reader']}"
    done = helper.read(result / "COMPLETE.json")
    assert done["status"] == "COMPLETE"
    assert helper.sha(result / "per_case.npz") == done["arrays_sha256"]
    with np.load(result / "per_case.npz", allow_pickle=False) as ours, np.load(
            c["reference"] / "per_case.npz", allow_pickle=False) as ref:
        np.testing.assert_array_equal(ours["keys"], ref["keys"])
        assert ours["keys"].shape == (1600, 5)
        np.testing.assert_array_equal(ours["null_error"], ref["null_error"])
        for arm in ("matched", "wrong", "zero", "null"):
            assert ours[arm + "_error"].shape == (1600, 4)
            assert np.isfinite(ours[arm + "_error"]).all()
    probe = helper.read(c["root"] / "cache" / f"CaDM_s{c['seed']}" / "PROBE.json")
    controls = {}
    for method in ("SPRII", "NOD", "FCRL"):
        control_root = c["old_root"] / "results" / f"{method}_s{c['seed']}_r{c['reader']}"
        d = helper.read(control_root / "COMPLETE.json")
        assert helper.sha(control_root / "per_case.npz") == d["arrays_sha256"]
        with np.load(control_root / "per_case.npz", allow_pickle=False) as a, np.load(
                result / "per_case.npz", allow_pickle=False) as b:
            np.testing.assert_array_equal(a["keys"], b["keys"])
        controls[method] = {"source_steps": 20000, "reader_steps": 20000,
                            "metrics": d["metrics"], "arrays_sha256": d["arrays_sha256"]}
    assert helper.sha(c["old_root"] / "PROTOCOL.json") == c["original_protocol_sha"]
    assert helper.sha(c["old_root"] / "readers" / "null_shared" / f"r{c['reader']}" / "final.pt") == c["null_checkpoint_sha"]
    summary = {
        "status": "COMPLETE", "method": "CaDM PyTorch adaptation + exact existing common reader",
        "source_seed": c["seed"], "reader_seed": c["reader"], "source_steps": 20000, "reader_steps": 20000,
        "horizons": [1, 4, 16, 32], "metrics": done["metrics"], "probe": probe,
        "controls_same_source_reader": controls, "cases": 1600, "report_systems": 100,
        "same_keys_and_null_errors": True, "physical_parameters_source_input": False,
        "physical_labels_probe_only": True, "formal_method_ranking": False,
        "source_budget_comparison": "All source20000; all reader20000. Different source objectives; no new control training.",
        "arrays_sha256": done["arrays_sha256"], "source_sha256": helper.sha(c["checkpoint"]),
        "protocol_sha256": helper.sha(c["root"] / "PROTOCOL.json"),
        "code_sha256": helper.sha(__file__), "test_read": False, "closed_loop": False,
    }
    helper.write(c["root"] / f"SUMMARY_r{c['reader']}.json", summary)
    helper.write(c["root"] / f"COMPLETE_r{c['reader']}.json", {"status": "COMPLETE", "summary_sha256": helper.sha(c["root"] / f"SUMMARY_r{c['reader']}.json")})
    print(json.dumps(summary, allow_nan=False), flush=True)


def run(args):
    helper, panel, pilot, c = bootstrap(args)
    if args.action in ("smoke", "all"):
        real_interface_smoke(helper, panel, pilot, c)
    if args.action in ("cache", "all"):
        panel.cache("CaDM", args.seed)
        verify_cache(helper, panel, c)
    if args.action in ("fit", "all"):
        assert helper.read(c["root"] / f"INTERFACE_SMOKE_r{c['reader']}.json")["status"] == "PASS"
        assert helper.read(c["root"] / "CACHE_INTERFACE_AUDIT.json")["status"] == "PASS"
        panel.fit("CaDM", args.seed, args.reader, "matched")
    if args.action in ("evaluate", "all"):
        panel.evaluate("CaDM", args.seed, args.reader)
    if args.action in ("probe", "all"):
        panel.probe("CaDM", args.seed)
    if args.action in ("finalize", "all"):
        final_audit(helper, panel, c)
    helper.write(c["root"] / f"EXIT_{args.action}_r{args.reader}.json", {
        "exit_code": 0, "action": args.action, "unix_time": time.time(),
        "code_sha256": helper.sha(__file__), "test_read": False})


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "smoke", "cache", "fit", "evaluate", "probe", "finalize", "all"))
    p.add_argument("--output", required=True)
    p.add_argument("--source-run", required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--reader", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--data-helper", default=str(here.parent / "missing_blocks" / "dclean_external.py"))
    p.add_argument("--panel-module", default=str(here.parent / "missing_blocks" / "dclean_panel.py"))
    p.add_argument("--pilot-module", default=str(here / "cadm_dclean_source_v2.py"))
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        run(args)
    except Exception as exc:
        # Preserve prior phase receipts and partial reader state for explicit diagnosis.
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"FAILED_{args.action}_{int(time.time())}.json"
        path.write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc(),
                                    "test_read": False}, indent=2) + "\n")
        raise
