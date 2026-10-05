import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.formal_data import load_arrays
from paper_c.coupled_sled.learner import (
    PersistentJEPA, RawHistoryPredictor, _loader, _normalizers,
    _predictions_and_z, _train_jepa, _train_raw,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _system_sufficiency(arrays, prediction, norms, seed, replicates=2000):
    target = (arrays.target - norms["target_mean"]) / norms["target_std"]
    model_loss = np.mean((prediction-target) ** 2, axis=1)
    trivial_loss = np.mean(target ** 2, axis=1)
    systems = np.unique(arrays.system_index)
    advantage = np.asarray([
        np.mean(trivial_loss[arrays.system_index == system] - model_loss[arrays.system_index == system])
        for system in systems
    ])
    rng = np.random.default_rng(seed)
    boot = advantage[rng.integers(0, len(advantage), size=(replicates, len(advantage)))].mean(axis=1)
    return {
        "model_mse": float(model_loss.mean()), "trivial_mse": float(trivial_loss.mean()),
        "trivial_minus_model_mse": float(advantage.mean()),
        "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
        "passes": bool(np.quantile(boot, .025) > 0),
    }


def run(root, config_path, data_root, output_root, device_name):
    root, config_path, data_root, output_root = map(Path, (root, config_path, data_root, output_root))
    config = json.loads(config_path.read_text())
    data_receipt = json.loads((data_root / "s2_data_receipt.json").read_text())
    if data_receipt["status"] != "S2_IMMUTABLE_TRAIN_SELECT_DATA_COMPLETE" or data_receipt["exact_train_select_overlap"] != 0:
        raise RuntimeError("S2 training requires disjoint immutable train/select data")
    if any(config["access"].values()):
        raise RuntimeError("S2 training must not access evaluation, sealed, A/B, or NAD")
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    seed = config["learner_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train = load_arrays(data_root / "train_arrays.npz")
    select = load_arrays(data_root / "select_arrays.npz")
    norms = _normalizers(train)
    train_loader = _loader(train, norms, config["batch_size"], True)
    select_loader = _loader(select, norms, config["batch_size"], False)
    history_dim, query_dim, target_dim = train.history.shape[2], train.query_action.shape[1], train.target.shape[1]
    raw = RawHistoryPredictor(history_dim, query_dim, config["hidden_dim"], target_dim).to(device)
    raw_curve, raw_best = _train_raw(raw, train_loader, select_loader, config, device)
    torch.manual_seed(seed + 1)
    jepa = PersistentJEPA(history_dim, query_dim, target_dim, config).to(device)
    jepa_curve, jepa_best = _train_jepa(jepa, train_loader, select_loader, config, device)
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(raw.state_dict(), output_root / "raw_frozen.pt")
    torch.save(jepa.state_dict(), output_root / "persistent_jepa_frozen.pt")
    np.savez(output_root / "train_only_normalization.npz", **norms)
    pd.DataFrame(raw_curve).assign(model="raw").to_csv(output_root / "raw_training_curve.csv", index=False)
    pd.DataFrame(jepa_curve).assign(model="jepa").to_csv(output_root / "jepa_training_curve.csv", index=False)
    raw_prediction, _ = _predictions_and_z(raw, select, norms, config, device, False)
    jepa_prediction, _ = _predictions_and_z(jepa, select, norms, config, device, True)
    sufficiency = {
        "raw": _system_sufficiency(select, raw_prediction, norms, seed + 10),
        "jepa": _system_sufficiency(select, jepa_prediction, norms, seed + 20),
    }
    optimization = {
        "raw": {"epochs": len(raw_curve), "best_epoch": int(np.argmin([r["selection_mse"] for r in raw_curve]) + 1),
                "finite": bool(np.isfinite([r["selection_mse"] for r in raw_curve]).all())},
        "jepa": {"epochs": len(jepa_curve), "best_epoch": int(np.argmin([r["selection_mse"] for r in jepa_curve]) + 1),
                 "finite": bool(np.isfinite([r["selection_mse"] for r in jepa_curve]).all())},
    }
    optimization_normal = all(row["finite"] and row["epochs"] >= config["minimum_epochs"] for row in optimization.values())
    status = "S2_LEARNER_SUFFICIENCY_GO" if all(row["passes"] for row in sufficiency.values()) and optimization_normal else "S2_LEARNER_SUFFICIENCY_NO_GO"
    receipt = {
        "status": status, "device": device_name, "raw_select_mse": raw_best, "jepa_select_mse": jepa_best,
        "sufficiency": sufficiency, "optimization": optimization,
        "checkpoint_selection_metric": "aggregate learner-select standardized query MSE",
        "conditionality_used_for_selection": False,
        "raw_and_jepa_sample_manifest": "EXACT_SAME_IMMUTABLE_DATA",
        "data_receipt_sha256": _sha256(data_root / "s2_data_receipt.json"),
        "config_sha256": _sha256(config_path), "sealed_accessed": False, "protected_scope_2_touched": False,
        "checkpoint_hashes": {"raw": _sha256(output_root / "raw_frozen.pt"), "jepa": _sha256(output_root / "persistent_jepa_frozen.pt"),
                              "normalization": _sha256(output_root / "train_only_normalization.npz")},
    }
    (output_root / "s2_training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("config"); parser.add_argument("data_root"); parser.add_argument("output_root")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.config, args.data_root, args.output_root, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
