"""Frozen matched-design layer-response extraction for Paper C Stage 1.

The ``freeze`` command writes row manifests and hashes before any Stage-1
learner forward pass.  Extraction never reads the completed formal learner
result tables: it reconstructs inputs from frozen systems/configs and runs the
frozen checkpoints directly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_c.coupled_sled.formal_data import arrays_from_sample_manifest, load_system_pool
from paper_c.coupled_sled.learner import (
    CANONICAL_ARCHITECTURE_TAG,
    _normalized,
    build_persistent_jepa,
)
from paper_c.coupled_sled.manifests import load_spec
from paper_c.coupled_sled.p3_learner import _stable_seed
from paper_c.coupled_sled.waveforms import history_probe_bank, query_bank
from paper_c.swimmer.lqa_evaluate import build_unique_learner_rows
from paper_c.swimmer.lqa_formal import formal_context_table
from paper_c.swimmer.lqa_prospective import _load_jepa


STATUS_FROZEN = "STAGE1_SYMPTOM_FEATURE_MANIFEST_FROZEN"
STATUS_SHARD = "STAGE1_SYMPTOM_EXTRACTION_SHARD_COMPLETE"
STATUS_MERGED = "STAGE1_SYMPTOM_EXTRACTION_COMPLETE"
KEYS = ["system_index", "realization", "history_index", "candidate_index", "query_index"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _root_path(root: Path, value: str) -> Path:
    return (root / value).resolve()


def choose_context(system_index: int, realizations: int, histories: int, salt: str) -> tuple[int, int]:
    choices = []
    for realization in range(realizations):
        for history in range(histories):
            digest = hashlib.sha256(f"{salt}|{system_index}|{realization}|{history}".encode()).hexdigest()
            choices.append((digest, realization, history))
    _, realization, history = min(choices)
    return realization, history


def _coupled_paths(root: Path) -> dict[str, Path]:
    return {
        "base_spec": root / "configs/phase2_trackA_spec.yaml",
        "formal_spec": root / "configs/phase2_trackA_formal_v1.yaml",
        "p3_spec": root / "configs/coupled_theory_validation_p3_v1.json",
        "system_pool": root / "runs/formal/coupled_theory_validation_p3_v1/prospective_systems.json",
        "pair_manifest": root / "runs/formal/coupled_theory_validation_p3_v1/pairs/PAIR_MANIFEST_FROZEN.csv",
        "pair_receipt": root / "runs/formal/coupled_theory_validation_p3_v1/pairs/PAIR_MANIFEST_FROZEN_RECEIPT.json",
        "checkpoint": root / "runs/formal/phase2_track_a_life_or_death_v1/models/persistent_jepa_formal_frozen.pt",
        "normalization": root / "runs/formal/phase2_track_a_life_or_death_v1/models/train_only_normalization.npz",
        "model_receipt": root / "runs/formal/phase2_track_a_life_or_death_v1/models/formal_training_receipt.json",
    }


def _articulated_paths(root: Path) -> dict[str, Path]:
    return {
        "reference_config": root / "configs/articulated_lqa_prospective_v1.json",
        "finalization_config": root / "configs/articulated_lqa_finalization_v1.json",
        "pair_manifest": root / "runs/formal/articulated_lqa_prospective_v1/postreference/finalized_512/formal_pair_manifest_frozen.csv.gz",
        "pair_receipt": root / "runs/formal/articulated_lqa_prospective_v1/postreference/finalized_512/formal_finalization_receipt.json",
        "checkpoint": root / "runs/formal/swimmer_s2r_learner_v1/models/persistent_jepa_frozen.pt",
        "normalization": root / "runs/formal/swimmer_s2r_learner_v1/models/train_only_normalization.npz",
        "model_receipt": root / "runs/formal/swimmer_s2r_learner_v1/models/s2_training_receipt.json",
    }


def coupled_rows(root: Path, cfg: dict) -> pd.DataFrame:
    paths = _coupled_paths(root)
    spec = load_spec(paths["base_spec"])
    development = spec["development_v0_1"]
    dt = spec["dynamics"]["reference_dt_s"]
    histories = sorted(history_probe_bank(development["experience_duration_s"], dt, development["history_energy"]))
    queries = sorted(query_bank(development["query_duration_s"], dt, development["query_energy"], tuple(development["query_chirp_hz"])))
    system_ids, _ = load_system_pool(paths["system_pool"])
    design = cfg["sample_design"]["coupled"]
    if len(system_ids) != int(design["systems"]) or len(histories) != 6 or len(queries) != 6:
        raise RuntimeError("Coupled frozen axes changed")
    rows = []
    for system_index, system_id in enumerate(system_ids):
        realization, history_index = choose_context(
            system_index, int(design["realizations_available"]), int(design["histories_available"]), str(design["context_salt"])
        )
        for query_index, query_id in enumerate(queries):
            seed = _stable_seed("coupled-p3-pair-crn-v1", system_id, realization, histories[history_index], query_id)
            common = {
                "system_index": system_index, "system_id": system_id,
                "realization": realization, "history_index": history_index,
                "anchor_index": history_index, "anchor_probe": histories[history_index],
                "query_index": query_index, "query": query_id, "group_seed": seed,
                "wrong_system_index": -1,
            }
            rows.append({**common, "candidate_index": -1, "condition_index": 0, "condition": "anchor", "added_probe": "NONE"})
            for candidate_index, candidate_id in enumerate(histories):
                rows.append({**common, "candidate_index": candidate_index, "condition_index": 1, "condition": "candidate", "added_probe": candidate_id})
    table = pd.DataFrame(rows)
    return _verify_rows(table, int(design["systems"]), "coupled")


def articulated_rows(root: Path, cfg: dict) -> pd.DataFrame:
    paths = _articulated_paths(root)
    reference = json.loads(paths["reference_config"].read_text())
    contexts = formal_context_table(reference).iloc[: int(cfg["sample_design"]["articulated"]["systems"])]
    rows = []
    for context in contexts.itertuples(index=False):
        for query_index in range(6):
            common = {
                "system_index": int(context.system_index), "realization": int(context.realization),
                "history_index": int(context.history_index), "query_index": query_index,
            }
            rows.append({**common, "candidate_index": -1, "condition": "anchor"})
            for candidate_index in range(6):
                rows.append({**common, "candidate_index": candidate_index, "condition": "candidate"})
    table = pd.DataFrame(rows)
    return _verify_rows(table, int(cfg["sample_design"]["articulated"]["systems"]), "articulated")


def _verify_rows(table: pd.DataFrame, systems: int, environment: str) -> pd.DataFrame:
    if table.duplicated(KEYS).any():
        raise RuntimeError(f"{environment} row manifest has duplicate keys")
    if table.system_index.nunique() != systems or set(table.candidate_index.unique()) != {-1, 0, 1, 2, 3, 4, 5}:
        raise RuntimeError(f"{environment} row manifest has incomplete system/candidate coverage")
    counts = table.groupby(["system_index", "query_index"]).size()
    if len(counts) != systems * 6 or not np.all(counts.to_numpy() == 7):
        raise RuntimeError(f"{environment} row manifest is not one baseline plus six candidates per query")
    return table.sort_values(KEYS).reset_index(drop=True)


def freeze(root: Path, config_path: Path, output_root: Path) -> dict:
    root, config_path, output_root = Path(root).resolve(), Path(config_path).resolve(), Path(output_root).resolve()
    if (output_root / "feature_manifest_frozen.json").exists():
        raise RuntimeError("Stage-1 feature manifest already frozen; refusing overwrite")
    cfg = json.loads(config_path.read_text())
    protocol_path = _root_path(root, cfg["protocol"])
    implementation_path = Path(__file__).resolve()
    sources = {"config": config_path, "protocol": protocol_path, "implementation": implementation_path}
    sources.update({f"coupled_{key}": value for key, value in _coupled_paths(root).items()})
    sources.update({f"articulated_{key}": value for key, value in _articulated_paths(root).items()})
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    coupled_receipt = json.loads(sources["coupled_pair_receipt"].read_text())
    articulated_receipt = json.loads(sources["articulated_pair_receipt"].read_text())
    if coupled_receipt.get("learner_outcomes_accessed") is not False:
        raise RuntimeError("Coupled pair receipt is not outcome-blind")
    if articulated_receipt.get("learner_outcomes_read") is not False:
        raise RuntimeError("Articulated pair receipt is not outcome-blind")
    if coupled_receipt.get("pair_manifest_sha256") != sha256(sources["coupled_pair_manifest"]):
        raise RuntimeError("Coupled pair manifest hash mismatch")
    if articulated_receipt.get("pair_manifest_sha256") != sha256(sources["articulated_pair_manifest"]):
        raise RuntimeError("Articulated pair manifest hash mismatch")

    coupled = coupled_rows(root, cfg)
    articulated = articulated_rows(root, cfg)
    coupled_path = output_root / "frozen" / "coupled_stage1_rows.csv.gz"
    articulated_path = output_root / "frozen" / "articulated_stage1_rows.csv.gz"
    _atomic_csv(coupled_path, coupled)
    _atomic_csv(articulated_path, articulated)
    receipt = {
        "schema_version": "1.0", "status": STATUS_FROZEN,
        "frozen_at_unix": time.time(),
        "environments": {
            "coupled": {"systems": 704, "rows": len(coupled), "row_manifest": str(coupled_path.relative_to(root)), "row_manifest_sha256": sha256(coupled_path)},
            "articulated": {"systems": 512, "rows": len(articulated), "row_manifest": str(articulated_path.relative_to(root)), "row_manifest_sha256": sha256(articulated_path)},
        },
        "source_hashes": {str(path.relative_to(root)): sha256(path) for path in sources.values()},
        "formal_learner_result_files_read": False,
        "stage1_learner_forward_runs": 0,
        "models_retrained": False, "sealed_accessed": False, "protected_scope_1_accessed": False, "protected_scope_2_accessed": False,
    }
    _atomic_text(output_root / "feature_manifest_frozen.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _verify_freeze(root: Path, output_root: Path) -> dict:
    receipt_path = output_root / "feature_manifest_frozen.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != STATUS_FROZEN or receipt.get("stage1_learner_forward_runs") != 0:
        raise RuntimeError("Stage-1 feature manifest is not a pre-forward freeze")
    for relative, expected in receipt["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"frozen source hash changed: {relative}")
    for env in ("coupled", "articulated"):
        item = receipt["environments"][env]
        if sha256(root / item["row_manifest"]) != item["row_manifest_sha256"]:
            raise RuntimeError(f"{env} frozen row manifest hash changed")
    return receipt


def _layer_forward(model, arrays, norms: dict, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    history, mask, query, target = _normalized(arrays, norms)
    results: dict[str, list[np.ndarray]] = {name: [] for name in ("anchor_segment", "segment", "persistent", "predicted", "prediction", "target")}
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            h = torch.from_numpy(history[start:start + batch_size]).to(device)
            m = torch.from_numpy(mask[start:start + batch_size]).to(device)
            q = torch.from_numpy(query[start:start + batch_size]).to(device)
            encoded = model.segment_encoder(h) * m[:, :, None]
            persistent = model.aggregate_encoded(encoded, m)
            qembed = model.query_encoder(q)
            predicted = model.latent_predictor(torch.cat((persistent, qembed), dim=1))
            prediction = model.target_decoder(predicted)
            results["anchor_segment"].append(encoded[:, 0].cpu().numpy())
            results["segment"].append(encoded[:, 1].cpu().numpy())
            results["persistent"].append(persistent.cpu().numpy())
            results["predicted"].append(predicted.cpu().numpy())
            results["prediction"].append(prediction.cpu().numpy())
            results["target"].append(target[start:start + batch_size])
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in results.items()}


def _baseline_scale_stats(table: pd.DataFrame, layers: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    baseline = table.candidate_index.to_numpy(int) == -1
    mapping = {
        "segment": layers["anchor_segment"][baseline],
        "persistent": layers["persistent"][baseline],
        "predicted": layers["predicted"][baseline],
        "prediction": layers["prediction"][baseline],
    }
    return {
        name: {"sum_squares": float(np.sum(values.astype(np.float64) ** 2)), "components": int(values.size)}
        for name, values in mapping.items()
    }


def _deltas(table: pd.DataFrame, layers: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    baseline_mask = table.candidate_index.to_numpy(int) == -1
    candidate_mask = ~baseline_mask
    baselines = table.loc[baseline_mask, ["system_index", "realization", "history_index", "query_index"]].copy()
    baselines["baseline_position"] = np.flatnonzero(baseline_mask)
    candidates = table.loc[candidate_mask, KEYS].copy()
    candidates["candidate_position"] = np.flatnonzero(candidate_mask)
    joined = candidates.merge(baselines, on=["system_index", "realization", "history_index", "query_index"], validate="many_to_one")
    if len(joined) != candidate_mask.sum():
        raise RuntimeError("candidate rows did not join exactly one baseline")
    cp = joined.candidate_position.to_numpy(int)
    bp = joined.baseline_position.to_numpy(int)
    delta = {
        "delta_segment": layers["segment"][cp],
        "delta_persistent": layers["persistent"][cp] - layers["persistent"][bp],
        "delta_predicted": layers["predicted"][cp] - layers["predicted"][bp],
        "delta_prediction": layers["prediction"][cp] - layers["prediction"][bp],
        "observed_target_residual": layers["target"][bp] - layers["prediction"][bp],
    }
    summary = joined[KEYS].copy()
    for name in ("delta_segment", "delta_persistent", "delta_predicted", "delta_prediction"):
        summary[name + "_norm"] = np.linalg.norm(delta[name], axis=1)
    left, right = delta["delta_prediction"], delta["observed_target_residual"]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    summary["cos_observed_target_residual_supporting_only"] = np.divide(
        np.sum(left * right, axis=1), denominator, out=np.full(len(left), np.nan), where=denominator > 0
    )
    return summary, delta


def _filter_shard(table: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    systems = table.system_index.to_numpy(int)
    return table.loc[systems % shard_count == shard_index].reset_index(drop=True)


def _load_coupled_model(root: Path, arrays, device: torch.device):
    paths = _coupled_paths(root)
    base, formal = load_spec(paths["base_spec"]), load_spec(paths["formal_spec"])
    cfg = copy.deepcopy(base["learner_development"])
    cfg.update(formal["learner"])
    model = build_persistent_jepa(
        arrays.history.shape[-1], arrays.query_action.shape[-1], arrays.target.shape[-1], cfg,
        expected_architecture_tag=CANONICAL_ARCHITECTURE_TAG,
    ).to(device)
    model.load_state_dict(torch.load(paths["checkpoint"], map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    norms_npz = np.load(paths["normalization"], allow_pickle=False)
    return model, {name: norms_npz[name] for name in norms_npz.files}, cfg


def _articulated_arrays(root: Path, table: pd.DataFrame):
    reference = json.loads(_articulated_paths(root)["reference_config"].read_text())
    rows = []
    for key, group in table[table.candidate_index >= 0].groupby(["system_index", "realization", "history_index", "query_index"]):
        candidates = sorted(group.candidate_index.astype(int).unique())
        for offset in range(0, len(candidates), 2):
            first = candidates[offset]
            second = candidates[min(offset + 1, len(candidates) - 1)]
            rows.append({"system_index": key[0], "realization": key[1], "history_index": key[2], "query_index": key[3],
                         "lqa_selected_candidate": first, "other_candidate": second})
    fake_manifest = pd.DataFrame(rows)
    baseline, baseline_index, candidate, candidate_index = build_unique_learner_rows(root, reference, fake_manifest)
    baseline_index["candidate_index"] = -1
    candidate_index = candidate_index.rename(columns={"candidate_index": "candidate_index"})
    # Reorder the generated arrays to the exact frozen row-manifest order.
    combined_index = pd.concat([baseline_index, candidate_index], ignore_index=True)
    source_arrays = [baseline, candidate]
    baseline_count = len(baseline.history)
    candidate_count = len(candidate.history)
    source_positions = np.concatenate([np.arange(baseline_count), np.arange(candidate_count)])
    source_kind = np.concatenate([np.zeros(baseline_count, dtype=int), np.ones(candidate_count, dtype=int)])
    lookup = combined_index.copy()
    lookup["source_position"] = source_positions
    lookup["source_kind"] = source_kind
    ordered = table.merge(lookup, on=KEYS, validate="one_to_one")
    from paper_c.coupled_sled.learner_data import LearnerArrays
    fields = {}
    for field in LearnerArrays.__dataclass_fields__:
        pieces = []
        for row in ordered.itertuples(index=False):
            source = source_arrays[int(row.source_kind)]
            pieces.append(getattr(source, field)[int(row.source_position)])
        fields[field] = np.asarray(pieces)
    return LearnerArrays(**fields)


def extract(root: Path, output_root: Path, environment: str, shard_index: int, shard_count: int, device_name: str, batch_size: int) -> dict:
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    freeze_receipt = _verify_freeze(root, output_root)
    item = freeze_receipt["environments"][environment]
    full_table = pd.read_csv(root / item["row_manifest"])
    table = _filter_shard(full_table, shard_index, shard_count)
    device = torch.device(device_name)
    started = time.perf_counter()
    if environment == "coupled":
        manifest_path = output_root / "work" / f"coupled_sample_manifest_shard_{shard_index:02d}_of_{shard_count:02d}.csv.gz"
        _atomic_csv(manifest_path, table)
        arrays = arrays_from_sample_manifest(_coupled_paths(root)["base_spec"], _coupled_paths(root)["system_pool"], manifest_path)
        model, norms, _ = _load_coupled_model(root, arrays, device)
    elif environment == "articulated":
        arrays = _articulated_arrays(root, table)
        reference = json.loads(_articulated_paths(root)["reference_config"].read_text())
        model, norms, _ = _load_jepa(root, reference, device)
    else:
        raise ValueError("environment must be coupled or articulated")
    layers = _layer_forward(model, arrays, norms, batch_size, device)
    baseline_scale_stats = _baseline_scale_stats(table, layers)
    summary, delta = _deltas(table, layers)
    shard_root = output_root / "shards" / f"{environment}_{shard_index:02d}_of_{shard_count:02d}"
    vector_path = shard_root / "layer_deltas.npz"
    summary_path = shard_root / "layer_norms.csv.gz"
    _atomic_npz(vector_path, **{name: value for name, value in delta.items()}, **{key: summary[key].to_numpy() for key in KEYS})
    _atomic_csv(summary_path, summary)
    receipt = {
        "status": STATUS_SHARD, "environment": environment, "shard_index": shard_index, "shard_count": shard_count,
        "systems": int(summary.system_index.nunique()), "candidate_query_rows": len(summary), "device": device_name,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_manifest_sha256": sha256(output_root / "feature_manifest_frozen.json"),
        "row_manifest_sha256": item["row_manifest_sha256"],
        "layer_vectors_sha256": sha256(vector_path), "layer_norms_sha256": sha256(summary_path),
        "baseline_scale_stats": baseline_scale_stats,
        "bayes_correction_joined": False, "models_retrained": False, "formal_learner_result_files_read": False,
        "sealed_accessed": False, "protected_scope_1_accessed": False, "protected_scope_2_accessed": False,
    }
    _atomic_text(shard_root / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def merge(root: Path, output_root: Path, environment: str, shard_count: int) -> dict:
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    frozen = _verify_freeze(root, output_root)
    summaries, vectors, shard_receipts = [], [], []
    for shard_index in range(shard_count):
        shard_root = output_root / "shards" / f"{environment}_{shard_index:02d}_of_{shard_count:02d}"
        receipt = json.loads((shard_root / "receipt.json").read_text())
        if receipt.get("status") != STATUS_SHARD or receipt.get("shard_count") != shard_count:
            raise RuntimeError("incompatible or incomplete Stage-1 shard")
        shard_receipts.append(receipt)
        summaries.append(pd.read_csv(shard_root / "layer_norms.csv.gz"))
        vectors.append(np.load(shard_root / "layer_deltas.npz", allow_pickle=False))
    summary = pd.concat(summaries, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if summary.duplicated(KEYS).any():
        raise RuntimeError("merged Stage-1 rows are duplicated")
    expected = int(frozen["environments"][environment]["systems"]) * 36
    if len(summary) != expected or summary.system_index.nunique() != int(frozen["environments"][environment]["systems"]):
        raise RuntimeError("merged Stage-1 rows are incomplete")
    order = np.lexsort(tuple(summary[key].to_numpy() for key in reversed(KEYS)))
    if not np.array_equal(order, np.arange(len(summary))):
        raise RuntimeError("summary sort invariant failed")
    merged_arrays = {}
    for name in ("delta_segment", "delta_persistent", "delta_predicted", "delta_prediction", "observed_target_residual"):
        frames = []
        for payload in vectors:
            keys = pd.DataFrame({key: payload[key] for key in KEYS})
            values = pd.DataFrame(payload[name])
            frames.append(pd.concat([keys, values], axis=1))
        frame = pd.concat(frames, ignore_index=True).sort_values(KEYS)
        merged_arrays[name] = frame.drop(columns=KEYS).to_numpy(np.float32)
    for key in KEYS: merged_arrays[key] = summary[key].to_numpy()
    final_root = output_root / "merged" / environment
    vector_path, summary_path = final_root / "layer_deltas.npz", final_root / "layer_norms.csv.gz"
    _atomic_npz(vector_path, **merged_arrays)
    _atomic_csv(summary_path, summary)
    scales = {}
    for layer in ("segment", "persistent", "predicted", "prediction"):
        sum_squares = sum(row["baseline_scale_stats"][layer]["sum_squares"] for row in shard_receipts)
        components = sum(row["baseline_scale_stats"][layer]["components"] for row in shard_receipts)
        scales[layer] = float(np.sqrt(sum_squares / components))
        if not np.isfinite(scales[layer]) or scales[layer] <= 0:
            raise RuntimeError(f"invalid baseline component RMS for {layer}")
        summary[f"delta_{layer}_standardized_norm"] = summary[f"delta_{layer}_norm"] / scales[layer]
    layer_summary = {}
    for column in ("delta_segment_norm", "delta_persistent_norm", "delta_predicted_norm", "delta_prediction_norm"):
        values = summary[column].to_numpy(float)
        layer_summary[column] = {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}
    receipt = {
        "status": STATUS_MERGED, "environment": environment,
        "systems": int(summary.system_index.nunique()), "candidate_query_rows": len(summary),
        "shards": shard_count, "device": sorted({row["device"] for row in shard_receipts}),
        "elapsed_seconds_sum": float(sum(row["elapsed_seconds"] for row in shard_receipts)),
        "layer_norm_descriptive_only": layer_summary,
        "baseline_component_rms": scales,
        "feature_manifest_sha256": sha256(output_root / "feature_manifest_frozen.json"),
        "layer_vectors_sha256": sha256(vector_path), "layer_norms_sha256": sha256(summary_path),
        "bayes_correction_joined": False, "models_retrained": False,
        "formal_learner_result_files_read": False, "sealed_accessed": False,
    }
    _atomic_text(final_root / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("root", type=Path); freeze_parser.add_argument("config", type=Path); freeze_parser.add_argument("output", type=Path)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("root", type=Path); extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("environment", choices=("coupled", "articulated"))
    extract_parser.add_argument("--shard-index", type=int, required=True); extract_parser.add_argument("--shard-count", type=int, required=True)
    extract_parser.add_argument("--device", default="cpu"); extract_parser.add_argument("--batch-size", type=int, default=1024)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("root", type=Path); merge_parser.add_argument("output", type=Path)
    merge_parser.add_argument("environment", choices=("coupled", "articulated")); merge_parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze": result = freeze(args.root, args.config, args.output)
    elif args.command == "extract": result = extract(args.root, args.output, args.environment, args.shard_index, args.shard_count, args.device, args.batch_size)
    else: result = merge(args.root, args.output, args.environment, args.shard_count)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
