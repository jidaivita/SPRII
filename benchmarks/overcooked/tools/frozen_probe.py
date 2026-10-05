"""Post-training identity and functional probes of frozen native AD representations.

The main panel has twenty-two fixed partners and sixteen fresh independent episodes
per partner. Identity classification sees twenty-two labels in its probe-fit partition;
the frozen encoders were trained on twenty partners. Functional regression
fits twenty training partners and measures two reserved partners. Identity is NOT
an interpretable parameter value or recovery of neural-network weights.

Original AD
uses its native256-dimensional final hidden state; its fixed DCT32 projection
is supplementary. Persistent methods use their actual learned pooled32 slot.
Only small ridge-linear probe heads and train-only standardizers are fitted.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

MODES = ("upstream_ad", "none", "VC", "I+VC")
STEPS, MODEL_SEED, PAIRS_PER_HISTORY = 20000, 4200, 4
RIDGE = 0.001


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify(reference):
    require(digest(reference["path"]) == reference["sha256"], f"Frozen artifact changed: {reference['path']}")


def select_episode_pairs(episodes, limit=PAIRS_PER_HISTORY):
    """Evenly cover a trajectory, without shared episodes between samples."""
    require(limit == PAIRS_PER_HISTORY, "The first probe fixes four pairs per history")
    episodes = sorted(episodes, key=lambda row: row["start"])
    available = list(range(0, len(episodes) - 1, 2))
    require(len(available) >= limit, "Each selected history needs eight explicit episode prefixes")
    chosen = [available[round(i * (len(available) - 1) / (limit - 1))] for i in range(limit)]
    require(len(set(chosen)) == limit, "Repeated support pair")
    return [[episodes[i], episodes[i + 1]] for i in chosen]


def build_sample_plan(root):
    raise ValueError("New20 run uses only the common fresh panel; no legacy four-policy supplement")


def build_panel_plan(root, panel_path, signature_path):
    import numpy as np
    panel, signature = read(panel_path), read(signature_path)
    require(panel["status"] == signature["status"] == "PASS", "Panel or signature collection is incomplete")
    verify(panel["source_isolation"])
    require(signature["panel_manifest"]["sha256"] == digest(panel_path), "Functional targets refer to another panel")
    verify(signature["panel_manifest"])
    verify(signature["npz"])
    isolation = read(root / "source_isolation.json")
    require(isolation["status"] == "SOURCE_AND_DATA_ISOLATION_PASS", "Model-training data isolation is incomplete")
    require(panel["source_isolation"]["sha256"] == digest(root / "source_isolation.json"),
            "Panel and frozen models must use the exact same source/data isolation receipt")
    for reference in isolation["data"].values():
        verify(reference)
    identities = sorted({row["teammate_params_sha256"] for row in isolation["sources"]})
    require(len(identities) == 22 and len(panel["episodes"]) == 352, "Main probe requires twenty-two fixed policies and352 episodes")
    expected_roles = {row["teammate_params_sha256"]: "train" if row["role"] == "training" else "heldout"
                      for row in isolation["sources"]}
    require(sum(role == "train" for role in expected_roles.values()) == 20
            and sum(role == "heldout" for role in expected_roles.values()) == 2,
            "Functional regression requires twenty training and two reserved policy identities")
    grouped, episodes_seen, paths_seen = {}, set(), set()
    for episode in panel["episodes"]:
        identity, number = episode["partner_identity_sha256"], int(episode["episode_index"])
        require(identity in expected_roles and episode["partner_role"] == expected_roles[identity], "Panel partner role mismatch")
        require(0 <= number < 16 and episode["probe_split"] == ("fit" if number < 8 else "test"),
                "Panel probe split must reserve independent episodes8..15")
        require((identity, number) not in episodes_seen, "Duplicate panel episode")
        absolute_npz = str(Path(episode["npz_path"]).resolve())
        require(absolute_npz not in paths_seen, "A panel episode artifact is reused under another identity/split")
        paths_seen.add(absolute_npz)
        episodes_seen.add((identity, number))
        verify({"path": episode["npz_path"], "sha256": episode["sha256"]})
        grouped.setdefault(identity, {})[number] = episode
    require(set(grouped) == set(identities) and all(set(group) == set(range(16)) for group in grouped.values()),
            "Missing independent episodes for one or more identities")
    samples = []
    for label, identity in enumerate(identities):
        for first in range(0, 16, 2):
            pair = [grouped[identity][first], grouped[identity][first + 1]]
            samples.append({"sample_id": f"{identity}/episodes{first}-{first+1}",
                "task_id": pair[0]["source_task_id"], "trajectory_group": f"{identity}/pair{first//2}",
                "probe_split": "probe_fit" if first < 8 else "probe_test", "context_index": first // 2,
                "policy_identity_sha256": identity, "identity_label": label, "partner_role": expected_roles[identity],
                "panel_episodes": pair})
    order = signature["partner_identity_sha256s"]
    require(len(order) == len(set(order)) == 22 and set(order) == set(identities), "Signature label order is incomplete")
    with np.load(signature["npz"]["path"], allow_pickle=False) as arrays:
        probabilities = arrays["probabilities"]
        require(probabilities.shape == (22, 40, 8, 6) and np.isfinite(probabilities).all()
                and (probabilities >= 0).all() and (probabilities <= 1).all()
                and np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5), "Invalid common-input action probabilities")
        anchor_sources = signature["anchors"]
        require(len(anchor_sources) == 40 and arrays["anchor_obs"].shape == (40, 8, 5, 5, 40)
                and arrays["anchor_avail_actions"].shape == (40, 8, 6)
                and arrays["anchor_done"].shape == (40, 8), "Common private-input bank shape changed")
        require(arrays["anchor_done"][:, 0].all() and not arrays["anchor_done"][:, 1:].any(),
                "Each signature snippet must reset recurrent state independently")
        episodes_by_path = {str(Path(ep["npz_path"]).resolve()): ep for ep in panel["episodes"]}
        actual_anchors = set()
        for i, anchor in enumerate(anchor_sources):
            source = episodes_by_path.get(str(Path(anchor["episode_npz"]["path"]).resolve()))
            require(source and source["sha256"] == anchor["episode_npz"]["sha256"] and source["partner_role"] == "train"
                    and source["probe_split"] == "fit" and source["episode_index"] == 0
                    and anchor["episode_index"] == 0 and anchor["partner_identity_sha256"] == source["partner_identity_sha256"]
                    and anchor["start"] in (0, 32) and anchor["length"] == 8,
                    "Functional input bank must come only from fixed training-partner fit episodes")
            actual_anchors.add((source["partner_identity_sha256"], anchor["start"]))
            with np.load(anchor["episode_npz"]["path"], allow_pickle=False) as trace:
                start = anchor["start"]
                np.testing.assert_array_equal(arrays["anchor_obs"][i], trace["obs_partner"][start:start + 8])
                np.testing.assert_array_equal(arrays["anchor_avail_actions"][i], trace["avail_actions"][start:start + 8, 1])
        expected_anchors = {(identity, start) for identity, role in expected_roles.items() if role == "train" for start in (0, 32)}
        require(actual_anchors == expected_anchors, "Common bank omitted or repeated a training partner snippet")
        targets = np.stack([probabilities[order.index(identity)].reshape(-1) for identity in identities])
    return {"format": "native-v4/frozen-common-panel-probe/1", "status": "INPUT_PLAN_FIXED",
            "data": isolation["data"], "source_isolation": ref(root / "source_isolation.json"),
            "panel_source_isolation": panel["source_isolation"],
            "panel_manifest": ref(panel_path), "signature_manifest": ref(signature_path), "signature_npz": signature["npz"],
            "labels": identities, "samples": samples, "history_tokens": 200, "supports": 2,
            "label_semantics": "verified fixed-policy identity, not physical/interpretable parameter values",
            "functional_target_semantics": "1920 action probabilities on one common40x8 private-input bank with zero initial GRU",
            "functional_fit_rule": "only twenty training partners and their probe_fit episode pairs",
            "functional_test_rule": "only two heldout partners and their independent probe_test episode pairs",
            "split_rule": "episodes0..7 fit;8..15 test; adjacent nonoverlapping two-episode supports",
            "encoder_pretraining_saw_these_histories": False, "encoder_pretraining_partner_count": 20,
            "identity_classifier_probe_fit_partner_count": 22, "reserved_4602_used": True,
            "claim_scope": "single-model-seed descriptive identity and functional-invariant readout; no weight recovery or population-level significance"}, targets


def panel_support_batch(samples):
    import numpy as np
    result = {key: [] for key in ("obs", "prev_actions", "prev_rewards", "attention_mask")}
    for sample in samples:
        support = {key: [] for key in result}
        for episode in sample["panel_episodes"]:
            with np.load(episode["npz_path"], allow_pickle=False) as arrays:
                observations = arrays["obs_ego"]
                actions, rewards = arrays["actions"], arrays["rewards"]
                require(observations.shape == (101, 5, 5, 40) and actions.shape == rewards.shape == (100, 2),
                        "Probe panel requires complete100-step native episodes")
                obs = np.asarray(observations[:100], np.float32)
                ego_actions, ego_rewards = np.asarray(actions[:, 0], np.int32), np.asarray(rewards[:, 0], np.float32)
                require(np.isfinite(obs).all() and np.isfinite(ego_rewards).all()
                        and ((0 <= ego_actions) & (ego_actions < 6)).all(), "Panel ego inputs are invalid")
                support["obs"].append(obs)
                support["prev_actions"].append(np.concatenate([np.zeros(1, np.int32), ego_actions[:-1]]))
                support["prev_rewards"].append(np.concatenate([np.zeros(1, np.float32), ego_rewards[:-1]]))
                support["attention_mask"].append(np.ones(100, np.float32))
        for key in result:
            result[key].append(np.stack(support[key]))
    # Partner-private observations, signature probabilities and identity never enter this dictionary.
    return {key: np.stack(value) for key, value in result.items()}


def support_batch(store, samples):
    import numpy as np
    result = {name: [] for name in ("obs", "prev_actions", "prev_rewards", "attention_mask")}
    for sample in samples:
        group = store[sample["h5_group"]]
        require(group.attrs["task_id"] == sample["task_id"] and group.attrs["env_idx"] == sample["env_idx"],
                "HDF5 observation source differs from the planned trajectory")
        pair = {name: [] for name in result}
        for segment in sample["supports"]:
            start, end = segment["start"], segment["end"]
            obs = np.asarray(group["obs"][start:end], np.float32)
            actions = np.asarray(group["actions"][start:end], np.int32)
            rewards = np.asarray(group["rewards"][start:end], np.float32)
            require(obs.shape == (100, 5, 5, 40) and actions.shape == rewards.shape == (100,), "Unexpected native prefix shapes")
            require(np.isfinite(obs).all() and np.isfinite(rewards).all() and ((0 <= actions) & (actions < 6)).all(),
                    "Nonfinite native input or invalid action")
            pair["obs"].append(obs)
            pair["prev_actions"].append(np.concatenate([np.zeros(1, np.int32), actions[:-1]]))
            pair["prev_rewards"].append(np.concatenate([np.zeros(1, np.float32), rewards[:-1]]))
            pair["attention_mask"].append(np.ones(100, np.float32))
        for name in result:
            result[name].append(np.stack(pair[name]))
    return {name: np.stack(value) for name, value in result.items()}


def load_frozen_model(repo, root, mode, data):
    from types import SimpleNamespace
    from evaluate_support import load_model
    isolation = read(root / "source_isolation.json")
    require(isolation["data"] == data, "Probe data differs from utility checkpoint binding")
    return load_model(SimpleNamespace(repo=str(repo), run_root=str(root), mode=mode), isolation)


def dct_projection(input_dim=256, output_dim=32):
    """Fixed analytical orthonormal columns; never fitted to data or labels."""
    import numpy as np
    columns = np.cos(np.pi / input_dim * (np.arange(input_dim)[:, None] + .5) * np.arange(output_dim)[None, :])
    columns[:, 0] *= math.sqrt(1 / input_dim)
    columns[:, 1:] *= math.sqrt(2 / input_dim)
    return columns.astype(np.float32)


def extract_model_features(repo, root, mode, plan, batch_size, loaded=None):
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np
    from native_a.model import ADHiddenBackbone
    model, params, cfg, checkpoint = loaded or load_frozen_model(repo, root, mode, plan["data"])
    equivalent = None
    if mode == "upstream_ad":
        hidden_model = ADHiddenBackbone(cfg)
        @jax.jit
        def encode(parameters, support):
            flat = {key: value.reshape((value.shape[0], 200) + value.shape[3:]) for key, value in support.items()}
            hidden = hidden_model.apply(parameters, **flat, train=False)
            return hidden[:, -1], hidden
    else:
        @jax.jit
        def encode(parameters, support):
            slots, pooled, weights = model.apply({"params": parameters}, **support,
                train=False, method=model.encode_supports)
            return pooled, slots
    features = []
    source = nullcontext(None) if "panel_manifest" in plan else h5py.File(plan["data"]["h5"]["path"], "r")
    with source as store:
        for start in range(0, len(plan["samples"]), batch_size):
            samples = plan["samples"][start:start + batch_size]
            support = panel_support_batch(samples) if store is None else support_batch(store, samples)
            representation, aux = encode(params, support)
            representation = np.asarray(jax.device_get(representation))
            require(np.isfinite(representation).all(), "Nonfinite frozen representation")
            if mode == "upstream_ad" and equivalent is None:
                flat = {key: value.reshape((value.shape[0], 200) + value.shape[3:]) for key, value in support.items()}
                original_logits = np.asarray(model.apply(params, **flat, train=False))
                head = params["params"]["action_head"]
                reconstructed = np.asarray(aux @ head["kernel"] + head["bias"])
                np.testing.assert_allclose(reconstructed, original_logits, atol=2e-5, rtol=2e-5)
                equivalent = {"status": "PASS", "max_abs_logit_error": float(np.max(np.abs(reconstructed - original_logits)))}
            features.append(representation)
    result = {mode: np.concatenate(features)}
    if mode == "upstream_ad":
        require(result[mode].shape[1] == 256, "Unexpected original AD hidden width")
        result["upstream_ad_DCT32_supplementary"] = result[mode] @ dct_projection()
    else:
        require(result[mode].shape[1] == 32, "Persistent slot width changed")
    return result, {"checkpoint": checkpoint, "representation_dim": result[mode].shape[1],
                    "frozen_encoder_updates": 0, "original_hidden_equivalence": equivalent}


def linear_identity_probe(features, labels, fit_mask, test_mask):
    """Fixed-ridge multi-output least-squares classifier; no tuning on test."""
    import numpy as np
    features, labels = np.asarray(features, np.float64), np.asarray(labels, np.int32)
    classes = np.unique(labels[fit_mask])
    require(len(classes) == 22 and np.array_equal(classes, np.arange(len(classes)))
            and set(labels[test_mask]) == set(classes), "Identity classes differ across splits")
    center = features[fit_mask].mean(axis=0)
    scale = features[fit_mask].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = (features - center) / scale
    targets = np.eye(len(classes))[labels[fit_mask]]
    target_mean = targets.mean(axis=0)
    coef = np.linalg.solve(x[fit_mask].T @ x[fit_mask] + RIDGE * np.eye(x.shape[1]),
                           x[fit_mask].T @ (targets - target_mean))
    predictions = (x @ coef + target_mean).argmax(axis=1)
    counts = np.bincount(labels[fit_mask], minlength=len(classes))
    majority = int(counts.argmax())
    metrics = {}
    for name, mask in (("probe_fit", fit_mask), ("probe_test", test_mask)):
        recall = [float((predictions[mask & (labels == cls)] == cls).mean()) for cls in classes]
        metrics[name] = {"accuracy": float((predictions[mask] == labels[mask]).mean()),
                         "macro_accuracy": float(np.mean(recall)), "per_identity_recall": recall,
                         "chance_accuracy": 1 / len(classes), "majority_baseline_accuracy": float((labels[mask] == majority).mean()),
                         "examples": int(mask.sum())}
    model = {"ridge": RIDGE, "center": center, "scale": scale, "coef": coef,
             "intercept": target_mean, "classes": classes, "predictions": predictions}
    return metrics, model, x


def functional_invariant_probe(features, labels, targets, fit_mask, test_mask, partner_roles):
    """A fixed linear readout of shared-input policy response, not NN weights."""
    import numpy as np
    features, labels, targets = np.asarray(features, np.float64), np.asarray(labels, np.int32), np.asarray(targets, np.float64)
    roles = np.asarray(partner_roles)
    fitting = fit_mask & (roles == "train")
    testing = test_mask & (roles == "heldout")
    fit_labels, test_labels = np.unique(labels[fitting]), np.unique(labels[testing])
    require(len(fit_labels) == 20 and len(test_labels) == 2 and not set(fit_labels).intersection(test_labels),
            "Functional readout must fit twenty identities and evaluate two unseen identities")
    center = features[fitting].mean(axis=0)
    scale = features[fitting].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = (features - center) / scale
    # Equal numbers of fit pairs per identity, and explicit equal-policy baseline.
    counts = [int((fitting & (labels == identity)).sum()) for identity in fit_labels]
    require(len(set(counts)) == 1, "Functional training partner weights would be unequal")
    target_mean = targets[fit_labels].mean(axis=0)
    coef = np.linalg.solve(x[fitting].T @ x[fitting] + RIDGE * np.eye(x.shape[1]),
                           x[fitting].T @ (targets[labels[fitting]] - target_mean))
    predictions = x @ coef + target_mean
    require(np.isfinite(predictions).all(), "Nonfinite frozen linear function readout")
    per_partner = []
    for label in test_labels:
        selected = testing & (labels == label)
        actual = targets[label]
        mse = float(np.mean((predictions[selected] - actual) ** 2))
        baseline_mse = float(np.mean((target_mean - actual) ** 2))
        per_partner.append({"identity_label": int(label), "episode_pairs": int(selected.sum()),
                            "mse": mse, "trainmean_baseline_mse": baseline_mse})
    macro_mse = float(np.mean([r["mse"] for r in per_partner]))
    baseline = float(np.mean([r["trainmean_baseline_mse"] for r in per_partner]))
    pair_mse = [float(np.mean((targets[i] - targets[j]) ** 2)) for i in range(22) for j in range(i + 1, 22)]
    diagnostics = {"exact_distinct_signatures": int(np.unique(targets, axis=0).shape[0]),
                   "across_partner_mean_component_variance": float(np.var(targets, axis=0).mean()),
                   "minimum_pair_signature_mse": min(pair_mse), "mean_pair_signature_mse": float(np.mean(pair_mse)),
                   "nearly_identical_pairs_mse_at_most1e-12": int(sum(value <= 1e-12 for value in pair_mse)),
                   "heldout_trainmean_baseline_nonzero": baseline > 1e-12}
    metric = {"heldout_partner_macro_mse": macro_mse, "trainmean_baseline_macro_mse": baseline,
              "fractional_mse_reduction_vs_trainmean": 1 - macro_mse / baseline if baseline > 1e-12 else None,
              "per_heldout_partner": per_partner, "functional_target_diagnostics": diagnostics,
              "evaluation_weighting": "each of two unseen partner identities has equal weight; pairs averaged within partner",
              "fit_policy_count": 20, "test_policy_count": 2, "label_is_full_weights": False,
              "no_significance_claim": True,
              "qualification_note": "Report signature variance and distances; nearly identical targets do not qualify a useful functional probe"}
    head = {"ridge": RIDGE, "center": center, "scale": scale, "coef": coef, "intercept": target_mean,
            "fit_identity_labels": fit_labels, "test_identity_labels": test_labels,
            "fit_mask": fitting, "test_mask": testing, "predictions": predictions}
    return metric, head


def formation_summary(features, standardized, labels, envs, test_mask):
    import numpy as np
    raw, x, y, e = features[test_mask], standardized[test_mask], labels[test_mask], envs[test_mask]
    norm = np.linalg.norm(x, axis=1)
    unit = x / np.maximum(norm[:, None], 1e-12)
    cosine_distance = 1 - unit @ unit.T
    pair = np.triu(np.ones((len(y), len(y)), dtype=bool), 1) & (e[:, None] != e[None, :])
    same, different = pair & (y[:, None] == y[None, :]), pair & (y[:, None] != y[None, :])
    require(same.any() and different.any(), "Need cross-env same and different partner pairs")
    return {"raw_mean_dimension_variance": float(np.var(raw, axis=0).mean()),
            "raw_noncollapsed_dimensions_std_above1e-6": int((np.std(raw, axis=0) > 1e-6).sum()),
            "same_partner_cross_env_mean_cosine_distance": float(cosine_distance[same].mean()),
            "different_partner_cross_env_mean_cosine_distance": float(cosine_distance[different].mean()),
            "same_pairs": int(same.sum()), "different_pairs": int(different.sum()),
            "distance_standardization": "only probe_fit means/stds; probe_test distance diagnostic",
            "not_independent_samples": "Pairs share histories; no confidence interval treats pairs/windows as independent"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--panel-manifest", help="Primary shared22-partner fresh352-episode panel")
    parser.add_argument("--signature-manifest", help="Required with --panel-manifest; common40x8 private-input target bank")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    require(platform.system() == "Linux", "Frozen model inference belongs on the existing Linux runtime")
    require(1 <= args.threads <= 24 and 1 <= args.batch_size <= 32, "Invalid read-only inference resource budget")
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.threads])
    repo, root, output = Path(args.repo).resolve(), Path(args.run_root).resolve(), Path(args.out_dir).resolve()
    sys.path.insert(0, str(repo))
    import jax
    import numpy as np
    require(jax.default_backend() == "cpu", "No GPU occupation for the small post-training probe")
    require(not output.exists(), "Refusing to overwrite or silently repeat a probe")
    for mode in MODES:
        config = read(root / "models" / mode.replace("+", "_") / "config.json")
        require(config["num_steps"] == STEPS and config["seed"] == MODEL_SEED, "All four planned frozen models required")
    require(bool(args.panel_manifest) == bool(args.signature_manifest), "Main panel requires its independently recorded signature manifest")
    if args.panel_manifest:
        plan, functional_targets = build_panel_plan(root, args.panel_manifest, args.signature_manifest)
    else:
        plan, functional_targets = build_sample_plan(root), None
    # Validate every actual terminal optimizer state before any representation
    # inference or probe fitting. A config that only plans20k is insufficient.
    frozen = {mode: load_frozen_model(repo, root, mode, plan["data"]) for mode in MODES}
    output.mkdir(parents=True)
    write(output / "probe_sample_plan.json", plan)
    script_ref = ref(__file__)
    started, representations, model_refs = time.monotonic(), {}, {}
    for mode in MODES:
        values, model_receipt = extract_model_features(repo, root, mode, plan, args.batch_size, loaded=frozen[mode])
        representations.update(values)
        model_refs[mode] = model_receipt
        print(json.dumps({"event": "frozen_representations_extracted", "mode": mode,
                          "samples": len(plan["samples"]), "encoder_updates": 0}), flush=True)
    labels = np.asarray([s["identity_label"] for s in plan["samples"]], np.int32)
    envs = np.asarray([s.get("env_idx", s.get("context_index")) for s in plan["samples"]], np.int32)
    fit_mask = np.asarray([s["probe_split"] == "probe_fit" for s in plan["samples"]])
    test_mask = ~fit_mask
    np.savez_compressed(output / "frozen_representations.npz", labels=labels, independent_context_index=envs,
                        probe_fit=fit_mask, probe_test=test_mask, **representations)
    results = {}
    for name, features in representations.items():
        metrics, head, standardized = linear_identity_probe(features, labels, fit_mask, test_mask)
        head_file = output / (name.replace("+", "_") + "_linear_probe.npz")
        np.savez_compressed(head_file, **head)
        results[name] = {"classification": metrics, "dimension": features.shape[1],
                         "formation": formation_summary(features, standardized, labels, envs, test_mask),
                         "frozen_linear_head": ref(head_file),
                         "primary": name != "upstream_ad_DCT32_supplementary"}
        if functional_targets is not None:
            functional_metrics, functional_head = functional_invariant_probe(
                features, labels, functional_targets, fit_mask, test_mask,
                [sample["partner_role"] for sample in plan["samples"]])
            path = output / (name.replace("+", "_") + "_functional_probe.npz")
            np.savez_compressed(path, **functional_head)
            results[name]["functional_invariant_regression"] = {**functional_metrics, "frozen_linear_head": ref(path)}
    for item in plan["data"].values():
        verify(item)
    if functional_targets is not None:
        for key in ("panel_manifest", "signature_manifest", "signature_npz"):
            verify(plan[key])
        for sample in plan["samples"]:
            for episode in sample["panel_episodes"]:
                verify({"path": episode["npz_path"], "sha256": episode["sha256"]})
    verify(script_ref)
    from native_a.original_checkpoint import inventory
    for mode, model_receipt in model_refs.items():
        ckpt = model_receipt["checkpoint"]
        if mode == "upstream_ad":
            require(inventory(ckpt["checkpoint"]) == ckpt["checkpoint_files"], "Frozen original model changed during probe")
        else:
            verify({"path": ckpt["path"], "sha256": ckpt["sha256"]})
    status = "PASS_FROZEN_COMMON_PANEL_PROBE" if functional_targets is not None else "PASS_DESCRIPTIVE_SEEN_PARTNER_PROBE"
    receipt = {"status": status, "results": results,
               "frozen_models": model_refs, "sample_plan": ref(output / "probe_sample_plan.json"),
               "representations": ref(output / "frozen_representations.npz"), "script": script_ref,
               "model_seed": MODEL_SEED, "new_backbone_training": False, "new_model_seeds": False,
               "linear_probe_ridge": RIDGE, "head_hyperparameter_selection": "fixed; no probe-test tuning",
               "parameter_value_recovery": False, "heldout4602_evaluated": functional_targets is not None,
               "claim_scope": plan["claim_scope"],
               "encoder_pretraining_saw_these_histories": plan["encoder_pretraining_saw_these_histories"],
               "identity_classifier_fit_sees_all_panel_identities": functional_targets is not None,
               "functional_regressor_fit_sees_reserved_identities": False,
               "warning": "Identity decodability is not parameter-value recovery; functional readout is of a small fixed response bank, with only two unseen policies",
               "elapsed_seconds": time.monotonic() - started}
    write(output / "probe_result.json", receipt)
    print(json.dumps({"status": receipt["status"], "result": str(output / "probe_result.json")}), flush=True)


if __name__ == "__main__":
    main()
