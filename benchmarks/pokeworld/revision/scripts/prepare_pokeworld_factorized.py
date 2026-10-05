#!/usr/bin/env python3
"""Generate the v2 factorized bank with shared levels and disjoint exact tuples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from persistent_jepa.pokeworld import PokeConfig, simulate_split


GENERATOR_SEED = 20260902
MASS_LEVELS = 10
DRAG_LEVELS = 10
STIFFNESS_LEVELS = 28
TRAIN_PER_BLOCK = 20
VALIDATION_PER_BLOCK = 4
CONFIRMATION_PER_BLOCK = 4
ROTATION_STEP = 9  # coprime with 28; rotates a balanced circular split word.


def digest_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def tuple_partition() -> dict[str, np.ndarray]:
    """Balanced 20/4/4 allocation for every (mass, drag) block.

    Individual stiffness levels recur across splits globally.  An exact
    (mass, drag, stiffness) tuple belongs to exactly one split.
    """
    base = np.random.default_rng(GENERATOR_SEED).permutation(STIFFNESS_LEVELS)
    # Interleaving the 20/4/4 labels as (TTTTTVC)x4 makes every residual
    # segment of the 100 block rotations balanced to within one count.
    split_word = np.asarray((["train"] * 5 + ["validation", "confirmation"]) * 4)
    output: dict[str, list[tuple[int, int, int]]] = {
        "train": [], "validation": [], "confirmation": []
    }
    for mi in range(MASS_LEVELS):
        for gi in range(DRAG_LEVELS):
            block = mi * DRAG_LEVELS + gi
            labels = np.roll(split_word, -(ROTATION_STEP * block) % STIFFNESS_LEVELS)
            allocation = {
                split: base[labels == split]
                for split in ("train", "validation", "confirmation")
            }
            for split, indices in allocation.items():
                output[split].extend((mi, gi, int(ki)) for ki in indices)
    return {name: np.asarray(value, dtype=np.int64) for name, value in output.items()}


def factor_arrays(
    tuples: np.ndarray,
    mass_levels: np.ndarray,
    drag_levels: np.ndarray,
    stiffness_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        mass_levels[tuples[:, 0]],
        drag_levels[tuples[:, 1]],
        stiffness_levels[tuples[:, 2]],
    )


def donor_pools(tuples: np.ndarray) -> dict[str, np.ndarray]:
    """Create three deterministic legal donors for G1, G2, and Random."""
    lookup = {tuple(value): index for index, value in enumerate(tuples.tolist())}
    blocks: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, (mi, gi, ki) in enumerate(tuples.tolist()):
        blocks.setdefault((mi, gi), []).append((ki, index))
    for values in blocks.values():
        values.sort()

    output = {
        name: np.empty((tuples.shape[0], 3), dtype=np.int64)
        for name in ("G1", "G2", "Random")
    }
    gamma_offsets = (1, 3, 7)
    for query, (mi, gi, ki) in enumerate(tuples.tolist()):
        same_block = [index for donor_k, index in blocks[(mi, gi)] if donor_k != ki]
        if len(same_block) < 3:
            raise AssertionError("every (mass, drag) block needs at least four stiffness tuples")
        for slot in range(3):
            output["G2"][query, slot] = same_block[(query + slot) % len(same_block)]

            g1_gi = (gi + gamma_offsets[slot]) % DRAG_LEVELS
            g1_candidates = [index for donor_k, index in blocks[(mi, g1_gi)] if donor_k != ki]
            if not g1_candidates:
                raise AssertionError("G1 requires a stiffness-mismatched donor")
            output["G1"][query, slot] = g1_candidates[(query + slot) % len(g1_candidates)]

    # Random is a null relation, not an anti-physical intervention.  Three
    # frozen derangements preserve the empirical system marginals exactly,
    # exclude only exact-system self-pairs, and allow accidental single-factor
    # matches at their natural chance rates.  Each system is used exactly once
    # as a donor per slot.
    rng = np.random.default_rng(GENERATOR_SEED + 104729 + tuples.shape[0])
    order = rng.permutation(tuples.shape[0])
    shifts = (1, max(2, tuples.shape[0] // 3), max(3, (2 * tuples.shape[0]) // 3))
    if len({shift % tuples.shape[0] for shift in shifts}) != 3:
        raise AssertionError("Random derangement shifts must be unique")
    for slot, shift in enumerate(shifts):
        donor = np.empty(tuples.shape[0], dtype=np.int64)
        donor[order] = np.roll(order, shift)
        if np.any(donor == np.arange(tuples.shape[0])):
            raise AssertionError("Random contains an exact-system collision")
        output["Random"][:, slot] = donor

    # Ensure all exact tuples are resolvable and no accidental duplicates exist.
    if len(lookup) != tuples.shape[0]:
        raise AssertionError("duplicate exact tuples within split")
    return output


def pool_audit(
    tuples: np.ndarray,
    factor: tuple[np.ndarray, np.ndarray, np.ndarray],
    pools: dict[str, np.ndarray],
) -> dict:
    mass, drag, stiffness = (np.asarray(x, dtype=np.float64) for x in factor)
    report = {}
    for name, pool in pools.items():
        query = np.repeat(np.arange(pool.shape[0]), 3)
        donor = pool.reshape(-1)
        if np.any(query == donor):
            raise AssertionError(f"{name} contains self-system donors")
        counts = np.bincount(donor, minlength=pool.shape[0])
        report[name] = {
            "shape": list(pool.shape),
            "unique_donors_per_query_min": int(min(np.unique(row).size for row in pool)),
            "donor_exposure_min": int(counts.min()),
            "donor_exposure_max": int(counts.max()),
            "same_mass_fraction": float(np.mean(mass[query] == mass[donor])),
            "same_drag_fraction": float(np.mean(drag[query] == drag[donor])),
            "same_stiffness_fraction": float(np.mean(stiffness[query] == stiffness[donor])),
            "mean_abs_log_mass_difference": float(np.mean(np.abs(np.log(mass[query]) - np.log(mass[donor])))),
            "mean_abs_drag_difference": float(np.mean(np.abs(drag[query] - drag[donor]))),
            "mean_abs_log_stiffness_difference": float(np.mean(np.abs(np.log(stiffness[query]) - np.log(stiffness[donor])))),
        }
    if report["G1"]["same_mass_fraction"] != 1.0:
        raise AssertionError("G1 must preserve mass")
    if report["G1"]["same_drag_fraction"] != 0.0 or report["G1"]["same_stiffness_fraction"] != 0.0:
        raise AssertionError("G1 kernel must vary drag and stiffness")
    if report["G2"]["same_mass_fraction"] != 1.0 or report["G2"]["same_drag_fraction"] != 1.0:
        raise AssertionError("G2 must preserve mass and drag")
    if report["G2"]["same_stiffness_fraction"] != 0.0:
        raise AssertionError("G2 kernel must vary stiffness")
    random_pool = pools["Random"]
    if np.any(random_pool == np.arange(random_pool.shape[0])[:, None]):
        raise AssertionError("Random kernel must be an exact-system derangement")
    random_counts = np.bincount(random_pool.reshape(-1), minlength=random_pool.shape[0])
    if random_counts.min() != 3 or random_counts.max() != 3:
        raise AssertionError("Random donor exposure must be exactly marginal matched")
    return report


def allocation_audit(partition: dict[str, np.ndarray]) -> dict:
    sets = {name: {tuple(row) for row in value.tolist()} for name, value in partition.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_confirmation": len(sets["train"] & sets["confirmation"]),
        "validation_confirmation": len(sets["validation"] & sets["confirmation"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"exact tuple overlap: {overlaps}")
    report = {"exact_tuple_overlap": overlaps, "splits": {}}
    for name, tuples in partition.items():
        counts = np.bincount(tuples[:, 2], minlength=STIFFNESS_LEVELS)
        per_block = np.bincount(
            tuples[:, 0] * DRAG_LEVELS + tuples[:, 1],
            minlength=MASS_LEVELS * DRAG_LEVELS,
        )
        report["splits"][name] = {
            "systems": int(tuples.shape[0]),
            "stiffness_levels_covered": int(np.count_nonzero(counts)),
            "stiffness_frequency_min": int(counts.min()),
            "stiffness_frequency_max": int(counts.max()),
            "tuples_per_mass_drag_min": int(per_block.min()),
            "tuples_per_mass_drag_max": int(per_block.max()),
        }
        if counts.max() - counts.min() > 1:
            raise AssertionError(f"{name} stiffness allocation is not balanced")
    return report


def geometry_control_audit(factor: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """Check that the three factor-distance controls are identifiable."""
    mass, drag, stiffness = (np.asarray(value, dtype=np.float64) for value in factor)
    left, right = np.triu_indices(mass.size, k=1)
    distance = np.column_stack([
        np.abs(np.log(mass[left]) - np.log(mass[right])),
        np.abs(drag[left] - drag[right]),
        np.abs(np.log(stiffness[left]) - np.log(stiffness[right])),
    ])
    scaled = (distance - distance.mean(0)) / distance.std(0).clip(1e-12)
    design = np.column_stack([np.ones(scaled.shape[0]), scaled])
    residual_variance = {}
    for target in range(3):
        controls = np.column_stack([np.ones(scaled.shape[0]), np.delete(scaled, target, axis=1)])
        gram = controls.T @ controls
        beta = np.linalg.solve(gram, controls.T @ scaled[:, target])
        residual = scaled[:, target] - controls @ beta
        residual_variance[("mass", "drag", "stiffness")[target]] = float(np.var(residual))
    report = {
        "system_pairs": int(distance.shape[0]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_columns": int(design.shape[1]),
        "condition_number_standardized_design": float(np.linalg.cond(design)),
        "residual_factor_distance_variance": residual_variance,
        "finite": bool(np.isfinite(design).all()),
    }
    if not report["finite"] or report["design_rank"] != report["design_columns"]:
        raise AssertionError(f"singular/nonfinite geometry controls: {report}")
    if any(value <= 0 for value in residual_variance.values()):
        raise AssertionError(f"zero residual factor-distance variance: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = PokeConfig(seed=GENERATOR_SEED)
    mass_levels = np.geomspace(cfg.mass_low, cfg.mass_high, MASS_LEVELS).astype(np.float32)
    drag_levels = np.linspace(cfg.gamma_low, cfg.gamma_high, DRAG_LEVELS).astype(np.float32)
    stiffness_levels = np.geomspace(
        cfg.stiffness_low, cfg.stiffness_high, STIFFNESS_LEVELS
    ).astype(np.float32)
    partition = tuple_partition()
    allocation = allocation_audit(partition)
    train_factor = factor_arrays(partition["train"], mass_levels, drag_levels, stiffness_levels)
    val_factor = factor_arrays(partition["validation"], mass_levels, drag_levels, stiffness_levels)

    seeds = np.random.SeedSequence(GENERATOR_SEED).spawn(2)
    train = simulate_split(2000, 0, seeds[0], cfg, system_parameters=train_factor)
    validation = simulate_split(400, 2000, seeds[1], cfg, system_parameters=val_factor)
    np.savez_compressed(args.output / "train.npz", **train)
    np.savez_compressed(args.output / "val.npz", **validation)
    np.savez_compressed(args.output / "tuple_allocation.npz", **partition)

    train_pools = donor_pools(partition["train"])
    val_pools = donor_pools(partition["validation"])
    np.savez_compressed(
        args.output / "donor_pools.npz",
        **{f"train_{name}": value for name, value in train_pools.items()},
        **{f"val_{name}": value for name, value in val_pools.items()},
    )
    confirmation_factor = factor_arrays(
        partition["confirmation"], mass_levels, drag_levels, stiffness_levels
    )
    pool_report = {
        "train": pool_audit(partition["train"], train_factor, train_pools),
        "validation": pool_audit(partition["validation"], val_factor, val_pools),
        "G3": {
            "semantics": "same exact (mass, drag, stiffness) system",
            "pairing_kernel": "different rollout",
            "rollouts_per_system": 4,
            "legal_donor_rollouts_per_query": 3,
        },
    }
    tuple_payload = {name: value.tolist() for name, value in partition.items()}
    manifest = {
        "schema_version": "pokeworld-factorized-2.0",
        "generator_seed": GENERATOR_SEED,
        "config": asdict(cfg),
        "system_identity_fields": ["mass", "gamma", "stiffness"],
        "factor_level_support_shared_across_splits": True,
        "exact_tuples_split_disjoint": True,
        "relation_semantics": {
            "G1": {"shared_key": ["mass"]},
            "G2": {"shared_key": ["mass", "drag"]},
            "G3": {"shared_key": ["mass", "drag", "stiffness"]},
        },
        "pairing_kernel": {
            "G1": "same mass; deliberately vary drag and stiffness",
            "G2": "same mass and drag; deliberately vary stiffness",
            "G3": "same exact system; deliberately vary rollout",
            "Random": "exact-system derangement with marginal-matched deterministic random pairing; accidental single-factor matches are retained",
        },
        "training_pair_distribution": "Pi_g(A,B)=P(A)Q_g(B|A)",
        "model_input_contains_factor_metadata": False,
        "k_pool": 3,
        "k_used": 1,
        "rollouts_per_system": 4,
        "factor_levels": {
            "mass": mass_levels.tolist(),
            "drag": drag_levels.tolist(),
            "stiffness": stiffness_levels.tolist(),
        },
        "allocation": {
            "method": "frozen base permutation plus block-wise cyclic rotation",
            "rotation_step": ROTATION_STEP,
            "per_mass_drag_block": {"train": 20, "validation": 4, "confirmation": 4},
            "audit": allocation,
            "tuple_allocation_sha256": digest_json(tuple_payload),
        },
        "geometry_control_audit": {
            "train": geometry_control_audit(train_factor),
            "validation": geometry_control_audit(val_factor),
            "confirmation_reserved_tuples": geometry_control_audit(confirmation_factor),
        },
        "splits": {
            "train": {"systems": 2000, "system_id_offset": 0, "generated": True},
            "validation": {"systems": 400, "system_id_offset": 2000, "generated": True},
            "confirmation": {"systems": 400, "system_id_offset": 2400, "generated": False},
        },
        "pool_audit": pool_report,
        "confirmation_accessed": False,
        "test_read": False,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "feasibility_audit.json").write_text(
        json.dumps({"allocation": allocation, "pools": pool_report}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
