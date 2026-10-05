from __future__ import annotations

import numpy as np

from persistent_jepa.pokeworld import (
    PokeConfig,
    generate_pokeworld_ood,
    sample_uniform_union,
    simulate_split,
)


def test_uniform_union_respects_support_and_width_weights() -> None:
    values = sample_uniform_union(
        np.random.default_rng(7), ((0.5, 1.5), (2.5, 4.0)), 100_000
    )
    assert np.all(((values >= 0.5) & (values < 1.5)) | ((values >= 2.5) & (values <= 4.0)))
    left_fraction = float((values < 1.5).mean())
    assert abs(left_fraction - 0.4) < 0.01


def test_ood_generation_separates_systems_and_gamma_support() -> None:
    cfg = PokeConfig(
        train_systems=12, val_systems=5, test_systems=6, rollouts_per_system=2,
        episode_states=20, substeps=1, seed=11,
    )
    data = generate_pokeworld_ood(cfg, ((0.5, 1.5), (2.5, 4.0)), ((1.5, 2.5),))
    train_gamma, val_gamma, ood_gamma = (data[key]["gamma"] for key in ("train", "val", "ood"))
    for values in (train_gamma, val_gamma):
        assert np.all(((values >= 0.5) & (values < 1.5)) | ((values >= 2.5) & (values <= 4.0)))
    assert np.all((ood_gamma >= 1.5) & (ood_gamma <= 2.5))
    ids = [set(data[key]["system_ids"].tolist()) for key in ("train", "val", "ood")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    for split in data.values():
        assert split["states"].shape[1] == 2
        assert np.allclose(split["gamma"][:, None], np.repeat(split["gamma"][:, None], 2, axis=1))


def test_fixed_parameters_generate_new_interactions_without_parameter_drift() -> None:
    cfg = PokeConfig(
        train_systems=3, val_systems=1, test_systems=1, rollouts_per_system=2,
        episode_states=20, substeps=1,
    )
    mass = np.asarray([0.6, 1.2, 2.4], dtype=np.float32)
    gamma = np.asarray([0.7, 2.1, 3.8], dtype=np.float32)
    stiffness = np.asarray([600.0, 1800.0, 5500.0], dtype=np.float32)
    first = simulate_split(
        3, 0, np.random.SeedSequence(21), cfg,
        system_parameters=(mass, gamma, stiffness),
    )
    second = simulate_split(
        3, 0, np.random.SeedSequence(22), cfg,
        system_parameters=(mass, gamma, stiffness),
    )
    for key, expected in (("mass", mass), ("gamma", gamma), ("stiffness", stiffness)):
        assert np.array_equal(first[key], expected)
        assert np.array_equal(second[key], expected)
    assert not np.array_equal(first["states"], second["states"])
    assert not np.array_equal(first["actions"], second["actions"])
