import numpy as np

from paper_c.swimmer.lqa_prospective import fixed_contexts, particle_pool, system_pool


PRIOR = {"mass_scale": [0.75, 1.25], "damping_scale": [0.60, 1.40]}


def test_system_pool_is_deterministic_and_supports_non_power_of_two_counts():
    first = system_pool(5, 91, PRIOR)
    second = system_pool(5, 91, PRIOR)
    assert first.shape == (5, 5)
    assert np.array_equal(first, second)


def test_particle_pool_requires_nested_power_of_two_design():
    particles = particle_pool(8, 93, PRIOR)
    assert particles.shape == (8, 5)
    try:
        particle_pool(7, 93, PRIOR)
    except ValueError:
        pass
    else:
        raise AssertionError("non-power-of-two particle pool was accepted")


def test_fixed_context_selection_is_hash_deterministic_and_system_unique():
    first = fixed_contexts(7, 4, 6, 19, "test")
    second = fixed_contexts(7, 4, 6, 19, "test")
    assert first.equals(second)
    assert len(first) == 7
    assert first.system_index.nunique() == 7
    assert len(first.drop_duplicates()) == 7


def test_fixed_context_selection_count_means_independent_systems():
    selected = fixed_contexts(128, 4, 6, 32, "test")
    assert len(selected) == 32
    assert selected.system_index.nunique() == 32
