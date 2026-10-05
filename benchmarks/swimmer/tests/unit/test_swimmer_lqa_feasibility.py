import numpy as np

from paper_c.swimmer.lqa_feasibility import formal_stopping_decision, paired_difference_bound


CONFIG = {
    "formal": {"stopping_rule": {
        "initial_systems": 512,
        "additional_block_systems": 128,
        "maximum_systems": 1024,
        "continue_when_contributing_below": 256,
        "limited_scope_minimum_contributing": 128,
    }}
}


def test_paired_difference_bound_uses_common_scramble_differences():
    shared = np.array([0.10, -0.20, 0.05, 0.15])
    first = shared + 0.002
    second = shared
    mean, se, bound = paired_difference_bound(first, second)
    assert np.isclose(mean, 0.002)
    assert se < 1e-15
    assert np.isclose(bound, 0.002)


def test_paired_difference_bound_is_orientation_symmetric():
    first = np.array([0.1, 0.2, 0.3, 0.4])
    second = np.array([0.0, 0.3, 0.1, 0.2])
    forward = paired_difference_bound(first, second)
    reverse = paired_difference_bound(second, first)
    assert np.isclose(forward[0], -reverse[0])
    assert np.isclose(forward[1], reverse[1])
    assert np.isclose(forward[2], reverse[2])


def test_formal_stopping_rule_is_outcome_blind_and_block_deterministic():
    assert formal_stopping_decision(512, 256, CONFIG)["scope"] == "NORMAL"
    assert formal_stopping_decision(512, 200, CONFIG) == {
        "action": "CONTINUE", "scope": "PENDING", "next_systems": 640,
    }
    assert formal_stopping_decision(1024, 200, CONFIG)["scope"] == "LIMITED"
    assert formal_stopping_decision(1024, 100, CONFIG)["scope"] == "LOW_COVERAGE_ASSAY_NO_GO"


def test_formal_stopping_rule_rejects_adaptive_nonblock_counts():
    try:
        formal_stopping_decision(700, 200, CONFIG)
    except ValueError:
        pass
    else:
        raise AssertionError("non-frozen formal block boundary was accepted")
