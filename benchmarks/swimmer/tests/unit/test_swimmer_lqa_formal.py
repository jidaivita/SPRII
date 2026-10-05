import json
from pathlib import Path

import pandas as pd

from paper_c.swimmer.lqa_formal import assigned_system_indices, formal_context_table


def test_formal_worker_assignment_is_complete_disjoint_and_resume_stable():
    assignments = [assigned_system_indices(0, 37, worker, 12) for worker in range(12)]
    flattened = [value for values in assignments for value in values]
    assert sorted(flattened) == list(range(37))
    assert len(flattened) == len(set(flattened))
    assert assignments[0] == [0, 12, 24, 36]


def test_formal_context_table_is_one_context_per_frozen_system():
    config = {
        "formal": {
            "pool_max": 32,
            "context_seed": 69103,
            "nuisance_candidates_per_system": 4,
        }
    }
    first = formal_context_table(config)
    second = formal_context_table(config)
    assert first.equals(second)
    assert len(first) == 32
    assert first.system_index.tolist() == list(range(32))
    assert first.system_index.nunique() == 32
    assert set(first.realization).issubset(set(range(4)))
    assert set(first.history_index).issubset(set(range(6)))
