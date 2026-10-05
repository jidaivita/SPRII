"""Small CPU-only example of independent interactions and relation controls.

The budget and seed below are illustrative, not the paper experiment recipe.
"""
from __future__ import annotations

import numpy as np

from persistent_jepa import DCleanConfig, generate_dataset
from persistent_jepa.sampling import (
    build_collision_free_pseudo_systems,
    sample_same_system_pairs,
)


def main() -> None:
    cfg = DCleanConfig(train_systems=8, val_systems=3, test_systems=3,
                       rollouts_per_system=4, seed=7)
    dataset = generate_dataset(cfg)
    pairs = sample_same_system_pairs(8, 4, pairs=16, seed=11)
    wrong = build_collision_free_pseudo_systems(8, 4, seed=13)
    assert np.all(pairs[:, 1] != pairs[:, 3])
    assert all(len(np.unique(row)) == 4 for row in wrong)
    split_ids = [set(dataset.system_ids[s].tolist()) for s in ('train', 'val', 'test')]
    assert all(not a.intersection(b) for i, a in enumerate(split_ids) for b in split_ids[i+1:])
    print('D-Clean independent-interaction example')
    print('Training states:', dataset.states['train'].shape)
    print('Pair columns: system, rollout A, anchor A, rollout B, anchor B')
    print(pairs[:3])
    print('Disjoint splits and collision-free wrong-relation control: verified')


if __name__ == '__main__':
    main()

