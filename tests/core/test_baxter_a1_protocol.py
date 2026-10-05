from __future__ import annotations

import numpy as np
import torch

from persistent_jepa.baxter_data import BaxterSplit, CONFIGS
from persistent_jepa.baxter_model import BaxterJEPA, BaxterModelConfig, baxter_objective
from persistent_jepa.losses import SIGReg


def test_pairing_blocks_encode_only_registered_semantics() -> None:
    split = object.__new__(BaxterSplit)
    rng = np.random.default_rng(7)
    split.condition = "R_H"
    for block in split._donor_config_blocks(rng, 8):
        for query, donor in block.items():
            assert query[-1] == donor[-1]
            assert query.split("_h")[0] != donor.split("_h")[0]
    split.condition = "R_S"
    for block in split._donor_config_blocks(rng, 8):
        for query, donor in block.items():
            assert query.split("_h")[0] == donor.split("_h")[0]
            assert query[-1] != donor[-1]
    split.condition = "Random"
    for block in split._donor_config_blocks(rng, 8):
        assert set(block) == set(CONFIGS)
        assert set(block.values()) == set(CONFIGS)
        assert all(query != donor for query, donor in block.items())


def test_baxter_objective_is_finite_and_backward_works() -> None:
    model = BaxterJEPA(BaxterModelConfig("R_H"))
    batch = type("Batch", (), {
        "query_history": torch.randn(12, 40, 16),
        "query_targets": torch.randn(12, 3, 16),
        "donor_history": torch.randn(12, 40, 16),
    })()
    loss, metrics = baxter_objective(model, batch, SIGReg(num_directions=32))
    assert torch.isfinite(loss)
    assert set(("loss_self", "loss_persist", "loss_cross")) <= set(metrics)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
