import json

import numpy as np
import pytest
import torch

from paper_c.coupled_sled.learner import (
    CANONICAL_ARCHITECTURE_TAG,
    MASKED_GRU_ARCHITECTURE_TAG,
    MaskedGRUPersistentAggregator,
    build_persistent_jepa,
    load_tagged_persistent_jepa,
    tagged_checkpoint_payload,
)
from paper_c.stage2.masked_gru_cross_architecture import (
    _job_table,
    exact_system_half_split,
    load_cfg,
)


def _cfg(aggregator_type="concat_mlp"):
    return {
        "hidden_dim": 16,
        "history_embedding_dim": 8,
        "persistent_dim": 6,
        "query_embedding_dim": 5,
        "aggregator_type": aggregator_type,
    }


def test_canonical_persistent_interface_is_bitwise_legacy_equivalent():
    torch.manual_seed(19)
    model = build_persistent_jepa(12, 10, 8, _cfg(), CANONICAL_ARCHITECTURE_TAG).eval()
    history = torch.randn(7, 2, 12)
    mask = torch.tensor([[1, 0], [1, 1], [0, 0], [1, 1], [1, 0], [1, 1], [1, 1]], dtype=torch.float32)
    encoded = model.segment_encoder(history) * mask[:, :, None]
    legacy = model.aggregator(torch.cat((encoded.flatten(1), mask), dim=1))
    interface = model.persistent(history, mask)
    assert torch.equal(interface, legacy)


def test_masked_gru_has_frozen_integer_width_counts_and_exact_mask_invariance():
    expected = [(128, 64, 160, 149504), (64, 32, 80, 37632)]
    for embedding, persistent, hidden, parameters in expected:
        module = MaskedGRUPersistentAggregator(embedding, persistent)
        assert module.hidden_dim == hidden
        assert module.parameter_count == parameters
    torch.manual_seed(23)
    model = build_persistent_jepa(12, 10, 8, _cfg("masked_gru"), MASKED_GRU_ARCHITECTURE_TAG).eval()
    history = torch.randn(6, 2, 12)
    changed = history.clone()
    changed[:, 1] = torch.randn_like(changed[:, 1]) * 1000
    mask = torch.tensor([[1, 0]] * 6, dtype=torch.float32)
    assert torch.equal(model.persistent(history, mask), model.persistent(changed, mask))


def test_masked_gru_rejects_nonintegral_width_and_nonbinary_mask():
    with pytest.raises(ValueError, match="divisible by four"):
        MaskedGRUPersistentAggregator(5, 3)
    module = MaskedGRUPersistentAggregator(8, 3)
    with pytest.raises(ValueError, match="binary"):
        module(torch.randn(2, 2, 8), torch.tensor([[1.0, 0.5], [1.0, 1.0]]))


def test_masked_gru_mixed_mask_backward_reaches_recurrent_parameters():
    torch.manual_seed(29)
    module = MaskedGRUPersistentAggregator(8, 4)
    encoded = torch.randn(5, 2, 8, requires_grad=True)
    mask = torch.tensor([[1, 0], [1, 1], [0, 0], [1, 1], [1, 0]], dtype=torch.float32)
    module(encoded, mask).square().mean().backward()
    assert module.cell.weight_ih.grad is not None
    assert module.cell.weight_hh.grad is not None
    assert torch.isfinite(module.cell.weight_ih.grad).all()


def test_factory_and_tagged_checkpoint_fail_closed(tmp_path):
    model = build_persistent_jepa(12, 10, 8, _cfg("masked_gru"), MASKED_GRU_ARCHITECTURE_TAG)
    checkpoint = tmp_path / "tagged.pt"
    torch.save(tagged_checkpoint_payload(model, {"history_dim": 12, "query_dim": 10, "target_dim": 8}), checkpoint)
    loaded = load_tagged_persistent_jepa(
        checkpoint, _cfg("masked_gru"), MASKED_GRU_ARCHITECTURE_TAG, torch.device("cpu"),
    )
    assert loaded.architecture_tag == MASKED_GRU_ARCHITECTURE_TAG
    with pytest.raises(RuntimeError, match="architecture tag mismatch"):
        load_tagged_persistent_jepa(
            checkpoint, _cfg(), CANONICAL_ARCHITECTURE_TAG, torch.device("cpu"),
        )
    raw_checkpoint = tmp_path / "raw.pt"
    torch.save(model.state_dict(), raw_checkpoint)
    with pytest.raises(RuntimeError, match="envelope"):
        load_tagged_persistent_jepa(
            raw_checkpoint, _cfg("masked_gru"), MASKED_GRU_ARCHITECTURE_TAG, torch.device("cpu"),
        )


def test_exact_select_responsibility_split_is_deterministic_disjoint_and_complete():
    systems = np.repeat(np.arange(256), 4)
    first = exact_system_half_split(systems, "articulated", 84501)
    second = exact_system_half_split(systems[::-1], "articulated", 84501)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    checkpoint, eligibility = first
    assert len(checkpoint) == len(eligibility) == 128
    assert np.intersect1d(checkpoint, eligibility).size == 0
    assert set(np.r_[checkpoint, eligibility]) == set(range(256))


@pytest.mark.skip(reason="Requires historical protocol/machine-command provenance fixtures, excluded from the source release")
def test_frozen_config_and_four_job_schedule(repo_root):
    config_path = repo_root / "configs/masked_gru_cross_architecture_v1.json"
    cfg = load_cfg(config_path)
    jobs = _job_table(repo_root, cfg)
    assert len(jobs) == 4
    assert set(zip(jobs.environment, jobs.base_seed.astype(int))) == {
        ("articulated", 64101), ("articulated", 64103),
        ("coupled", 47001), ("coupled", 47003),
    }
    assert cfg["adapter"]["fits"] == 12
    assert cfg["formal"]["enabled_in_this_implementation_stage"] is False
    assert cfg["probe_protocol_sha256"] == __import__("hashlib").sha256(
        (repo_root / cfg["probe_protocol"]).read_bytes()
    ).hexdigest()


@pytest.fixture
def repo_root():
    return __import__("pathlib").Path(__file__).resolve().parents[2]
