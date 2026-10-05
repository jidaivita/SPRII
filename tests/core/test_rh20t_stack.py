from __future__ import annotations

from types import SimpleNamespace

import torch

from persistent_jepa.losses import SIGReg
from persistent_jepa.rh20t_model import RH20TJEPA, RH20TModelConfig, rh20t_objective


def synthetic_batch(batch_size: int = 4, lowdim_only: bool = False) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(7)
    image = None if lowdim_only else torch.randn(batch_size, 24, 2, 96, 96, generator=generator)
    target_image = (
        None if lowdim_only else torch.randn(batch_size, 3, 2, 96, 96, generator=generator)
    )
    return SimpleNamespace(
        history_image=image,
        history_lowdim=torch.randn(batch_size, 24, 14, generator=generator),
        history_actions=torch.randn(batch_size, 23, 1, generator=generator),
        donor_history_image=(
            None if lowdim_only else torch.randn(batch_size, 24, 2, 96, 96, generator=generator)
        ),
        donor_history_lowdim=torch.randn(batch_size, 24, 14, generator=generator),
        donor_history_actions=torch.randn(batch_size, 23, 1, generator=generator),
        target_image=target_image,
        target_lowdim=torch.randn(batch_size, 3, 14, generator=generator),
        target_force=torch.randn(batch_size, 3, 6, generator=generator),
        target_tcp_xyz=torch.randn(batch_size, 3, 3, generator=generator),
        future_actions=torch.randn(batch_size, 3, 16, 1, generator=generator),
        action_masks=torch.ones(batch_size, 3, 16),
    )


def test_all_conditions_have_same_prediction_heads() -> None:
    b0 = RH20TJEPA(RH20TModelConfig("B0"))
    b3 = RH20TJEPA(RH20TModelConfig("B3-Indep"))
    assert b0.predictor.latent.weight.shape == b3.predictor.latent.weight.shape
    assert b0.predictor.force.weight.shape == b3.predictor.force.weight.shape
    assert b0.predictor.tcp_xyz.weight.shape == b3.predictor.tcp_xyz.weight.shape


def test_monolithic_qd_has_one_joint_context_and_common_objective_only() -> None:
    model = RH20TJEPA(RH20TModelConfig("Mono-QD-Indep"))
    batch = synthetic_batch()
    loss, metrics = rh20t_objective(
        model,
        batch,
        SIGReg(knots=5, num_directions=16),
    )
    assert torch.isfinite(loss)
    assert model.joint_context is not None
    assert model.context is None
    assert model.transient is None
    assert model.persistent is None
    assert "loss_self" in metrics
    assert "loss_sigreg" in metrics
    assert "loss_persist" not in metrics
    assert "loss_cross" not in metrics
    loss.backward()
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.joint_context.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_monolithic_qd_prediction_changes_with_donor() -> None:
    model = RH20TJEPA(RH20TModelConfig("Mono-QD-Indep"))
    model.eval()
    batch = synthetic_batch()
    with torch.no_grad():
        query_h, target_h = model.encode(
            batch.history_image, batch.history_lowdim, batch.target_image, batch.target_lowdim
        )
        donor_h = model.observation(batch.donor_history_image, batch.donor_history_lowdim)
        context_a = model.joint_context(
            query_h, batch.history_actions, donor_h, batch.donor_history_actions
        )
        rolled = donor_h.roll(1, dims=0)
        rolled_actions = batch.donor_history_actions.roll(1, dims=0)
        context_b = model.joint_context(
            query_h, batch.history_actions, rolled, rolled_actions
        )
    assert not torch.allclose(context_a, context_b)


def test_b3_objective_is_finite_and_backpropagates() -> None:
    model = RH20TJEPA(RH20TModelConfig("B3-Indep"))
    loss, metrics = rh20t_objective(
        model,
        synthetic_batch(),
        SIGReg(knots=5, num_directions=16),
        lambda_p=1.0,
        lambda_x=0.1,
    )
    assert torch.isfinite(loss)
    assert {"loss_self", "loss_cross", "loss_persist"} <= set(metrics)
    loss.backward()
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.persistent.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_donor_changes_shared_trunk_force_and_tcp_and_receives_gradients() -> None:
    model = RH20TJEPA(RH20TModelConfig("B3-Indep"))
    model.eval()
    batch = synthetic_batch()
    history_h, _ = model.encode(
        batch.history_image, batch.history_lowdim, batch.target_image, batch.target_lowdim
    )
    z_s, z_p, _ = model.codes(history_h, batch.history_actions)
    query = 2
    horizon_index = torch.tensor([2])
    output_a = model.predictor(
        torch.cat([z_s[query : query + 1], z_p[0:1]], dim=-1),
        batch.future_actions[query : query + 1, 2],
        batch.action_masks[query : query + 1, 2],
        horizon_index,
    )
    output_b = model.predictor(
        torch.cat([z_s[query : query + 1], z_p[1:2]], dim=-1),
        batch.future_actions[query : query + 1, 2],
        batch.action_masks[query : query + 1, 2],
        horizon_index,
    )
    assert not torch.allclose(output_a["q_h"], output_b["q_h"])
    assert not torch.allclose(output_a["force"], output_b["force"])
    assert not torch.allclose(output_a["tcp_xyz"], output_b["tcp_xyz"])
    model.zero_grad(set_to_none=True)
    (output_a["force"].square().mean() + output_a["tcp_xyz"].square().mean()).backward()
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.persistent.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_lowdim_condition_never_requires_rgb() -> None:
    model = RH20TJEPA(RH20TModelConfig("B3-LowDim-Same"))
    loss, _ = rh20t_objective(
        model,
        synthetic_batch(lowdim_only=True),
        SIGReg(knots=5, num_directions=16),
    )
    assert torch.isfinite(loss)


def test_closure_condition_semantics_are_exact() -> None:
    expected = {
        "B0split": (True, False, False, False, "independent"),
        "B2-Indep": (True, True, False, False, "independent"),
        "B2-Random": (True, True, False, False, "random"),
        "Bx-Random": (True, False, True, False, "random"),
        "LowDim-B0": (False, False, False, True, None),
        "LowDim-Bx-Indep": (True, False, True, True, "independent"),
        "Mono-QD-Indep": (False, False, False, False, "independent"),
    }
    for condition, values in expected.items():
        cfg = RH20TModelConfig(condition)
        assert (cfg.split, cfg.persistence, cfg.cross, cfg.lowdim_only, cfg.donor_relation) == values


def test_closure_objective_decomposition() -> None:
    expected_metrics = {
        "B0split": {"loss_self", "loss_sigreg"},
        "B2-Indep": {"loss_self", "loss_sigreg", "loss_persist"},
        "B2-Random": {"loss_self", "loss_sigreg", "loss_persist"},
        "Bx-Random": {"loss_self", "loss_sigreg", "loss_cross"},
        "LowDim-B0": {"loss_self", "loss_sigreg"},
        "LowDim-Bx-Indep": {"loss_self", "loss_sigreg", "loss_cross"},
    }
    for condition, required in expected_metrics.items():
        cfg = RH20TModelConfig(condition)
        model = RH20TJEPA(cfg)
        _, metrics = rh20t_objective(
            model,
            synthetic_batch(lowdim_only=cfg.lowdim_only),
            SIGReg(knots=5, num_directions=16),
        )
        assert required <= set(metrics)
        assert ("loss_persist" in metrics) == cfg.persistence
        assert ("loss_cross" in metrics) == cfg.cross
