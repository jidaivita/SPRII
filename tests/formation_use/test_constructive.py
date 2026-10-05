import torch

from sprii_next.constructive import RecipientBase, ResidualReader


def batch(n=5):
    return (torch.randn(n, 128), torch.randn(n, 64), torch.randn(n, 16, 2),
            torch.ones(n, 16), torch.tensor([0, 1, 2, 3, 4][:n]))


def test_constructive_shapes_and_finiteness():
    q, p, actions, mask, horizon = batch()
    base = RecipientBase(0)
    for arm in ResidualReader.ARMS[:-1]:
        route = ResidualReader(arm, 1, base_state=base.state_dict(),
                               w=torch.linspace(.25, 1., 8))
        pred = route(q, p, actions, mask, horizon)
        assert pred.shape == (len(q), 8)
        assert torch.isfinite(pred).all()


def test_m2_zero_is_exact_b0_and_m1_is_persistent_invariant():
    q, p, actions, mask, horizon = batch()
    base = RecipientBase(0)
    m1 = ResidualReader('m1', 1, base_state=base.state_dict())
    m2 = ResidualReader('m2', 1, base_state=base.state_dict())
    assert m2.zero_parity(q, actions, mask, horizon) < 2e-6
    p2 = torch.randn_like(p)
    assert torch.allclose(m1(q, p, actions, mask, horizon),
                          m1(q, p2, actions, mask, horizon))


def test_physics_weights_are_shared_between_m1_and_m2():
    q, p, actions, mask, horizon = batch()
    base = RecipientBase(0)
    w = torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8])
    m1 = ResidualReader('m1_phys', 1, base_state=base.state_dict(), w=w)
    m2 = ResidualReader('m2_phys', 1, base_state=base.state_dict(), w=w)
    assert torch.equal(m1.physics_weight, m2.physics_weight)
    assert m1.architecture()['parameters'] == m2.architecture()['parameters']
    assert torch.allclose(m1.physics_weight, w)
