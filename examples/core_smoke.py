"""A small CPU demonstration of simulation, pairing and objective gradients.

This uses synthetic data and an untrained model; it is not a paper result.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import torch
from persistent_jepa.simulator import DCleanConfig, generate_dataset
from persistent_jepa.certificates import system_gamma_estimates, r2_score_system
from persistent_jepa.torch_data import SplitArrays
from persistent_jepa.model import ModelConfig, PersistentJEPA
from persistent_jepa.losses import SIGReg
from persistent_jepa.objective import compute_objective


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = DCleanConfig(train_systems=12, val_systems=3, test_systems=3,
                       rollouts_per_system=4)
    dataset = generate_dataset(cfg)
    estimate, coverage = system_gamma_estimates(dataset.states['train'],
                                               dataset.actions['train'], cfg.dt)
    with TemporaryDirectory() as directory:
        dataset.save(Path(directory))
        batch = SplitArrays(Path(directory), 'train').paired_batch(4, 0)
        model = PersistentJEPA(ModelConfig(variant='B3', transformer_depth=1,
                                           dropout=0.0))
        loss, components = compute_objective(model, batch, SIGReg(num_directions=16),
                                             sigreg_weight=0.02,
                                             lambda_p=0.3, lambda_x=0.3)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.parameters())
        print(json.dumps({
            'purpose': 'synthetic interface smoke test; untrained model',
            'history_shape': list(batch.history_states.shape),
            'loss_is_finite': bool(torch.isfinite(loss)),
            'parameter_tensors_with_gradient': sum(p.grad is not None for p in model.parameters()),
            'analytic_drag_probe_r2': float(r2_score_system(dataset.gamma['train'], estimate)),
            'loss_terms': sorted(components),
        }, indent=2))


if __name__ == '__main__':
    main()
