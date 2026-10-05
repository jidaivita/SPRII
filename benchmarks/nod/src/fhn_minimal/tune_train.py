"""Explicit late-stage adaptations around the unchanged minimal trainer."""
import argparse
import sys

import torch

from fhn_minimal import train as core
from nod_sprii.canonical_losses import canonical_vicreg


def weighted_prediction_loss(prediction, target, indicators, power):
    per_target = (prediction - target).square().mean(dim=(2, 3, 4))
    offsets = torch.tensor([1, 3, 9, 27], device=indicators.device)[indicators].cumsum(1)
    weights = offsets.float().pow(power)
    weights = weights / weights.mean(dim=1, keepdim=True)
    return (per_target * weights).mean()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--horizon-weight-power', type=float, default=0.)
    parser.add_argument('--freeze-encoder', action='store_true')
    options, remaining = parser.parse_known_args()
    if options.horizon_weight_power < 0:
        raise ValueError('horizon weight power must be nonnegative')
    if options.freeze_encoder and '--lambda-align' in remaining:
        if float(remaining[remaining.index('--lambda-align') + 1]) != 0:
            raise ValueError('frozen encoder requires zero Align weight')
    original_step = core.training_step
    original_save = core.atomic_save

    def step(model, optimizer, batch, criterion, locs, align_weight):
        if options.freeze_encoder:
            model.conditioning_encoder.requires_grad_(False)
            model.conditioning_encoder.eval()
        if options.horizon_weight_power == 0:
            return original_step(model, optimizer, batch, criterion, locs, align_weight)
        x, c1, c2, target, indicators, _ = batch
        x, c1, target, indicators = [v.to(locs.device) for v in (x, c1, target, indicators)]
        prediction, z1 = core.predict_four(model, x, c1, indicators, locs)
        base = criterion(prediction, target)
        objective = weighted_prediction_loss(prediction, target, indicators, options.horizon_weight_power)
        align = base.new_zeros(())
        metrics = {}
        if align_weight > 0:
            z2 = model.conditioning_encoder(c2.to(locs.device))
            align, am = canonical_vicreg(z1, z2)
            metrics.update({k: float(v) for k, v in am.items()})
        total = objective + align_weight * align
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        metrics.update(base_loss=float(base.detach()), prediction_objective=float(objective.detach()),
                       align_loss=float(align.detach()), total_loss=float(total.detach()), lambda_align=align_weight)
        return metrics

    def save(payload, path):
        payload['tuning_options'] = vars(options)
        payload['resume_data_policy'] = 'seeded data stream restart; optimizer and scheduler restored'
        original_save(payload, path)

    core.training_step = step
    core.atomic_save = save
    sys.argv = [sys.argv[0]] + remaining
    core.main()


if __name__ == '__main__':
    main()
