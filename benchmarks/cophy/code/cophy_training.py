"""Native/A/Random/Param-known execution through the same PT adapter."""
import json
import time

import torch

from cophy_adapter import Targets, objective


def targets_from_batch(batch, device):
    return Targets(batch['pose_3D_cd'][:, 1:].to(device),
                   batch['stab_cd'][:, 1:].to(device), batch['presence_cd'].to(device))


def mse_per_recipient(pred, target, presence, dims, allow_uncovered=False):
    if dims not in (2, 3) or pred.shape != target.shape:
        raise ValueError('Invalid prediction target or dimensions')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite model prediction/target; do not filter by model outcomes')
    count=presence.sum(1)*pred.shape[1]
    if (count<=0).any() and not allow_uncovered:
        raise ValueError('Invalid prediction target or zero visual coverage')
    error = (pred[..., :dims]-target[..., :dims]).square().mean(-1)
    score=(error * presence[:, None]).sum((1, 2)) / count.clamp_min(1)
    return torch.where(count>0,score,torch.full_like(score,float('nan')))


def train_adapter_epoch(model, device, loader, optimizer, log_file, *, epoch,
                        pair_provider=None, parameter_store=None, lambda_x=.1,
                        lambda_p=1., is_rgb=False, dims=3):
    if (model.method in {'A', 'Random'}) != (pair_provider is not None):
        raise ValueError('Training method and donor provider disagree')
    if (model.method == 'Param-known') != (parameter_store is not None):
        raise ValueError('Training method and parameter permissions disagree')
    model.train()
    loader.dataset.is_rgb = is_rgb
    started = time.perf_counter()
    recipient_ids, donor_ids = set(), set()
    totals = {'epoch': epoch, 'method': model.method, 'updates': 0, 'recipients': 0,
              'paired': 0, 'encoder_calls': 0, 'donor_examples': 0,
              'randomization_skipped': 0, 'random_same': 0,
              'loss_native_sum': 0., 'loss_cross_sum': 0., 'loss_persist_sum': 0.,
              'vicreg_skipped_batches': 0}
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for batch_number, batch in enumerate(loader):
        visual = model.input_from_batch(batch, device, is_rgb=is_rgb)
        target = targets_from_batch(batch, device)
        pairs, pair_stats = None, {}
        if pair_provider is not None:
            pairs, pair_stats = pair_provider.make(batch['id'], model.method, epoch, batch_number, visual)
        recipient_ids.update(map(str, batch['id']))
        donor_ids.update(pair_stats.get('donor_ids', []))
        parameters = (parameter_store.batch(batch['id'], visual.c.shape[2], device)
                      if parameter_store is not None else None)
        optimizer.zero_grad(set_to_none=True)
        loss, prediction, stats = objective(model, visual, target, pairs, parameters, lambda_x, lambda_p)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite adapter loss')
        loss.backward()
        gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        if gradients and not torch.stack([torch.isfinite(g).all() for _, g in gradients]).all():
            failed = [name for name, g in gradients if not torch.isfinite(g).all()]
            raise FloatingPointError(f'Nonfinite gradients in {failed}')
        optimizer.step()
        totals['updates'] += 1
        totals['recipients'] += len(visual.c)
        totals['paired'] += stats['paired']
        totals['donor_examples'] += stats['paired']
        totals['encoder_calls'] += stats['encoder_calls']
        totals['vicreg_skipped_batches'] += int(stats['vicreg_skipped'])
        for name in ('native', 'cross', 'persist'):
            totals[f'loss_{name}_sum'] += float(stats[name])
        for name in ('randomization_skipped', 'random_same'):
            totals[name] += pair_stats.get(name, 0)
    if not totals['updates']:
        raise ValueError('Empty training loader')
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
        totals['peak_memory_bytes'] = torch.cuda.max_memory_allocated(device)
    totals['seconds'] = time.perf_counter()-started
    totals['unique_recipient_episodes'] = len(recipient_ids)
    totals['unique_donor_episodes'] = len(donor_ids)
    with open(log_file, 'a') as stream:
        stream.write(json.dumps(totals, allow_nan=False)+'\n')
    return totals


@torch.no_grad()
def validate_adapter(model, device, loader, log_file, dims=3, is_rgb=False, parameter_store=None, epoch=None):
    if (model.method == 'Param-known') != (parameter_store is not None):
        raise ValueError('Evaluation parameter permissions disagree')
    model.eval()
    loader.dataset.is_rgb = is_rgb
    values = []; uncovered = 0; recipients = 0
    for batch in loader:
        visual = model.input_from_batch(batch, device, is_rgb=is_rgb)
        parameters = (parameter_store.batch(batch['id'], visual.c.shape[2], device)
                      if parameter_store is not None else None)
        # Evaluation never invokes a training pair sampler.
        pred, presence, _ = model.predict_code(model.code_for_task(visual, parameters), visual)
        target = batch['pose_3D_cd'][:, 1:].to(device)
        score=mse_per_recipient(pred,target,presence,dims,allow_uncovered=True)
        # This mask is fixed by the shared visual frontend, independent of model
        # predictions or donor condition. Record failures rather than zero MSE.
        valid=presence.sum(1)>0
        values.extend(score[valid].cpu().tolist())
        recipients+=len(score); uncovered+=int((~valid).sum())
    if not values:
        raise ValueError('Empty validation loader')
    score = sum(values)/len(values)
    with open(log_file, 'a') as stream:
        stream.write(json.dumps({'mse':score,'recipients':recipients,'scored_recipients':len(values),
            'uncovered_recipients':uncovered,'visual_coverage':len(values)/recipients,
            'method':model.method,'epoch':epoch})+'\n')
    return score
