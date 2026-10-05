"""CoPhy A adapter: swap only P; retain recipient T and C.

The public input types deliberately separate observations from supervision.
P/T denote trainable roles, not guaranteed disentanglement. No new backbone
parameters are introduced for Native, A or Random.
"""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from cophy_protocol import vicreg_focal

ADAPTER_VERSION = 'cophy-pt16-v3'
P_DIM = 16
U_DIM = 32


def _finite(x, name):
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be a finite floating tensor')


@dataclass(frozen=True)
class ABObservation:
    pose: torch.Tensor                 # B,T,K,3, estimated from AB images
    presence: torch.Tensor             # B,K, estimated from AB

    def __post_init__(self):
        if (self.pose.ndim != 4 or self.pose.shape[-1] != 3 or
                self.pose.shape[1] < 2 or
                self.presence.shape != (self.pose.shape[0], self.pose.shape[2])):
            raise ValueError('AB requires B,T>=2,K,3 and B,K presence')
        _finite(self.pose, 'AB pose')
        _finite(self.presence, 'AB presence')
        if not ((self.presence == 0) | (self.presence == 1)).all():
            raise ValueError('AB presence must be binary')


@dataclass(frozen=True)
class VisualInput:
    ab: ABObservation
    c: torch.Tensor                    # B,1,K,3, exactly one image
    presence_c: torch.Tensor           # B,K

    def __post_init__(self):
        b, _, k, _ = self.ab.pose.shape
        if self.c.shape != (b, 1, k, 3) or self.presence_c.shape != (b, k):
            raise ValueError('Recipient input requires exactly one C frame')
        _finite(self.c, 'C pose')
        _finite(self.presence_c, 'C presence')
        if not ((self.presence_c == 0) | (self.presence_c == 1)).all():
            raise ValueError('C presence must be binary')

    def select(self, rows):
        return VisualInput(ABObservation(self.ab.pose[rows], self.ab.presence[rows]),
                           self.c[rows], self.presence_c[rows])


@dataclass(frozen=True)
class Targets:
    pose: torch.Tensor
    stationary: torch.Tensor
    presence: torch.Tensor

    def select(self, rows):
        return Targets(self.pose[rows], self.stationary[rows], self.presence[rows])


@dataclass(frozen=True)
class DonorPairs:
    rows: torch.Tensor                 # eligible recipient rows only
    focal: torch.Tensor
    donor: ABObservation               # donor has no C/D/labels/parameters
    donor_slot: torch.Tensor


class PersistentNullBank:
    """Active train-object P means, bound to one selected checkpoint.

    Keys are the frozen (slot, known_type) strata, never physical attributes.
    Fit after selecting a checkpoint; do not update it on val/test objects.
    """
    def __init__(self, checkpoint_sha256):
        if not checkpoint_sha256:
            raise ValueError('Null bank requires a checkpoint identity')
        self.checkpoint_sha256 = checkpoint_sha256
        self.sums, self.counts = {}, {}

    def update(self, persistent, active, keys, *, split):
        if split != 'train':
            raise ValueError('Null means are fitted only on the training split')
        if persistent.ndim != 3 or persistent.shape[-1] != P_DIM or active.shape != persistent.shape[:2]:
            raise ValueError('Null bank requires P16 and active-object masks')
        if len(keys) != len(persistent) or any(len(row) != persistent.shape[1] for row in keys):
            raise ValueError('Null strata do not align with objects')
        _finite(persistent, 'Null bank P')
        for b, row in enumerate(keys):
            for k, key in enumerate(row):
                if len(key) != 2 or key[0] != k:
                    raise ValueError('Null strata must retain the actual object slot and known type')
                if active[b, k] > 0:
                    key = tuple(key)
                    p = persistent[b, k].detach().cpu().double()
                    self.sums[key] = self.sums.get(key, torch.zeros(P_DIM, dtype=torch.float64)) + p
                    self.counts[key] = self.counts.get(key, 0) + 1

    def lookup(self, keys, *, checkpoint_sha256, device, dtype=torch.float32):
        if checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError('Null bank belongs to another checkpoint')
        values = []
        for key in keys:
            key = tuple(key)
            if key not in self.counts:
                raise ValueError('Uncovered null stratum; do not silently use another mean')
            values.append(self.sums[key]/self.counts[key])
        return torch.stack(values).to(device=device, dtype=dtype)


def split_code(u):
    if u.ndim != 3 or u.shape[-1] != U_DIM:
        raise ValueError('Expected B,K,32 code')
    return u[..., :P_DIM], u[..., P_DIM:]


def replace_p(u, focal, p):
    """Functional replacement: preserve T and every nonfocal U exactly."""
    persistent, transient = split_code(u)
    b, k, _ = u.shape
    if (focal.shape != (b,) or focal.dtype != torch.long or
            (focal < 0).any() or (focal >= k).any() or p.shape != (b, P_DIM)):
        raise ValueError('Invalid focal slot or persistent replacement')
    _finite(p, 'P replacement')
    mask = F.one_hot(focal, k).to(u.dtype).unsqueeze(-1)
    return torch.cat((persistent * (1-mask) + p[:, None] * mask, transient), -1)


class PTCoPhy(nn.Module):
    def __init__(self, backbone, method='Native', parameter_dim=None):
        super().__init__()
        if method not in {'Native', 'A', 'Random', 'Param-known'}:
            raise ValueError('Unknown method')
        if (method == 'Param-known') != (parameter_dim is not None):
            raise ValueError('Only Param-known accepts audited parameter inputs')
        self.backbone = backbone
        self.method = method
        self.parameter_dim = parameter_dim
        self.parameter_encoder = (nn.Sequential(nn.Linear(parameter_dim, 32), nn.ReLU(),
                                               nn.Linear(32, P_DIM))
                                  if parameter_dim is not None else None)
        for p in self.backbone.derendering.parameters():
            p.requires_grad_(False)
        self.backbone.derendering.eval()

    @property
    def derendering(self):
        return self.backbone.derendering

    def train(self, mode=True):
        super().train(mode)
        self.derendering.eval()
        return self

    def encode_ab(self, ab):
        return self.backbone.rnn_on_AB_up(self.backbone.gcn_on_AB(ab.pose, ab.presence))

    def input_from_batch(self, batch, device, is_rgb=False):
        # Allowlist fields: this function never reads targets, metadata, or future CD.
        if is_rgb:
            from cf_learning.model import extract_pose_ab_c
            pa, pc, xa, xc = extract_pose_ab_c(
                self.derendering, batch['rgb_ab'].to(device), batch['rgb_cd'][:, :1].to(device))
        else:
            pa = batch['pred_presence_ab'].to(device)
            pc = batch['pred_presence_cd'].to(device)
            xa = batch['pred_pose_3D_ab'].to(device)
            xc = batch['pred_pose_3D_cd'].to(device)  # reject a future-bearing cache
        return VisualInput(ABObservation(xa, pa), xc, pc)

    def code_for_task(self, visual, parameters=None):
        u = self.encode_ab(visual.ab)
        if self.method == 'Param-known':
            if parameters is None or parameters.shape != (*u.shape[:2], self.parameter_dim):
                raise ValueError('Missing all audited parameter features')
            _finite(parameters, 'Parameter features')
            _, transient = split_code(u)
            u = torch.cat((self.parameter_encoder(parameters), transient), -1)
        elif parameters is not None:
            raise ValueError('GT parameter input forbidden for this method')
        return u

    def predict_code(self, u, visual):
        split_code(u)
        pose, stability = self.backbone.pred_D(
            u, visual.c[:, 0], visual.presence_c, T=visual.ab.pose.shape[1]-1)
        return pose, visual.presence_c, stability.squeeze(-1)

    def forward(self, rgb_ab, rgb_c, pred_presence_ab=None, pred_pose_3d_ab=None,
                pred_presence_c=None, pred_pose_3d_c=None, *, parameters=None):
        if rgb_ab is not None:
            from cf_learning.model import extract_pose_ab_c
            pa, pc, xa, xc = extract_pose_ab_c(self.derendering, rgb_ab, rgb_c)
        else:
            pa, pc, xa, xc = pred_presence_ab, pred_presence_c, pred_pose_3d_ab, pred_pose_3d_c
        visual = VisualInput(ABObservation(xa, pa), xc, pc)
        return self.predict_code(self.code_for_task(visual, parameters), visual)


def task_loss(prediction, target):
    pose, _, stability = prediction
    if (pose.shape != target.pose.shape or stability.shape != target.stationary.shape or
            target.presence.shape != (pose.shape[0], pose.shape[2])):
        raise ValueError('Targets do not align with recipient prediction')
    _finite(target.pose, 'Target pose')
    _finite(target.stationary, 'Stationary target')
    mask = target.presence[:, None].expand_as(stability)
    if mask.sum() <= 0:
        raise ValueError('No active target objects')
    mse = ((pose-target.pose).square().mean(-1) * mask).sum() / mask.sum()
    bce = (F.binary_cross_entropy_with_logits(stability, target.stationary,
                                             reduction='none') * mask).sum() / mask.sum()
    return 10*mse + bce


def objective(model, visual, target, pairs=None, parameters=None, lambda_x=.1, lambda_p=1.):
    """One native task plus eligible P-only cross and invariance tasks."""
    if lambda_x < 0 or lambda_p < 0:
        raise ValueError('Negative loss coefficient')
    u = model.code_for_task(visual, parameters)
    native_pred = model.predict_code(u, visual)
    native = task_loss(native_pred, target)
    zero = u.sum() * 0
    cross, persist = zero, zero
    details = {'native': native.detach(), 'cross': zero.detach(), 'persist': zero.detach(),
               'paired': 0, 'vicreg_skipped': True}
    if model.method in {'A', 'Random'}:
        if pairs is None:
            raise ValueError('A/Random requires an explicit pair batch, possibly empty')
        rows = pairs.rows
        if (rows.dtype != torch.long or rows.ndim != 1 or len(rows.unique()) != len(rows) or
                (rows < 0).any() or (rows >= len(u)).any()):
            raise ValueError('Invalid recipient row selection')
        if len(rows):
            recipient = visual.select(rows)
            if len(pairs.focal) != len(rows) or len(pairs.donor.pose) != len(rows):
                raise ValueError('Misaligned pair batch')
            if (pairs.focal.shape != rows.shape or pairs.focal.dtype != torch.long or
                    (pairs.focal < 0).any() or (pairs.focal >= u.shape[1]).any()):
                raise ValueError('Recipient focal out of range')
            if pairs.donor_slot.shape != pairs.focal.shape or pairs.donor_slot.dtype != torch.long:
                raise ValueError('Invalid donor slots')
            donor_u = model.encode_ab(pairs.donor)
            if (pairs.donor_slot < 0).any() or (pairs.donor_slot >= donor_u.shape[1]).any():
                raise ValueError('Donor slot out of range')
            arange = torch.arange(len(rows), device=u.device)
            if (recipient.ab.presence[arange, pairs.focal] <= 0).any() or (recipient.presence_c[arange, pairs.focal] <= 0).any():
                raise ValueError('Recipient focal is not visible in AB and C')
            if (pairs.donor.presence[arange, pairs.donor_slot] <= 0).any():
                raise ValueError('Donor focal is absent')
            recipient_p = split_code(u[rows])[0][arange, pairs.focal]
            donor_p = split_code(donor_u)[0][arange, pairs.donor_slot]
            mixed = replace_p(u[rows], pairs.focal, donor_p)
            cross = task_loss(model.predict_code(mixed, recipient), target.select(rows))
            persist, stats = vicreg_focal(recipient_p, donor_p)
            details.update(cross=cross.detach(), persist=persist.detach(),
                           paired=len(rows), vicreg_skipped=stats['skipped'])
    elif pairs is not None:
        raise ValueError('Native/Param-known must not consume training donors')
    details['encoder_calls'] = 1 + int(details['paired'] > 0)
    return native + lambda_x*cross + lambda_p*persist, native_pred, details


@torch.no_grad()
def probe_views(model, visual):
    """Same checkpoint: P16/T16/U32 readouts distinguish organization and accessibility."""
    model.eval()
    if model.method == 'Param-known':
        raise ValueError('GT parameter embedding is not a learned formation probe')
    u = model.encode_ab(visual.ab)
    p, t = split_code(u)
    return {'P': p, 'T': t, 'U': u}


@torch.no_grad()
def donor_assay(model, visual, target, focal, donors: Mapping, null_p, metric_dims=3):
    """All rows are pre-frozen recipient-focal opportunities; reduce by recipient later.

    donors maps condition -> (ABObservation, donor_slot). Caller obtains null_p
    from the same checkpoint's train-only, active-slot/type P mean.
    """
    model.eval()
    if model.method == 'Param-known' or metric_dims not in (2, 3):
        raise ValueError('Invalid donor assay method or metric dimensions')
    if not {'Correct', 'Wrong-any'} <= donors.keys() or set(donors) & {'Own', 'Null', 'Null-zero'}:
        raise ValueError('Need Correct/Wrong-any with reserved names unchanged')
    u = model.encode_ab(visual.ab)
    b = len(u)
    rows = torch.arange(b, device=u.device)
    if focal.shape != (b,) or focal.dtype != torch.long or (focal < 0).any() or (focal >= u.shape[1]).any():
        raise ValueError('Invalid assay focal')
    if (visual.ab.presence[rows, focal] <= 0).any() or (visual.presence_c[rows, focal] <= 0).any():
        raise ValueError('Assay focal must be observed in AB and C')
    codes = {'Own': u, 'Null': replace_p(u, focal, null_p),
             'Null-zero': replace_p(u, focal, torch.zeros_like(null_p))}
    for condition, (ab, slots) in donors.items():
        if len(ab.pose) != b or slots.shape != (b,) or slots.dtype != torch.long:
            raise ValueError('Misaligned assay donor')
        if (slots < 0).any() or (slots >= ab.pose.shape[2]).any() or (ab.presence[rows, slots] <= 0).any():
            raise ValueError('Invalid assay donor slot')
        p = split_code(model.encode_ab(ab))[0][rows, slots]
        codes[condition] = replace_p(u, focal, p)
    result = {}
    for condition, code in codes.items():
        pred, mask, _ = model.predict_code(code, visual)
        if pred.shape != target.pose.shape or (mask.sum(1) <= 0).any():
            raise ValueError('Invalid assay targets or scene coverage')
        error = (pred[..., :metric_dims] - target.pose[..., :metric_dims]).square().mean(-1)
        result[condition] = {
            'focal_mse': error[rows, :, focal].mean(1).cpu(),
            'scene_mse': ((error * mask[:, None]).sum((1, 2)) /
                          (mask.sum(1)*error.shape[1])).cpu(),
        }
    return result
