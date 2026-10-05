"""Project verified private A plans to image/action supervision only.

No physical state files are read. Image targets are supervision, kept separate
from causal histories; StrictVisualJEPA controls target normalization/gradients.
This is a data/training-objective bridge, not a formal experiment entry point.
"""
from pathlib import Path
import hashlib
import numpy as np
from .observations import image_history
from .schema import Config
from .adapters import z_experience


def _visible(root, row):
    root = Path(root).resolve(); asset = row['assets']['64']; path = (root/asset['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('public image asset outside bank or missing')
    expected = asset['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError('public image asset hash differs')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'images', 'actions', 'timestamps'}: raise ValueError('unexpected public image fields')
        images, actions, times = (data[k].copy() for k in ('images', 'actions', 'timestamps'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError('public image asset changed during load')
    n = len(images)
    if images.dtype != np.uint8 or images.shape != (n, 64, 64) or actions.shape != (n-1, 2) or times.shape != (n,):
        raise ValueError('public A training episode shape differs')
    if n != row['raw_frames'] or not np.isfinite(actions).all() or np.any(np.linalg.norm(actions, axis=1) > 1+1e-7):
        raise ValueError('public A episode action or duration support differs')
    if not np.allclose(times, np.arange(n)*.05, rtol=0, atol=1e-8): raise ValueError('public A episode time grid differs')
    return images, actions


def _window(images, actions, start, length):
    anchor = start+length-1
    if start < 0 or anchor+16 >= len(images): raise ValueError('planned A observed/target support missing; no replacement')
    x = images[start:anchor+1].astype(np.float32)/255
    difference = np.zeros_like(x); difference[1:] = x[1:]-x[:-1]
    observations = np.stack((x, difference), axis=1)
    past = np.zeros((length, 2), np.float32); past[1:] = actions[start:anchor]
    mask = np.ones(length, bool); mask[0] = False
    payload = dict(observations=observations, past_actions=past, past_action_mask=mask, relative_times=np.arange(length)*.05)
    image_history(z_experience(payload), Config(resolution=64))
    targets = []; future = np.zeros((3, 16, 2), np.float32); masks = np.zeros((3, 16), np.float32)
    for hi, horizon in enumerate((1, 4, 16)):
        frame = images[anchor+horizon].astype(np.float32)/255
        previous = images[anchor+horizon-1].astype(np.float32)/255
        targets.append(np.stack((frame, frame-previous)))
        future[hi, :horizon] = actions[anchor:anchor+horizon]; masks[hi, :horizon] = 1
    return observations, past[1:], np.stack(targets), future, masks


def make_batch(schedule, plan, batch_index, bank_root):
    """Return (label-free VisualBatch, private receipt), donor branch first."""
    import torch
    from strict_model import VisualBatch
    schedule.validate(plan)
    size = schedule.batch_pairs; total = len(plan['pairs'])//size
    if type(batch_index) is not int or not 0 <= batch_index < total: raise ValueError('incomplete or invalid A pair batch')
    pairs = plan['pairs'][batch_index*size:(batch_index+1)*size]
    windows = []; input_assets = {}; frames_read = 0
    for branch in ('donor', 'recipient'):
        for pair in pairs:
            row = schedule.rows[pair[branch+'_episode']]; start = pair[branch+'_start']
            if start+schedule.length-1+16 >= row['raw_frames']:
                raise ValueError('planned A observed/target support missing; no replacement')
            images, actions = _visible(bank_root, row); frames_read += len(images)
            windows.append(_window(images, actions, start, schedule.length))
            input_assets[row['episode_key']] = row['assets']['64']['sha256']
    # No IDs, theta, split, relation, correctness label or absolute time survives.
    batch = VisualBatch(*[torch.from_numpy(np.stack([window[i] for window in windows])) for i in range(5)])
    batch.validate(schedule.length)
    receipt = dict(plan_sha256=plan['plan_sha256'], batch_index=batch_index, input_assets=input_assets,
        pair_sha256=[p['pair_sha256'] for p in pairs], windows=len(windows), raw_image_frames_read=frames_read,
        presented_history_frames=len(windows)*schedule.length, observed_action_intervals=len(windows)*(schedule.length-1),
        target_frames=len(windows)*3, direction='donor_first_half_to_recipient_second_half',
        target_horizons=[1, 4, 16], decoder=None, physical_labels_read=False, test_read=False, formal_training=False)
    return batch, receipt


def objective(model, batch, sigreg, config):
    """Dispatch the exact existing A objective for one registered configuration."""
    from .a_pairing import configuration
    from strict_model import StrictVisualJEPA, strict_objective
    if config != configuration(config['name']): raise ValueError('A objective configuration changed')
    if not isinstance(model, StrictVisualJEPA) or model.variant != config['variant']:
        raise ValueError('A model variant or strict normalization does not match configuration')
    if getattr(sigreg, 'num_directions', None) != 1024:
        raise ValueError('registered A SIGReg uses 1024 directions; microbatch alternatives need explicit calibration')
    return strict_objective(model, batch, sigreg, sigreg_weight=config['sigreg_weight'],
        lambda_p=config['lambda_p'], lambda_x=config['lambda_x'])
