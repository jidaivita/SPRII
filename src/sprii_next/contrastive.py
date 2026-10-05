"""Fixed-budget FCRL-style relation contrast, adapted to temporal histories.

This is Rel-InfoNCE, not a reproduction of the paper's set encoder. Architecture,
temperature and budget are fixed before outcomes. Diagnostics never gate export.
See BASELINE_RECIPE.md for the original-paper correspondence and adaptations.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .io import read, write, sha, digest, development_path

RECIPE_ID = 'fcrl_style_temporal_v1'
TEMPERATURE = .07
SOURCE_STEPS = 10000
PAIRS_PER_BATCH = 48
METRIC_NAMES = ('loss', 'uniform_ce_margin', 'positive_negative_gap',
                'mean_normalized_std', 'effective_rank')


@dataclass
class HistoryBatch:
    """Only the two tensors consumed by the fixed contrastive objective."""
    history_images: torch.Tensor
    history_actions: torch.Tensor

    def validate(self, history_length):
        size = len(self.history_images)
        if history_length != 96 or size < 4 or size % 2:
            raise ValueError('native 96-frame paired history batch required')
        for value, shape in ((self.history_images, (size, 96, 2, 128, 128)),
                             (self.history_actions, (size, 95, 2))):
            if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('history batch shape, dtype, or finite values differ')
        if torch.any(self.history_images[:, 0, 1] != 0):
            raise ValueError('history must not expose a hidden predecessor')

    def to(self, device):
        return HistoryBatch(self.history_images.to(device), self.history_actions.to(device))


def make_history_batch(schedule, plan, batch_index, bank_root):
    """Native pairs and float32 inputs, without unused future tensors/copies.

    visible() retains native asset hashes, uint8 shape, action and time checks.
    The original window arithmetic and donor-then-recipient order are unchanged.
    Constructed tensors are checked once by objective(), before any optimization.
    """
    from native_training import visible
    schedule.validate(plan)
    size = schedule.batch_pairs
    if type(batch_index) is not int or not 0 <= batch_index < len(plan['pairs']) // size:
        raise ValueError('pair batch index')
    if plan['configuration']['name'] != 'Both':
        raise ValueError('contrastive training uses the unchanged Both pair schedule')
    selected = plan['pairs'][batch_index * size:(batch_index + 1) * size]
    history = np.empty((2 * size, 96, 2, 128, 128), np.float32)
    past = np.empty((2 * size, 95, 2), np.float32)
    assets = {}
    raw_read = 0
    for branch_index, branch in enumerate(('donor', 'recipient')):
        for pair_index, pair in enumerate(selected):
            row = schedule.rows[pair[branch + '_episode']]
            if row['split'] != 'train':
                raise PermissionError('nontrain pretraining row')
            asset = row['assets']['128']
            images, actions = visible(str(bank_root), asset['path'], asset['sha256'])
            start = pair[branch + '_start']
            anchor = start + 95
            if anchor + 16 >= len(images):
                raise ValueError('missing planned support; no replacement')
            index = branch_index * size + pair_index
            x = images[start:anchor + 1].astype(np.float32) / 255
            history[index, :, 0] = x
            history[index, 0, 1] = 0
            history[index, 1:, 1] = x[1:] - x[:-1]
            past[index] = actions[start:anchor].astype(np.float32)
            assets[row['episode_key']] = asset['sha256']
            raw_read += len(images)
    batch = HistoryBatch(torch.from_numpy(history), torch.from_numpy(past))
    receipt = dict(plan_sha256=plan['plan_sha256'], batch_index=batch_index,
        input_assets=assets, pair_sha256=[p['pair_sha256'] for p in selected], windows=2 * size,
        raw_image_frames_read=raw_read, presented_history_frames=2 * size * 96,
        observed_action_intervals=2 * size * 95, target_frames=0, target_horizons=[],
        direction='donor_to_recipient', resolution=128, physical_labels_read=False,
        target_statistics=None, test_read=False, future_targets_consumed=0,
        data_preparation='history_only_native_pairs_v2')
    return batch, receipt


def symmetric_infonce(projected, raw, temperature=TEMPERATURE):
    if len(projected) % 2 or len(projected) < 4:
        raise ValueError('paired batch with at least two systems required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    # Disable autocast explicitly: casting operands alone does not protect matmul.
    with torch.autocast(device_type=projected.device.type, enabled=False):
        z = F.normalize(projected.float(), dim=-1)
        p = F.normalize(raw.float(), dim=-1)
        n = len(z) // 2
        cos = z[:n] @ z[n:].T
        logits = cos / temperature
        labels = torch.arange(n, device=z.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * .5
        with torch.no_grad():
            diagonal = torch.eye(n, device=z.device, dtype=torch.bool)
            eig = torch.linalg.svdvals(p - p.mean(0)).square()
            total = eig.sum()
            prob = eig / total.clamp_min(1e-20)
            rank = torch.exp(-(prob * prob.clamp_min(1e-20).log()).sum()) if total > 0 else total
            metrics = dict(loss=float(loss), uniform_ce=math.log(n),
                uniform_ce_margin=math.log(n) - float(loss),
                positive_negative_gap=float(cos.diag().mean() - cos[~diagonal].mean()),
                mean_normalized_std=float(p.std(0, unbiased=False).mean()),
                effective_rank=float(rank), similarity_dtype=str(logits.dtype))
    return loss, metrics


def new_source(seed, device='cpu'):
    """Keep the native visual/history encoder; train only the contrastive objective."""
    from native128_model import Native128JEPA

    class RelationSource(Native128JEPA):
        def __init__(self):
            super().__init__('Bx', projection_seed=seed)
            self.transient = None
            self.predictor = None
            # Supplement A.1: nonlinear critic, one hidden layer with BatchNorm.
            self.projector = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128),
                                           nn.ReLU(), nn.Linear(128, 64))
            self.temperature = TEMPERATURE

        def persistent_code(self, history, actions):
            return self.persistent(history, actions)

        def codes(self, history_h, history_actions):
            persistent = self.persistent_code(history_h, history_actions)
            transient = torch.zeros_like(persistent)
            return transient, persistent, torch.cat((transient, persistent), -1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RelationSource()
    return model.to(device=device, dtype=torch.float32)


def objective(model, batch, keys):
    batch.validate(96)
    half = len(batch.history_images) // 2
    if len(keys) != half or len(set(keys)) != half:
        raise ValueError('same-system false negatives forbidden')
    history = model.observation(batch.history_images, train_history=model.training)
    code = model.persistent_code(history, batch.history_actions)
    with torch.autocast(device_type=code.device.type, enabled=False):
        projected = model.projector(code.float())
        return symmetric_infonce(projected, code, model.temperature)


def summarize_training(events, steps):
    """Describe optimization without selecting or rejecting scientific outcomes."""
    if not events:
        raise ValueError('no optimization records')
    tail = list(events)[-100:]
    values = {k: float(np.mean([e[k] for e in tail])) for k in METRIC_NAMES}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('nonfinite optimization records are a numerical failure')
    return dict(status='COMPLETE' if steps == SOURCE_STEPS else 'SMOKE_ONLY',
        tail_mean=values, tail_updates=len(tail), updates=steps, recipe_id=RECIPE_ID,
        outcome_threshold_applied=False, downstream_result_used=False,
        reporting_policy='Report every completed fixed-recipe source, including weak or collapsed representations.',
        test_read=False)


def setup(native_root, bank):
    from .protocol import activate_native
    root = development_path(native_root).resolve()
    bank = development_path(bank).resolve()
    paths = activate_native(root)
    files = {str(p.resolve()): sha(p) for p in sorted(root.rglob('*.py'))}
    if not (root / 'native128_model.py').exists() or not (root / 'native_training.py').exists():
        raise FileNotFoundError('native-root must contain native128_model.py and native_training.py')
    manifest = read(bank / 'MANIFEST.private.json')
    if any(r['split'] not in ('train', 'validation') for r in manifest['episodes']):
        raise PermissionError('development bank required')
    return root, bank, paths, files, manifest


def implementation_hashes():
    return {'contrastive.py': sha(__file__)}


def train(native_root, bank, output, seed, device='cuda:0', smoke_steps=None, resume=False):
    if seed not in (0, 1, 2):
        raise ValueError('fixed source seeds are 0, 1, 2; there is no selection seed')
    if smoke_steps is not None and not 1 <= smoke_steps < SOURCE_STEPS:
        raise ValueError('smoke must be shorter than the formal source budget')
    root, bank, paths, source_files, manifest = setup(native_root, bank)
    from native_training import NativePairSchedule
    from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec, pretraining_learning_rate, _step_rng
    from persistbench.envs.visual_elastic_coupling.a_head_features import model_state_sha256
    from .engine import math_profile
    device = torch.device(device)
    math_profile(device)
    torch.set_num_threads(1)
    steps = SOURCE_STEPS if smoke_steps is None else smoke_steps
    spec = PretrainingSpec(steps=SOURCE_STEPS, pairs_per_batch=PAIRS_PER_BATCH,
        history_frames=96, model_seed=seed, sampling_seed=seed, stochastic_seed=seed,
        save_every=2500, log_every=100, milestone_steps=(1000, 2500, 5000, 10000))
    schedule = NativePairSchedule(manifest, seed=seed, pairs_per_batch=PAIRS_PER_BATCH)
    count = len({r['system_key'] for r in manifest['episodes'] if r['split'] == 'train'})
    if count != 144:
        raise ValueError('the matched source bank requires 144 physical training systems')
    batches_per_sweep = math.ceil(count / PAIRS_PER_BATCH)
    model = new_source(seed, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay, foreach=False, fused=False)
    out = Path(output)
    run = dict(recipe_id=RECIPE_ID, method='RelInfoNCE', source_seed=seed,
        temperature=TEMPERATURE, temperature_selection='fixed before outcomes; no sweep',
        history_frames=96, history_encoder='unchanged native temporal HistoryEncoder P64',
        query_encoder='same observation encoder as donor history',
        data_preparation='history_only_native_pairs_v2',
        projector='Linear64x128-BatchNorm128-ReLU-Linear128x64; discarded downstream',
        source_updates=steps, source_files=source_files, implementation_sha256=implementation_hashes(),
        spec=spec.record(), parameters=sum(p.numel() for p in model.parameters()),
        bank_snapshot_sha256=sha(bank / 'BANK_SNAPSHOT.json'),
        manifest_sha256=sha(bank / 'MANIFEST.private.json'),
        native_root=str(root), bank=str(bank), selection_rule='final_step_only',
        outcome_threshold_applied=False, test_read=False, smoke=smoke_steps is not None)
    events = deque(maxlen=100)
    sequence = []
    start = 0
    if resume:
        previous = read(out / 'RUN.json')
        if previous != run:
            raise ValueError('resume recipe, code, source files, or bank changed')
        if (out / 'COMPLETE.json').exists():
            return read(out / 'COMPLETE.json')
        checkpoint = torch.load(out / 'resume.pt', map_location=device, weights_only=True)
        if checkpoint['run_sha256'] != sha(out / 'RUN.json'):
            raise ValueError('resume checkpoint belongs to another run')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        events.extend(checkpoint['events'])
        sequence = checkpoint['sequence']
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'RUN.json', run)
    plan = None
    with (out / 'training.jsonl').open('a' if resume else 'x') as log:
        for step in range(start, steps):
            sweep, bi = divmod(step, batches_per_sweep)
            if plan is None or bi == 0:
                plan = schedule.sweep(sweep, 'Both')
            batch, receipt = make_history_batch(schedule, plan, bi, bank)
            if receipt['physical_labels_read'] or receipt['test_read']:
                raise PermissionError('contrastive source saw private evidence')
            keys = [x['recipient_system'] for x in plan['pairs'][bi * PAIRS_PER_BATCH:(bi + 1) * PAIRS_PER_BATCH]]
            batch = batch.to(device)
            sequence.append(digest(receipt))
            for group in optimizer.param_groups:
                group['lr'] = pretraining_learning_rate(spec, step)
            optimizer.zero_grad(set_to_none=True)
            model.begin_train_step()
            with _step_rng(seed, step, device):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss, metrics = objective(model, batch, keys)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite contrastive objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                model.finish_train_step()
            event = dict(step=step + 1, **metrics)
            events.append(event)
            if (step + 1) % spec.log_every == 0 or step + 1 == steps:
                log.write(json.dumps(event) + '\n')
                log.flush()
                print(json.dumps(event), flush=True)
            if (step + 1) % spec.save_every == 0 or step + 1 in spec.milestone_steps or step + 1 == steps:
                checkpoint = dict(step=step + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    events=list(events), sequence=sequence, run_sha256=sha(out / 'RUN.json'))
                torch.save(checkpoint, out / 'resume.pending.pt')
                (out / 'resume.pending.pt').replace(out / 'resume.pt')
    diagnostics = summarize_training(events, steps)
    write(out / 'DIAGNOSTICS.json', diagnostics)
    if implementation_hashes() != run['implementation_sha256'] or any(sha(p) != h for p, h in source_files.items()):
        raise ValueError('baseline implementation changed during training')
    if sha(bank / 'BANK_SNAPSHOT.json') != run['bank_snapshot_sha256'] or sha(bank / 'MANIFEST.private.json') != run['manifest_sha256']:
        raise ValueError('bank identity changed during training')
    checkpoint = dict(recipe_id=RECIPE_ID, source_seed=seed, step=steps,
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        model_sha256=model_state_sha256(model))
    torch.save(checkpoint, out / 'source.pt')
    complete = dict(status='COMPLETE', recipe_id=RECIPE_ID, method='RelInfoNCE', source_seed=seed,
        selected_step=steps, optimizer_updates=steps, selection_rule='final_step_only',
        checkpoint=str((out / 'source.pt').resolve()), checkpoint_sha256=sha(out / 'source.pt'),
        model_state_sha256=checkpoint['model_sha256'], run_sha256=sha(out / 'RUN.json'),
        diagnostics_sha256=sha(out / 'DIAGNOSTICS.json'), outcome_threshold_applied=False,
        temperature=TEMPERATURE, training_sequence_sha256=digest(sequence),
        bank_snapshot_sha256=run['bank_snapshot_sha256'], test_read=False, smoke=smoke_steps is not None)
    write(out / 'COMPLETE.json', complete)
    return complete


def export(native_root, bank, completion, output, device='cuda:0'):
    root, bank, paths, source_files, manifest = setup(native_root, bank)
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan, AHeadDataAccess
    from persistbench.envs.visual_elastic_coupling.a_head_features import extract_features, AHeadFeatureCache, model_state_sha256
    from persistbench.envs.visual_elastic_coupling.a_head_targets import extract_targets
    cp = development_path(completion).resolve()
    c = read(cp)
    r = read(cp.parent / 'RUN.json')
    if c['status'] != 'COMPLETE' or c['source_seed'] not in (0, 1, 2) or c['smoke'] or c['selected_step'] != SOURCE_STEPS:
        raise PermissionError('complete fixed-budget source required; smoke is not a result')
    if c['recipe_id'] != RECIPE_ID or c['outcome_threshold_applied'] is not False:
        raise ValueError('baseline recipe changed')
    if sha(cp.parent / 'RUN.json') != c['run_sha256'] or sha(cp.parent / 'DIAGNOSTICS.json') != c['diagnostics_sha256']:
        raise ValueError('source receipts changed')
    if r['implementation_sha256'] != implementation_hashes() or r['source_files'] != source_files:
        raise ValueError('source implementation changed')
    if sha(bank / 'BANK_SNAPSHOT.json') != r['bank_snapshot_sha256'] or sha(bank / 'MANIFEST.private.json') != r['manifest_sha256']:
        raise ValueError('source and export bank differ')
    if sha(c['checkpoint']) != c['checkpoint_sha256']:
        raise ValueError('source checkpoint changed')
    checkpoint = torch.load(c['checkpoint'], map_location='cpu', weights_only=True)
    model = new_source(c['source_seed'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    if checkpoint['step'] != SOURCE_STEPS or model_state_sha256(model) != c['model_state_sha256']:
        raise ValueError('source content differs')
    plan = AHeadCasePlan(manifest, seed=0, history_frames=96)
    access = AHeadDataAccess(bank, plan, snapshot_sha256=r['bank_snapshot_sha256'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    extract_features(access, model, out / 'features', expected_model_state_sha256=c['model_state_sha256'], workers=8,
        progress=lambda event: print(json.dumps(event), flush=True))
    fr = out / 'features/FEATURES.json'
    features = AHeadFeatureCache(fr.parent, plan, receipt_sha256=sha(fr),
        model_state_sha256=c['model_state_sha256'], bank_snapshot_sha256=r['bank_snapshot_sha256'])
    extract_targets(access, features, out / 'targets', workers=8)
    descriptor = dict(environment='springworld', method='RelInfoNCE', source_seed=c['source_seed'],
        native_files=source_files, native_paths=paths, model_state_sha256=c['model_state_sha256'],
        source_completion=str(cp), source_completion_sha256=sha(cp))
    for key, path in dict(manifest=bank / 'MANIFEST.private.json', features_receipt=fr,
            targets_receipt=out / 'targets/SUPERVISION.json', checkpoint=Path(c['checkpoint'])).items():
        descriptor[key] = str(path.resolve())
        descriptor[key + '_sha256'] = sha(path)
    from .providers import SpringCache
    SpringCache(descriptor)
    write(out / 'SOURCE.json', descriptor)
    return descriptor


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('train', 'export'):
        c = sub.add_parser(command)
        c.add_argument('--native-root', required=True)
        c.add_argument('--bank', required=True)
        c.add_argument('--output', required=True)
        c.add_argument('--device', default='cuda:0')
        if command == 'train':
            c.add_argument('--seed', type=int, required=True)
            c.add_argument('--smoke-steps', type=int)
            c.add_argument('--resume', action='store_true')
        else:
            c.add_argument('--completion', required=True)
    args = vars(p.parse_args())
    command = args.pop('command')
    print(json.dumps(dict(train=train, export=export)[command](**args), indent=2))


if __name__ == '__main__':
    main()
