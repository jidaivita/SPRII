"""Extract the unchanged frozen geometry inputs without downstream query/target caches.

Uses the native per-donor inference path and the frozen source_geometry function.
Only manifest-selected train/validation donor pixels are read; their native
reader verifies each asset against BANK_SNAPSHOT before and after loading it.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sprii_next.io import development_path, read, write, npz, sha
from sprii_next.protocol import activate_native, verify_spring_source
from sprii_next.geometry import source_geometry


def extract_donors(access, model, expected_model_sha256, progress=None):
    """Exactly the donor loop of native extract_features, with no query pass."""
    from persistbench.envs.visual_elastic_coupling.a_head_features import _donors, model_state_sha256
    from persistbench.envs.visual_elastic_coupling.a_fresh_head import AFreshFeatures
    from persistbench.envs.visual_elastic_coupling.adapters import z_experience
    from persistbench.envs.visual_elastic_coupling.schema import history_payload
    plan = access.plan
    plan._guard()
    if model_state_sha256(model) != expected_model_sha256:
        raise ValueError('source model differs from completion')
    if model.cfg.history_length != plan.frames or plan.frames != 96:
        raise ValueError('the registered 96-frame donor plan is required')
    if getattr(model, 'normalization_profile', None) != 'train_history_running_statistics_pre_step_v1':
        raise ValueError('source normalization profile differs')
    features = AFreshFeatures(model)
    features.initialize(None)
    if features.representation_dim != 64:
        raise ValueError('geometry requires the native frozen P64')
    donors = _donors(plan)
    values = np.zeros((len(donors), 64), np.float32)
    audit_start = len(access.audit)
    for index, key in enumerate(donors):
        if plan.rows[key]['raw_frames'] < plan.frames:
            raise ValueError('planned donor has insufficient support: ' + key)
        features._guard()
        features.initialize(None)
        episode = access.public_episode(key, purpose='donor')
        features.ingest(z_experience(history_payload(episode, 0, plan.frames - 1)))
        values[index] = features.history_code().cpu().numpy()[0]
        del episode
        if progress is not None and ((index + 1) % 128 == 0 or index + 1 == len(donors)):
            progress(dict(phase='donor', completed=index + 1, total=len(donors)))
    features._guard()
    plan._guard()
    if model_state_sha256(model) != expected_model_sha256 or not np.isfinite(values).all():
        raise ValueError('source changed or returned nonfinite donor codes')
    events = access.audit[audit_start:]
    if len(events) != len(donors) or any(e['kind'] != 'public_observation' or e['purpose'] != 'donor' for e in events):
        raise ValueError('geometry extraction must read exactly the declared public donors')
    return donors, values, events


class DonorProvider:
    environment = 'springworld'

    def __init__(self, descriptor, plan, donor_ids, values):
        self.descriptor = descriptor
        self.data = {}
        for split in ('train', 'validation'):
            indices = [i for i, key in enumerate(donor_ids) if plan.rows[key]['split'] == split]
            keys = [donor_ids[i] for i in indices]
            systems = np.asarray([plan.rows[key]['system_key'] for key in keys])
            theta = np.asarray([plan.systems[sid]['theta'] for sid in systems])
            self.data[split] = (values[indices].copy(), theta, systems, np.asarray(keys))

    def donors(self, split):
        if split not in self.data:
            raise PermissionError('development donors only')
        return self.data[split]


def export(native_root, bank, completion, method, seed, output, device):
    paths = activate_native(native_root)
    model, source, checkpoint = verify_spring_source(completion, method, seed)
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan, AHeadDataAccess
    from persistbench.envs.visual_elastic_coupling.training_protocol import source_fingerprint
    bank = development_path(bank).resolve()
    manifest = bank / 'MANIFEST.private.json'
    snapshot = bank / 'BANK_SNAPSHOT.json'
    expected = source['binding']['bank_snapshot_sha256']
    if sha(snapshot) != expected:
        raise ValueError('geometry bank differs from the trained source')
    plan = AHeadCasePlan(read(manifest), seed=0, history_frames=96)
    access = AHeadDataAccess(bank, plan, snapshot_sha256=expected)
    out = development_path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    descriptor = dict(environment='springworld', method=method, source_seed=seed,
        extraction='geometry_only_native_donors_v1', native_paths=paths,
        native_files={str(p.resolve()): sha(p) for p in sorted(Path(native_root).rglob('*.py'))},
        model_state_sha256=source['model_state_sha256'], bank_snapshot_sha256=expected,
        plan_sha256=plan.plan_sha256, manifest_semantic_sha256=plan.manifest_semantic_sha256)
    for key, path in dict(source_completion=completion, checkpoint=checkpoint, manifest=manifest).items():
        descriptor[key] = str(Path(path).resolve())
        descriptor[key + '_sha256'] = sha(path)
    fingerprint = source_fingerprint()
    model.to(device)
    started = time.monotonic()
    donor_ids, values, events = extract_donors(access, model, source['model_state_sha256'],
        progress=lambda event: print(json.dumps(event), flush=True))
    if sha(snapshot) != expected or source_fingerprint() != fingerprint:
        raise ValueError('geometry input commitment or extractor changed')
    provider = DonorProvider(descriptor, plan, donor_ids, values)
    result = source_geometry(provider)
    for split, filename in (('train', 'training_donor_vectors.npz'), ('validation', 'source_vectors.npz')):
        p, theta, systems, keys = provider.donors(split)
        npz(out / filename, persistent_code=p, physical_parameters=theta, system_id=systems, donor_id=keys)
    used_paths = sorted({event['path'] for event in events})
    receipt = dict(schema='sprii-next.geometry-donors.v1', status='COMPLETE', test_read=False,
        source=descriptor, extractor_script_sha256=sha(__file__), native_source_fingerprint=fingerprint,
        history_frames=96, inference='native AFreshFeatures per-donor reset, float32, model.eval; unchanged',
        donors=len(donor_ids), public_donor_reads=len(events), query_reads=0, target_reads=0,
        checked_assets=[access.files[path] for path in used_paths],
        asset_check='native AHeadDataAccess validates each used donor asset before and after loading',
        training_vectors_sha256=sha(out / 'training_donor_vectors.npz'),
        validation_vectors_sha256=sha(out / 'source_vectors.npz'), elapsed_seconds=time.monotonic() - started)
    write(out / 'DONOR_EXTRACTION.json', receipt)
    result['source_vectors_sha256'] = receipt['validation_vectors_sha256']
    result['donor_extraction_receipt_sha256'] = sha(out / 'DONOR_EXTRACTION.json')
    write(out / 'RESULT.json', result)
    return dict(output=str(out), method=method, source_seed=seed, donors=len(donor_ids),
                systems=result['systems'], test_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-root', 'bank', 'completion', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--method', required=True, choices=('Structure', 'Align', 'Cross', 'Both'))
    parser.add_argument('--seed', required=True, type=int, choices=(0, 1, 2))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    result = export(args.native_root, args.bank, args.completion, args.method, args.seed, args.output, args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
