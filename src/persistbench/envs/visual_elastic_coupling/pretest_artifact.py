"""Admit an existing development model before an independent frozen test.

Original training files stay unchanged. This audits a finished artifact against
a newly frozen evaluation budget; it does not claim training was preregistered.
It never opens test data or grants permission by itself.
"""
import ast
import hashlib
import json
import math
import tarfile
from dataclasses import asdict
from pathlib import Path

SCHEMA = 'vec.pretest-existing-model-admission.v1'
MODE = 'pretest_existing'
EVIDENCE_ROLES = ('predictor', 'training_config', 'training_report', 'training_log',
                  'training_source_archive', 'training_launch_budget', 'training_budget',
                  'training_data_admission', 'training_content_admission', 'training_target_statistics')
REQUIRED_ROLES = (*EVIDENCE_ROLES, 'artifact_admission')
PACKAGE = 'src/persistbench/envs/visual_elastic_coupling/'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_compatibility(archive_path):
    """Compare the original model/sampler and reachable development helpers."""
    with tarfile.open(archive_path) as archive:
        members = {}
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile() or member.name in members:
                raise ValueError('invalid or duplicate original source member')
            members[member.name] = archive.extractfile(member).read()
    sources = {name[len(PACKAGE):]: value for name, value in members.items()
               if name.startswith(PACKAGE) and '/' not in name[len(PACKAGE):] and name.endswith('.py')}
    fingerprint = hashlib.sha256()
    for name, value in sorted(sources.items()):
        fingerprint.update(name.encode() + b'\0' + value)
    current = Path(__file__).parent
    direct = ('pixel_models.py', 'pixel_training.py', 'target_scaling.py')
    for name in direct:
        if sources.get(name) != (current / name).read_bytes():
            raise ValueError('original model or sampling implementation differs: ' + name)
    name = 'training_protocol.py'
    old = ast.parse(sources[name]); new = ast.parse((current / name).read_text())
    def functions(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old_functions, new_functions = functions(old), functions(new)
    helpers = ('file_digest', 'source_fingerprint', 'verify_admission', 'validation_case')
    for helper in helpers:
        if old_functions.get(helper) != new_functions.get(helper):
            raise ValueError('original development training helper differs: ' + helper)
    # The formal-only helper may have evolved. Imports used by the unchanged
    # development functions must still have identical definitions.
    imports = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body
                            if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports(old) != imports(new):
        raise ValueError('training helper imports differ')
    return dict(original_source_fingerprint=fingerprint.hexdigest(),
                unchanged_files={name: hashlib.sha256(sources[name]).hexdigest() for name in direct},
                unchanged_development_helpers=list(helpers))


def audit_existing(paths, *, slot, training_bank):
    """Reconstruct eligibility from bound original files, not an asserted PASS."""
    paths = {key: Path(value) for key, value in paths.items()}
    if not set(EVIDENCE_ROLES) <= set(paths):
        raise ValueError('existing artifact evidence is incomplete')
    if slot.get('admission_mode') != MODE:
        raise PermissionError('existing-artifact admission must be explicitly frozen')
    before = {key: _sha(paths[key]) for key in EVIDENCE_ROLES}
    read = lambda key: json.loads(paths[key].read_text())
    config, report = read('training_config'), read('training_report')
    launch, budget = read('training_launch_budget'), read('training_budget')
    admission, content = read('training_data_admission'), read('training_content_admission')
    statistics = read('training_target_statistics')
    if budget.get('status') != 'FROZEN' or before['training_budget'] != slot.get('training_protocol_sha256'):
        raise ValueError('evaluation training budget is not the frozen commitment')
    if config.get('stage') != 'development_training' or report.get('stage') != config['stage']:
        raise ValueError('preserve the original development training identity')
    if any(value.get('test_read') is not False for value in (config, report, launch, admission, content)):
        raise PermissionError('existing artifact evidence must precede test access')
    if report.get('status') != 'EXECUTED' or config.get('family') != slot.get('family') or config.get('seed') != slot.get('seed'):
        raise ValueError('existing model did not finish the registered family/seed')
    opt = budget.get('optimization', {})
    for key in ('updates', 'batch_size', 'accumulation', 'learning_rate', 'weight_decay', 'gradient_clip'):
        if config.get(key) != opt.get(key) or key not in opt:
            raise ValueError('existing training budget differs: ' + key)
    for key in ('updates', 'batch_size', 'accumulation', 'learning_rate', 'validate_every', 'validation_batches'):
        if launch.get(key) != opt.get(key) or key not in opt:
            raise ValueError('original launch budget differs: ' + key)
    if opt.get('warmup_updates') != 100 or opt.get('schedule') != 'cosine_floor_0.1':
        raise ValueError('registered learning-rate schedule differs')
    if launch.get('seed') != slot['seed'] or slot['seed'] not in opt.get('seeds', []) or slot['family'] not in budget.get('families', []):
        raise ValueError('existing family/seed is not registered')
    integer_budgets = ('updates', 'batch_size', 'accumulation', 'validate_every', 'validation_batches')
    if any(type(opt[key]) is not int or opt[key] < 1 for key in integer_budgets) or opt['validation_batches'] % 80:
        raise ValueError('invalid complete training/paired validation budget')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
        value = opt[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError('invalid optimization setting: ' + key)
    if report.get('updates') != opt['updates'] or budget.get('selection') != 'paired_matched_null_all_query_budgets_all_horizons':
        raise ValueError('completed update count or checkpoint selection differs')
    if launch.get('source_sha256') != before['training_source_archive'] or launch.get('admission_sha256') != before['training_data_admission'] or config.get('data_admission_sha256') != before['training_data_admission']:
        raise ValueError('original launch/source/data admission lineage differs')
    checked_source = source_compatibility(paths['training_source_archive'])
    if config.get('source_fingerprint') != checked_source['original_source_fingerprint']:
        raise ValueError('checkpoint training source is not the original archive')
    if admission.get('status') != 'PASS' or admission.get('problems') or content.get('status') != 'PASS' or content.get('original_archive_files_verified', 0) < 1:
        raise ValueError('training data integrity or original content audit did not pass')
    expected = dict(manifest_sha256=config.get('bank_manifest_sha256'),
                    target_statistics_sha256=before['training_target_statistics'],
                    snapshot_sha256=content.get('snapshot_sha256'), content_sha256=content.get('content_sha256'))
    if expected != training_bank or admission.get('manifest_sha256') != expected['manifest_sha256'] or admission.get('target_statistics_sha256') != expected['target_statistics_sha256']:
        raise ValueError('existing training bank or statistics differs')
    if statistics.get('source_split') != 'train' or config.get('target_statistics') != statistics:
        raise ValueError('existing normalization differs from audited train-only statistics')
    for short, full in (('manifest_sha256', 'bank_manifest_sha256'), ('target_statistics_sha256', 'target_statistics_sha256'),
                        ('snapshot_sha256', 'bank_snapshot_sha256'), ('content_sha256', 'bank_content_sha256')):
        if budget.get(full) != expected[short]:
            raise ValueError('frozen training budget data binding differs')
    trace = [json.loads(line) for line in paths['training_log'].read_text().splitlines()]
    if [r.get('update') for r in trace] != list(range(1, opt['updates'] + 1)):
        raise ValueError('training log is incomplete or duplicated')
    for row in trace:
        if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0 for k in ('train_loss', 'gradient_norm')):
            raise ValueError('nonfinite or invalid completed training update')
    validation = [r for r in trace if 'validation_loss' in r]
    steps = sorted(set(range(opt['validate_every'], opt['updates'] + 1, opt['validate_every'])) | {opt['updates']})
    if [r['update'] for r in validation] != steps:
        raise ValueError('validation checkpoints do not match the frozen schedule')
    components = ('cold/matched', 'cold/null', 'moving/matched', 'moving/null')
    for row in validation:
        values = row.get('validation_components', {})
        if set(values) != set(components) or not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('validation does not preserve paired cold/moving conditions')
        if not math.isclose(sum(values.values()) / 4, row['validation_loss'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('validation selection score differs from recorded conditions')
    selected = min(validation, key=lambda row: row['validation_loss'])
    if slot.get('checkpoint_name') != 'best.pt' or report.get('best_validation_loss') != selected['validation_loss'] or report.get('checkpoints', {}).get('best.pt') != before['predictor']:
        raise ValueError('selected checkpoint differs from original validation selection')
    import torch
    from .pixel_models import PixelDynamicsModel, PixelModelConfig
    artifact = torch.load(paths['predictor'], map_location='cpu', weights_only=True)
    if artifact.get('config') != config or artifact.get('update') != selected['update'] or artifact.get('validation_loss') != selected['validation_loss']:
        raise ValueError('checkpoint embedded configuration or selection differs')
    cfg = PixelModelConfig(family=slot['family'], resolution=budget.get('resolution'))
    if config.get('model') != asdict(cfg):
        raise ValueError('existing model architecture differs from the common recipe')
    with torch.random.fork_rng(devices=[]):
        reference = PixelDynamicsModel(cfg)
    reference.set_target_statistics(statistics)
    for name in ('target_mean', 'target_scale', 'target_horizons'):
        if not torch.equal(artifact['model'][name], reference.state_dict()[name]):
            raise ValueError('checkpoint normalization does not match training statistics')
    reference.load_state_dict(artifact['model'], strict=True)
    if config.get('parameters') != sum(p.numel() for p in reference.parameters()) or any(not torch.isfinite(t).all() for t in reference.state_dict().values()):
        raise ValueError('checkpoint contains invalid model tensors')
    if before != {key: _sha(paths[key]) for key in EVIDENCE_ROLES}:
        raise ValueError('existing evidence changed during admission')
    return dict(schema=SCHEMA, status='PASS', admission_mode=MODE, files=before,
                family=slot['family'], seed=slot['seed'], training_budget_sha256=before['training_budget'],
                training_bank=expected, source_compatibility=checked_source,
                original_training_stage=config['stage'], original_training_protocol_sha256=config.get('protocol_sha256'),
                completed_updates=len(trace), selected_update=selected['update'], selected_validation_loss=selected['validation_loss'],
                training_was_preregistered=False, test_read=False,
                scope='Existing development artifact audited before independent frozen test; original files remain unchanged; no claim of prospective training completion.')


def validate_existing(paths, *, slot, training_bank):
    if 'artifact_admission' not in paths:
        raise PermissionError('explicit existing-artifact admission receipt required')
    actual = json.loads(Path(paths['artifact_admission']).read_text())
    expected = audit_existing(paths, slot=slot, training_bank=training_bank)
    if actual != expected:
        raise ValueError('existing-artifact receipt does not match independent evidence audit')
    return expected
