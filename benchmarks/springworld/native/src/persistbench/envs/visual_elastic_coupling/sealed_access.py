"""Fail-closed frozen-protocol and selected-artifact authorization.

This is an accidental-access/provenance boundary for trusted reference runs,
not an operating-system sandbox for arbitrary submitted Python. Only evaluator
code receives this object; model inputs remain the public observation codec.
No constructor in this module generates or reads test trajectories.
"""
import json
from dataclasses import asdict
from pathlib import Path
from .dataset_snapshot import checked_asset,stable_digest
from .training_protocol import source_fingerprint
from .evaluation_blocks import digest,validate_plan
from .schema import Config

PROTOCOL_SCHEMA='vec.sealed-evaluation-protocol.v1'
SELECTION_SCHEMA='vec.frozen-method-selection.v1'
CAPABILITIES=('recoverability','formation','composition','delayed_value','realized_utility','specificity','transfer','control_competency')
ASSAYS=('raw_recoverability','formation','conditional_prediction','history_composition','delayed_prediction','factor_specificity','predictive_transfer','closed_loop_control')


def read_bound(root,descriptor):
    if set(descriptor)!={'path','sha256'}:raise ValueError('exact relative path and digest binding required')
    path=checked_asset(root,descriptor['path'])
    if stable_digest(path)['sha256']!=descriptor['sha256']:raise ValueError('bound artifact content changed: '+descriptor['path'])
    return path


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_protocol(protocol):
    if protocol.get('schema')!=PROTOCOL_SCHEMA or protocol.get('status')!='FROZEN':raise PermissionError('sealed protocol is not frozen')
    if protocol.get('allow_test_generation') is not True or protocol.get('allow_test_evaluation') is not True:
        raise PermissionError('sealed generation/evaluation not authorized in frozen protocol')
    if protocol.get('source_fingerprint')!=source_fingerprint():raise ValueError('frozen evaluator/generator source differs')
    if protocol.get('physics_config')!=asdict(Config()):raise ValueError('registered physical/camera base configuration differs from generator')
    runtime=protocol.get('render_runtime',{})
    if not runtime.get('mujoco_version') or runtime.get('backend')!='osmesa':raise ValueError('explicit renderer runtime binding required')
    training=protocol.get('training_bank',{})
    required=('manifest_sha256','target_statistics_sha256','snapshot_sha256','content_sha256')
    if set(training)!=set(required) or not all(_sha(training[k]) for k in required):raise ValueError('incomplete frozen training-bank binding')
    if set(protocol.get('capability_receipts',{}))!=set(CAPABILITIES):raise ValueError('comprehensive capability admission receipts are missing')
    assays=protocol.get('assays',{})
    if not set(ASSAYS).issubset(assays):raise ValueError('comprehensive sealed assay registration incomplete')
    for name,spec in assays.items():
        if not spec.get('profiles') or not spec.get('metrics') or not spec.get('failure_policy') or not spec.get('budget') or not _sha(spec.get('configuration_sha256')):
            raise ValueError('incomplete frozen assay: '+name)
    stats=protocol.get('statistics',{})
    if stats.get('independent_unit')!='registered_plan_blocks' or stats.get('seed_policy')!='fixed_artifacts_within_blocks':raise ValueError('unregistered statistical units')
    if not 0<float(stats.get('alpha',0))<1 or not stats.get('comparison_families'):raise ValueError('missing alpha/comparison families')
    for name,size in stats['comparison_families'].items():
        if not isinstance(size,int) or size<1:raise ValueError('invalid frozen comparison family size')
    slots=protocol.get('method_slots',[])
    if not slots or len({s['slot_id'] for s in slots})!=len(slots):raise ValueError('empty or duplicate method slots')
    for slot in slots:
        if slot.get('kind') not in ('learned','explicit') or not slot.get('required_roles'):raise ValueError('invalid method slot')
        if slot['kind']=='learned':
            mode=slot.get('admission_mode','prospective')
            if mode not in ('prospective','pretest_existing'):raise ValueError('unknown learned artifact admission mode')
            if mode=='pretest_existing':
                from .pretest_artifact import REQUIRED_ROLES
                required=set(REQUIRED_ROLES)
            else:required={'predictor','training_config','training_report','formal_completion'}
            if not required.issubset(slot['required_roles']):raise ValueError('learned slot lacks registered training evidence')
            if not isinstance(slot.get('seed'),int) or not slot.get('family') or not _sha(slot.get('training_protocol_sha256')):raise ValueError('learned training registration incomplete')
            name=slot.get('checkpoint_name')
            if not isinstance(name,str) or not name or Path(name).name!=name:raise ValueError('exact selected checkpoint name must be frozen')
    if not protocol.get('populations'):raise ValueError('no frozen evaluation populations')
    for name,spec in protocol['populations'].items():
        if not _sha(spec.get('plan_sha256')) or not _sha(spec.get('population_sha256')) or not spec.get('episode_namespace'):raise ValueError('incomplete population/episode commitment')
    return protocol


class SealedAuthorization:
    """Explicit hash commitments are required; paths alone never authorize."""
    def __init__(self,protocol_path,selection_path,*,protocol_sha256,selection_sha256):
        self.protocol_path=Path(protocol_path);self.selection_path=Path(selection_path)
        self.protocol_sha256=protocol_sha256;self.selection_sha256=selection_sha256
        if not _sha(protocol_sha256) or not _sha(selection_sha256):raise PermissionError('explicit frozen content commitments required')
        self.revalidate()

    def revalidate(self):
        if stable_digest(self.protocol_path)['sha256']!=self.protocol_sha256:raise ValueError('protocol commitment changed')
        if stable_digest(self.selection_path)['sha256']!=self.selection_sha256:raise ValueError('selection commitment changed')
        protocol=validate_protocol(json.loads(self.protocol_path.read_text()));selection=json.loads(self.selection_path.read_text())
        if selection.get('schema')!=SELECTION_SCHEMA or selection.get('status')!='FROZEN' or selection.get('test_read') is not False:
            raise PermissionError('method selection is not frozen before test access')
        if selection.get('protocol_sha256')!=self.protocol_sha256:raise ValueError('selection refers to another evaluation protocol')
        for capability,descriptor in protocol['capability_receipts'].items():
            receipt=json.loads(read_bound(self.protocol_path.parent,descriptor).read_text())
            if receipt.get('status') not in ('PASS','EXECUTED') or receipt.get('test_read') is not False:raise ValueError('unqualified capability admission: '+capability)
        entries=selection.get('methods',[]);by_slot={r['slot_id']:r for r in entries}
        if len(by_slot)!=len(entries) or set(by_slot)!={s['slot_id'] for s in protocol['method_slots']}:raise ValueError('selected method matrix differs from registration')
        resolved={}
        for slot in protocol['method_slots']:
            files=by_slot[slot['slot_id']]['files']
            if set(files)!=set(slot['required_roles']):raise ValueError('missing or extra selected model roles')
            paths={role:read_bound(self.selection_path.parent,descriptor) for role,descriptor in files.items()}
            if slot['kind']=='learned':
                if slot.get('admission_mode','prospective')=='pretest_existing':
                    from .pretest_artifact import validate_existing
                    validate_existing(paths,slot=slot,training_bank=protocol['training_bank'])
                    resolved[slot['slot_id']]=paths
                    continue
                completion=json.loads(paths['formal_completion'].read_text());config=json.loads(paths['training_config'].read_text());report=json.loads(paths['training_report'].read_text())
                if completion.get('schema')!='vec.formal-training-completion.v1.1' or completion.get('status')!='PASS' or completion.get('test_read') is not False:
                    raise PermissionError('development checkpoint cannot be admitted as formal training')
                if completion.get('protocol_sha256')!=slot['training_protocol_sha256'] or completion.get('train_report_sha256')!=files['training_report']['sha256']:
                    raise ValueError('formal training receipt binding differs')
                if config.get('family')!=slot['family'] or config.get('seed')!=slot['seed'] or config.get('stage')!='formal' or report.get('stage')!='formal' or report.get('status')!='EXECUTED':
                    raise ValueError('selected family/seed/training stage differs')
                if config.get('bank_manifest_sha256')!=protocol['training_bank']['manifest_sha256']:raise ValueError('selected training bank differs')
                if report.get('checkpoints')!=completion.get('checkpoints') or files['predictor']['sha256']!=completion.get('checkpoints',{}).get(slot['checkpoint_name']):
                    raise ValueError('predictor is not the preregistered completed checkpoint')
                content_check=completion.get('after_run_content_check',{})
                if content_check.get('status')!='PASS' or content_check.get('content_sha256')!=protocol['training_bank']['content_sha256']:
                    raise ValueError('formal after-training content verification absent')
                bindings=completion.get('input_bindings',{})
                for short,full in (('manifest_sha256','bank_manifest_sha256'),('target_statistics_sha256','target_statistics_sha256'),('snapshot_sha256','bank_snapshot_sha256'),('content_sha256','bank_content_sha256')):
                    if bindings.get(full)!=protocol['training_bank'][short]:raise ValueError('formal data-content lineage differs')
            resolved[slot['slot_id']]=paths
        self.protocol=protocol;self.selection=selection;self.selected_paths=resolved
        return self

    def validate_population(self,name,plan,systems):
        validate_plan(plan)
        if name not in self.protocol['populations']:raise PermissionError('population not frozen')
        spec=self.protocol['populations'][name]
        if spec['plan_sha256']!=plan['plan_sha256'] or spec['population_sha256']!=digest(systems):raise ValueError('population or sampling plan differs')
        if plan['stratum']!=name or [s['system_key'] for s in systems]!=plan['system_generation_order']:raise ValueError('population order or stratum differs')
        if any(s['split']!='test' or s['stratum']!=name for s in systems):raise ValueError('sealed population split differs')
        return spec['episode_namespace']
