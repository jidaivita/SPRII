"""Controlled fixed-artifact prediction evaluation with post-run admission.

All method/data/profile arguments are read from their frozen commitments.
No CLI switch can turn development checkpoints into formal artifacts, change
query budgets, reduce case counts or override the test population after opening.
"""
import argparse,json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from .sealed_access import SealedAuthorization
from .sealed_bank import SealedBank
from .sealed_sdk import SealedVisualPredictionAdapter
from .sdk_bridge import IndependentVisualAgent,metric_registry
from .sdk_registry import build_registry
from .dataset_snapshot import stable_digest


def load_method(authorization,slot_id,resolution,device):
    slots={s['slot_id']:s for s in authorization.protocol['method_slots']}
    if slot_id not in slots:raise PermissionError('method slot not frozen')
    slot=slots[slot_id];paths=authorization.selected_paths[slot_id]
    if slot['kind']=='explicit':
        from .schema import Config
        from .persistent_reference import CompressedVisualReference
        config=json.loads(paths['configuration'].read_text())
        if set(config)!={'method','resolution','samples','max_histories','noise_floor_m'} or config['method']!='compressed_visual_reference' or config['resolution']!=resolution:
            raise ValueError('explicit reference configuration differs')
        if not isinstance(config['samples'],int) or config['samples']<1 or not isinstance(config['max_histories'],int) or not 1<=config['max_histories']<=8 or config['noise_floor_m']<=0:
            raise ValueError('invalid frozen explicit budgets')
        inner=CompressedVisualReference(Config(resolution=resolution),samples=config['samples'],max_histories=config['max_histories'],noise_floor_m=config['noise_floor_m'])
        return IndependentVisualAgent(inner,stable_digest(paths['configuration'])['sha256'])
    import torch
    from .pixel_models import PixelDynamicsModel,PixelModelConfig
    from .pixel_agent import PixelMemoryAgent
    artifact=torch.load(paths['predictor'],map_location='cpu',weights_only=True);config=artifact['config']
    if slot.get('admission_mode','prospective')=='pretest_existing':
        if config.get('stage')!='development_training' or config!=json.loads(paths['training_config'].read_text()):
            raise ValueError('existing checkpoint embedded training identity differs')
    elif config.get('stage')!='formal' or config.get('protocol_sha256')!=slot['training_protocol_sha256']:
        raise ValueError('checkpoint embedded training identity differs')
    if config.get('family')!=slot['family'] or config.get('seed')!=slot['seed']:
        raise ValueError('checkpoint family or seed differs')
    if config['bank_manifest_sha256']!=authorization.protocol['training_bank']['manifest_sha256'] or config['model']['resolution']!=resolution:raise ValueError('checkpoint data or observation profile differs')
    model=PixelDynamicsModel(PixelModelConfig(**config['model']));model.load_state_dict(artifact['model'],strict=True)
    model.to(device).eval().requires_grad_(False)
    return IndependentVisualAgent(PixelMemoryAgent(model),stable_digest(paths['predictor'])['sha256'])


def run(args):
    from persistbench.contracts import EvaluationRequest,Split,ComputeTier
    from persistbench.registry import BenchmarkRegistry
    from persistbench.evaluator import Evaluator
    from persistbench.results import RunManifest
    from .pixel_training import runtime_policy
    import torch
    authorization=SealedAuthorization(args.protocol,args.selection,protocol_sha256=args.protocol_sha256,selection_sha256=args.selection_sha256)
    spec=authorization.protocol['assays'][args.assay];profile=spec['profiles'][args.profile]
    runtime_policy(profile['evaluation_seed']);torch.set_num_threads(1)
    agent=load_method(authorization,args.slot,profile['resolution'],args.device)
    if args.output.exists():raise ValueError('sealed result attempt exists; do not overwrite')
    args.output.mkdir(parents=True);result=None;bank=None
    try:
        factor_arguments=[getattr(args,n,None) for n in ('factor_bank','factor_admission','factor_admission_sha256')]
        if args.assay=='factor_specificity' and not all(factor_arguments):raise ValueError('complete sealed factor-bank commitment required')
        if args.assay!='factor_specificity' and any(factor_arguments):raise ValueError('factor access not registered for this assay')
        with ExitStack() as stack:
            bank=stack.enter_context(SealedBank(args.bank,authorization,admission_path=args.admission,admission_sha256=args.admission_sha256,
                        audit_path=args.output/'ACCESS.private.jsonl',allow_labels=True,resolution=profile['resolution']))
            factors=None
            if args.assay=='factor_specificity':
                from .sealed_factors import SealedFactorBank
                factors=stack.enter_context(SealedFactorBank(args.factor_bank,bank,admission_path=args.factor_admission,admission_sha256=args.factor_admission_sha256))
            adapter=SealedVisualPredictionAdapter(bank,assay=args.assay,profile_id=args.profile,factor_bank=factors)
            registry_path=args.output/'REGISTRY.json';registry_path.write_text(json.dumps(build_registry(),indent=2)+'\n')
            registry=BenchmarkRegistry.load(registry_path);assay=next(s for s in registry.assays if s.assay_id=='visual_elastic_coupling/'+args.assay)
            request=EvaluationRequest(assay.assay_id,Split.TEST,ComputeTier(profile['compute_tier']),profile['evaluation_seed'],
                limit=profile['case_limit'],test_authorization=adapter.request_commitment)
            result=Evaluator(metric_registry()).run(assay=assay,request=request,adapter=adapter,agent=agent,agent_id=args.slot)
            expected=len(adapter.plan['cases'])*len(adapter.conditions) if request.limit is None else request.limit
            if len(result.case_commitments)!=expected or len(agent.trace.cases)!=expected:raise ValueError('incomplete sealed case execution')
            private_plan=adapter.emitted_private_plan
        # close() has rechecked the complete data bytes and frozen artifacts.
        provenance=dict(result.manifest.source_provenance,formal_results=True,postrun_input_verification='PASS')
        old=result.manifest
        manifest=RunManifest.create(request,agent_id=old.agent_id,adapter_id=old.adapter_id,adapter_version=old.adapter_version,
            source_provenance=provenance,protocol_version=old.protocol_version,evaluator_digest=old.evaluator_digest,
            agent_artifact_digest=old.agent_artifact_digest,agent_checkpoint_digest=old.agent_checkpoint_digest)
        result=replace(result,manifest=manifest);result.write(args.output/'RESULT.private.json');result.write_public(args.output/'PUBLIC_RESULT.json')
        (args.output/'CASE_PLAN.private.json').write_text(json.dumps(private_plan,indent=2)+'\n')
        (args.output/'LIFECYCLE.private.json').write_text(json.dumps(agent.trace.cases,indent=2)+'\n')
        completion=dict(schema='vec.formal-evaluation-completion.v1',status='PASS',protocol_sha256=authorization.protocol_sha256,
            selection_sha256=authorization.selection_sha256,bank_admission_sha256=args.admission_sha256,method_slot=args.slot,
            assay=args.assay,profile=args.profile,cases=expected,metric_records=len(result.records),test_read=True,
            artifacts={p.name:stable_digest(p)['sha256'] for p in sorted(args.output.glob('*.json'))},
            statistical_status='raw registered metric outcomes; final paired block analysis and comparison families are separate required steps')
    except Exception as exc:
        completion=dict(schema='vec.formal-evaluation-completion.v1',status='FAIL',error=repr(exc),
            test_read=bank is not None,formal_results=False)
        (args.output/'FORMAL_COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n');raise
    (args.output/'FORMAL_COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n');return completion


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','selection','bank','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','selection-sha256','admission-sha256','slot','assay','profile'):p.add_argument('--'+name,required=True)
    p.add_argument('--factor-bank',type=Path);p.add_argument('--factor-admission',type=Path);p.add_argument('--factor-admission-sha256')
    p.add_argument('--device',default='cuda:0');a=p.parse_args();print(json.dumps(run(a)),flush=True)


if __name__=='__main__':main()
