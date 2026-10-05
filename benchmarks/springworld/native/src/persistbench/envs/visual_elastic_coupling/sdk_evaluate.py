"""Real data and frozen methods through the existing PersistBench Evaluator.

One run fixes the query profile, horizon and population stratum. Those axes are
never silently pooled into a single OOD or all-horizons aggregate.
"""
import argparse,hashlib,json
from pathlib import Path
from .sdk_bridge import ASSAYS,IndependentVisualAgent,VisualPredictionAdapter,metric_registry
from .sdk_registry import build_registry,evaluator_fingerprint
from .schema import Config


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assay',choices=tuple(ASSAYS),required=True);p.add_argument('--stratum',default='continuous_new_systems')
    p.add_argument('--query-kind',choices=('cold','moving'),default='cold');p.add_argument('--query-budget',type=int,default=0);p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--replicates',type=int,default=1);p.add_argument('--systems-limit',type=int,default=2);p.add_argument('--factor-bank',type=Path)
    p.add_argument('--explicit',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--model-source',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--posterior-samples',type=int,default=256);a=p.parse_args()
    from persistbench.contracts import EvaluationRequest,Split,ComputeTier
    from persistbench.registry import BenchmarkRegistry
    from persistbench.evaluator import Evaluator
    from .pixel_training import TrainingBank,runtime_policy
    from .specificity_bank import FactorDonorBank
    import torch
    if a.output.exists():raise ValueError('SDK evaluation attempt exists')
    if a.explicit and a.checkpoint is not None:raise ValueError('select exactly one model source')
    runtime_policy(17);torch.set_num_threads(1);bank=TrainingBank(a.bank)
    factors=None if a.factor_bank is None else FactorDonorBank(a.factor_bank,bank)
    adapter=VisualPredictionAdapter(bank,assay=a.assay,stratum=a.stratum,query_kind=a.query_kind,query_budget=a.query_budget,horizon=a.horizon,
        replicates=a.replicates,systems_limit=a.systems_limit,factor_bank=factors)
    checkpoint_digest=None
    if a.explicit:
        from .persistent_reference import CompressedVisualReference
        inner=CompressedVisualReference(Config(resolution=128),samples=a.posterior_samples);name='explicit'
    else:
        from .pixel_models import PixelDynamicsModel,PixelModelConfig
        from .pixel_agent import PixelMemoryAgent
        if a.checkpoint is None or a.model_source is None:raise ValueError('frozen model and exact source required')
        for filename in ('pixel_models.py','pixel_training.py'):
            original=a.model_source/'src/persistbench/envs/visual_elastic_coupling'/filename
            if original.read_bytes()!=(Path(__file__).parent/filename).read_bytes():raise ValueError('SDK model source differs from training')
        artifact=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if artifact['config']['bank_manifest_sha256']!=bank.manifest_sha256:raise ValueError('SDK checkpoint bank binding differs')
        model=PixelDynamicsModel(PixelModelConfig(**artifact['config']['model']));model.load_state_dict(artifact['model'],strict=True)
        model.to(a.device).eval().requires_grad_(False);inner=PixelMemoryAgent(model);name=model.cfg.family
        checkpoint_digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    a.output.mkdir(parents=True);registry_path=a.output/'REGISTRY.json';registry_path.write_text(json.dumps(build_registry(),indent=2)+'\n')
    registry=BenchmarkRegistry.load(registry_path);assay=next(x for x in registry.assays if x.assay_id=='visual_elastic_coupling/'+a.assay)
    agent=IndependentVisualAgent(inner,checkpoint_digest)
    request=EvaluationRequest(assay.assay_id,Split.VALIDATION,ComputeTier.STANDARD,990301)
    result=Evaluator(metric_registry()).run(assay=assay,request=request,adapter=adapter,agent=agent,agent_id=name)
    result.write(a.output/'RESULT.json');result.write_public(a.output/'PUBLIC_RESULT.json')
    (a.output/'PRIVATE_CASE_PLAN.json').write_text(json.dumps(adapter.emitted_private_plan,indent=2)+'\n')
    (a.output/'LIFECYCLE_TRACE.json').write_text(json.dumps(agent.trace.cases,indent=2)+'\n')
    report=dict(status='EXECUTED',cases=len(agent.trace.cases),metric_records=len(result.records),conditions=list(adapter.conditions),
        registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),evaluator_fingerprint=evaluator_fingerprint(),
        checkpoint_sha256=checkpoint_digest,aggregates=result.aggregates,formal_results=False,test_read=False)
    (a.output/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
