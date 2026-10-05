"""Fixed-query factor-specificity assay with actual ingest-time compression.

Each condition ingests its one historical episode once and answers four fresh
query packets from retained state. Physical targets remain evaluator-owned.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from persistbench.contracts import RunContext,EpisodeContext,ComputeTier,Split
from .schema import Config
from .adapters import z_experience,z_query,_digest
from .prediction_assay import prepare_system,query_case,donor_assignment,artifact_digest
from .calibration import component_errors

CONDITIONS=('null','matched96','wrong96','factor_m_only','factor_gamma_only','factor_k_only','factor_surface_all')


def initialize_worker(settings):
    global BANK,FACTORS,AGENT,SETTINGS
    import torch
    from .pixel_training import TrainingBank,runtime_policy
    from .specificity_bank import FactorDonorBank
    SETTINGS=settings;runtime_policy(17);torch.set_num_threads(1)
    BANK=TrainingBank(settings['bank']);FACTORS=FactorDonorBank(settings['factor_bank'],BANK)
    if settings['explicit']:
        from .persistent_reference import CompressedVisualReference
        AGENT=CompressedVisualReference(Config(resolution=128),samples=settings['samples'])
    else:
        from .pixel_models import PixelDynamicsModel,PixelModelConfig
        from .pixel_agent import PixelMemoryAgent
        for name in ('pixel_models.py','pixel_training.py'):
            original=Path(settings['model_source'])/'src/persistbench/envs/visual_elastic_coupling'/name
            if original.read_bytes()!=(Path(__file__).parent/name).read_bytes():raise ValueError('factor assay model source differs')
        checkpoint=torch.load(settings['checkpoint'],map_location='cpu',weights_only=True)
        if checkpoint['config']['bank_manifest_sha256']!=BANK.manifest_sha256:raise ValueError('factor checkpoint bank binding differs')
        model=PixelDynamicsModel(PixelModelConfig(**checkpoint['config']['model']));model.load_state_dict(checkpoint['model'],strict=True)
        model.to(settings['device']).eval().requires_grad_(False);AGENT=PixelMemoryAgent(model)


def evaluate(job):
    index,key,wrong_key,replicate=job;start=time.monotonic()
    histories,budgets,qrows=prepare_system(BANK,key,wrong_key,replicate,990321+index%SETTINGS['systems'],factor_bank=FACTORS)
    if any(name not in histories for name in CONDITIONS):raise ValueError('factor family absent for registered system')
    queries=[(kind,budget,horizon,*query_case(BANK,qrows[kind],budget,horizon))
        for kind,budget in (('cold',0),('moving',95)) for horizon in (16,32)]
    context=RunContext('visual_elastic_coupling/factor-specificity','1.1',Split.VALIDATION,ComputeTier.STANDARD,990301)
    frozen=artifact_digest(AGENT);rows=[]
    for condition in CONDITIONS:
        AGENT.initialize(context);began=time.monotonic()
        for history in histories[condition]:
            AGENT.reset(EpisodeContext('opaque_donor'));experience=z_experience(history);AGENT.ingest(experience)
            experience.observations.fill(0);experience.actions.fill(0)
        ingestion=time.monotonic()-began
        for kind,budget,horizon,packet,target in queries:
            packet_hash=_digest(packet);AGENT.reset(EpisodeContext('opaque_fresh_query'));incoming=z_query(packet)
            before=(_digest(incoming.observations),_digest(incoming.actions));began=time.monotonic();response=AGENT.respond(incoming)
            if before!=(_digest(incoming.observations),_digest(incoming.actions)) or packet_hash!=_digest(packet):raise ValueError('factor method mutated query')
            prediction=np.asarray(response.values['joint_state_delta'],float)
            if prediction.shape!=(8,) or not np.isfinite(prediction).all():raise ValueError('invalid factor physical prediction')
            errors=component_errors(prediction,target);errors['joint_train_standardized_mse']=float(np.mean(((prediction-target)/BANK.statistics['scale'][str(horizon)])**2))
            rows.append(dict(condition=condition,query_kind=kind,query_budget=budget,horizon=horizon,prediction=prediction.tolist(),target=target.tolist(),errors=errors,
                budget=budgets[condition],query_episode=qrows[kind]['episode_key'],query_fingerprint=packet_hash,
                diagnostic=dict(ingestion_seconds=ingestion,ingestion_shared_across_four_queries=True,response_seconds=time.monotonic()-began,
                    memory_bytes=AGENT.mutable_state_bytes(),method=response.diagnostics)))
    if artifact_digest(AGENT)!=frozen:raise ValueError('factor assay changed frozen model weights/buffers')
    result=dict(index=index,system_key=key[1],stratum=BANK.systems[key]['stratum'],theta=BANK.systems[key]['theta'],replicate=replicate,
        wrong_system_key=wrong_key[1],rows=rows,seconds=time.monotonic()-start)
    (Path(SETTINGS['output'])/f'case_{index:04d}.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--factor-bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--explicit',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--model-source',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=1);p.add_argument('--replicates',type=int,choices=(1,2),default=2)
    p.add_argument('--samples',type=int,default=256);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    from .pixel_training import TrainingBank
    from .specificity_bank import FactorDonorBank
    from .training_protocol import source_fingerprint
    if a.output.exists():raise ValueError('factor assay attempt exists')
    if a.workers<1 or a.samples<1:raise ValueError('invalid factor assay compute budget')
    if a.explicit and a.checkpoint is not None:raise ValueError('select exactly one factor method')
    if not a.explicit and (a.checkpoint is None or a.model_source is None):raise ValueError('checkpoint and model source are required')
    if a.device.startswith('cuda') and a.workers!=1:raise ValueError('one GPU worker per registered method')
    bank=TrainingBank(a.bank);factors=FactorDonorBank(a.factor_bank,bank);keys=bank.selection_keys
    if a.replicates>factors.config['replicates']:raise ValueError('factor replicate support incomplete')
    if a.smoke:keys=[next(k for k in keys if bank.systems[k]['stratum']==name) for name in ('continuous_new_systems','heldout_factorial_combinations')]
    assignments=[donor_assignment(keys,991052+rep) for rep in range(a.replicates)]
    settings=dict(bank=str(a.bank),factor_bank=str(a.factor_bank),output=str(a.output),explicit=a.explicit,checkpoint=str(a.checkpoint) if a.checkpoint else None,
        model_source=str(a.model_source) if a.model_source else None,device=a.device,samples=a.samples,replicates=a.replicates,systems=len(keys))
    jobs=[(len(keys)*rep+i,key,keys[assignments[rep][i]],rep) for rep in range(a.replicates) for i,key in enumerate(keys)]
    a.output.mkdir(parents=True);config=dict(schema='vec.factor-specificity-development.v1.1',bank_manifest_sha256=bank.manifest_sha256,
        factor_manifest_sha256=factors.manifest_sha256,source_fingerprint=source_fingerprint(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
        conditions=CONDITIONS,systems=len(keys),replicates=a.replicates,cases=len(jobs),samples=a.samples if a.explicit else None,
        assignments=[{key[1]:keys[order[i]][1] for i,key in enumerate(keys)} for order in assignments],
        scope='matched96 fixed initial-state/action donor vs single-factor and all-factor replacements; coldq0/movingq95 h16/32; main validation systems only',
        lifecycle='initialize per condition, ingest once, erase passed raw payload, reset for each fresh query; frozen weights and query bytes checked',formal_results=False,test_read=False)
    (a.output/'EVALUATION_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');rows=[];errors=[];began=time.monotonic()
    def accept(job,index):
        try:rows.append(job())
        except Exception as exc:errors.append(dict(index=index,error=repr(exc)))
        print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors))),flush=True)
    if a.workers==1:
        initialize_worker(settings)
        for job in jobs:accept(lambda job=job:evaluate(job),job[0])
    else:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(settings,)) as pool:
            futures={pool.submit(evaluate,job):job[0] for job in jobs}
            for future in as_completed(futures):accept(future.result,futures[future])
    (a.output/'EVALUATION_REPORT.json').write_text(json.dumps(dict(config=config,cases=rows,errors=errors,seconds=time.monotonic()-began),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
