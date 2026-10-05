"""Fixed-artifact prediction, composition, delayed-value and transfer assays.

The case manifest and targets stay evaluator-side. Every condition independently
initializes the same artifact and ingests the registered history before query.
No raw donor is supplied to respond, including after query interference.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from persistbench.contracts import RunContext,EpisodeContext,ComputeTier,Split
from .schema import Episode,Config,history_payload,query_packet
from .adapters import z_experience,z_query,_digest
from .calibration import component_errors


def encoded_episode(bank,row):
    images,actions=bank.visible(row['episode_key'])
    # The episode codec never sees simulator states or source identities.
    return Episode(images,actions,np.arange(len(images))*.05,np.zeros((len(images),8)),{})


def query_case(bank,row,budget,horizon):
    packet=query_packet(encoded_episode(bank,row),row['anchor'],budget,horizon)
    packet['target_spec']='cold_rest_joint_state_delta_8d' if row['kind']=='cold' else 'passive_prefix95_joint_state_delta_8d'
    states=bank.labels(row['episode_key']);target=states[row['anchor']+horizon]-states[row['anchor']]
    return packet,target


def history_case(bank,row,frames):return history_payload(encoded_episode(bank,row),0,frames-1)


def artifact_digest(agent):
    model=getattr(agent,'model',None)
    if model is None:return None
    return _digest({key:value.detach().cpu().numpy() for key,value in model.state_dict().items()})


def run_condition(agent,context,histories,packet,*,interference=()):
    agent.initialize(context);begin=time.monotonic()
    for history in histories:
        agent.reset(EpisodeContext('opaque_donor'))
        experience=z_experience(history);agent.ingest(experience)
        # Only method-retained state can be used later. No original payload is
        # passed again; erasing this copy also catches accidental shared buffers.
        experience.observations.fill(0);experience.actions.fill(0)
    ingestion=time.monotonic()-begin
    for distractor in interference:
        agent.reset(EpisodeContext('opaque_intermediate_query'));incoming=z_query(distractor)
        before=(_digest(incoming.observations),_digest(incoming.actions));agent.respond(incoming)
        if before!=(_digest(incoming.observations),_digest(incoming.actions)):raise ValueError('method mutated intermediate query')
    agent.reset(EpisodeContext('opaque_fresh_query'));incoming=z_query(packet)
    before=(_digest(incoming.observations),_digest(incoming.actions));begin=time.monotonic();response=agent.respond(incoming)
    elapsed=time.monotonic()-begin
    if before!=(_digest(incoming.observations),_digest(incoming.actions)):raise ValueError('method mutated fixed query')
    value=np.asarray(response.values['joint_state_delta'],float)
    if value.shape!=(8,) or not np.isfinite(value).all():raise ValueError('invalid physical prediction')
    return value,dict(ingestion_seconds=ingestion,response_seconds=elapsed,intermediate_queries=len(interference),
        memory_bytes=agent.mutable_state_bytes(),method_diagnostics=response.diagnostics)


def choose_distinct(rng,rows,count):
    if len(rows)<count:raise ValueError('registered independent donor count is unavailable')
    return [rows[i] for i in rng.choice(len(rows),count,replace=False)]


def donor_assignment(keys,seed):
    """Balanced wrong-source assignment, independently randomized per repeat."""
    if len(keys)<2:raise ValueError('wrong-source assay requires at least two systems')
    rng=np.random.default_rng(seed)
    while True:
        order=rng.permutation(len(keys))
        if np.all(order!=np.arange(len(keys))):return order


def prepare_system(bank,key,wrong_key,replicate,seed,*,factor_bank=None):
    rng=np.random.default_rng(np.random.SeedSequence([seed,replicate]));groups=bank.by_system[key]
    histories={};metadata={}
    forced=choose_distinct(rng,bank.eligible(key,'forced',96),1)[0]
    factor_histories={};factor_metadata={}
    if factor_bank is not None:
        source,factor_histories,factor_metadata=factor_bank.histories(key,replicate,bank.resolution)
        if source is not None:
            forced=bank.rows[source]
            if (forced['split'],forced['system_key'])!=key:raise ValueError('paired factor source belongs to another system')
    wrong=choose_distinct(rng,bank.eligible(wrong_key,'forced',96),1)[0]
    members={'matched96':[(forced,96)],'wrong96':[(wrong,96)],'matched24':[(forced,24)],'matched48':[(forced,48)],'null':[]}
    m=choose_distinct(rng,bank.eligible(key,'mass',48),2);f=choose_distinct(rng,bank.eligible(key,'free',48),2)
    members.update(M=[(m[0],48)],F=[(f[0],48)],MM=[(m[0],48),(m[1],48)],FF=[(f[0],48),(f[1],48)],
        MF=[(m[0],48),(f[0],48)],repeat_M=[(m[0],48),(m[0],48)],repeat_F=[(f[0],48),(f[0],48)])
    for condition,selected in members.items():
        histories[condition]=[history_case(bank,r,n) for r,n in selected]
        unique={r['episode_key']:n for r,n in selected}
        metadata[condition]=dict(donor_episode_keys=[r['episode_key'] for r,n in selected],processed_frames=sum(n for _,n in selected),
            unique_frames=sum(unique.values()),unique_transitions=sum(n-1 for n in unique.values()),episodes=len(selected),
            effort_n2_s=sum(float(np.sum(bank.visible(r['episode_key'])[1][:n-1]**2)*.05) for r,n in selected))
    histories.update(factor_histories);metadata.update(factor_metadata)
    cold=[r for r in groups['cold'] if r['template']=='pulse_050'];moving=groups['moving']
    qrows={'cold':cold[replicate%len(cold)],'moving':moving[replicate%len(moving)]}
    if any(r['episode_key'] in {x for meta in metadata.values() for x in meta['donor_episode_keys']} for r in qrows.values()):
        raise ValueError('query and donor episode supports overlap')
    return histories,metadata,qrows


def evaluate_system(agent,bank,key,wrong_key,replicate,seed,*,full_budgets=False,include_composition=True,delay_queries=4,factor_bank=None):
    histories,budgets,qrows=prepare_system(bank,key,wrong_key,replicate,seed,factor_bank=factor_bank)
    context=RunContext('visual_elastic_coupling/prediction','1.1',Split.VALIDATION,ComputeTier.STANDARD,990301)
    frozen=artifact_digest(agent);rows=[];start=time.monotonic()
    for kind,row in qrows.items():
        choices=(0,1) if kind=='cold' else ((0,1,3,7,15,31,63,95) if full_budgets else (0,7,95))
        for q in choices:
            for h in (1,4,16,32):
                packet,target=query_case(bank,row,q,h);fingerprint=_digest(packet)
                conditions=['null','matched96','wrong96']
                if kind=='cold' and q==0:conditions+=['matched24','matched48']
                if include_composition and kind=='cold' and q==0 and h in (16,32):conditions+=['M','F','MM','FF','MF','repeat_M','repeat_F']
                if h in (16,32) and ((kind=='cold' and q==0) or (kind=='moving' and q==95)):
                    conditions += [name for name in histories if name.startswith('factor_')]
                for condition in conditions:
                    prediction,diagnostic=run_condition(agent,context,histories[condition],packet)
                    if _digest(packet)!=fingerprint:raise ValueError('condition changed fixed public query')
                    error=component_errors(prediction,target)
                    error['joint_train_standardized_mse']=float(np.mean(((prediction-target)/bank.statistics['scale'][str(h)])**2))
                    rows.append(dict(condition=condition,query_kind=kind,query_budget=q,horizon=h,prediction=prediction.tolist(),target=target.tolist(),
                        errors=error,budget=budgets[condition],diagnostic=diagnostic,query_episode=row['episode_key'],query_fingerprint=fingerprint))
                    if delay_queries and condition=='matched96' and kind=='cold' and q==0 and h==16:
                        unrelated=bank.by_system[wrong_key]['moving'];interference=[query_case(bank,unrelated[i%len(unrelated)],7,16)[0] for i in range(delay_queries)]
                        delayed,delay_diag=run_condition(agent,context,histories[condition],packet,interference=interference)
                        delay_error=component_errors(delayed,target);delay_error['joint_train_standardized_mse']=float(np.mean(((delayed-target)/bank.statistics['scale']['16'])**2))
                        rows.append(dict(condition='matched96_after_query_interference',query_kind=kind,query_budget=q,horizon=h,prediction=delayed.tolist(),target=target.tolist(),
                            errors=delay_error,budget=budgets[condition],diagnostic=delay_diag,query_episode=row['episode_key'],
                            max_abs_prediction_change=float(np.max(np.abs(delayed-prediction))),query_fingerprint=fingerprint))
    if artifact_digest(agent)!=frozen:raise ValueError('artifact weights or normalization changed during evaluation')
    return dict(system_key=key[1],wrong_system_key=wrong_key[1],stratum=bank.systems[key]['stratum'],replicate=replicate,rows=rows,seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--factor-bank',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--model-source',type=Path);p.add_argument('--explicit',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--systems-per-stratum',type=int,default=2);p.add_argument('--replicates',type=int,default=1)
    p.add_argument('--full-budgets',action='store_true');p.add_argument('--posterior-samples',type=int,default=256);a=p.parse_args()
    from .pixel_training import TrainingBank,runtime_policy
    from .training_protocol import source_fingerprint,file_digest
    import torch
    if a.systems_per_stratum<0 or a.replicates<1 or a.posterior_samples<1:raise ValueError('invalid assay budget')
    if a.explicit and a.checkpoint is not None:raise ValueError('select exactly one method source')
    bank=TrainingBank(a.bank,128)
    from .specificity_bank import FactorDonorBank
    factor_bank=None if a.factor_bank is None else FactorDonorBank(a.factor_bank,bank)
    if factor_bank is not None and a.replicates>factor_bank.config['replicates']:raise ValueError('factor donor replicate support is incomplete')
    if a.output.exists():raise ValueError('evaluation attempt already exists')
    a.output.mkdir(parents=True);runtime_policy(17);torch.set_num_threads(1)
    if a.explicit:
        from .persistent_reference import CompressedVisualReference
        agent=CompressedVisualReference(Config(resolution=128),samples=a.posterior_samples)
    else:
        from .pixel_models import PixelDynamicsModel,PixelModelConfig
        from .pixel_agent import PixelMemoryAgent
        if a.checkpoint is None or a.model_source is None:raise ValueError('frozen checkpoint and exact model source are required')
        for name in ('pixel_models.py','pixel_training.py'):
            if (Path(__file__).parent/name).read_bytes()!=(a.model_source/name).read_bytes():raise ValueError('evaluation model source differs from training: '+name)
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['config']['bank_manifest_sha256']!=bank.manifest_sha256:raise ValueError('checkpoint data provenance differs')
        model=PixelDynamicsModel(PixelModelConfig(**checkpoint['config']['model']));model.load_state_dict(checkpoint['model']);model.to(a.device);agent=PixelMemoryAgent(model)
    groups={}
    for key in bank.keys['validation']:groups.setdefault(bank.systems[key]['stratum'],[]).append(key)
    selected=[k for group in groups.values() for k in (group[:a.systems_per_stratum] if a.systems_per_stratum else group)]
    orders=[donor_assignment(selected,990311+replicate) for replicate in range(a.replicates)]
    config=dict(schema='vec.fixed-artifact-validation.v1.1',bank_manifest_sha256=bank.manifest_sha256,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
        source_fingerprint=source_fingerprint(),training_source_fingerprint=None if a.explicit else checkpoint['config']['source_fingerprint'],
        selected_update=None if a.explicit else checkpoint['update'],target_statistics_sha256=file_digest(bank.root/'TRAIN_TARGET_STATISTICS.json'),
        method='explicit' if a.explicit else model.cfg.family,posterior_samples=a.posterior_samples if a.explicit else None,systems=len(selected),replicates=a.replicates,
        assignments=[{key[1]:selected[order[i]][1] for i,key in enumerate(selected)} for order in orders],
        factor_manifest_sha256=None if factor_bank is None else factor_bank.manifest_sha256,
        factor_scope=None if factor_bank is None else 'registered in-range continuous and heldout-combination validation systems; coldq0/movingq95 at h16/32',
        split='validation',full_budgets=a.full_budgets,test_read=False,formal_results=False,
        delay='registered query interference from another system, fresh boundaries and no extra ingest; does not claim resistance to arbitrary history pollution')
    (a.output/'EVALUATION_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');results=[];errors=[]
    for i,key in enumerate(selected):
        for replicate in range(a.replicates):
            try:
                row=evaluate_system(agent,bank,key,selected[orders[replicate][i]],replicate,990321+i,full_budgets=a.full_budgets,factor_bank=factor_bank)
                results.append(row);(a.output/f'case_{i:04d}_{replicate}.json').write_text(json.dumps(row,indent=2)+'\n')
            except Exception as exc:errors.append(dict(system_key=key[1],replicate=replicate,error=repr(exc)))
            print(json.dumps(dict(completed=len(results)+len(errors),total=len(selected)*a.replicates,errors=len(errors))),flush=True)
    (a.output/'EVALUATION_REPORT.json').write_text(json.dumps(dict(config=config,cases=results,errors=errors),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
