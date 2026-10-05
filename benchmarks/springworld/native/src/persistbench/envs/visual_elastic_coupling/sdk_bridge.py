"""Visual assays on the existing public PersistentAgent/Evaluator contracts.

Case setup is evaluator orchestration: it delivers independent history calls,
then optional unrelated query calls, then the scored query. The inner method
receives only the original white-listed public packets. Every comparison case
gets a clean initialization; reset within a case retains authorized history.
"""
from dataclasses import dataclass,field
from copy import deepcopy
import hashlib,json,time
import numpy as np
from persistbench.contracts import (AgentOutput,CapabilityDeclaration,ComputeTier,EpisodeContext,
    EvaluationCase,ExperienceBatch,OutputType,Split)
from .adapters import z_experience,z_query,_digest
from .prediction_assay import prepare_system,query_case,artifact_digest,donor_assignment
from .training_protocol import source_fingerprint

ASSAYS={
    'conditional_prediction':('null','matched24','matched48','matched96','wrong96'),
    'history_composition':('M','F','repeat_M','repeat_F','MM','FF','MF'),
    'delayed_prediction':('null','matched96','matched96_after_query_interference'),
    'factor_specificity':('null','matched96','wrong96','factor_m_only','factor_gamma_only','factor_k_only','factor_surface_all'),
    'predictive_transfer':('null','matched96','wrong96')}
METRICS=('position_mse','velocity_mse','center_position_mse','relative_position_mse','joint_standardized_mse')


@dataclass
class RuntimeTrace:
    cases:list=field(default_factory=list)
    pending:dict=field(default_factory=dict)


class IndependentVisualAgent:
    """Lifecycle adapter, not an alternative learned method."""
    def __init__(self,inner,checkpoint_digest=None):
        self.inner=inner;self.checkpoint_digest=checkpoint_digest
        self.implementation_bindings=dict(source=source_fingerprint(),inner_type=type(inner).__module__+'.'+type(inner).__qualname__)
        self.trace=RuntimeTrace();self._run_context=None;self._episode_context=None;self._fingerprint=None

    def initialize(self,context):
        self._run_context=context;self.trace=RuntimeTrace();declaration=self.inner.initialize(context)
        if OutputType.PREDICTION not in declaration.output_types or 'joint_state_delta' not in declaration.output_keys:
            raise ValueError('visual prediction method has no joint-state output')
        self._fingerprint=artifact_digest(self.inner)
        return CapabilityDeclaration((OutputType.PREDICTION,),output_keys=('joint_state_delta',))

    def reset(self,context):
        if self._run_context is None:raise ValueError('SDK run has not initialized')
        self.inner.initialize(self._run_context);self._episode_context=context
        self.trace.pending=dict(history_calls=0,interference_queries=0,ingestion_seconds=0.,interference_seconds=0.)
        if artifact_digest(self.inner)!=self._fingerprint:raise ValueError('case initialization changed frozen weights')

    def ingest(self,experience):
        experience.validate();setup=experience.observations
        if experience.metadata.get('observation_schema')!='vec_independent_case_setup_v1' or experience.actions is not None:
            raise ValueError('unexpected independent-case setup schema')
        if not isinstance(setup,dict) or set(setup)!={'histories','interference'}:raise ValueError('non-public case setup fields')
        if len(setup['histories'])>8 or len(setup['interference'])>4:raise ValueError('case setup exceeds registered lifecycle budget')
        for history in setup['histories']:
            self.inner.reset(EpisodeContext('opaque_donor'));payload=z_experience(history);began=time.monotonic();self.inner.ingest(payload)
            self.trace.pending['ingestion_seconds']+=time.monotonic()-began;self.trace.pending['history_calls']+=1
            payload.observations.fill(0);payload.actions.fill(0)
        for packet in setup['interference']:
            self.inner.reset(EpisodeContext('opaque_intermediate_query'));query=z_query(packet)
            before=(_digest(query.observations),_digest(query.actions));began=time.monotonic();self.inner.respond(query)
            self.trace.pending['interference_seconds']+=time.monotonic()-began;self.trace.pending['interference_queries']+=1
            if before!=(_digest(query.observations),_digest(query.actions)):raise ValueError('method mutated intermediate query')
        self.inner.reset(self._episode_context)

    def respond(self,query):
        incoming=deepcopy(query);before=(_digest(incoming.observations),_digest(incoming.actions));began=time.monotonic();output=self.inner.respond(incoming)
        if before!=(_digest(incoming.observations),_digest(incoming.actions)):raise ValueError('scored query changed')
        if artifact_digest(self.inner)!=self._fingerprint:raise ValueError('evaluation changed frozen method artifact')
        if output.output_type is not OutputType.PREDICTION:raise ValueError('method returned wrong output type')
        value=np.asarray(output.values['joint_state_delta'],np.float64)
        if value.shape!=(8,) or not np.isfinite(value).all():raise ValueError('invalid joint state prediction')
        audit=dict(self.trace.pending,response_seconds=time.monotonic()-began,persistent_numeric_bytes=self.inner.mutable_state_bytes(),
            method_diagnostics=output.diagnostics)
        self.trace.cases.append(audit)
        return AgentOutput(OutputType.PREDICTION,{'joint_state_delta':value.copy()},audit)


class VisualPredictionAdapter:
    adapter_id='visual_elastic_coupling.sdk_prediction'
    adapter_version='1.1.0-development'

    def __init__(self,bank,*,assay='conditional_prediction',stratum='continuous_new_systems',query_kind='cold',query_budget=0,horizon=16,
                 replicates=2,systems_limit=0,factor_bank=None,sampling_seed=991101):
        if assay not in ASSAYS:raise ValueError('unregistered visual prediction assay')
        if query_kind not in ('cold','moving') or horizon not in (1,4,16,32):raise ValueError('unregistered query profile')
        if query_budget not in ((0,1) if query_kind=='cold' else (0,1,3,7,15,31,63,95)):raise ValueError('unregistered query budget')
        if replicates not in (1,2) or systems_limit<0:raise ValueError('invalid paired sampling budget')
        if assay=='history_composition' and (query_kind,query_budget) != ('cold',0):raise ValueError('composition primary profile is coldq0')
        if assay=='delayed_prediction' and (query_kind,query_budget,horizon)!=('cold',0,16):raise ValueError('unregistered delayed query')
        if assay=='factor_specificity' and (factor_bank is None or horizon not in (16,32) or (query_kind,query_budget) not in (('cold',0),('moving',95))):
            raise ValueError('registered factor bank/profile required')
        self.bank=bank;self.factor_bank=factor_bank;self.assay=assay;self.stratum=stratum;self.query_kind=query_kind
        self.query_budget=query_budget;self.horizon=horizon;self.replicates=replicates;self.sampling_seed=sampling_seed
        self.keys=[k for k in bank.keys['validation'] if bank.systems[k]['stratum']==stratum]
        if systems_limit:self.keys=self.keys[:systems_limit]
        if len(self.keys)<2:raise ValueError('wrong-source controls require at least two registered systems')
        self.conditions=ASSAYS[assay];self.orders=[donor_assignment(self.keys,sampling_seed+rep) for rep in range(replicates)]
        self.plan=dict(assay=assay,stratum=stratum,query_kind=query_kind,query_budget=query_budget,horizon=horizon,
            replicates=replicates,systems=[k[1] for k in self.keys],
            assignments=[[self.keys[i][1] for i in order] for order in self.orders])
        self.emitted_private_plan=[]

    def provenance(self):
        return dict(bank_manifest_sha256=self.bank.manifest_sha256,source_fingerprint=source_fingerprint(),
            factor_manifest_sha256=None if self.factor_bank is None else self.factor_bank.manifest_sha256,
            sampling_plan_sha256=hashlib.sha256(json.dumps(self.plan,sort_keys=True).encode()).hexdigest(),
            stratum=self.stratum,query_kind=self.query_kind,query_budget=self.query_budget,horizon=self.horizon,
            systems=len(self.keys),replicates=self.replicates,conditions=self.conditions,
            lineage='visual upgrade within the coupled-sled family; not counted as an additional independent physical domain',
            test_read=False,formal_results=False)

    def iter_cases(self,request):
        request.validate()
        if request.assay_id!='visual_elastic_coupling/'+self.assay:raise ValueError('wrong visual assay request')
        if request.split is not Split.VALIDATION:raise PermissionError('development adapter cannot access sealed or training outcomes')
        block=len(self.conditions)*self.replicates
        if request.limit is not None and request.limit%block:raise ValueError('case limit would break paired system blocks')
        self.emitted_private_plan=[]
        for i,key in enumerate(self.keys):
            if request.limit is not None and i*block>=request.limit:break
            for rep in range(self.replicates):
                wrong_key=self.keys[self.orders[rep][i]]
                histories,budgets,qrows=prepare_system(self.bank,key,wrong_key,rep,self.sampling_seed+i,factor_bank=self.factor_bank)
                packet,target=query_case(self.bank,qrows[self.query_kind],self.query_budget,self.horizon)
                for condition in self.conditions:
                    source='matched96' if condition=='matched96_after_query_interference' else condition
                    interference=[]
                    if condition=='matched96_after_query_interference':
                        alternatives=self.bank.by_system[wrong_key]['moving']
                        interference=[query_case(self.bank,alternatives[j%len(alternatives)],7,16)[0] for j in range(4)]
                    setup=ExperienceBatch(('opaque_case',),('opaque_case_setup',),
                        dict(histories=deepcopy(histories[source]),interference=interference),None,
                        metadata={'observation_schema':'vec_independent_case_setup_v1','action_schema':'per_history_previous_actions'})
                    query=z_query(packet);case_id=f'{i:04d}/{rep}/{condition}';private=dict(system_key=key[1],replicate=rep,
                        stratum=self.stratum,query_episode=qrows[self.query_kind]['episode_key'],donor_budget=budgets[source],query_fingerprint=_digest(packet))
                    self.emitted_private_plan.append(dict(case_id=case_id,**private))
                    metric_target=dict(delta=target.copy(),scale=np.asarray(self.bank.statistics['scale'][str(self.horizon)]))
                    yield EvaluationCase(case_id,self.stratum+'/'+key[1],'opaque_fresh',condition,setup,query,
                        {'vec_'+name:deepcopy(metric_target) for name in METRICS},private)


def component_metric(kind,prediction,target):
    prediction=np.asarray(prediction,float);truth=np.asarray(target['delta'],float);scale=np.asarray(target['scale'],float)
    if prediction.shape!=(8,) or truth.shape!=(8,) or scale.shape!=(8,) or not np.isfinite(np.r_[prediction,truth,scale]).all() or np.any(scale<=0):
        raise ValueError('invalid scored physical state/normalization')
    error=prediction-truth
    values={'position_mse':error[:4],'velocity_mse':error[4:],
        'center_position_mse':(error[:2]+error[2:4])/2,'relative_position_mse':error[2:4]-error[:2],
        'joint_standardized_mse':error/scale}
    if kind not in values:raise ValueError('unregistered metric')
    return float(np.mean(values[kind]**2))


def metric_registry():
    from persistbench.metrics import MetricRegistry
    registry=MetricRegistry()
    for kind in METRICS:registry.register('vec_'+kind,lambda prediction,target,kind=kind:component_metric(kind,prediction,target))
    return registry
