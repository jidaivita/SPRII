"""Preallocated independent-block cases through the public SDK evaluator.

This base entry accepts validation rehearsals only. The separate sealed_sdk
subclass requires the frozen authorization and a separate test-bank reader.
"""
from copy import deepcopy
from collections import Counter
import numpy as np
from persistbench.contracts import EvaluationCase,ExperienceBatch,Split
from .sdk_bridge import ASSAYS,METRICS
from .planned_support import PlannedSupport
from .prediction_assay import query_case
from .adapters import z_query,_digest
from .training_protocol import source_fingerprint


class PlannedVisualPredictionAdapter:
    adapter_id='visual_elastic_coupling.planned_sdk_prediction'
    adapter_version='1.1.0-development'
    allowed_split=Split.VALIDATION

    def __init__(self,bank,plan,*,assay='conditional_prediction',query_kind='cold',query_budget=0,horizon=16,factor_bank=None):
        if assay not in ASSAYS or query_kind not in ('cold','moving') or horizon not in (1,4,16,32):raise ValueError('unknown planned query/assay')
        if query_budget not in ((0,1) if query_kind=='cold' else (0,1,3,7,15,31,63,95)):raise ValueError('unknown planned query budget')
        if assay=='history_composition' and (query_kind,query_budget)!=('cold',0):raise ValueError('composition primary query differs')
        if assay=='delayed_prediction' and (query_kind,query_budget,horizon)!=('cold',0,16):raise ValueError('delay profile differs')
        if assay=='factor_specificity' and (factor_bank is None or horizon not in (16,32) or (query_kind,query_budget) not in (('cold',0),('moving',95))):
            raise ValueError('factor assay requires its paired bank and profile')
        self.bank=bank;self.plan=plan;self.assay=assay;self.conditions=ASSAYS[assay];self.factor_bank=factor_bank
        self.query_kind=query_kind;self.query_budget=query_budget;self.horizon=horizon
        self.support=PlannedSupport(bank,plan,split=self.allowed_split.value);self.emitted_private_plan=[]

    def provenance(self):
        return dict(bank_manifest_sha256=self.bank.manifest_sha256,source_fingerprint=source_fingerprint(),
            factor_manifest_sha256=None if self.factor_bank is None else self.factor_bank.manifest_sha256,
            sampling_plan_sha256=self.plan['plan_sha256'],stratum=self.plan['stratum'],query_kind=self.query_kind,
            query_budget=self.query_budget,horizon=self.horizon,conditions=self.conditions,
            physical_systems=len(self.plan['system_generation_order']),replicates=self.plan['replicates'],
            independent_blocks=len({r['block_id'] for r in self.plan['cases']}),independent_unit=self.plan['independent_unit'],
            split='validation',test_read=False,formal_results=False)

    def iter_cases(self,request):
        self.check_request(request)
        block_counts=Counter(row['block_id'] for row in self.plan['cases'])
        if len(set(block_counts.values()))!=1:raise ValueError('unequal planned independent block sizes')
        cases_per_block=next(iter(block_counts.values()))*len(self.conditions)
        if request.limit is not None and request.limit%cases_per_block:raise ValueError('limit would split an independent block')
        self.emitted_private_plan=[];emitted=0
        for planned in self.plan['cases']:
            if request.limit is not None and emitted>=request.limit:break
            histories,budgets,qrows,interference_rows=self.support.prepare(planned['case_id'],query_kind=self.query_kind,
                query_budget=self.query_budget,horizon=self.horizon,conditions=self.conditions,factor_bank=self.factor_bank)
            packet,target=query_case(self.bank,qrows[self.query_kind],self.query_budget,self.horizon)
            for condition in self.conditions:
                source='matched96' if condition=='matched96_after_query_interference' else condition
                interference=[query_case(self.bank,row,7,16)[0] for row in interference_rows] if condition=='matched96_after_query_interference' else []
                setup=ExperienceBatch(('opaque_case',),('opaque_case_setup',),dict(histories=deepcopy(histories[source]),interference=interference),None,
                    metadata={'observation_schema':'vec_independent_case_setup_v1','action_schema':'per_history_previous_actions'})
                case_id=planned['case_id']+'/'+condition
                private=dict(base_case_id=planned['case_id'],block_id=planned['block_id'],system_key=planned['system_key'],
                    wrong_system_key=planned['wrong_system_key'],replicate=planned['replicate'],stratum=self.plan['stratum'],
                    query_episode=qrows[self.query_kind]['episode_key'],donor_budget=budgets[source],query_fingerprint=_digest(packet),
                    interference_episodes=[r['episode_key'] for r in interference_rows] if interference else [])
                self.emitted_private_plan.append(dict(case_id=case_id,**private));emitted+=1
                metric_target=dict(delta=target.copy(),scale=np.asarray(self.bank.statistics['scale'][str(self.horizon)]))
                yield EvaluationCase(case_id,self.plan['stratum']+'/'+planned['system_key'],'opaque_fresh',condition,setup,z_query(packet),
                    {'vec_'+name:deepcopy(metric_target) for name in METRICS},private)

    def check_request(self,request):
        request.validate()
        if request.split is not Split.VALIDATION:raise PermissionError('planned rehearsal adapter cannot open sealed test')
        if request.assay_id!='visual_elastic_coupling/'+self.assay:raise ValueError('wrong planned assay')


def paired_records(bundle,private_plan,*,profile_key,model_seed):
    """Convert SDK metric rows for the strict independent-block analyzer."""
    plans={r['case_id']:r for r in private_plan};rows={}
    for record in bundle.records:
        if record.case_id not in plans:raise ValueError('SDK result has no private planned identity')
        plan=plans[record.case_id]
        row=rows.setdefault(record.case_id,dict(case_id=plan['base_case_id'],model_seed=model_seed,condition=record.condition,
            profile_key=profile_key,status='OK',metrics={},query_fingerprint=plan['query_fingerprint'],
            target_fingerprint=bundle.case_commitments[record.case_id]))
        if record.metric in row['metrics']:raise ValueError('duplicate SDK metric')
        row['metrics'][record.metric]=record.value
    if set(rows)!=set(plans):raise ValueError('SDK result omitted planned outcomes')
    return list(rows.values())
