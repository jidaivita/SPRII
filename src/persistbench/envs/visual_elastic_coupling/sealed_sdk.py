"""Frozen test profiles on the same observation/lifecycle and SDK path."""
from persistbench.contracts import Split
from .planned_sdk import PlannedVisualPredictionAdapter
from .sealed_bank import SealedBank
from .sdk_bridge import ASSAYS,METRICS
from .evaluation_blocks import digest


class SealedVisualPredictionAdapter(PlannedVisualPredictionAdapter):
    adapter_id='visual_elastic_coupling.sealed_prediction'
    adapter_version='1.1.0'
    allowed_split=Split.TEST

    def __init__(self,bank,*,assay,profile_id,factor_bank=None):
        if not isinstance(bank,SealedBank) or bank.closed or not bank.allow_labels:raise PermissionError('authorized evaluator target reader required')
        if assay not in ASSAYS:raise ValueError('unimplemented sealed prediction assay')
        spec=bank.authorization.protocol['assays'][assay]
        if not isinstance(spec['profiles'],dict) or profile_id not in spec['profiles']:raise PermissionError('profile not frozen')
        profile=spec['profiles'][profile_id]
        fields={'stratum','query_kind','query_budget','horizon','resolution','conditions','metrics','compute_tier','evaluation_seed','case_limit'}
        if set(profile)!=fields:raise ValueError('incomplete or unrecognized frozen prediction profile')
        if profile['resolution']!=bank.resolution or profile['conditions']!=list(ASSAYS[assay]) or profile['metrics']!=list(METRICS):raise ValueError('frozen prediction permissions/conditions/metrics differ')
        if profile['stratum'] not in bank.plans:raise ValueError('profile population not in sealed bank')
        if assay=='factor_specificity':
            from .sealed_factors import SealedFactorBank
            if not isinstance(factor_bank,SealedFactorBank) or factor_bank.closed or factor_bank.bank is not bank:raise PermissionError('factor assay requires the exact authorized sealed factor bank')
        elif factor_bank is not None:raise ValueError('factor access is not part of this frozen assay')
        self.profile=profile;self.profile_id=profile_id
        self.request_commitment=digest([bank.authorization.protocol_sha256,bank.authorization.selection_sha256,bank.admission_sha256,assay,profile_id])
        if factor_bank is not None:self.request_commitment=digest([self.request_commitment,factor_bank.admission_sha256])
        super().__init__(bank,bank.plans[profile['stratum']]['plan'],assay=assay,query_kind=profile['query_kind'],query_budget=profile['query_budget'],horizon=profile['horizon'],factor_bank=factor_bank)

    def check_request(self,request):
        request.validate()
        if self.bank.closed or request.split is not Split.TEST or request.test_authorization!=self.request_commitment:raise PermissionError('request is not bound to this frozen test profile')
        if request.assay_id!='visual_elastic_coupling/'+self.assay:raise ValueError('wrong sealed assay')
        if request.limit!=self.profile['case_limit'] or request.seed!=self.profile['evaluation_seed'] or request.compute_tier.value!=self.profile['compute_tier']:
            raise ValueError('sealed case budget, evaluation randomness or compute tier differs')

    def provenance(self):
        result=super().provenance()
        result.update(split='test',test_read=bool(self.emitted_private_plan),formal_results=False,
            protocol_sha256=self.bank.authorization.protocol_sha256,selection_sha256=self.bank.authorization.selection_sha256,
            bank_admission_sha256=self.bank.admission_sha256,profile_id=self.profile_id,
            completion_requirement='runner must close and reverify bank/model inputs before admitting a formal result')
        if self.factor_bank is not None:result['factor_admission_sha256']=self.factor_bank.admission_sha256
        return result
