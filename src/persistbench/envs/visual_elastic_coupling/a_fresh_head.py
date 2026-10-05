"""Candidate A fresh heads with common 128d context slots and frozen features.

This preserves B0's full native128d context and pads split64d/Oracle3d without
truncation. Nominal head parameters match; effective information dimensions do
not. Independent head fitting and fixed-head interventions are separate jobs.
No formal head fitting, checkpoint qualification or test access occurs here.
"""
import numpy as np
import torch
from torch import nn
from .a_memory import CompressedAHistoryAgent
from .a_head_statistics import HORIZONS,AHeadStatistics
from .observations import public_query_images

HEAD_PROFILE='common_context128_slot64_horizon16_mlp256x2_v1'
INPUT_KEYS={'query_embedding','context_slot','future_actions','action_mask','horizon_index'}


class AFreshFeatures(CompressedAHistoryAgent):
    def __init__(self,model,*,max_histories=1,aggregation='last'):
        super().__init__(model,max_histories=max_histories,aggregation=aggregation,no_history_policy='zero_code')

    def head_inputs(self,query):
        self._guard()
        images,actions=public_query_images(query,self.config)
        if len(images) not in (1,2) or len(actions) not in HORIZONS:
            raise ValueError('candidate A fresh-head query requires q0/1 and h1/2/4/8/16')
        device=next(self.model.parameters()).device
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            q=self.model.observation(torch.tensor(images[None],device=device,dtype=torch.float32))[:,-1]
            slot=torch.zeros((1,128),device=device,dtype=torch.float32)
            slot[:,:self.representation_dim]=self.history_code()
        future=torch.zeros((1,16,2),device=device);mask=torch.zeros((1,16),device=device)
        future[0,:len(actions)]=torch.tensor(actions,device=device);mask[0,:len(actions)]=1
        self._guard()
        return dict(query_embedding=q.detach().clone(),context_slot=slot.detach(),future_actions=future,
            action_mask=mask,horizon_index=torch.tensor([HORIZONS.index(len(actions))],device=device,dtype=torch.long))


class AFreshHead(nn.Module):
    profile=HEAD_PROFILE
    def __init__(self):
        super().__init__()
        self.slot=nn.Linear(128,64)
        self.horizon=nn.Embedding(len(HORIZONS),16)
        self.predictor=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,8))

    def forward(self,inputs):
        if set(inputs)!=INPUT_KEYS:raise ValueError('unexpected or private A head input fields')
        q,slot,actions,mask,index=(inputs[k] for k in ('query_embedding','context_slot','future_actions','action_mask','horizon_index'))
        b=len(q);device=next(self.parameters()).device
        if b<1:raise ValueError('empty A head batch')
        if any(p.dtype!=torch.float32 or p.device!=device for p in self.parameters()):raise ValueError('A head requires common float32 parameter profile')
        for value,shape in ((q,(b,128)),(slot,(b,128)),(actions,(b,16,2)),(mask,(b,16))):
            if value.shape!=shape or value.dtype!=torch.float32 or value.device!=device or not torch.isfinite(value).all():
                raise ValueError('A head feature shape/dtype/device/value mismatch')
        if index.shape!=(b,) or index.dtype!=torch.long or index.device!=device or torch.any(index<0) or torch.any(index>=len(HORIZONS)):
            raise ValueError('invalid A head horizon index')
        horizons=torch.tensor(HORIZONS,device=device)[index]
        expected=torch.arange(16,device=device)[None,:]<horizons[:,None]
        if not torch.equal(mask,expected.float()) or torch.any(actions[~expected]!=0) or torch.any(torch.linalg.vector_norm(actions,dim=-1)>1+1e-7):
            raise ValueError('A head action/mask support differs from registered horizon')
        with torch.autocast(device_type=device.type,enabled=False):
            x=torch.cat((q,self.slot(slot),actions.flatten(1),mask,self.horizon(index)),dim=-1)
            return self.predictor(x)

    def architecture(self):
        return dict(profile=self.profile,query_dim=128,context_slot_dim=128,adapted_slot_dim=64,horizon_embedding_dim=16,
            hidden_widths=[256,256],output_dim=8,horizons=list(HORIZONS),parameters=sum(p.numel() for p in self.parameters()),
            context_active_dimensions=dict(monolithic=128,split=64,Oracle=3,Null=0),
            capacity_statement='common nominal head; native information dimensions differ; not exact effective-capacity isolation')


def _stamp(module):
    return tuple((kind,name,id(v),v._version,str(v.dtype),str(v.device))
        for kind,items in (('parameter',module.named_parameters()),('buffer',module.named_buffers())) for name,v in items)


def evaluator_context(inputs,*,role,statistics=None,theta=None):
    """Evaluator-only head arm construction; role/known theta never enter agent packets.

    History contains whichever donor the caller encoded. Wrong-context training
    uses an independently fitted head and a separately sampled wrong donor; it
    is not implemented by modifying weights during a matched-head intervention.
    """
    if set(inputs)!=INPUT_KEYS:raise ValueError('invalid public head features')
    result={k:v.detach().clone() for k,v in inputs.items()}
    if role=='history':
        if theta is not None or statistics is not None:raise ValueError('known parameters cannot enter history arm')
    elif role=='null':
        if theta is not None or statistics is not None:raise ValueError('known parameters cannot enter Null arm')
        result['context_slot'].zero_()
    elif role=='oracle':
        if not isinstance(statistics,AHeadStatistics) or theta is None:raise ValueError('Oracle requires separately permitted known parameters and training statistics')
        slot=statistics.oracle_slot(theta)
        if slot.shape==(128,):slot=slot[None]
        if slot.shape!=tuple(result['context_slot'].shape):raise ValueError('Oracle system/query batch mismatch')
        result['context_slot']=torch.tensor(slot,device=result['context_slot'].device,dtype=torch.float32)
    else:raise ValueError('unregistered evaluator context arm')
    return result


class AFreshStateAgent:
    """Frozen-head public inference; privileged Oracle is a separate evaluator path."""
    def __init__(self,model,head,statistics,*,context_policy='history_or_zero',max_histories=1,aggregation='last'):
        if not isinstance(head,AFreshHead) or not isinstance(statistics,AHeadStatistics):raise ValueError('declared A head and target statistics required')
        if context_policy not in ('null_only','history_or_zero'):raise ValueError('unregistered A head missing-context policy')
        self.model=model;self.head=head;self.statistics=AHeadStatistics(statistics.record);self.context_policy=context_policy
        self.features=AFreshFeatures(model,max_histories=max_histories,aggregation=aggregation);self._ready=False

    def initialize(self,context):
        from persistbench.contracts import CapabilityDeclaration,OutputType
        self._ready=False;self.features.initialize(context);self.head.eval();self.statistics._check()
        device=next(self.model.parameters()).device
        if any(p.device!=device or p.dtype!=torch.float32 for p in self.head.parameters()):raise ValueError('head and backbone require common device/float32 profile')
        self._head_stamp=_stamp(self.head);self._statistics_sha=self.statistics.sha256;self._policy=self.context_policy
        self._ready=True
        return CapabilityDeclaration((OutputType.PREDICTION,),output_keys=('joint_state_delta',))

    def _guard(self):
        if not self._ready:raise RuntimeError('initialize A frozen state agent first')
        self.features._guard();self.statistics._check()
        if (any(m.training for m in self.head.modules()) or _stamp(self.head)!=self._head_stamp or
                self.statistics.record['statistics_sha256']!=self._statistics_sha or self.context_policy!=self._policy):
            raise ValueError('A head, normalization or context policy changed during fixed-artifact evaluation')

    def ingest(self,experience):
        self._guard()
        if self.context_policy=='null_only':raise ValueError('independently fitted Null does not ingest donor history')
        self.features.ingest(experience)

    def reset(self,context):self._guard();self.features.reset(context)
    def mutable_state_bytes(self):self._guard();return self.features.mutable_state_bytes()

    def respond(self,query):
        from persistbench.contracts import AgentOutput,OutputType
        self._guard();inputs=self.features.head_inputs(query)
        with torch.no_grad():normalized=self.head(inputs).cpu().numpy()[0]
        delta=self.statistics.physical(normalized,len(query.actions));self._guard()
        return AgentOutput(OutputType.PREDICTION,{'joint_state_delta':delta},
            dict(self.features.diagnostics(),head_profile=self.head.profile,head_parameters=self.head.architecture()['parameters'],
                 context_policy=self.context_policy,missing_context='zero slot',normalization_sha256=self._statistics_sha,
                 independent_fitted_Null_comparison=False,formal_head_provenance_verified=False))
