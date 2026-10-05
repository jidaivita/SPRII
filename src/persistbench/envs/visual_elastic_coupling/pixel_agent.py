"""Ingest-time compression for the independent pixel reference families."""
import numpy as np
import torch
from .schema import Config
from .observations import image_history,public_query_images
from .pixel_models import QUERY_PROFILES,query_profile_tensor




class PixelMemoryAgent:
    def __init__(self,model):
        self.model=model;self.config=Config(resolution=model.cfg.resolution,control_dt=model.cfg.control_dt)
        self._clear()

    def _clear(self):self._sum=None;self._max=None;self.count=0;self.episode_token=None

    def initialize(self,context):
        from persistbench.contracts import CapabilityDeclaration,OutputType
        self._clear();self.model.eval()
        return CapabilityDeclaration((OutputType.PREDICTION,OutputType.REPRESENTATION),
            representation_dim=2*self.model.cfg.episode_width,output_keys=('joint_state_delta','representation'))

    def reset(self,context):self.episode_token=context.episode_token

    def ingest(self,experience):
        _,actions=image_history(experience,self.config)
        if self.count>=self.model.cfg.max_histories:raise ValueError('registered episode budget exceeded')
        device=next(self.model.parameters()).device
        with torch.no_grad():
            code=self.model.encode_history(torch.as_tensor(experience.observations,device=device,dtype=torch.float32)[None],
                torch.as_tensor(actions,device=device,dtype=torch.float32)[None]).detach().clone()
        self._sum=code if self._sum is None else self._sum+code
        self._max=code.clone() if self._max is None else torch.maximum(self._max,code)
        self.count+=1

    def memory(self):
        if not self.count:
            device=next(self.model.parameters()).device
            return torch.zeros(1,2*self.model.cfg.episode_width,device=device),torch.zeros(1,1,device=device)
        return torch.cat((self._sum/self.count,self._max),dim=-1),torch.tensor([[self.count]],device=self._sum.device,dtype=self._sum.dtype)

    def mutable_state_bytes(self):
        # Serialized numeric payload, excluding framework/container overhead.
        return 8 if not self.count else 8+self._sum.numel()*self._sum.element_size()+self._max.numel()*self._max.element_size()

    def respond(self,query):
        from persistbench.contracts import AgentOutput,OutputType
        query.validate()
        payload=query.observations
        if isinstance(payload,dict) and payload.get('target_spec')=='persistent_representation_128d':
            allowed={'observations','relative_times','horizon_seconds','target_spec'}
            if set(payload)-allowed:raise ValueError('non-public representation query')
            if self.model.cfg.episode_width!=64:raise ValueError('representation version does not match artifact')
            memory,_=self.memory()
            return AgentOutput(OutputType.REPRESENTATION,{'representation':memory.detach().cpu().numpy()[0].copy()})
        images,actions=public_query_images(query,self.config,allowed_targets=QUERY_PROFILES)
        memory,count=self.memory();device=memory.device
        profile=query_profile_tensor(payload['target_spec'],1,device)
        with torch.no_grad():
            output=self.model.predict(memory,count,torch.as_tensor(images,device=device)[None],torch.as_tensor(actions,device=device)[None],profile)
        return AgentOutput(OutputType.PREDICTION,{'joint_state_delta':output.detach().cpu().numpy()[0].copy()},
            diagnostics=dict(persistent_numeric_payload_bytes=self.mutable_state_bytes(),processed_histories=self.count,
                             raw_history_retained=False))
