"""Object Gaussian RSSM conditioned on persistent AB history.

The current prefix is filtered with zero history context. Its recurrent state
therefore depends on CD[:3] only. Future forecasting uses prior transitions only;
future posterior states are confined to the training reconstruction/KL branch.
This is a CoPhy-adapted RSSM, not a complete Dreamer agent.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from components import FrameProjection, Interaction, time_mask, relation_loss, replace_focal_p

VERSION = 'cophy-gaussian-rssm-v6.3'


@dataclass(frozen=True)
class ModelConfig:
    family: str = 'RSSM'
    feature_dim: int = 784
    width: int = 128
    persistent_dim: int = 64
    stochastic_dim: int = 32
    hidden_layers: int = 2
    query_frames: int = 3
    min_std: float = .1
    reconstruction_weight: float = 1.
    rollout_weight: float = 1.
    kl_weight: float = .01
    free_nats: float = 1.
    kl_balance: float = .8
    lambda_cross: float = 1.
    lambda_align: float = .1
    normalization_train_episodes: int = 256
    normalization_scale_floor: float = .001

    def __post_init__(self):
        if self.family != 'RSSM' or (self.feature_dim,self.width,self.persistent_dim,self.stochastic_dim)!=(784,128,64,32):
            raise ValueError('Fixed Gaussian RSSM: F784/h128/P64/z32')
        if self.query_frames != 3: raise ValueError('Only three current frames are visible')


def gaussian_kl(q,p):
    qm,qs=q;pm,ps=p
    return (torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).sum(-1)


def draw(distribution,sample):
    mean,std=distribution
    return mean+std*torch.randn_like(mean) if sample else mean


class GaussianRSSM(nn.Module):
    def __init__(self,config=ModelConfig()):
        super().__init__();self.config=config
        self.project=FrameProjection(config)
        self.history_interaction=Interaction(128)
        self.history=nn.GRU(128,128,config.hidden_layers,batch_first=True)
        self.history_output=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64))
        self.observation_interaction=Interaction(128)
        self.dynamics_interaction=Interaction(128)
        self.transition_input=nn.Sequential(nn.Linear(128+32+64,128),nn.GELU())
        self.recurrent=nn.GRUCell(128,128)
        self.prior=nn.Sequential(nn.Linear(128+64,128),nn.GELU(),nn.Linear(128,64))
        self.posterior=nn.Sequential(nn.Linear(128+128,128),nn.GELU(),nn.Linear(128,64))
        self.decoder=nn.Sequential(nn.Linear(128+32+64,256),nn.GELU(),nn.Linear(256,784))
        self.register_buffer('feature_mean',torch.zeros(784))
        self.register_buffer('feature_scale',torch.ones(784))

    def encode(self,features_ab,mask):
        mask=time_mask(mask,features_ab)
        seq=self.history_interaction(self.project(features_ab),mask)
        b,t,k,_=seq.shape
        out,_=self.history(seq.permute(0,2,1,3).reshape(b*k,t,128))
        present=mask.permute(0,2,1).reshape(b*k,t)
        last=(present*torch.arange(1,t+1,device=seq.device)).amax(1)-1
        result=out[torch.arange(b*k,device=seq.device),last.clamp_min(0)].reshape(b,k,128)
        return self.history_output(result)*(last>=0).reshape(b,k,1)

    def distribution(self,values):
        mean,raw_std=values.float().chunk(2,-1)
        return mean,F.softplus(raw_std)+self.config.min_std

    def initial(self,b,k,device):
        return {'h':torch.zeros(b,k,128,device=device),'z':torch.zeros(b,k,32,device=device)}

    def transition(self,state,p,active):
        b,k,_=p.shape
        context=self.dynamics_interaction(state['h'],active)
        inputs=self.transition_input(torch.cat((context,state['z'],p),-1))
        h=self.recurrent(inputs.reshape(b*k,128),state['h'].reshape(b*k,128)).reshape(b,k,128)
        h=h*active[...,None]
        distribution=self.distribution(self.prior(torch.cat((h,p),-1)))
        return h,distribution

    def filter_prefix(self,query,mask,sample=False):
        if query.shape[1]!=3:raise ValueError('Prefix filtering must consume CD[:3] only')
        mask=time_mask(mask,query);active=mask.any(1)
        b,t,k,_=query.shape;state=self.initial(b,k,query.device)
        observations=self.observation_interaction(self.project(query),mask)
        zero_p=query.new_zeros(b,k,64)
        for step in range(t):
            h,prior=self.transition(state,zero_p,active)
            post=self.distribution(self.posterior(torch.cat((h,observations[:,step]),-1)))
            z=torch.where(mask[:,step,:,None],draw(post,sample),draw(prior,sample))
            state={'h':h,'z':z*active[...,None]}
        return state,active

    def imagine(self,p,prefix,active,steps,sample=False):
        state={k:v for k,v in prefix.items()};outputs=[]
        for _ in range(steps):
            h,prior=self.transition(state,p,active)
            z=draw(prior,sample)*active[...,None]
            state={'h':h,'z':z}
            outputs.append(self.decoder(torch.cat((h,z,p),-1)))
        return torch.stack(outputs,1)

    def predict(self,p,query,mask,steps,sample=False):
        prefix,active=self.filter_prefix(query,mask,sample=sample)
        return self.imagine(p,prefix,active,steps,sample=sample)

    def target(self,full_cd):
        # The target is fixed RGB-derived feature784, not a learned latent or GT pose.
        return ((full_cd.float()-self.feature_mean)/self.feature_scale)[:,3:].detach()

    def observe_future(self,p,prefix,active,full_cd,mask_cd,sample=True):
        # Train-only teacher-forced filtering; NEVER called by predict/imagine.
        mask_cd=time_mask(mask_cd,full_cd)
        observations=self.observation_interaction(self.project(full_cd),mask_cd)[:,3:]
        state={k:v for k,v in prefix.items()};reconstructed=[];kls=[];raw=[]
        for step in range(observations.shape[1]):
            h,prior=self.transition(state,p,active)
            post=self.distribution(self.posterior(torch.cat((h,observations[:,step]),-1)))
            available=mask_cd[:,step+3] & active
            z=torch.where(available[...,None],draw(post,sample),draw(prior,sample))*active[...,None]
            state={'h':h,'z':z}
            reconstructed.append(self.decoder(torch.cat((h,z,p),-1)))
            raw_kl=gaussian_kl(post,prior)
            dynamics_kl=gaussian_kl(tuple(v.detach() for v in post),prior)
            representation_kl=gaussian_kl(post,tuple(v.detach() for v in prior))
            balance=self.config.kl_balance
            kl=balance*dynamics_kl.clamp_min(self.config.free_nats)+(1-balance)*representation_kl.clamp_min(self.config.free_nats)
            kls.append(kl);raw.append(raw_kl)
        return torch.stack(reconstructed,1),torch.stack(kls,1),torch.stack(raw,1)

    def history_parameters(self):
        return [p for module in (self.history_interaction,self.history,self.history_output) for p in module.parameters()]

    def set_normalization(self,mean,scale):
        mean=torch.as_tensor(mean,dtype=torch.float32,device=self.feature_mean.device)
        scale=torch.as_tensor(scale,dtype=torch.float32,device=self.feature_mean.device)
        if mean.shape!=(784,) or scale.shape!=(784,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Invalid train-only fixed feature normalization')
        self.feature_mean.copy_(mean);self.feature_scale.copy_(scale)

    def artifact_config(self):return {'version':VERSION,**asdict(self.config)}


def make_model(family='RSSM',config=None):
    if family!='RSSM':raise ValueError('This independent prototype is RSSM only')
    if isinstance(config,dict):config=ModelConfig(**{k:v for k,v in config.items() if k!='version'})
    return GaussianRSSM(config or ModelConfig())


def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version')!=VERSION:raise ValueError('Not a Gaussian RSSM v6.3 checkpoint')
    model=make_model(config=ck['model_config']);model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck
