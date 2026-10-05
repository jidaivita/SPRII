"""Native and structured object Gaussian RSSM over frozen RGB features.

Native filters AB and then a boundary plus the current prefix through one h/z
state machine, with no P tower. Structured models retain the prior P64 tower
and query-only filtering. Future posterior states are training-only; full state
exports consume AB and the visible query prefix and use deterministic means.
This is a CoPhy RSSM adaptation, not a complete Dreamer agent.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
def time_mask(mask, features):
    if mask.ndim == 2:
        mask = mask[:, None].expand(features.shape[:3])
    if mask.shape != features.shape[:3]:
        raise ValueError('Expected object presence [B,T,O] or [B,O]')
    return mask.bool()


class FrameProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * cfg.feature_dim),
                                 nn.Linear(2 * cfg.feature_dim, 256), nn.GELU(),
                                 nn.Linear(256, cfg.width), nn.LayerNorm(cfg.width))

    def forward(self, features, zero_delta=False):
        if features.ndim != 4 or features.shape[-1] != 784:
            raise ValueError('Motion projection requires [B,T,O,784] from one video')
        current = features.float()
        delta = torch.zeros_like(current)
        if not zero_delta:
            delta[:, 1:] = current[:, 1:] - current[:, :-1]
        return self.net(torch.cat((current, delta), -1))


class Interaction(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                    nn.Linear(width, width), nn.LayerNorm(width))

    def forward(self, tokens, mask):
        k = tokens.shape[-2]
        receiver = tokens[..., :, None, :].expand(*tokens.shape[:-2], k, k, tokens.shape[-1])
        sender = tokens[..., None, :, :].expand_as(receiver)
        edges = self.edge(torch.cat((receiver, sender), -1))
        pair = mask[..., :, None] & mask[..., None, :]
        pair = pair & ~torch.eye(k, device=tokens.device, dtype=torch.bool)
        messages = (edges * pair[..., None]).sum(-2) / pair.sum(-1).clamp_min(1)[..., None]
        return (tokens + self.update(torch.cat((tokens, messages), -1))) * mask[..., None]




VERSION = 'cophy-native-structured-rssm-v7.0'
LEGACY_VERSION = 'cophy-gaussian-rssm-v6.3'


@dataclass(frozen=True)
class ModelConfig:
    family: str = 'RSSM'
    architecture: str = 'structured'
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
        if self.architecture not in ('native','structured'): raise ValueError('Unknown RSSM architecture')
        if self.query_frames != 3: raise ValueError('Only three current frames are visible')


def gaussian_kl(q,p):
    qm,qs=q;pm,ps=p
    return (torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).sum(-1)


def draw(distribution,sample):
    mean,std=distribution
    return mean+std*torch.randn_like(mean) if sample else mean


class GaussianRSSM(nn.Module):
    context_dim = 224
    representation_dim = 224
    export_kind = "rssm"
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

    def encode_history_state(self, ab, am):
        am = time_mask(am, ab)
        return {'p': self.encode(ab, am), 'present': am.any(1)}

    def encode_current_tokens(self, c, cm):
        if c.shape[1] != self.config.query_frames:
            raise ValueError('RSSM export requires the bound visible query prefix')
        return self.observation_interaction(self.project(c), time_mask(cm, c))

    def filter_tokens(self, tokens, mask, state=None, sample=False):
        mask = mask.bool(); active = mask.any(1)
        b, t, k, _ = tokens.shape
        state = self.initial(b, k, tokens.device) if state is None else state
        zero_p = tokens.new_zeros(b, k, 64)
        for step in range(t):
            h, prior = self.transition(state, zero_p, active)
            post = self.distribution(self.posterior(torch.cat((h, tokens[:, step]), -1)))
            z = torch.where(mask[:, step, :, None], draw(post, sample), draw(prior, sample))
            state = {'h': h, 'z': z * active[..., None]}
        return state, active

    def encode_joint_from_cached(self, state, tokens, cm):
        prefix, active = self.filter_tokens(tokens, cm, sample=False)
        p = state['p'] * state['present'][..., None]
        return torch.cat((p, prefix['h'], prefix['z']), -1) * active[..., None]

    def encode_joint_from_state(self, state, c, cm):
        return self.encode_joint_from_cached(state, self.encode_current_tokens(c, cm), cm)

    def encode_joint(self, ab, am, c, cm):
        return self.encode_joint_from_state(self.encode_history_state(ab, am), c, cm)

    def history_parameters(self):
        return [p for module in (self.history_interaction,self.history,self.history_output) for p in module.parameters()]

    def set_normalization(self,mean,scale):
        mean=torch.as_tensor(mean,dtype=torch.float32,device=self.feature_mean.device)
        scale=torch.as_tensor(scale,dtype=torch.float32,device=self.feature_mean.device)
        if mean.shape!=(784,) or scale.shape!=(784,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Invalid train-only fixed feature normalization')
        self.feature_mean.copy_(mean);self.feature_scale.copy_(scale)

    def artifact_config(self):return {'version':VERSION,**asdict(self.config)}


class NativeRSSM(nn.Module):
    """A single RSSM filters AB, a boundary token, then the visible CD prefix.

    No independent history GRU, persistent projection or P-conditioned transition
    exists. A boundary is one learned observation token, not a fabricated RGB
    difference. The same stochastic state machinery processes both clips.
    """
    context_dim = 160
    representation_dim = 160
    export_kind = 'rssm'

    def __init__(self, config):
        super().__init__(); self.config = config
        self.project = FrameProjection(config)
        self.observation_interaction = Interaction(128)
        self.dynamics_interaction = Interaction(128)
        self.transition_input = nn.Sequential(nn.Linear(128 + 32, 128), nn.GELU())
        self.recurrent = nn.GRUCell(128, 128)
        self.prior = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))
        self.posterior = nn.Sequential(nn.Linear(128 + 128, 128), nn.GELU(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(128 + 32, 256), nn.GELU(), nn.Linear(256, 784))
        self.boundary = nn.Parameter(torch.zeros(128))
        self.register_buffer('feature_mean', torch.zeros(784))
        self.register_buffer('feature_scale', torch.ones(784))

    distribution = GaussianRSSM.distribution
    initial = GaussianRSSM.initial
    target = GaussianRSSM.target
    set_normalization = GaussianRSSM.set_normalization
    artifact_config = GaussianRSSM.artifact_config
    encode_current_tokens = GaussianRSSM.encode_current_tokens

    def transition(self, state, active):
        b, k = active.shape
        context = self.dynamics_interaction(state['h'], active)
        inputs = self.transition_input(torch.cat((context, state['z']), -1))
        h = self.recurrent(inputs.reshape(b*k, 128), state['h'].reshape(b*k, 128)).reshape(b,k,128)
        h = h * active[..., None]
        return h, self.distribution(self.prior(h))

    def filter_tokens(self, tokens, mask, state=None, sample=False):
        mask = mask.bool(); active = mask.any(1)
        b, t, k, _ = tokens.shape
        state = self.initial(b, k, tokens.device) if state is None else state
        for step in range(t):
            h, prior = self.transition(state, active)
            post = self.distribution(self.posterior(torch.cat((h, tokens[:, step]), -1)))
            z = torch.where(mask[:, step, :, None], draw(post, sample), draw(prior, sample))
            state = {'h': h, 'z': z * active[..., None]}
        return state, active

    def history_state(self, ab, am, sample=False):
        am = time_mask(am, ab)
        tokens = self.observation_interaction(self.project(ab), am)
        state, active = self.filter_tokens(tokens, am, sample=sample)
        return dict(state, present=active)

    def encode_history_state(self, ab, am):
        return self.history_state(ab, am, sample=False)

    def continue_query(self, state, tokens, cm, sample=False):
        active = cm.bool().any(1)
        # A new episode can contain objects absent in the history. They start
        # from zero; current tokens and presence remain available to every arm.
        present = state['present'].bool()
        recurrent = {k: state[k] * present[..., None] for k in ('h', 'z')}
        boundary_mask = (present | active)[:, None]
        boundary = self.boundary[None, None, None].expand(tokens.shape[0],1,tokens.shape[2],-1)
        recurrent, _ = self.filter_tokens(boundary, boundary_mask, recurrent, sample)
        return self.filter_tokens(tokens, cm, recurrent, sample)

    def encode_joint_from_cached(self, state, tokens, cm):
        result, active = self.continue_query(state, tokens, cm, sample=False)
        return torch.cat((result['h'], result['z']), -1) * active[..., None]

    encode_joint_from_state = GaussianRSSM.encode_joint_from_state
    encode_joint = GaussianRSSM.encode_joint

    def source_prefix(self, ab, am, c, cm, sample=False):
        state = self.history_state(ab, am, sample)
        return self.continue_query(state, self.encode_current_tokens(c,cm), cm, sample)

    def imagine(self, prefix, active, steps, sample=False):
        state = {k:v for k,v in prefix.items()}; outputs=[]
        for _ in range(steps):
            h, prior = self.transition(state, active)
            z = draw(prior, sample) * active[..., None]
            state = {'h':h, 'z':z}
            outputs.append(self.decoder(torch.cat((h,z), -1)))
        return torch.stack(outputs,1)

    def predict_observed(self, ab, am, c, cm, steps, sample=False):
        prefix, active = self.source_prefix(ab,am,c,cm,sample)
        return self.imagine(prefix,active,steps,sample)

    def observe_future(self, prefix, active, full_cd, mask_cd, sample=True):
        # Only the training objective calls this function. Export/prediction do
        # not accept full_cd and cannot obtain a future-conditioned posterior.
        mask_cd = time_mask(mask_cd, full_cd)
        tokens = self.observation_interaction(self.project(full_cd), mask_cd)[:,3:]
        state = {k:v for k,v in prefix.items()}; outputs=[]; kls=[]; raws=[]
        for step in range(tokens.shape[1]):
            h, prior = self.transition(state, active)
            post = self.distribution(self.posterior(torch.cat((h,tokens[:,step]),-1)))
            available = mask_cd[:,step+3] & active
            z = torch.where(available[...,None],draw(post,sample),draw(prior,sample))*active[...,None]
            state = {'h':h,'z':z}
            outputs.append(self.decoder(torch.cat((h,z),-1)))
            raw = gaussian_kl(post,prior)
            dyn = gaussian_kl(tuple(v.detach() for v in post),prior).clamp_min(self.config.free_nats)
            rep = gaussian_kl(post,tuple(v.detach() for v in prior)).clamp_min(self.config.free_nats)
            kls.append(self.config.kl_balance*dyn+(1-self.config.kl_balance)*rep);raws.append(raw)
        return torch.stack(outputs,1),torch.stack(kls,1),torch.stack(raws,1)


def make_model(family='RSSM', config=None, method=None):
    if family != 'RSSM': raise ValueError('RSSM only')
    if isinstance(config,dict):
        config = ModelConfig(**{k:v for k,v in config.items() if k != 'version'})
    if config is None:
        config = ModelConfig(architecture='native' if method == 'Native' else 'structured')
    return NativeRSSM(config) if config.architecture == 'native' else GaussianRSSM(config)


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version') not in (VERSION,LEGACY_VERSION): raise ValueError('Unknown RSSM checkpoint version')
    if ck.get('test_read') is not False: raise ValueError('Only bound train/validation sources are supported')
    cfg = dict(ck['model_config'])
    if ck.get('version') == LEGACY_VERSION: cfg['architecture'] = 'structured'
    model = make_model(config=cfg);model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck
