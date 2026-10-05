"""Independent pixel baselines with one common observation and query interface.

Every history is encoded at ingest. Only an episode sum, maximum, and count
persist. The four variants differ in their within-episode temporal encoder.
Per-frame normalization never pools statistics across examples or future frames.
"""
from dataclasses import asdict,dataclass
import torch
from torch import nn
from torch.nn import functional as F

QUERY_PROFILES=('joint_state_delta_8d','cold_rest_joint_state_delta_8d','passive_prefix95_joint_state_delta_8d')


def query_profile_tensor(target_spec,batch_size,device):
    if target_spec not in QUERY_PROFILES:raise ValueError('unregistered public query profile')
    out=torch.zeros(batch_size,len(QUERY_PROFILES),device=device)
    out[:,QUERY_PROFILES.index(target_spec)]=1.
    return out


@dataclass(frozen=True)
class PixelModelConfig:
    family: str = 'gru'
    resolution: int = 128
    image_width: int = 128
    episode_width: int = 64
    temporal_layers: int = 2
    max_history_frames: int = 96
    max_query_frames: int = 97
    max_future_actions: int = 32
    max_histories: int = 8
    control_dt: float = .05

    def validate(self):
        if self.family not in ('gru','transformer','transition_deepsets','causal_tcn'):
            raise ValueError('unregistered pixel history family')
        if self.resolution not in (64,128) or self.episode_width%4:
            raise ValueError('unsupported observation/attention dimensions')
        if min(self.temporal_layers,self.max_histories,self.max_query_frames,self.max_future_actions)<1 or self.max_history_frames<2:
            raise ValueError('nonpositive observation/memory budget')


class ImageEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.resolution=cfg.resolution;layers=[];channels=2
        for out in (32,64,128,256):
            layers.extend((nn.Conv2d(channels,out,3,2,1),nn.GELU()));channels=out
        layers.extend((nn.Flatten(),nn.Linear(256*(cfg.resolution//16)**2,cfg.image_width),
                       nn.LayerNorm(cfg.image_width),nn.GELU()))
        self.net=nn.Sequential(*layers)

    def forward(self,images):
        if images.ndim!=5 or images.shape[2:]!=(2,self.resolution,self.resolution):
            raise ValueError('expected batch x frames x 2 x resolution x resolution')
        b,l=images.shape[:2]
        return self.net(images.reshape(b*l,*images.shape[2:])).reshape(b,l,-1)


class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.padding=2*dilation
        self.conv=nn.Conv1d(width,width,3,dilation=dilation)
        self.norm=nn.LayerNorm(width)

    def forward(self,x):
        update=self.conv(F.pad(x.transpose(1,2),(self.padding,0))).transpose(1,2)
        return self.norm(x+F.gelu(update))


class HistoryEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;d=cfg.episode_width;input_width=2*cfg.image_width+2
        self.single=nn.Linear(cfg.image_width,d)
        if cfg.family=='gru':
            self.temporal=nn.GRU(input_width,d,cfg.temporal_layers,batch_first=True)
        elif cfg.family=='transformer':
            self.input=nn.Linear(input_width,d)
            self.position=nn.Parameter(torch.zeros(1,cfg.max_history_frames,d))
            self.summary=nn.Parameter(torch.zeros(1,1,d))
            nn.init.normal_(self.position,std=.02);nn.init.normal_(self.summary,std=.02)
            layer=nn.TransformerEncoderLayer(d,4,4*d,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,cfg.temporal_layers,enable_nested_tensor=False)
        elif cfg.family=='transition_deepsets':
            self.temporal=nn.Sequential(nn.Linear(input_width,2*d),nn.GELU(),nn.Linear(2*d,d),nn.GELU())
            self.pool=nn.Linear(2*d,d)
        else:
            self.input=nn.Linear(input_width,d)
            self.temporal=nn.Sequential(*(CausalBlock(d,2**i) for i in range(cfg.temporal_layers)))
            self.pool=nn.Linear(2*d,d)
        self.output=nn.LayerNorm(d)

    def encode_transitions(self,tokens):
        if tokens.ndim!=3 or not 0<len(tokens[0])<self.cfg.max_history_frames:
            raise ValueError('observed transition support exceeds registered budget')
        family=self.cfg.family
        if family=='gru':
            _,last=self.temporal(tokens);result=last[-1]
        elif family=='transformer':
            sequence=torch.cat((self.input(tokens),self.summary.expand(len(tokens),-1,-1)),dim=1)
            result=self.temporal(sequence+self.position[:,:sequence.shape[1]])[:,-1]
        else:
            sequence=self.temporal(tokens if family=='transition_deepsets' else self.input(tokens))
            result=self.pool(torch.cat((sequence.mean(1),sequence.amax(1)),dim=-1))
        return self.output(result)

    def forward(self,embeddings,actions):
        if actions.shape!=(*embeddings.shape[:1],embeddings.shape[1]-1,2):
            raise ValueError('actions must exactly cover observed transitions')
        if embeddings.shape[1]==1:return self.output(self.single(embeddings[:,0]))
        tokens=torch.cat((embeddings[:,:-1],embeddings[:,1:]-embeddings[:,:-1],actions),dim=-1)
        return self.encode_transitions(tokens)


class PixelDynamicsModel(nn.Module):
    def __init__(self,cfg=PixelModelConfig()):
        super().__init__();cfg.validate();self.cfg=cfg
        self.image=ImageEncoder(cfg);self.history=HistoryEncoder(cfg)
        self.query_temporal=nn.GRU(cfg.image_width,cfg.episode_width,batch_first=True)
        self.future_temporal=nn.GRU(2,32,batch_first=True)
        width=3*cfg.episode_width+32+2+len(QUERY_PROFILES)
        self.head=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,8))
        self.register_buffer('target_mean',torch.zeros(cfg.max_future_actions+1,8))
        self.register_buffer('target_scale',torch.ones(cfg.max_future_actions+1,8))
        self.register_buffer('target_horizons',torch.ones(cfg.max_future_actions+1,dtype=torch.bool))

    def set_target_statistics(self,statistics):
        if statistics.get('source_split')!='train':raise ValueError('model normalization requires training-only statistics')
        means=torch.zeros_like(self.target_mean);scales=torch.ones_like(self.target_scale)
        horizons=torch.zeros_like(self.target_horizons)
        if set(statistics['mean'])!=set(statistics['scale']):raise ValueError('inconsistent target statistics')
        for key in statistics['scale']:
            h=int(key)
            if not 1<=h<=self.cfg.max_future_actions:raise ValueError('normalization horizon exceeds model support')
            mean=torch.as_tensor(statistics['mean'][key],device=means.device,dtype=means.dtype)
            scale=torch.as_tensor(statistics['scale'][key],device=scales.device,dtype=scales.dtype)
            if mean.shape!=(8,) or scale.shape!=(8,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('invalid physical normalization')
            means[h]=mean;scales[h]=scale;horizons[h]=True
        if not horizons.any():raise ValueError('empty normalized target support')
        self.target_mean.copy_(means);self.target_scale.copy_(scales);self.target_horizons.copy_(horizons)

    def artifact_config(self):return asdict(self.cfg)

    def encode_history(self,images,actions):
        if images.shape[1]>self.cfg.max_history_frames:raise ValueError('history frame budget exceeded')
        return self.history(self.image(images),actions)

    def aggregate(self,episode_codes,batch_size=None,device=None):
        if not episode_codes:
            return torch.zeros(batch_size,2*self.cfg.episode_width,device=device),torch.zeros(batch_size,1,device=device)
        if len(episode_codes)>self.cfg.max_histories:raise ValueError('episode budget exceeded')
        codes=torch.stack(episode_codes,dim=1)
        return torch.cat((codes.mean(1),codes.amax(1)),dim=-1),torch.full((len(codes),1),len(episode_codes),device=codes.device,dtype=codes.dtype)

    def predict(self,memory,count,query_images,future_actions,query_profile=None):
        if not 1<=query_images.shape[1]<=self.cfg.max_query_frames:raise ValueError('query frame budget exceeded')
        if future_actions.ndim!=3 or future_actions.shape[0]!=len(query_images) or future_actions.shape[2]!=2 or not 1<=future_actions.shape[1]<=self.cfg.max_future_actions:
            raise ValueError('future action interval budget exceeded')
        if memory.shape!=(len(query_images),2*self.cfg.episode_width) or count.shape!=(len(query_images),1):
            raise ValueError('persistent state dimensions')
        _,query=self.query_temporal(self.image(query_images));_,future=self.future_temporal(future_actions)
        duration=torch.full_like(count,future_actions.shape[1]*self.cfg.control_dt)
        if query_profile is None:query_profile=query_profile_tensor(QUERY_PROFILES[0],len(count),count.device)
        if query_profile.shape!=(len(count),len(QUERY_PROFILES)) or not torch.all((query_profile==0)|(query_profile==1)) or not torch.all(query_profile.sum(1)==1):
            raise ValueError('query profile must be a registered public one-hot declaration')
        horizon=future_actions.shape[1]
        if not self.target_horizons[horizon]:raise ValueError('prediction horizon has no registered training normalization')
        normalized=self.head(torch.cat((memory,query[-1],future[-1],duration,torch.log1p(count),query_profile),dim=-1))
        return normalized*self.target_scale[horizon]+self.target_mean[horizon]

    def forward(self,histories,query_images,future_actions,context_mask=None,query_profile=None):
        codes=[self.encode_history(images,actions) for images,actions in histories]
        memory,count=self.aggregate(codes,len(query_images),query_images.device)
        if context_mask is not None:
            if context_mask.shape!=(len(query_images),1):raise ValueError('context mask must select complete case histories')
            memory=memory*context_mask;count=count*context_mask
        return self.predict(memory,count,query_images,future_actions,query_profile)
