"""New-environment A adapter; the frozen PokeWorld implementation is unchanged.

All observation forwards are pointwise under a pre-step statistics snapshot.
Only train-history features update running statistics, after the forward/backward
work for the step. Future targets retain their ordinary encoder gradients.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from persistent_jepa.poke_model import PokeJEPA, PixelObservationEncoder, poke_objective


@dataclass
class VisualBatch:
    history_images: torch.Tensor
    history_actions: torch.Tensor
    target_images: torch.Tensor
    future_actions: torch.Tensor
    action_masks: torch.Tensor

    def validate(self, history_length):
        b=len(self.history_images)
        shapes={"history_images":(b,history_length,2,64,64),
                "history_actions":(b,history_length-1,2),
                "target_images":(b,3,2,64,64),"future_actions":(b,3,16,2),"action_masks":(b,3,16)}
        for name,shape in shapes.items():
            tensor=getattr(self,name)
            if tensor.shape!=shape or not torch.isfinite(tensor).all(): raise ValueError(name)
        if b<4 or b%2: raise ValueError("paired branch batches require an even size >=4")
        if torch.any(self.history_images[:,0,1]!=0): raise ValueError("strict history has no predecessor before frame zero")
        for hi,h in enumerate((1,4,16)):
            if not torch.all(self.action_masks[:,hi,:h]==1) or torch.any(self.action_masks[:,hi,h:]!=0):
                raise ValueError("future-action mask does not match its horizon")
            if torch.any(self.future_actions[:,hi,h:]!=0): raise ValueError("unobserved future-action padding must be zero")

    def to(self,device):
        return VisualBatch(**{k:v.to(device) for k,v in vars(self).items()})


class StrictPixelObservationEncoder(PixelObservationEncoder):
    def __init__(self):
        super().__init__()
        self._snapshot=None
        self._history_sum=None
        self._history_square_sum=None
        self._history_count=0

    def begin_train_step(self):
        if not self.training: raise RuntimeError("statistics collection requires training mode")
        if self._snapshot is not None: raise RuntimeError("previous step was not finished or discarded")
        self._snapshot=(self.norm.running_mean.detach().clone(),self.norm.running_var.detach().clone())
        self._history_sum=None;self._history_square_sum=None;self._history_count=0

    def forward(self,image,*,train_history=False):
        shape=image.shape[:-3]
        feature=self.project(self.cnn(image.reshape(-1,*image.shape[-3:])).flatten(1))
        if train_history:
            if not self.training or self._snapshot is None: raise RuntimeError("begin_train_step must precede train history")
            x=feature.detach().double()
            if not torch.isfinite(x).all():raise ValueError("nonfinite train history features")
            total=x.sum(0);square=x.square().sum(0)
            self._history_sum=total if self._history_sum is None else self._history_sum+total
            self._history_square_sum=square if self._history_square_sum is None else self._history_square_sum+square
            self._history_count+=len(x)
        # Detached clones prevent subsequent buffer updates from changing saved
        # backward tensors. No instantaneous context or target batch statistics.
        mean,var=self._snapshot if self._snapshot is not None else (self.norm.running_mean.detach().clone(),self.norm.running_var.detach().clone())
        result=F.batch_norm(feature,mean,var,self.norm.weight,self.norm.bias,training=False,momentum=0.,eps=self.norm.eps)
        return result.reshape(*shape,128)

    @torch.no_grad()
    def finish_train_step(self,*,split):
        if split!="train":raise ValueError("validation/test must not update observation statistics")
        if self._snapshot is None or self._history_count<2:raise RuntimeError("no complete training-history step")
        n=self._history_count;mean=self._history_sum/n
        variance=((self._history_square_sum-self._history_sum.square()/n)/(n-1)).clamp_min(0)
        self.norm.num_batches_tracked.add_(1)
        factor=self.norm.momentum if self.norm.momentum is not None else 1/float(self.norm.num_batches_tracked)
        self.norm.running_mean.lerp_(mean.to(self.norm.running_mean),factor)
        self.norm.running_var.lerp_(variance.to(self.norm.running_var),factor)
        self.discard_train_step()

    def discard_train_step(self):
        self._snapshot=None;self._history_sum=None;self._history_square_sum=None;self._history_count=0


class StrictVisualJEPA(PokeJEPA):
    normalization_profile="train_history_running_statistics_pre_step_v1"

    def __init__(self,variant="B3",history_length=24):
        if variant=="Sup":raise ValueError("parameter-supervised variant is outside this extension")
        super().__init__(variant,history_length)
        original=self.observation.state_dict()
        self.observation=StrictPixelObservationEncoder()
        self.observation.load_state_dict(original)

    def encode_batch(self,batch):
        batch.validate(self.cfg.history_length)
        history=self.observation(batch.history_images,train_history=self.training)
        targets=self.observation(batch.target_images)
        return history,targets

    def begin_train_step(self):self.observation.begin_train_step()
    def finish_train_step(self,*,split="train"):self.observation.finish_train_step(split=split)
    def discard_train_step(self):self.observation.discard_train_step()


def strict_objective(model,batch,sigreg,**kwargs):
    # Preserve the original self/SIGReg/VICReg and one-way cross implementation.
    # Pair construction and its complete observed supports are checked upstream.
    return poke_objective(model,batch,sigreg,**kwargs)
