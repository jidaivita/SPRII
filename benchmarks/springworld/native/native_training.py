"""Versioned independent96-window training bridge, shared across objective arms."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import torch
from dataclasses import replace
from persistbench.envs.visual_elastic_coupling.a_pairing import APairSchedule, digest, configuration
from persistbench.envs.visual_elastic_coupling.schema import Config
from persistbench.envs.visual_elastic_coupling.observations import image_history
from persistbench.envs.visual_elastic_coupling.adapters import z_experience
from native128_model import NativeVisualBatch, Native128JEPA
from strict_model import strict_objective


class NativePairSchedule(APairSchedule):
    def __init__(self, manifest, *, seed, history_frames=96, pairs_per_batch=48):
        if history_frames != 96:raise ValueError('native profile requires96')
        super().__init__(manifest,seed=seed,history_frames=24,pairs_per_batch=pairs_per_batch)
        self.length=96
        end=self.requested_frames-self.length-16
        if end<0:raise ValueError('independent96 history and future16 do not fit')
        self.windows=[(d,r) for d in range(end+1) for r in range(end+1)]
        self._settings=digest([self.seed,self.kind,self.length,self.batch_pairs,self.keys,self.systems,
                              self.episodes,self.episode_count,self.requested_frames,self.windows])

    def sweep(self,sweep,name):
        if configuration(name)['pairing_profile']!='Independent':
            raise ValueError('SameEp96 requires separately registered longer-episode matched control')
        return super().sweep(sweep,name)

    def inventory(self):
        row=super().inventory()
        row.update(schema='springworld.A-native128-independent96.v1',resolution=128,
                   template_policy='uniform Cartesian product of per-episode legal96 histories with h16 support',
                   strict_A_compatible=False,training_profile='native128-history96',
                   same_episode_admitted=False)
        return row


@lru_cache(maxsize=None)
def visible(root,relative,expected):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('asset escapes bank')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('public asset hash differs')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'images','actions','timestamps'}:raise ValueError('nonpublic image fields')
        images,actions,times=(z[k].copy() for k in ('images','actions','timestamps'))
    n=len(images)
    if images.dtype!=np.uint8 or images.shape!=(n,128,128) or actions.shape!=(n-1,2):raise ValueError('native public shape')
    if not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-7):raise ValueError('action values')
    if times.shape!=(n,) or not np.allclose(times,np.arange(n)*.05,rtol=0,atol=1e-8):raise ValueError('time support')
    images.flags.writeable=False;actions.flags.writeable=False
    return images,actions


@lru_cache(maxsize=None)
def train_state(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('label path escapes bank')
    # Caller only admits train forced rows. The engine verifies all source hashes
    # against the frozen snapshot both before and after execution.
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'state'}:raise ValueError('unexpected state labels')
        state=z['state'].copy()
    if state.ndim!=2 or state.shape[1]!=8 or not np.isfinite(state).all():raise ValueError('invalid state labels')
    state.flags.writeable=False
    return state


_STATS={}
def target_statistics(schedule,root):
    key=(str(root),schedule.selected_manifest_sha256)
    if key in _STATS:return _STATS[key]
    sums=np.zeros((3,8));squares=np.zeros((3,8));count=0
    for row in schedule.rows.values():
        if row['split']!='train' or row['kind']!='forced':raise PermissionError('supervised statistics require train forced only')
        s=train_state(str(root),row['private_state_path'])
        if len(s)!=row['requested_frames']:raise ValueError('failed trajectory retained; no successful-only statistics')
        anchors=np.arange(95,len(s)-16)
        y=np.stack([s[anchors+h]-s[anchors] for h in (1,4,16)],axis=1)
        sums+=y.sum(0);squares+=(y*y).sum(0);count+=len(y)
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,1e-12))
    out=dict(mean=mean.tolist(),std=std.tolist(),window_count=count,
             policy='every legal train-forced96 history anchor, three horizons, no validation labels')
    out['sha256']=digest(out);_STATS[key]=out
    return out


def make_batch(schedule,plan,batch_index,bank_root):
    schedule.validate(plan);size=schedule.batch_pairs
    if type(batch_index)is not int or not 0<=batch_index<len(plan['pairs'])//size:raise ValueError('pair batch index')
    selected=plan['pairs'][batch_index*size:(batch_index+1)*size]
    supervised=plan['configuration']['name']=='Supervised-Split'
    stats=target_statistics(schedule,bank_root) if supervised else None
    windows=[];labels=[];assets={};raw_read=0
    for branch in ('donor','recipient'):
        for pair in selected:
            row=schedule.rows[pair[branch+'_episode']]
            if row['split']!='train':raise PermissionError('nontrain pretraining row')
            asset=row['assets']['128'];images,actions=visible(str(bank_root),asset['path'],asset['sha256'])
            start=pair[branch+'_start'];anchor=start+95
            if anchor+16>=len(images):raise ValueError('missing planned support; no replacement')
            x=images[start:anchor+1].astype(np.float32)/255
            diff=np.zeros_like(x);diff[1:]=x[1:]-x[:-1]
            history=np.stack((x,diff),axis=1);past=actions[start:anchor].astype(np.float32)
            payload=dict(observations=history,past_actions=np.r_[np.zeros((1,2)),past],
                         past_action_mask=np.r_[False,np.ones(95,bool)],relative_times=np.arange(96)*.05)
            image_history(z_experience(payload),Config(resolution=128))
            targets=[];future=np.zeros((3,16,2),np.float32);masks=np.zeros((3,16),np.float32)
            for i,h in enumerate((1,4,16)):
                current=images[anchor+h].astype(np.float32)/255
                previous=images[anchor+h-1].astype(np.float32)/255
                targets.append(np.stack((current,current-previous)))
                future[i,:h]=actions[anchor:anchor+h];masks[i,:h]=1
            windows.append((history,past,np.stack(targets),future,masks))
            assets[row['episode_key']]=asset['sha256'];raw_read+=len(images)
            if supervised:
                state=train_state(str(bank_root),row['private_state_path'])
                labels.append(np.stack([state[anchor+h]-state[anchor] for h in (1,4,16)]))
    batch=NativeVisualBatch(*[torch.from_numpy(np.stack([w[i] for w in windows])) for i in range(5)])
    if supervised:
        batch.state_targets=torch.tensor((np.stack(labels)-np.asarray(stats['mean']))/np.asarray(stats['std']),dtype=torch.float32)
    batch.validate(96)
    receipt=dict(plan_sha256=plan['plan_sha256'],batch_index=batch_index,input_assets=assets,
                 pair_sha256=[p['pair_sha256'] for p in selected],windows=len(windows),
                 raw_image_frames_read=raw_read,presented_history_frames=len(windows)*96,
                 observed_action_intervals=len(windows)*95,target_frames=len(windows)*3,
                 target_horizons=[1,4,16],direction='donor_to_recipient',resolution=128,
                 physical_labels_read=supervised,target_statistics=stats if supervised else None,test_read=False)
    return batch,receipt


def new_model(name,spec,device):
    from persistent_jepa.model import Predictor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.model_seed)
        model=Native128JEPA(configuration(name)['variant'],projection_seed=spec.model_seed)
        if name=='Supervised-Split':
            model.predictor=Predictor(replace(model.cfg,observation_dim=8),context_dim=128)
    model.training_objective='state_mse' if name=='Supervised-Split' else 'jepa'
    return model.to(device=device,dtype=torch.float32)


def objective(model,batch,sigreg,config):
    if config!=configuration(config['name']):raise ValueError('objective configuration differs')
    if config['name']!='Supervised-Split':
        if getattr(sigreg,'num_directions',None)!=1024:raise ValueError('full-batch SIGReg1024 required')
        return strict_objective(model,batch,sigreg,sigreg_weight=.02,lambda_p=config['lambda_p'],lambda_x=config['lambda_x'])
    batch.validate(96)
    # The supervised arm never encodes target images; labels are used only here.
    if not hasattr(batch,'state_targets') or batch.state_targets.shape!=(len(batch.history_images),3,8) or not torch.isfinite(batch.state_targets).all():
        raise ValueError('supervised state target shape or values')
    h=model.observation(batch.history_images,train_history=model.training)
    context=model.codes(h,batch.history_actions)[2];terms=[]
    for i in range(3):
        index=torch.full((len(context),),i,device=context.device,dtype=torch.long)
        y=model.predictor(context,batch.future_actions[:,i],batch.action_masks[:,i],index)
        terms.append(torch.nn.functional.mse_loss(y,batch.state_targets[:,i]))
    loss=torch.stack(terms).mean()
    return loss,dict(loss_total=loss.detach(),loss_state_mse=loss.detach())
