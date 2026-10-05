"""Continuous, resumable monolithic JEPA source training: fixed 50/100 snapshots.

This module reads only frozen RGB features, visual presence, and episode IDs.
No RelationIndex, confounder metadata, donor, pose, or coordinate labels enter.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recompute

# Avoid collision with the original split-model modules in a shared process.
_MODEL_ALIAS = '_cophy_monolithic_jepa_v66_models'
_model_path = Path(__file__).with_name('models.py').resolve()
if _MODEL_ALIAS in sys.modules:
    model_module = sys.modules[_MODEL_ALIAS]
    if Path(model_module.__file__).resolve() != _model_path:
        raise ValueError('Different monolithic model implementation is already loaded')
else:
    spec = importlib.util.spec_from_file_location(_MODEL_ALIAS, _model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[_MODEL_ALIAS] = model_module
    spec.loader.exec_module(model_module)
VERSION, make_model, ModelConfig = model_module.VERSION, model_module.make_model, model_module.ModelConfig
SPECS = {'balls': (30, 9), 'collision': (15, 4), 'blocktower': (30, 4)}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(data, tmp); os.replace(tmp, path)


def emit(event, **kwargs):
    print(json.dumps({'event': event, 'time': time.time(), **kwargs}, allow_nan=False), flush=True)



def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('Frozen artifact differs: ' + str(path))
    write(path, value)


def seeded(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        with torch.cuda.device(device): torch.cuda.manual_seed(seed)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state(device)] if device.type == 'cuda' else [])


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        if len(state['cuda']) != 1: raise ValueError('Missing active-device CUDA RNG')
        torch.cuda.set_rng_state(state['cuda'][0], device=device)


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class FeatureData:
    def __init__(self, path, scene):
        self.path, self.scene = Path(path).resolve(), scene
        self.ids, self.arrays, self.files = {}, {}, {}
        self.frames, self.slots = SPECS[scene]
        marker = self.path / 'scene_COMPLETE.json'
        if read(marker).get('status') != 'COMPLETE':
            raise ValueError('Feature scene is not completely and atomically prepared')
        self.files[str(marker)] = digest(marker)
        for split in ('train', 'val'):
            folder = self.path / split
            for name in ('manifest.json', 'COMPLETE.json', 'ids.json'):
                path = folder / name
                self.files[str(path)] = digest(path)
            if read(folder / 'COMPLETE.json').get('status') != 'COMPLETE':
                raise ValueError('Feature split is incomplete')
            ids = list(map(str, read(folder / 'ids.json')))
            if len(set(ids)) != len(ids) or not ids:
                raise ValueError('Feature IDs must be unique and nonempty')
            self.ids[split] = ids
            self.arrays[split] = {}
            for name in ('features_ab', 'features_cd', 'presence_ab', 'presence_cd'):
                arr = np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                shape = (len(ids), self.frames, self.slots) + ((784,) if name.startswith('features') else ())
                if arr.shape != shape:
                    raise ValueError(f'Unexpected {split}/{name} shape {arr.shape}; expected {shape}')
                if name.startswith('features') and arr.dtype not in (np.float16, np.float32):
                    raise ValueError('RGB features must be floating point')
                if name.startswith('presence') and arr.dtype not in (np.uint8, np.bool_):
                    raise ValueError('Visual presence must be uint8/bool')
                self.arrays[split][name] = arr
        if set(self.ids['train']) & set(self.ids['val']):
            raise ValueError('Train/validation ID overlap')

    def _tensor(self, split, name, indices, device, only_c=False):
        arr = self.arrays[split][name]
        # Index before conversion: no worker loads or copies the whole memmap.
        values = np.array((arr[:, :3] if only_c else arr)[np.asarray(indices)], copy=True)
        if name.startswith('presence'):
            if not np.isin(values, (0, 1)).all():
                raise ValueError('Invalid public visual presence')
            return torch.as_tensor(values, device=device, dtype=torch.bool)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite public RGB feature')
        return torch.as_tensor(values, device=device, dtype=torch.float32)

    def history(self, split, indices, device):
        return (self._tensor(split, 'features_ab', indices, device),
                self._tensor(split, 'presence_ab', indices, device))

    def context(self, split, indices, device):
        ab, mask = self.history(split, indices, device)
        return (ab, mask, self._tensor(split, 'features_cd', indices, device, True),
                self._tensor(split, 'presence_cd', indices, device, True))

    def target(self, split, indices, device):
        # Target-only full video gives index3 its legal index2 predecessor.
        # Context() separately exposes only CD[:3].
        return (self._tensor(split, 'features_cd', indices, device),
                self._tensor(split, 'presence_cd', indices, device))


class EpochPlanner:
    """The existing recipient-shuffle rule, without relation metadata."""
    def __init__(self, data, seed):
        self.ids, self.seed = data.ids['train'], seed

    def make(self, epoch, batch_size):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 6001]))
        order = rng.permutation(len(self.ids))
        batches = [order[s:s+batch_size] for s in range(0, len(order), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.r_[batches[-2], batches.pop()]
        if not np.array_equal(np.sort(np.concatenate(batches)), np.arange(len(self.ids))):
            raise ValueError('Recipient exposure is not a permutation')
        return dict(batches=batches, order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                    query_exposures=len(order))


def setup(args):
    torch.set_num_threads(args.threads)
    data = FeatureData(args.features, args.scene)
    files = dict(data.files)
    for path in (Path(__file__).resolve(), _model_path): files[str(path)] = digest(path)
    if args.protocol: files[str(Path(args.protocol).resolve())] = digest(args.protocol)
    body = dict(version=VERSION, scene=args.scene, method='Monolithic', family='JEPA',
        features=str(data.path), files=files, seed=args.seed, epochs=100,
        batch_size=args.batch_size, initial_microbatch=args.microbatch, learning_rate=args.lr,
        weight_decay=1e-4, clip_norm=1., model_config=make_model_config(),
        source_task='one joint E(AB,boundary,CD[:3])->U128 predicts live motion g(CD)[3:]',
        source_selection='fixed epoch50 and epoch100; no latent loss selection',
        parameter_policy='random initialization; no P/T split, relations, donor or parameter labels',
        frontend='frozen supervised official last_cnn784',
        target_gradients='live shared online motion projection, no EMA or detached teacher',
        sigreg_statistics='same fixed AB/future-CD times; independent episode axis by time/slot',
        shuffle_rule='SeedSequence([seed,epoch,6001]); each train recipient once each epoch',
        test_read=False, coordinate_labels_read=False, relation_index_read=False)
    body['sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return data, EpochPlanner(data,args.seed), body


def make_model_config():
    from dataclasses import asdict
    return {'version': VERSION, **asdict(ModelConfig())}


def parameter_inventory(model):
    return dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total=sum(p.numel() for p in model.parameters()),
                modules={name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()})


def prepare(args):
    data, planner, binding = setup(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    immutable(out/'binding.json',binding)
    seeded(args.seed,torch.device('cpu')); model=make_model()
    initial=dict(version=VERSION,binding_sha256=binding['sha256'],state_sha256=state_sha(model),
                 parameter_counts=parameter_inventory(model),seed=args.seed,source_pretrained_checkpoint=None)
    immutable(out/'initialization.json',initial)
    path=out/'initialization.pt'
    if path.exists():
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        check=make_model();check.load_state_dict(loaded['model'])
        if loaded['binding_sha256']!=binding['sha256'] or state_sha(check)!=initial['state_sha256']:
            raise ValueError('Changed initial model artifact')
    else:
        save(path,dict(version=VERSION,model=model.state_dict(),model_config=model.artifact_config(),
             binding_sha256=binding['sha256'],epoch=0,step=0))
    first=planner.make(1,args.batch_size)
    marker=dict(status='COMPLETE',version=VERSION,binding_sha256=binding['sha256'],
        initialization_sha256=initial['state_sha256'],initialization_file_sha256=digest(path),
        train_recipients=len(data.ids['train']),val_recipients=len(data.ids['val']),
        steps_per_epoch=len(first['batches']),expected_steps50=50*len(first['batches']),
        expected_steps100=100*len(first['batches']),first_order_sha256=first['order_sha256'],
        parameter_counts=initial['parameter_counts'],optimizer_steps=0,test_read=False)
    immutable(out/'prepared.json',marker);emit('PREPARED',**marker)


def checked_setup(args):
    data,planner,binding=setup(args);out=Path(args.out)
    if read(out/'binding.json')!=binding or read(out/'prepared.json')['binding_sha256']!=binding['sha256']:
        raise ValueError('Prepare this exact fixed100 binding first')
    if digest(out/'initialization.pt')!=read(out/'prepared.json')['initialization_file_sha256']:
        raise ValueError('Changed initialization artifact')
    return data,planner,binding


def encode_micro(model,ab,am,c,cm,microbatch):
    output=[]
    for start in range(0,len(ab),microbatch):
        values=(ab[start:start+microbatch],am[start:start+microbatch],c[start:start+microbatch],cm[start:start+microbatch])
        if model.training and microbatch<len(ab):
            u=recompute(model.encode_joint,*values,use_reentrant=False)
        else: u=model.encode_joint(*values)
        output.append(u)
    return torch.cat(output)


def predict_micro(model,u,active,steps,microbatch):
    output=[]
    for start in range(0,len(u),microbatch):
        uu,mm=u[start:start+microbatch],active[start:start+microbatch]
        if model.training and microbatch<len(u):
            y=recompute(lambda a,b:model.predict_from_context(a,b,steps),uu,mm,use_reentrant=False)
        else: y=model.predict_from_context(uu,mm,steps)
        output.append(y)
    return torch.cat(output)


def masked_mse(prediction,target,mask):
    values=(prediction.float()-target.float()).square().mean(-1)
    weight=mask.to(values.dtype)
    return (values*weight).sum()/weight.sum().clamp_min(1)


def batch_objective(model,data,indices,microbatch):
    device=next(model.parameters()).device
    ab,am,c,cm=data.context('train',indices,device)
    cd,mask_cd=data.target('train',indices,device)
    active=cm.any(1);target_mask=mask_cd[:,3:]&active[:,None]
    u=encode_micro(model,ab,am,c,cm,microbatch)
    prediction=predict_micro(model,u,active,data.frames-3,microbatch)
    target=model.target(cd)
    self_loss=masked_mse(prediction,target,target_mask)
    reg=model.regularization(ab,am,cd,mask_cd)
    loss=self_loss+model.config.sigreg_weight*reg
    metrics=dict(loss=float(loss.detach()),self_loss=float(self_loss.detach()),sigreg=float(reg.detach()),
        u_std_all=float(u.detach().float().flatten(0,1).std(0).mean()),
        target_std=float(target.detach().float().flatten(0,2).std(0).mean()),
        cross=0.,align=0.,donor_encodes=0)
    return loss,metrics


@torch.no_grad()
def evaluate(model,data,batch_size=32,limit=512):
    model.eval();device=next(model.parameters()).device
    values,ids,uncovered=[],[],[]
    count=min(limit,len(data.ids['val'])) if limit else len(data.ids['val'])
    for start in range(0,count,batch_size):
        indices=np.arange(start,min(start+batch_size,count))
        ab,am,c,cm=data.context('val',indices,device)
        cd,mask_cd=data.target('val',indices,device)
        mask=mask_cd[:,3:]&cm.any(1)[:,None]
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            u=model.encode_joint(ab,am,c,cm)
            prediction=model.predict_from_context(u,cm.any(1),data.frames-3)
            target=model.target(cd)
        errors=(prediction.float()-target.float()).square().mean(-1)
        denominator=mask.sum((1,2));scores=(errors*mask).sum((1,2))/denominator.clamp_min(1)
        for i,score,n in zip(indices,scores.tolist(),denominator.tolist()):
            if n: ids.append(data.ids['val'][int(i)]);values.append(score)
            else: uncovered.append(data.ids['val'][int(i)])
    if not values or not np.isfinite(values).all(): raise ValueError('No finite validation scores')
    return dict(mse=float(np.mean(values)),ids=ids,per_recipient_mse=values,uncovered_ids=uncovered,
                recipients=count,scored=len(values),metric='live latent diagnostic only, not cross-model ranking or selection')


def device_for(args):
    device=torch.device(args.device)
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    return device


def smoke(args):
    data,planner,binding=checked_setup(args);device=device_for(args);seeded(args.seed,device)
    model=make_model().to(device);initial=read(Path(args.out)/'initialization.json')
    if state_sha(model)!=initial['state_sha256']: raise ValueError('Smoke initialization differs')
    model.train();indices=planner.make(1,args.batch_size)['batches'][0]
    before=state_sha(model)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        loss,metrics=batch_objective(model,data,indices,args.microbatch)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite real smoke loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if state_sha(model)!=before: raise ValueError('Smoke changed model state without optimizer')
    model.eval()
    with torch.no_grad():
        ab,am,c,cm=data.context('train',indices[:2],device)
        direct=model.encode_joint(ab,am,c,cm)
        cached=model.encode_joint_from_cached(model.encode_history_state(ab,am),model.encode_current_tokens(c,cm),cm)
        gap=float((direct-cached).abs().max())
        if gap!=0.: raise ValueError('Cached joint path differs from direct context path')
        # A changed future cannot alter the encoder's three-frame context input.
        if c.shape[1]!=3 or direct.shape!=(len(c),data.slots,128):
            raise ValueError('Joint context interface differs')
    result=dict(status='PASS',version=VERSION,binding_sha256=binding['sha256'],metrics=metrics,
        gradient_norm=float(norm),cached_path_max_abs_difference=gap,joint_shape=list(direct.shape),
        source_model_unchanged=True,optimizer_steps=0,donor_encodes=0,test_read=False,coordinate_labels_read=False)
    write(Path(args.out)/'smoke.json',result);emit('SMOKE_PASS',**result)


def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'owner.lock','a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Monolithic source already running') from e
        data,planner,binding=checked_setup(args)
        smoke_record=read(out/'smoke.json')
        if smoke_record.get('status')!='PASS' or smoke_record['binding_sha256']!=binding['sha256']:
            raise ValueError('Matching real-data smoke required')
        if (out/'complete.json').exists():
            done=read(out/'complete.json')
            if done.get('binding_sha256')!=binding['sha256'] or done.get('epochs')!=100:
                raise ValueError('Different completed source binding')
            emit('ALREADY_COMPLETE',**done);return
        device=device_for(args);seeded(args.seed,device)
        model=make_model().to(device)
        initial=read(out/'initialization.json')
        if state_sha(model)!=initial['state_sha256']: raise ValueError('Training initialization differs')
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        epoch,next_batch,step,history,running=1,0,0,[],{}
        prior_seconds=0.;microbatch=args.microbatch;latest=out/'latest.pt'
        if latest.exists():
            if not args.resume: raise ValueError('Existing partial source requires --resume')
            old=torch.load(latest,map_location='cpu',weights_only=False)
            if old.get('version')!=VERSION or old.get('binding_sha256')!=binding['sha256']:
                raise ValueError('Different continuation checkpoint')
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
            epoch,next_batch,step=old['next_epoch'],old['next_batch'],old['step']
            history,running=old['history'],old['running'];microbatch=old['microbatch']
            prior_seconds=old['seconds'];restore_rng(old['rng'],device)
        elif args.resume: emit('RESUME_NEW_RUN',reason='No existing checkpoint; starting the common random initialization')
        started=time.time()
        def record(ne,nb):
            return dict(version=VERSION,model=model.state_dict(),model_config=model.artifact_config(),
                optimizer=optimizer.state_dict(),rng=rng_state(device),rng_device_index=0 if device.type=='cuda' else None,
                scene=args.scene,method='Monolithic',family='JEPA',binding=binding,binding_sha256=binding['sha256'],
                epoch=ne-1 if nb==0 else ne,next_epoch=ne,next_batch=nb,step=step,history=history,running=running,
                microbatch=microbatch,initialization_sha256=initial['state_sha256'],
                seconds=prior_seconds+time.time()-started,test_read=False,coordinate_labels_read=False)
        def publish_checkpoint(completed, validation):
            checkpoint=out/f'checkpoint_{completed}.pt'
            marker=out/f'checkpoint_{completed}_complete.json'
            if marker.exists():
                old_marker=read(marker)
                if (old_marker.get('binding_sha256')!=binding['sha256'] or old_marker.get('epoch')!=completed
                        or old_marker.get('steps')!=step or old_marker.get('checkpoint_sha256')!=digest(checkpoint)):
                    raise ValueError('Conflicting published source snapshot')
                return
            save(checkpoint,record(completed+1,0))
            write(out/f'validation_{completed}.json',dict(epoch=completed,**validation))
            write(marker,dict(status='COMPLETE',version=VERSION,scene=args.scene,method='Monolithic',family='JEPA',
                epoch=completed,epochs=completed,steps=step,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                binding_sha256=binding['sha256'],initialization_sha256=initial['state_sha256'],selected_epoch=completed,
                selection='fixed source budget, no latent-loss selection',test_read=False,coordinate_labels_read=False))

        write(out/'worker.json',dict(status='RUNNING',pid=os.getpid(),host=socket.gethostname(),device=str(device),
            started_at=started,binding_sha256=binding['sha256']))
        try:
            # Repair only an unpublished fixed-budget boundary after an interrupted
            # commit. Once a marker exists its checkpoint is never overwritten.
            if next_batch==0 and epoch-1 in (50,100) and not (out/f'checkpoint_{epoch-1}_complete.json').exists():
                publish_checkpoint(epoch-1,evaluate(model,data,min(microbatch,32)))
            while epoch<=args.stop_epoch:
                plan=planner.make(epoch,args.batch_size)
                if next_batch==0:
                    running=dict(weighted={},examples=0,steps=0,started_at=time.time(),order_sha256=plan['order_sha256'])
                elif running.get('order_sha256')!=plan['order_sha256']: raise ValueError('Resumed query order differs')
                model.train()
                for batch_number in range(next_batch,len(plan['batches'])):
                    indices=plan['batches'][batch_number];optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        loss,metrics=batch_objective(model,data,indices,microbatch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite monolithic objective')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();del loss
                    step+=1;next_batch=batch_number+1
                    running['examples']+=len(indices);running['steps']+=1
                    for key,value in metrics.items():running['weighted'][key]=running['weighted'].get(key,0.)+value*len(indices)
                    if step%50==0 or next_batch==len(plan['batches']):
                        save(latest,record(epoch,next_batch))
                        status=dict(status='RUNNING',epoch=epoch,batch=next_batch,step=step,history=history,latest=metrics,
                            grad_norm=float(norm),order_sha256=plan['order_sha256'],microbatch=microbatch,
                            seconds=prior_seconds+time.time()-started,test_read=False)
                        write(out/'progress.json',status);emit('PROGRESS',**{k:v for k,v in status.items() if k!='history'})
                validation=evaluate(model,data,min(microbatch,32))
                if running['examples']!=len(data.ids['train']): raise ValueError('Incomplete query exposure')
                row=dict(epoch=epoch,mse=validation['mse'],train={k:v/running['examples'] for k,v in running['weighted'].items()},
                    seconds=time.time()-running['started_at'],query_exposures=running['examples'],
                    order_sha256=plan['order_sha256'],metric='live latent diagnostic only')
                history.append(row);completed=epoch;epoch+=1;next_batch=0;running={}
                if completed in (50,100): publish_checkpoint(completed,validation)
                save(latest,record(epoch,0))
                write(out/'progress.json',dict(status='RUNNING' if completed<args.stop_epoch else ('PAUSED' if completed<100 else 'COMPLETE'),
                    epoch=completed,step=step,history=history,microbatch=microbatch,seconds=prior_seconds+time.time()-started,test_read=False))
                emit('EPOCH_COMPLETE',**row)
            if args.stop_epoch<100:
                result=dict(status='PAUSED',version=VERSION,scene=args.scene,epoch=args.stop_epoch,step=step,
                    next_epoch=epoch,next_batch=next_batch,binding_sha256=binding['sha256'],test_read=False,
                    resume_hint='Same fixed100 plan: rerun train --stop-epoch100 --resume')
                write(out/'paused.json',result);write(out/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('PAUSED',**result);return
            expected=read(out/'prepared.json')['expected_steps100']
            if step!=expected or len(history)!=100: raise ValueError('Final source100 budget differs')
            result=dict(read(out/'checkpoint_100_complete.json'),epochs=100,seconds=prior_seconds+time.time()-started,
                source_pretrained_checkpoint=None,parameter_counts=initial['parameter_counts'],finished_at=time.time())
            write(out/'complete.json',result);write(out/'worker.json',dict(result,pid=os.getpid(),exit_code=0));emit('COMPLETE',**result)
        except Exception as error:
            result=dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),epoch=epoch,
                next_batch=next_batch,step=step,binding_sha256=binding['sha256'],test_read=False)
            write(out/'failure.json',result);emit('FAILED',**result);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','smoke','train'))
    p.add_argument('--scene',required=True,choices=tuple(SPECS));p.add_argument('--features',required=True)
    p.add_argument('--out',required=True);p.add_argument('--protocol');p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,choices=(100,),default=100)
    p.add_argument('--stop-epoch',type=int,choices=(50,100),default=100)
    p.add_argument('--seed',type=int,choices=(0,),default=0);p.add_argument('--batch-size',type=int,choices=(32,),default=32)
    p.add_argument('--microbatch',type=int,default=32);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    return p


def main():
    p=parser();args=p.parse_args()
    if args.microbatch<1 or args.microbatch>args.batch_size or args.threads<1 or args.lr!=.0003:
        p.error('Expected microbatch1..32, positive threads, and fixed learning rate3e-4')
    globals()[args.command](args)


if __name__=='__main__':main()
