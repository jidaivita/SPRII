"""Common supervised physical-prediction training for the four Z references.

This objective is explicitly distinct from Paper A self-supervised pretraining.
Private train labels enter the loss only. Model arguments are tensors containing
the authorized pixels, past donor actions, future query actions and null mask.
"""
import argparse,collections,hashlib,json,random,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .pixel_models import PixelModelConfig,PixelDynamicsModel,query_profile_tensor
from .target_scaling import HORIZONS
from .training_protocol import file_digest,source_fingerprint,verify_admission,verify_formal,validation_case

CONTEXTS=('forced24','forced48','forced96','MM48','FF48','MF48','repeatM48','repeatF48','M48','F48')
CONTEXT_PROBABILITIES=(.15,.10,.30,.08,.08,.15,.04,.04,.03,.03)
QUERY_BUDGETS=(0,1,3,7,15,31,63,95)


def public_pixels(images):
    x=np.asarray(images)
    if x.dtype!=np.uint8 or x.ndim!=3 or not len(x):raise ValueError('raw gray image support required')
    gray=x.astype(np.float32)/255.;difference=np.zeros_like(gray);difference[1:]=gray[1:]-gray[:-1]
    return np.stack((gray,difference),axis=1)


class TrainingBank:
    def __init__(self,root,resolution=128):
        self.root=Path(root);self.resolution=resolution
        self.config=json.loads((self.root/'BANK_CONFIG.json').read_text())
        report=json.loads((self.root/'BANK_REPORT.json').read_text())
        if self.config.get('test_generated') or report.get('test_read') or report['execution_errors']:
            raise ValueError('unqualified training bank or test access')
        self.manifest_path=self.root/'MANIFEST.private.json'
        manifest=json.loads(self.manifest_path.read_text())
        if any(r['split'] not in ('train','validation') for r in manifest['episodes']):
            raise ValueError('test episode manifest forbidden in training reader')
        self.manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.statistics=json.loads((self.root/'TRAIN_TARGET_STATISTICS.json').read_text())
        if self.statistics['source_split']!='train':raise ValueError('non-training target statistics')
        self.by_system=collections.defaultdict(lambda:collections.defaultdict(list));self.systems={};self.rows={}
        for row in manifest['episodes']:
            self.rows[row['episode_key']]=row;key=(row['split'],row['system_key'])
            self.by_system[key][row['kind']].append(row);self.systems[key]=row['system']
        self.keys={split:sorted(k for k in self.by_system if k[0]==split) for split in ('train','validation')}
        self.selection_keys=[k for k in self.keys['validation'] if self.systems[k]['stratum'] in ('continuous_new_systems','heldout_factorial_combinations')]
        for groups in self.by_system.values():
            for rows in groups.values():rows.sort(key=lambda r:r['episode_key'])

    def _path(self,relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('asset outside registered bank')
        return path

    @lru_cache(maxsize=192)
    def visible(self,key):
        row=self.rows[key]
        with np.load(self._path(row['assets'][str(self.resolution)]['path']),allow_pickle=False) as data:
            images=data['images'];actions=data['actions']
        return images,actions

    @lru_cache(maxsize=256)
    def labels(self,key):
        row=self.rows[key]
        if row['split'] not in ('train','validation'):raise ValueError('private label split is not permitted')
        with np.load(self._path(row['private_state_path']),allow_pickle=False) as data:return data['state']

    def eligible(self,key,kind,frames):
        return [r for r in self.by_system[key][kind] if r['raw_frames']>=frames]

    def sample_batch(self,split,seed,batch_size=16,*,device='cpu',forced_context=None,forced_query=None,forced_horizon=None,forced_null=None,forced_budget=None):
        rng=np.random.default_rng(seed)
        if split not in ('train','validation'):raise ValueError('training sampler cannot open test')
        keys=self.keys['train'] if split=='train' else self.selection_keys
        if not keys:raise ValueError('empty registered sampling population')
        context=forced_context or rng.choice(CONTEXTS,p=CONTEXT_PROBABILITIES)
        if context not in CONTEXTS:raise ValueError('unregistered context mixture')
        history_kinds={'MM48':['mass','mass'],'FF48':['free','free'],'MF48':['mass','free'],
            'repeatM48':['mass','mass'],'repeatF48':['free','free'],'M48':['mass'],'F48':['free']}.get(context,['forced'])
        frames=int(''.join(x for x in context if x.isdigit()))
        kind=forced_query or str(rng.choice(['cold','moving']))
        budget=int(rng.choice([0,1] if kind=='cold' else QUERY_BUDGETS)) if forced_budget is None else int(forced_budget)
        horizon=int(forced_horizon or rng.choice(HORIZONS))
        if kind not in ('cold','moving') or horizon not in HORIZONS:raise ValueError('unregistered physical query')
        if budget not in ((0,1) if kind=='cold' else QUERY_BUDGETS):raise ValueError('query budget outside declared profile')
        histories=[([],[]) for _ in history_kinds];queries=[];future=[];targets=[];null=[];audit=[]
        for _ in range(batch_size):
            system=keys[int(rng.integers(len(keys)))];groups=self.by_system[system]
            candidates=[r for r in groups[kind] if r['anchor'] is not None and r['raw_frames']>r['anchor']+horizon]
            if not candidates:raise ValueError('query support absent; do not silently resample systems')
            qrow=candidates[int(rng.integers(len(candidates)))];images,actions=self.visible(qrow['episode_key']);anchor=qrow['anchor']
            queries.append(public_pixels(images[anchor-budget:anchor+1]));future.append(actions[anchor:anchor+horizon])
            state=self.labels(qrow['episode_key']);targets.append(state[anchor+horizon]-state[anchor])
            selected=[]
            for hi,hkind in enumerate(history_kinds):
                if context.startswith('repeat') and hi==1:hrow=selected[0]
                else:
                    options=[r for r in self.eligible(system,hkind,frames) if r['episode_key'] not in [x['episode_key'] for x in selected]]
                    if not options:raise ValueError('independent donor support absent; no same-episode fallback')
                    hrow=options[int(rng.integers(len(options)))]
                if hrow['episode_key']==qrow['episode_key']:raise ValueError('same-episode query donor leak')
                selected.append(hrow);himages,hactions=self.visible(hrow['episode_key'])
                histories[hi][0].append(public_pixels(himages[:frames]));histories[hi][1].append(hactions[:frames-1])
            is_null=bool(rng.random()<.25) if forced_null is None else bool(forced_null)
            null.append([0. if is_null else 1.])
            audit.append(dict(query=qrow['episode_key'],donors=[r['episode_key'] for r in selected],split=split,system=system[1]))
        tensor=lambda x:torch.from_numpy(np.asarray(x,np.float32)).to(device)
        profile='cold_rest_joint_state_delta_8d' if kind=='cold' else 'passive_prefix95_joint_state_delta_8d'
        public=dict(histories=[(tensor(x),tensor(u)) for x,u in histories],query_images=tensor(queries),future_actions=tensor(future),context_mask=tensor(null),query_profile=query_profile_tensor(profile,batch_size,device))
        return public,tensor(targets),dict(context=context,query_kind=kind,budget=budget,horizon=horizon,cases=audit)


def runtime_policy(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')


def train(args):
    if args.output.exists():raise ValueError('training attempt exists; choose a new immutable attempt')
    args.output.mkdir(parents=True);runtime_policy(args.seed);torch.set_num_threads(1)
    device=torch.device(args.device);bank=TrainingBank(args.bank,args.resolution)
    bindings=None;source=source_fingerprint();admission=None
    if args.stage!='development_smoke':
        if args.data_admission is None:raise ValueError('training requires a verified data-admission report')
        admission=json.loads(args.data_admission.read_text());verify_admission(admission,bank)
        if args.validation_batches%80:raise ValueError('selection requires complete paired80-batch query/budget/horizon cycles')
    if args.stage=='formal':
        if args.protocol is None:raise ValueError('formal training requires a frozen protocol')
        protocol=json.loads(args.protocol.read_text())
        bindings=verify_formal(protocol,args,bank,admission,source)
        (args.output/'FROZEN_PROTOCOL.json').write_bytes(args.protocol.read_bytes())
    model=PixelDynamicsModel(PixelModelConfig(family=args.family,resolution=args.resolution)).to(device)
    model.set_target_statistics(bank.statistics)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    config=dict(stage=args.stage,family=args.family,seed=args.seed,batch_size=args.batch_size,accumulation=args.accumulation,
        updates=args.max_updates,learning_rate=args.learning_rate,weight_decay=1e-4,gradient_clip=5.,
        model=model.artifact_config(),parameters=sum(p.numel() for p in model.parameters()),
        bank_manifest_sha256=bank.manifest_sha256,target_statistics=bank.statistics,source_fingerprint=source,
        data_admission_sha256=file_digest(args.data_admission) if args.data_admission else None,
        formal_bindings=bindings,protocol_sha256=file_digest(args.protocol) if args.protocol else None,
        context_mixture=dict(zip(CONTEXTS,CONTEXT_PROBABILITIES)),null_probability=.25,
        query_profile='public cold-rest/passive-prefix95 declaration supplied to every head, matching explicit-reference prior permissions; no donor correctness or identity flags',
        loss='mean squared physical target error divided by training-only per-horizon per-component standard deviations',
        labels='train physical states used in supervised loss only; this is Z reference training, not A self-supervised pretraining',
        runtime='float32; CUDA and cuDNN TF32 disabled; per-frame LayerNorm; no across-example normalization',
        validation='fixed train-normalized loss on heldout-combination/continuous validation systems; range extrapolation excluded from checkpoint selection',
        selection='equal cold/moving weight; within each profile equal paired matched/null, query-budget and horizon weight; identical case draws within pairs; each80 batches is a complete cycle',test_read=False)
    (args.output/'TRAIN_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');start=time.monotonic()
    schedule=lambda step: min(1.,(step+1)/100)*(.1+.9*.5*(1+np.cos(np.pi*step/max(1,args.max_updates))))
    with (args.output/'training.jsonl').open('w') as log:
        for step in range(args.max_updates):
            model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;step_start=time.monotonic()
            for micro in range(args.accumulation):
                public,target,audit=bank.sample_batch('train',987000000+args.seed*1000000+step*args.accumulation+micro,args.batch_size,device=device)
                prediction=model(**public);scale=model.target_scale[audit['horizon']]
                loss=((prediction-target)/scale).square().mean()/args.accumulation
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
                loss.backward();loss_sum+=float(loss.detach())
                del public,target,prediction,loss
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
            if not np.isfinite(grad):raise FloatingPointError('nonfinite gradient')
            for group in optimizer.param_groups:group['lr']=args.learning_rate*schedule(step)
            optimizer.step();row=dict(update=step+1,train_loss=loss_sum,gradient_norm=grad,seconds=time.monotonic()-step_start)
            if (step+1)%args.validate_every==0 or step+1==args.max_updates:
                model.eval();scores=collections.defaultdict(list);condition_scores=collections.defaultdict(list)
                with torch.inference_mode():
                    for vi in range(args.validation_batches):
                        specification=validation_case(vi)
                        public,target,audit=bank.sample_batch('validation',batch_size=args.batch_size,device=device,**specification)
                        value=float(((model(**public)-target)/model.target_scale[audit['horizon']]).square().mean())
                        scores[audit['query_kind']].append(value)
                        condition_scores[audit['query_kind']+('/null' if specification['forced_null'] else '/matched')].append(value)
                score=float(np.mean([np.mean(values) for values in scores.values()]));row['validation_loss']=score
                row['validation_components']={key:float(np.mean(values)) for key,values in condition_scores.items()}
                if score<best:
                    best=score
                    torch.save(dict(model=model.state_dict(),config=config,update=step+1,validation_loss=score),args.output/'best.pt')
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,update=step+1,validation_loss=score),args.output/'last.pt')
            log.write(json.dumps(row)+'\n');log.flush()
            if (step+1)%10==0 or 'validation_loss' in row:print(json.dumps(row),flush=True)
    report=dict(status='EXECUTED',stage=args.stage,updates=args.max_updates,best_validation_loss=best,seconds=time.monotonic()-start,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.pt')},test_read=False)
    (args.output/'TRAIN_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('gru','transformer','transition_deepsets','causal_tcn'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resolution',type=int,default=128);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--stage',choices=('development_smoke','development_training','formal'),default='development_smoke')
    p.add_argument('--protocol',type=Path);p.add_argument('--data-admission',type=Path);p.add_argument('--max-updates',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--validation-batches',type=int,default=80);a=p.parse_args()
    if min(a.max_updates,a.batch_size,a.accumulation,a.validate_every,a.validation_batches)<1:raise ValueError('positive training budgets required')
    train(a)


if __name__=='__main__':main()
