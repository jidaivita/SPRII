"""Train/validation-separated linear and nonlinear decodability assays.

Physical and episode-nuisance labels are evaluator-owned targets. Frozen model
representations are computed from actual public history pixels/actions only.
Probe failure is a decoder-specific lower bound, not proof of information loss.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

LABELS=('log_m','log_gamma','log_k','log_k_over_m','initial_center_x','initial_center_y',
        'initial_orientation_sin','initial_orientation_cos','initial_strain','observed_action_effort')
SELECTION_STRATA=('continuous_new_systems','heldout_factorial_combinations')
FEATURE_DEFINITIONS={'neural':'ingested_mean_max_128d_v1','explicit':'posterior_log_mean_sd_physical_mean_12d_v1'}
EXPLICIT_CONFIGURATION=dict(method='compressed_visual_reference',resolution=128,samples=256,max_histories=8,noise_floor_m=.0006)


def private_labels(bank,row,frames):
    state=bank.labels(row['episode_key'])[0];_,actions=bank.visible(row['episode_key'])
    m,g,k=row['theta'];relative=state[2:4]-state[:2];length=np.linalg.norm(relative);center=(state[:2]+state[2:4])/2
    return np.r_[np.log([m,g,k,k/m]),center,relative[1]/length,relative[0]/length,length-.35,
                 np.sum(actions[:frames-1]**2)*.05]


def explicit_feature(job):
    root,row,frames=job
    from persistbench.contracts import RunContext,ComputeTier,Split
    from .schema import Config,Episode,history_payload
    from .adapters import z_experience
    from .persistent_reference import CompressedVisualReference
    with np.load(Path(root)/row['assets']['128']['path'],allow_pickle=False) as data:
        images=data['images'][:frames];actions=data['actions'][:frames-1]
    episode=Episode(images,actions,np.arange(frames)*.05,np.zeros((frames,8)),{})
    agent=CompressedVisualReference(Config(resolution=128),samples=256)
    agent.initialize(RunContext('visual_elastic_coupling/formation','1.1',Split.VALIDATION,ComputeTier.STANDARD,990017))
    payload=z_experience(history_payload(episode,0,frames-1));agent.ingest(payload)
    payload.observations.fill(0);payload.actions.fill(0)
    theta=agent.parameter_samples();physical=np.c_[theta,theta[:,2]/theta[:,0]];logs=np.log(physical)
    # This is a declared belief readout of the stored likelihood factors. No
    # private theta is available to the fit or posterior-moment calculation.
    code=np.r_[logs.mean(0),logs.std(0),physical.mean(0)]
    return code,dict(fit_failures=agent.fit_failures,memory_bytes=agent.mutable_state_bytes())


def extract(args):
    import torch
    from .pixel_training import TrainingBank,public_pixels,runtime_policy
    from .pixel_models import PixelDynamicsModel,PixelModelConfig
    from .training_protocol import source_fingerprint
    from .dataset_snapshot import stable_digest,verify
    if not args.explicit and (args.checkpoint is None or args.model_source is None):raise ValueError('neural extraction requires checkpoint and original model source directory')
    bank=TrainingBank(args.bank,128);rows=sorted([r for r in bank.rows.values() if r['kind']=='forced' and r['raw_frames']>=args.frames],key=lambda r:r['episode_key'])
    snapshot_path=bank.root/'BANK_SNAPSHOT.json';snapshot_sha=stable_digest(snapshot_path)['sha256']
    snapshot=json.loads(snapshot_path.read_text());before_content=verify(bank.root,snapshot);source_before=source_fingerprint()
    checkpoint_before=stable_digest(args.checkpoint)['sha256'] if not args.explicit else None
    if args.limit_per_system:
        counts={};selected=[]
        for row in rows:
            key=(row['split'],row['system_key']);count=counts.get(key,0)
            if count<args.limit_per_system:selected.append(row);counts[key]=count+1
        rows=selected
    if args.output.exists():raise ValueError('formation extraction attempt exists')
    args.output.mkdir(parents=True);features=[];diagnostics=[];start=time.monotonic();runtime_policy(17);torch.set_num_threads(1)
    config=dict(schema='vec.formation-feature.v1.1',method='explicit' if args.explicit else 'neural',frames=args.frames,
        bank_manifest_sha256=bank.manifest_sha256,labels=LABELS,kind='forced',limit_per_system=args.limit_per_system,
        frozen=True,test_read=False,formal_results=False,resolution=128,feature_seed=990017,
        feature_definition=FEATURE_DEFINITIONS['explicit' if args.explicit else 'neural'],source_fingerprint=source_fingerprint())
    if args.explicit:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for code,diag in pool.map(explicit_feature,[(str(args.bank),r,args.frames) for r in rows]):features.append(code);diagnostics.append(diag)
        config['representation']='12D posterior moments: mean(log m,gamma,k,k/m), sd(log...), mean(physical...); derived from retained H/b with256 samples'
        config['physical_prior_bounds']=[[.5,.25,4.],[2.,1.5,25.]]
        config['explicit_configuration']=EXPLICIT_CONFIGURATION
        config['extrapolation_note']='This development artifact retains its original prior support; range-extrapolation limitations must be reported separately from broader-prior raw-evidence calibration.'
    else:
        if args.checkpoint is None or args.model_source is None:raise ValueError('neural extraction requires checkpoint and original model source directory')
        # The evaluation package may add assays, but the actual encoding and
        # preprocessing implementations must match the artifact's source.
        for name in ('pixel_models.py','pixel_training.py'):
            original=args.model_source/'src/persistbench/envs/visual_elastic_coupling'/name
            if hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256((Path(__file__).parent/name).read_bytes()).digest():
                raise ValueError('model/observation implementation differs from checkpoint source: '+name)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['config'].get('bank_manifest_sha256')!=bank.manifest_sha256:
            raise ValueError('formation training/validation bank differs from checkpoint training bank')
        runtime_policy(checkpoint['config']['seed']);model=PixelDynamicsModel(PixelModelConfig(**checkpoint['config']['model']))
        if not args.random_init:model.load_state_dict(checkpoint['model'],strict=True)
        model.to(args.device).eval()
        config.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),training_source_fingerprint=checkpoint['config']['source_fingerprint'],
            family=model.cfg.family,selected_update=0 if args.random_init else checkpoint['update'],random_initialization_control=args.random_init,
            representation='128D mean/max persistent state for one ingested episode')
        with torch.inference_mode():
            for first in range(0,len(rows),args.batch_size):
                rs=rows[first:first+args.batch_size];images=[];actions=[]
                for row in rs:
                    x,u=bank.visible(row['episode_key']);images.append(public_pixels(x[:args.frames]));actions.append(u[:args.frames-1])
                code=model.encode_history(torch.as_tensor(np.asarray(images),device=args.device),torch.as_tensor(np.asarray(actions),device=args.device))
                memory,_=model.aggregate([code]);features.extend(memory.cpu().numpy())
        diagnostics=[dict(memory_bytes=520) for _ in rows]
    labels=np.asarray([private_labels(bank,r,args.frames) for r in rows]);features=np.asarray(features)
    if not np.isfinite(features).all() or not np.isfinite(labels).all():raise ValueError('nonfinite frozen representation or target')
    np.savez_compressed(args.output/'FEATURES.private.npz',features=features,labels=labels,
        split=np.array([r['split'] for r in rows]),stratum=np.array([r['stratum'] for r in rows]),system_key=np.array([r['system_key'] for r in rows]),
        episode_key=np.array([r['episode_key'] for r in rows]))
    config.update(rows=len(rows),representation_dim=features.shape[1],seconds=time.monotonic()-start,
        feature_sha256=hashlib.sha256((args.output/'FEATURES.private.npz').read_bytes()).hexdigest(),diagnostics=diagnostics)
    after_content=verify(bank.root,snapshot)
    if stable_digest(snapshot_path)['sha256']!=snapshot_sha or source_fingerprint()!=source_before or (not args.explicit and stable_digest(args.checkpoint)['sha256']!=checkpoint_before):
        raise ValueError('formation input snapshot/source/checkpoint changed during extraction')
    config['bank_content_verification']=dict(snapshot_sha256=snapshot_sha,before=before_content,after=after_content)
    (args.output/'EXTRACTION.json').write_text(json.dumps(config,indent=2)+'\n');print(json.dumps({k:config[k] for k in ('method','rows','representation_dim','seconds')}),flush=True)


def train_standardize(values,train):
    if train.dtype!=bool or train.shape!=(len(values),) or not train.any():raise ValueError('explicit nonempty training mask required')
    mean=values[train].mean(0);sd=values[train].std(0);active=sd>1e-10;scale=np.where(active,sd,1.)
    return (values-mean)/scale,mean,scale,active


def grouped_scores(prediction,truth,systems,scale):
    unique=np.unique(systems);weights=np.zeros(len(systems))
    for key in unique:
        selected=systems==key;weights[selected]=1/(len(unique)*selected.sum())
    error=np.sum(weights[:,None]*(prediction-truth)**2,axis=0)
    mean=np.sum(weights[:,None]*truth,axis=0);variance=np.sum(weights[:,None]*(truth-mean)**2,axis=0)
    r2=[float(1-e/v) if v>1e-16 else None for e,v in zip(error,variance)]
    return dict(systems=len(unique),episodes=len(systems),mse=error.tolist(),train_standardized_mse=(error/scale**2).tolist(),r2=r2)


def fit_probes(args):
    import torch
    from torch import nn
    from .pixel_training import runtime_policy
    from .training_protocol import source_fingerprint
    from .dataset_snapshot import stable_digest
    before={name:stable_digest(args.features/name)['sha256'] for name in ('FEATURES.private.npz','EXTRACTION.json')};source_before=source_fingerprint()
    if args.output.exists():raise ValueError('probe attempt already exists')
    args.output.mkdir(parents=True);data=np.load(args.features/'FEATURES.private.npz',allow_pickle=False)
    if np.any(~np.isin(data['split'],['train','validation'])):raise ValueError('probe development fitter cannot read test bundles')
    train=data['split']=='train';select=(data['split']=='validation')&np.isin(data['stratum'],SELECTION_STRATA)
    if not select.any():raise ValueError('missing independent validation selection population')
    x,xmean,xscale,xactive=train_standardize(data['features'],train);y,ymean,yscale,yactive=train_standardize(data['labels'],train)
    x=x[:,xactive];xt=x[train];yt=y[train];xv=x[select];yv=y[select]
    gram=xt.T@xt/len(xt);rhs=xt.T@yt/len(xt);alphas=np.array([1e-5,1e-3,.1,1.,10.,100.]);weights=[];losses=[]
    for alpha in alphas:
        weight=np.linalg.solve(gram+alpha*np.eye(len(gram)),rhs);weights.append(weight);losses.append(np.mean((xv@weight-yv)**2,axis=0))
    chosen=np.argmin(losses,axis=0);weight=np.column_stack([weights[chosen[i]][:,i] for i in range(y.shape[1])])
    predictions={'ridge':(x@weight)*yscale+ymean}
    np.savez_compressed(args.output/'ridge.npz',weights=weight,xmean=xmean,xscale=xscale,xactive=xactive,ymean=ymean,yscale=yscale,alphas=alphas[chosen])
    runtime_policy(args.seed);torch.set_num_threads(1);model=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,y.shape[1]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=np.random.default_rng(args.seed)
    tx=torch.tensor(xt,dtype=torch.float32);ty=torch.tensor(yt,dtype=torch.float32);vx=torch.tensor(xv,dtype=torch.float32);vy=torch.tensor(yv,dtype=torch.float32)
    best=float('inf');best_state=None;trace=[];start=time.monotonic()
    for step in range(args.updates):
        idx=rng.integers(len(tx),size=min(128,len(tx)));model.train();optimizer.zero_grad(set_to_none=True)
        loss=(model(tx[idx])-ty[idx]).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if (step+1)%100==0 or step+1==args.updates:
            model.eval()
            with torch.no_grad():score=float((model(vx)-vy).square().mean())
            trace.append(dict(update=step+1,validation_loss=score))
            if score<best:best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():predictions['mlp']=model(torch.tensor(x,dtype=torch.float32)).numpy()*yscale+ymean
    torch.save(dict(model=best_state,input_dim=x.shape[1],output_dim=y.shape[1],xmean=xmean.tolist(),xscale=xscale.tolist(),xactive=xactive.tolist(),
        ymean=ymean.tolist(),yscale=yscale.tolist()),args.output/'mlp.pt')
    reports={}
    for method,pred in predictions.items():
        reports[method]={}
        for split in ('train','validation'):
            for stratum in np.unique(data['stratum'][data['split']==split]):
                mask=(data['split']==split)&(data['stratum']==stratum)
                reports[method][split+'/'+stratum]=grouped_scores(pred[mask],data['labels'][mask],data['system_key'][mask],yscale)
    config=dict(schema='vec.formation-probes.v1.1',features_sha256=hashlib.sha256((args.features/'FEATURES.private.npz').read_bytes()).hexdigest(),
        extraction_sha256=hashlib.sha256((args.features/'EXTRACTION.json').read_bytes()).hexdigest(),source_fingerprint=source_fingerprint(),
        artifacts={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ('ridge.npz','mlp.pt')},
        labels=LABELS,physical_targets=list(range(4)),nuisance_targets=list(range(4,10)),feature_dimension=x.shape[1],
        ridge='per-target alpha selected on independent validation systems; standardized train-only features and targets',ridge_alphas=alphas[chosen].tolist(),
        mlp=dict(hidden_widths=[128,128],updates=args.updates,seed=args.seed,lr=.001,weight_decay=.0001,parameters=sum(p.numel() for p in model.parameters()),selection_trace=trace),
        scores=reports,seconds=time.monotonic()-start,interpretation='decoder-specific linear/nonlinear decodability lower bounds; poor scores do not prove absence of information',
        formal_results=False,test_read=False)
    if before!={name:stable_digest(args.features/name)['sha256'] for name in before} or source_fingerprint()!=source_before:
        raise ValueError('formation fit inputs/source changed during optimization')
    (args.output/'PROBE_REPORT.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez_compressed(args.output/'PREDICTIONS.private.npz',**predictions)
    print(json.dumps(dict(status='EXECUTED',features=x.shape[1],train_rows=int(train.sum()),validation_selection_rows=int(select.sum()),seconds=config['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract');e.add_argument('--bank',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--checkpoint',type=Path);e.add_argument('--model-source',type=Path);e.add_argument('--explicit',action='store_true')
    e.add_argument('--frames',type=int,choices=(24,48,96),default=96);e.add_argument('--device',default='cuda:0');e.add_argument('--batch-size',type=int,default=32)
    e.add_argument('--random-init',action='store_true')
    e.add_argument('--workers',type=int,default=32);e.add_argument('--limit-per-system',type=int,default=0)
    f=sub.add_parser('fit');f.add_argument('--features',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--updates',type=int,default=2000);f.add_argument('--seed',type=int,default=17)
    a=p.parse_args();extract(a) if a.command=='extract' else fit_probes(a)


if __name__=='__main__':main()
