"""Actual fixed Query-only/Known heads; no source/probe is invented for refs."""
from pathlib import Path
import numpy as np
import torch
from runtime import VERSION, artifact, checked, load_core, read, sha, write


def parameter_vectors(scene, labels, gravity, mask):
    labels=np.asarray(labels,np.int64);mask=np.asarray(mask,np.float32)
    count=2 if scene=='blocktower' else 3
    if labels.shape[-1]!=(2 if scene=='blocktower' else 3) or np.any((labels<0)|(labels>=count)):
        raise ValueError('Known parameter category outside the frozen training encoding')
    vectors=np.eye(count,dtype=np.float32)[labels].reshape(*labels.shape[:-1],-1)
    if scene=='blocktower':
        g=np.asarray(gravity,np.float32)
        g=np.broadcast_to(g[:,None,:],(*labels.shape[:-1],2))
        vectors=np.concatenate((vectors,g),-1)
    return vectors*mask[...,None]


def freeze_reference(e,bind):
    rp=Path(e['results']);result=read(rp);prior=read(rp.parent/'checkpoint_freeze.json');done=read(rp.parent/'complete.json')
    kind='query' if e['reference']=='Query-only' else 'known'
    if result.get('status')!='COMPLETE' or result.get('test_read') is not False or result.get('reference')!=kind:
        raise ValueError('Uncommitted reference full-validation result')
    if done.get('results_sha256')!=sha(rp) or prior.get('head_budget')!=100 or prior.get('test_read') is not False:
        raise ValueError('Reference fixed-head receipt differs')
    head=bind(prior['checkpoint']);ck=torch.load(head['path'],map_location='cpu',weights_only=False)
    folder=Path(head['path']).parent;conf=read(folder/'config.json');selected=read(folder/'selected_validation.json')
    core=bind(e['core'])
    if core['sha256']!=prior['head_implementation_sha256'] or head['sha256']!=prior['checkpoint_sha256']:
        raise ValueError('Wrong reference implementation/checkpoint')
    if (ck['config']!=conf or conf['epochs']!=100 or conf['seed']!=0 or conf['supports']!=3 or conf['reference']!=kind
            or conf['test_read'] is not False or ck['epoch']!=selected['epoch'] or len(selected['ids'])!=512):
        raise ValueError('Reference is not the original validation512-selected head100')
    for path,h in conf['input_sha256'].items():
        if bind(path)['sha256']!=h:raise ValueError('Changed reference training/query input')
    transform=None
    if kind=='known':
        params=next(Path(p) for p in conf['input_sha256'] if Path(p).name=='parameters_train.npz')
        base=params.parent;manifest=read(base/'manifest.json');part=manifest['splits']['train'];ids=part['query_ids']
        labels=np.asarray([part['physical'][q] for q in ids],np.int64);mask=np.asarray([part['presence'][q] for q in ids],np.float32)
        gravity=[part['gravity'][q] for q in ids] if e['scene']=='blocktower' else None
        expected=parameter_vectors(e['scene'],labels,gravity,mask)
        with np.load(params,allow_pickle=False) as z:
            if not np.array_equal(z['values'],expected):raise ValueError('Known test encoder does not exactly reproduce stored train parameter inputs')
        transform=dict(kind='categorical_onehot_then_raw_gravity',class_count=2 if e['scene']=='blocktower' else 3,
            gravity_transform='none; original raw gx/gy',train_exact_verified=True,
            train_parameters=bind(params),train_manifest=bind(base/'manifest.json'))
    for p in (rp,rp.parent/'complete.json',rp.parent/'checkpoint_freeze.json',folder/'config.json',folder/'selected_validation.json',folder/'results.json',folder/'complete.json'):bind(p)
    return dict(e,core=core,head=head,config=conf,reference_kind=kind,selected_head_epoch=ck['epoch'],head_epochs=100,
                known_transform=transform,profile='shared-pose-q3-head100-task-reference',source_epochs=None,
                probe_required=False,attribution='Known is a finite trained parameter reference, not a guaranteed oracle upper bound')


def evaluate_reference(args,freeze,entry):
    bundle=read(args.bundle)
    if bundle['status']!='PREPARED' or bundle['split']!='test' or bundle['freeze_sha256']!=sha(args.freeze) or bundle['scene']!=entry['scene']:
        raise ValueError('Reference requires the same frozen test bundle')
    for item in bundle['files']:checked(item)
    out=Path(args.out);identity=dict(version=VERSION,entry=entry['id'],freeze_sha256=sha(args.freeze),bundle_sha256=sha(args.bundle),head_sha256=entry['head']['sha256'])
    write(out/'binding.json',identity,immutable=True)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json')
        if saved['binding_sha256']!=sha(out/'binding.json'):raise ValueError('Existing reference test binding differs')
        checked(saved['results']);print('FINAL_TEST_REFERENCE_ALREADY_COMPLETE');return
    core=load_core(checked(entry['core']));ck=torch.load(checked(entry['head']),map_location='cpu',weights_only=False)
    if ck['config']!=entry['config']:raise ValueError('Reference head configuration changed')
    with np.load(checked(bundle['inputs']['input']),allow_pickle=False) as z:row={k:z[k].copy() for k in z.files}
    with np.load(checked(bundle['inputs']['target']),allow_pickle=False) as z:target=z['pose'].copy()
    ids=list(map(str,row['ids']));n=len(ids);slots=row['pose'].shape[2];dims=row['pose'].shape[-1]
    detdims=1 if row['detected'].ndim==3 else row['detected'].shape[-1]
    head=core.Head(dims,detdims,entry['config']['support_dims'],target.shape[1]).to(args.device)
    head.load_state_dict(ck['model'],strict=True);head.eval();head.requires_grad_(False)
    params=None
    if entry['reference_kind']=='known':
        for item in (entry['known_transform']['train_parameters'],entry['known_transform']['train_manifest']):checked(item)
        records=read(checked(bundle['inputs']['raw_relations']));meta={(str(r['id']),int(r['slot'])):r for r in records}
        # The field auditor records real objects only. Padded color slots have
        # no metadata and retain the same masked zero-fill as training.
        labels=np.zeros((n,slots,2 if entry['scene']=='blocktower' else 3),np.int64)
        gravity=np.zeros((n,2),np.float32) if entry['scene']=='blocktower' else None
        for i,q in enumerate(ids):
            episode=[]
            for k in range(slots):
                record=meta.get((q,k))
                if record is None:
                    if row['presence'][i,k]>0:raise ValueError('Active known-parameter object has no audited metadata')
                    continue
                labels[i,k]=record['physical'];episode.append(record)
            if gravity is not None:
                unique={tuple(r['raw_gravity']) for r in episode}
                if len(unique)!=1:raise ValueError('Missing or inconsistent per-episode global gravity')
                gravity[i]=next(iter(unique))
        params=parameter_vectors(entry['scene'],labels,gravity,row['presence'])
    values=[];thirds=[]
    with torch.no_grad():
        for start in range(0,n,128):
            sl=slice(start,min(start+128,n));t=lambda v:torch.as_tensor(v,dtype=torch.float32,device=args.device)
            q,det,mask,y=t(row['pose'][sl]),t(row['detected'][sl]),t(row['presence'][sl]),t(target[sl])
            support=None if params is None else t(params[sl,:,None])
            pred=head(q,det,mask,support)
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite fixed reference prediction')
            values.extend(core.scores(pred,y,mask).cpu().tolist())
            thirds.extend(torch.stack([core.scores(pred[:,p],y[:,p],mask) for p in np.array_split(np.arange(target.shape[1]),3)],1).cpu().tolist())
    with np.load(checked(bundle['plans']),allow_pickle=False) as plans:eligible=np.flatnonzero(plans['correct_eligible'])
    def report(ix):return dict(mse=float(np.mean(np.asarray(values)[ix])) if len(ix) else None,recipients=len(ix),
        ids=[ids[i] for i in ix],per_recipient_mse=np.asarray(values)[ix].tolist())
    result=dict(identity,status='COMPLETE',stage='final_test',scene=entry['scene'],split='test',reference=entry['reference_kind'],
        family='reference',role=entry['reference'],arms=dict(matched=report(np.arange(n)),matched_on_learned_cohort=report(eligible)),
        thirds=np.mean(thirds,0).tolist(),
        official_test_count=n,correct_eligible_count=len(eligible),
        source_epochs=None,head_budget=100,selected_head_epoch=ck['epoch'],seed=0,probe_required=False,
        known_transform=entry['known_transform'],test_fit_steps=0,test_model_selection=False,optimizer_steps=0,test_read=True)
    write(out/'results.json',result,immutable=True)
    write(out/'complete.json',dict(status='COMPLETE',version=VERSION,binding_sha256=sha(out/'binding.json'),results=artifact(out/'results.json'),test_read=True),immutable=True)
    print('FINAL_TEST_REFERENCE_COMPLETE')
