"""Selected-checkpoint task, formation and P-only reuse evaluation.

No training, donor resampling, checkpoint selection or hyperparameter search is
performed here. Raw future states are read only as recipient scoring targets.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from cophy_adapter import ABObservation, VisualInput, Targets, PTCoPhy, PersistentNullBank, donor_assay
from cophy_protocol import digest, verify_preflight, verify_adapter_binding, ridge_probe
from cophy_prepare_artifacts import SPECS, checked_cache, frozen_write
from cophy_relations import read_artifact, artifact_path, ParameterFeatures
from cophy_training import mse_per_recipient


def chunks(rows, size):
    for start in range(0,len(rows),size): yield rows[start:start+size]


def observation(cache, ids, device):
    def tensor(key):
        return torch.from_numpy(np.stack([cache[ident][key] for ident in ids])).to(device)
    return VisualInput(ABObservation(tensor('pose_ab'),tensor('presence_ab')),
                       tensor('pose_c'),tensor('presence_c'))


def targets(profile, ids, device):
    from dataloaders.utils import get_pose_3D, get_stab
    poses=[];masks=[];stable=[]
    for ident in ids:
        root=Path(profile['dataset_dir'])
        if profile['scene']!='collision':root=root/str(profile['num_objects'])
        pose,mask=get_pose_3D(str(root/ident/'cd'))
        poses.append(pose[1:]);masks.append(mask);stable.append(get_stab(pose,mask)[1:])
    tensor=lambda x:torch.as_tensor(np.stack(x),dtype=torch.float32,device=device)
    return Targets(tensor(poses),tensor(stable),tensor(masks))


def classification(y,pred,n_classes):
    y=np.asarray(y,int);pred=np.asarray(pred,int)
    cm=np.zeros((n_classes,n_classes),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    actual=cm.sum(1); guessed=cm.sum(0);tp=cm.diagonal()
    supported=actual>0
    recall=np.divide(tp,actual,out=np.zeros(n_classes,float),where=supported)
    f1=np.divide(2*tp,actual+guessed,out=np.zeros(n_classes,float),where=actual+guessed>0)
    return {'balanced_accuracy':float(recall[supported].mean()),
        'macro_f1':float(f1[(actual+guessed)>0].mean()),'class_support':actual.tolist(),
        'confusion':cm.tolist(),'examples':len(y)}


def _prior(keys,y,eval_keys,n_classes):
    counts=defaultdict(lambda:np.zeros(n_classes,int))
    for key,label in zip(keys,y):counts[tuple(key)][label]+=1
    default=np.bincount(y,minlength=n_classes).argmax()
    return np.array([counts[tuple(key)].argmax() if tuple(key) in counts else default for key in eval_keys])


def representation_scatter(x,rows,include_gravity=False):
    """Descriptive conditional variance, not a rank or success gate."""
    x=np.asarray(x,float);strata=defaultdict(list);classes=defaultdict(list)
    for i,row in enumerate(rows):
        key=(row['slot'],row['known_type'])
        if include_gravity:key+=tuple(row['raw_gravity'])
        strata[key].append(i);classes[(key,tuple(row['physical']))].append(i)
    def residual(groups):
        return sum(float(((x[ix]-x[ix].mean(0))**2).sum()) for ix in groups.values())/len(x)
    total=residual(strata);within=residual(classes)
    return {'feature_std':x.std(0).tolist(),'global_variance_trace':float(x.var(0).sum()),
        'within_slot_type_variance_trace':total,'within_physical_class_variance_trace':within,
        'between_physical_class_variance_trace':max(0.,total-within),
        'gravity_also_conditioned':include_gravity,'physical_classes':len(classes),
        'singleton_classes':sum(len(ix)==1 for ix in classes.values()),
        'descriptive_only_not_a_collapse_gate':True}


def formation_report(train_codes,eval_codes,train_rows,eval_rows,fields,seed=20260911,include_gravity=False):
    """Frozen alpha=1; inner split is by experiment, never by object rows."""
    def arrays(codes, rows):
        rows=sorted((r for r in rows if r['id'] in codes and codes[r['id']][1][r['slot']]>0),
                    key=lambda r:(r['id'],r['slot']))
        x=np.stack([codes[r['id']][0][r['slot']] for r in rows])
        y=np.array([r['physical'] for r in rows],int)
        return rows,x,y,[(r['slot'],r['known_type']) for r in rows]
    tr,x,y,keys=arrays(train_codes,train_rows);ev,e,ey,ekeys=arrays(eval_codes,eval_rows)
    if set(r['id'] for r in tr)&set(r['id'] for r in ev): raise ValueError('Probe experiment split leak')
    ids=sorted({r['id'] for r in tr},key=lambda ident:hashlib.sha256(f'{seed}:{ident}'.encode()).digest())
    fit_ids=set(ids[:max(1,min(len(ids)-1,int(.8*len(ids))))])
    fit=np.array([r['id'] in fit_ids for r in tr]);inner=~fit
    if not fit.any() or not inner.any(): raise ValueError('Need independent train probe partitions')
    permutation=np.random.default_rng(seed).permutation(len(y))
    result={'alpha':1.,'seed':seed,'train_objects':len(tr),'eval_objects':len(ev),
            'inner_fit_experiments':len(fit_ids),'inner_eval_experiments':len(ids)-len(fit_ids),
            'fields':{},'joint_exact_match':{},'rows':[]}
    predictions={};views={'P':slice(0,16),'T':slice(16,32),'U':slice(0,32)}
    result['representation_scatter']={split:{view:representation_scatter(value[:,dims],rows,include_gravity)
        for view,dims in views.items()} for split,value,rows in [('train',x,tr),('eval',e,ev)]}
    for j,name in enumerate(fields):
        classes=int(max(y[:,j].max(),ey[:,j].max()))+1
        supported=np.unique(y[:,j]);is_varying=len(supported)>1
        record={'varying_in_train':is_varying,'train_class_support':np.bincount(y[:,j],minlength=classes).tolist(),
                'unseen_eval_classes':[int(v) for v in sorted(set(ey[:,j])-set(supported))]}
        # Values are stored for completeness; a constant attribute is not evidence of recovery.
        for view,dims in views.items():
            p=ridge_probe(x[:,dims],y[:,j],e[:,dims],classes)
            shuffled=ridge_probe(x[:,dims],y[permutation,j],e[:,dims],classes)
            diagnostic=ridge_probe(x[fit,dims],y[fit,j],x[inner,dims],classes)
            predictions[(view,name)]=p
            record[view]=dict(classification(ey[:,j],p,classes),
                inner=classification(y[inner,j],diagnostic,classes),
                train_label_permutation=classification(ey[:,j],shuffled,classes))
        prior=_prior(keys,y[:,j],ekeys,classes);predictions[('slot_type_prior',name)]=prior
        record['slot_type_prior']=classification(ey[:,j],prior,classes)
        result['fields'][name]=record
    for view in list(views)+['slot_type_prior']:
        p=np.stack([predictions[(view,f)] for f in fields],axis=1)
        result['joint_exact_match'][view]=float((p==ey).all(1).mean())
    if fields==['mass','friction']:
        # Dedicated Blocktower mass x friction four-class readout, not two separate argmaxes.
        joint=y[:,0]*2+y[:,1];ejoint=ey[:,0]*2+ey[:,1]
        result['blocktower_four_class']={view:classification(ejoint,
            ridge_probe(x[:,dims],joint,e[:,dims],4),4) for view,dims in views.items()}
    for i,row in enumerate(ev):
        result['rows'].append({'id':row['id'],'slot':row['slot'],'labels':ey[i].tolist(),
            'predicted':{view:[int(predictions[(view,f)][i]) for f in fields] for view in list(views)+['slot_type_prior']}})
    return result


@torch.no_grad()
def encode_cache(model,cache,device,batch_size):
    result={}
    for ids in chunks(sorted(cache),batch_size):
        v=observation(cache,ids,device)
        u=model.encode_ab(v.ab).cpu().numpy()
        for i,ident in enumerate(ids):result[ident]=(u[i],cache[ident]['presence_ab'])
    return result


@torch.no_grad()
def run(args):
    from cf_learning.model import CoPhyNet
    verify_preflight(args.preflight,release=args.release,require_release=args.split=='test')
    bound=verify_adapter_binding(args.preflight);scene=bound['scene'];spec=SPECS[scene]
    source=args.release if args.split=='test' else args.preflight
    # A sealed release contains bound test cache, metadata and both manifests.
    if args.split=='test':
        seal=json.loads(Path(source).read_text())
        if digest(args.checkpoint) not in seal.get('checkpoint_sha256',[]):
            raise ValueError('Test checkpoint was not selected before release')
        for item in seal['artifacts'].values():
            if digest(item['path'])!=item['sha256']:raise ValueError('Changed released test artifact')
    splits=read_artifact(args.preflight,'splits');device=torch.device(args.device)
    caches={split:checked_cache(artifact_path(source if split=='test' else args.preflight,f'cache_{split}'),
              splits[split]['ids'],spec) for split in ['train',args.split]}
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=state['run_config'];method=config['method'];checkpoint_sha=digest(args.checkpoint)
    if (config['data_binding']['preflight_sha256']!=digest(args.preflight) or
        config['code_sha256']!=bound['code_sha256'] or method not in bound['allowed_methods']):
        raise ValueError('Checkpoint configuration is not bound to the qualified run')
    parameters=None
    if method=='Param-known':
        parameters=ParameterFeatures(read_artifact(source,f'parameters_{args.split}'),scene,args.split)
        if parameters.schema!=config['parameter_schema']:raise ValueError('Parameter schema changed at evaluation')
    model=PTCoPhy(CoPhyNet(spec['slots']),method,len(parameters.fields) if parameters else None).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    # The checkpoint carries the frozen visual module; cached AB+C remains the actual input.
    official=[]
    for ids in chunks(splits[args.split]['ids'],args.batch_size):
        visual=observation(caches[args.split],ids,device);target=targets(bound['input_profile'],ids,device)
        params=parameters.batch(ids,spec['slots'],device) if parameters else None
        pred,presence,_=model.predict_code(model.code_for_task(visual,params),visual)
        copy=visual.c.expand(-1,pred.shape[1],-1,-1)
        scores={}
        for label,value in [('model',pred),('CopyC',copy)]:
            scores[label]=mse_per_recipient(value,target.pose,presence,2 if scene=='balls' else 3,True).cpu()
            scores[label+'_gtmask']=mse_per_recipient(value,target.pose,target.presence,2 if scene=='balls' else 3,True).cpu()
        for i,ident in enumerate(ids):official.append({'id':ident,**{
            name:float(value[i]) if torch.isfinite(value[i]) else None for name,value in scores.items()}})
    mean=lambda key:float(np.mean([r[key] for r in official if r[key] is not None]))
    result={'scene':scene,'split':args.split,'method':method,'seed':config['seed'],'epoch':state['epoch'],
        'checkpoint_sha256':checkpoint_sha,'preflight_sha256':digest(args.preflight),
        'official':{k:mean(k) for k in ['model','CopyC','model_gtmask','CopyC_gtmask']},
        'visual_coverage':sum(r['model'] is not None for r in official)/len(official),
        'official_rows':official,'test_read':args.split=='test','parameter_reference_is_upper_bound':False}
    if method!='Param-known':
        codes={s:encode_cache(model,cache,device,args.batch_size) for s,cache in caches.items()}
        raw={s:read_artifact(source if s=='test' else args.preflight,f'raw_relations_{s}') for s in caches}
        raw_audit=read_artifact(args.preflight,'raw_audit')
        audit=read_artifact(args.preflight,'audit')
        result['formation']=formation_report(codes['train'],codes[args.split],raw['train'],raw[args.split],raw_audit['fields'],
            include_gravity=audit.get('gravity_branch')=='varying_verified')
        null=PersistentNullBank(checkpoint_sha)
        # No absent GT slot or visually missed AB object enters a slot/type mean.
        active=defaultdict(dict)
        for row in raw['train']:active[row['id']][row['slot']]=row['known_type']
        for ids in chunks(sorted(codes['train']),args.batch_size):
            u=torch.from_numpy(np.stack([codes['train'][i][0] for i in ids]))
            masks=torch.zeros(len(ids),spec['slots']);keys=[]
            for b,ident in enumerate(ids):
                keys.append([(k,active[ident].get(k,'absent')) for k in range(spec['slots'])])
                for slot in active[ident]:masks[b,slot]=float(codes['train'][ident][1][slot]>0)
            null.update(u[...,:16],masks,keys,split='train')
        result['null_counts']={json.dumps(key):value for key,value in null.counts.items()}
        result['assays']={}
        for kind,artifact_name in [('primary',f'{"test" if args.split=="test" else "validation"}_primary'),
                                  ('wrong1',f'{"test" if args.split=="test" else "validation"}_wrong_1')]:
            manifest=read_artifact(source,artifact_name);rows=manifest['rows'];output=[]
            for batch in chunks(rows,args.batch_size):
                ids=[r['recipient'] for r in batch];visual=observation(caches[args.split],ids,device)
                target=targets(bound['input_profile'],ids,device)
                focal=torch.tensor([r['focal'] for r in batch],device=device,dtype=torch.long)
                donors={}
                for arm in ['Correct','Wrong-any']+(['Wrong-1'] if kind=='wrong1' else []):
                    donor_ids=[r[arm]['id'] for r in batch]
                    # AB-only donor construction. No donor target/data-loader call.
                    donor_ab=ABObservation(torch.from_numpy(np.stack([caches[args.split][i]['pose_ab'] for i in donor_ids])).to(device),
                        torch.from_numpy(np.stack([caches[args.split][i]['presence_ab'] for i in donor_ids])).to(device))
                    donors[arm]=(donor_ab,torch.tensor([r[arm]['slot'] for r in batch],device=device,dtype=torch.long))
                null_p=null.lookup([(r['focal'],r['known_type']) for r in batch],checkpoint_sha256=checkpoint_sha,device=device)
                scored=donor_assay(model,visual,target,focal,donors,null_p,2 if scene=='balls' else 3)
                for i,row in enumerate(batch):output.append(dict(row,errors={arm:{m:float(v[i]) for m,v in vals.items()}
                                                                          for arm,vals in scored.items()}))
            result['assays'][kind]={'manifest_sha256':digest(artifact_path(source,artifact_name)),
                'coverage':manifest.get('coverage'),'status':manifest.get('status','EVALUATED'),'rows':output}
    frozen_write(args.output,result)
    print(json.dumps({key:result[key] for key in ['scene','split','method','seed','epoch','official','visual_coverage']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['val','test'],default='val');parser.add_argument('--release')
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(args.threads);run(args)
