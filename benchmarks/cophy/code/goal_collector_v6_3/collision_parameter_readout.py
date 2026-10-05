"""CPU diagnostics of existing Collision JEPA P64 and actual S3 head memory.

Only new parameter readouts are fitted, on official training episodes. Existing
source models and prediction heads stay frozen. Validation never selects a probe
checkpoint, capacity, or learning rate. No new source model is instantiated.
"""

import os
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

VERSION='collision-fixed-P64-parameter-readout-v2-audited-mapping'
SOURCE_VERSION='cophy-latent-v6.2-sig02'
CORE_VERSION='latent-relation-v6.2-frozen-P64-pose-prefix-readout'
METHODS=('Base','Cross','Align','Both','Random-Both')
FIELDS=('mass','friction','restitution')
COLORS=('yellow','green','blue','red')
TYPES=('sphere','cylinder_up','cylinder_down')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.pending.json');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)


def identity_keys(keys):return hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def load_core(path):
    spec=importlib.util.spec_from_file_location('frozen_collision_readout_for_parameter_diagnostic',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    require(core.VERSION==CORE_VERSION,'Unexpected frozen readout version')
    return core


def audited_rows(preflight):
    pre=read(preflight);arrays={};bindings={str(preflight):sha(preflight)}
    for split in ('train','val'):
        artifact=pre['artifacts']['raw_relations_'+split];p=Path(artifact['path'])
        require(sha(p)==artifact['sha256'],'Audited relation rows changed: '+split)
        bindings[str(p)]=artifact['sha256'];rows={}
        for r in read(p):
            require(r['split']==split,'Wrong split in physical labels')
            key=(str(r['id']),int(r['slot']))
            require(key not in rows,'Duplicate (episode, object-slot) physical row')
            require(0<=key[1]<4 and r['known_type'] in TYPES,'Unexpected Collision object slot/type')
            lab=np.asarray(r['physical'],np.int64);raw=np.asarray(r['raw_physical'],float)
            require(lab.shape==raw.shape==(3,) and (lab>=0).all() and np.isfinite(raw).all(),'Unexpected physical label dimensions/values')
            require(np.array_equal(lab,np.asarray(r['physical'])),'Physical categories must be integer IDs')
            rows[key]=r
        arrays[split]=rows
    # The committed audit is authoritative. Infer a single category -> raw
    # value map from TRAIN only; validation can check but never alter it.
    schema=[]
    for j,name in enumerate(FIELDS):
        mapping={};train_counts=Counter();val_counts=Counter()
        for r in arrays['train'].values():
            category=int(r['physical'][j]);value=float(r['raw_physical'][j])
            if category in mapping:
                require(value==mapping[category],
                        f'Training audit has non-unique {name} category {category}: {mapping[category]} versus {value}')
            else:mapping[category]=value
            train_counts[category]+=1
        require(bool(mapping),'Empty audited training support: '+name)
        for r in arrays['val'].values():
            category=int(r['physical'][j]);value=float(r['raw_physical'][j])
            require(category in mapping,f'Validation has unseen {name} category {category}')
            require(value==mapping[category],
                    f'Validation audit disagrees with training {name} category {category}: {mapping[category]} versus {value}')
            val_counts[category]+=1
        categories=sorted(mapping)
        schema.append(dict(field=name,category_ids=categories,raw_values=[mapping[c] for c in categories],
                           nominal_display_values=[.1 if mapping[c]==float(np.float16(.1)) else mapping[c] for c in categories],
                           train_rows=[train_counts[c] for c in categories],val_rows=[val_counts[c] for c in categories]))
    tr={k[0] for k in arrays['train']};va={k[0] for k in arrays['val']}
    require(not tr&va,'Train/val experiment IDs overlap; never split object rows randomly')
    return arrays,bindings,schema


def cache_for(args,method):
    source=args.run/'sources/collision/JEPA'/method
    out=args.run/'readouts/collision'/method/'source50'
    if not (source/'complete.json').exists():return None,'source50 not complete'
    done=read(source/'complete.json')
    require(all(done.get(k)==v for k,v in dict(status='COMPLETE',version=SOURCE_VERSION,scene='collision',family='JEPA',method=method,
                                              epochs=50,selected_epoch=50,test_read=False).items()),'Wrong source identity: '+method)
    if not (out/'codes_complete.json').exists():return None,'P64 cache not complete'
    marker=read(out/'codes_complete.json')
    require(all(marker.get(k)==v for k,v in dict(status='COMPLETE',version=CORE_VERSION,representation='P64',test_read=False).items()),'Wrong P64 cache identity')
    require(marker['source_sha256']==done['checkpoint_sha256'],'Source/code checkpoint binding differs')
    bindings={str(source/'complete.json'):sha(source/'complete.json'),str(out/'codes_complete.json'):sha(out/'codes_complete.json')}
    cache={}
    for split in ('train','val'):
        p=out/f'codes_{split}.npz';require(marker['files'][str(p)]==sha(p),'Frozen code cache changed')
        bindings[str(p)]=marker['files'][str(p)]
        with np.load(p,allow_pickle=False) as z:
            ids=list(map(str,z['ids']));code=z['p'].copy();seen=z['presence'].copy()
        require(code.shape==(len(ids),4,64) and seen.shape==(len(ids),4),'Expected four-slot P64 cache')
        require(len(set(ids))==len(ids) and np.isfinite(code).all(),'Invalid code IDs or values')
        cache[split]=dict(ids=ids,lookup={s:i for i,s in enumerate(ids)},p=code,presence=seen)
    require(not set(cache['train']['ids'])&set(cache['val']['ids']),'Frozen cache split IDs overlap')
    return dict(source=source,out=out,cache=cache,bindings=bindings),None


def fixed_cohort(manifest,raw,available):
    keys={}
    for split in ('train','val'):
        part=manifest['splits'][split];ids=part['query_ids'];selected=part.get('selection_query_ids',ids[:512])
        if split=='val':ids=selected
        require(split!='val' or len(ids)==512,'Diagnostic must use the existing fixed512 selection cohort')
        eligible=[]
        for ident in ids:
            for slot in range(4):
                key=(str(ident),slot);r=raw[split].get(key)
                if not r or not r['in_C'] or part['presence'][str(ident)][slot]<=0:continue
                if all(str(ident) in c['cache'][split]['lookup'] and c['cache'][split]['presence'][c['cache'][split]['lookup'][str(ident)],slot]>0 for c in available.values()):
                    eligible.append(key)
        require(bool(eligible),'Empty diagnostic object cohort')
        keys[split]=eligible
    require(not {k[0] for k in keys['train']}&{k[0] for k in keys['val']},'Object cohorts leak experiment IDs')
    return keys


def p_view(cache,keys):
    return {split:np.asarray([cache[split]['p'][cache[split]['lookup'][ident],slot] for ident,slot in keys[split]],np.float32)
            for split in ('train','val')}


def actual_memory(args,core,method,entry,keys,raw):
    folder=entry['out']/'S3/learned'
    if not (folder/'complete.json').exists():return None,dict(status='PENDING',reason='S3 prediction head100 not complete')
    done=read(folder/'complete.json');config=read(folder/'config.json');selected=read(folder/'selected_validation.json')
    require(done.get('status')=='COMPLETE' and done.get('epochs')==100,'Incomplete/wrong frozen prediction head')
    require(all(config.get(k)==v for k,v in dict(version=CORE_VERSION,scene='collision',reference='learned',supports=3,epochs=100,
                                                encoder_frozen=True,selection_rows=512,test_read=False).items()),'Wrong head config')
    require(config['code_sha256']==sha(entry['out']/'codes_complete.json'),'Frozen head/code binding differs')
    require(config['base_sha256']==sha(args.base/'manifest.json'),'Frozen head/base binding differs')
    ck=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    require(ck['config']==config and ck['epoch']==selected['epoch'],'Selected prediction head differs')
    data=core.Data(argparse.Namespace(scene='collision',base=str(args.base),out=str(entry['out']),reference='learned',supports=3))
    head=core.Head(data.dims,data.det_dims,data.support_dims,data.horizon)
    head.load_state_dict(ck['model'],strict=True);head.requires_grad_(False);head.eval()
    values={};concatenated={};plans={}
    with torch.inference_mode():
        for split in ('train','val'):
            row=data.data[split];plan=data.plan(split,0);lookup={ident:i for i,ident in enumerate(row['ids'])}
            all_ids=data.manifest['splits'][split]['all_ids'];memory=[];raw_support=[]
            for off in range(0,len(keys[split]),512):
                batch=keys[split][off:off+512];ri=np.asarray([lookup[ident] for ident,k in batch]);slots=np.asarray([k for ident,k in batch])
                donors=plan[ri,slots]
                for b,(ident,k) in enumerate(batch):
                    require(len(set(donors[b].tolist()))==3,'Repeated donor in S3 memory')
                    for donor in donors[b]:
                        did=str(all_ids[int(donor)]);r=raw[split].get((did,k));recipient=raw[split][(ident,k)]
                        require(did!=ident and r is not None and r['physical']==recipient['physical'] and r['known_type']==recipient['known_type'],
                                'Correct support violates independent matching or slot/type/physics labels')
                # Exact Head.forward route: normalize each donor P using Data
                # train statistics, support MLP per donor, then mean over S.
                support=row['codes'][donors,slots[:,None]]*row['mask'][ri,slots][:,None,None]
                raw_support.append(np.asarray(support,np.float32).reshape(len(batch),192))
                memory.append(head.support(torch.from_numpy(np.asarray(support,np.float32))).mean(1).numpy())
            values[split]=np.concatenate(memory)
            concatenated[split]=np.concatenate(raw_support)
            plans[split]=dict(epoch=0,sha256=hashlib.sha256(plan.tobytes()).hexdigest(),query_rows=len(row['ids']),supports=3)
    bindings={str(folder/name):sha(folder/name) for name in ('complete.json','config.json','selected.pt','selected_validation.json')}
    return dict(actual_memory64=values,support_P_concat192=concatenated),dict(status='COMPLETE',bindings=bindings,selected_epoch=ck['epoch'],plan=plans,
        definition='mean_s frozen_head.support((P_s - original_train_code_mean) / original_train_code_scale)',
        support_P_concat192_definition='Same three normalized donor P64 vectors flattened in fixed sampler order; this order is not temporal',
        position='64D support memory before concatenation with current query and head.init',source_and_head_frozen=True,
        prefix_and_future_targets_read_by_original_Data=True,future_targets_used_as_probe_inputs=False)


def labels_for(keys,raw,schema):
    dense=[{category:i for i,category in enumerate(s['category_ids'])} for s in schema]
    return {split:dict(values=np.asarray([raw[split][key]['raw_physical'] for key in keys[split]],np.float64),
                        labels=np.asarray([[dense[j][int(raw[split][key]['physical'][j])] for j in range(3)] for key in keys[split]],np.int64),
                        public=[(key[1],raw[split][key]['known_type']) for key in keys[split]]) for split in ('train','val')}


def metrics(values,labels,pred,classes,schema):
    result={}
    for j,name in enumerate(FIELDS):
        err=np.square(values[:,j]-pred[:,j]);variance=float(values[:,j].var());actual=labels[:,j];guess=classes[:,j]
        count=len(schema[j]['category_ids']);support=[int((actual==k).sum()) for k in range(count)]
        recall=[float((guess[actual==k]==k).mean()) if support[k] else None for k in range(count)]
        result[name]=dict(mse=float(err.mean()),target_variance=variance,r2=1-float(err.mean())/variance if variance>0 else None,
            accuracy=float((actual==guess).mean()),balanced_accuracy=float(np.mean([r for r in recall if r is not None])),
            class_values=schema[j]['raw_values'],category_ids=schema[j]['category_ids'],class_rows=support,per_class_recall=recall,object_rows=len(actual),
            classification_applicable=len([n for n in support if n])>=2)
    return result


def baselines(labels,schema):
    tr,va=labels['train'],labels['val'];mean=tr['values'].mean(0)
    majority=np.asarray([np.bincount(tr['labels'][:,j],minlength=len(schema[j]['category_ids'])).argmax() for j in range(3)])
    pred=np.tile(mean,(len(va['values']),1));classes=np.tile(majority,(len(va['values']),1))
    overall=metrics(va['values'],va['labels'],pred,classes,schema)
    for i,public in enumerate(va['public']):
        ix=np.asarray([p==public for p in tr['public']])
        if ix.any():
            pred[i]=tr['values'][ix].mean(0)
            classes[i]=[np.bincount(tr['labels'][ix,j],minlength=len(schema[j]['category_ids'])).argmax() for j in range(3)]
    return dict(train_mean_and_majority=overall,train_mean_and_majority_given_slot_public_type=metrics(va['values'],va['labels'],pred,classes,schema),
                classification_majority_ids=[schema[j]['category_ids'][int(majority[j])] for j in range(3)],no_episode_id_input=True)


def fit_readouts(args,views,labels,schema,extra_val=None):
    tr,va=labels['train'],labels['val'];mean=views['train'].mean(0);scale=views['train'].std(0).clip(1e-6)
    x=((views['train']-mean)/scale).astype(np.float64);xv=((views['val']-mean)/scale).astype(np.float64)
    ymean=tr['values'].mean(0);yscale=tr['values'].std(0).clip(1e-6)
    counts=[len(s['category_ids']) for s in schema];offsets=np.cumsum([3]+counts)
    onehot=np.concatenate([np.eye(counts[j])[tr['labels'][:,j]] for j in range(3)],1)
    target=np.concatenate(((tr['values']-ymean)/yscale,onehot),1);center=target.mean(0)
    dims=x.shape[1]
    weight=np.linalg.solve(x.T@x+np.eye(dims),x.T@(target-center));linear=xv@weight+center
    def unpack(a):return a[:,:3]*yscale+ymean,np.stack([a[:,offsets[j]:offsets[j+1]].argmax(1) for j in range(3)],1)
    lp,lc=unpack(linear);linear_metrics=metrics(va['values'],va['labels'],lp,lc,schema)
    torch.manual_seed(0);net=nn.Sequential(nn.Linear(dims,64),nn.GELU(),nn.Linear(64,target.shape[1]))
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    tx=torch.from_numpy(x.astype(np.float32));ty=torch.from_numpy(target.astype(np.float32));history=[]
    for epoch in range(1,args.epochs+1):
        order=np.random.default_rng(9000+epoch).permutation(len(tx));total=0.
        for off in range(0,len(order),512):
            ix=order[off:off+512];optimizer.zero_grad(set_to_none=True);loss=(net(tx[ix])-ty[ix]).square().mean()
            require(torch.isfinite(loss).item(),'Nonfinite parameter readout loss');loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=loss.item()*len(ix)
        if epoch==1 or epoch%10==0 or epoch==args.epochs:history.append(dict(epoch=epoch,train_mse=total/len(tx)))
    net.eval()
    with torch.inference_mode():
        nonlinear=net(torch.from_numpy(xv.astype(np.float32))).numpy();trainpred=net(tx).numpy()
    mp,mc=unpack(nonlinear);tp,tc=unpack(trainpred)
    result=dict(input_dims=dims,probe_seed=0,linear_ridge=dict(alpha=1.,validation=linear_metrics),
                small_MLP=dict(width=64,epochs=args.epochs,selected_epoch=args.epochs,selection='fixed last epoch, no validation feedback',
                    trainable_parameters=sum(p.numel() for p in net.parameters()),optimizer='AdamW',learning_rate=.001,weight_decay=.0001,batch_size=512,
                    validation=metrics(va['values'],va['labels'],mp,mc,schema),train=metrics(tr['values'],tr['labels'],tp,tc,schema),train_curve=history),
                matched_targets=f'3 train-standardized physical values + {sum(counts)} audited category one-hot columns; common squared loss',
                normalization='each view and physical-value target normalized from training objects only')
    if extra_val:
        xx=(extra_val['x'].astype(np.float64)-mean)/scale
        ep,ec=unpack(xx@weight+center)
        with torch.inference_mode():mp,mc=unpack(net(torch.from_numpy(xx.astype(np.float32))).numpy())
        result['fullval_single_P_supplement']=dict(cohort=extra_val['cohort'],
            linear_ridge=metrics(extra_val['labels']['values'],extra_val['labels']['labels'],ep,ec,schema),
            small_MLP=metrics(extra_val['labels']['values'],extra_val['labels']['labels'],mp,mc,schema),
            additional_probe_training=False)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--p-only',action='store_true');args=p.parse_args()
    require(args.epochs>0,'Positive fixed probe budget required');torch.set_num_threads(args.threads)
    args.run=args.root/'latent_v6_2_sigcal/weight02';args.base=args.root/'xep_discovery_collision_v4_4'
    prepath=args.root/'prepared_v3/collision/training_preflight.json';corepath=args.root/'source/latent_v6_2_sig02/readout.py'
    raw,binding,schema=audited_rows(prepath);manifest=read(args.base/'manifest.json')
    require(manifest.get('test_read') is False,'Test-containing base is not permitted')
    binding[str(args.base/'manifest.json')]=sha(args.base/'manifest.json');binding[str(corepath)]=sha(corepath)
    result=dict(version=VERSION,status='PARTIAL',scene='collision',source_family='JEPA',source_epochs=50,
        parameter_label_source='audited confounders.npy mapped to (experiment ID, fixed color slot)',
        colors=list(COLORS),physical_fields=list(FIELDS),audited_category_mapping=schema,public_type_values=TYPES,
        mapping_rule='Unique category-to-stored-raw mapping inferred from committed training relation rows; validation must match the same stored numeric value exactly',
        storage_note='The audited friction/restitution nominal 0.1 is stored as float16 value 0.0999755859375; nominal display is explanatory only and never used to define labels or regression targets',
        mapping_sha256=hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        repair=dict(supersedes='collision-fixed-P64-parameter-readout-v1',reason='Replaced nominal-decimal support comparison with audited stored-value mapping; previous execution stopped on float16 0.1 representation before probe fitting, not on category semantics',
                    frozen_source_or_prediction_head_changed=False),
        split_rule='Official train/val experiments are disjoint; no object-row random split; identity/color/type are not model inputs',
        object_interpretation='Each (experiment,slot) is one object instance; matching parameters across experiments do not establish persistent physical identity',
        methods={},bindings=binding,test_read=False,source_optimizer_steps=0,prediction_head_optimizer_steps=0,
        diagnostic_probe_epochs=args.epochs,validation_used_for_probe_tuning=False,script_sha256=sha(__file__))
    write(args.out/'audited_mapping.json',dict(version=VERSION,status='PASS',mapping=schema,bindings=binding,
        rule=result['mapping_rule'],storage_note=result['storage_note'],test_read=False,probe_fit_started=False))
    print(json.dumps(dict(event='audited_mapping_verified',version=VERSION,mapping=schema)),flush=True)
    available={}
    for method in METHODS:
        try:
            entry,pending=cache_for(args,method)
            if pending:result['methods'][method]=dict(status='PENDING',reason=pending)
            else:available[method]=entry
        except Exception as exc:result['methods'][method]=dict(status='CONTRADICTED',reason=str(exc))
    if not available:
        result['status']='WAITING_SOURCE_CODES';write(args.out/'summary.json',result);print(json.dumps(dict(status=result['status'])));return
    keys=fixed_cohort(manifest,raw,available);labels=labels_for(keys,raw,schema)
    fullkeys=[key for key,r in raw['val'].items() if r['in_C'] and all(key[0] in e['cache']['val']['lookup'] and
              e['cache']['val']['presence'][e['cache']['val']['lookup'][key[0]],key[1]]>0 for e in available.values())]
    fulllabels=labels_for(dict(train=keys['train'],val=fullkeys),raw,schema)['val']
    result['cohorts']={split:dict(object_rows=len(keys[split]),episode_groups=len({k[0] for k in keys[split]}),
        keys_sha256=identity_keys(keys[split]),keys=keys[split]) for split in ('train','val')}
    result['cohort_rule']='Intersection of active in_C object rows across available methods, restricted to common original query domain; val is fixed512 episodes'
    result['baselines']=baselines(labels,schema);core=None if args.p_only else load_core(corepath)
    for method,entry in available.items():
        began=time.time();r=dict(status='RUNNING',bindings=entry['bindings'],views={})
        try:
            cache=entry['cache']['val'];extra=dict(x=np.asarray([cache['p'][cache['lookup'][i],k] for i,k in fullkeys],np.float32),labels=fulllabels,
                cohort=dict(object_rows=len(fullkeys),episode_groups=len({k[0] for k in fullkeys}),keys_sha256=identity_keys(fullkeys)))
            r['views']['single_P64']=fit_readouts(args,p_view(entry['cache'],keys),labels,schema,extra_val=extra)
            oldprobe=entry['out']/'probes.json'
            if oldprobe.exists():
                old=read(oldprobe);require(old.get('source_codes_sha256')==sha(entry['out']/'codes_complete.json'),'Original probe source differs')
                r['existing_fullval_linear_probe']=dict(path=str(oldprobe),sha256=sha(oldprobe),train_objects=old.get('train_objects'),
                    val_objects=old.get('val_objects'),representations=old.get('representations'),
                    same_training_object_count=old.get('train_objects')==len(keys['train']))
                ours=r['views']['single_P64']['fullval_single_P_supplement']['linear_ridge'];oldfields=old.get('representations',{}).get('P',{})
                r['existing_fullval_linear_probe']['diagnostic_minus_existing_r2']={name:(ours[name]['r2']-oldfields[name]['r2'])
                    if ours[name]['r2'] is not None and oldfields.get(name,{}).get('r2') is not None else None for name in FIELDS}
            if not args.p_only:
                memory,evidence=actual_memory(args,core,method,entry,keys,raw);r['actual_memory_binding']=evidence
                if memory is not None:
                    for view in ('support_P_concat192','actual_memory64'):r['views'][view]=fit_readouts(args,memory[view],labels,schema)
            r['status']='COMPLETE' if args.p_only or r.get('actual_memory_binding',{}).get('status')=='COMPLETE' else 'P64_COMPLETE_MEMORY_PENDING'
        except Exception as exc:r['status']='DIAGNOSTIC_FAILURE';r['error']=repr(exc)
        r['seconds']=time.time()-began;result['methods'][method]=r
        write(args.out/(method+'.json'),dict(version=VERSION,method=method,**r));write(args.out/'summary.json',result)
        print(json.dumps(dict(method=method,status=r['status'],seconds=r['seconds'])),flush=True)
    result['status']='COMPLETE' if all(result['methods'][m]['status']=='COMPLETE' for m in METHODS) else 'PARTIAL'
    result['interpretation']=[
        'A higher nonlinear than linear parameter readout indicates accessible nonlinear parameter information, not proof that the prediction head uses it.',
        'Low scores from both finite-capacity readouts do not prove information is absent; train fit and public baselines delimit this diagnostic.',
        'Actual memory uses independent S3 histories and a task-trained support projection; single_P64 is the recipient own AB history. Their difference cannot isolate projection loss.',
        'support_P_concat192 and actual_memory64 share exactly the same independent S3 donors. Their readability difference diagnoses the head support path, but dimensions and probe parameter counts differ, so it is not a causal proof.',
        'Functional use still requires matched/null/wrong downstream intervention results on their proper cohorts; parameter readability is a separate measurement.',
        'All numbers are development validation. The frozen support projection was selected by earlier validation prediction MSE, not by these new parameter labels.']
    write(args.out/'summary.json',result)


if __name__=='__main__':main()
