"""Fixed sealed-test inference on explicitly named TEST data; never fit/select.

Reuses the bound common Head and Data.raw_context/frozen_source computation.
No common.Data constructor is called and no test-as-val dictionary is created.
"""
import argparse
import fcntl
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from runtime import VERSION, artifact, checked, entry_by_id, load_core, physical_values, read, sha, verify_freeze, write


def encode(core, entry, bundle, out, device):
    dest=out/'encoding.json'
    identity=dict(freeze_entry=entry['id'],source_sha256=entry['source']['sha256'],bundle_sha256=sha(bundle['_path']))
    if dest.exists():
        old=read(dest)
        if old['identity']!=identity:raise ValueError('Different existing test encoding')
        for item in old['files']:checked(item)
        return old
    model,_=core.source_model(checked(entry['model_code']),checked(entry['source']),device)
    inputs=bundle['inputs']; arrays={k:np.load(checked(inputs[k]),mmap_mode='r',allow_pickle=False)
        for k in ('features_ab','features_c','presence_ab','presence_c')}
    count=len(bundle['query_ids']); binding=entry['encoding_binding']; axes=binding.get('history_batch_axes',{'hidden':1,'present':0})
    cache={};paths={};core_kind=binding['kind']
    for first in range(0,count,64):
        end=min(first+64,count)
        t=lambda x:torch.from_numpy(np.array(x[first:end],copy=True)).to(device)
        ab=t(arrays['features_ab']).float();am=t(arrays['presence_ab']).bool()
        current=t(arrays['features_c']).float();cm=t(arrays['presence_c']).bool()
        with torch.no_grad():
            if core_kind=='split':
                p=model.encode(ab,am);tokens=model.encode_current(current,cm)
                own=torch.cat((p,tokens),-1);null=torch.cat((torch.zeros_like(p),tokens),-1)
                values=dict(p=p,t=tokens)
            else:
                state=model.encode_history_state(ab,am);tokens=model.encode_current_tokens(current,cm)
                own=model.encode_joint_from_cached(state,tokens,cm)
                null=model.encode_joint_from_cached({k:torch.zeros_like(v) for k,v in state.items()},tokens,cm)
                if first==0:torch.testing.assert_close(model.encode_joint(ab,am,current,cm),own,atol=2e-5,rtol=2e-5)
                if core_kind=='cached':values={'state__'+k:v.movedim(axes.get(k,0),0) for k,v in state.items()}
                else:values={'hidden':state['hidden'].movedim(1,0),'present':state['present']}
            values.update(tokens=tokens,present=am.any(1),current_mask=cm,null_u=null,own_u=own)
        for name,value in values.items():
            v=value.detach().cpu().numpy();dtype=np.uint8 if v.dtype==np.bool_ else np.float32
            if name not in cache:
                path=out/(name+'.npy');paths[name]=path
                cache[name]=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=(count,*v.shape[1:]))
            cache[name][first:end]=v
    for value in cache.values():value.flush()
    del cache
    result=dict(status='COMPLETE',version=VERSION,identity=identity,cache={k:artifact(p) for k,p in paths.items()},
                files=[artifact(p) for p in paths.values()],history_batch_axes=axes,test_read=True,optimizer_steps=0)
    write(dest,result,immutable=True);return result


class TestData:
    def __init__(self,core,entry,bundle,encoding,device):
        self.args=SimpleNamespace(supports=3);self.binding=entry['encoding_binding'];self.source=None
        self.support_dims=entry['support_dims'];self.slots={'balls':9,'collision':4,'blocktower':4}[bundle['scene']]
        self.cache={'test':{k:np.load(checked(v),mmap_mode='r') for k,v in encoding['cache'].items()}}
        ids=bundle['query_ids'];self.cache_lut={'test':{q:i for i,q in enumerate(ids)}}
        self.history_index={'test':np.arange(len(ids),dtype=np.int64)}
        with np.load(checked(bundle['inputs']['input']),allow_pickle=False) as x:
            row=dict(ids=ids,q=x['pose'].copy(),det=x['detected'].copy(),mask=x['presence'].copy())
        with np.load(checked(bundle['inputs']['target']),allow_pickle=False) as y:row['target']=y['pose'].copy()
        self.data={'test':row};self.dims=row['q'].shape[-1];self.det_dims=1 if row['det'].ndim==3 else row['det'].shape[-1]
        self.horizon=row['target'].shape[1]
        norm=read(checked(entry['normalization']));self.code_mean=np.asarray(norm['mean'],np.float32);self.code_scale=np.asarray(norm['scale'],np.float32)
        self.raw_context=MethodType(core.Data.raw_context,self);self.frozen_source=MethodType(core.Data.frozen_source,self)
        with np.load(checked(bundle['plans']),allow_pickle=False) as plans:self.plans={k:plans[k].copy() for k in plans.files}

    def batch(self,ix,arm,device):
        row=self.data['test'];plan=None if arm=='null' else self.plans['wrong' if arm=='wrong' else 'correct'][ix]
        u,null=self.raw_context('test',ix,plan,device)
        mean=torch.as_tensor(self.code_mean,device=device);scale=torch.as_tensor(self.code_scale,device=device)
        t=lambda x:torch.as_tensor(np.asarray(x,dtype=np.float32),device=device)
        mask=t(row['mask'][ix]);support=dict(u=((u-mean)/scale)*mask[:,:,None,None],null_u=((null-mean)/scale)*mask[:,:,None,None],
            available=torch.ones(len(ix),self.slots,1,device=device)*(0. if arm=='null' else 1.))
        return t(row['q'][ix]),t(row['det'][ix]),mask,support,t(row['target'][ix])


@torch.no_grad()
def score(core,head,data,ix,arm,device):
    values=[];thirds=[];per_step=[]
    for start in range(0,len(ix),128):
        sub=ix[start:start+128];q,det,mask,support,y=data.batch(sub,arm,device);prediction=head(q,det,mask,support)
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite fixed test prediction')
        values.extend(core.scores(prediction,y,mask).cpu().tolist())
        pieces=np.array_split(np.arange(data.horizon),3)
        thirds.extend(torch.stack([core.scores(prediction[:,p],y[:,p],mask) for p in pieces],1).cpu().tolist())
        per_step.extend(torch.stack([core.scores(prediction[:,t:t+1],y[:,t:t+1],mask) for t in range(data.horizon)],1).cpu().tolist())
    return dict(mse=float(np.mean(values)) if values else None,recipients=len(values),
        ids=[data.data['test']['ids'][i] for i in ix],per_recipient_mse=values,
        thirds=np.mean(thirds,0).tolist() if thirds else [],per_step=np.mean(per_step,0).tolist() if per_step else [])


def probe_apply(fitfile,x,y,labels,names):
    with np.load(checked(fitfile),allow_pickle=False) as z:
        x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);labels=np.asarray(labels,np.int64)
        if len(x)==0:return dict(rows=0,fields={})
        prediction=((x-z['mean'])/z['scale'])@z['weight']+z['center'];result={}
        for j,name in enumerate(names):
            mse=float(np.square(prediction[:,j]-y[:,j]).mean());var=float(y[:,j].var())
            result[name]=dict(mse=mse,r2=1-mse/var if var else None,variance_denominator=var,
                              train_mean_mse=float(np.square(y[:,j]-z['raw_train_mean'][j]).mean()))
        offset=len(names)
        for j in range(int(z['classes_count'])):
            c=z['classes_'+str(j)];guess=c[prediction[:,offset:offset+len(c)].argmax(1)];actual=labels[:,j]
            result[names[j]].update(accuracy=float((guess==actual).mean()),
                balanced_accuracy=float(np.mean([(guess[actual==v]==v).mean() for v in np.unique(actual)])),
                train_majority_accuracy=float((actual==z['majority'][j]).mean()),
                unseen_test_label_count=int((~np.isin(actual,c)).sum()))
            offset+=len(c)
    return dict(rows=len(x),fields=result,fit_split='train',test_fit_steps=0)


@torch.no_grad()
def probes(head,data,entry,bundle,device):
    fit=read(checked(entry['probe_fit']));records=read(checked(bundle['inputs']['raw_relations']))
    meta={(str(r['id']),int(r['slot'])):r for r in records};ids=bundle['query_ids'];lookup={q:i for i,q in enumerate(ids)}
    channels={k:dict(x=[],y=[],labels=[]) for k in ('own','support_mean','memory')}
    def add(channel,vector,r):
        dest=channels[channel];dest['x'].append(vector);dest['y'].append(physical_values(r,bundle['scene']));dest['labels'].append(r['physical'])
    own=data.cache['test']['own_u'];visible=data.cache['test']['current_mask'].any(1)
    for r in records:
        ident,slot=str(r['id']),int(r['slot'])
        if r['in_C'] and visible[lookup[ident],slot]:add('own',own[lookup[ident],slot],r)
    ix=np.flatnonzero(data.plans['correct_eligible'])
    for start in range(0,len(ix),128):
        sub=ix[start:start+128];u,_=data.raw_context('test',sub,data.plans['correct'][sub],device)
        _,_,mask,support,_=data.batch(sub,'matched',device)
        memory=head.support(support['u']).mean(2).cpu().numpy();average=u.mean(2).cpu().numpy()
        for row,slot in zip(*np.where(data.data['test']['mask'][sub]>0)):
            r=meta[(ids[sub[row]],int(slot))];add('support_mean',average[row,slot],r);add('memory',memory[row,slot],r)
    return {k:dict(probe_apply(fit['channels'][k]['artifact'],**v,names=fit['fields']),
                   attribution=fit['channels'][k]['attribution'],fit_artifact=fit['channels'][k]['artifact']) for k,v in channels.items()}


def run(args):
    freeze=verify_freeze(args.freeze,verify_all=False)
    ref=[r for r in freeze['references'] if r['id']==args.entry]
    if ref:
        from references_adapter import evaluate_reference
        evaluate_reference(args,freeze,ref[0]);return
    entry=entry_by_id(freeze,args.entry)
    if entry['family']=='supervised':
        from supervised_adapter import evaluate as supervised_evaluate
        bundle=read(args.bundle);bundle['_path']=str(Path(args.bundle).resolve())
        if bundle['status']!='PREPARED' or bundle['split']!='test' or bundle['freeze_sha256']!=sha(args.freeze) or bundle['scene']!=entry['scene']:
            raise ValueError('No qualified supervised test bundle')
        for item in bundle['files']:checked(item)
        identity=dict(version=VERSION,entry=args.entry,freeze_sha256=sha(args.freeze),bundle_sha256=sha(args.bundle),
            source_sha256=entry['source']['sha256'],head_sha256=entry['head']['sha256'],probe_fit_sha256=entry['probe_fit']['sha256'])
        out=Path(args.out);write(out/'binding.json',identity,immutable=True)
        if (out/'complete.json').exists():
            saved=read(out/'complete.json')
            if saved['binding_sha256']!=sha(out/'binding.json'):raise ValueError('Existing supervised test result binding differs')
            checked(saved['results']);print('FINAL_TEST_SUPERVISED_ALREADY_COMPLETE');return
        result=supervised_evaluate(entry,bundle,out,args.device)
        write(out/'results.json',dict(result,**identity,status='COMPLETE',stage='final_test',split='test',scene=entry['scene'],
            family='supervised',role=entry['role'],source_epochs=100,source_selected_epoch=entry['source_selected_epoch'],
            head_budget=100,source_optimizer_steps=0,head_optimizer_steps=0,test_probe_fit_steps=0,test_read=True),immutable=True)
        write(out/'complete.json',dict(status='COMPLETE',version=VERSION,binding_sha256=sha(out/'binding.json'),
              results=artifact(out/'results.json'),test_read=True,optimizer_steps=0),immutable=True)
        print('FINAL_TEST_SUPERVISED_COMPLETE');return
    for key in ('core','source','model_code','head','probe_fit','normalization'):checked(entry[key])
    bundle=read(args.bundle);bundle['_path']=str(Path(args.bundle).resolve())
    if bundle['version']!=VERSION or bundle['status']!='PREPARED' or bundle['split']!='test' or bundle['freeze_sha256']!=sha(args.freeze) or bundle['scene']!=entry['scene']:
        raise ValueError('No qualified frozen test input bundle for this entry')
    for item in bundle['files']:checked(item)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    identity=dict(version=VERSION,entry=args.entry,freeze_sha256=sha(args.freeze),bundle_sha256=sha(args.bundle),
        source_sha256=entry['source']['sha256'],head_sha256=entry['head']['sha256'],probe_fit_sha256=entry['probe_fit']['sha256'])
    write(out/'binding.json',identity,immutable=True)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json')
        if saved['binding_sha256']!=sha(out/'binding.json'):raise ValueError('Existing test result binding differs')
        checked(saved['results']);print('FINAL_TEST_ALREADY_COMPLETE');return
    core=load_core(checked(entry['core']));encoding=encode(core,entry,bundle,out,args.device)
    data=TestData(core,entry,bundle,encoding,args.device)
    head=core.Head(data.dims,data.det_dims,data.support_dims,data.horizon).to(args.device)
    ck=torch.load(checked(entry['head']),map_location='cpu',weights_only=False);head.load_state_dict(ck['model'],strict=True)
    head.eval();head.requires_grad_(False)
    good=np.flatnonzero(data.plans['correct_eligible']);wrong=np.flatnonzero(data.plans['wrong_eligible'])
    arms={a:score(core,head,data,wrong if a=='wrong' else good,a,args.device) for a in ('matched','null','wrong')}
    arms['matched_on_wrong_cohort']=score(core,head,data,wrong,'matched',args.device)
    arms['null_on_wrong_cohort']=score(core,head,data,wrong,'null',args.device)
    result=dict(identity,scene=entry['scene'],family=entry['family'],role=entry['role'],stage='final_test',split='test',
        status='COMPLETE',source_epochs=100,head_budget=100,selected_head_epoch=ck['epoch'],seed=0,
        official_test_count=len(bundle['query_ids']),correct_eligible_count=len(good),wrong_eligible_count=len(wrong),
        unsupported_query_ids=bundle['unsupported_query_ids'],arms=arms,probes=probes(head,data,entry,bundle,args.device),
        source_optimizer_steps=0,head_optimizer_steps=0,test_probe_fit_steps=0,test_model_selection=False,test_read=True)
    write(out/'results.json',result,immutable=True)
    write(out/'complete.json',dict(status='COMPLETE',version=VERSION,binding_sha256=sha(out/'binding.json'),
        results=artifact(out/'results.json'),test_read=True,optimizer_steps=0),immutable=True)
    print('FINAL_TEST_COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','entry','bundle','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
    Path(a.out).mkdir(parents=True,exist_ok=True)
    with open(Path(a.out)/'worker.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)


if __name__=='__main__':main()
