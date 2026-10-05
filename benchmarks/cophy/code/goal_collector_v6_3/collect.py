"""Read-only, finite acceptance ledger for the approved CoPhy workload.

No GPU, model forward pass, training, checkpoint selection, or scientific pass
threshold. Metadata mode defers large-file hashes; --final verifies them once.
Every metric retains its recipient cohort and denominator. Missing work is not
a failed experiment. Existing negative results are valid completed results.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

VERSION = 'cophy-goal-artifact-collector-v1'
FULL_VERSION = 'latent-v6.2-fullval-fixed-P64-readout-1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7)


class Collector:
    def __init__(self, args, spec):
        self.args, self.spec = args, spec
        self.root = Path(args.root or spec['root'])
        self.hashes, self.documents, self.code_checks = {}, {}, {}
        self.rows, self.raw_results = [], {}

    def path(self, path):
        p = Path(path)
        declared = Path(self.spec['root'])
        if not p.is_relative_to(declared):
            raise ValueError('Path outside the declared experiment root: '+str(p))
        return self.root / p.relative_to(declared)

    def digest(self, path):
        p = self.path(path); st = p.stat(); key = (str(p), st.st_size, st.st_mtime_ns)
        if key not in self.hashes:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8<<20), b''): h.update(block)
            self.hashes[key]=h.hexdigest()
        return self.hashes[key]

    def doc(self, path, row=None, required=True):
        p=self.path(path)
        if not p.is_file():
            if row is not None and required: row['missing_files'].append(str(path))
            return None
        if str(p) not in self.documents:
            try: self.documents[str(p)]=json.loads(p.read_text())
            except (ValueError, OSError) as exc:
                if row is not None: row['contradictions'].append('Unreadable JSON '+str(path)+': '+str(exc))
                return None
        if row is not None: row['evidence'][str(path)] = self.digest(path)
        return self.documents[str(p)]

    def need(self, row, condition, reason):
        if not condition: row['contradictions'].append(reason)

    def fields(self, row, d, expected, label):
        if d is None: return
        for key, value in expected.items():
            self.need(row, d.get(key)==value, f'{label}.{key}: expected {value!r}, found {d.get(key)!r}')

    def bound(self, row, path, expected):
        if not expected or not isinstance(expected, str) or len(expected)!=64:
            self.need(row, False, 'Absent or invalid SHA256 for '+str(path)); return
        p=self.path(path)
        if not p.is_file(): row['missing_files'].append(str(path)); return
        # Checkpoint and NPZ/NPY assets are read only in final mode. Code and
        # JSON receipts are small and checked immediately, even across runs.
        if not self.args.final and (p.suffix in ('.pt','.pth','.npy','.npz') or p.stat().st_size > 8<<20):
            row['deferred_hashes'][str(path)] = expected; return
        self.need(row, self.digest(path)==expected, 'SHA256 differs: '+str(path))

    def filemap(self, row, files):
        if not isinstance(files,dict): self.need(row,False,'Missing bound-file map'); return
        for p,h in files.items(): self.bound(row,p,h)

    def identity(self, row, d, label, version=None):
        expected={'status':'COMPLETE','test_read':False}
        if version: expected['version']=version
        self.fields(row,d,expected,label)

    def checkpoint(self,row,path,expected,config=None):
        if not self.args.final:
            row['deferred_checkpoint_metadata'].append(str(path));return
        if not self.path(path).is_file(): return
        import torch
        ck=torch.load(self.path(path),map_location='cpu',weights_only=False)
        self.fields(row,ck,expected,'checkpoint')
        if config is not None: self.need(row,ck.get('config')==config,'Selected checkpoint config differs')
        self.need(row,isinstance(ck.get('model'),dict) and bool(ck['model']),'Checkpoint has no model state')

    def budget_history(self,row,folder,epochs):
        progress=self.doc(Path(folder)/'progress.json',row)
        if progress:
            self.fields(row,progress,dict(epoch=epochs),'final progress')
            history=progress.get('history',[])
            self.need(row,[x.get('epoch') for x in history]==list(range(1,epochs+1)),
                      'Epoch history does not cover the complete fixed budget')

    def arm(self,row,d,label,expected_rows=None):
        if not isinstance(d,dict): self.need(row,False,'Missing arm '+label);return
        ids=d.get('ids',[]); values=d.get('per_recipient_mse',[]); n=d.get('recipients')
        self.need(row,len(ids)==len(values)==n and len(set(ids))==len(ids),label+' recipient IDs/counts differ')
        if expected_rows is not None: self.need(row,n==expected_rows,label+' unexpected cohort size')
        self.need(row,all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values),label+' invalid MSE values')
        if values: self.need(row,close(sum(values)/len(values),d.get('mse')),label+' mean differs from per-recipient MSE')
        elif n==0: self.need(row,d.get('mse') is None,label+' nonempty mean for empty cohort')
        row.setdefault('metrics',{})[label]=dict(mse=d.get('mse'),recipients=n,
            ids_sha256=hashlib.sha256(canonical(ids)).hexdigest(),aggregation='equal recipient mean',thirds=d.get('thirds'))

    def arms(self,row,d,expected_rows,learned):
        self.arm(row,d.get('matched'),'matched',expected_rows)
        if not learned: return
        self.arm(row,d.get('null'),'null',expected_rows)
        self.arm(row,d.get('wrong'),'wrong')
        self.arm(row,d.get('matched_on_wrong_cohort'),'matched_on_wrong_cohort')
        self.arm(row,d.get('null_on_wrong_cohort'),'null_on_wrong_cohort')
        ids=d.get('matched',{}).get('ids',[]); wrong=d.get('wrong',{}).get('ids',[])
        self.need(row,ids==d.get('null',{}).get('ids'),'Matched/Null cohort differs')
        self.need(row,set(wrong)<=set(ids),'Wrong cohort is not a subset of Matched')
        for arm in ('matched_on_wrong_cohort','null_on_wrong_cohort'):
            self.need(row,d.get(arm,{}).get('ids')==wrong,arm+' differs from Wrong cohort')
        m=d.get('matched',{}).get('mse'); n=d.get('null',{}).get('mse')
        if n and m is not None:
            gain=100*(n-m)/n
            self.need(row,close(gain,d.get('history_gain_percent')),'History gain denominator/result differs')
            row['history_gain']=dict(percent=gain,numerator=n-m,denominator=n,cohort='Matched=Null',rows=expected_rows)
        w=d.get('wrong',{}).get('mse'); mw=d.get('matched_on_wrong_cohort',{}).get('mse')
        if w and mw is not None:
            row['correct_vs_wrong']=dict(percent=100*(w-mw)/w,numerator=w-mw,denominator=w,cohort='Wrong eligible',rows=len(wrong))
            if 'correct_vs_wrong_percent' in d:
                self.need(row,close(row['correct_vs_wrong']['percent'],d['correct_vs_wrong_percent']),'Correct/Wrong denominator differs')

    def source(self,row,s):
        folder=Path(s['folder']); done=self.doc(folder/'complete.json',row)
        if not done: return
        self.identity(row,done,'source',self.spec['source_versions'][s['family']])
        self.fields(row,done,dict(scene=s['scene'],family=s['family'],method=s['method'],epochs=50,selected_epoch=50,
                                coordinate_labels_read=False),'source')
        binding=self.doc(folder.parent/'binding.json',row); initial=self.doc(folder.parent/'initialization.json',row)
        if binding:
            body={k:v for k,v in binding.items() if k!='sha256'}
            bh=hashlib.sha256(canonical(body)).hexdigest()
            self.need(row,binding.get('sha256')==bh==done.get('binding_sha256'),'Source canonical binding SHA differs')
            self.fields(row,binding,dict(version=self.spec['source_versions'][s['family']],scene=s['scene'],epochs=50,test_read=False),'binding')
            self.filemap(row,binding.get('files'))
        if initial: self.need(row,initial.get('state_sha256')==done.get('initialization_sha256'),'Source initialization differs')
        ck=folder/'checkpoint_50.pt'
        self.need(row,done.get('checkpoint')==str(ck),'Source checkpoint path differs')
        self.bound(row,ck,done.get('checkpoint_sha256'))
        self.checkpoint(row,ck,dict(version=done.get('version'),epoch=50,scene=s['scene'],family=s['family'],method=s['method'],
                                   binding_sha256=done.get('binding_sha256'),initialization_sha256=done.get('initialization_sha256'),test_read=False))
        worker=self.doc(folder/'worker.json',row)
        if worker: self.fields(row,worker,dict(status='COMPLETE',exit_code=0),'source worker')
        self.budget_history(row,folder,50)
        row['budget']=dict(source_epochs=50,steps=done.get('steps'),checkpoint_epoch=50)

    def codes(self,row,s):
        readout=Path(s.get('readout',s['folder'])); marker=readout/'codes_complete.json'
        d=self.doc(marker,row)
        if not d:return
        self.identity(row,d,'codes',self.spec['core_versions'][s['family']]);self.fields(row,d,{'representation':'P64'},'codes')
        src=self.doc(Path(s['source'])/'complete.json',row)
        if src:self.need(row,d.get('source_sha256')==src.get('checkpoint_sha256'),'Codes source checkpoint hash differs')
        self.filemap(row,d.get('files'))
        feature_root=Path(d.get('feature_root',str(Path(self.spec['root'])/'latent_v6/features'/s['scene'])))
        for split,bindings in d.get('feature_binding',{}).items():
            self.bound(row,feature_root/split/'COMPLETE.json',bindings.get('complete_sha256'))
            self.bound(row,feature_root/split/'ids.json',bindings.get('ids_sha256'))
            fc=self.doc(feature_root/split/'COMPLETE.json',row)
            if fc:
                self.fields(row,fc,dict(status='COMPLETE',scene=s['scene'],split=split,test_read=False,optimizer_steps=0),'feature cache')
                self.bound(row,feature_root/split/'manifest.json',fc.get('manifest_sha256'))
                self.bound(row,feature_root/split/'chunks.jsonl',fc.get('chunks_sha256'))
                for info in fc.get('files',{}).values():
                    p=feature_root/split/info['path']
                    self.need(row,self.path(p).is_file() and self.path(p).stat().st_size==info['bytes'],'Feature asset missing or size differs: '+str(p))
        return d

    def probe(self,row,s):
        self.codes(row,s)
        d=self.doc(Path(s['folder'])/'probes.json',row)
        if not d:return
        self.identity(row,d,'probe',self.spec['core_versions'][s['family']]);self.fields(row,d,dict(scene=s['scene']),'probe')
        self.bound(row,Path(s['folder'])/'codes_complete.json',d.get('source_codes_sha256'))
        prepath=Path(self.spec['root'])/'prepared_v3'/s['scene']/'training_preflight.json'
        if s['scene']=='blocktower':
            profiles=self.doc(Path(self.spec['root'])/'runtime_profiles.json',row)
            if profiles:
                pre=profiles['scenes']['blocktower']['training_preflight'];prepath=Path(pre['path'])
                self.bound(row,prepath,pre['sha256'])
        self.bound(row,prepath,d.get('preflight_sha256'))
        self.need(row,set(d.get('representations',{}))=={'P'},'Expected frozen P64 probe only')
        self.need(row,d.get('train_objects',0)>0 and d.get('val_objects',0)>0,'Probe object cohorts empty')
        for field,metrics in d.get('representations',{}).get('P',{}).items():
            for key,value in metrics.items():
                if isinstance(value,(int,float)):
                    self.need(row,math.isfinite(value),'Nonfinite saved probe metric '+field+'/'+key)
        row['probe']=dict(fields=d.get('fields'),representations=d.get('representations'),train_objects=d.get('train_objects'),val_objects=d.get('val_objects'))

    def head(self,row,s):
        folder=Path(s['folder']); learned=s['reference']=='learned'
        if learned:self.codes(row,s)
        complete=self.doc(folder/'complete.json',row); config=self.doc(folder/'config.json',row)
        if not complete or not config:return
        self.identity(row,complete,'head');self.fields(row,complete,dict(epochs=100),'head')
        self.fields(row,config,dict(version=self.spec['core_versions'][s['family']],scene=s['scene'],reference=s['reference'],supports=s['supports'],
                                  epochs=100,selection_rows=512,validation_rows=512,encoder_frozen=True,seed=0,test_read=False),'head config')
        self.bound(row,Path(s['base'])/'manifest.json',config.get('base_sha256'));self.filemap(row,config.get('input_sha256'))
        base=self.doc(Path(s['base'])/'manifest.json',row)
        if base:self.fields(row,base,dict(test_read=False),'head base manifest')
        if learned:self.bound(row,Path(s['readout'])/'codes_complete.json',config.get('code_sha256'))
        result=self.doc(folder/'results.json',row); selected=self.doc(folder/'selected_validation.json',row)
        if not result or not selected:return
        self.identity(row,result,'head results');self.need(row,result.get('config')==config,'Head results config differs')
        epoch=result.get('selected_epoch');self.need(row,isinstance(epoch,int) and 1<=epoch<=100,'Invalid head selected epoch')
        self.need(row,epoch==selected.get('epoch'),'Head selected receipts disagree')
        self.arm(row,selected,'selection',512);self.arms(row,result,512,learned)
        if base:
            part=base['splits']['val'];ids=part.get('selection_query_ids',part['query_ids'][:512])
            self.need(row,selected.get('ids')==ids,'Head selected cohort differs from bound manifest')
        self.need(row,selected.get('ids')==result.get('selection',{}).get('ids'),'Head selection IDs differ')
        self.need(row,selected.get('per_recipient_mse')==result.get('selection',{}).get('per_recipient_mse'),'Head selection values differ')
        self.need(row,close(complete.get('mse'),result.get('matched',{}).get('mse')),'Head completion MSE differs')
        if not self.path(folder/'selected.pt').is_file():row['missing_files'].append(str(folder/'selected.pt'))
        self.checkpoint(row,folder/'selected.pt',dict(epoch=epoch),config)
        self.budget_history(row,folder,100)
        row['budget']=dict(source_epochs=50 if learned else None,head_epochs=100,selected_epoch=epoch,selection_rows=512)
        self.raw_results[row['id']]=result

    def fullval(self,row,s):
        folder=Path(s['folder']); h=Path(s['readout'])/f"S{s['supports']}"/s['reference'];learned=s['reference']=='learned'
        complete=self.doc(folder/'complete.json',row);result=self.doc(folder/'results.json',row);freeze=self.doc(folder/'checkpoint_freeze.json',row)
        if not complete or not result or not freeze:return
        self.identity(row,complete,'fullval complete',FULL_VERSION);self.identity(row,result,'fullval results',FULL_VERSION)
        for d,label in ((complete,'fullval complete'),(result,'fullval results')):
            self.fields(row,d,dict(reference=s['reference'],supports=s['supports'],full_validation_rows=self.spec['full_rows'][s['scene']],optimizer_steps=0),label)
        self.fields(row,result,dict(scene=s['scene'],selection_rows=512),'fullval results')
        self.bound(row,folder/'results.json',complete.get('results_sha256'));self.bound(row,folder/'checkpoint_freeze.json',result.get('checkpoint_freeze_sha256'))
        self.fields(row,freeze,dict(version=FULL_VERSION,scene=s['scene'],reference=s['reference'],supports=s['supports'],head_budget=100,
                                  test_read=False,optimizer_steps=0,encoder_frozen=True,head_frozen=True),'fullval freeze')
        self.need(row,freeze.get('checkpoint')==str(h/'selected.pt'),'Fullval selected checkpoint path differs')
        for p,key in ((h/'selected.pt','checkpoint_sha256'),(h/'selected_validation.json','selected_receipt_sha256'),
                      (h/'results.json','original_results_sha256'),(Path(s['prepared'])/'prepared.json','prepared_sha256'),
                      (Path(s['core']),'head_implementation_sha256'),(Path(s['evaluator']),'evaluation_implementation_sha256')):
            self.bound(row,p,freeze.get(key))
        if learned:self.bound(row,Path(s['readout'])/'codes_complete.json',freeze.get('source_codes_sha256'))
        prepared=self.doc(Path(s['prepared'])/'prepared.json',row);old=self.doc(h/'results.json',row);selected=self.doc(h/'selected_validation.json',row)
        if prepared:
            self.identity(row,prepared,'prepared',FULL_VERSION);self.fields(row,prepared,dict(scene=s['scene'],optimizer_steps=0),'prepared')
            self.filemap(row,prepared.get('files'))
            self.need(row,result.get('wrong_donor_domain')==prepared.get('wrong_donor_domain'),'Wrong donor domain changed')
            self.need(row,len(prepared.get('selection_ids',[]))==512,'Prepared selection is not 512')
            self.need(row,result['matched']['ids']==prepared.get('query_ids'),'Fullval recipient domain differs from prepared list')
        if old and selected:
            epoch=selected.get('epoch')
            for d in (complete,result,freeze,old):self.need(row,d.get('selected_epoch')==epoch,'Fullval checkpoint reselection or receipt mismatch')
            for name in ('selected','matched','null','wrong') if learned else ('selected','matched'):
                r=result.get('reproduction',{}).get(name,{})
                prior=selected if name=='selected' else old.get(name,{})
                self.need(row,r.get('matched_ids_exact') is True and r.get('recipients')==prior.get('recipients'),'Original512 reproduction cohort missing: '+name)
                self.need(row,isinstance(r.get('max_absolute_error'),(int,float)) and math.isfinite(r['max_absolute_error']),'Original512 reproduction error missing: '+name)
                # Recompute identity from saved full values; this is receipt
                # verification, not a new inference/evaluation pass.
                if name!='selected':
                    vals=dict(zip(result.get(name,{}).get('ids',[]),result.get(name,{}).get('per_recipient_mse',[])))
                    good=all(q in vals and abs(vals[q]-v)<=2e-5+2e-5*abs(v) for q,v in zip(prior.get('ids',[]),prior.get('per_recipient_mse',[])))
                    self.need(row,good,'Full values do not reproduce selected512: '+name)
        self.arms(row,result,self.spec['full_rows'][s['scene']],learned)
        if 'plan_sha256' in result:self.bound(row,folder/'plans.npz',result['plan_sha256'])
        if s['family']!='JEPA':
            family=self.doc(folder/'family_complete.json',row);fb=self.doc(folder/'family_binding.json',row)
            if family and fb:
                self.identity(row,family,'family complete')
                for d in (family,fb):
                    self.fields(row,d,dict(family=s['family'],family_core_version=self.spec['core_versions'][s['family']],scene=s['scene'],supports=s['supports'],
                                          source_checkpoint_epoch=50,head_budget_epochs=100,test_read=False,optimizer_steps=0,
                                          representation='P64',reference='learned',adapter_version='cophy-cpc-rssm-family-fullval-v6.3-1',
                                          common_eval_version=FULL_VERSION),'family binding')
                for p,key in ((folder/'family_binding.json','family_binding_sha256'),(folder/'complete.json','common_complete_sha256'),
                              (folder/'results.json','results_sha256'),(folder/'checkpoint_freeze.json','checkpoint_freeze_sha256')):
                    self.bound(row,p,family.get(key))
                for pkey,hkey in (('family_core','family_core_sha256'),('common_eval','common_eval_sha256'),('wrapper','wrapper_sha256')):
                    self.bound(row,fb[pkey],fb.get(hkey))
        row['budget']=dict(source_epochs=50 if learned else None,head_epochs=100,selected_epoch=result.get('selected_epoch'),selection_rows=512,full_rows=self.spec['full_rows'][s['scene']])
        self.raw_results[row['id']]=result

    def newrow(self,s):
        return dict(id=':'.join(str(s.get(k,'')) for k in ('kind','family','scene','method','supports')),kind=s['kind'],
            family=s.get('family'),scene=s.get('scene'),method=s.get('method'),supports=s.get('supports'),folder=s['folder'],
            verification_scope='bound-file hashes and checkpoint metadata' if self.args.final else 'metadata and small-file hashes',
            contradictions=[],missing_files=[],deferred_hashes={},deferred_checkpoint_metadata=[],evidence={})

    def finish(self,row,claimed,exists):
        row['missing_files']=sorted(set(row['missing_files']))
        row['status']='contradicted' if row['contradictions'] or (claimed and row['missing_files']) else (
            'verified' if claimed and not row['missing_files'] else ('incomplete' if exists else 'missing'))
        row['final_verified']=row['status']=='verified' and self.args.final and not row['deferred_hashes'] and not row['deferred_checkpoint_metadata']
        self.rows.append(row)

    def collect_item(self,s):
        row=self.newrow(s);folder=self.path(s['folder']);marker=self.doc(s['marker'],required=False)
        claimed=bool(marker and marker.get('status')=='COMPLETE')
        try:
            manifest=self.doc(s['manifest'],row) if s.get('manifest') else None
            if manifest:
                task=next((t for t in manifest.get('tasks',[]) if t.get('id')==s['task']),None)
                self.need(row,task is not None and task.get('marker')==s['marker'],'Task not found in original manifest with the same marker')
                queue=Path(s['manifest']).parent
                if s['manifest'].endswith('fullval_tail_manifest.json'):queue=queue/'fullval_tail'
                job=self.doc(queue/'jobs'/s['task']/'status.json',row,required=False)
                failure=self.doc(queue/'jobs'/s['task']/'failure.json',row,required=False)
                if failure:row['runtime_failure']=failure
                if claimed and job and job.get('status')=='COMPLETE' and 'exit_code' in job:
                    self.need(row,job['exit_code']==0,'Completed queue task has a nonzero exit')
            if claimed:
                {'source':self.source,'probe':self.probe,'head':self.head,'reference':self.head,
                 'fullval':self.fullval,'reference_fullval':self.fullval}[s['kind']](row,s)
            else:
                progress=self.doc(Path(s['folder'])/'progress.json',row,required=False)
                if progress:row['progress']={k:progress[k] for k in ('status','epoch','step','best') if k in progress}
                failure=self.doc(Path(s['folder'])/'failure.json',row,required=False)
                if failure:row['runtime_failure']=failure
        except Exception as exc:row['contradictions'].append(type(exc).__name__+': '+str(exc))
        self.finish(row,claimed,folder.exists())

    def legacy(self,s):
        row=self.newrow(dict(kind='legacy_tail',family='supervised',method='all',**s));folder=Path(s['folder'])
        name='controller_status.json' if s['scene']=='blocktower' else 'summary.json'
        d=self.doc(folder/name,row,required=False)
        if s['scene']=='blocktower':
            # Method-level complete+validation files are authoritative even
            # if a historical controller receipt uses another filename.
            done=[];row['methods']={}
            for method in s['methods']:
                f=folder/'runs'/method;c=self.doc(f/'complete.json',row);v=self.doc(f/'validation.json',row)
                if not c or not v:continue
                self.identity(row,c,'legacy Blocktower '+method,s['version']);self.fields(row,c,dict(method=method,optimizer_steps=0),'legacy Blocktower')
                self.bound(row,f/'validation.json',c.get('result_sha256'));self.bound(row,c['checkpoint'],c.get('checkpoint_sha256'))
                cache=c.get('target_cache',{});self.fields(row,cache,dict(status='COMPLETE',rows=8088,test_read=False,optimizer_steps=0),'Blocktower target cache')
                if method!='Param-known':
                    self.need(row,isinstance(v.get('formation'),dict) and bool(v['formation']),'Legacy formation missing '+method)
                    for assay in ('primary','wrong1'):
                        self.fields(row,v.get('assays',{}).get(assay,{}),dict(status='EVALUATED',recipient_groups=8088),'Legacy '+method+' '+assay)
                row['methods'][method]=dict(epoch=c.get('epoch'),official=c.get('official'),formation=v.get('formation'),assays=v.get('assays'))
                done.append(method)
            row['expected_methods']=s['methods'];row['observed_methods']=done
            claimed=len(done)==len(s['methods'])
        else:
            if d:
                self.identity(row,d,'legacy summary',s['version']);self.fields(row,d,dict(head_budget=100,optimizer_steps=0),'legacy summary')
                self.fields(row,d.get('cohort_counts',{}),dict(original_selection=512,all_eligible=self.spec['full_rows'][s['scene']]),'legacy cohorts')
                self.need(row,len(d.get('methods',{}))==s['heads'],'Legacy head matrix count differs')
                freeze=self.doc(folder/'checkpoint_freeze.json',row);self.bound(row,folder/'checkpoint_freeze.json',d.get('checkpoint_freeze_sha256'))
                row['methods']=d.get('methods',{})
                for name,item in d.get('methods',{}).items():
                    relative=('scores/'+name if s['scene']=='balls' else 'runs/'+name)+'.json'
                    score=self.doc(folder/relative,row)
                    if score:self.arm(row,score,name,self.spec['full_rows'][s['scene']])
                    if freeze:
                        fr=freeze.get('heads',freeze.get('selected',{})).get(name,{})
                        ck=fr.get('checkpoint',fr.get('path'))
                        if not ck:self.need(row,False,'Legacy missing checkpoint path '+name);continue
                        self.bound(row,ck,fr.get('sha256'))
                        head_done=self.doc(Path(ck).parent/'complete.json',row)
                        if head_done:
                            self.fields(row,head_done,dict(epochs=100,test_read=False),'Legacy fixed head')
                            if 'status' in head_done:self.fields(row,head_done,dict(status='COMPLETE'),'Legacy fixed head')
                        self.bound(row,Path(ck).parent/'selected_validation.json',fr.get('receipt_sha256',fr.get('selection_receipt_sha256')))
                        sel=self.doc(Path(ck).parent/'selected_validation.json',row)
                        if sel:
                            self.need(row,sel.get('epoch')==fr.get('epoch')==item.get('selected_epoch'),'Legacy selected epoch differs '+name)
                            self.need(row,len(sel.get('ids',[]))==512,'Legacy selection not 512 '+name)
                            if score:
                                self.need(row,score.get('ids',[])[:512]==sel['ids'],'Legacy first512 IDs differ '+name)
                                self.need(row,all(abs(a-b)<=2e-5+2e-5*abs(b) for a,b in zip(score.get('per_recipient_mse',[])[:512],sel.get('per_recipient_mse',[]))),'Legacy old512 reproduction differs '+name)
                        if isinstance(fr.get('code_sha256'),dict):self.filemap(row,fr['code_sha256'])
            claimed=bool(d and d.get('status')=='COMPLETE')
        self.finish(row,claimed,self.path(folder).exists())

    def comparisons(self):
        groups=defaultdict(list)
        for row in self.rows:
            if row['kind']=='fullval' and row['status']=='verified':groups[row['family'],row['scene'],row['supports']].append(row)
        result=[]
        for (family,scene,supports),rows in groups.items():
            raw={r['method']:self.raw_results[r['id']] for r in rows};base=raw.get('Base'); table={}
            common=None
            for d in raw.values():
                ids=set(d.get('wrong',{}).get('ids',[]));common=ids if common is None else common&ids
            for method,d in raw.items():
                entry=dict(matched=d['matched']['mse'],null=d['null']['mse'],wrong=d['wrong']['mse'],
                           matched_rows=d['matched']['recipients'],wrong_rows=d['wrong']['recipients'])
                if base and d['matched']['ids']==base['matched']['ids']:
                    b=base['matched']['mse'];entry['improvement_vs_Base_percent']=100*(b-d['matched']['mse'])/b
                    entry['base_denominator']=b
                entry['arms_on_common_wrong_cohort']={}
                for arm in ('matched','null','wrong'):
                    values=dict(zip(d[arm]['ids'],d[arm]['per_recipient_mse']))
                    entry['arms_on_common_wrong_cohort'][arm]=sum(values[i] for i in common)/len(common) if common else None
                table[method]=entry
            result.append(dict(family=family,scene=scene,supports=supports,methods=table,common_wrong_rows=len(common or []),
                               common_wrong_ids_sha256=hashlib.sha256(canonical(sorted(common or []))).hexdigest(),
                               scope='Only currently verified methods; common cohort may change as pending methods finish'))
        return result

    def run(self):
        manifest_rows=[]
        for m in self.spec['manifests']:
            row=self.newrow(dict(kind='manifest',folder=str(Path(m['path']).parent)))
            d=self.doc(m['path'],row)
            if d:
                self.fields(row,d,dict(version=m['version'],test_read=False),'manifest')
                if m.get('sha256'):self.bound(row,m['path'],m['sha256'])
                if d.get('code_sha256'):self.filemap(row,d['code_sha256'])
            self.finish(row,bool(d),self.path(m['path']).exists());manifest_rows.append(row)
        for s in self.spec['items']:self.collect_item(s)
        for s in self.spec['old_tails']:
            try:self.legacy(s)
            except Exception as exc:
                row=self.newrow(dict(kind='legacy_tail',family='supervised',method='all',**s))
                row['contradictions'].append(type(exc).__name__+': '+str(exc));self.finish(row,False,True)
        counts=defaultdict(Counter)
        for row in self.rows:counts[row['kind']][row['status']]+=1
        all_verified=all(r['status']=='verified' for r in self.rows)
        return dict(version=VERSION,captured_at_unix=time.time(),mode='final' if self.args.final else 'metadata',
                    scope=self.spec['scope'],status='VERIFIED_COMPLETE' if all_verified and self.args.final else 'PARTIAL_INVENTORY',
                    all_metadata_verified=all_verified,expected=self.spec['counts'],counts=dict(counts),items=self.rows,
                    comparisons=self.comparisons(),hash_files_read=len(self.hashes),test_read=False,optimizer_steps=0,
                    semantics={'missing':'No artifact at the declared path','incomplete':'Work exists but completion has not been claimed',
                               'contradicted':'A receipt, identity, hash, count or completion claim disagrees',
                               'verified':'Declared completed artifacts agree within the explicitly stated verification scope'},
                    scientific_outcome_is_not_an_acceptance_gate=True)


def report(data):
    lines=['# CoPhy 实验产物总账','',f"模式：{data['mode']}；状态：{data['status']}。只核验产物和预算，不要求方法获胜。",'',
           '| 阶段 | 已核验 | 进行中/未齐 | 尚无工件 | 矛盾 |','|---|---:|---:|---:|---:|']
    for kind,c in data['counts'].items():lines.append(f"| {kind} | {c.get('verified',0)} | {c.get('incomplete',0)} | {c.get('missing',0)} | {c.get('contradicted',0)} |")
    lines+=['','metadata 模式的已核验不代表大文件已复核；最终以 --final 的绑定哈希核验为准。所有结果仍为 validation。','',
            '| Family | 场景 | S | 方法 | Matched MSE | Null MSE | Wrong MSE | 相对 Base 改善 | Wrong 人数 |',
            '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for group in data['comparisons']:
        for method,r in group['methods'].items():
            fmt=lambda x:'—' if x is None else f'{x:.6f}'
            lines.append(f"| {group['family']} | {group['scene']} | {group['supports']} | {method} | {fmt(r['matched'])} | {fmt(r['null'])} | {fmt(r['wrong'])} | {fmt(r.get('improvement_vs_Base_percent'))}% | {r['wrong_rows']} |")
    lines+=['','Matched/Null 使用完整相同 recipient 队列；Wrong 只在自己的合法队列内比较。JSON 保留分母、人数、队列哈希及共同队列结果。','']
    for row in data['items']:
        if row['status']=='contradicted':
            lines.append('需核对：'+row['id'])
            lines.extend('- '+reason for reason in row['contradictions']+row['missing_files'])
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--root');p.add_argument('--final',action='store_true');a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text());c=Collector(a,spec);data=c.run()
    data['spec_path']=str(Path(a.spec).resolve());data['spec_sha256']=hashlib.sha256(Path(a.spec).read_bytes()).hexdigest()
    data['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'inventory.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'report.md').write_text(report(data))
    print(json.dumps({k:data[k] for k in ('version','status','mode','counts','hash_files_read','test_read')}))


if __name__=='__main__':main()
