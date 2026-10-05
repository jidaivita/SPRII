"""Bind existing full-validation XEP arrays to the unchanged v7 U32 reader.

CPU-only; no video decoding, model inference, target-dependent sampling or test.
Original512 query inputs/targets remain exact and exclusively select new heads.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import numpy as np
import jepa_source as io

VERSION='cophy-v7-supervised-fullval-base-2'

def main(a):
 base,full,out=Path(a.base).resolve(),Path(a.full_data).resolve(),Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 old=io.read(base/'manifest.json');fm=io.read(full/'manifest.json');ready=io.read(full/'data_ready.json')
 if any(d.get('test_read') is not False for d in (old,fm,ready)):raise ValueError('Only validation artifacts permitted')
 if old['prefix']!=3:raise ValueError('Wrong query horizon')
 part=fm['metadata'];ids=list(map(str,fm['query_ids']));orig=old['splits']['val'];oldids=orig['query_ids']
 if len(ids)!={'balls':2000,'collision':4000}[a.scene] or ids[:512]!=oldids or len(oldids)!=512:raise ValueError('Original selection/full cohorts changed')
 allids=part.get('all_ids',fm.get('all_history_ids'))
 if allids!=orig['all_ids']:raise ValueError('History population changed')
 for k in ('presence','physical','known_type','gravity'):
  if k in orig and any(orig[k][ident]!=part[k][ident] for ident in oldids):raise ValueError('Original metadata changed '+k)
 m=copy.deepcopy(old);val=m['splits']['val'];val.update(copy.deepcopy(part));val.update(all_ids=allids,query_ids=ids,full_query_ids=ids,selection_query_ids=oldids)
 # Explicit and compact candidate tables must not coexist ambiguously.
 if 'candidates' in part:val.pop('candidate_groups',None);val.pop('candidate_keys',None)
 m['v7_original_manifest']=dict(path=str(base/'manifest.json'),sha256=io.sha(base/'manifest.json'))
 m['v7_full_manifest']=dict(path=str(full/'manifest.json'),sha256=io.sha(full/'manifest.json'))
 m['v7_wrong_domain']='all audited full-validation metadata; same source-independent domain in every v7 method; old tables retained separately'
 files={str(p):io.sha(p) for p in (base/'manifest.json',full/'manifest.json',full/'data_ready.json')}
 for path,digest in ready.get('file_sha256',{}).items():
  if io.sha(path)!=digest:raise ValueError('Full data binding changed '+path)
 bound={Path(path).resolve():digest for path,digest in ready.get('file_sha256',{}).items()}
 manifest_path=(full/'manifest.json').resolve();manifest_digests=[]
 if manifest_path in bound:manifest_digests.append(bound[manifest_path])
 if 'manifest_sha256' in ready:manifest_digests.append(ready['manifest_sha256'])
 if not manifest_digests or any(digest!=io.sha(manifest_path) for digest in manifest_digests):raise ValueError('Missing or changed full manifest SHA binding')
 for path in (full/'input_val.npz',full/'target_val.npz'):
  if bound.get(path.resolve())!=io.sha(path):raise ValueError('Missing full data SHA binding '+str(path))
 for name in ('input','target'):
  path=full/(name+'_val.npz');oldpath=base/(name+'_val.npz')
  with np.load(path,allow_pickle=False) as x,np.load(oldpath,allow_pickle=False) as y:
   for archive,expected_ids in ((x,ids),(y,oldids)):
    if name=='input' and 'ids' not in archive:raise ValueError('Input array IDs missing')
    if 'ids' in archive and archive['ids'].astype(str).tolist()!=expected_ids:raise ValueError('Array ID order changed')
   # Legacy targets contain pose only; their row order is bound by the checked
   # input IDs, shared manifest/ready hashes, and exact original512 targets.
   for k in ('pose','detected','presence') if name=='input' else ('pose',):
    current,original=x[k],y[k]
    if not current.ndim or not original.ndim or current.shape[0]!=len(ids) or original.shape[0]!=len(oldids) or current.shape[1:]!=original.shape[1:]:raise ValueError('Array row count or shape changed: '+name+'/'+k)
    if not np.array_equal(current[:512],original):raise ValueError('Original512 changed: '+name+'/'+k)
  files[str(path)]=io.sha(path)
 labels=np.asarray([val['physical'][ident] for ident in ids],np.int64);mask=np.asarray([val['presence'][ident] for ident in ids],np.float32)
 if labels.shape[-1]!=3 or ((labels<0)|(labels>=3)).any():raise ValueError('Unexpected audited physical category support')
 params=np.eye(3,dtype=np.float32)[labels].reshape(len(ids),mask.shape[1],9)*mask[...,None]
 with np.load(base/'parameters_val.npz',allow_pickle=False) as y:
  if not np.array_equal(params[:512],y['values']):raise ValueError('Original512 Known vector changed')
 for split in ('train','val'):
  for name in ('input','target','parameters'):
   target=out/f'{name}_{split}.npz'
   if split=='val' and name=='parameters':
    if not target.exists():
     tmp=target.with_suffix('.pending.npz');np.savez(tmp,ids=np.asarray(ids),values=params);tmp.replace(target)
    with np.load(target,allow_pickle=False) as z:
     if not np.array_equal(z['values'],params):raise ValueError('Existing parameter vectors changed')
    files[str(target)]=io.sha(target);continue
   source=(full if split=='val' else base)/f'{name}_{split}.npz'
   if target.exists() or target.is_symlink():
    if target.resolve()!=source:raise ValueError('Different base alias')
   else:target.symlink_to(source)
   files[str(source)]=io.sha(source)
 io.immutable(out/'manifest.json',m);io.write(out/'prepared.json',dict(status='COMPLETE',version=VERSION,scene=a.scene,validation_rows=len(ids),selection_rows=512,files=files,source_manifest_sha256=io.sha(base/'manifest.json'),old512_exact=True,test_read=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=('balls','collision'),required=True);p.add_argument('--base',required=True);p.add_argument('--full-data',required=True);p.add_argument('--out',required=True);a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
 with open(Path(a.out)/'prepare.lock','a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
