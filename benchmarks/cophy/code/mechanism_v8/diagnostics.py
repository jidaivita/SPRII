"""Frozen train/validation mechanism assays. Never trains a source or task head."""
import argparse,hashlib,importlib.util,json,sys,time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from controlled import load_controlled

VERSION='cophy-mechanism-v8-1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little')
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending.json');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def module(path):
 name='_mech_'+sha(path)[:12];s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def core_path(root,h):
 for name in ('readout.py','supervised_readout.py'):
  for path in sorted((Path(root)/'source').rglob(name)):
   if sha(path)==h:return path
 raise ValueError('Exact bound reader not found '+h)

def load(spec,device):
 root=Path(spec['root']);out=Path(spec['readout']);legacy=spec['family']=='CoPhyNet'
 if legacy:
  marker=read(out/'codes_complete.json');core=core_path(root,marker['implementation_sha256'])
  conf=read(out/'S3/learned/config.json');base=str(next(Path(p).parent for p in conf['input_sha256'] if Path(p).name=='input_train.npz'))
 else:
  marker=read(out/'encoding_complete.json')['binding'];core=core_path(root,marker['readout_code_sha256']);base=marker['base']
 recipe=spec.get('recipe');m=load_controlled(core,recipe) if recipe else module(core)
 args=SimpleNamespace(root=str(root),out=str(out),base=base,scene=spec['scene'],reference='learned',supports=3,epochs=100,device=device,prepared=None)
 if not legacy:args.prepared=spec['prepared']
 data=m.Data(args);ck=torch.load(out/'S3/learned/selected.pt',map_location=device,weights_only=False)
 head=m.Head(data.dims,data.det_dims,data.support_dims,data.horizon).to(device);head.load_state_dict(ck['model'],strict=True);head.eval();head.requires_grad_(False)
 return m,args,data,head,core

def metadata(root,scene):
 p=Path(root)/'prepared_v3'/scene/'training_preflight.json'
 if scene=='blocktower':p=Path(read(Path(root)/'runtime_profiles.json')['scenes'][scene]['training_preflight']['path'])
 pre=read(p);result={}
 for split in ('train','val'):
  item=pre['artifacts']['raw_relations_'+split]
  if sha(item['path'])!=item['sha256']:raise ValueError('Changed physical audit')
  result[split]={(str(r['id']),int(r['slot'])):r for r in read(item['path'])}
 return result,sha(p)

def values(r,scene):return list(r['raw_physical'])+(list(r['raw_gravity']) if scene=='blocktower' else [])
def names(scene):return ['mass','friction','gravity_x','gravity_y'] if scene=='blocktower' else ['mass','friction','restitution']

def ridge(train,val,field_names):
 x=train['x'].astype(np.float64);z=val['x'].astype(np.float64);y=train['y'].astype(np.float64)
 mu=x.mean(0);sc=x.std(0).clip(1e-8);x=(x-mu)/sc;z=(z-mu)/sc
 classes=[np.unique(train['labels'][:,j]) for j in range(train['labels'].shape[1])]
 one=np.concatenate([(train['labels'][:,j,None]==c).astype(float) for j,c in enumerate(classes)],1)
 target=np.concatenate((y,one),1);center=target.mean(0);w=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@(target-center));pred=z@w+center
 fields={}
 for j,n in enumerate(field_names):
  mse=float(((pred[:,j]-val['y'][:,j])**2).mean());var=float(val['y'][:,j].var());fields[n]=dict(mse=mse,r2=1-mse/var if var>1e-16 else None)
 at=len(field_names)
 for j,c in enumerate(classes):
  guessed=c[pred[:,at:at+len(c)].argmax(1)];actual=val['labels'][:,j];at+=len(c)
  fields[field_names[j]].update(accuracy=float((guessed==actual).mean()),balanced_accuracy=float(np.mean([(guessed[actual==v]==v).mean() for v in c if (actual==v).any()])))
 cov=x.T@x/max(1,len(x)-1);eig=np.linalg.eigvalsh(cov).clip(0);p=eig/eig.sum() if eig.sum()>0 else eig
 erank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if eig.sum()>0 else 0.
 return dict(fields=fields,train_objects=len(x),validation_objects=len(z),effective_rank=erank,dimensions=x.shape[1],ridge_alpha=1.,normalization='train only')

def tensor_u(s):return s['u'] if isinstance(s,dict) else s

@torch.no_grad()
def trace_set(data,head,args,meta,split,limit=None):
 row=data.data[split];ix=np.arange(len(row['ids']))
 if limit and len(ix)>limit:ix=np.asarray(sorted(ix,key=lambda i:seed('mech-probe:'+row['ids'][i]))[:limit])
 state={};hooks=[];steps={1,max(1,data.horizon//2),data.horizon}
 def mem_hook(mod,inp,out):state['memory']=out.mean(2)
 def init_hook(mod,inp,out):state['h0']=out;state['step']=0
 def cell_hook(mod,inp,out):
  state['step']+=1
  if state['step'] in steps:state['h'+str(state['step'])]=out.reshape(state['batch'],data.slots,-1)
 hooks=[head.support.register_forward_hook(mem_hook),head.init.register_forward_hook(init_hook),head.cell.register_forward_hook(cell_hook)]
 arrays=defaultdict(list);ys=[];ls=[];object_ids=[]
 try:
  for off in range(0,len(ix),128):
   batch=ix[off:off+128];q,det,mask,s,target=data.batch(split,batch,0,args.device);state.clear();state['batch']=len(batch)
   head(q,det,mask,s);u=tensor_u(s);features=dict(source_support_mean=u.mean(2),**{k:v for k,v in state.items() if torch.is_tensor(v)})
   vel=(q[:,-1]-q[:,0])/2;public=det[:,-1];public=public[...,None] if public.ndim==2 else public
   slots=torch.eye(data.slots,device=q.device)[None].expand(len(batch),-1,-1)
   features['current_public_only']=torch.cat((q[:,-1],vel,public,slots),-1)
   # These are anatomical slices only where the model actually defines P.
   if args.p_width:
    features['P_support']=u[...,:args.p_width].mean(2);features['T_or_current_state']=u[...,args.p_width:].mean(2)
   active=mask.cpu().numpy()>0
   for k,v in features.items():arrays[k].append(v.cpu().numpy()[active])
   for local,slot in zip(*np.where(active)):
    r=meta[(row['ids'][batch[local]],int(slot))];ys.append(values(r,args.scene));ls.append(r['physical']);object_ids.append([r['id'],int(slot)])
 finally:
  for h in hooks:h.remove()
 y=np.asarray(ys,np.float64);labels=np.asarray(ls,np.int64)
 return {k:dict(x=np.concatenate(v),y=y,labels=labels) for k,v in arrays.items()},object_ids

def donor_groups(data,meta):
 groups=defaultdict(list);all_ids=data.manifest['splits']['val']['all_ids']
 for i,ident in enumerate(all_ids):
  for k in np.flatnonzero(data.data['val']['donor_seen'][i]>0):
   r=meta.get((ident,int(k)))
   if r is None:continue
   public=(int(k),json.dumps(r.get('known_type'),sort_keys=True),tuple(r.get('raw_gravity',[])))
   groups[(public,tuple(r['physical']))].append(i)
 return groups,{s:i for i,s in enumerate(all_ids)}

def make_plan(data,meta,groups,lut,kind,count=3,factor=None):
 base=data.val_plan;plan=np.repeat(base[:,:,:1],count,axis=2) if count!=3 else base.copy()
 if count>3:plan[:,:,:3]=base
 eligible=np.ones(len(base),bool);focal=np.full(len(base),-1,np.int64)
 for i,ident in enumerate(data.data['val']['ids']):
  active=np.flatnonzero(data.data['val']['mask'][i]>0)
  focal[i]=int(active[seed('mech-focal:'+ident)%len(active)])
  slots=active if kind=='extra' else [focal[i]]
  for k in slots:
   r=meta[(ident,int(k))];public=(int(k),json.dumps(r.get('known_type'),sort_keys=True),tuple(r.get('raw_gravity',[])));label=tuple(r['physical'])
   if kind=='extra':pool=groups.get((public,label),[]);excluded={lut[ident],*base[i,k].tolist()};n=count-3
   else:
    pool=[v for (p,l),vs in groups.items() if p==public and len(l)==len(label) and sum(a!=b for a,b in zip(l,label))==1 and l[factor]!=label[factor] for v in vs]
    excluded={lut[ident]};n=3
   pool=np.asarray(sorted(set(pool)-excluded),np.int64)
   if len(pool)<n:eligible[i]=False;continue
   selected=np.random.default_rng(seed(f'mech:{kind}:{factor}:{ident}:{k}')).choice(pool,n,replace=False)
   if kind=='extra':plan[i,k,3:]=selected
   else:plan[i,k]=selected
 return plan,eligible,focal

@torch.no_grad()
def score(data,head,args,indices,plan=None,arm='matched',geometry=False):
 original=data.val_plan;olds=args.supports;data.args.supports=plan.shape[-1] if plan is not None else 3
 if plan is not None:data.val_plan=plan
 rows=[];frames=[];objects=[];within=[];contexts=[];within_p=[]
 try:
  for off in range(0,len(indices),128):
   ix=indices[off:off+128];q,det,mask,s,y=data.batch('val',ix,0,args.device,arm=arm);pred=head(q,det,mask,s)
   err=(pred-y).square().mean(-1);obj=err.mean(1);frame=(err*mask[:,None]).sum(2)/mask.sum(1).clamp_min(1)[:,None]
   rows.extend(frame.mean(1).cpu().tolist());frames.extend(frame.cpu().tolist());objects.extend(obj.cpu().tolist())
   if geometry and s is not None:
    u=tensor_u(s);within.extend((((u[:,:,0]-u[:,:,1])**2).mean(-1)*mask).sum(1).div(mask.sum(1).clamp_min(1)).cpu().tolist());contexts.extend(u.mean(2).cpu().numpy())
    if args.p_width:within_p.extend((((u[:,:,0,:args.p_width]-u[:,:,1,:args.p_width])**2).mean(-1)*mask).sum(1).div(mask.sum(1).clamp_min(1)).cpu().tolist())
 finally:data.val_plan=original;data.args.supports=olds
 return dict(indices=np.asarray(indices),mse=np.asarray(rows),frames=np.asarray(frames),objects=np.asarray(objects),within=np.asarray(within),context=np.asarray(contexts),within_p=np.asarray(within_p))

def run(spec_path,device):
 spec=read(spec_path);dest=Path(spec['out']);dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 m,args,data,head,core=load(spec,device);torch.set_num_threads(2)
 args.p_width=0 if spec['role']=='Native' else (16 if spec['family']=='CoPhyNet' else 64)
 meta,audit_sha=metadata(spec['root'],spec['scene']);report=dict(status='RUNNING',version=VERSION,spec=spec,core_sha256=sha(core),head_sha256=sha(Path(spec['readout'])/'S3/learned/selected.pt'),physical_audit_sha256=audit_sha,source_optimizer_steps=0,head_optimizer_steps=0,test_read=False)
 all_ix=np.arange(len(data.data['val']['ids']));cohorts={};arrays={}
 def record(name,sc,focal=None):
  ix=sc['indices'];entry=dict(mse=float(sc['mse'].mean()) if len(ix) else None,recipients=len(ix),coverage=len(ix)/len(all_ix),plan_sha256=None)
  if len(ix):entry['thirds']=[float(sc['frames'][:,s].mean()) for s in np.array_split(np.arange(data.horizon),3)]
  if focal is not None and len(ix):entry['focal_mse']=float(sc['objects'][np.arange(len(ix)),focal[ix]].mean())
  for k,v in sc.items():
   if k!='context':arrays[name+'__'+k]=v
  cohorts[name]=entry
 correct=score(data,head,args,all_ix,geometry=True);record('correct_S3',correct)
 saved_path=Path(spec['readout'])/'S3/learned'/('results.json' if spec['family']=='CoPhyNet' else 'fullval/results.json')
 saved=read(saved_path);expected=saved['matched']['mse']
 if not np.isclose(correct['mse'].mean(),expected,rtol=2e-5,atol=2e-6):raise ValueError('Fixed-head full-validation reproduction failed')
 report['reproduction']=dict(saved_mse=expected,current_mse=float(correct['mse'].mean()),matched=True)
 cohorts['correct_S3']['within_correct_history_distance']=float(correct['within'].mean())
 for j in range(3):record('repeat_one_history_'+str(j),score(data,head,args,all_ix,np.repeat(data.val_plan[:,:,j:j+1],3,2)))
 record('null',score(data,head,args,all_ix,arm='null'))
 ix=np.flatnonzero((data.wrong_plan>=0).all((1,2)))
 if len(ix):
  wrong=score(data,head,args,ix,arm='wrong',geometry=True);record('wrong_any',wrong)
  mask=data.data['val']['mask'][ix];delta=correct['context'][ix]-wrong['context'];denom=mask.sum(1).clip(1)
  dist=((delta**2).mean(-1)*mask).sum(1)/denom
  report['geometry']=dict(within_same_physics=float(correct['within'][ix].mean()),between_wrong_physics=float(dist.mean()),
   within_between_ratio=float(correct['within'][ix].mean()/dist.mean()) if dist.mean()>0 else None,
   interpretation='fixed query, normalized U; includes nuisance variation, not proof of exclusive parameter encoding')
  if args.p_width:
   pdist=((delta[...,:args.p_width]**2).mean(-1)*mask).sum(1)/denom
   report['geometry']['P']=dict(within=float(correct['within_p'][ix].mean()),between=float(pdist.mean()),ratio=float(correct['within_p'][ix].mean()/pdist.mean()) if pdist.mean()>0 else None)
 else:cohorts['wrong_any']=dict(mse=None,recipients=0,coverage=0)
 groups,lut=donor_groups(data,meta['val'])
 plans={'correct_S3':hashlib.sha256(np.ascontiguousarray(data.val_plan).tobytes()).hexdigest()}
 for s in (5,8):
  plan,valid,focal=make_plan(data,meta['val'],groups,lut,'extra',s);ix=np.flatnonzero(valid);name='S'+str(s)
  record(name,score(data,head,args,ix,plan) if len(ix) else dict(indices=ix,mse=np.array([]),frames=np.array([]),objects=np.array([]),within=np.array([])))
  plans[name]=hashlib.sha256(plan.tobytes()).hexdigest()
 for factor,name in enumerate(names(args.scene)[:2 if args.scene=='blocktower' else 3]):
  plan,valid,focal=make_plan(data,meta['val'],groups,lut,'wrong1',factor=factor);ix=np.flatnonzero(valid);label='wrong1_'+name
  sc=score(data,head,args,ix,plan) if len(ix) else dict(indices=ix,mse=np.array([]),frames=np.array([]),objects=np.array([]),within=np.array([]))
  record(label,sc,focal);plans[label]=hashlib.sha256(plan.tobytes()).hexdigest()
  if len(ix):cohorts[label]['paired_focal_delta']=float((sc['objects'][np.arange(len(ix)),focal[ix]]-correct['objects'][ix,focal[ix]]).mean())
 for name,cohort in cohorts.items():
  ix=arrays.get(name+'__indices',np.array([],dtype=int))
  if len(ix):cohort['correct_mse_same_cohort']=float(correct['mse'][ix].mean());cohort['paired_mse_delta']=float((arrays[name+'__mse']-correct['mse'][ix]).mean())
 # Trajectory-change strata use train targets only for thresholds, never selection.
 def activity(row):
  trajectory=np.concatenate((row['q'][:,-1:],row['target']),1);acc=np.diff(trajectory,n=2,axis=1)
  return (np.linalg.norm(acc,axis=-1)*row['mask'][:,None]).max((1,2))
 train_activity=activity(data.data['train']);cuts=np.quantile(train_activity,[1/3,2/3]);va_activity=activity(data.data['val']);arrays['activity']=va_activity
 strata={}
 for j in range(3):
  ix=np.flatnonzero(np.digitize(va_activity,cuts)==j);strata[str(j)]=dict(recipients=len(ix),correct_mse=float(correct['mse'][ix].mean()) if len(ix) else None)
 report.update(cohorts=cohorts,plans=plans,activity_strata=dict(proxy='maximum trajectory second difference; not a contact label',train_thresholds=cuts.tolist(),groups=strata),
  support_curve_scope='fixed S3-trained head; S5/S8 are input-budget sensitivity, not a trained sample-efficiency claim')
 np.savez_compressed(dest/'per_recipient.npz',ids=np.asarray(data.data['val']['ids']),**arrays)
 write(dest/'utility.json',report)
 tr,trids=trace_set(data,head,args,meta['train'],'train',4096);va,vaids=trace_set(data,head,args,meta['val'],'val')
 fields=names(args.scene);probes={k:ridge(tr[k],va[k],fields) for k in tr}
 report.update(status='COMPLETE',probes=probes,probe_train_query_cap=4096,probe_train_object_ids_sha256=hashlib.sha256(json.dumps(trids).encode()).hexdigest(),
  probe_validation_object_ids_sha256=hashlib.sha256(json.dumps(vaids).encode()).hexdigest(),
  probe_interpretation='P/T only where structurally defined; RSSM current state can depend on P; h/memory include supervised head learning; readout is not a proof of causal mediation',
  source_probe_reference=str(Path(spec['readout'])/'probes.json'),completed=time.time())
 write(dest/'results.json',report);write(dest/'complete.json',dict(status='COMPLETE',version=VERSION,source_optimizer_steps=0,head_optimizer_steps=0,test_read=False,results_sha256=sha(dest/'results.json')))
 print(json.dumps(dict(status='COMPLETE',id=spec['id'],mse=cohorts['correct_S3']['mse'])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.spec,a.device)
