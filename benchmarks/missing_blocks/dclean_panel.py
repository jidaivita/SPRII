"""Same frozen-code reader, parameter probe, and donor intervention across sources."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dclean_external import ROOT,NATIVE,Encoder,batch,contract,data,module,read,seed_all,sha,specs,stats,write
H=(1,4,16,32)
REFS=[('runs/wave2/b3_lp1.0_lx0.1_s0/checkpoints/step_020000.pt','bab118a8f3bb790ce89cfbdbc87f48285b9871599e9b3f7f0e0fd8d5260f9393'),('runs/final_repeats/selected_lp1.0_lx0.1_s1/checkpoints/step_020000.pt','ef612f00f4410717ba589e81f7523cfa274845e51c04e994e72ab8699c7cc748'),('runs/final_repeats/selected_lp1.0_lx0.1_s2/checkpoints/step_020000.pt','0c4b7abe66b63900644fe16d776c0a67cf19056ab41cef48011da4bc5882b25b')]
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def atomic_ck(p,value):
 t=p.with_suffix('.partial');torch.save(value,t);t.replace(p)
def native():
 assert sha(NATIVE)=='04f4cefa54a2044bd70c251950b4b43fa2007413dfe3e2d679a2714b7789f928'
 return module(NATIVE,'dclean_native_panel')
def load_source(method,seed):
 if method=='SPRII':
  p,h=REFS[seed];p=Path(p);assert sha(p)==h;ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['step']==20000
  mod=native();m=mod.PersistentJEPA(mod.ModelConfig(**ck['config']['model']));m.load_state_dict(ck['model'],strict=True)
 else:
  p=ROOT/'sources'/f'{method}_s{seed}'/'final.pt';done=read(p.parent/'COMPLETE.json');assert done['status']=='COMPLETE' and done['checkpoint_sha256']==sha(p)
  ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['step']==20000 and ck['method']==method and ck['seed']==seed
  m=Encoder(method);m.load_state_dict(ck['model'],strict=True)
 return m.cuda().eval().requires_grad_(False),dict(checkpoint=str(p),sha256=sha(p),method=method,seed=seed)
def encode(m,method,x,u):return m.persistent(m.observation(x),u) if method=='SPRII' else m.encode(x,u)
def cache_path(method,seed):return ROOT/'cache'/f'{method}_s{seed}'
def cache(method,seed):
 contract();out=cache_path(method,seed);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():load_cache(method,seed,'val');return
 m,source=load_source(method,seed);before=digest(m);files={};norm=None
 for split in ['train','val']:
  d=data(split);n=len(d['states']);codes=[]
  with torch.inference_mode():
   for start in range(0,n*72,256):
    ix=np.arange(start,min(start+256,n*72));s=ix//72;r=ix%72//9;t=23+ix%9
    x=torch.as_tensor(d['states'][s[:,None],r[:,None],t[:,None]+np.arange(-23,1)],device='cuda:0');u=torch.as_tensor(d['actions'][s[:,None],r[:,None],t[:,None]+np.arange(-23,0)],device='cuda:0')
    z=encode(m,method,x,u).cpu().numpy();assert np.isfinite(z).all();codes.append(z)
  z=np.concatenate(codes);assert 0<z.shape[1]<=64
  if split=='train':norm=dict(mean=z.astype('float64').mean(0).tolist(),scale=z.astype('float64').std(0).clip(1e-8).tolist())
  p=out/f'{split}.npy';tmp=p.with_suffix('.partial')
  with tmp.open('wb') as f:np.save(f,z.reshape(n,8,9,-1))
  tmp.replace(p);files[split]=sha(p)
 assert digest(m)==before
 write(out/'COMPLETE.json',dict(status='COMPLETE',source=source,files=files,normalization=norm,source_frozen=True,windows='24 history states / 23 observed actions; all anchors23..31',test_read=False))
def load_cache(method,seed,split):
 p=cache_path(method,seed);c=read(p/'COMPLETE.json');assert c['status']=='COMPLETE' and sha(Path(c['source']['checkpoint']))==c['source']['sha256'];assert sha(p/f'{split}.npy')==c['files'][split]
 return np.load(p/f'{split}.npy',mmap_mode='r'),c

def slot(raw,normalization,sp,wrong=None):
 s=sp[:,0] if wrong is None else np.asarray(wrong);z=np.concatenate([raw[s,sp[:,3],sp[:,4]-23],raw[s,sp[:,1],sp[:,2]-23]])
 z=(z-np.asarray(normalization['mean']))/np.asarray(normalization['scale']);out=np.zeros((len(z),64),np.float32);out[:,:z.shape[1]]=z;return torch.as_tensor(out,device='cuda:0')
class Head(nn.Module):
 def __init__(self):
  super().__init__()
  for k,v in stats().items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
  self.horizon=nn.Embedding(4,32);self.net=native().MLP([196,256,256,4])
 def forward(self,x,u,z):
  out=[]
  for j,h in enumerate(H):
   mask=torch.zeros((len(x),32),device=x.device);mask[:,:h]=1
   features=torch.cat([(x-self.state_mean)/self.state_scale,z,((u/self.action_scale)*mask[:,:,None]).flatten(1),mask,self.horizon(torch.full((len(x),),j,dtype=torch.long,device=x.device))],1)
   out.append(x+self.net(features)*self.state_scale)
  return torch.stack(out,1)
def fit(method,seed,reader,arm):
 out=ROOT/'readers'/('null_shared' if arm=='null' else f'{method}_s{seed}')/f'r{reader}';out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():assert read(out/'COMPLETE.json')['checkpoint_sha256']==sha(out/'final.pt');return
 assert not (out/'RUN.json').exists(),'Existing incomplete fit needs explicit resume'
 if arm=='matched':raw,c=load_cache(method,seed,'train')
 seed_all(2026091600+reader);m=Head().cuda().train();initial=digest(m);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.05)
 write(out/'RUN.json',dict(method=method if arm=='matched' else 'source-independent',source_seed=seed if arm=='matched' else None,reader_seed=reader,arm=arm,steps=20000,initial_sha256=initial,code_sha256=sha(__file__),source_cache_sha256=sha(cache_path(method,seed)/'COMPLETE.json') if arm=='matched' else None,protocol_sha256=sha(ROOT/'PROTOCOL.json'),test_read=False))
 began=time.monotonic()
 with (out/'train.jsonl').open('x') as log:
  for step in range(1,20001):
   sp=specs(reader,step,True);hs,ha,u,y=batch('train',sp,H);z=slot(raw,c['normalization'],sp) if arm=='matched' else torch.zeros((len(hs),64),device='cuda:0')
   t=step-1;factor=t/500 if t<=500 else .5*(1+math.cos(math.pi*(t-500)/19500))
   for g in opt.param_groups:g['lr']=.0003*factor
   opt.zero_grad(set_to_none=True);pred=m(hs[:,-1],u,z);loss=((pred-y)/m.state_scale).square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
   if step==1 or step%100==0:
    row=dict(step=step,loss=float(loss),seconds=time.monotonic()-began);write(out/'progress.json',row);log.write(json.dumps(row)+'\n');log.flush()
   if step%1000==0:atomic_ck(out/'latest.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,initial_sha256=initial,run_sha256=sha(out/'RUN.json'),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
 os.link(out/'latest.pt',out/'final.pt');write(out/'COMPLETE.json',dict(status='COMPLETE',step=20000,checkpoint_sha256=sha(out/'final.pt'),initial_sha256=initial,test_read=False))
def eval_specs():
 c=contract();d=data('val');ids=set(c['report_ids']);rows=[i for i,x in enumerate(d['system_ids']) if int(x) in ids];sp=[]
 for i in rows:
  ident=int(d['system_ids'][i]);rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'sprii-dclean-eval-20260916:{ident}'.encode()).digest()[:8],'little'))
  for _ in range(8):
   a=int(rng.integers(8));b=(a+int(rng.integers(1,8)))%8;sp.append([i,a,int(rng.integers(23,32)),b,int(rng.integers(23,32))])
 return np.array(sp),dict(zip(rows,np.roll(rows,-1)))
def evaluate(method,seed,reader):
 out=ROOT/'results'/f'{method}_s{seed}_r{reader}';out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():assert read(out/'COMPLETE.json')['arrays_sha256']==sha(out/'per_case.npz');return
 raw,c=load_cache(method,seed,'val');sp,wrong=eval_specs();records={};metrics={};initial=[];head_refs={}
 for arm in ['null','matched']:
  p=ROOT/'readers'/('null_shared' if arm=='null' else f'{method}_s{seed}')/f'r{reader}'
  done=read(p/'COMPLETE.json');assert sha(p/'final.pt')==done['checkpoint_sha256'];ck=torch.load(p/'final.pt',map_location='cuda:0',weights_only=False);assert ck['step']==20000
  m=Head().cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'],strict=True);initial.append(ck['initial_sha256']);before=digest(m);head_refs[arm]=done
  for condition in (['null'] if arm=='null' else ['matched','wrong','zero']):
   errors=[];keys=[]
   with torch.inference_mode():
    for start in range(0,len(sp),48):
     s=sp[start:start+48];hs,ha,u,y=batch('val',s,H)
     z=torch.zeros((len(hs),64),device='cuda:0') if condition in ['null','zero'] else slot(raw,c['normalization'],s,[wrong[i] for i in s[:,0]] if condition=='wrong' else None)
     errors.append((m(hs[:,-1],u,z)-y).square().mean(-1).cpu().numpy());keys.append(np.concatenate([s,s[:,[0,3,4,1,2]]]))
   err=np.concatenate(errors);key=np.concatenate(keys);records[condition+'_error']=err
   if 'keys' in records:np.testing.assert_array_equal(records['keys'],key)
   else:records['keys']=key
   system=np.stack([err[key[:,0]==s].mean(0) for s in sorted(set(key[:,0]))]);metrics[condition]=system.mean(0).tolist()
  assert digest(m)==before
 assert initial[0]==initial[1]
 np.savez_compressed(out/'per_case.npz',**records);write(out/'COMPLETE.json',dict(status='COMPLETE',method=method,source_seed=seed,reader_seed=reader,metrics=metrics,horizons=H,primary='h32 raw state MSE, macro100 report systems',arrays_sha256=sha(out/'per_case.npz'),heads=head_refs,cache_sha256=sha(cache_path(method,seed)/'COMPLETE.json'),test_read=False))
def probe(method,seed):
 from scipy.stats import spearmanr
 raw,c=load_cache(method,seed,'train');val,_=load_cache(method,seed,'val');mu=np.array(c['normalization']['mean']);sd=np.array(c['normalization']['scale']);x=(raw.reshape(-1,raw.shape[-1])-mu)/sd
 y=np.repeat(np.log(data('train')['gamma']),72).astype('float64');ym=y.mean();coef=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@(y-ym))
 keep=np.array([i for i,ident in enumerate(data('val')['system_ids']) if int(ident) in contract()['report_ids']]);v=(val[keep].reshape(-1,val.shape[-1])-mu)/sd;t=np.repeat(np.log(data('val')['gamma'][keep]),72);pred=v@coef+ym
 means=v.reshape(len(keep),72,-1).mean(1);within=float(np.square(v.reshape(len(keep),72,-1)-means[:,None]).sum(-1).mean());between=float(np.square(means-means.mean(0)).sum(-1).mean())
 out=dict(parameter='log_gamma',fit='all training histories only, fixed ridge1; report100 systems, equal72windows/system',r2=float(1-np.square(t-pred).sum()/np.square(t-t.mean()).sum()),spearman=float(spearmanr(t,pred).statistic),geometry=dict(within=within,between=between,ratio=between/max(within,1e-12),normalization='training latent statistics'),cache_sha256=sha(cache_path(method,seed)/'COMPLETE.json'),test_read=False)
 out['spearman_defined']=bool(np.isfinite(out['spearman']))
 if not out['spearman_defined']:out['spearman']=None
 out['constant_prediction']=bool(np.ptp(pred)==0)
 out['latent_train_std']=np.asarray(raw).reshape(-1,raw.shape[-1]).astype('float64').std(0).tolist()
 write(cache_path(method,seed)/'PROBE.json',out)
def smoke():
 contract();seed_all(0);m=Head().cuda();hs,ha,u,y=batch('train',specs(0,1,True),H);z=torch.zeros((len(hs),64),device='cuda:0');p=m(hs[:,-1],u,z);loss=((p-y)/m.state_scale).square().mean();loss.backward();assert torch.isfinite(loss);assert p.shape==(96,4,4)
 # Inputs beyond each requested horizon are masked; future unavailable forces cannot affect that horizon.
 with torch.no_grad():
  u2=u.clone();u2[:,1:]+=123;torch.testing.assert_close(m(hs[:,-1],u2,z)[:,0],m(hs[:,-1],u,z)[:,0],rtol=0,atol=0)
 write(ROOT/'PANEL_SMOKE.json',dict(status='PASS',loss=float(loss),future_mask_verified=True,code_sha256=sha(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','cache','fit','evaluate','probe']);p.add_argument('--method',choices=['NOD','FCRL','SPRII'],default='SPRII');p.add_argument('--seed',type=int,default=0);p.add_argument('--reader',type=int,default=0);p.add_argument('--arm',choices=['null','matched'],default='matched');a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if a.action=='smoke':smoke()
 elif a.action=='cache':cache(a.method,a.seed)
 elif a.action=='fit':fit(a.method,a.seed,a.reader,a.arm)
 elif a.action=='evaluate':evaluate(a.method,a.seed,a.reader)
 else:probe(a.method,a.seed)
