import argparse,sys,types,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(add_help=False);p.add_argument('--tune-cross',type=float,required=True);p.add_argument('--tune-align',type=float,required=True);a,rest=p.parse_known_args()
path=Path(__file__).resolve().parents[1]/'cophy_complete_v7/legacy/jepa_source.py'
source=path.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='05b01cfbc61d23da0f1517d0998dd66ac1a9891f3d290dcc1d1dfa81b69a6aa4'
old="p.add_argument('--seed',type=int,choices=(0,),default=0)"
assert source.count(old)==1
source=source.replace(old,"p.add_argument('--seed',type=int,choices=(0,1,2),default=0)")
prefix,cli=source.split("if __name__=='__main__':",1)
m=types.ModuleType('_tuned_jepa');m.__file__=str(path);sys.modules[m.__name__]=m;sys.path.insert(0,str(path.parent));exec(compile(prefix,str(path),'exec'),m.__dict__)
original_setup=m.setup;original_objective=m.objective
from canonical_losses import canonical_vicreg
import torch,numpy as np
from dataclasses import replace
def setup(args):
 core,route,data,planner,model,b=original_setup(args);model.config=replace(model.config,lambda_cross=a.tune_cross,lambda_align=a.tune_align)
 b.update(lambda_cross=a.tune_cross,lambda_align=a.tune_align,tuning_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());b.pop('sha256',None);b['sha256']=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return core,route,data,planner,model,b
def objective(core,route,model,data,plan,ix,method,mode,micro):
 loss,metrics,extra=original_objective(core,route,model,data,plan,ix,method,mode,micro)
 if a.tune_align:
  assert mode=='all';rows=extra['rows'];eligible=np.flatnonzero(plan['common'][ix]);mask=torch.as_tensor(plan['external'][ix[eligible]]>=0,device=loss.device)
  own=extra['own'][rows][mask];donor=extra['mixed'][mask]
  if len(own)>=2:
   align,am=canonical_vicreg(own,donor);loss=loss+a.tune_align*align;metrics['align']=float(align.detach());extra['terms']['align']=a.tune_align*align
 metrics['loss']=float(loss.detach());return loss,metrics,extra
m.setup=setup;m.objective=objective;sys.argv=[str(path)]+rest;m.__dict__['__name__']='__main__';exec(compile("if __name__=='__main__':"+cli,str(path),'exec'),m.__dict__)
