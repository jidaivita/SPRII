"""CPU-only certificate of completed old source budgets; never train or read test."""
import argparse,hashlib,json,os
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text());made=[]
for row in manifest['rows']:
 if not row.get('needs_existing_budget_audit'):continue
 cp=Path(row['source_checkpoint']);ck=torch.load(cp,map_location='cpu',weights_only=False);selected=ck.get('epoch');budget=row['source_budget'];proof={'checkpoint_sha256':sha(cp),'selected_epoch':selected}
 if row.get('source_checkpoint_sha256') and proof['checkpoint_sha256']!=row['source_checkpoint_sha256']:raise ValueError('Existing checkpoint changed: '+str(cp))
 if row['family']=='CoPhyNet':
  latest=cp.parent/('latest_resume.pt' if (cp.parent/'latest_resume.pt').exists() else 'latest.pt')
  lc=torch.load(latest,map_location='cpu',weights_only=False)
  if lc.get('epoch')!=budget or not lc.get('optimizer') or selected is None or not 0<selected<=budget:raise ValueError('Original supervised budget not completed: '+str(cp))
  cfg=ck.get('run_config',ck.get('config',{}));lcfg=lc.get('run_config',lc.get('config',{}))
  if cfg.get('method')!=row['source_method'] or (lcfg.get('method') and lcfg['method']!=cfg['method']):raise ValueError('Wrong supervised source role')
  proof.update(latest=str(latest),latest_sha256=sha(latest),original_training_epoch=lc['epoch'],source_config=cfg)
 else:
  if selected!=budget or ck.get('next_epoch')!=budget+1 or ck.get('next_batch')!=0 or ck.get('test_read') is not False:raise ValueError('Source snapshot is not exact completed budget: '+str(cp))
  proof.update(steps=ck.get('step'),source_version=ck.get('version'))
 result=dict(status='COMPLETE',family=row['family'],scene=row['scene'],source_budget=budget,epochs=budget,checkpoint=str(cp),test_read=False,optimizer_steps_performed=0,**proof)
 path=Path(row['source_marker']);write(path,result);made.append(str(path))
write(Path(a.out),dict(status='COMPLETE',source_count=len(made),markers=made,manifest_sha256=sha(a.manifest),test_read=False));print(json.dumps({'status':'COMPLETE','sources':len(made)}))
