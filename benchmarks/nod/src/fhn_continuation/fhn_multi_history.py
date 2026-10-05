"""K-history aggregation of frozen FHN codes with matched observation budgets."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from fhn_minimal.data import SYSTEMS,CONDITION_FRAMES,one_hot_head,load_trajectory
from fhn_minimal.evaluate import head_sequence,load_model,rollout,ridge_probe
from fhn_minimal.train import create_locs,model_checksum,set_seed

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def predict(model,initial,z,frame,horizon,device):
 x=initial[frame:frame+1].to(device);z=z.view(1,-1,1,1).expand(1,-1,128,128)
 for head in head_sequence(horizon):x=model.predict(x,z,one_hot_head(torch.tensor([head],device=device)).to(device))
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--official-code',required=True,type=Path);p.add_argument('--device',default='cuda');a=p.parse_args()
 data=a.data_dir;sys.path.insert(0,str(a.official_code))
 from ngs.neuralnetworks import NGS_metaNet_Hier
 torch.set_num_threads(4);torch.set_num_interop_threads(1);set_seed(42);device=torch.device(a.device);model=NGS_metaNet_Hier(64,3,2,128).to(device);ck=load_model(model,a.checkpoint,device);before=model_checksum(model)
 assert not a.output.exists(),'Completed evaluation must not be overwritten'
 pool=[36,5,15,24,45];targets=[5,15,24,45];rows=[];latent_rows=[];checked=set()
 for split,systems in SYSTEMS.items():
  for si,system in enumerate(systems):
   trajectories={i:load_trajectory(data,system,i) for i in pool}
   with torch.no_grad():z=torch.cat([model.conditioning_encoder(trajectories[i][list(CONDITION_FRAMES)].unsqueeze(0).to(device)) for i in pool])
   for initial in targets:
    donor_ids=[36]+[i for i in pool if i not in [36,initial]];assert len(donor_ids)==4 and initial not in donor_ids
    for k in [1,2,4]:
     code=z[[pool.index(i) for i in donor_ids[:k]]].mean(0);latent_rows.append(dict(split=split,system_index=si,k=system[0],beta=system[1],initial_id=initial,K=k,donor_ids=donor_ids[:k],z=code.cpu().tolist()))
     for frame in [12,42,72,92]:
      for h in [1,5,50]:
       if frame+h>100:continue
       prediction=predict(model,trajectories[initial],code,frame,h,device);target=trajectories[initial][frame+h:frame+h+1].to(device)
       mse=float((prediction-target).square().mean());l2=float(torch.linalg.vector_norm(prediction-target)/torch.linalg.vector_norm(target))
       if k==1 and h not in checked:
        original=rollout(model,trajectories[initial],trajectories[36],frame,h,create_locs(device),device)
        np.testing.assert_allclose([mse,l2],original,rtol=2e-6,atol=1e-10);checked.add(h)
       rows.append(dict(split=split,system_index=si,k=system[0],beta=system[1],initial_id=initial,frame=frame,horizon=h,K=k,mse=mse,l2_relative=l2))
 assert checked=={1,5,50} and before==model_checksum(model)
 train_z=[];train_y=[]
 for system in SYSTEMS['train']:
  histories=torch.stack([load_trajectory(data,system,i)[list(CONDITION_FRAMES)] for i in range(50,90)])
  with torch.no_grad():zs=torch.cat([model.conditioning_encoder(chunk.to(device)).cpu() for chunk in histories.split(8)]).numpy()
  train_z.append(zs);train_y.extend([system]*40)
 train_z=np.stack(train_z);train_y=np.asarray(train_y);probes={}
 for k in [1,2,4]:
  tz=np.mean([np.roll(train_z,-j,axis=1) for j in range(k)],axis=0).reshape(-1,2)
  probes[str(k)]={}
  for split in SYSTEMS:
   selected=[r for r in latent_rows if r['K']==k and r['split']==split]
   probes[str(k)][split]=ridge_probe(tz,train_y,np.asarray([r['z'] for r in selected]),np.asarray([[r['k'],r['beta']] for r in selected]))
 assert before==model_checksum(model)
 aggregate=[]
 for split in SYSTEMS:
  for k in [1,2,4]:
   for h in [1,5,50]:
    selected=[x for x in rows if x['split']==split and x['K']==k and x['horizon']==h]
    aggregate.append(dict(split=split,K=k,horizon=h,n=len(selected),mse=float(np.mean([x['mse'] for x in selected])),l2_relative=float(np.mean([x['l2_relative'] for x in selected]))))
 result=dict(status='COMPLETE',checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),step=ck['step'],K1_equivalence_pass=True,model_unchanged=True,protocol=dict(targets=targets,donor_pool=pool,donor_rule='36 first, then fixed pool order excluding recipient; K independent histories from same system',budgets=[1,2,4],frames=[12,42,72,92],horizons=[1,5,50],interpretation='observation-budget adaptation; comparisons only at matched K',test_read=False,historical_evaluation=True),aggregate=aggregate,probe=probes,probe_policy="train-fitted same-K cyclic aggregation of distinct histories50..89; evaluation donors exclude recipient",cases=rows,latents=latent_rows)
 a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(a.output)
 print(json.dumps(aggregate),flush=True)
if __name__=='__main__':main()
