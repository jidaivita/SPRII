"""Report every registered replicate, including reversals, without selecting a seed."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();plan=read(a.plan);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];groups={};missing=[];probes=[]
 for s in plan['diagnostics']+plan['seed0_references']:
  folder=Path(s['out'])
  if not (folder/'complete.json').exists():missing.append(s['id']);continue
  result=read(folder/'results.json')
  with np.load(folder/'per_recipient.npz',allow_pickle=False) as z: ids=z['ids'].copy();mse=z['correct_S3__mse'].copy()
  key=(s['family'],s['scene'],s['source_budget'],json.dumps(s.get('recipe'),sort_keys=True))
  row=dict(id=s['id'],family=s['family'],scene=s['scene'],source_budget=s['source_budget'],seed=s['replicate_seed'],role=s['role'],recipe=s.get('recipe'),mse=float(mse.mean()),recipients=len(ids),cohort_sha256=hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest(),result_sha256=hashlib.sha256((folder/'results.json').read_bytes()).hexdigest())
  rows.append(row);groups.setdefault(key,{}).setdefault(s['replicate_seed'],{})[s['role']]=(row,ids,mse)
  for channel,value in result['probes'].items():
   for param,metric in value['fields'].items():probes.append(dict(id=s['id'],seed=s['replicate_seed'],channel=channel,parameter=param,**metric))
 if missing:(out/'progress.json').write_text(json.dumps(dict(status='PENDING',missing=missing,complete=len(rows)),indent=2));raise SystemExit(75)
 effects=[];summary=[]
 for key,seeds in groups.items():
  for baseline,method in [('Native','A'),('Structure','A'),('Random','A')]:
   local=[]
   for seed,models in sorted(seeds.items()):
    if baseline not in models or method not in models:continue
    n,ni,nm=models[baseline];r,ri,rm=models[method]
    if not np.array_equal(ni,ri):raise ValueError('Unmatched within-seed recipients')
    gain=100*(1-rm.mean()/nm.mean());record=dict(family=key[0],scene=key[1],source_budget=key[2],recipe=key[3],seed=seed,baseline=baseline,method=method,baseline_mse=float(nm.mean()),method_mse=float(rm.mean()),gain_percent=float(gain));effects.append(record);local.append(record)
   if local:
    vals=np.array([v['gain_percent'] for v in local]);summary.append(dict(family=key[0],scene=key[1],source_budget=key[2],recipe=key[3],baseline=baseline,method=method,seeds=[v['seed'] for v in local],gain_each_percent=vals.tolist(),mean_gain_percent=float(vals.mean()),sample_sd_percent=float(vals.std(ddof=1)) if len(vals)>1 else None,positive_seeds=int((vals>0).sum())))
 for filename,data in [('rows.csv',rows),('effects.csv',effects),('probes.csv',probes)]:
  keys=list(dict.fromkeys(k for r in data for k in r))
  with (out/filename).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in data])
 report=['# CoPhy seed replication v9','', 'All registered seeds are retained. Source100 replicas vary source and readout random seeds; source50 reader replicas keep the original source frozen and vary only head randomness. Validation only. Three seeds do not establish universal generalization.','', '|Family/scene|Comparison|Source budget|Recipe|Seeds gains %|Mean ± sample SD|','|---|---|---:|---|---|---|']
 for r in summary:report.append(f"|{r['family']}/{r['scene']}|{r['method']} vs {r['baseline']}|{r['source_budget']}|{r['recipe']}|{r['gain_each_percent']}|{r['mean_gain_percent']:.3f} ± {r['sample_sd_percent']}|")
 (out/'REPORT.md').write_text('\n'.join(report)+'\n');(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE',rows=len(rows),comparisons=summary,test_read=False),indent=2));(out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',rows=len(rows),plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),test_read=False)))
if __name__=='__main__':main()
