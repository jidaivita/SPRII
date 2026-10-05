"""Collect all fixed mechanism cells, including negative results; no model selection."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def interval(delta):
 if not len(delta):return None
 rng=np.random.default_rng(20260913);samples=[]
 for _ in range(1000):samples.append(float(delta[rng.integers(len(delta),size=len(delta))].mean()))
 return dict(mean=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),unit='recipient',bootstrap_replicates=1000)

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 plan=read(a.plan);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];reports={};errors=[];pairs=[];probes=[];groups={}
 for s in plan['diagnostics']:
  path=Path(s['out'])/'results.json'
  if not path.exists():errors.append(s['id']);continue
  r=read(path)
  if r.get('status')!='COMPLETE':errors.append(s['id']);continue
  reports[s['id']]=r
  key=(s['phase'],s['family'],s['scene'],s['source_budget'],json.dumps(s.get('recipe'),sort_keys=True))
  groups.setdefault(key,{})[s['role']]=s
  for arm,v in r['cohorts'].items():rows.append(dict(id=s['id'],phase=s['phase'],family=s['family'],scene=s['scene'],role=s['role'],source_budget=s['source_budget'],arm=arm,**v))
  for channel,pr in r['probes'].items():
   for parameter,v in pr['fields'].items():probes.append(dict(id=s['id'],family=s['family'],scene=s['scene'],role=s['role'],channel=channel,parameter=parameter,**v))
 if errors:
  (out/'progress.json').write_text(json.dumps(dict(status='PENDING',complete=len(reports),missing=errors),indent=2));raise SystemExit(75)
 def load(s):return np.load(Path(s['out'])/'per_recipient.npz',allow_pickle=False)
 for key,g in groups.items():
  for left,right in [('Native','Structure'),('Native','A'),('Structure','A'),('Random','A'),('Structure','Align'),('Structure','Both'),('A','Both'),('Random-Both','Both')]:
   if left not in g or right not in g:continue
   with load(g[left]) as x,load(g[right]) as y:
    lut={q:i for i,q in enumerate(y['ids'])};ix=np.asarray([i for i,q in enumerate(x['ids']) if q in lut]);iy=np.asarray([lut[x['ids'][i]] for i in ix])
    one=x['correct_S3__mse'][ix];two=y['correct_S3__mse'][iy];stat=interval(one-two)
    pairs.append(dict(phase=key[0],family=key[1],scene=key[2],source_budget=key[3],recipe=key[4],baseline=left,method=right,recipients=len(ix),baseline_mse=float(one.mean()),method_mse=float(two.mean()),reduction_percent=float(100*(one.mean()-two.mean())/one.mean()),paired_delta=stat))
 def table(filename,data):
  keys=list(dict.fromkeys(k for r in data for k in r))
  with (out/filename).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data])
 table('utility.csv',rows);table('probes.csv',probes);table('paired_effects.csv',pairs)
 # Paired difference-in-differences across reader recipes, same recipients.
 interactions=[]
 for scene in ('balls','collision','blocktower'):
  candidates=[(k,g) for k,g in groups.items() if k[1:4]==('CoPhyNet',scene,50) and 'A' in g and 'Native' in g]
  default=next(((k,g) for k,g in candidates if k[0]=='supervised50_reference'),None)
  if default is None:continue
  for k,g in candidates:
   if k==default[0]:continue
   ds=[];ids=None
   for pair in (default[1],g):
    with load(pair['Native']) as n,load(pair['A']) as ar:
     if not np.array_equal(n['ids'],ar['ids']):raise ValueError('Head interaction requires identical recipients')
     if ids is not None and not np.array_equal(ids,n['ids']):raise ValueError('Head recipe cohorts differ')
     ids=n['ids'].copy();ds.append(n['correct_S3__mse']-ar['correct_S3__mse'])
   interactions.append(dict(scene=scene,recipe=k[4],new_recipe_minus_v7_A_advantage=interval(ds[1]-ds[0])))
 table('readout_interactions.csv',interactions)
 text=['# CoPhy mechanism analysis v8','',f'Completed {len(reports)} fixed checkpoint/reader assays; seed0; train/validation only.','',
 'Source100 is the primary cross-method mechanism budget. Source50 supervised reader factorial diagnoses the old/new reversal separately.',
 'All physical probes are trained on train objects; memory and recurrent probes include learned downstream processing. No result establishes exclusive physical encoding or full causal mediation.',
 'S5/S8 are frozen S3-head input sensitivity, not a newly trained sample-efficiency claim. Native capacity/path differences remain reported in v7.',
 '', '|Phase|Family|Scene|Comparison|Error reduction|','|---|---|---|---|---:|']
 for r in pairs:text.append(f"|{r['phase']}|{r['family']}|{r['scene']}|{r['method']} vs {r['baseline']}|{r['reduction_percent']:.3f}%|")
 text+=['','The complete CSVs retain every reader recipe, diagnostic arm, parameter and negative result. Existing official CoPhy task results and old FT/MQ results remain separate.']
 (out/'REPORT.md').write_text('\n'.join(text)+'\n');(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE',assays=len(reports),comparisons=pairs,readout_interactions=interactions,test_read=False),indent=2))
 (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',assays=len(reports),plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),test_read=False)))

if __name__=='__main__':main()
