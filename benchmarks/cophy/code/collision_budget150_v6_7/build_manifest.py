"""Build the bounded three-source continuation and readout DAG."""

import os
import hashlib,json
from pathlib import Path
E=Path(__file__).resolve().parents[2]
C=Path(__file__).resolve().parent
R=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
S=R/'source/collision_budget150_v6_7'
O=R/'collision_budget150_v6_7'
PY=str(R/'venv/bin/python')
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((E/'code/monolithic_jepa_v6_6_1/recovery_manifest.json').read_text())
refs=[x for x in old['readout_runs'] if x['scene']=='collision' and x['source_epochs']==100 and x['method'] in ('Monolithic','Base','Cross-all')]
assert len(refs)==3
dump(C/'reference100.json',dict(test_read=False,readout_runs=refs))
files=['source_continue.py','readout.py','collect_results.py','PROTOCOL.md','reference100.json']
bind={str(S/n):sha(C/n) for n in files}
for folder,names in [('monolithic_jepa_v6_6',['models.py','train.py']),('latent_v6_2_sig02',['models.py','train.py']),('collision_base100_v6_5',['source_base.py']),('collision_adapt_v6_4',['source_route.py','adapt_dispatcher.py']),('latent_extension_v6_3',['gpu_guard.py'])]:
 for name in names:bind[str(R/'source'/folder/name)]=sha(E/'code'/folder/name)
tasks=[];runs=[];finaldeps=[]
for entry in refs:
 method=entry['method'];kind=entry['kind'];out=O/'sources'/method
 runtime=R/'source'/('monolithic_jepa_v6_6' if kind=='mono' else 'latent_v6_2_sig02')
 trainer=runtime/'train.py' if kind=='mono' else R/'source'/('collision_base100_v6_5/source_base.py' if method=='Base' else 'collision_adapt_v6_4/source_route.py')
 args=['--method',method,'--runtime-dir',str(runtime),'--parent-trainer',str(trainer),'--parent-checkpoint',entry['checkpoint'],'--out',str(out),'--epochs','150','--protocol',str(S/'PROTOCOL.md'),'--device','{device}']
 commands=[[PY,str(S/'source_continue.py'),stage,*args,*(['--resume'] if stage=='train' else [])] for stage in ('prepare','smoke','train')]
 tasks.append(dict(id=method+'-source150',resource='gpu',dependencies=[],files=[entry['checkpoint']],marker=str(out/'complete.json'),commands=commands))
 ro=O/'readouts/collision'/method/'source150';ck=out/'checkpoint_150.pt'
 run=dict(scene='collision',method=method,kind=kind,source_epochs=150,checkpoint=str(ck),out=str(ro));runs.append(run)
 common=[PY,str(S/'readout.py')]
 commands=[common+['encode','--out',str(ro),'--device','{device}','--scene','collision','--base',str(R/'xep_discovery_collision_v4_4'),'--features',str(R/'latent_v6/features/collision'),'--checkpoint',str(ck),'--model-code',str(runtime/'models.py'),'--kind',kind,'--method',method,'--source-epochs','150'],common+['smoke','--out',str(ro),'--device','{device}'],common+['train','--out',str(ro),'--device','{device}','--epochs','100','--supports','3'],common+['evaluate','--out',str(ro),'--device','{device}','--prepared',str(R/'latent_v6_2/fullval_inputs/collision')]]
 full=str(ro/'S3/learned/fullval/complete.json');probe=str(ro/'probes.json')
 tasks.append(dict(id=method+'-readout150',resource='gpu',dependencies=[str(out/'complete.json')],files=[str(ck)],marker=full,commands=commands))
 tasks.append(dict(id=method+'-probe150',resource='cpu',dependencies=[full],files=[],marker=probe,commands=[common+['probe','--out',str(ro),'--device','cpu']]))
 finaldeps.extend([full,probe])
manifest=dict(version='collision-budget150-v6.7-1',test_read=False,source_continuations=3,new_head_training_jobs=3,readout_runs=runs,code_sha256=bind,tasks=tasks)
dump(C/'manifest.json',manifest)
# Collector lives in a separate manifest so it cannot pollute parsed readout runs.
tail=dict(version='collision-budget150-collector-v6.7-1',test_read=False,code_sha256={**bind,str(S/'manifest.json'):sha(C/'manifest.json')},tasks=[dict(id='collision-budget100-150-comparison',resource='cpu',dependencies=finaldeps,files=[],marker=str(O/'comparison/complete.json'),commands=[[PY,str(S/'collect_results.py'),'--manifest',str(S/'manifest.json'),'--reference-manifest',str(S/'reference100.json'),'--out',str(O/'comparison'),'--require-complete']])])
dump(C/'comparison_manifest.json',tail)
print(json.dumps(dict(tasks=len(tasks),heads=len(runs),source_continuations=3,manifest_sha256=sha(C/'manifest.json'))))
