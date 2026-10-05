"""Build task descriptions only; no remote calls, subprocesses, or training."""

import os
import json
from pathlib import Path
import hashlib
R=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"));HERE=Path(__file__).resolve().parent;E=HERE.parents[2];REMOTE=R+'/source/cophy_complete_v7/legacy'

def write(name,doc):
 (HERE/name).write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')

def sig(file):return hashlib.sha256((HERE/file).read_bytes()).hexdigest()

def tasks():
 rows=[]
 for scene in ('balls','blocktower'):
  for method in ('Monolithic','Base','Cross'):
   mono=method=='Monolithic';parent=f'{R}/monolithic_jepa_v6_6/sources/{scene}/checkpoint_100.pt' if mono else f'{R}/latent_v6_2_sigcal/weight02/sources/{scene}/JEPA/{method}/checkpoint_50.pt'
   runtime=f'{R}/source/'+('monolithic_jepa_v6_6' if mono else 'latent_v6_2_sig02');out=f'{R}/cophy_complete_v7/sources/JEPA/{scene}/{method}'
   args=['--runtime-dir',runtime,'--parent-checkpoint',parent,'--scene',scene,'--method',method,'--out',out,'--device','{device}']
   for budget in ((150,) if mono else (100,150)):
    cmds=([['python','-u',REMOTE+'/jepa_continue.py',phase]+args for phase in ('prepare','smoke')] if budget==(150 if mono else 100) else [])+[['python','-u',REMOTE+'/jepa_continue.py','train']+args+['--through',str(budget)]]
    rows.append(dict(id=f'jepa-{scene}-{method}-{budget}',resource='gpu',commands=cmds,dependencies=[parent] if budget==(150 if mono else 100) else [out+'/budget_100_complete.json'],marker=out+f'/budget_{budget}_complete.json',priority=110 if budget==100 else 210,status='IMPLEMENTED_PARENT_RNG_NEEDS_RECEIPT' if not mono else 'IMPLEMENTED_NOT_REMOTE_VERIFIED',parent_rng_instruction='Append --parent-rng-index from the audited old worker/launch receipt when parent has four saved streams; no heuristic in runner' if not mono else 'saved single active stream index0'))
 write('jepa_continuations.json',dict(version='cophy-v7-jepa-continued-spec-1',source_code_sha256=sig('jepa_continue.py'),helper_sha256=sig('jepa_source.py'),tasks=rows,test_read=False))
 rows=[]
 for scene in ('balls','collision','blocktower'):
  packed=f'{R}/cophy_complete_v7/prepared/supervised_source/{scene}'
  rows.append(dict(id=f'supervised-{scene}-source-pack',resource='cpu',commands=[['python','-u',REMOTE+'/supervised_continue.py','prepare-data','--root',R,'--scene',scene,'--packed',packed,'--out',packed]],dependencies=[],marker=packed+'/manifest.json',priority=1,status='IMPLEMENTED_NOT_REMOTE_VERIFIED'))
  for method in ('Native','A','Random'):
   v51=scene=='collision' and method!='Native';actual=('Both-new' if method=='A' else 'Random-Both-new') if v51 else method
   parentdir=f'{R}/source_formation_v5_1/collision/source/runs/{actual}' if v51 else f'{R}/runs_seed0/'+('blocktower_gate1' if scene=='blocktower' else scene)+'/'+method
   parent=parentdir+('/latest.pt' if v51 else '/latest_resume.pt');selected=parentdir+('/selected.pt' if v51 else '/model_state_dict.pt');out=f'{R}/cophy_complete_v7/sources/supervised/{scene}/{method}'
   args=['--root',R,'--scene',scene,'--method',method,'--profile','v51' if v51 else 'v3','--packed',packed,'--parent-checkpoint',parent,'--parent-selected',selected,'--out',out,'--device','{device}']
   if v51:args+=['--v51-code',R+'/discovery_v4_1/source_formation_v51.py']
   for budget in (100,150):
    commands=([['python','-u',REMOTE+'/supervised_continue.py',p]+args for p in ('prepare','smoke')] if budget==100 else [])+[['python','-u',REMOTE+'/supervised_continue.py','train']+args+['--through',str(budget)]]
    rows.append(dict(id=f'supervised-{scene}-{method}-{budget}',resource='gpu',commands=commands,dependencies=[packed+'/manifest.json',parent,selected] if budget==100 else [out+'/budget_100_complete.json'],marker=out+f'/budget_{budget}_complete.json',priority=110 if budget==100 else 210,status='IMPLEMENTED_NOT_REMOTE_VERIFIED',source_method=actual,readout_checkpoint=out+f'/selected_budget{budget}.pt',source_budget=budget,head_method_argument=actual,source_selection='original v51 every epoch' if v51 else 'original v3 every5; includes previous budget best'))
 write('supervised_continuations.json',dict(version='cophy-v7-original-supervised-continued-spec-1',source_code_sha256=sig('supervised_continue.py'),helper_sha256=sig('jepa_source.py'),tasks=rows,test_read=False))


def assets():
 # Import exact final frozen checkpoint/reader paths, not filenames inferred from labels.
 rows=[]
 p=E/'receipts/Monolithic18_Final_summary_20260912.json';doc=json.loads(p.read_text())
 for run in doc['runs']:
  rows.append(dict(family='JEPA',scene=run['scene'],method=run['method'],budget=run['source_epochs'],selected_source_epoch=run['source_epoch'],checkpoint=run['checkpoint'],checkpoint_sha256=run['source_checkpoint_sha256'],source_updates=run['source_updates'],readout_out=run['out'],head_budget=100,fullval=run['fullval_results_path'],fullval_sha256=run['fullval_results_sha256'],reuse='SOURCE_HEAD_FULLVAL_PROBE_COMPLETE',head_context='U128; Mono joint context; Split [donorP64,recipientT64]',evidence=str(p)))
 p=E/'receipts/Collision_Budget150_Final_20260912.json';doc=json.loads(p.read_text())
 for run in doc['summary']['runs']:
  rows.append(dict(family='JEPA',scene='collision',method=run['method'],budget=150,checkpoint=run['checkpoint'],checkpoint_sha256=run['source_checkpoint_sha256'],source_updates=run['source_updates'],readout_out=run['out'],head_budget=100,fullval=run['fullval_results_path'],fullval_sha256=run['fullval_results_sha256'],reuse='SOURCE_HEAD_FULLVAL_PROBE_COMPLETE',route_note='Cross-all has focal1-50/all51-150; not new fixed-all budget curve' if 'Cross' in run['method'] else None,evidence=str(p)))
 for scene in ('balls','collision','blocktower'):
  for method in ('Native','A','Random'):
   v51=scene=='collision' and method!='Native';actual=('Both-new' if method=='A' else 'Random-Both-new') if v51 else method
   source=f'{R}/source_formation_v5_1/collision/source/runs/{actual}/selected.pt' if v51 else f'{R}/runs_seed0/'+('blocktower_gate1' if scene=='blocktower' else scene)+f'/{method}/model_state_dict.pt'
   rows.append(dict(family='supervised',scene=scene,method=method,source_method=actual,budget=50,checkpoint=source,continuation_checkpoint=str(Path(source).with_name('latest.pt' if v51 else 'latest_resume.pt')),reuse='SOURCE_OFFICIAL_PROBE_COMPLETE; BLOCKTOWER_NEW_HEAD_GAP' if scene=='blocktower' else 'SOURCE_HEAD_FULLVAL_PROBE_COMPLETE_OLD_READOUT',head_context='old U32 source, legacy readout table; new generic v7 head is a separate profile',profile='v51' if v51 else ('v3-gate1' if scene=='blocktower' else 'v3')))
 write('reuse_assets.json',dict(version='cophy-v7-qualified-legacy-reuse-1',test_read=False,full_validation={'balls':2000,'collision':4000,'blocktower':8088},rows=rows,not_reusable_as_strict_controls=['JEPA Random-Both for Cross-only attribution','Collision focal-to-all 50-to100 as fixed-route budget curve','Native-FT/MQ as original Native','source best weights as optimizer/RNG continuation'],pending=['remote realbatch verification of new wrappers','parent multi-CUDA RNG mapping for original split JEPA50','source/head final summaries of newly scheduled tasks']))

if __name__=='__main__':tasks();assets()
