"""Recognize exactly Align and Random-Both sources; do not relabel other roles."""
from pathlib import Path
import hashlib,json,sys,types,os
P=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/source/cophy_complete_v7/common/readout.py'))
EXPECTED='929815334f7b0773ea8bbc8b06d77926d7b539ab8d50a6498ddde5a3817bf870'
s=P.read_text()
if hashlib.sha256(s.encode()).hexdigest()!=EXPECTED:raise ValueError('Parent changed')
old="'Both':{'Both'},'A':{'A','Cross','Both'},'Random':{'Random'}"
new="'Both':{'Both'},'Align':{'Align'},'Random-Both':{'Random-Both'},'A':{'A','Cross','Both'},'Random':{'Random'}"
if s.count(old)!=1:raise ValueError('Role registry changed')
s=s.replace(old,new)
args=sys.argv[1:]
if '--out' not in args:raise ValueError('Missing output binding')
out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
r=dict(status='APPLIED',repair='v8-role-registry-1',original_sha256=EXPECTED,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),effective_code_sha256=hashlib.sha256(s.encode()).hexdigest(),only_change='add exact Align->Align and Random-Both->Random-Both roles',source_optimizer_steps=0,test_read=False)
rp=out/'role_registry_repair1.json'
if rp.exists() and json.loads(rp.read_text())!=r:raise ValueError('Repair changed')
tmp=rp.with_name(rp.name+'.tmp');tmp.write_text(json.dumps(r,indent=2));os.replace(tmp,rp)
sys.argv=[str(P),*args];sys.path.insert(0,str(P.parent));m=types.ModuleType('__main__');m.__file__=str(P);sys.modules['__main__']=m;exec(compile(s,str(P),'exec'),m.__dict__)
