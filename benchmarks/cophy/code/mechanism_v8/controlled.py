"""Isolated supervised readout interventions; existing v7 code/weights stay immutable.

The original source-cache VERSION is retained inside the loaded implementation.
Every new head additionally binds the exact controlled source and recipe.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

VERSION = 'cophy-mechanism-v8-1'


def load_controlled(core, recipe):
    core = Path(core).resolve()
    source = core.read_text()
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError('Unexpected parent implementation: '+old[:100])
        source = source.replace(old, new)
    replace("torch.manual_seed(seedof('v6-", "torch.manual_seed(module_seed('v6-", 5)
    replace("null_dropout=.1 if data.learned else 0.,", "mechanism_recipe=CONTROL, mechanism_implementation_sha256=CONTROL_SHA, null_dropout=CONTROL['dropout'] if data.learned else 0.,")
    replace("torch.rand(len(ix), device=args.device) < .1 if data.learned else None", "torch.rand(len(ix), device=args.device) < CONTROL['dropout'] if data.learned else None")
    replace("self.cell = nn.GRUCell(width+2*dims, width)", "self.cell = nn.GRUCell(width+2*dims+(65 if CONTROL['memory']=='every_step' else 0), width)")
    replace("self.cell(torch.cat((agg, position, velocity), -1).reshape(b*slots, -1)", "self.cell(torch.cat((agg, position, velocity, memory, available) if CONTROL['memory']=='every_step' else (agg, position, velocity), -1).reshape(b*slots, -1)")
    replace("head_history_access='support is read only for recurrent-state initialization; original v4.1/v4.4 head',", "head_history_access=CONTROL['memory'],")
    name='_controlled_'+hashlib.sha256((str(core)+json.dumps(recipe,sort_keys=True)).encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_loader(name,loader=None);module=importlib.util.module_from_spec(spec)
    module.__file__=str(core);module.CONTROL=dict(recipe)
    module.CONTROL_SHA=hashlib.sha256((source+Path(__file__).read_text()).encode()).hexdigest()
    def module_seed(label):
        if recipe['initialization']=='legacy':
            return int.from_bytes(hashlib.sha256(('xep-head0:'+label.removeprefix('v6-')).encode()).digest()[:4],'little')
        return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')
    module.module_seed=module_seed
    sys.modules[name]=module
    exec(compile(source,str(core),'exec'),module.__dict__)
    return module


def run(spec_path, device, smoke=False):
    import torch
    spec=json.loads(Path(spec_path).read_text());out=Path(spec['out']);out.mkdir(parents=True,exist_ok=True)
    parent=Path(spec['parent_readout']);core=Path(spec['core'])
    if hashlib.sha256(core.read_bytes()).hexdigest()!=spec['core_sha256']:raise ValueError('Changed parent code')
    for name in ('codes_train.npz','codes_val.npz','codes_complete.json'):
        src=parent/name;dst=out/name
        if not src.exists():raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            if dst.resolve()!=src.resolve():raise ValueError('Alias differs')
        else:dst.symlink_to(src)
    m=load_controlled(core,spec['recipe'])
    args=argparse.Namespace(out=str(out),base=spec['base'],scene=spec['scene'],reference='learned',
        root=spec['root'],device=device,supports=3,epochs=100)
    torch.set_num_threads(2)
    # A real batch checks the new recurrent input shape. No optimizer or source update.
    if smoke:
        m.smoke(args)
        print(json.dumps({'status':'PASS','weights_discarded':True,'source_optimizer_steps':0}))
        return
    m.train(args);m.probe(args)
    m.write(out/'mechanism_complete.json',dict(status='COMPLETE',version=VERSION,recipe=spec['recipe'],
        source_optimizer_steps=0,source_codes_sha256=m.sha(out/'codes_complete.json'),
        head_sha256=m.sha(out/'S3/learned/selected.pt'),paired_probe=m.sha(out/'probes.json'),test_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.spec,a.device,a.smoke)
