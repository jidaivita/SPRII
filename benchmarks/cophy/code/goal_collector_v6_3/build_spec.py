"""Build a finite read-only inventory from the already approved task lists."""

import os
import hashlib
import importlib.util
import json
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
RUN = ROOT/'latent_v6_2_sigcal/weight02'


def read(p):
    return json.loads(Path(p).read_text())


def arg(command, name):
    return command[command.index(name)+1]


def main():
    path = EXT/'code/latent_v6_2_sig02/dispatcher.py'
    loader = importlib.util.spec_from_file_location('inventory_source_dispatcher', path)
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    primary = module.task_definitions()
    extra_path = EXT/'protocol/latent_extension_v6_3/manifest.json'
    extra = read(extra_path)
    full = read(EXT/'receipts/fullval_readout_dispatch_integration_20260912.json')['tasks']
    extfull_path = EXT/'protocol/latent_extension_fullval_v6_3/manifest.json'
    extfull = read(extfull_path)
    items = []
    manifests = [dict(path=str(RUN/'queue/manifest.json'), version=module.VERSION),
                 dict(path=str(ROOT/'latent_extensions_v6_3/manifest.json'), version=extra['version'],
                      sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest()),
                 dict(path=str(ROOT/'latent_extensions_v6_3/fullval_tail_manifest.json'), version=extfull['version'],
                      sha256=hashlib.sha256(extfull_path.read_bytes()).hexdigest())]
    for tasks, family_default, manifest in ((primary, 'JEPA', manifests[0]['path']),
                                             (extra['tasks'], None, manifests[1]['path'])):
        for t in tasks:
            commands = t['commands']
            if not commands: continue
            c = commands[-1]; marker = Path(t['marker'])
            family = t.get('family', family_default)
            if '--scene' not in c: continue
            scene = arg(c, '--scene')
            common = dict(family=family, scene=scene, manifest=manifest, task=t['id'], marker=str(marker))
            if 'train.py' in ' '.join(c) and 'train' in c and '--max-steps' not in c:
                items.append(dict(kind='source', method=arg(c, '--method'), folder=str(marker.parent), **common))
            elif family == 'JEPA' and marker.name == 'probes.json' and 'source50' in marker.parts:
                encode = commands[0]
                source = Path(arg(encode, '--checkpoint')).parent
                items.append(dict(kind='probe', method=source.name, folder=str(marker.parent),
                                  source=str(source), core=encode[2], **common))
            elif marker.name == 'complete.json' and '--reference' in c and arg(c, '--epochs') == '100':
                reference = arg(c, '--reference'); out = Path(arg(c, '--out'))
                if reference == 'learned' and out.name != 'source50': continue
                items.append(dict(kind='head' if reference == 'learned' else 'reference',
                                  reference=reference, method=out.parent.name if family == 'JEPA' else t.get('method', out.parent.name),
                                  supports=int(arg(c, '--supports')), base=arg(c, '--base'),
                                  folder=str(marker.parent), readout=str(out), core=c[2], **common))
    # Extension encode and probe are separate tasks, unlike the JEPA queue.
    for t in extra['tasks']:
        if t.get('kind') != 'probe': continue
        c=t['commands'][-1]; out=Path(arg(c,'--out'))
        if any(x['kind']=='probe' and x['folder']==str(out) for x in items): continue
        family=t['family']; scene=arg(c,'--scene'); method=out.parent.name
        source=ROOT/('latent_'+family.lower()+'_v6_3')/'sources'/scene/family/method
        items.append(dict(kind='probe', family=family, scene=scene, method=method, folder=str(out),
                          source=str(source), core=c[2], manifest=manifests[1]['path'], task=t['id'], marker=t['marker']))
    for t in full + extfull['tasks']:
        c=t.get('command',t.get('argv',[]))
        if 'evaluate' not in c: continue
        family=t.get('family','JEPA'); scene=arg(c,'--scene'); readout=Path(arg(c,'--readout'))
        reference=arg(c,'--reference') if '--reference' in c else 'learned'
        kind='fullval' if reference=='learned' else 'reference_fullval'
        items.append(dict(kind=kind, family=family, scene=scene, reference=reference,
            method=readout.parent.name if reference=='learned' else reference,
            supports=int(arg(c,'--supports')), folder=arg(c,'--out'), readout=str(readout),
            prepared=arg(c,'--prepared'), marker=t['marker'], task=t['id'],
            manifest=manifests[2]['path'] if family!='JEPA' else None,
            core=arg(c,'--family-core') if family!='JEPA' else str(ROOT/'source/latent_v6/readout.py'),
            evaluator=str(ROOT/'source/latent_v6/fullval_readout.py')))
    counts = {k:sum(x['kind']==k for x in items) for k in ('source','probe','head','reference','fullval','reference_fullval')}
    assert counts == dict(source=39,probe=39,head=52,reference=6,fullval=52,reference_fullval=6), counts
    # Source paths for learned heads/codes are taken from their paired probe records.
    probes={(x['family'],x['scene'],x['method']):x for x in items if x['kind']=='probe'}
    for x in items:
        if x['kind'] in ('head','fullval'):
            x['source']=probes[x['family'],x['scene'],x['method']]['source']
        if x['kind']=='reference': x['method']=x['reference']
    spec=dict(version='cophy-goal-collector-spec-v1',root=str(ROOT),counts=counts,manifests=manifests,
              source_versions=dict(JEPA='cophy-latent-v6.2-sig02',CPC='cophy-cpc-v6.3-sig02',RSSM='cophy-gaussian-rssm-v6.3'),
              core_versions=dict(JEPA='latent-relation-v6.2-frozen-P64-pose-prefix-readout',
                 CPC='cophy-cpc-v6.3-frozen-P64-pose-prefix-readout',RSSM='cophy-rssm-v6.3-frozen-P64-pose-prefix-readout'),
              full_rows=dict(balls=2000,collision=4000,blocktower=8088),items=items,
              old_tails=[dict(scene='balls',folder=str(ROOT/'supervised_tail_v6/balls_fullval'),heads=18,
                             version='supervised-balls-fullval-v6-1'),
                         dict(scene='collision',folder=str(ROOT/'supervised_tail_v6/collision_v51_fullval'),heads=15,
                             version='supervised-v51-collision-fullval-v6-1'),
                         dict(scene='blocktower',folder=str(ROOT/'supervised_tail_v6/blocktower_bound_runtime'),
                             methods=['Native','A','Random','Param-known'],version='supervised-blocktower-tail-v6-2')],
              scope='Existing fixed work only. Negative scientific results can be verified complete. No new scientific gate.',
              test_read=False)
    dest=EXT/'protocol/goal_collector_v6_3/spec.json'
    dest.write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(path=str(dest),counts=counts,items=len(items))))


if __name__=='__main__': main()
