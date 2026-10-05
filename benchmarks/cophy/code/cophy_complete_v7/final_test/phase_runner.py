"""Run one bounded v7 finalization stage in the existing four-GPU dispatcher.

Semantic receipts retain their meaning; a separate COMPLETE envelope is made
only after the specific stage has really finished. No scheduler edits needed.
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys
from runtime import artifact, read, sha, verify_freeze, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('specification','fit','freeze','extract','produce','bundle','evaluate','collect'))
    p.add_argument('--root',default=(os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")));p.add_argument('--entry');p.add_argument('--scene')
    p.add_argument('--device',default='cpu');p.add_argument('--marker',required=True)
    a=p.parse_args();root=Path(a.root);run=root/'cophy_complete_v7';out=run/'final_test';code=Path(__file__).parent
    out.mkdir(parents=True,exist_ok=True);spec_path=out/'specification.json';freeze_path=out/'freeze/freeze.json'
    def command(name,*args):subprocess.run([sys.executable,str(code/name),*map(str,args)],check=True,cwd=root)
    if a.stage=='specification':
        rules={}
        for scene,(folder,suffix) in {'balls':('ballsCF','4'),'collision':('collisionCF','normal'),'blocktower':('blocktowerCF','3_normal')}.items():
            source=root/'source';feature=root/'latent_v6/features'/scene/'train/manifest.json';fm=read(feature)
            checkpoint=Path(fm['checkpoint']['path'])
            rules[scene]=dict(query_frames=3,supports=3,history_domain='same_test_split',
                official_split_path=str(source/'dataloaders/splits'/f'{folder}_test_{suffix}.txt'),
                feature_producer_path=str(code/'produce_data.py'),pose_producer_path=str(code/'produce_data.py'),
                field_auditor_path=str(source/'cophy_fields.py'),
                development_id_files=[artifact(root/'latent_v6/features'/scene/s/'ids.json') for s in ('train','val')],
                archive_receipt=artifact(root/'raw/download_complete.json'),archive_path=str(root/'raw/cophy_224.tar.gz'),
                source_root=str(source),frontend_checkpoint=artifact(checkpoint),frontend_checkpoint_sha256=sha(checkpoint),
                feature_training_manifest=artifact(feature),visual_source_files=[artifact(source/x) for x in
                    ('derendering/model.py','dataloaders/utils.py','cf_learning/model.py','cophy_fields.py','cophy_metadata.py','cophy_protocol.py')])
        rules_path=out/'scene_rules.json';write(rules_path,rules,immutable=True)
        command('build_spec.py','--presentation-manifest',run/'presentation_manifest.json',
            '--reference-assets',root/'source/cophy_complete_v7/common/reference_official_assets.json',
            '--scene-rules',rules_path,'--protocol',code/'PROTOCOL.md',
            '--main-summary-dir',run/'summary/main','--references-summary-dir',run/'summary/references',
            '--wrong-gravity-complete',run/'wrong_gravity/comparison/complete.json',
            '--probe-fit-root',out/'probe_fits','--code-root',root/'source','--out',spec_path)
        receipt=spec_path;expected='SPECIFICATION_ONLY_NOT_TEST_PERMIT'
    elif a.stage=='fit':
        spec=read(spec_path);entries=[e for e in spec['learned_entries'] if e['id']==a.entry]
        if len(entries)!=1:raise ValueError('Unknown fixed probe entry')
        e=entries[0];pre=read(root/'prepared_v3'/e['scene']/'training_preflight.json');raw=pre['artifacts']['raw_relations_train']
        command('fit_probes.py','--readout',e['readout'],'--core',e['core'],'--base',e['base'],
            '--raw-train-relations',raw['path'],'--raw-train-sha256',raw['sha256'],
            '--out',Path(e['probe_fit']).parent,'--device',a.device)
        receipt=Path(e['probe_fit']);expected='TRAIN_FIT_COMPLETE'
    elif a.stage=='freeze':
        command('freeze_gate.py','freeze','--specification',spec_path,'--out',out/'freeze')
        receipt=freeze_path;expected='FROZEN_FOR_FINAL_TEST'
    else:
        verify_freeze(freeze_path,verify_all=False)
        write(out/'TEST_ACCESS_STARTED.json',dict(status='TEST_ACCESS_STARTED',freeze_sha256=sha(freeze_path),
            test_read=True,scope='formal test phase has begun; authoritative over the old dispatcher training-only status field'),immutable=True)
        scene_out=out/'data'/str(a.scene)
        if a.stage in ('extract','produce'):
            command('produce_data.py',a.stage,'--freeze',freeze_path,'--scene',a.scene,'--out',scene_out,'--device',a.device)
            receipt=scene_out/('extraction.json' if a.stage=='extract' else 'inputs/producer_receipt.json');expected='COMPLETE'
        elif a.stage=='bundle':
            command('prepare_bundle.py','--freeze',freeze_path,'--scene',a.scene,
                '--producer-receipt',scene_out/'inputs/producer_receipt.json','--out',scene_out/'bundle')
            receipt=scene_out/'bundle/prepared.json';expected='PREPARED'
        elif a.stage=='evaluate':
            frozen=read(freeze_path);entries=[e for e in frozen['learned_entries']+frozen['references'] if e['id']==a.entry]
            if len(entries)!=1:raise ValueError('Unknown fixed evaluation entry')
            scene_out=out/'data'/entries[0]['scene']
            command('evaluate.py','--freeze',freeze_path,'--entry',a.entry,'--bundle',scene_out/'bundle/prepared.json',
                '--out',out/'results'/a.entry,'--device',a.device)
            receipt=out/'results'/a.entry/'complete.json';expected='COMPLETE'
        else:
            command('collect_results.py','--freeze',freeze_path,'--results',out/'results','--out',out/'summary')
            receipt=out/'summary/complete.json';expected='COMPLETE'
    value=read(receipt)
    if value.get('status')!=expected:raise ValueError('Stage has not met its semantic completion: '+str(receipt))
    write(a.marker,dict(status='COMPLETE',stage=a.stage,entry=a.entry,scene=a.scene,
        receipt=artifact(receipt),semantic_status=expected,test_read=a.stage in ('extract','produce','bundle','evaluate','collect')),
        immutable=True)
    print('STAGE_COMPLETE',a.stage,a.entry or a.scene or '')


if __name__=='__main__':main()
