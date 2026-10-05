"""Freeze all final choices, then prepare test with the existing evaluation rules.

These commands are inert until explicitly invoked. No test body is opened by
the freeze command. Preparation is resumable under one frozen global plan.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import torch
from cophy_prepare_artifacts import SPECS, read, artifact, frozen_write, checked_cache
from cophy_protocol import digest, verify_preflight, verify_adapter_binding
from cophy_relations import read_artifact, artifact_path
from cophy_fields import inspect_episode
from cophy_manifests import qualify_objects, build_donor_manifests, parameter_documents


def freeze(specification,output):
    """Specification names final run checkpoints and their completed validation reports."""
    entries=read(specification)['scenes']
    if set(entries)!=set(SPECS):raise ValueError('Freeze must account for all three original scenes')
    result={'status':'FROZEN_FOR_TEST','adapter_version':'cophy-pt16-v3','scenes':{},'test_read':False}
    for scene,entry in entries.items():
        preflight=entry['preflight'];verify_preflight(preflight);bound=verify_adapter_binding(preflight)
        if scene!=bound['scene']:raise ValueError('Scene mismatch in final selection')
        budget=read(entry['budget_decision'])['epochs']
        if budget not in [25,50]:raise ValueError('Unregistered training budget')
        methods=bound['allowed_methods'];eligible='A' in methods
        expected={(m,s) for m in methods for s in ([0] if m=='Param-known' else [0,1,2])}
        found=set();models=[];a_settings=set()
        for selected in entry['checkpoints']:
            state=torch.load(selected['path'],map_location='cpu',weights_only=False)
            config=state['run_config'];key=(config['method'],config['seed'])
            if key not in expected or key in found:raise ValueError('Missing/duplicate/unregistered final method-seed')
            if config['data_binding']['preflight_sha256']!=digest(preflight) or config['code_sha256']!=bound['code_sha256']:
                raise ValueError('Final checkpoint is not bound to this data/code')
            log=Path(selected['path']).parent/'val.txt'
            observations=[json.loads(line) for line in log.read_text().splitlines()]
            if max(r['epoch'] for r in observations)!=budget:raise ValueError('Final run did not finish its common budget')
            candidates=[r for r in observations if r['epoch'] in [10,15,20,25,30,35,40,45,50] and r['epoch']<=budget]
            best=min(candidates,key=lambda r:(r['mse'],r['epoch']))
            if state['epoch']!=best['epoch']:raise ValueError('Checkpoint is not the registered validation selection')
            report=read(selected['validation'])
            if report['checkpoint_sha256']!=digest(selected['path']) or report['split']!='val':
                raise ValueError('Need this selected checkpoint\'s completed validation evaluation')
            if key[0] in {'A','Random'}:a_settings.add((config['lambda_x'],config['lambda_p']))
            models.append({'method':key[0],'seed':key[1],'epoch':state['epoch'],
                'checkpoint':artifact(selected['path']),'validation':artifact(selected['validation']),
                'selection_log':artifact(log),'lambda_x':config['lambda_x'],'lambda_p':config['lambda_p']})
            found.add(key)
        if found!=expected:raise ValueError('Required final method-seed runs are incomplete')
        if len(a_settings)>1:raise ValueError('Final A/Random settings differ across seeds')
        # Preserve default/limited-candidate validation evidence, including negatives.
        diagnostics=[artifact(path) for path in entry.get('diagnostic_reports',[])]
        result['scenes'][scene]={'preflight':artifact(preflight),'budget_decision':artifact(entry['budget_decision']),
            'checkpoints':models,'diagnostic_reports':diagnostics,'epochs':budget,'relation_eligible':eligible}
    path=frozen_write(Path(output)/'global_freeze.json',result)
    for scene,entry in result['scenes'].items():
        artifacts={'global_freeze':artifact(path),'preflight':entry['preflight'],'budget_decision':entry['budget_decision']}
        for i,model in enumerate(entry['checkpoints']):
            for key in ['checkpoint','validation','selection_log']:artifacts[f'{key}_{i}']=model[key]
        permit={'status':'FROZEN_FOR_TEST','stage':'preparation','scene':scene,
            'preflight_sha256':entry['preflight']['sha256'],'artifacts':artifacts,
            'checkpoint_sha256':[m['checkpoint']['sha256'] for m in entry['checkpoints']],
            'test_read':False}
        frozen_write(Path(output)/scene/'preparation_permit.json',permit)
    return path


def extract(root,freeze_dir):
    root=Path(root);freeze_dir=Path(freeze_dir);plan=read(freeze_dir/'global_freeze.json')
    allowed=set();archive_sha=None;data_roots=set()
    for scene,entry in plan['scenes'].items():
        preflight=entry['preflight']['path'];permit=freeze_dir/scene/'preparation_permit.json'
        verify_preflight(preflight,release=permit,require_release=True)
        data_roots.add(str(Path(read(preflight)['input_profile']['dataset_dir']).parent))
        receipt=read_artifact(preflight,'archive_receipt')
        if archive_sha is not None and archive_sha!=receipt['sha256']:raise ValueError('Scenes refer to different archives')
        archive_sha=receipt['sha256'];spec=SPECS[scene]
        ids=read_artifact(preflight,'splits')['test']['ids']
        for ident in ids:allowed.add((spec['folder'],ident) if scene=='collision' else (spec['folder'],str(spec['num_objects']),ident))
    if len(data_roots)!=1:raise ValueError('Three scenes must use the same qualified dataset root')
    data_root=Path(next(iter(data_roots)))
    complete=freeze_dir/'test_extraction.json'
    if complete.exists():
        old=read(complete)
        if old['global_freeze_sha256']!=digest(freeze_dir/'global_freeze.json'):raise ValueError('Different test freeze')
        return complete
    found=set();nfiles=0
    with tarfile.open(root/'raw/cophy_224.tar.gz','r|*') as stream:
        for member in stream:
            parts=Path(member.name).parts
            start=next((i for i,p in enumerate(parts) if p in {s['folder'] for s in SPECS.values()}),None)
            if start is None:continue
            parts=parts[start:];width=2 if parts[0]=='collisionCF' else 3
            if len(parts)<=width or tuple(parts[:width]) not in allowed:continue
            if '..' in parts or member.issym() or member.islnk():raise ValueError('Unsafe archive member')
            if not member.isfile():continue
            target=data_root/Path(*parts);target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out,1024*1024)
            found.add(tuple(parts[:width]));nfiles+=1
    if found!=allowed:raise ValueError('Missing official test episodes')
    return frozen_write(complete,{'status':'EXTRACTED_AFTER_GLOBAL_FREEZE','test_read':True,
        'global_freeze_sha256':digest(freeze_dir/'global_freeze.json'),'archive_sha256':archive_sha,
        'episodes':len(found),'files':nfiles})


def prepare_scene(root,freeze_dir,scene):
    root=Path(root);freeze_dir=Path(freeze_dir);folder=freeze_dir/scene
    permit=read(folder/'preparation_permit.json');preflight=permit['artifacts']['preflight']['path']
    verify_preflight(preflight,release=folder/'preparation_permit.json',require_release=True)
    verify_adapter_binding(preflight)
    extraction=read(freeze_dir/'test_extraction.json')
    if extraction['global_freeze_sha256']!=digest(freeze_dir/'global_freeze.json'):raise ValueError('Test extraction freeze mismatch')
    spec=SPECS[scene];splits=read_artifact(preflight,'splits');ids=splits['test']['ids']
    cache_path=root/'features'/f"{spec['cache']}_test_extracted_prop.pickle"
    cache=checked_cache(cache_path,ids,spec)
    raw=[]
    data_root=Path(read(preflight)['input_profile']['dataset_dir']).parent
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for value in pool.map(lambda ident:inspect_episode(data_root,scene,'test',ident),ids):raw.extend(value['rows'])
    audit=read_artifact(preflight,'audit');raw_audit=read_artifact(preflight,'raw_audit')
    branch=audit.get('gravity_branch','not_applicable');eligible='A' in audit['allowed_methods']
    fields=raw_audit['fields'];varying=raw_audit['varying_object_fields']
    if eligible:
        records=qualify_objects(raw,cache,scene=scene,split='test',field_indices=[fields.index(f) for f in varying],gravity_branch=branch)
        primary,wrong1=build_donor_manifests(records,scene=scene,split='test',physical_fields=varying)
    else:primary=wrong1={'status':'NOT_APPLICABLE','reason':'unreliable_global_gravity','rows':[]}
    artifacts=dict(permit['artifacts']);artifacts['test_extraction']=artifact(freeze_dir/'test_extraction.json')
    for name,value in [('raw_relations_test',raw),('test_primary',primary),('test_wrong_1',wrong1)]:
        artifacts[name]=artifact(frozen_write(folder/(name+'.json'),value))
    artifacts['cache_test']=artifact(cache_path)
    if 'Param-known' in audit['allowed_methods']:
        data=parameter_documents({'train':read_artifact(preflight,'raw_relations_train'),'test':raw},
            scene=scene,slots=spec['slots'],object_fields=fields,include_gravity=scene=='blocktower' and branch=='varying_verified')
        if data['test']['fields']!=read_artifact(preflight,'parameters_train')['fields']:
            raise ValueError('Test preparation changed train-only parameter normalization')
        artifacts['parameters_test']=artifact(frozen_write(folder/'parameters_test.json',data['test']))
    return frozen_write(folder/'test_release.json',dict(permit,stage='evaluation',artifacts=artifacts,test_read=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze_parser=sub.add_parser('freeze');freeze_parser.add_argument('--specification',required=True);freeze_parser.add_argument('--output',required=True)
    for name in ['extract','prepare-scene']:
        command=sub.add_parser(name);command.add_argument('--root',required=True);command.add_argument('--freeze-dir',required=True)
        if name=='prepare-scene':command.add_argument('--scene',choices=SPECS,required=True)
    args=parser.parse_args()
    if args.action=='freeze':result=freeze(args.specification,args.output)
    elif args.action=='extract':result=extract(args.root,args.freeze_dir)
    else:result=prepare_scene(args.root,args.freeze_dir,args.scene)
    print(result)
