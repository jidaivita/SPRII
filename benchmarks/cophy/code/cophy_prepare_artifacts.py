"""Build real-data feature/training receipts. This module never fabricates audit qualification."""
import argparse
import hashlib
import json
import pickle
import socket
from pathlib import Path
import numpy as np

from cophy_protocol import digest, verify_preflight, verify_adapter_binding
from cophy_manifests import qualify_objects, build_donor_manifests, parameter_documents

SPECS={
    'collision':dict(folder='collisionCF',num_objects=4,type='normal',suffix='normal',slots=4,frames=15,cache='collision_normal'),
    'balls':dict(folder='ballsCF',num_objects=4,type='normal',suffix='4',slots=9,frames=30,cache='balls_4'),
    'blocktower':dict(folder='blocktowerCF',num_objects=3,type='normal',suffix='3_normal',slots=4,frames=30,cache='blocktower_3_normal'),
}


def read(path):
    return json.loads(Path(path).read_text())


def frozen_write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'Frozen artifact differs; use a new version directory: {path}')
    path.write_text(text)
    return path


def artifact(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':digest(path)}


def code_hashes(source):
    return {str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*.py'))}


def official_ids(source,spec):
    result={}
    for split in ['train','val','test']:
        path=source/'dataloaders/splits'/f"{spec['folder']}_{split}_{spec['suffix']}.txt"
        ids=path.read_text().split()
        if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty/duplicate official split')
        result[split]={'ids':ids,'file':artifact(path)}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(result[a]['ids']) & set(result[b]['ids']): raise ValueError('Official primary split overlap')
    return result


def feature_preflight(root,scene,out):
    root=Path(root);out=Path(out);source=Path(__file__).resolve().parent;spec=SPECS[scene]
    download=read(root/'raw/download_complete.json')
    extraction=read(root/'train_val_extraction_complete.json')
    if download.get('phase')!='COMPLETE_VERIFIED_SEALED' or extraction.get('archive_sha256')!=download.get('sha256'):
        raise ValueError('Verified archive/extraction evidence missing')
    if extraction.get('dataset_host',socket.gethostname())!=socket.gethostname():
        raise ValueError('Prepare caches on the host holding the qualified local dataset')
    raw_path=root/'audit'/f'{scene}_raw_field_audit.json';raw=read(raw_path)
    splits=official_ids(source,spec)
    if (raw.get('scene')!=scene or raw.get('phase')!='RAW_FIELDS_INSPECTED' or raw.get('errors_count')!=0 or
        raw.get('episodes')!=len(splits['train']['ids'])+len(splits['val']['ids'])):
        raise ValueError('Full primary train/val field audit not complete')
    split_path=frozen_write(out/'splits.json',splits)
    profile={'scene':scene,'num_objects':spec['num_objects'],'type':spec['type'],
             'dataset_dir':str((Path(extraction.get('dataset_root',str(root/'data')))/spec['folder']).resolve())}
    result={'status':'PASS','stage':'features','scene':scene,'adapter_version':'cophy-pt16-v3',
        'input_profile':profile,'code_sha256':code_hashes(source),
        'artifacts':{'audit':artifact(raw_path),'splits':artifact(split_path),
            'derenderer':artifact(source/'ckpts/derendering'/spec['folder']/'model_state_dict.pt'),
            'protocol':artifact(source/'protocol/Adapter_Contract_v3.md'),
            'archive_receipt':artifact(root/'raw/download_complete.json'),
            'extraction_receipt':artifact(root/'train_val_extraction_complete.json')},
        'test_read':False,'training_authorized_by_this_receipt':False}
    path=frozen_write(out/'feature_preflight.json',result)
    verify_preflight(path,stage='features');verify_adapter_binding(path)
    return path


def checked_cache(path,ids,spec):
    # Locally generated cache, never an unverified downloaded pickle.
    with Path(path).open('rb') as stream: cache=pickle.load(stream)
    if set(cache)!=set(ids): raise ValueError('Cache does not cover exactly this official split')
    expected={'cache_version','presence_ab','presence_c','pose_ab','pose_c'}
    for ident,row in cache.items():
        if set(row)!=expected or row['cache_version']!='ab_c_float32_v2':
            raise ValueError('Unqualified or future-bearing cache record')
        for name,shape in [('presence_ab',(spec['slots'],)),('presence_c',(spec['slots'],)),
                ('pose_ab',(spec['frames'],spec['slots'],3)),('pose_c',(1,spec['slots'],3))]:
            x=np.asarray(row[name])
            if x.shape!=shape or x.dtype!=np.float32 or not np.isfinite(x).all():
                raise ValueError(f'Invalid cache shape/dtype/values: {ident}/{name}')
            if name.startswith('presence') and not np.isin(x,[0.,1.]).all():
                raise ValueError('Nonbinary cached presence')
    return cache


def training_preflight(root,scene,out,qualification_path,frontend_check_path):
    root=Path(root);out=Path(out);spec=SPECS[scene]
    feature_path=out/'feature_preflight.json'
    verify_preflight(feature_path,stage='features');base=verify_adapter_binding(feature_path)
    qualification=read(qualification_path);raw_path=root/'audit'/f'{scene}_raw_field_audit.json';raw=read(raw_path)
    if (qualification.get('status')!='PASS' or qualification.get('scene')!=scene or
        qualification.get('raw_audit_sha256')!=digest(raw_path)):
        raise ValueError('Scene qualification must refer to the actual full raw-field audit')
    branch=qualification.get('gravity_branch','not_applicable')
    if scene=='blocktower' and branch not in {'constant_verified','varying_verified','unreliable'}:
        raise ValueError('Unresolved gravity branch')
    eligible=scene!='blocktower' or branch!='unreliable'
    methods=['Native','A','Random','Param-known'] if eligible else ['Native']
    check=read(frontend_check_path)
    if (check.get('status')!='PASS' or check.get('scene')!=scene or check.get('split')!='train' or
        check.get('derenderer_sha256')!=base['artifacts']['derenderer']['sha256'] or
        check.get('feature_preflight_sha256')!=digest(feature_path)):
        raise ValueError('Need this scene\'s actual train-video/cache frontend check')
    splits=read(base['artifacts']['splits']['path']);caches={};raw_rows={};cache_paths={}
    for split in ['train','val']:
        cache_paths[split]=root/'features'/f"{spec['cache']}_{split}_extracted_prop.pickle"
        caches[split]=checked_cache(cache_paths[split],splits[split]['ids'],spec)
        raw_rows[split]=read(root/'audit'/f'{scene}_{split}_raw_relations.json')
        if {r['id'] for r in raw_rows[split]}!=set(splits[split]['ids']):
            raise ValueError('Metadata rows do not cover the same split as the cache')
    if check.get('cache_train_sha256') != digest(cache_paths['train']):
        raise ValueError('Frontend check must exercise the actual training cache')
    object_fields=raw['fields'];varying=raw['varying_object_fields'];indices=[object_fields.index(f) for f in varying]
    if eligible and not indices: raise ValueError('No varying object physical attributes were audited')
    if eligible:
        train_records=qualify_objects(raw_rows['train'],caches['train'],scene=scene,split='train',field_indices=indices,gravity_branch=branch)
        val_records=qualify_objects(raw_rows['val'],caches['val'],scene=scene,split='val',field_indices=indices,gravity_branch=branch)
        primary,wrong1=build_donor_manifests(val_records,scene=scene,split='val',physical_fields=varying)
        index={'version':'cophy-relation-index-v3','scene':scene,'split':'train','records':train_records}
        parameter_data=parameter_documents(raw_rows,scene=scene,slots=spec['slots'],object_fields=object_fields,
                                           include_gravity=scene=='blocktower' and branch=='varying_verified')
        parameter_fields=[f['name'] for f in parameter_data['train']['fields']]
    else:
        index={'version':'cophy-relation-index-v3','scene':scene,'split':'train','records':[],
               'status':'NOT_APPLICABLE','reason':'unreliable_global_gravity'}
        primary=wrong1={'status':'NOT_APPLICABLE','reason':'unreliable_global_gravity','rows':[]}
        parameter_data={};parameter_fields=[]
    audit=dict(qualification,parameter_fields=parameter_fields,relation_fields=varying,
               allowed_methods=methods,coverage=primary.get('coverage'),test_read=False)
    artifacts=dict(base['artifacts'])
    documents={'audit':audit,'relation_index':index,'validation_primary':primary,'validation_wrong_1':wrong1,
        'sampler':{'version':'full_epoch_v3','seed_derivation':'sha256(cophy-pairs-v3:seed:epoch:0)',
                   'focal_rule':'uniform legal focal per recipient','recipient_order':'same independent loader generator'},
        'random_rule':{'version':'full_epoch_stratified_permutation','max_attempts':256,
                       'preserve_donor_multiset':True,'allow_incidental_same_physics':True,'exclude_same_experiment':True},
        'test_generator':{'module':'cophy_manifests','source_sha256':digest(Path(__file__).with_name('cophy_manifests.py')),
                          'seed':20260911,'donor_split':'test','test_read':False}}
    for name,document in documents.items(): artifacts[name]=artifact(frozen_write(out/(name+'.json'),document))
    artifacts['validation_correct']=artifacts['validation_primary']
    artifacts['validation_wrong_any']=artifacts['validation_primary']
    artifacts['raw_audit']=artifact(raw_path);artifacts['frontend_check']=artifact(frontend_check_path)
    for split in ['train','val']:
        artifacts[f'cache_{split}']=artifact(cache_paths[split])
        artifacts[f'raw_relations_{split}']=artifact(root/'audit'/f'{scene}_{split}_raw_relations.json')
        if split in parameter_data:
            artifacts[f'parameters_{split}']=artifact(frozen_write(out/f'parameters_{split}.json',parameter_data[split]))
    result=dict(base,status='PASS',stage='training',allowed_methods=methods,artifacts=artifacts,
                training_authorized_by_this_receipt=True,test_read=False)
    path=frozen_write(out/'training_preflight.json',result)
    verify_preflight(path);verify_adapter_binding(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--scene',choices=SPECS,required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--stage',choices=['features','training'],required=True)
    parser.add_argument('--qualification');parser.add_argument('--frontend-check')
    args=parser.parse_args()
    if args.stage=='features': path=feature_preflight(args.root,args.scene,args.out)
    else:
        if not args.qualification or not args.frontend_check: parser.error('training needs qualification and frontend check')
        path=training_preflight(args.root,args.scene,args.out,args.qualification,args.frontend_check)
    print(json.dumps({'preflight':str(path),'sha256':digest(path)}))
