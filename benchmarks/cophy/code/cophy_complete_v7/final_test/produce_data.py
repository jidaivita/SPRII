"""Explicit test extraction and frozen visual inputs, gated by the v7 freeze.

No work on import. Decode AB and only CD[:3]. Future coordinates are saved
separately as targets; relation fields are never given to a visual encoder.
Keep Balls/Collision FP32 query poses and Blocktower's cached-AMP pose route.
"""
import argparse
import concurrent.futures
import fcntl
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from runtime import artifact, checked, read, sha, verify_freeze, write

VERSION = 'cophy-v7-test-producer-1'
SPECS = {'balls': ('ballsCF', '4', 30, 9, 2),
         'collision': ('collisionCF', '', 15, 4, 3),
         'blocktower': ('blocktowerCF', '3', 30, 4, 3)}


def frozen_rule(args):
    freeze = verify_freeze(args.freeze, verify_all=False)
    rule = freeze['scene_rules'][args.scene]
    for name in ('feature_producer', 'pose_producer'):
        if checked(rule[name]).resolve() != Path(__file__).resolve():
            raise ValueError('Frozen producer is a different implementation')
    checked(rule['field_auditor'])
    split = checked(rule['official_split'])
    ids = sorted(split.read_text().split())
    if not ids or len(ids) != len(set(ids)) or any(not q.isdigit() for q in ids):
        raise ValueError('Invalid fixed official test IDs')
    for item in rule['development_id_files']:
        if set(ids) & set(map(str, read(checked(item)))):
            raise ValueError('Test/development episode overlap')
    return freeze, rule, ids


def extract(args):
    _, rule, ids = frozen_rule(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data_root = out/'raw'; dest = out/'extraction.json'
    receipt = read(checked(rule['archive_receipt']))
    archive = Path(rule['archive_path'])
    expected = receipt.get('sha256') or receipt.get('archive_sha256')
    if not expected: raise ValueError('Missing original archive digest')
    identity = dict(version=VERSION, freeze_sha256=sha(args.freeze), scene=args.scene,
                    split='test', ids=ids, archive_path=str(archive), archive_sha256=expected,
                    dataset_root=str(data_root), test_read=True)
    if dest.exists():
        saved = read(dest)
        if any(saved[k] != v for k, v in identity.items()): raise ValueError('Changed extraction binding')
        for item in saved['files']: checked(item)
        print('TEST_EXTRACTION_ALREADY_COMPLETE'); return
    if sha(archive) != expected: raise ValueError('Original data archive changed')
    folder, subdir, _, _, _ = SPECS[args.scene]
    keys = {(folder, subdir, q) if subdir else (folder, q) for q in ids}
    required = {'ab/rgb.mp4', 'ab/states.npy', 'cd/rgb.mp4', 'cd/states.npy', 'confounders.npy'}
    if args.scene in ('collision', 'blocktower'): required.add('ab/colors.txt')
    if args.scene == 'blocktower': required.add('gravity.txt')
    found = {k: set() for k in keys}; files = []
    with tarfile.open(archive, 'r|*') as stream:
        for member in stream:
            parts = Path(member.name).parts
            if folder not in parts: continue
            parts = parts[parts.index(folder):]; width = 3 if subdir else 2
            key = tuple(parts[:width]); suffix = '/'.join(parts[width:])
            if key not in keys or suffix not in required: continue
            if '..' in parts or member.issym() or member.islnk() or not member.isfile():
                raise ValueError('Unsafe selected archive member')
            if suffix in found[key]: raise ValueError('Duplicate selected archive member')
            target = data_root/Path(*parts); target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extract-' + str(os.getpid()))
            with stream.extractfile(member) as source, temporary.open('wb') as sink:
                shutil.copyfileobj(source, sink, 2**20)
            temporary.replace(target); found[key].add(suffix); files.append(artifact(target))
    missing = {str(k): sorted(required-v) for k, v in found.items() if v != required}
    if missing: raise ValueError('Missing official test fields: ' + str(list(missing.items())[:8]))
    write(dest, dict(identity, status='COMPLETE', files=files, episodes=len(ids)), immutable=True)
    print('TEST_EXTRACTION_COMPLETE')


def install(rule):
    source = Path(rule['source_root'])
    for item in rule['visual_source_files']: checked(item)
    sys.path.insert(0, str(source))
    from derendering.model import DeRendering
    from dataloaders.utils import get_rgb
    from cophy_fields import inspect_episode
    import cophy_fields
    if Path(cophy_fields.__file__).resolve() != checked(rule['field_auditor']).resolve():
        raise ValueError('Field auditor import differs from the freeze')
    return DeRendering, get_rgb, inspect_episode


def npz(path, **values):
    import numpy as np
    path = Path(path); temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as f: np.savez(f, **values)
    temporary.replace(path)


def emit(event, **values):
    import json
    print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)


def produce(args):
    import numpy as np
    import torch
    _, rule, ids = frozen_rule(args)
    extraction = read(Path(args.out)/'extraction.json')
    if extraction.get('status') != 'COMPLETE' or extraction['freeze_sha256'] != sha(args.freeze) or extraction['ids'] != ids:
        raise ValueError('Wait for this frozen official test extraction')
    for item in extraction['files']: checked(item)
    folder, subdir, frames, slots, dims = SPECS[args.scene]
    out = Path(args.out)/'inputs'; out.mkdir(parents=True, exist_ok=True)
    cls, get_rgb, inspect_episode = install(rule)
    feature_train = read(checked(rule['feature_training_manifest']))
    amp = feature_train['inference_precision'] == 'amp_float16'
    if feature_train['inference_precision'] not in ('amp_float16', 'float32'):
        raise ValueError('Unrecognized frozen feature precision')
    checkpoint = checked(rule['frontend_checkpoint'])
    if sha(checkpoint) != rule['frontend_checkpoint_sha256'] or feature_train['checkpoint']['sha256'] != sha(checkpoint):
        raise ValueError('Feature/pose frontend mismatch')
    identity = dict(version=VERSION, scene=args.scene, split='test', ids=ids,
        freeze_sha256=sha(args.freeze), extraction_sha256=sha(Path(args.out)/'extraction.json'),
        feature_precision=feature_train['inference_precision'], feature_dtype='float16',
        query_pose_route=('cached_fp16_mlp_pose_matching_amp' if args.scene == 'blocktower' else 'direct_fp32_derenderer'),
        supervised_ab_pose_route='direct_fp32; presence is AB frame0, pose unmasked',
        producer_sha256=sha(__file__), frontend_checkpoint_sha256=sha(checkpoint), test_read=True)
    manifest_path = out/'manifest.json'; write(manifest_path, identity, immutable=True)
    done_path = out/'producer_receipt.json'
    if done_path.exists():
        saved = read(done_path)
        if saved['manifest_sha256'] != sha(manifest_path): raise ValueError('Completed producer identity changed')
        for item in saved['artifacts'].values(): checked(item)
        emit('test_inputs_already_complete', scene=args.scene); return
    torch.set_num_threads(4); model = cls(slots).to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    arrays = {}
    schemas = {
        'features_ab': ((len(ids), frames, slots, 784), 'float16'),
        'features_c': ((len(ids), 3, slots, 784), 'float16'),
        'presence_ab': ((len(ids), frames, slots), 'uint8'),
        'presence_c': ((len(ids), 3, slots), 'uint8'),
        'pose_ab': ((len(ids), frames, slots, 3), 'float32'),
        'pose_presence_ab': ((len(ids), slots), 'float32'),
        'official_pose_c': ((len(ids), 1, slots, 3), 'float32'),
        'official_presence_c': ((len(ids), slots), 'float32'),
        'query_pose': ((len(ids), 3, slots, dims), 'float32'),
        'query_detection': ((len(ids), 3, slots), 'float32')}
    progress_path = out/'progress.json'
    progress = read(progress_path) if progress_path.exists() else dict(committed=0, manifest_sha256=sha(manifest_path))
    if progress['manifest_sha256'] != sha(manifest_path): raise ValueError('Resume binding changed')
    if not isinstance(progress['committed'], int) or not 0 <= progress['committed'] <= len(ids):
        raise ValueError('Invalid committed episode prefix')
    if progress['committed'] and any(not (out/(name+'.npy')).exists() for name in schemas):
        raise ValueError('A committed cache array is missing; cannot fill its prefix with zeros')
    for name, (shape, dtype) in schemas.items():
        path = out/(name+'.npy'); mode = 'r+' if path.exists() else 'w+'
        arrays[name] = np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype): raise ValueError('Resume array schema changed')
    root = Path(extraction['dataset_root']); data = root/folder/subdir

    def load(q):
        ab = get_rgb(str(data/q/'ab')); c = get_rgb(str(data/q/'cd'), max_frames=3)
        if ab.shape != (frames,3,224,224) or c.shape != (3,3,224,224): raise ValueError('Unexpected test video length')
        return np.concatenate((ab,c))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool, torch.inference_mode():
        for first in range(progress['committed'], len(ids), args.episode_batch):
            end = min(first+args.episode_batch, len(ids)); rgb = np.concatenate(list(pool.map(load, ids[first:end])))
            f_all=[]; p_all=[]; poses=[]; seen=[]
            for start in range(0,len(rgb),args.frame_batch):
                x = torch.from_numpy(rgb[start:start+args.frame_batch]).contiguous().to(args.device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    cnn=model.cnn; h=cnn.maxpool(cnn.relu(cnn.bn1(cnn.conv1(x))))
                    h=cnn.layer4(cnn.layer3(cnn.layer2(cnn.layer1(h))))
                    p=model.mlp_presence(h.mean((2,3)))>0
                    f=model.last_cnn(h).reshape(-1,slots,784)
                f_all.append(f.to(torch.float16).cpu().numpy()); p_all.append(p.to(torch.uint8).cpu().numpy())
                with torch.autocast('cuda', enabled=False):
                    fp_presence, fp_pose, _=model(x.float())
                poses.append(fp_pose.float().cpu().numpy()); seen.append((fp_presence>0).float().cpu().numpy())
            shape=(end-first,frames+3,slots)
            features=np.concatenate(f_all).reshape(*shape,784); presence=np.concatenate(p_all).reshape(shape)
            pose=np.concatenate(poses).reshape(*shape,3); detection=np.concatenate(seen).reshape(shape)
            if not np.isfinite(features).all() or not np.isfinite(pose).all(): raise ValueError('Nonfinite frozen frontend')
            qpose=pose[:,frames:,:,:dims]; qdet=detection[:,frames:]
            if args.scene=='blocktower':
                x=torch.from_numpy(features[:,frames:].astype(np.float32)).to(args.device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp): cached_pose=model.mlp_pose(x)[...,:dims]
                qpose=cached_pose.float().cpu().numpy(); qdet=presence[:,frames:].astype(np.float32)
            block=dict(features_ab=features[:,:frames],features_c=features[:,frames:],
                presence_ab=presence[:,:frames],presence_c=presence[:,frames:],
                pose_ab=pose[:,:frames],pose_presence_ab=detection[:,0],
                official_pose_c=pose[:,frames:frames+1],official_presence_c=detection[:,frames],
                query_pose=qpose,query_detection=qdet)
            for name,value in block.items(): arrays[name][first:end]=value; arrays[name].flush()
            progress.update(committed=end); write(progress_path,progress)
            emit('test_visual_progress',scene=args.scene,done=end,total=len(ids))
    # Field audit and targets are kept outside the image encoder path.
    records=[]; targets=[]; official_targets=[]; masks=[]; types=[]
    for q in ids:
        r=inspect_episode(root,args.scene,'test',q); records.extend(r['rows'])
        state=np.load(data/q/'cd/states.npy',allow_pickle=False,mmap_mode='r')
        if state.shape[:2] != (frames,slots): raise ValueError('Target shape differs')
        mask=(np.abs(state[0,:,:3]).sum(-1)>0).astype(np.float32)
        expected=np.zeros(slots,np.float32)
        type_codes=np.zeros((slots,3),np.float32)
        for item in r['rows']:
            if item['in_C']:
                expected[item['slot']]=1
                if args.scene=='collision': type_codes[item['slot'],['sphere','cylinder_up','cylinder_down'].index(item['known_type'])]=1
        if not np.array_equal(mask,expected): raise ValueError('C object identity mismatch')
        future=np.asarray(state[3:,:,:dims],np.float32)
        if not np.isfinite(future).all(): raise ValueError('Nonfinite test target')
        official_future=np.asarray(state[1:,:,:dims],np.float32)
        if not np.isfinite(official_future).all(): raise ValueError('Nonfinite official test target')
        masks.append(mask); types.append(type_codes); targets.append(future); official_targets.append(official_future)
    det=np.asarray(arrays['query_detection'])
    if args.scene=='collision': det=np.concatenate((det[...,None],np.broadcast_to(np.asarray(types)[:,None],(len(ids),3,slots,3))),-1)
    elif args.scene=='blocktower': det=det[...,None]
    npz(out/'input.npz',ids=np.asarray(ids),pose=np.asarray(arrays['query_pose']),detected=det,presence=np.asarray(masks))
    npz(out/'target.npz',ids=np.asarray(ids),pose=np.asarray(targets))
    npz(out/'official_c.npz',ids=np.asarray(ids),pose=np.asarray(arrays['official_pose_c']),presence=np.asarray(arrays['official_presence_c']))
    npz(out/'target_official.npz',ids=np.asarray(ids),pose=np.asarray(official_targets))
    write(out/'ids.json',ids,immutable=True); write(out/'raw_relations.json',records,immutable=True)
    artifacts={name:artifact(out/(name+'.npy')) for name in schemas if name not in ('query_pose','query_detection')}
    artifacts.update(ids=artifact(out/'ids.json'),input=artifact(out/'input.npz'),target=artifact(out/'target.npz'),
        official_c=artifact(out/'official_c.npz'),target_official=artifact(out/'target_official.npz'),raw_relations=artifact(out/'raw_relations.json'))
    receipt=dict(status='COMPLETE',version=VERSION,scene=args.scene,split='test',freeze_sha256=sha(args.freeze),
        query_frames=3,pose_source='same_frozen_derenderer_as_training',gt_current_pose_used=False,
        frontend_checkpoint_sha256=sha(checkpoint),feature_producer=artifact(__file__),pose_producer=artifact(__file__),
        field_auditor=rule['field_auditor'],artifacts=artifacts,manifest_sha256=sha(manifest_path),
        optimizer_steps=0,test_read=True,future_rgb_decoded=False)
    write(done_path,receipt,immutable=True);emit('test_inputs_complete',scene=args.scene,rows=len(ids))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('extract','produce'))
    p.add_argument('--freeze',required=True);p.add_argument('--scene',required=True,choices=SPECS);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--episode-batch',type=int,default=4);p.add_argument('--frame-batch',type=int,default=64)
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
    with (Path(a.out)/(a.command+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (extract if a.command=='extract' else produce)(a)


if __name__=='__main__': main()
