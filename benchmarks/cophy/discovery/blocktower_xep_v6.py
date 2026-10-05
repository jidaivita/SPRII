"""Blocktower cross-experience inputs: visual prefix pose, sealed future targets.

This prepares data only. Relations use audited AB metadata and public C object
presence; no future state is used to choose queries or independent supports.
The pose bridge consumes only CD frames 0, 1, 2 from the frozen v6 RGB cache.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

VERSION = 'blocktower-xep-pose-v6-1'
K, D, PREFIX, HORIZON = 4, 3, 3, 27


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with open(tmp, 'wb') as stream: np.savez(stream, **values)
    os.replace(tmp, path)


def emit(event, **values):
    print(json.dumps(dict(event=event, time_unix=time.time(), **values), allow_nan=False), flush=True)


def artifact(bound, name):
    entry = bound['artifacts'][name]; path = Path(entry['path'])
    if digest(path) != entry['sha256']: raise ValueError('Changed bound artifact: ' + name)
    return path


def install_runtime(root):
    profile = read(Path(root) / 'runtime_profiles.json')['scenes']['blocktower']
    preflight = Path(profile['training_preflight']['path'])
    if digest(preflight) != profile['training_preflight']['sha256']:
        raise ValueError('Changed registered Blocktower preflight')
    bound = read(preflight); source = Path(bound['runtime_source_root']).resolve()
    if source != Path(profile['source']).resolve(): raise ValueError('Registered runtime mismatch')
    sys.path.insert(0, str(source))
    import cophy_protocol
    if Path(cophy_protocol.__file__).resolve().parent != source: raise ValueError('Wrong scene runtime imported')
    cophy_protocol.verify_adapter_binding(str(preflight))
    return profile, preflight, bound


def relation_key(row):
    physical = tuple(int(x) for x in row['physical'])
    gravity = tuple(float(x) for x in row['raw_gravity'])
    if len(physical) != 2 or len(gravity) != 2: raise ValueError('Unexpected Blocktower attributes')
    return json.dumps([int(row['slot']), row['known_type'], *physical, *gravity], separators=(',', ':'))


def prepare_manifest(root, out):
    import torch
    profile, prepath, p = install_runtime(root)
    fn = out / 'manifest.json'
    if fn.exists():
        m = read(fn)
        if m['version'] != VERSION or m['preflight_sha256'] != digest(prepath):
            raise ValueError('Changed manifest binding')
        return m
    splits = read(artifact(p, 'splits'))
    m = dict(version=VERSION, scene='blocktower-normal', prefix=PREFIX, supports=3,
        maximum_supports=8, slots=K, dims=D, horizon=HORIZON, test_read=False,
        preflight=str(prepath), preflight_sha256=digest(prepath), runtime_source=profile['source'],
        data_root=str(Path(p['input_profile']['dataset_dir']) / str(p['input_profile']['num_objects'])),
        derenderer=str(artifact(p, 'derenderer')), derenderer_sha256=p['artifacts']['derenderer']['sha256'],
        candidate_encoding='candidate_groups[key] are indices into all_ids; remove the recipient index',
        relation_fields=['slot', 'public_type', 'mass_class', 'friction_class', 'gravity_x', 'gravity_y'],
        gravity_role='matching and Known-parameters only; absent from ordinary prediction inputs',
        query_input='frozen official mlp_pose of v6 last_cnn cache at CD frames 0,1,2; visual detection',
        known_parameter_columns=['mass_0','mass_1','friction_0','friction_1','gravity_x','gravity_y'],
        source_models={}, splits={})
    for method in ('Native', 'A', 'Random'):
        ckpt = Path(profile['runs_seed0']) / method / 'model_state_dict.pt'
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        if state['run_config']['method'] != method or state['run_config']['data_binding']['preflight_sha256'] != digest(prepath):
            raise ValueError('Changed source checkpoint binding')
        m['source_models'][method] = dict(path=str(ckpt), sha256=digest(ckpt), epoch=state['epoch'],
                                        source_run_config=state['run_config'])
        del state
    for split in ('train', 'val'):
        ids = sorted(map(str, splits[split]['ids'])); index = {x:i for i,x in enumerate(ids)}
        rows = read(artifact(p, 'raw_relations_' + split)); cache_path = artifact(p, 'cache_' + split)
        with open(cache_path, 'rb') as stream: cache = pickle.load(stream)
        byid, groups = defaultdict(list), defaultdict(set)
        for row in rows:
            ident = str(row['id']); slot = int(row['slot'])
            if row['split'] != split or ident not in index or not 0 <= slot < K:
                raise ValueError('Relation split/slot mismatch')
            byid[ident].append(row)
            if cache[ident]['presence_ab'][slot] > 0: groups[relation_key(row)].add(index[ident])
        groups = {key:sorted(values) for key,values in groups.items()}
        presence, physical, gravity, keys, eligible, excluded, pool_sizes = {}, {}, {}, {}, [], [], []
        for ident in ids:
            active = [r for r in byid[ident] if r['in_C']]
            ks, pm, ph = [None] * K, [0.] * K, [[0,0] for _ in range(K)]
            gs = {tuple(map(float,r['raw_gravity'])) for r in byid[ident]}
            if len(gs) != 1: raise ValueError('Inconsistent global gravity')
            for row in active:
                slot = int(row['slot']); ks[slot] = relation_key(row); pm[slot] = 1.; ph[slot] = row['physical']
            lengths = [len(groups[ks[int(r['slot'])]]) - int(index[ident] in groups[ks[int(r['slot'])]]) for r in active]
            if active and min(lengths) >= 8:
                eligible.append(ident); keys[ident] = ks; presence[ident] = pm; physical[ident] = ph
                gravity[ident] = list(next(iter(gs))); pool_sizes.extend(lengths)
            else: excluded.append(ident)
        eligible.sort(key=lambda q: hashlib.sha256(('xep:20260911:' + q).encode()).digest())
        query = eligible if split == 'train' else eligible[:512]
        m['splits'][split] = dict(all_ids=ids, query_ids=query, full_query_ids=eligible,
            eligible_total=len(eligible), excluded_ids=excluded, cache=str(cache_path), cache_sha256=digest(cache_path),
            candidate_groups=groups, candidate_keys=keys, presence=presence, physical=physical, gravity=gravity,
            minimum_independent_pool=min(pool_sizes) if pool_sizes else 0,
            source_pool_size_by_key={key:len(pool) for key,pool in groups.items()})
        emit('manifest_split', split=split, episodes=len(ids), eligible=len(eligible), pilot=len(query),
             candidate_groups=len(groups), minimum_pool=min(pool_sizes) if pool_sizes else 0)
        del cache
    if set(m['splits']['train']['all_ids']) & set(m['splits']['val']['all_ids']):
        raise ValueError('Overlapping train/validation episode IDs')
    write(fn, m); return m


def target_row(request):
    root, ident = request
    state = np.load(Path(root) / ident / 'cd/states.npy', allow_pickle=False)
    if state.shape[:2] != (30,K): raise ValueError('Unexpected Blocktower future shape')
    actual = (np.abs(state[0,:,:D]).sum(-1) > 0).astype(np.float32)
    future = np.asarray(state[PREFIX:,:,:D], np.float32)
    if not np.isfinite(future).all(): raise ValueError('Nonfinite target')
    return future, actual


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True); m = prepare_manifest(root, out)
    marker = out / 'targets_complete.json'
    if marker.exists():
        r = read(marker)
        if r['manifest_sha256'] != digest(out / 'manifest.json'): raise ValueError('Changed target manifest')
        emit('targets_already_complete'); return
    start = time.time(); files = {}
    for split in ('train','val'):
        part = m['splits'][split]; ids = part['full_query_ids']; gt, masks, params = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for ident, (future, actual) in zip(ids, pool.map(target_row, [(m['data_root'], q) for q in ids])):
                if not np.array_equal(actual, np.asarray(part['presence'][ident])):
                    raise ValueError('Public C presence differs from target mask')
                labels = np.asarray(part['physical'][ident], np.int64)
                if np.any((labels < 0) | (labels >= 2)): raise ValueError('Invalid physical class')
                onehot = np.eye(2, dtype=np.float32)[labels].reshape(K,4)
                g = np.broadcast_to(np.asarray(part['gravity'][ident],np.float32), (K,2))
                params.append(np.concatenate((onehot,g),-1) * actual[:,None]); masks.append(actual); gt.append(future)
        gt, masks, params = np.asarray(gt), np.asarray(masks), np.asarray(params)
        suffix = split if split == 'train' else 'val_full'
        save_npz(out / ('target_' + suffix + '.npz'), ids=np.asarray(ids), pose=gt)
        save_npz(out / ('parameters_' + suffix + '.npz'), ids=np.asarray(ids), values=params)
        if split == 'val':
            n = len(part['query_ids'])
            save_npz(out / 'target_val.npz', ids=np.asarray(ids[:n]), pose=gt[:n])
            save_npz(out / 'parameters_val.npz', ids=np.asarray(ids[:n]), values=params[:n])
        emit('targets_split', split=split, queries=len(ids), seconds=time.time()-start)
    # Two actual RGB prefixes permit the bridge to be checked on the other host.
    from dataloaders.utils import get_rgb
    ref_ids = [m['splits'][s]['query_ids'][0] for s in ('train','val')]
    rgb = np.stack([get_rgb(str(Path(m['data_root'])/q/'cd'), max_frames=PREFIX) for q in ref_ids])
    save_npz(out / 'bridge_rgb_reference.npz', ids=np.asarray(ref_ids), splits=np.asarray(['train','val']), rgb=rgb)
    for fn in list(out.glob('target_*.npz')) + list(out.glob('parameters_*.npz')) + [out/'bridge_rgb_reference.npz']:
        files[fn.name] = dict(sha256=digest(fn), bytes=fn.stat().st_size)
    write(marker, dict(version=VERSION, status='COMPLETE', manifest_sha256=digest(out/'manifest.json'),
                       files=files, seconds=time.time()-start, test_read=False, optimizer_steps=0))


def bridge(root, out, features, device, wait_seconds):
    import torch
    torch.set_num_threads(4)
    start = time.time()
    ready = [out/'targets_complete.json'] + [features/s/'COMPLETE.json' for s in ('train','val')]
    while not all(p.is_file() for p in ready):
        if time.time() - start >= wait_seconds: raise RuntimeError('Waiting for targets/features: ' + ', '.join(str(p) for p in ready if not p.exists()))
        time.sleep(10)
    _, _, _ = install_runtime(root); m = read(out/'manifest.json')
    if read(out/'targets_complete.json')['manifest_sha256'] != digest(out/'manifest.json'):
        raise ValueError('Target cache belongs to a different manifest')
    from derendering.model import DeRendering
    if digest(m['derenderer']) != m['derenderer_sha256']: raise ValueError('Changed visual checkpoint')
    model = DeRendering(K).to(device).eval()
    model.load_state_dict(torch.load(m['derenderer'], map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    bindings, memories = {}, {}
    with torch.inference_mode():
        for split in ('train','val'):
            folder = features/split; fm = read(folder/'manifest.json'); fc = read(folder/'COMPLETE.json')
            if fc['status'] != 'COMPLETE' or fc['manifest_sha256'] != digest(folder/'manifest.json'):
                raise ValueError('Invalid feature completion binding')
            if fm['scene'] != 'blocktower' or fm['split'] != split or fm['test_read']:
                raise ValueError('Wrong feature scene/split')
            if fm['checkpoint']['sha256'] != m['derenderer_sha256']:
                raise ValueError('Feature and pose checkpoint differ')
            ids = read(folder/'ids.json'); part = m['splits'][split]
            if ids != part['all_ids']: raise ValueError('Feature IDs differ from manifest')
            index = {q:i for i,q in enumerate(ids)}; queries = part['full_query_ids']
            f = np.load(folder/'features_cd.npy', mmap_mode='r', allow_pickle=False)
            p = np.load(folder/'presence_cd.npy', mmap_mode='r', allow_pickle=False)
            if f.shape != (len(ids),30,K,784) or p.shape != (len(ids),30,K): raise ValueError('Feature shape differs')
            amp = fm['inference_precision'] == 'amp_float16'; poses, detections = [], []
            for off in range(0,len(queries),256):
                ix = [index[q] for q in queries[off:off+256]]
                x = torch.from_numpy(np.array(f[ix,:PREFIX], dtype=np.float32)).to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp): pose = model.mlp_pose(x)[...,:D]
                poses.append(pose.float().cpu().numpy()); detections.append(np.asarray(p[ix,:PREFIX],np.float32)[...,None])
            pose = np.concatenate(poses); detection = np.concatenate(detections)
            if not np.isfinite(pose).all(): raise ValueError('Nonfinite visual query pose')
            mask = np.asarray([part['presence'][q] for q in queries],np.float32)
            suffix = split if split == 'train' else 'val_full'
            save_npz(out/('input_'+suffix+'.npz'), ids=np.asarray(queries), pose=pose, detected=detection, presence=mask)
            if split == 'val':
                n = len(part['query_ids'])
                save_npz(out/'input_val.npz', ids=np.asarray(queries[:n]), pose=pose[:n], detected=detection[:n], presence=mask[:n])
            bindings[split] = dict(manifest_sha256=digest(folder/'manifest.json'), complete_sha256=digest(folder/'COMPLETE.json'),
                feature_files=fc['files'], precision=fm['inference_precision'], input_queries=len(queries),
                prefix_frames=[0,1,2], query_shape=list(pose.shape), detected_shape=list(detection.shape))
            memories[split] = (index, f, p, amp)
            emit('pose_bridge_split',split=split,queries=len(queries),seconds=time.time()-start)
        reference = np.load(out/'bridge_rgb_reference.npz', allow_pickle=False); checks = []
        for ident, split, rgb in zip(reference['ids'],reference['splits'],reference['rgb']):
            ident,split = str(ident),str(split); index,f,p,amp = memories[split]
            x = torch.from_numpy(np.array(rgb)).to(device)
            feature = torch.from_numpy(np.asarray(f[index[ident],:PREFIX],np.float32).copy()).to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                raw_presence, raw_pose, _ = model(x); cache_pose = model.mlp_pose(feature)[...,:D]
            delta = (raw_pose.float()-cache_pose.float()).abs().cpu().numpy()
            with torch.autocast('cuda', enabled=False): _, fp32_pose, _ = model(x)
            fp32_delta = (fp32_pose.float()-cache_pose.float()).abs().cpu().numpy()
            seen = (raw_presence>0).cpu().numpy(); cached_seen=np.asarray(p[index[ident],:PREFIX])>0
            check = dict(id=ident, split=split, amp_same_path_max_abs=float(delta.max()),
                amp_same_path_mean_abs=float(delta.mean()), fp32_vs_v6_max_abs=float(fp32_delta.max()),
                fp32_vs_v6_mean_abs=float(fp32_delta.mean()), detection_mismatches=int(np.sum(seen!=cached_seen)))
            if not np.isfinite(delta).all() or float(delta.max()) > .05:
                raise ValueError('Visual bridge sanity mismatch: '+json.dumps(check))
            checks.append(check)
    files = {}
    for pattern in ('input_*.npz','target_*.npz','parameters_*.npz'):
        for path in out.glob(pattern): files[path.name]=dict(sha256=digest(path),bytes=path.stat().st_size)
    receipt = dict(version=VERSION,status='COMPLETE',manifest_sha256=digest(out/'manifest.json'),
        implementation_sha256=digest(__file__),features=bindings,files=files,visual_bridge_checks=checks,
        test_read=False,optimizer_steps=0,seconds=time.time()-start,
        input_precision_note='Blocktower v6 pose from frozen FP16 cached last_cnn with matching AMP mlp_pose; all heads share this input. Not asserted bit-identical to the legacy FP32 frontend.')
    write(out/'data_ready.json',receipt); emit('data_ready',seconds=receipt['seconds'],checks=checks)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','bridge'))
    parser.add_argument('--root',type=Path,default=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))))
    parser.add_argument('--out',type=Path); parser.add_argument('--features',type=Path)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--wait-seconds',type=float,default=0)
    args=parser.parse_args(); out=args.out or args.root/'xep_discovery_blocktower_v6'
    if args.command=='prepare': prepare(args.root,out)
    else: bridge(args.root,out,args.features or args.root/'latent_v6/features/blocktower',args.device,args.wait_seconds)


if __name__=='__main__': main()
