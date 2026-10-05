"""Resumable real RGB feature cache for CoPhy v6. No GT state or test reads.

Each scene/split has one writer. Arrays are committed in contiguous episode
blocks; an interrupted uncommitted block is overwritten on resume. Readers
must wait for COMPLETE.json (or scene_COMPLETE.json for both splits).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch

VERSION = 'cophy-rgb-object-feature784-v6.1'
SCENES = {
    'collision': dict(folder='collisionCF', subdir='', suffix='normal', frames=15, slots=4),
    'balls': dict(folder='ballsCF', subdir='4', suffix='4', frames=30, slots=9),
    'blocktower': dict(folder='blocktowerCF', subdir='3', suffix='3_normal', frames=30, slots=4),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit(event, **values):
    print(json.dumps(dict(event=event, time_unix=time.time(), **values), allow_nan=False), flush=True)


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def binding(args, scene, split, smoke=False):
    if scene not in SCENES or split not in ('train', 'val'):
        raise ValueError('Only registered scenes and train/val are permitted')
    spec = SCENES[scene]
    split_path = args.source / 'dataloaders/splits' / f"{spec['folder']}_{split}_{spec['suffix']}.txt"
    ids = sorted(split_path.read_text().split())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty/duplicate split IDs')
    if any(not ident.isdigit() for ident in ids):
        raise ValueError('Only audited numeric episode IDs are permitted')
    if smoke:
        ids = ids[:2]
    receipt = read(args.root / 'train_val_extraction_complete.json')
    download = read(args.root / 'raw/download_complete.json')
    if receipt['archive_sha256'] != download['sha256'] or download['phase'] != 'COMPLETE_VERIFIED_SEALED':
        raise ValueError('Archive/extraction identity is not verified')
    if receipt.get('test_samples_extracted') != 0:
        raise ValueError('Unexpected extraction scope')
    data_root = Path(receipt['dataset_root']) / spec['folder'] / spec['subdir']
    checkpoint = args.source / 'ckpts/derendering' / spec['folder'] / 'model_state_dict.pt'
    m = dict(version=VERSION, scene=scene, split=split, ids=ids, episodes=len(ids),
        frames=spec['frames'], slots=spec['slots'], feature_dim=784, dtype='float16',
        presence_dtype='uint8', presence_semantics='per frame visual presence logit > 0',
        source_image_size=[224, 224], inference_precision=args.precision,
        model_mode='eval; all parameters frozen; last_cnn before mlp_pose',
        frontend_scope='official supervised visual frontend; no GT state read by this extractor',
        preprocess='official get_rgb: RGB/255 * [0.229,0.224,0.225] + [0.485,0.456,0.406]',
        data_root=str(data_root), split_file=artifact(split_path), checkpoint=artifact(checkpoint),
        archive_sha256=receipt['archive_sha256'], extraction_receipt=artifact(args.root/'train_val_extraction_complete.json'),
        source_files={name: sha(args.source/name) for name in ['derendering/model.py','dataloaders/utils.py']},
        extractor=artifact(__file__), only_source_fields=['ab/rgb.mp4','cd/rgb.mp4'],
        future_role='CD future is target-only; downstream input prefix remains separate reader policy',
        no_gt_coordinates=True, optimizer_steps=0, test_read=False, smoke=smoke)
    return m


def install_source(args):
    sys.path.insert(0, str(args.source))
    from derendering.model import DeRendering
    from dataloaders.utils import get_rgb
    return DeRendering, get_rgb


def load_visual(m, args):
    cls, _ = install_source(args)
    checkpoint = m['checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Visual checkpoint changed')
    model = cls(m['slots']).eval()
    model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    model = model.to(args.device).eval()
    return model


def visual_features(model, x):
    c = model.cnn
    h = c.maxpool(c.relu(c.bn1(c.conv1(x))))
    h = c.layer4(c.layer3(c.layer2(c.layer1(h))))
    # Original forward's dropout is the identity in eval mode.
    presence = model.mlp_presence(h.mean((2, 3)))
    features = model.last_cnn(h).reshape(-1, model.num_objects, 784)
    return features, presence > 0


def load_video(request):
    from dataloaders.utils import get_rgb
    ident, branch, root, expected_frames = request
    folder = Path(root) / ident / branch
    file = folder / 'rgb.mp4'
    video_sha256 = sha(file)
    rgb = get_rgb(str(folder))
    if rgb.shape != (expected_frames, 3, 224, 224) or not np.isfinite(rgb).all():
        raise ValueError(f'Unexpected RGB content: {file}, shape={rgb.shape}')
    return rgb, dict(id=ident, branch=branch, relative_path=f'{ident}/{branch}/rgb.mp4',
                     sha256=video_sha256, bytes=file.stat().st_size)


@torch.inference_mode()
def infer_videos(model, videos, args):
    n = len(videos)
    length = len(videos[0])
    # Keep at most one decode block in RAM; GPU receives only a frame microbatch.
    joined = np.concatenate(videos)
    outputs, presences = [], []
    for off in range(0, len(joined), args.frame_batch):
        x = torch.from_numpy(joined[off:off+args.frame_batch]).contiguous().to(args.device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.precision == 'amp_float16'):
            f, p = visual_features(model, x)
        outputs.append(f.to(dtype=torch.float16).cpu().numpy())
        presences.append(p.to(dtype=torch.uint8).cpu().numpy())
    features = np.concatenate(outputs).reshape(n, length, model.num_objects, 784)
    presence = np.concatenate(presences).reshape(n, length, model.num_objects)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite visual features')
    return features, presence


def array_specs(m):
    n, t, o = m['episodes'], m['frames'], m['slots']
    return {name: ((n,t,o,784),np.dtype('float16')) if name.startswith('features') else ((n,t,o),np.dtype('uint8'))
            for name in ['features_ab','features_cd','presence_ab','presence_cd']}


def check_complete(directory, m):
    complete = read(directory/'COMPLETE.json')
    if complete.get('status') != 'COMPLETE' or complete['manifest_sha256'] != sha(directory/'manifest.json'):
        raise ValueError('Invalid completed-cache identity')
    if read(directory/'manifest.json') != m or read(directory/'ids.json') != m['ids']:
        raise ValueError('Completed-cache input binding differs')
    for name, (shape,dtype) in array_specs(m).items():
        x = np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if x.shape != shape or x.dtype != dtype or (directory/(name+'.npy')).stat().st_size != complete['files'][name]['bytes']:
            raise ValueError('Completed array shape/dtype/size differs: '+name)
    if sha(directory/'chunks.jsonl') != complete['chunks_sha256']:
        raise ValueError('Completed chunk ledger differs')
    return complete


def process_split(args, scene, split):
    m = binding(args,scene,split)
    directory = args.out/scene/split
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'COMPLETE.json').exists():
            check_complete(directory,m)
            emit('split_already_complete',scene=scene,split=split)
            return
        if (directory/'manifest.json').exists():
            if read(directory/'manifest.json') != m:
                raise ValueError('Input/code/precision changed; use a new output version')
        else:
            write(directory/'manifest.json',m)
            write(directory/'ids.json',m['ids'])
        if read(directory/'ids.json') != m['ids']:
            raise ValueError('ID binding differs')
        progress_path = directory/'PROGRESS.json'
        progress = read(progress_path) if progress_path.exists() else dict(committed_episodes=0, ledger_bytes=0, chunks=0)
        arrays = {}
        for name,(shape,dtype) in array_specs(m).items():
            fn = directory/(name+'.npy')
            if fn.exists():
                arrays[name] = np.lib.format.open_memmap(fn,mode='r+')
            else:
                if progress['committed_episodes']:
                    raise ValueError('Missing committed array '+name)
                arrays[name] = np.lib.format.open_memmap(fn,mode='w+',dtype=dtype,shape=shape)
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise ValueError('Resume array schema differs '+name)
        ledger_path = directory/'chunks.jsonl'
        with ledger_path.open('a+b') as ledger:
            if ledger.seek(0,2) < progress['ledger_bytes']:
                raise ValueError('Resume ledger shorter than committed data')
            ledger.truncate(progress['ledger_bytes'])
            ledger.seek(0,2)
            model = load_visual(m,args)
            begin = time.perf_counter()
            resumed_at = int(progress['committed_episodes'])
            emit('split_started',scene=scene,split=split,episodes=m['episodes'],resumed_at=resumed_at,device=args.device)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for start in range(resumed_at,m['episodes'],args.episode_batch):
                    end = min(start+args.episode_batch,m['episodes'])
                    requests = [(i,b,m['data_root'],m['frames']) for i in m['ids'][start:end] for b in ('ab','cd')]
                    t0 = time.perf_counter()
                    decoded = list(pool.map(load_video,requests))
                    decode_seconds = time.perf_counter()-t0
                    t0 = time.perf_counter()
                    f,p = infer_videos(model,[item[0] for item in decoded],args)
                    forward_seconds = time.perf_counter()-t0
                    block = dict(features_ab=f[0::2],features_cd=f[1::2],presence_ab=p[0::2],presence_cd=p[1::2])
                    t0 = time.perf_counter()
                    checks = {}
                    for name,value in block.items():
                        arrays[name][start:end] = value
                        arrays[name].flush()
                        checks[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    row = dict(start=start,end=end,inputs=[item[1] for item in decoded],array_block_sha256=checks)
                    ledger.write((json.dumps(row,sort_keys=True)+'\n').encode())
                    ledger.flush();os.fsync(ledger.fileno())
                    write_seconds = time.perf_counter()-t0
                    elapsed = time.perf_counter()-begin
                    completed_frames = (end-resumed_at)*m['frames']*2
                    progress = dict(version=VERSION,scene=scene,split=split,committed_episodes=end,
                        episodes=m['episodes'],ledger_bytes=ledger.tell(),chunks=progress['chunks']+1,
                        manifest_sha256=sha(directory/'manifest.json'),seconds_this_process=elapsed,
                        frames_per_second_this_process=completed_frames/elapsed,
                        estimated_remaining_seconds=(m['episodes']-end)*m['frames']*2/(completed_frames/elapsed),
                        last_decode_seconds=decode_seconds,last_forward_seconds=forward_seconds,last_write_seconds=write_seconds,
                        pid=os.getpid(),host=socket.gethostname(),updated_unix=time.time(),test_read=False)
                    write(progress_path,progress)
                    emit('cache_progress',**progress)
                    del decoded,f,p,block
            if sha(m['checkpoint']['path']) != m['checkpoint']['sha256']:
                raise ValueError('Checkpoint changed during extraction')
            complete = dict(status='COMPLETE',version=VERSION,scene=scene,split=split,
                episodes=m['episodes'],manifest_sha256=sha(directory/'manifest.json'),ids_sha256=sha(directory/'ids.json'),
                chunks_sha256=sha(ledger_path),files={name:dict(path=name+'.npy',shape=list(shape),dtype=str(dtype),
                    bytes=(directory/(name+'.npy')).stat().st_size) for name,(shape,dtype) in array_specs(m).items()},
                checkpoint_sha256=m['checkpoint']['sha256'],optimizer_steps=0,test_read=False,
                committed_episodes=m['episodes'],finished_unix=time.time())
            write(directory/'COMPLETE.json',complete)
            emit('split_complete',**complete)
            del model,arrays
            torch.cuda.empty_cache()


@torch.inference_mode()
def smoke(args,scene):
    m = binding(args,scene,'train',smoke=True)
    model = load_visual(m,args)
    requests = [(i,b,m['data_root'],m['frames']) for i in m['ids'] for b in ('ab','cd')]
    decoded = [load_video(r) for r in requests]
    values,presence = infer_videos(model,[d[0] for d in decoded],args)
    hooked = []
    handle = model.last_cnn.register_forward_hook(lambda module, inputs, output: hooked.append(output.detach()))
    rgb = np.concatenate([d[0] for d in decoded])
    reference,reference_p = [],[]
    for off in range(0,len(rgb),args.frame_batch):
        x = torch.from_numpy(rgb[off:off+args.frame_batch]).contiguous().to(args.device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=args.precision=='amp_float16'):
            logits,_,_ = model(x)
        reference.append(hooked.pop().reshape(-1,m['slots'],784).to(dtype=torch.float16).cpu().numpy())
        reference_p.append((logits>0).to(dtype=torch.uint8).cpu().numpy())
    handle.remove()
    expected = np.concatenate(reference).reshape(values.shape)
    expected_p = np.concatenate(reference_p).reshape(presence.shape)
    difference = float(np.max(np.abs(values.astype(np.float32)-expected.astype(np.float32))))
    if difference != 0 or not np.array_equal(presence,expected_p):
        raise ValueError('Official online forward and cache feature path differ')
    directory=args.out/'smoke'/scene
    directory.mkdir(parents=True,exist_ok=True)
    fn=directory/'features.npy'
    mm=np.lib.format.open_memmap(fn,mode='w+',dtype=np.float16,shape=values.shape)
    mm[:]=values;mm.flush();del mm
    if not np.array_equal(np.load(fn,mmap_mode='r'),expected):
        raise ValueError('Real memmap roundtrip differs')
    write(directory/'SMOKE.json',dict(status='PASS',version=VERSION,scene=scene,ids=m['ids'],
        feature_shape=list(values.shape),presence_shape=list(presence.shape),
        max_online_cache_difference=difference,memmap_roundtrip_exact=True,
        checkpoint_sha256=m['checkpoint']['sha256'],extractor_sha256=sha(__file__),
        precision=args.precision,frame_batch=args.frame_batch,optimizer_steps=0,test_read=False,time_unix=time.time()))
    emit('smoke_pass',scene=scene,features=list(values.shape),max_difference=difference)


def run(args):
    scenes=list(SCENES) if args.scene=='all' else [args.scene]
    for scene in scenes:
        smoke_path=args.out/'smoke'/scene/'SMOKE.json'
        if not smoke_path.exists() or read(smoke_path).get('extractor_sha256')!=sha(__file__):
            smoke(args,scene)
        for split in ('train','val'):
            process_split(args,scene,split)
        write(args.out/scene/'scene_COMPLETE.json',dict(status='COMPLETE',scene=scene,version=VERSION,
            splits={s:artifact(args.out/scene/s/'COMPLETE.json') for s in ('train','val')},test_read=False))
        emit('scene_complete',scene=scene)
    write(args.out/('COMPLETE.json' if args.scene=='all' else args.scene+'_RUN_COMPLETE.json'),
        dict(status='COMPLETE',version=VERSION,scenes=scenes,test_read=False,optimizer_steps=0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['smoke','run'])
    parser.add_argument('--root',type=Path,default=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy"))))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--scene',choices=['all',*SCENES],default='all')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--episode-batch',type=int,default=16)
    parser.add_argument('--frame-batch',type=int,default=128)
    parser.add_argument('--precision',choices=['float32','amp_float16'],default='amp_float16')
    args=parser.parse_args()
    args.source=args.source or args.root/'source'
    args.out=args.out or args.root/'latent_v6/features'
    if min(args.workers,args.episode_batch,args.frame_batch)<1:
        raise ValueError('Positive batch/worker sizes required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    install_source(args)
    if args.command=='smoke':
        for scene in (list(SCENES) if args.scene=='all' else [args.scene]):smoke(args,scene)
    else:
        try:run(args)
        except Exception as error:
            write(args.out/'failures'/f'{int(time.time())}_{os.getpid()}.json',
                dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),test_read=False))
            raise


if __name__=='__main__':
    main()
