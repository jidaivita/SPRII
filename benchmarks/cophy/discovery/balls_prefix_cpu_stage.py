"""Resumable, lossless CPU decoding of the 1488 missing Balls query prefixes.

No model is loaded and no future state is opened. The GPU stage later consumes
the exact arrays returned by the original load_prefix function.
"""

import os
import argparse
import concurrent.futures
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import xep_discovery as xep

VERSION = 'balls-fullval-prefix-cpu-stage-1'


def prepare(base, out, workers=2):
    out.mkdir(parents=True, exist_ok=True)
    source = xep.read(base / 'manifest.json')
    part = source['splits']['val']
    exclude = set(part['excluded_ids']) | set(part['query_ids'])
    ids = sorted((q for q in part['all_ids'] if q not in exclude),
                 key=lambda q: hashlib.sha256(('xep:20260911:' + q).encode()).digest())
    if len(ids) != 1488 or len(part['query_ids']) != 512:
        raise ValueError('Unexpected original validation cohort')
    import dataloaders.utils as utils
    binding = dict(version=VERSION, base=str(base), data_root=source['data_root'],
                   ids=ids, batch_size=24, test_read=False, future_states_read=False,
                   file_sha256={str(p): xep.digest(p) for p in
                                (base / 'manifest.json', Path(xep.__file__), Path(utils.__file__))})
    if (out / 'binding.json').exists() and xep.read(out / 'binding.json') != binding:
        raise ValueError('CPU prefix stage binding changed')
    xep.write(out / 'binding.json', binding)
    if (out / 'complete.json').exists():
        receipt = xep.read(out / 'complete.json')
        for name, sha in receipt['file_sha256'].items():
            if xep.digest(name) != sha:
                raise ValueError('Completed prefix shard changed: ' + name)
        return receipt
    began = time.perf_counter()
    shards = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for off in range(0, len(ids), 24):
            batch = ids[off:off + 24]
            path = out / f'prefix_{off:04d}.npz'
            receipt_path = path.with_suffix('.json')
            if path.exists() and receipt_path.exists():
                receipt = xep.read(receipt_path)
                if receipt['ids'] != batch or receipt['sha256'] != xep.digest(path):
                    raise ValueError('Resumed prefix shard changed')
            else:
                loaded = list(pool.map(xep.load_prefix, [(q, source['data_root']) for q in batch]))
                if [v[0] for v in loaded] != batch:
                    raise ValueError('Decoded prefix order differs')
                rgb = np.stack([v[1] for v in loaded])
                if rgb.shape != (len(batch), 3, 3, 224, 224) or not np.isfinite(rgb).all():
                    raise ValueError('Invalid decoded prefix')
                # np.savez is lossless, with no dtype conversion or quantization.
                xep.save_npz(path, ids=np.asarray(batch), rgb=rgb)
                with np.load(path, allow_pickle=False) as stored:
                    if not np.array_equal(stored['rgb'], rgb):
                        raise ValueError('Lossless cache verification failed')
                receipt = dict(ids=batch, sha256=xep.digest(path), shape=list(rgb.shape),
                               dtype=str(rgb.dtype), bytes=path.stat().st_size)
                xep.write(receipt_path, receipt)
            shards.append(dict(path=str(path), **receipt))
            if off % 240 == 0:
                xep.emit('balls_cpu_prefix', done=off + len(batch), total=len(ids),
                         seconds=time.perf_counter() - began)
    receipt = dict(version=VERSION, status='COMPLETE', ids=ids, shards=shards,
                   binding_sha256=xep.digest(out / 'binding.json'),
                   file_sha256={s['path']: s['sha256'] for s in shards},
                   seconds=time.perf_counter() - began, test_read=False,
                   future_states_read=False, model_forward_calls=0, gpu_used=False,
                   bytes=sum(s['bytes'] for s in shards))
    xep.write(out / 'complete.json', receipt)
    xep.emit('balls_cpu_prefix_complete', seconds=receipt['seconds'], bytes=receipt['bytes'])
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
    parser.add_argument('--base', type=Path, default=root / 'xep_discovery_balls_v4_1')
    parser.add_argument('--out', type=Path, default=root / 'supervised_tail_v6/balls_prefix_cpu')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'stage.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.base, args.out, args.workers)
