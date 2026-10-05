"""Lossless uint8 HOST observations; each GPU microbatch remains exact float32.

Only the already verified integer observation corpus may use this adapter.
No model, loss, sampling decision, batch size, or optimizer changes.
"""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

class HostNumpy:
    def __init__(self, original):
        self.original = original
        self.compact_allocations = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def empty(self, shape, dtype=float, *args, **kwargs):
        shape = tuple(shape)
        if len(shape) in (5, 6) and shape[-3:] == (5, 5, 40) and np.dtype(dtype) == np.dtype('float32'):
            self.compact_allocations += 1
            return self.original.empty(shape, dtype=np.uint8, *args, **kwargs)
        return self.original.empty(shape, dtype=dtype, *args, **kwargs)

def cast_micro(tree, start, stop):
    if isinstance(tree, dict):
        return {key: cast_micro(value, start, stop) for key, value in tree.items()}
    value = tree[start:stop]
    if value.ndim in (5, 6) and value.shape[-3:] == (5, 5, 40):
        assert value.dtype in (np.dtype('uint8'), np.dtype('float32'))
        return np.asarray(value, dtype=np.float32)
    return value

def install(repo, manifest):
    info = json.loads(Path(manifest).read_text())
    assert info['status'] == 'PASS_LOSSLESS_MAPPED_HISTORY' and info['all_arrays_byte_exact']
    obs = info['fields']['obs']
    assert np.dtype(obs['storage_dtype']) == np.dtype('uint8') and np.dtype(obs['dtype']) == np.dtype('float32')
    from native_a import large_batch, joint_batch
    assert hashlib.sha256((Path(repo)/'native_a/large_batch.py').read_bytes()).hexdigest() == 'fc15b1be19ef831e6ecf03ed94233da28d3706eece376bd3a522f2cc7c056bfd'
    assert hashlib.sha256((Path(repo)/'native_a/joint_batch.py').read_bytes()).hexdigest() == 'e3736de12c764e03bae1c5c6d2d1f22c62719c03781a0c1040aab26b2ad64424'
    proxy = HostNumpy(large_batch.np)
    large_batch.np = proxy
    joint_batch._slice = cast_micro
    return proxy

def float32_input_fingerprint(tree):
    """Hash logical original-dtype values using bounded conversion blocks."""
    h = hashlib.sha256()
    def walk(value):
        if isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode()); walk(value[key])
            return
        value = np.asarray(value)
        compact = value.dtype == np.uint8 and value.ndim in (5, 6) and value.shape[-3:] == (5, 5, 40)
        dtype = np.dtype('float32') if compact else value.dtype
        h.update(str(dtype).encode()); h.update(str(value.shape).encode())
        flat = value.reshape(-1)
        for start in range(0, flat.size, 1 << 18):
            h.update(np.ascontiguousarray(flat[start:start+(1 << 18)], dtype=dtype).tobytes())
    walk(tree)
    return h.hexdigest()

def trim_unused_host_heap():
    libc = ctypes.CDLL(None)
    if hasattr(libc, 'malloc_trim'):
        libc.malloc_trim(0)
