"""Lossless local-disk mirror for unchanged HistoryStore slice/sampling semantics.

Observations use uint8 only when every value round-trips byte-exactly. A failed
round-trip promotes the entire bank to its original dtype, without clipping.
All reads expose original dtypes. No training labels, RNG, or episodes change.
"""
import argparse
import hashlib
import json
import mmap
import os
import time
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def drop_clean_pages(array):
    array.flush()
    if hasattr(array._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
        array._mmap.madvise(mmap.MADV_DONTNEED)
    if hasattr(os, 'posix_fadvise'):
        fd = os.open(array.filename, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def build(h5_path, index_path, directory):
    h5_path, index_path, directory = map(Path, (h5_path, index_path, directory))
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'manifest.json').exists(), 'Do not replace a bound mirror'
    assert not list(directory.glob('*.npy')), 'Inspect partial mirror before rebuilding'
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    assert [r['history_id'] for r in rows] == list(range(len(rows)))
    lengths = {int(r['T']) for r in rows}
    assert len(lengths) == 1, 'This bank requires the actual fixed-length native corpus'
    length = lengths.pop()
    groups = [r['h5_group'].lstrip('/') for r in rows]
    assert len(set(groups)) == len(rows)
    fields, arrays, receipts = {}, {}, []
    with h5py.File(h5_path, 'r', rdcc_nbytes=4 << 20) as source:
        first = source[groups[0]]
        for name in sorted(first):
            ds = first[name]
            shape = (len(rows), *ds.shape)
            assert ds.shape[0] == length
            dtype = ds.dtype
            storage = np.dtype('uint8') if name == 'obs' else dtype
            path = directory / (name + '.npy')
            arrays[name] = np.lib.format.open_memmap(path, mode='w+', dtype=storage, shape=shape)
            fields[name] = dict(file=path.name, shape=list(shape), dtype=dtype.str,
                                storage_dtype=storage.str, promoted=False)
        for hid, group_name in enumerate(groups):
            group = source[group_name]
            assert sorted(group) == sorted(fields)
            for name, spec in fields.items():
                ds = group[name]
                assert list(ds.shape) == spec['shape'][1:] and ds.dtype.str == spec['dtype']
                h = hashlib.sha256()
                for start in range(0, length, 512):
                    end = min(start + 512, length)
                    original = ds[start:end]
                    candidate = original.astype(arrays[name].dtype)
                    raw = original.tobytes()
                    if candidate.astype(ds.dtype).tobytes() != raw:
                        assert name == 'obs' and arrays[name].dtype == np.uint8
                        old = arrays[name]
                        promoted_path = directory / (name + '_original_dtype.npy')
                        promoted = np.lib.format.open_memmap(promoted_path, mode='w+',
                            dtype=ds.dtype, shape=tuple(spec['shape']))
                        for prior in range(hid + 1):
                            stop = length if prior < hid else start
                            for offset in range(0, stop, 512):
                                promoted[prior, offset:min(offset + 512, stop)] = old[prior, offset:min(offset + 512, stop)]
                            drop_clean_pages(promoted)
                        old_path = Path(old.filename)
                        old._mmap.close()
                        old_path.unlink()
                        arrays[name] = promoted
                        spec.update(file=promoted_path.name, storage_dtype=ds.dtype.str, promoted=True)
                        candidate = original
                    arrays[name][hid, start:end] = candidate
                    # Verify stored values, not just the intended conversion.
                    assert arrays[name][hid, start:end].astype(ds.dtype).tobytes() == raw
                    h.update(raw)
                drop_clean_pages(arrays[name])
                receipts.append(dict(history_id=hid, field=name, original_array_sha256=h.hexdigest(),
                                     stored_round_trip_byte_exact=True))
            write(directory / 'progress.json', dict(status='BUILDING_LOSSLESS_MIRROR',
                completed_histories=hid + 1, total_histories=len(rows), at=time.time()))
            if hid % 64 == 0:
                print(json.dumps(dict(event='mirror_history_complete', history=hid + 1, total=len(rows))), flush=True)
    for name, array in arrays.items():
        drop_clean_pages(array)
        array._mmap.close()
        fields[name]['file_sha256'] = digest(directory / fields[name]['file'])
        st = (directory / fields[name]['file']).stat()
        fields[name]['file_stat'] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
    proof = directory / 'array_equivalence.json'
    write(proof, receipts)
    result = dict(status='PASS_LOSSLESS_MAPPED_HISTORY', schema=1, at=time.time(),
        source_h5=dict(path=str(h5_path.resolve()), sha256=digest(h5_path)),
        source_index=dict(path=str(index_path.resolve()), sha256=digest(index_path)),
        adapter_sha256=digest(__file__), histories=len(rows), length=length,
        groups=groups, fields=fields, array_equivalence_sha256=digest(proof),
        source_dtype_preserved_on_read=True, all_arrays_byte_exact=True)
    write(directory / 'manifest.json', result)
    return result


class MappedDataset:
    def __init__(self, array, history, dtype):
        self.array, self.history, self.dtype = array, history, np.dtype(dtype)
        self.shape = array.shape[1:]

    def __getitem__(self, key):
        # A bounded slice copy matches HDF5 ownership and its original dtype.
        return np.array(self.array[self.history][key], dtype=self.dtype, copy=True)


class MappedGroup:
    def __init__(self, owner, history):
        self.owner, self.history = owner, history

    def __getitem__(self, key):
        return MappedDataset(self.owner.arrays[key], self.history, self.owner.info['fields'][key]['dtype'])

    def __contains__(self, key):
        return key in self.owner.arrays

    def __iter__(self):
        return iter(self.owner.arrays)

    def __len__(self):
        return len(self.owner.arrays)


class MappedFile:
    mode = 'r'

    def __init__(self, manifest):
        self.path = Path(manifest)
        self.info = json.loads(self.path.read_text())
        assert self.info['status'] == 'PASS_LOSSLESS_MAPPED_HISTORY'
        assert self.info['adapter_sha256'] == digest(__file__)
        self.ids = {name: i for i, name in enumerate(self.info['groups'])}
        self.arrays = {}
        for name, spec in self.info['fields'].items():
            st = (self.path.parent / spec['file']).stat()
            assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino) == spec['file_stat']
            array = np.load(self.path.parent / spec['file'], mmap_mode='r', allow_pickle=False)
            assert list(array.shape) == spec['shape'] and array.dtype.str == spec['storage_dtype']
            self.arrays[name] = array

    def __getitem__(self, key):
        return MappedGroup(self, self.ids[key.lstrip('/')])

    def __contains__(self, key):
        return key.lstrip('/') in self.ids

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


def enable_mapped_store(store, manifest=None):
    if isinstance(store._h5f, MappedFile):
        return store
    manifest = Path(manifest or (store.h5_path.parent / 'mapped/manifest.json'))
    mapped = MappedFile(manifest)
    assert Path(mapped.info['source_h5']['path']).resolve() == store.h5_path.resolve()
    assert digest(store.index_path) == mapped.info['source_index']['sha256']
    assert len(store) == mapped.info['histories']
    assert [r['h5_group'].lstrip('/') for r in store._index] == mapped.info['groups']
    store._h5f.close()
    store._h5f = mapped
    return store


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = build(args.h5, args.index, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'histories', 'length', 'all_arrays_byte_exact')}), flush=True)
