"""Check legacy relation eligibility against committed RGB visibility, on CPU.

Only train AB presence and train CD[:3] presence are read. No training, model,
validation/test data, coordinate target, or feature784 array is opened.
COMPLETE means all rows were checked; only all_zero proves shared eligibility.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

VERSION = 'cophy-v7-relation-visibility-proof-1'
SCENES = {'balls': (30, 9), 'collision': (15, 4), 'blocktower': (30, 4)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def check_scene(scene, feature_root, index_path):
    root = Path(feature_root).resolve() / scene
    train = root / 'train'
    index_path = Path(index_path).resolve()
    files = [root / 'scene_COMPLETE.json', train / 'COMPLETE.json',
             train / 'manifest.json', train / 'ids.json',
             train / 'presence_ab.npy', train / 'presence_cd.npy', index_path]
    before = {str(p): sha(p) for p in files}
    complete, scene_complete, manifest = (read(train / 'COMPLETE.json'),
                                         read(root / 'scene_COMPLETE.json'),
                                         read(train / 'manifest.json'))
    if complete.get('status') != 'COMPLETE' or scene_complete.get('status') != 'COMPLETE':
        raise ValueError(f'{scene}: wait for committed full cache')
    if complete.get('manifest_sha256') != before[str(train / 'manifest.json')]:
        raise ValueError(f'{scene}: feature manifest hash mismatch')
    if manifest.get('scene') != scene or manifest.get('split') != 'train' or manifest.get('test_read') is not False:
        raise ValueError(f'{scene}: unexpected cache data domain')
    ids = list(map(str, read(train / 'ids.json')))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'{scene}: empty/duplicate training IDs')
    if complete.get('ids_sha256') != before[str(train / 'ids.json')] or ids != list(map(str, manifest['ids'])):
        raise ValueError(f'{scene}: committed training IDs differ from the manifest')
    lut = {ident: i for i, ident in enumerate(ids)}
    index = read(index_path)
    if (index.get('version') != 'cophy-relation-index-v3' or index.get('scene') != scene
            or index.get('split') != 'train'):
        raise ValueError(f'{scene}: wrong RelationIndex')
    ab = np.load(train / 'presence_ab.npy', mmap_mode='r', allow_pickle=False)
    cd = np.load(train / 'presence_cd.npy', mmap_mode='r', allow_pickle=False)
    frames, slots = SCENES[scene]
    if ab.shape != (len(ids), frames, slots) or cd.shape != ab.shape:
        raise ValueError(f'{scene}: unexpected visibility tensor shapes')
    ab_values = np.asarray(ab)
    current_values = np.asarray(cd[:, :3])
    if not np.isin(ab_values, (0, 1)).all() or not np.isin(current_values, (0, 1)).all():
        raise ValueError(f'{scene}: presence mask is not binary')
    seen = ab_values.astype(bool).any(1)
    current = current_values.astype(bool).any(1)
    failures = {'recipient_ab_absent': [], 'recipient_query3_absent': [], 'donor_ab_absent': []}
    eligible = {'recipient': [], 'donor': []}
    seen_keys = set()
    for record in index['records']:
        ident, slot = str(record['id']), record['slot']
        if ident not in lut or type(slot) is not int or not 0 <= slot < slots or (ident, slot) in seen_keys:
            raise ValueError(f'{scene}: invalid/duplicate RelationIndex object')
        if type(record['recipient']) is not bool or type(record['donor']) is not bool:
            raise ValueError(f'{scene}: eligibility flags are not audited booleans')
        seen_keys.add((ident, slot))
        i, item = lut[ident], {'id': ident, 'slot': slot}
        if record['recipient']:
            eligible['recipient'].append(item)
            if not seen[i, slot]: failures['recipient_ab_absent'].append(item)
            if not current[i, slot]: failures['recipient_query3_absent'].append(item)
        if record['donor']:
            eligible['donor'].append(item)
            if not seen[i, slot]: failures['donor_ab_absent'].append(item)
    if not all(eligible.values()):
        raise ValueError(f'{scene}: no eligible recipient or donor objects; cannot give a vacuous proof')
    after = {str(p): sha(p) for p in files}
    if before != after:
        raise ValueError(f'{scene}: input changed during visibility scan')
    counts = {}
    for name, rows in failures.items():
        role = 'donor' if name.startswith('donor_') else 'recipient'
        denominator = len(eligible[role])
        counts[name] = dict(objects=len(rows), unique_ids=len({r['id'] for r in rows}),
                            eligible_objects=denominator,
                            object_fraction=len(rows) / denominator if denominator else None,
                            objects_by_id_slot=rows)
    all_zero = all(not rows for rows in failures.values())
    return dict(status='COMPLETE', scene=scene, all_zero=all_zero,
                training_episodes=len(ids), indexed_objects=len(seen_keys),
                eligibility={role: dict(objects=len(rows), unique_ids=len({r['id'] for r in rows}))
                             for role, rows in eligible.items()},
                violations=counts, files=before, query_frames=3,
                proof_scope='All audited eligible recipients and donors; hence every sampled focal pair if all_zero.',
                not_proven='Does not equate Collision focal and all-object routing, different Random permutations, or objectives.',
                future_cd_values_read=False, coordinate_targets_read=False, test_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--relation-index', action='append', required=True, metavar='SCENE=PATH')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    indices = {}
    for entry in args.relation_index:
        scene, separator, path = entry.partition('=')
        if not separator or scene not in SCENES or scene in indices or not path:
            parser.error('Provide exactly one --relation-index SCENE=PATH per registered scene')
        indices[scene] = path
    if set(indices) != set(SCENES):
        parser.error('All three scenes are required')
    began = time.monotonic()
    scenes = {scene: check_scene(scene, args.features_root, indices[scene]) for scene in SCENES}
    result = dict(version=VERSION, status='COMPLETE', all_zero=all(s['all_zero'] for s in scenes.values()),
                  scenes=scenes, implementation_sha256=sha(__file__), seconds=time.monotonic()-began,
                  optimizer_steps=0, training_split_only=True, test_read=False)
    atomic_write(args.out, result)
    print(json.dumps(dict(status=result['status'], all_zero=result['all_zero'],
                          scenes={s: {k: v['objects'] for k, v in r['violations'].items()}
                                  for s, r in scenes.items()}, out=str(Path(args.out).resolve())), allow_nan=False))


if __name__ == '__main__':
    main()
