"""One-time sealed test input/plan binding AFTER the global freeze.

Consumes already extracted/audited test artifacts; does not extract raw data.
The producer receipt must certify the unchanged frozen RGB/pose input route.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from runtime import VERSION, artifact, checked, read, sha, verify_freeze, write


def seedof(s): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'little')


def prepare(args):
    freeze = verify_freeze(args.freeze); rule = freeze['scene_rules'][args.scene]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dest = out/'prepared.json'
    if dest.exists():
        saved = read(dest)
        if saved['freeze_sha256'] != sha(args.freeze) or saved['scene'] != args.scene: raise ValueError('Existing sealed bundle belongs to another freeze')
        for item in saved['files']: checked(item)
        print('SEALED_BUNDLE_ALREADY_PREPARED'); return
    producer = read(args.producer_receipt)
    if (producer.get('status') != 'COMPLETE' or producer.get('split') != 'test' or producer.get('scene') != args.scene
            or producer.get('freeze_sha256') != sha(args.freeze) or producer.get('query_frames') != 3
            or producer.get('pose_source') != 'same_frozen_derenderer_as_training' or producer.get('gt_current_pose_used') is not False):
        raise ValueError('Test producers are not qualified for the frozen predicted-current input route')
    for name in ('feature_producer', 'pose_producer', 'field_auditor'):
        if producer[name]['sha256'] != rule[name]['sha256']: raise ValueError('Changed test producer/auditor')
        checked(producer[name])
    if producer['frontend_checkpoint_sha256'] != rule['frontend_checkpoint_sha256']:
        raise ValueError('Test frontend differs from the frozen training frontend')
    files = [artifact(args.producer_receipt)]
    def use(name):
        item = producer['artifacts'][name]; path = checked(item); files.append(item); return path
    ids_path = use('ids'); ids = list(map(str, read(ids_path)))
    split_path = checked(rule['official_split']); official_ids = sorted(split_path.read_text().split())
    if ids != official_ids or len(ids) != len(set(ids)) or not ids:
        raise ValueError('Use every official test episode in fixed sorted order')
    # The source development domains remain frozen and distinct from test.
    for item in rule['development_id_files']:
        path = checked(item)
        if set(ids) & set(map(str, read(path))): raise ValueError('Test/development episode overlap')
    feature_names = ('features_ab', 'features_c', 'presence_ab', 'presence_c')
    arrays = {name: np.load(use(name), mmap_mode='r', allow_pickle=False) for name in feature_names}
    frames, slots, dims, horizon = {'balls':(30,9,2,27),'collision':(15,4,3,12),'blocktower':(30,4,3,27)}[args.scene]
    if arrays['features_ab'].shape != (len(ids),frames,slots,784) or arrays['features_c'].shape != (len(ids),3,slots,784):
        raise ValueError('Test features must be full AB and exactly three current frames')
    if arrays['presence_ab'].shape != (len(ids),frames,slots) or arrays['presence_c'].shape != (len(ids),3,slots):
        raise ValueError('Test feature presence shape changed')
    xp, yp, raw_path = use('input'), use('target'), use('raw_relations')
    pose_ab_path, pose_presence_path = use('pose_ab'), use('pose_presence_ab')
    pose_ab = np.load(pose_ab_path, mmap_mode='r', allow_pickle=False)
    pose_presence = np.load(pose_presence_path, mmap_mode='r', allow_pickle=False)
    if pose_ab.shape != (len(ids),frames,slots,3) or pose_presence.shape != (len(ids),slots):
        raise ValueError('Supervised AB pose/first-frame presence shape differs')
    with np.load(xp, allow_pickle=False) as x:
        if list(map(str,x['ids'])) != ids or x['pose'].shape != (len(ids),3,slots,dims): raise ValueError('Predicted-current input differs')
        mask = np.array(x['presence'], copy=True)
        if mask.shape != (len(ids),slots) or not np.isfinite(x['pose']).all() or not np.isfinite(x['detected']).all(): raise ValueError('Invalid predicted input')
    with np.load(yp, allow_pickle=False) as y:
        if list(map(str,y['ids'])) != ids or y['pose'].shape != (len(ids),horizon,slots,dims) or not np.isfinite(y['pose']).all():
            raise ValueError('Test target/order differs from CD[3:]')
    official_c, official_target = use('official_c'), use('target_official')
    with np.load(official_c, allow_pickle=False) as c:
        if (list(map(str,c['ids'])) != ids or c['pose'].shape != (len(ids),1,slots,3)
                or c['presence'].shape != (len(ids),slots) or not np.isfinite(c['pose']).all()):
            raise ValueError('Official predicted C input differs')
    with np.load(official_target, allow_pickle=False) as y:
        if list(map(str,y['ids'])) != ids or y['pose'].shape != (len(ids),frames-1,slots,dims) or not np.isfinite(y['pose']).all():
            raise ValueError('Official task targets must be CD[1:]')
    records = read(raw_path); by_id = {}; groups = defaultdict(lambda:defaultdict(list))
    rgb_seen = np.asarray(arrays['presence_ab'],bool).any(1)
    seen = rgb_seen & (np.asarray(pose_presence)>0); lookup = {q:i for i,q in enumerate(ids)}
    for r in records:
        ident, slot = str(r['id']), int(r['slot'])
        if r['split'] != 'test' or ident not in lookup or not 0 <= slot < slots or (ident,slot) in by_id:
            raise ValueError('Invalid audited test object')
        by_id[(ident,slot)] = r
        typ = r['known_type']; gravity = r.get('raw_gravity') if args.scene == 'blocktower' else None
        public = (slot,__import__('json').dumps(typ,sort_keys=True),__import__('json').dumps(gravity,sort_keys=True))
        label = tuple(r['physical'])
        if seen[lookup[ident],slot]: groups[public][label].append(lookup[ident])
    correct = np.zeros((len(ids),slots,3),np.int64); wrong = correct.copy()
    for i, ident in enumerate(ids):
        for arm, plan in (('correct',correct),('wrong',wrong)):
            rng = np.random.default_rng(seedof(f'xep:20260911:test:{ident}:0'))
            for slot in np.flatnonzero(mask[i] > 0):
                r = by_id[(ident,int(slot))]; gravity = r.get('raw_gravity') if args.scene == 'blocktower' else None
                public = (int(slot),__import__('json').dumps(r['known_type'],sort_keys=True),__import__('json').dumps(gravity,sort_keys=True))
                label = tuple(r['physical']); group = groups[public]
                pool = sorted(v for other, members in group.items() for v in members if (other == label) == (arm == 'correct') and v != i)
                plan[i,slot] = np.asarray(pool)[rng.choice(len(pool),3,replace=False)] if len(pool)>=3 else -1
    eligible = ((correct>=0).all(2) | (mask<=0)).all(1) & (mask>0).any(1)
    wrong_eligible = eligible & (((wrong>=0).all(2) | (mask<=0)).all(1))
    plan_path = out/'plans.npz'; tmp = out/'plans.pending.npz'
    np.savez(tmp, correct=correct, wrong=wrong, correct_eligible=eligible, wrong_eligible=wrong_eligible); tmp.replace(plan_path)
    files.append(artifact(plan_path))
    inputs = {name: producer['artifacts'][name] for name in (*feature_names,'ids','input','target','raw_relations','pose_ab','pose_presence_ab','official_c','target_official')}
    result = dict(version=VERSION,status='PREPARED' if eligible.any() else 'UNQUALIFIED_NO_CORRECT_SUPPORT',
        scene=args.scene,split='test',freeze_sha256=sha(args.freeze),query_frames=3,supports=3,
        query_ids=ids,history_domain='same_test_split; independent episode; donor reads AB only',
        sampler='fixed seed xep:20260911:test:ID:0; same slot/public type/global gravity; Wrong-any differs in object physical key',
        donor_visibility='intersection of RGB AB any-visible and supervised AB first-frame pose presence; one common plan for all45 roles',
        donor_objects_rgb_visible=int(rgb_seen.sum()),donor_objects_shared_visible=int(seen.sum()),
        plans=artifact(plan_path),inputs=inputs,files=files,official_test_count=len(ids),
        correct_eligible_count=int(eligible.sum()),wrong_eligible_count=int(wrong_eligible.sum()),
        unsupported_query_ids=[q for i,q in enumerate(ids) if not eligible[i]],
        gt_current_pose_used=False,manifest_resampling_allowed=False,test_read=True,optimizer_steps=0)
    write(dest,result,immutable=True); print(result['status'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',required=True);p.add_argument('--scene',required=True,choices=('balls','collision','blocktower'))
    p.add_argument('--producer-receipt',required=True);p.add_argument('--out',required=True)
    prepare(p.parse_args())


if __name__=='__main__':main()
