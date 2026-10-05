"""Generate the original deterministic nested 25/50/100% history subsets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0,str(a.repo.resolve()))
from native_a.large_batch import make_replacement_sampler_class
names = {'h5':'histories.h5','index':'histories_index.jsonl','task_manifest':'train_manifest.jsonl',
         'episode_index':'native_episode_index.jsonl'}
data = {k:{'path':str((a.data/v).resolve()),'sha256':hashlib.sha256((a.data/v).read_bytes()).hexdigest()}
        for k,v in names.items() if k != 'h5'}
sampler = make_replacement_sampler_class()(*[str((a.data/names[k]).resolve())
                                           for k in ('h5','index','task_manifest','episode_index')])
assert len(sampler.identities)==20 and all(len(v)==128 for v in sampler.identity_histories.values())
salt = 'overcooked-history-efficiency-v1'
a.out.mkdir(parents=True,exist_ok=False)
for pct,n in ((25,32),(50,64),(100,128)):
    chosen = {}
    for identity,hids in sampler.identity_histories.items():
        rank = sorted(hids,key=lambda h:hashlib.sha256((salt+'/'+identity+'/'+str(h)).encode()).hexdigest())
        chosen[identity] = sorted(rank[:n])
    selected = [h for hs in chosen.values() for h in hs]
    manifest = {'schema':'overcooked-history-subset/1','budget_percent':pct,'per_partner_histories':n,
                'selection_salt':salt,'identities':chosen,'metadata':data,'history_count':len(selected),
                'transition_count':sum(sampler.store.get_history_meta(h)['T'] for h in selected),
                'recorded_episode_prefixes':sum(len(sampler.episodes[h]) for h in selected),
                'episode_count_range':[min(len(sampler.episodes[h]) for h in selected),
                                       max(len(sampler.episodes[h]) for h in selected)],
                'entire_recorded_learning_time_range_retained':True}
    (a.out/f'{pct}.json').write_text(json.dumps(manifest,indent=2)+'\n')
