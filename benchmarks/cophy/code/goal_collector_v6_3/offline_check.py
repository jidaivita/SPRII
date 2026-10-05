"""Limited adapter check against real, previously captured receipts.

This never represents a current remote inventory, and does not manufacture
checkpoints or training data. It exercises the collector's metric parsing on
full saved arrays and the identity/budget fields on the latest local snapshot.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collect import Collector, close


def main():
    ext=Path(__file__).resolve().parents[2]
    spec=json.loads((ext/'protocol/goal_collector_v6_3/spec.json').read_text())
    c=Collector(argparse.Namespace(root=None,final=False),spec)
    rawpath=ext/'receipts/raw/balls_source10_readout_raw_20260912.json.gz'
    raw=json.load(gzip.open(rawpath,'rt'));checks=[]
    for name,item in raw['files'].items():
        if not name.endswith('/results.json'):continue
        d=item['data'];row=c.newrow(dict(kind='offline_arm_schema',folder=spec['root']+'/'+name))
        c.arms(row,d,512,d['config']['reference']=='learned')
        checks.append(dict(source=name,declared_original_sha256=item['sha256'],
                           source_epoch=10 if '/source10/' in name else None,
                           head_budget=d['config']['epochs'],excluded_from_final_source50_inventory='/source10/' in name,
                           contradictions=row['contradictions'],metric_cohorts=row['metrics']))
    snapshot=ext/'receipts/JEPA_v62_STAGE_AND_FULLVAL_INVENTORY_20260912.json'
    d=json.loads(snapshot.read_text())['original512_and_fullval_evidence']
    complete_sources=0
    for scene,content in d['scenes'].items():
        for method,r in content['methods'].items():
            source=r['source'].get('complete')
            if not source or source.get('status')!='COMPLETE':continue
            row=c.newrow(dict(kind='offline_source_schema',folder=spec['root'],scene=scene,method=method))
            c.fields(row,source,dict(version=spec['source_versions']['JEPA'],family='JEPA',scene=scene,method=method,
                                    epochs=50,selected_epoch=50,test_read=False),'source')
            checks.append(dict(source=source.get('checkpoint'),contradictions=row['contradictions']))
            complete_sources+=1
    summary=dict(status='PASS_OFFLINE_SCHEMA_ONLY' if not any(x['contradictions'] for x in checks) else 'ADAPTER_REVIEW_NEEDED',
                 snapshot_captured_at_unix=d['captured_at_unix'],current_remote_state_verified=False,
                 original_receipts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (rawpath,snapshot)},
                 checks=checks,completed_source_receipts_checked=complete_sources,
                 file_hash_chains_rechecked=False,checkpoints_loaded=False,new_model_samples=0,test_read=False,
                 limitation='Only existing local receipt schemas and saved arm arrays. Run collect.py remotely after access returns.')
    out=ext/'receipts/Goal_Collector_Offline_Adapter_Check_20260912.json'
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(checks),path=str(out),current_remote_state_verified=False)))


if __name__=='__main__':main()
