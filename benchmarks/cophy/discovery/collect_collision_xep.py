"""Compact receipt of real Collision progress and selected validation results."""
import time
from pathlib import Path
from collision_xep import ROOT, read, write

out=ROOT/'xep_discovery_collision_v4_4'
result={'time':time.time(),'test_read':False,'controller':read(out/'controller_status.json'),'methods':{}}
if (out/'manifest.json').exists():
    m=read(out/'manifest.json')
    result['coverage']={s:{'total':len(p['all_ids']),'eligible':p['eligible_total'],
        'queries':len(p['query_ids']),'minimum_support_pool':min(len(c) for rows in p['candidates'].values() for c in rows if c)}
        for s,p in m['splits'].items()}
    result['sources']={k:{f:v[f] for f in ['path','sha256','epoch']} for k,v in m['source_models'].items()}
for method in ['Native-U','A-U','Random-U','Known-parameters','A-P','Query-only']:
    run=out/'runs'/method
    if not (run/'progress.json').exists():continue
    progress=read(run/'progress.json');chosen=read(run/'selected_validation.json')
    result['methods'][method]={'trained_epoch':progress['epoch'],'selected':chosen,
        'seconds_per_epoch':progress['history'][-1]['seconds']}
write(out/'result_snapshot.json',result)
print('STATUS',result['controller']['status'],result['controller'].get('current_stage'))
print('COVERAGE',result.get('coverage'))
for method,entry in result['methods'].items():
    print(method,'trained',entry['trained_epoch'],'selected',entry['selected']['epoch'],
          'mse',entry['selected']['mse'],'seconds_per_epoch',entry['seconds_per_epoch'])
