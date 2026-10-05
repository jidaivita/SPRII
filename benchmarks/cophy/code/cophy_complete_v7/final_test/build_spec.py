"""Discover exact completed TRAIN/VAL artifacts for the fixed final-test scope.

Does not open test samples or launch anything. An unresolved path is an error,
not permission to substitute another checkpoint, budget, head, or code version.
"""
import argparse
from pathlib import Path
from runtime import VERSION, read, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('presentation-manifest','reference-assets','scene-rules','protocol','main-summary-dir','references-summary-dir','wrong-gravity-complete','probe-fit-root','out'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--code-root',action='append',required=True)
    a=p.parse_args();manifest=read(a.presentation_manifest);assets=read(a.reference_assets)
    if manifest.get('source_budget_main')!=100 or manifest.get('seed')!=0 or manifest.get('test_read') is not False:
        raise ValueError('Use the already registered source100 primary manifest')
    cores={}
    for root in map(Path,a.code_root):
        for name in ('readout.py','supervised_readout.py'):
            for path in sorted(root.rglob(name)):cores.setdefault(sha(path),str(path.resolve()))
    def core_for(h):
        if h not in cores:raise ValueError('Exact bound readout code not found in the supplied code roots: '+h)
        return cores[h]
    entries=[]
    for row in manifest['rows']:
        if row.get('source_budget')!=100 or row.get('required',True) is False:continue
        rd=Path(row['readout_out']);family='supervised' if row['family'] in ('CoPhyNet','supervised') else row['family']
        if family=='supervised':
            encoded=read(rd/'codes_complete.json');core=core_for(encoded['implementation_sha256'])
            params=read(rd/'S3/learned/config.json')['input_sha256']
            base=str(next(Path(k).parent for k in params if Path(k).name=='input_train.npz'))
        else:
            encoded=read(rd/'encoding_complete.json')['binding'];core=core_for(encoded['readout_code_sha256']);base=encoded['base']
        full=Path(row.get('fullval_results',rd/'S3/learned/fullval/results.json'))
        entry=dict(id=row['id'],family=family,scene=row['scene'],role=row['role'],source_method=row['source_method'],
            readout=str(rd),core=core,base=base,source_checkpoint=row['source_checkpoint'],
            probe_fit=str(Path(a.probe_fit_root)/row['id']/'probe_fit.json'),
            fullval_results=str(full),fullval_marker=str(row.get('fullval_marker',full.parent/'complete.json')),
            probe_report=str(row.get('probe_path',rd/'probes.json')))
        if family=='supervised':
            entry['source_budget_receipt']=str(Path(row['source_checkpoint']).parent/'budget_100_complete.json')
        entries.append(entry)
    references=[]
    for row in assets['references']:
        freeze=read(Path(row['results']).parent/'checkpoint_freeze.json')
        references.append(dict(id=f"Reference-{row['scene']}-{row['reference']}",scene=row['scene'],
            reference='Query-only' if row['reference']=='query' else 'Known-parameters',results=row['results'],
            core=core_for(freeze['head_implementation_sha256'])))
    if len(entries)!=45 or len(references)!=6:raise ValueError('The final source100 matrix must contain 45 learned + 6 references')
    m=Path(a.main_summary_dir);r=Path(a.references_summary_dir)
    spec=dict(version=VERSION,status='SPECIFICATION_ONLY_NOT_TEST_PERMIT',seed=0,main_source_epochs=100,supports=3,query_frames=3,
        learned_entries=entries,references=references,expected_ids=[e['id'] for e in entries+references],
        scene_rules=read(a.scene_rules),protocol=str(Path(a.protocol).resolve()),
        validation_closure=dict(main_complete=str(m/'complete.json'),main_summary=str(m/'summary.json'),
            references_official_complete=str(r/'references_official_complete.json'),references_official_summary=str(r/'references_official.json'),
            wrong_gravity_complete=str(Path(a.wrong_gravity_complete).resolve())),
        input_manifest=dict(path=str(Path(a.presentation_manifest).resolve()),sha256=sha(a.presentation_manifest)),
        reference_assets=dict(path=str(Path(a.reference_assets).resolve()),sha256=sha(a.reference_assets)),test_read=False)
    write(a.out,spec,immutable=True);print('SPECIFICATION_WRITTEN; test remains unopened')


if __name__=='__main__':main()
