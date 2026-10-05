"""Freeze the fixed source100/seed0 main matrix without opening test samples.

The explicit specification must list all 45 learned roles and 6 references.
Incomplete artifacts produce NOT_READY, never a preparation permit.
"""
import argparse
from collections import Counter
from pathlib import Path
import torch
from runtime import VERSION, SCENES, artifact, checked, read, sha, verify_freeze, write


def freeze(specification, out):
    spec = read(specification); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if spec.get('main_source_epochs') != 100 or spec.get('seed') != 0 or spec.get('supports') != 3 or spec.get('query_frames') != 3:
        raise ValueError('This stage is source100/seed0/S3/query3 only; no test budget selection')
    entries = spec['learned_entries']; references = spec['references']
    expected = spec['expected_ids']
    if len(entries) != 45 or len(references) != 6 or sorted(expected) != sorted(e['id'] for e in entries+references) or len(set(expected)) != 51:
        raise ValueError('Final freeze needs the explicit complete 45-role + 6-reference matrix')
    counts = Counter((e['scene'], e['family']) for e in entries)
    if counts != Counter({(s, f): (3 if f == 'supervised' else 4) for s in SCENES for f in ('supervised', 'JEPA', 'CPC', 'RSSM')}):
        raise ValueError('Learned role matrix differs from the registered four-family scope')
    if Counter((e['scene'], e['reference']) for e in references) != Counter({(s, r): 1 for s in SCENES for r in ('Query-only', 'Known-parameters')}):
        raise ValueError('Reference scope differs')
    if set(spec['scene_rules']) != set(SCENES): raise ValueError('All three scene preparation rules are required')
    files = {}; rows = []; problems = []
    def bind(path):
        item = artifact(path); files[item['path']] = item; return item
    bind(specification); bind(spec['protocol'])
    for key in ('input_manifest','reference_assets'):bind(checked(spec[key]))
    for path in sorted(Path(__file__).parent.glob('*.py')): bind(path)
    # Do not open test while the registered source50/100/150 curves are still
    # being run or inspected. A source100 arriving early is insufficient.
    closure=spec['validation_closure']
    try:
        main_marker=read(closure['main_complete']);main_summary=read(closure['main_summary'])
        reference_marker=read(closure['references_official_complete']);reference_summary=read(closure['references_official_summary'])
        gravity_marker=read(closure['wrong_gravity_complete'])
    except FileNotFoundError as exc:
        write(out/'freeze_readiness.json',dict(version=VERSION,status='NOT_READY',test_read=False,
            reason='Registered validation closure is incomplete',missing=str(exc)))
        print('NOT_READY: no test preparation permit issued');return
    if (main_marker.get('status')!='COMPLETE' or main_marker.get('test_read') is not False
            or main_marker.get('summary_sha256')!=sha(closure['main_summary'])
            or main_summary.get('counts',{}).get('required_complete')!=135
            or main_summary.get('counts',{}).get('required')!=135
            or main_summary.get('checkpoint_contents_verified') is not True
            or reference_marker.get('status')!='COMPLETE' or reference_marker.get('test_read') is not False
            or reference_marker.get('summary_sha256')!=sha(closure['references_official_summary'])
            or reference_summary.get('allow_archives') is not False
            or gravity_marker.get('status')!='COMPLETE' or gravity_marker.get('test_read') is not False):
        raise ValueError('All validation budgets/physical probes/references/official results must finish and be verified before test freeze')
    for path in closure.values():bind(path)
    # Binding official ID files is allowed here; raw test examples are unopened.
    rules = {}
    for scene, rule in spec['scene_rules'].items():
        if rule.get('query_frames') != 3 or rule.get('supports') != 3 or rule.get('history_domain') != 'same_test_split':
            raise ValueError('Unregistered sealed support/input policy')
        rules[scene] = dict(rule, official_split=bind(rule['official_split_path']),
                           feature_producer=bind(rule['feature_producer_path']),
                           pose_producer=bind(rule['pose_producer_path']),
                           field_auditor=bind(rule['field_auditor_path']))
        for key in ('archive_receipt','feature_training_manifest','frontend_checkpoint'):
            rules[scene][key]=bind(checked(rule[key]))
        for key in ('visual_source_files','development_id_files'):
            rules[scene][key]=[bind(checked(item)) for item in rule[key]]
        if rules[scene]['frontend_checkpoint']['sha256']!=rule['frontend_checkpoint_sha256']:
            raise ValueError('Frontend checkpoint does not match the registered training frontend')
    for e in entries:
        try:
            full=read(e['fullval_results']);full_marker=read(e['fullval_marker']);prior_probe=read(e['probe_report'])
            if full.get('status')!='COMPLETE' or full.get('test_read') is not False or full_marker.get('status')!='COMPLETE' or prior_probe.get('status')!='COMPLETE' or prior_probe.get('test_read') is not False:
                raise ValueError('Full validation and existing physical probe are required')
            arms=full.get('arms',full);match=arms.get('matched')
            if match is None or len(match['ids'])!={'balls':2000,'collision':4000,'blocktower':8088}[e['scene']]:
                raise ValueError('Full validation must cover the registered whole scene')
            marker_hash=full_marker.get('results_sha256')
            if marker_hash is not None and marker_hash!=sha(e['fullval_results']):raise ValueError('Full-validation result hash changed')
            for key in ('fullval_results','fullval_marker','probe_report'):bind(e[key])
            if e['family']=='supervised':
                from supervised_adapter import freeze_entry
                rows.append(freeze_entry(e,bind));continue
            rd = Path(e['readout']); enc = read(rd/'encoding_complete.json'); b = enc['binding']
            folder = rd/'S3/learned'; conf = read(folder/'config.json'); done = read(folder/'complete.json')
            selection = read(folder/'selected_validation.json'); results = read(folder/'results.json')
            if enc['status'] != 'COMPLETE' or b['scene'] != e['scene'] or b['source_epochs'] != 100 or b['method'] != e['source_method']:
                raise ValueError('Source identity/budget mismatch')
            if b['test_read'] is not False or done['status'] != 'COMPLETE' or done['epochs'] != 100 or conf['epochs'] != 100 or conf['seed'] != 0:
                raise ValueError('Head budget/data domain mismatch')
            core = bind(e['core'])
            if b['readout_code_sha256'] != core['sha256']: raise ValueError('Wrong bound common readout code')
            source = bind(b['checkpoint']); model_code = bind(b['model_code'])
            if source['sha256'] != b['checkpoint_sha256'] or model_code['sha256'] != b['model_code_sha256']:
                raise ValueError('Source source/code hash changed')
            if Path(e['source_checkpoint']).resolve()!=Path(source['path']):raise ValueError('Source differs from the primary presentation row')
            source_ck = torch.load(source['path'], map_location='cpu', weights_only=False)
            if source_ck['epoch'] != 100 or source_ck.get('test_read') is not False:
                raise ValueError('Not the fixed actual source100 checkpoint')
            head = bind(folder/'selected.pt'); head_ck = torch.load(head['path'], map_location='cpu', weights_only=False)
            if head_ck['config'] != conf or head_ck['epoch'] != selection['epoch'] or head_ck['epoch'] != results['selected_epoch']:
                raise ValueError('Selected head identity differs')
            full_freeze_path=Path(e['fullval_results']).parent/'checkpoint_freeze.json'
            full_freeze=read(full_freeze_path)
            if (full_freeze.get('source_checkpoint_sha256')!=source['sha256']
                    or full_freeze.get('head_sha256')!=head['sha256']
                    or full_freeze.get('encoding_sha256')!=sha(rd/'encoding_complete.json')
                    or full_freeze.get('readout_code_sha256')!=core['sha256']
                    or full_freeze.get('head_budget')!=100 or full_freeze.get('source_epochs')!=100
                    or full.get('checkpoint_freeze_sha256')!=sha(full_freeze_path)):
                raise ValueError('Full validation did not evaluate this fixed source/head')
            bind(full_freeze_path)
            if prior_probe.get('encoding_sha256')!=sha(rd/'encoding_complete.json'):
                raise ValueError('Validation parameter probe belongs to another source encoding')
            if len(selection['ids']) != 512 or conf['selection_rows'] != 512 or conf['test_read'] is not False:
                raise ValueError('Only original validation512 may choose the head')
            if conf['code_sha256'] != sha(rd/'encoding_complete.json') or conf['normalization_sha256'] != sha(rd/'normalization.json'):
                raise ValueError('Head encoder/normalization binding differs')
            for path, expected_hash in enc['files'].items():
                if bind(path)['sha256'] != expected_hash: raise ValueError('Changed bound encoder cache')
            for path, expected_hash in conf['input_sha256'].items():
                if bind(path)['sha256'] != expected_hash: raise ValueError('Changed train/validation readout input')
            fit_path = Path(e['probe_fit']); fit = read(fit_path)
            if (fit['status'] != 'TRAIN_FIT_COMPLETE' or fit['fit_split'] != 'train' or fit['validation_used'] is not False
                    or fit['test_read'] is not False or fit['head_checkpoint']['sha256'] != head['sha256']
                    or fit['source_checkpoint']['sha256'] != source['sha256'] or fit['source_epochs'] != 100):
                raise ValueError('Probe must be the training-only fit for this source/head')
            if set(fit['channels']) != {'own', 'support_mean', 'memory'}: raise ValueError('Incomplete source/support/head probe channels')
            for channel in fit['channels'].values(): bind(checked(channel['artifact']))
            for key in ('readout', 'encoding', 'normalization', 'source_checkpoint', 'head_checkpoint', 'raw_train_relations', 'implementation', 'runtime'):
                bind(checked(fit[key]))
            for name in ('config.json', 'complete.json', 'selected_validation.json', 'results.json'): bind(folder/name)
            for name in ('encoding_config.json', 'encoding_complete.json', 'normalization.json', 'prepared.json'): bind(rd/name)
            fit_item = bind(fit_path)
            rows.append(dict(e, readout=str(rd.resolve()), core=core, source=source, model_code=model_code,
                             head=head, probe_fit=fit_item, support_dims=conf.get('support_dims',128), head_epochs=100,
                             selected_head_epoch=head_ck['epoch'], source_epochs=100,
                             normalization=bind(rd/'normalization.json'), encoding_binding=b))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            problems.append(dict(id=e['id'], state='MISSING_OR_NOT_QUALIFIED', reason=str(exc)))
    ref_rows = []
    for e in references:
        try:
            from references_adapter import freeze_reference
            ref_rows.append(freeze_reference(e,bind))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            problems.append(dict(id=e['id'], state='MISSING_OR_NOT_QUALIFIED', reason=str(exc)))
    if problems:
        write(out/'freeze_readiness.json', dict(version=VERSION, status='NOT_READY', problems=problems,
              learned_qualified=len(rows), references_qualified=len(ref_rows), test_read=False))
        print('NOT_READY: no test preparation permit issued'); return
    value = dict(version=VERSION, status='FROZEN_FOR_FINAL_TEST', seed=0, main_source_epochs=100,
        supports=3, query_frames=3, head_epochs=100, expected_ids=expected, learned_entries=rows,
        references=ref_rows, scene_rules=rules, bound_files=list(files.values()),
        selection='source100 fixed in advance; head selected once on original validation512; probes fitted only on train',
        test_manifest_policy='generate once after this freeze; no coverage/result-driven resampling',
        test_read=False, scope='45 learned roles plus six matched references; no secondary scale/seed expansion')
    write(out/'freeze.json', value, immutable=True)
    print('FROZEN_FOR_FINAL_TEST')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'verify')); p.add_argument('--specification'); p.add_argument('--out')
    p.add_argument('--freeze'); a = p.parse_args()
    if a.command == 'freeze':
        if not a.specification or not a.out: p.error('freeze requires --specification and --out')
        freeze(a.specification, a.out)
    else:
        if not a.freeze: p.error('verify requires --freeze')
        verify_freeze(a.freeze); print('FROZEN_FILES_VERIFIED; test not read')


if __name__ == '__main__': main()
