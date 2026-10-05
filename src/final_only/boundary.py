"""Freeze commitments and one-shot evaluator handoff, without data discovery.

This is an accidental-access/provenance boundary, not an OS sandbox. The native
SealedAuthorization/SealedBank checks remain mandatory in the final evaluator.
No final evaluation is authorized merely by installing this package.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
from sprii_next.io import read,write,sha,digest,code_hashes,development_path

ROLES=('development_protocol','reader_recipe','statistical_plan','claims','primary_figures',
       'source_checkpoints','reader_checkpoints','baseline_qualification','decode_recipe','native_evaluator_contract')
METHODS=('Native','Structure','Cross','Both','RelInfoNCE')


def boundary_hashes():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def freeze(roles,output):
    if set(roles)!=set(ROLES):raise ValueError('every required freeze role must be explicitly supplied')
    artifacts={}
    for role,paths in roles.items():
        if not isinstance(paths,list) or not paths:raise ValueError('nonempty file list required for '+role)
        artifacts[role]=[{"path":str(development_path(p).resolve()),"sha256":sha(development_path(p))} for p in paths]
    qualifications=[read(d['path']) for d in artifacts['baseline_qualification']]
    if not qualifications or any(q.get('status')!='COMPLETE' or q.get('test_read') is not False
                                or q.get('recipe_id')!='fcrl_style_temporal_v1' or q.get('smoke',False) for q in qualifications):
        raise PermissionError('completed fixed-recipe baseline must be frozen before test; downstream strength is not a gate')
    contract=read(artifacts['native_evaluator_contract'][0]['path'])
    if contract.get('status')!='VALIDATED_ON_SYNTHETIC_DATA' or contract.get('native_sealed_authorization_required') is not True:
        raise PermissionError('native evaluator integration must be validated before final freeze')
    if contract.get('methods')!=list(METHODS) or contract.get('statistical_unit')!='physical_system':
        raise ValueError('final method matrix/statistical unit differs')
    result=dict(schema='sprii-next.final-freeze.v1',status='FROZEN',test_read=False,artifacts=artifacts,
        development_code=code_hashes(),boundary_code=boundary_hashes(),methods=list(METHODS),source_seeds=[0,1,2],reader_seeds=[0,1,2],
        primary_comparisons=['1-E_Both/E_Native','1-E_Both/E_RelInfoNCE'],secondary_comparisons=['Cross vs Both'],
        test_policy='one claim per freeze identity; no development reuse; failure consumes claim; no automatic retry')
    write(output,result);return result


def verify_freeze(path,expected_sha256):
    if sha(path)!=expected_sha256:raise ValueError('freeze receipt changed')
    record=read(path)
    if record.get('schema')!='sprii-next.final-freeze.v1' or record.get('status')!='FROZEN' or record.get('test_read') is not False:
        raise PermissionError('pre-test frozen receipt required')
    if record['development_code']!=code_hashes() or record['boundary_code']!=boundary_hashes():raise ValueError('code changed after freeze')
    for values in record['artifacts'].values():
        for d in values:
            if sha(d['path'])!=d['sha256']:raise ValueError('frozen role changed: '+d['path'])
    return record


@contextmanager
def evaluator_ticket(freeze_path,expected_sha256,ledger_root):
    """Called before constructing native SealedAuthorization or any test loader.

The ledger directory is itself committed by the evaluator contract. Changing an
output directory cannot silently create a second attempt for the same freeze.
"""
    record=verify_freeze(freeze_path,expected_sha256)
    contract=read(record['artifacts']['native_evaluator_contract'][0]['path'])
    ledger=Path(ledger_root).resolve()
    if ledger!=Path(contract['one_shot_ledger']).resolve():raise PermissionError('one-shot ledger differs from frozen evaluator contract')
    ledger.mkdir(parents=True,exist_ok=True)
    claim=ledger/(expected_sha256+'.CLAIM.json')
    # Exclusive create happens before the caller is allowed to inspect test data.
    fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(dict(freeze_sha256=expected_sha256,status='CLAIMED',test_read=False),f)
        f.flush();os.fsync(f.fileno())
    try:
        yield record
        verify_freeze(freeze_path,expected_sha256)
    except BaseException as error:
        write(ledger/(expected_sha256+'.FAILURE.json'),dict(status='FAILED',error=repr(error),test_may_have_been_read=True,retry_allowed=False))
        raise
    else:
        write(ledger/(expected_sha256+'.CLOSED.json'),dict(status='CLOSED',freeze_sha256=expected_sha256,
            note='native evaluator must separately attest complete rows and post-run verification'))
