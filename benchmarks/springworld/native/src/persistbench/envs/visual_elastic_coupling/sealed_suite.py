"""Run and collect exact registered cells, retaining native scoring contracts.

Only run-cell can open a bank, after frozen authorization. Collection reads
saved outputs only. EXECUTED means complete, bound outcomes; neither it nor a
native PASS means the scientific claim passed or the benchmark is released.
Existing directories are never resumed, overwritten or selected by score.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from .dataset_snapshot import checked_asset, stable_digest
from .evaluation_blocks import digest, validate_plan
from .execution_matrix import load_matrix, ROUTES, PREDICTION_ASSAYS
from .sealed_access import SealedAuthorization
from .sealed_bank import SealedBank
from .training_protocol import source_fingerprint

RECEIPT = 'CELL_RECEIPT.json'


def write_new(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in result attempt')
        if path.is_file() and path != root / RECEIPT:
            name = path.relative_to(root).as_posix()
            result[name] = stable_digest(checked_asset(root, name))
    return result


def select_cell(matrix, cell_id):
    found = [c for c in matrix['cells'] if c['cell_id'] == cell_id]
    if len(found) != 1:
        raise ValueError('cell is absent or repeated in execution matrix')
    return found[0]


def bindings(matrix, *, selection_sha256, admission_sha256, factor_admission_sha256=None):
    from .sealed_access import _sha
    if not _sha(selection_sha256) or not _sha(admission_sha256) or (factor_admission_sha256 is not None and not _sha(factor_admission_sha256)):
        raise ValueError('explicit selection and bank commitments required')
    return dict(matrix_sha256=matrix['matrix_sha256'], protocol_sha256=matrix['protocol_sha256'],
        selection_sha256=selection_sha256, admission_sha256=admission_sha256,
        factor_admission_sha256=factor_admission_sha256, source_fingerprint=matrix['source_fingerprint'],
        evaluator_fingerprint=matrix['evaluator_fingerprint'])


def _exact(rows, key, expected, message):
    actual = [key(r) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(message)


def validate_native(root, cell, profile, plan, bound):
    """Check each native denominator against the original population plan.

    Does not recompute metrics, reinterpret failures, or accept an incomplete
    continuous cost by averaging successful cases. The native result remains
    the scoring authority; the downstream statistical decision is separate.
    """
    root = Path(root)
    native = root / 'native'
    report = json.loads(checked_asset(native, 'RESULT.private.json').read_text())
    prediction = cell['assay'] in PREDICTION_ASSAYS
    filename = 'FORMAL_COMPLETION.json' if prediction else 'COMPLETION.json'
    completion = json.loads(checked_asset(native, filename).read_text())
    if completion.get('status') != 'PASS' or completion.get('test_read') is not True:
        raise ValueError('native execution incomplete or unqualified')
    provenance = report['manifest']['source_provenance'] if prediction else report
    if provenance.get('formal_results') is not True:
        raise ValueError('native result lacks post-run formal admission')
    for key in ('protocol_sha256', 'selection_sha256'):
        if provenance.get(key) != bound[key]:
            raise ValueError('native protocol/selection binding differs')
    admission_key = 'admission_sha256' if cell['assay'] == 'raw_recoverability' else 'bank_admission_sha256'
    if provenance.get(admission_key) != bound['admission_sha256']:
        raise ValueError('native bank admission differs')
    cases = plan['cases']
    expected_meta = {c['case_id']: c for c in cases}
    ids = set(expected_meta)
    if prediction:
        manifest = report['manifest']
        if manifest.get('agent_id') != cell['method_slot'] or manifest.get('assay_id') != 'visual_elastic_coupling/' + cell['assay'] or provenance.get('profile_id') != cell['profile']:
            raise ValueError('native prediction method/assay/profile differs')
        if provenance.get('source_fingerprint') != bound['source_fingerprint'] or provenance.get('postrun_input_verification') != 'PASS':
            raise ValueError('native prediction source or post-run verification differs')
        if cell['assay'] == 'factor_specificity' and provenance.get('factor_admission_sha256') != bound['factor_admission_sha256']:
            raise ValueError('native factor-bank admission differs')
        for name, sha in completion['artifacts'].items():
            if stable_digest(checked_asset(native, name))['sha256'] != sha:
                raise ValueError('native prediction artifact changed')
        expected = {(i + '/' + c, c, 'vec_' + m) for i in ids for c in profile['conditions'] for m in profile['metrics']}
        _exact(report['records'], lambda r: (r['case_id'], r['condition'], r['metric']), expected,
               'prediction case/condition/metric denominator differs')
        if any(not math.isfinite(r['value']) for r in report['records']):
            raise ValueError('nonfinite prediction metric')
        case_ids = {i + '/' + c for i in ids for c in profile['conditions']}
        if set(report['case_commitments']) != case_ids:
            raise ValueError('prediction case commitments incomplete')
        private = json.loads(checked_asset(native, 'CASE_PLAN.private.json').read_text())
        _exact(private, lambda r: (r['base_case_id'], r['condition']) if 'condition' in r else
               (r['base_case_id'], r['case_id'][len(r['base_case_id']) + 1:]),
               {(i, c) for i in ids for c in profile['conditions']}, 'prediction emitted plan differs')
        if completion.get('cases') != len(case_ids) or completion.get('metric_records') != len(expected):
            raise ValueError('prediction completion counts differ')
        rows = private
        pairing_sha256 = digest(report['case_commitments'])
    else:
        if report.get('profile') != profile or (cell['method_slot'] is not None and report.get('method_slot') != cell['method_slot']):
            raise ValueError('native method or exact profile differs')
        if stable_digest(native / 'RESULT.private.json')['sha256'] != completion.get('result_sha256'):
            raise ValueError('native result changed after completion')
        if cell['assay'] == 'formation':
            rows = report['records']
            _exact(rows, lambda r: (r['case_id'], r['decoder']),
                   {(i, c) for i in ids for c in profile['decoders']}, 'formation case/decoder denominator differs')
            if report.get('cases') != len(ids) or completion.get('cases') != len(ids):
                raise ValueError('formation completion counts differ')
            sha = stable_digest(checked_asset(native, 'PREDICTIONS_BEFORE_LABELS.npz'))['sha256']
            if sha != report.get('predictions_committed_before_labels_sha256') or sha != completion.get('predictions_before_labels_sha256'):
                raise ValueError('formation prediction commitment differs')
            if completion.get('source_fingerprint') != bound['source_fingerprint']:
                raise ValueError('formation evaluator source differs')
            pairing = {}
            for r in rows:
                identity = {k: r[k] for k in ('episode_key', 'history_fingerprint', 'target_fingerprint')}
                if r['case_id'] in pairing and pairing[r['case_id']] != identity:
                    raise ValueError('formation decoders scored different evidence/targets')
                pairing[r['case_id']] = identity
            pairing_sha256 = digest(pairing)
        else:
            if report.get('source_fingerprint') != bound['source_fingerprint']:
                raise ValueError('native evaluator source differs')
            rows = report['cases']
            _exact(rows, lambda r: (r['case_id'], r['condition']),
                   {(i, c) for i in ids for c in profile['conditions']}, 'native case/condition denominator differs')
            if completion.get('cases') != len(rows):
                raise ValueError('native completion counts differ')
            if cell['assay'] == 'closed_loop_control':
                if report.get('errors') != [] or report.get('planned_cases') != len(rows):
                    raise ValueError('control execution errors or planned denominator differ')
                for name, value in report['assets'].items():
                    if stable_digest(checked_asset(native, name)) != value:
                        raise ValueError('control trajectory asset changed')
                pairing = {}
                for r in rows:
                    identity = {k: r[k] for k in ('source_query_episode', 'initial_query_sha256', 'initial_state_sha256')}
                    if r['case_id'] in pairing and pairing[r['case_id']] != identity:
                        raise ValueError('control conditions started from different states')
                    pairing[r['case_id']] = identity
                pairing_sha256 = digest(pairing)
            else:
                if report.get('execution_errors') != []:
                    raise ValueError('raw evidence execution errors retained')
                if stable_digest(checked_asset(native, 'ESTIMATES_BEFORE_LABELS.private.jsonl')) != report.get('estimates_commitment'):
                    raise ValueError('raw estimates changed after truth access')
                pairing_sha256 = None  # One raw diagnostic, not a model-family comparison.
    for row in rows:
        base = row.get('base_case_id', row.get('case_id'))
        original = expected_meta[base]
        if any(row.get(k) != original[k] for k in ('block_id', 'system_key', 'replicate')):
            raise ValueError('native independent-block/system assignment differs')
    return dict(native_completion=filename, base_cases=len(ids), outcome_rows=len(rows),
        native_result_sha256=stable_digest(native / 'RESULT.private.json')['sha256'], pairing_sha256=pairing_sha256)


def run_cell(args):
    matrix = load_matrix(args.matrix, args.protocol)
    cell = select_cell(matrix, args.cell)
    if matrix['protocol_status'] != 'FROZEN':
        raise PermissionError('candidate execution matrix cannot open test data')
    from .analysis_contract import validate_registration
    validate_registration(json.loads(Path(args.protocol).read_text()), matrix)
    authority = SealedAuthorization(args.protocol, args.selection,
        protocol_sha256=matrix['protocol_sha256'], selection_sha256=args.selection_sha256)
    factor = [getattr(args, n, None) for n in ('factor_bank', 'factor_admission', 'factor_admission_sha256')]
    if (cell['assay'] == 'factor_specificity' and not all(factor)) or (cell['assay'] != 'factor_specificity' and any(factor)):
        raise ValueError('factor arguments must match the selected assay')
    bound = bindings(matrix, selection_sha256=args.selection_sha256, admission_sha256=args.admission_sha256,
        factor_admission_sha256=factor[2])
    profile = authority.protocol['assays'][cell['assay']]['profiles'][cell['profile']]
    directory = Path(args.root) / cell['cell_id']
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / 'REQUEST.json', dict(cell=cell, bindings=bound))
    try:
        # Metadata is evaluator-owned. This pass reads no observation/label arrays.
        # It binds an independent denominator before invoking the native runner.
        with SealedBank(args.bank, authority, admission_path=args.admission,
                admission_sha256=args.admission_sha256, resolution=cell['resolution'],
                audit_path=directory / 'PLAN_ACCESS.private.jsonl', allow_labels=False) as bank:
            plan = bank.plans[cell['stratum']]['plan']
            write_new(directory / 'EXPECTED_PLAN.private.json', plan)
        native_args = SimpleNamespace(**vars(args))
        native_args.protocol_sha256 = matrix['protocol_sha256']
        native_args.assay, native_args.profile, native_args.slot = cell['assay'], cell['profile'], cell['method_slot']
        native_args.output = directory / 'native'
        module = importlib.import_module('.' + ROUTES[cell['assay']], __package__)
        module.run(native_args)
        authority.revalidate()
        # Recheck matrix/source and the pre-run denominator after native execution.
        load_matrix(args.matrix, args.protocol)
        if json.loads((directory / 'EXPECTED_PLAN.private.json').read_text()) != plan:
            raise ValueError('expected population changed during native execution')
        evidence = validate_native(directory, cell, profile, plan, bound)
        receipt = dict(schema='vec.execution-cell.v1', status='EXECUTED', cell=cell, bindings=bound,
            evidence=evidence, artifacts=inventory(directory), scientific_acceptance=False,
            note='complete bound native outcomes; downstream statistics and scientific acceptance remain required')
    except Exception as exc:
        write_new(directory / RECEIPT, dict(schema='vec.execution-cell.v1', status='FAILED',
            cell=cell, bindings=bound, error=repr(exc), scientific_acceptance=False,
            note='retain this attempt; no overwrite, automatic retry, or successful-case selection'))
        raise
    write_new(directory / RECEIPT, receipt)
    return receipt


def collect(matrix, protocol, root, *, selection_sha256, admission_sha256, factor_admission_sha256):
    """Outputs only, without opening any bank, model or target trajectory."""
    root = Path(root)
    if matrix['protocol_status'] != 'FROZEN':
        raise PermissionError('candidate protocol cannot admit formal execution outcomes')
    expected_dirs = {c['cell_id'] for c in matrix['cells']}
    unexpected = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in expected_dirs) if root.exists() else []
    records = []
    for cell in matrix['cells']:
        directory = root / cell['cell_id']
        row = dict(cell_id=cell['cell_id'], assay=cell['assay'], profile=cell['profile'], method_slot=cell['method_slot'])
        try:
            if not directory.exists():
                row['status'] = 'NOT_RUN'
            elif not (directory / RECEIPT).exists():
                row['status'] = 'INCOMPLETE_OR_RUNNING'
            else:
                if directory.is_symlink():
                    raise ValueError('symlink attempt directory')
                receipt_path = checked_asset(directory, RECEIPT)
                before = stable_digest(receipt_path)
                receipt = json.loads(receipt_path.read_text())
                bound = bindings(matrix, selection_sha256=selection_sha256, admission_sha256=admission_sha256,
                    factor_admission_sha256=factor_admission_sha256 if cell['assay'] == 'factor_specificity' else None)
                if receipt.get('schema') != 'vec.execution-cell.v1' or receipt.get('cell') != cell or receipt.get('bindings') != bound:
                    raise ValueError('cell receipt source/protocol/selection/data identity differs')
                if receipt.get('status') == 'FAILED':
                    row.update(status='FAILED', error=receipt.get('error'))
                else:
                    if receipt.get('status') != 'EXECUTED' or receipt.get('artifacts') != inventory(directory):
                        raise ValueError('cell result files missing, changed or added after commitment')
                    plan = json.loads(checked_asset(directory, 'EXPECTED_PLAN.private.json').read_text())
                    validate_plan(plan)
                    if plan['plan_sha256'] != protocol['populations'][cell['stratum']]['plan_sha256']:
                        raise ValueError('output denominator differs from frozen population')
                    evidence = validate_native(directory, cell, protocol['assays'][cell['assay']]['profiles'][cell['profile']], plan, bound)
                    if evidence != receipt.get('evidence'):
                        raise ValueError('cell execution evidence differs')
                    if receipt['artifacts'] != inventory(directory):
                        raise ValueError('result artifacts changed during collection')
                    row.update(status='EXECUTED', **evidence)
                if stable_digest(receipt_path) != before:
                    raise ValueError('cell receipt changed during collection')
                row['receipt_sha256'] = before['sha256']
        except Exception as exc:
            row.update(status='INVALID', error=repr(exc))
        records.append(row)
    # Matched conditions must use the same evidence/targets across model slots.
    # Control actions may differ; compare only the shared physical starting point.
    paired = {}
    mismatches = []
    for row, cell in zip(records, matrix['cells']):
        if row['status'] != 'EXECUTED' or row['pairing_sha256'] is None:
            continue
        profile = protocol['assays'][cell['assay']]['profiles'][cell['profile']]
        if cell['assay'] == 'closed_loop_control':
            comparison = [cell['assay'], profile['stratum'], profile['contract']]
        elif cell['assay'] == 'formation':
            comparison = [cell['assay'], {k: v for k, v in profile.items() if k not in ('feature_condition', 'readout_roles', 'compute_tier')}]
        else:
            comparison = [cell['assay'], cell['profile']]
        key = digest(comparison)
        group = paired.setdefault(key, [])
        group.append(row)
    for key, group in paired.items():
        if len({r['pairing_sha256'] for r in group}) != 1:
            mismatches.append(dict(comparison_sha256=key, cells=[r['cell_id'] for r in group]))
    complete = not unexpected and not mismatches and all(r['status'] == 'EXECUTED' for r in records)
    return dict(schema='vec.execution-index.v1', status='EXECUTED' if complete else 'INCOMPLETE',
        matrix_sha256=matrix['matrix_sha256'], protocol_sha256=matrix['protocol_sha256'],
        selection_sha256=selection_sha256, admission_sha256=admission_sha256,
        cells=records, unexpected_directories=unexpected, pairing_mismatches=mismatches, expected_cells=len(records),
        executed_cells=sum(r['status'] == 'EXECUTED' for r in records),
        statistical_analysis='PENDING_SEPARATE_ANALYSIS', scientific_acceptance=False,
        release_ready=False, full_goal_complete=False,
        scope='saved-output integrity and execution coverage; collection does not re-open test banks or score scientific acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run-cell', 'collect'))
    for name in ('matrix', 'protocol', 'root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('selection-sha256', 'admission-sha256'):
        parser.add_argument('--' + name, required=True)
    for name in ('selection', 'bank', 'admission', 'factor-bank', 'factor-admission', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--factor-admission-sha256')
    parser.add_argument('--cell')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.mode == 'run-cell':
        if not all(getattr(args, k) for k in ('selection', 'bank', 'admission', 'cell')):
            parser.error('run-cell requires selection, bank, admission and cell')
        result = run_cell(args)
        print(json.dumps(dict(status=result['status'], cell=result['cell']['cell_id'])))
    else:
        if args.output is None:
            parser.error('collect requires a new output path')
        matrix = load_matrix(args.matrix, args.protocol)
        result = collect(matrix, json.loads(args.protocol.read_text()), args.root,
            selection_sha256=args.selection_sha256, admission_sha256=args.admission_sha256,
            factor_admission_sha256=args.factor_admission_sha256)
        write_new(args.output, result)
        print(json.dumps(dict(status=result['status'], expected_cells=result['expected_cells'], executed_cells=result['executed_cells'])))


if __name__ == '__main__':
    main()
