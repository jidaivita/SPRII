"""Frozen mean/linear-contrast analysis of completed native result files.

No training, bank opening, checkpoint selection or endpoint selection occurs.
Independent blocks, not windows or training seeds, define uncertainty. Missing
outcomes retain the planned denominator and unqualify that comparison.
"""
import argparse
import json
import math
from pathlib import Path
from collections import Counter
from functools import lru_cache
import numpy as np
from .analysis_contract import validate_registration, term_cells, task_signature, RAW_PARAMETER, RAW_PREDICTION
from .execution_matrix import load_matrix, PREDICTION_ASSAYS
from .sealed_suite import collect, write_new
from .dataset_snapshot import checked_asset, stable_digest
from .evaluation_blocks import digest, validate_plan
from .formation import LABELS
from .sealed_control import CONTRACT


def scalar_rows(report, private_plan, comparison, term):
    """Convert a named endpoint without re-scoring or changing its failures."""
    assay, metric = comparison['assay'], comparison['metric']
    condition = term['condition']; rows = []
    if assay in PREDICTION_ASSAYS:
        plans = {r['case_id']: r for r in private_plan}
        for r in report['records']:
            if r['condition'] != condition or r['metric'] != 'vec_' + metric:
                continue
            p = plans[r['case_id']]
            rows.append(dict(case_id=p['base_case_id'], block_id=p['block_id'], system_key=p['system_key'], replicate=p['replicate'],
                value=r['value'], missing_reason=None, outcome_status='EXECUTED', query_fingerprint=digest(p['query_fingerprint']),
                target_fingerprint=report['case_commitments'][r['case_id']]))
        return rows
    if assay == 'formation':
        if report['labels'] != list(LABELS):
            raise ValueError('formation label ordering differs')
        index = LABELS.index(comparison['target'])
        for r in report['records']:
            if r['decoder'] != comparison['decoder']:
                continue
            values = r[metric]
            if len(values) != len(LABELS):
                raise ValueError('formation target vector incomplete')
            rows.append(dict(**{k: r[k] for k in ('case_id', 'block_id', 'system_key', 'replicate')}, value=values[index],
                missing_reason=None, outcome_status='EXECUTED', query_fingerprint=digest(r['history_fingerprint']),
                target_fingerprint=r['target_fingerprint']))
        return rows
    if assay == 'closed_loop_control':
        for r in report['cases']:
            if r['condition'] != condition:
                continue
            selected = [m for m in r['metrics'] if m['horizon_s'] == comparison['horizon_s']]
            if len(selected) != 1:
                raise ValueError('control horizon missing or duplicated')
            m = selected[0]
            missing = None
            if metric == 'sustained_success':
                if type(m['success']) is not bool:
                    raise ValueError('control success must retain the native boolean failure policy')
                value = float(m['success'])
            elif m['status'] != 'EXECUTED':
                value, missing = None, 'incomplete control horizon; continuous cost undefined'
            else:
                value = m[metric]
            rows.append(dict(**{k: r[k] for k in ('case_id', 'block_id', 'system_key', 'replicate')},
                value=value, missing_reason=missing, outcome_status='PHYSICS_FAILURE' if r['failure'] else 'EXECUTED',
                query_fingerprint=r['initial_query_sha256'],
                target_fingerprint=digest([r['initial_state_sha256'], r['theta'], CONTRACT['goal_xy'], CONTRACT])))
        return rows
    base, separator, layer = condition.partition(':')
    for r in report['cases']:
        if r['condition'] != base:
            continue
        value, missing = None, None
        if metric in RAW_PARAMETER:
            names = dict(parameter_log_squared_error='log_squared_error', parameter_relative_error='relative_error', parameter_covered80='covered80')
            if r['parameter_errors'] is not None:
                value = float(r['parameter_errors'][comparison['target']][names[metric]])
            else:
                missing = 'parameter estimate absent'
        elif metric in RAW_PREDICTION:
            errors = (r['references'] or {}).get(layer) if separator else r['prediction_errors']
            if errors is not None:
                value = errors[metric]
            else:
                missing = 'prediction or authorized target unavailable'
        elif metric == 'missing_parameter':
            value = float(r['parameter_errors'] is None)
        elif metric == 'method_or_support_failure':
            value = float(r['outcome_status'] != 'EXECUTED')
        elif metric == 'query_tracking_rmse_m':
            value = r.get('query_tracking_rmse_m')
            if value is None:
                missing = 'query tracking unavailable'
        else:
            raise ValueError('unsupported raw scalar')
        query = r['input_commitment']['query']
        rows.append(dict(**{k: r[k] for k in ('case_id', 'block_id', 'system_key', 'replicate')},
            value=value, missing_reason=missing, outcome_status=r['outcome_status'],
            query_fingerprint=digest(query) if query is not None else digest('raw donor-only measurement task'),
            target_fingerprint=digest(r['theta']) if metric not in RAW_PREDICTION else r.get('target_sha256')))
    return rows


def block_reduce(case_values, plan):
    """Replicates within each system, then systems within each block."""
    by_system = {}
    for c in plan['cases']:
        by_system.setdefault((c['block_id'], c['system_key']), []).append(case_values[c['case_id']])
    by_block = {}
    for (block, _), values in by_system.items():
        by_block.setdefault(block, []).append(float(np.mean(values)))
    if len({len(v) for v in by_block.values()}) != 1:
        raise ValueError('unequal independent-block physical-system counts')
    return {k: float(np.mean(by_block[k])) for k in sorted(by_block)}


def assemble(comparison, plan, sources):
    """Sources are term-indexed lists of exact cell identities and scalar rows."""
    validate_plan(plan)
    if len(sources) != len(comparison['terms']) or any(not group for group in sources):
        raise ValueError('contrast term or fixed-model source missing')
    expected = {c['case_id']: c for c in plan['cases']}
    missing, outcomes, checked, model_means = [], [], [], []
    for ti, group in enumerate(sources):
        term_sources, seeds = [], set()
        for source in group:
            cell = source['cell']; rows = source['rows']; seed = cell['model_seed']
            if seed in seeds:
                raise ValueError('duplicate fixed seed in a statistical term')
            seeds.add(seed)
            by_id = {r['case_id']: r for r in rows}
            if len(by_id) != len(rows) or set(by_id) != set(expected):
                raise ValueError('scalar outcomes omit or duplicate planned cases')
            values = {}
            for identity, row in by_id.items():
                if any(row[k] != expected[identity][k] for k in ('block_id', 'system_key', 'replicate')):
                    raise ValueError('scalar outcomes changed independent-block assignments')
                value = row['value']
                valid = type(value) in (int, float) and math.isfinite(value)
                if not valid or row['missing_reason'] is not None:
                    missing.append(dict(term=ti, cell_id=cell['cell_id'], case_id=identity, reason=row['missing_reason'] or 'nonfinite scalar'))
                elif comparison['interval_method'] == 'bounded_hoeffding' and not 0 <= value <= 1:
                    raise ValueError('bounded endpoint outside its registered [0,1] range')
                values[identity] = float(value) if valid else None
                outcomes.append(dict(term=ti, cell_id=cell['cell_id'], status=row['outcome_status']))
            if any(v is None for v in values.values()) or any(r['missing_reason'] is not None for r in rows):
                model_mean = None
            else:
                model_mean = float(np.mean(list(block_reduce(values, plan).values())))
            model_means.append(dict(term=ti, cell_id=cell['cell_id'], method_family=cell['method_family'], seed=seed, mean=model_mean))
            term_sources.append(by_id)
        checked.append(term_sources)
    # Retain the entire planned denominator even when a target itself is absent.
    result = dict(status='MISSING_OUTCOMES' if missing else 'ASSEMBLED', planned_cases=len(expected),
        physical_systems=len(plan['system_generation_order']), episode_replicates=plan['replicates'],
        independent_unit=plan['independent_unit'], independent_blocks=len({c['block_id'] for c in expected.values()}),
        plan_sha256=plan['plan_sha256'], missing=missing, fixed_model_means=model_means,
        outcome_status_counts=dict(Counter(o['status'] for o in outcomes)),
        seed_policy='average each registered family ensemble per case; seeds are fixed artifacts, not extra independent test units')
    # Even missing comparisons must expose altered available query/target pairs.
    for identity in expected:
        fingerprints = {(r[identity]['query_fingerprint'], r[identity]['target_fingerprint'])
                        for group in checked for r in group if r[identity]['value'] is not None}
        if any(not q or not t for q, t in fingerprints) or len(fingerprints) > 1:
            raise ValueError('paired evidence or target changed across conditions/models')
    if missing:
        result.update(mean=None, block_means=None)
        return result
    values = {}
    for identity in expected:
        values[identity] = float(sum(term['coefficient'] * np.mean([rows[identity]['value'] for rows in group])
            for term, group in zip(comparison['terms'], checked)))
    blocks = block_reduce(values, plan)
    result.update(mean=float(np.mean(list(blocks.values()))), block_means=blocks)
    return result


def infer(assembled, comparison, *, alpha, family_size, min_blocks, bootstrap_draws, bootstrap_seed):
    """Prespecified simultaneous intervals, with no post-result method choice."""
    result = dict(assembled, interval_method=comparison['interval_method'], alpha=alpha,
        registered_family_size=family_size, precision_target=comparison['precision_absolute_halfwidth'],
        decision='UNAVAILABLE', precision_status='UNAVAILABLE')
    if assembled['status'] != 'ASSEMBLED':
        return result
    values = np.asarray(list(assembled['block_means'].values()), float)
    count = len(values); mean = float(values.mean())
    if count < min_blocks:
        result.update(status='INSUFFICIENT_INDEPENDENT_BLOCKS')
        return result
    if comparison['interval_method'] == 'bounded_hoeffding':
        coefficients = [t['coefficient'] for t in comparison['terms']]
        lower, upper = sum(min(c, 0) for c in coefficients), sum(max(c, 0) for c in coefficients)
        half = (upper - lower) * math.sqrt(math.log(2 / alpha) / (2 * count))
        family_half = (upper - lower) * math.sqrt(math.log(2 * family_size / alpha) / (2 * count))
        interval = [max(lower, mean - half), min(upper, mean + half)]
        family = [max(lower, mean - family_half), min(upper, mean + family_half)]
        assumptions = 'independent preallocated blocks with bounded scalar contrasts; no normality assumption; may be conservative'
    else:
        se = float(values.std(ddof=1) / np.sqrt(count))
        result['standard_error'] = se
        if se == 0:
            result.update(status='DEGENERATE_EMPIRICAL_VARIANCE',
                reason='zero sample variance is not proof of zero population uncertainty; no automatic t-interval claim')
            return result
        from scipy.stats import t
        half = float(t.ppf(1 - alpha / 2, count - 1)) * se
        family_half = float(t.ppf(1 - alpha / (2 * family_size), count - 1)) * se
        interval = [mean - half, mean + half]
        family = [mean - family_half, mean + family_half]
        assumptions = 'approximate independent-cluster mean t interval; conditional on fixed training artifacts; no exact finite-sample guarantee'
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = []
    for start in range(0, bootstrap_draws, 256):
        indices = rng.integers(count, size=(min(256, bootstrap_draws - start), count))
        bootstrap.extend(values[indices].mean(1))
    # Precision uses the simultaneous interval, never a more favorable alternative.
    radius = max(mean - family[0], family[1] - mean)
    conclusion = comparison['conclusion']; kind = conclusion['kind']; threshold = conclusion['threshold']
    decision = 'ESTIMATE_ONLY'
    if kind == 'greater_than':
        decision = 'SUPPORTED' if family[0] > threshold else ('CONTRADICTED' if family[1] <= threshold else 'INCONCLUSIVE')
    elif kind == 'less_than':
        decision = 'SUPPORTED' if family[1] < threshold else ('CONTRADICTED' if family[0] >= threshold else 'INCONCLUSIVE')
    elif kind == 'equivalence':
        lo, hi = threshold - conclusion['margin'], threshold + conclusion['margin']
        decision = 'SUPPORTED' if lo < family[0] and family[1] < hi else ('CONTRADICTED' if family[1] <= lo or family[0] >= hi else 'INCONCLUSIVE')
    result.update(status='ANALYZED', interval=interval, family_interval=family, family_radius=radius,
        bootstrap_sensitivity_interval=np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2]).tolist(),
        bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed, interval_assumptions=assumptions,
        decision=decision, conclusion=conclusion,
        precision_status='MET' if radius <= comparison['precision_absolute_halfwidth'] else 'NOT_MET',
        reliability_scope='population means only; system rankings/heterogeneity/personalization require separately registered repeat reliability')
    return result


def run(args):
    matrix = load_matrix(args.matrix, args.protocol)
    if matrix['protocol_status'] != 'FROZEN':
        raise PermissionError('candidate protocol cannot analyze formal outcomes')
    protocol = json.loads(Path(args.protocol).read_text())
    analysis = validate_registration(protocol, matrix)
    common = dict(selection_sha256=args.selection_sha256, admission_sha256=args.admission_sha256,
                  factor_admission_sha256=args.factor_admission_sha256)
    before = collect(matrix, protocol, args.root, **common)
    if before['status'] != 'EXECUTED':
        raise ValueError('complete and bound native execution index required before formal analysis')
    output = Path(args.output)
    if output.resolve().is_relative_to(Path(args.root).resolve()):
        raise ValueError('statistical output must be outside the immutable execution root')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'INPUT_EXECUTION_INDEX.json', before)
    results = []
    @lru_cache(maxsize=8)
    def load_source(cell_id, assay):
        directory = Path(args.root) / cell_id
        report = json.loads(checked_asset(directory, 'native/RESULT.private.json').read_text())
        plan = json.loads(checked_asset(directory, 'EXPECTED_PLAN.private.json').read_text())
        private = json.loads(checked_asset(directory, 'native/CASE_PLAN.private.json').read_text()) if assay in PREDICTION_ASSAYS else None
        return report, plan, private
    try:
        for comparison in analysis['comparisons']:
            sources, plans = [], []
            for term in comparison['terms']:
                group = []
                for cell in term_cells(matrix, comparison['assay'], term):
                    report, plan, private = load_source(cell['cell_id'], comparison['assay']); plans.append(plan)
                    group.append(dict(cell=cell, rows=scalar_rows(report, private, comparison, term)))
                sources.append(group)
            if len({p['plan_sha256'] for p in plans}) != 1:
                raise ValueError('comparison changes planned sampling population')
            assembled = assemble(comparison, plans[0], sources)
            summary = infer(assembled, comparison, alpha=protocol['statistics']['alpha'],
                family_size=protocol['statistics']['comparison_families'][comparison['family']],
                min_blocks=analysis['min_blocks'], bootstrap_draws=analysis['bootstrap_draws'], bootstrap_seed=analysis['bootstrap_seed'])
            results.append(dict(comparison_id=comparison['comparison_id'], specification=comparison, **summary))
        if load_matrix(args.matrix, args.protocol) != matrix:
            raise ValueError('analysis protocol changed')
        after = collect(matrix, protocol, args.root, **common)
        if before != after:
            raise ValueError('native outcomes changed during analysis')
        result = dict(schema='vec.statistical-results.v1', status='EXECUTED',
            matrix_sha256=matrix['matrix_sha256'], protocol_sha256=matrix['protocol_sha256'],
            analysis_registration_sha256=digest(analysis), input_index_sha256=stable_digest(output / 'INPUT_EXECUTION_INDEX.json')['sha256'],
            source_fingerprint=matrix['source_fingerprint'], comparisons=results,
            missing_comparisons=sum(r['status'] == 'MISSING_OUTCOMES' for r in results),
            scientific_acceptance=False, release_ready=False, full_goal_complete=False,
            scope='all preregistered population-mean estimates and linear contrasts; zero/missing/negative outcomes retained; no system-ranking or global identifiability claim')
        write_new(output / 'STATISTICS.private.json', result)
        public = dict(result, comparisons=[{k: v for k, v in r.items() if k not in ('block_means', 'missing')} | {'missing_outcome_count': len(r['missing'])} for r in results])
        write_new(output / 'PUBLIC_STATISTICS.json', public)
        receipt = dict(status='EXECUTED', result_sha256=stable_digest(output / 'STATISTICS.private.json')['sha256'],
            public_sha256=stable_digest(output / 'PUBLIC_STATISTICS.json')['sha256'], comparison_count=len(results),
            scientific_acceptance=False, note='execution of analysis is separate from support of individual claims')
        write_new(output / 'COMPLETION.json', receipt)
        return receipt
    except Exception as exc:
        write_new(output / 'FAILED_PARTIAL.private.json', dict(error=repr(exc), completed_comparisons=results))
        write_new(output / 'COMPLETION.json', dict(status='FAILED', error=repr(exc), scientific_acceptance=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'protocol', 'root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('selection-sha256', 'admission-sha256', 'factor-admission-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); print(json.dumps(run(args)))


if __name__ == '__main__':
    main()
