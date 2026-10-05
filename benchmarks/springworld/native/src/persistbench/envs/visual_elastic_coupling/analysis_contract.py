"""Metadata-only, preregistered scalar estimands and conclusion rules."""
import argparse
import json
import math
from pathlib import Path
from collections import Counter
from .evaluation_blocks import digest
from .sdk_bridge import ASSAYS as PREDICTION_ASSAYS, METRICS as PREDICTION_METRICS
from .formation import LABELS
from .raw_evidence import PARAMETERS

SCHEMA = 'vec.statistical-analysis.v1'
RAW_PREDICTION = ('position_mse_m2', 'velocity_mse_m2_s2', 'center_position_mse_m2',
    'relative_position_mse_m2', 'center_velocity_mse_m2_s2', 'relative_velocity_mse_m2_s2', 'joint_train_standardized_mse')
RAW_PARAMETER = ('parameter_log_squared_error', 'parameter_relative_error', 'parameter_covered80')
CONTROL_METRICS = ('sustained_success', 'integrated_center_error_m2_s', 'effort_n2_s')


def finite_number(x):
    return type(x) in (float, int) and math.isfinite(x)


def term_cells(matrix, assay, term):
    cells = [c for c in matrix['cells'] if c['assay'] == assay and c['profile'] == term['profile']
             and c['method_family'] == term['method_family']]
    if not cells:
        raise ValueError('statistical term has no registered method/profile cells')
    return cells


def task_signature(assay, profile):
    """Only compare the same physical target and authorized observation task."""
    if assay in PREDICTION_ASSAYS:
        fields = ('stratum', 'query_kind', 'query_budget', 'horizon', 'resolution')
    elif assay == 'formation':
        fields = ('stratum', 'frames', 'resolution', 'feature_seed', 'probe_updates', 'probe_seed', 'decoders')
    elif assay == 'closed_loop_control':
        fields = ('stratum', 'contract')
    else:
        fields = ('stratum', 'resolution', 'mode', 'query_kind', 'query_budget', 'horizon', 'configuration')
    return digest({k: profile[k] for k in fields})


def validate_registration(protocol, matrix):
    stats = protocol.get('statistics', {})
    analysis = stats.get('analysis', {})
    if set(analysis) != {'schema', 'comparisons', 'bootstrap_draws', 'bootstrap_seed', 'min_blocks'} or analysis['schema'] != SCHEMA:
        raise ValueError('explicit frozen statistical analysis registration required')
    if stats.get('independent_unit') != 'registered_plan_blocks' or stats.get('seed_policy') != 'fixed_artifacts_within_blocks':
        raise ValueError('statistical unit or seed policy differs')
    if not finite_number(stats.get('alpha')) or not 0 < stats['alpha'] < 1:
        raise ValueError('invalid preregistered alpha')
    if type(analysis['bootstrap_draws']) is not int or analysis['bootstrap_draws'] < 100 or type(analysis['bootstrap_seed']) is not int or analysis['bootstrap_seed'] < 0:
        raise ValueError('invalid sensitivity bootstrap budget')
    if type(analysis['min_blocks']) is not int or analysis['min_blocks'] < 16:
        raise ValueError('at least 16 independent blocks required for inferential intervals')
    comparisons = analysis['comparisons']
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError('empty statistical comparison plan')
    ids, families = set(), Counter()
    for c in comparisons:
        fields = {'comparison_id', 'family', 'assay', 'metric', 'target', 'decoder', 'horizon_s',
                  'terms', 'conclusion', 'precision_absolute_halfwidth', 'interval_method',
                  'missing_policy', 'claim_scope'}
        if set(c) != fields or not isinstance(c['comparison_id'], str) or not c['comparison_id'] or c['comparison_id'] in ids:
            raise ValueError('incomplete or duplicate statistical comparison')
        ids.add(c['comparison_id']); families[c['family']] += 1
        if c['claim_scope'] != 'population_mean' or c['missing_policy'] != 'retain_and_unqualify':
            raise ValueError('mean analysis cannot certify rankings or discard missing cases')
        if not finite_number(c['precision_absolute_halfwidth']) or c['precision_absolute_halfwidth'] <= 0:
            raise ValueError('positive absolute precision target must be preregistered')
        if c['interval_method'] not in ('cluster_t', 'bounded_hoeffding'):
            raise ValueError('unregistered interval method')
        conclusion = c['conclusion']
        if set(conclusion) != {'kind', 'threshold', 'margin'} or conclusion['kind'] not in ('estimate', 'greater_than', 'less_than', 'equivalence'):
            raise ValueError('unregistered conclusion type')
        if conclusion['kind'] == 'estimate':
            if conclusion['threshold'] is not None or conclusion['margin'] is not None:
                raise ValueError('descriptive mean estimate has no decision threshold')
        elif conclusion['kind'] == 'equivalence':
            if not finite_number(conclusion['margin']) or conclusion['margin'] <= 0 or not finite_number(conclusion['threshold']):
                raise ValueError('equivalence center and positive margin required before test')
        elif not finite_number(conclusion['threshold']) or conclusion['margin'] is not None:
            raise ValueError('directional claim requires an explicit threshold')
        assay, metric = c['assay'], c['metric']
        if assay not in protocol['assays']:
            raise ValueError('comparison assay not registered')
        if assay in PREDICTION_ASSAYS:
            if metric not in PREDICTION_METRICS or any(c[k] is not None for k in ('target', 'decoder', 'horizon_s')):
                raise ValueError('unregistered prediction scalar endpoint')
        elif assay == 'formation':
            if metric not in ('squared_error', 'train_standardized_squared_error') or c['target'] not in LABELS or c['decoder'] not in ('ridge', 'mlp') or c['horizon_s'] is not None:
                raise ValueError('formation requires a named label, decoder and squared-error endpoint')
        elif assay == 'closed_loop_control':
            if metric not in CONTROL_METRICS or c['horizon_s'] not in (8., 12., 16.) or c['target'] is not None or c['decoder'] is not None:
                raise ValueError('unregistered control scalar endpoint')
        else:
            if metric not in RAW_PARAMETER + RAW_PREDICTION + ('missing_parameter', 'method_or_support_failure', 'query_tracking_rmse_m'):
                raise ValueError('unregistered raw evidence scalar endpoint')
            if (c['target'] not in PARAMETERS if metric in RAW_PARAMETER else c['target'] is not None) or c['decoder'] is not None or c['horizon_s'] is not None:
                raise ValueError('raw metric parameter/decoder/horizon differs')
        bounded = metric in ('sustained_success', 'parameter_covered80', 'missing_parameter', 'method_or_support_failure')
        if c['interval_method'] == 'bounded_hoeffding' and not bounded:
            raise ValueError('distribution-free bounded interval requires a declared [0,1] scalar')
        terms = c['terms']; seen = set(); signatures = set()
        if not isinstance(terms, list) or not terms:
            raise ValueError('a mean or linear contrast requires explicit terms')
        for term in terms:
            if set(term) != {'profile', 'method_family', 'condition', 'coefficient'} or not finite_number(term['coefficient']) or term['coefficient'] == 0:
                raise ValueError('invalid linear contrast term')
            identity = (term['profile'], term['method_family'], term['condition'])
            if identity in seen:
                raise ValueError('duplicate statistical term; combine coefficients before freeze')
            seen.add(identity)
            cells = term_cells(matrix, assay, term)
            profile = protocol['assays'][assay]['profiles'][term['profile']]
            signatures.add(task_signature(assay, profile))
            allowed = [profile['feature_condition']] if assay == 'formation' else profile['conditions']
            if assay == 'raw_recoverability':
                allowed = allowed + [name + ':' + layer for name in profile['conditions'] for layer in profile['reference_layers']]
                if ':' in term['condition'] and metric not in RAW_PREDICTION:
                    raise ValueError('privileged reference supports prediction errors only')
                if profile['mode'] == 'donor_only' and metric in RAW_PREDICTION + ('query_tracking_rmse_m',):
                    raise ValueError('donor-only profile has no prediction/query tracking endpoint')
            if term['condition'] not in allowed:
                raise ValueError('statistical condition not registered in source profile')
            seeds = [x['model_seed'] for x in cells]
            if len(seeds) != len(set(seeds)):
                raise ValueError('term has duplicate fixed model seed')
        if len(signatures) != 1:
            raise ValueError('comparison mixes physical populations, query support or target budgets')
        coefficient_sum = sum(t['coefficient'] for t in terms)
        if not (math.isclose(coefficient_sum, 0, abs_tol=1e-12) or (len(terms) == 1 and terms[0]['coefficient'] == 1)):
            raise ValueError('register a single mean or a zero-sum linear contrast')
    if dict(families) != stats.get('comparison_families'):
        raise ValueError('registered family sizes must exactly equal all declared comparisons')
    if {c['assay'] for c in comparisons} != set(protocol['assays']):
        raise ValueError('comprehensive statistical registration omits an assay')
    return analysis


def build_analysis_plan(protocol, matrix):
    """A reviewable, data-free plan; candidate status grants no test access."""
    analysis = validate_registration(protocol, matrix)
    resolved = []
    for comparison in analysis['comparisons']:
        terms = []
        for term in comparison['terms']:
            cells = term_cells(matrix, comparison['assay'], term)
            terms.append(dict(specification=term, cells=[c['cell_id'] for c in cells],
                              fixed_model_seeds=[c['model_seed'] for c in cells]))
        resolved.append(dict(specification=comparison, terms=terms,
                             simultaneous_family_size=protocol['statistics']['comparison_families'][comparison['family']]))
    result = dict(schema='vec.statistical-plan.v1', protocol_sha256=matrix['protocol_sha256'],
        matrix_sha256=matrix['matrix_sha256'], source_fingerprint=matrix['source_fingerprint'],
        status=matrix['protocol_status'], analysis_registration_sha256=digest(analysis), comparisons=resolved,
        alpha=protocol['statistics']['alpha'], test_data_opened=False, scientific_acceptance=False)
    result['analysis_plan_sha256'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'matrix', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from .execution_matrix import load_matrix
    value = build_analysis_plan(json.loads(args.protocol.read_text()), load_matrix(args.matrix, args.protocol))
    with args.output.open('x') as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), comparisons=len(value['comparisons']), test_data_opened=False)))


if __name__ == '__main__':
    main()
