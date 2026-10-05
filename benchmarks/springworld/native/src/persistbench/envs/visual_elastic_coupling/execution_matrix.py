"""Complete assay catalog and metadata-only execution planning.

This module never opens a trajectory or grants test permission. The catalog
describes native runners; its three non-SDK routes are not generic AgentOutput
plugins. A complete execution matrix is not a scientific acceptance decision.
"""
import argparse
import json
from pathlib import Path
from .evaluation_blocks import digest
from .dataset_snapshot import stable_digest
from .sdk_bridge import ASSAYS as PREDICTION_ASSAYS, METRICS as PREDICTION_METRICS
from .sdk_registry import build_registry, evaluator_fingerprint
from .sealed_access import ASSAYS, PROTOCOL_SCHEMA
from .training_protocol import source_fingerprint

SCHEMA = 'vec.execution-matrix.v1'
FAMILIES = ('gru', 'transformer', 'transition_deepsets', 'causal_tcn')
ROUTES = {name: 'sealed_evaluate' for name in PREDICTION_ASSAYS}
ROUTES.update(raw_recoverability='sealed_recoverability', formation='sealed_formation',
              closed_loop_control='sealed_control')


def build_full_registry():
    """Preserve the legacy five-assay builder and extend a fresh copy."""
    from .sealed_recoverability import DONOR_CONDITIONS, QUERY_CONDITIONS, METRICS as raw_metrics
    from .sealed_formation import METRICS as formation_metrics
    from .sealed_control import MAIN_CONDITIONS, REFERENCE_CONDITIONS, METRICS as control_metrics
    result = build_registry()
    extra = [
        ('raw_recoverability', 'representation', 'offline', ['recoverability'],
         DONOR_CONDITIONS + QUERY_CONDITIONS,
         [('parameter_log_squared_error', 'lower', 'dimensionless')], raw_metrics),
        ('formation', 'representation', 'offline', ['formation'],
         ('trained', 'random_initialization'),
         [('train_standardized_squared_error', 'lower', 'dimensionless')], formation_metrics),
        ('closed_loop_control', 'control', 'online', ['learner_utility'],
         MAIN_CONDITIONS + REFERENCE_CONDITIONS,
         [('sustained_success', 'higher', 'fraction'), ('integrated_center_error_m2_s', 'lower', 'm^2 s'),
          ('effort_n2_s', 'lower', 'N^2 s')], control_metrics),
    ]
    result['native_routes'] = {name: dict(module='persistbench.envs.visual_elastic_coupling.' + route,
        entry='run', output_contract='native result and completion; use sealed_suite',
        generic_sdk=name in PREDICTION_ASSAYS) for name, route in ROUTES.items()}
    for name, task, interaction, capabilities, conditions, metrics, native_metrics in extra:
        keys = []
        for metric, direction, unit in metrics:
            key = 'vec_' + name + '_' + metric
            keys.append(key)
            result['metric_catalog'][key] = dict(implementation=ROUTES[name] + ':' + metric,
                direction=direction, unit=unit, version='1.1', scope='suite', primary=True,
                per_system_aggregation='native scorer; no silent missing-case reduction',
                suite_aggregation='registered plan blocks', uncertainty='registered downstream block analysis')
        result['assays'].append(dict(assay_id='visual_elastic_coupling/' + name,
            environment_id='visual_elastic_coupling', protocol_version='1.1', task_type=task,
            interaction_mode=interaction, capabilities=capabilities,
            controls=[dict(control_id=c, kind='condition') for c in conditions], metrics=keys,
            group_split_keys=['physical_system'], release_state='incubating', leaderboard=False,
            compute_tiers=['standard']))
        result['native_routes'][name].update(native_metrics=list(native_metrics),
            metric_policy='Catalog scalars are summaries only. Native scorer retains failures, missing support, '
                          'vector diagnostics and reference layers; generic SDK fail_run policies do not replace it.')
    result['environments'][0]['status_note'] = (
        'All eight implemented assay routes catalogued. Full execution, statistical analysis, '
        'scientific acceptance and redistribution audit remain separate requirements.')
    result['execution_scope'] = dict(assays=list(ASSAYS), full_goal_complete=False,
        scientific_acceptance='not implied by catalog or execution coverage')
    return result


def validate_prediction_profile(profile, assay):
    fields = {'stratum', 'query_kind', 'query_budget', 'horizon', 'resolution', 'conditions',
              'metrics', 'compute_tier', 'evaluation_seed', 'case_limit'}
    if set(profile) != fields or profile['conditions'] != list(PREDICTION_ASSAYS[assay]) or profile['metrics'] != list(PREDICTION_METRICS):
        raise ValueError('prediction profile fields, conditions or metrics differ')
    kind, budget, horizon = profile['query_kind'], profile['query_budget'], profile['horizon']
    if kind not in ('cold', 'moving') or budget not in ((0, 1) if kind == 'cold' else (0, 1, 3, 7, 15, 31, 63, 95)) or horizon not in (1, 4, 16, 32):
        raise ValueError('unimplemented prediction query profile')
    if assay == 'history_composition' and (kind, budget) != ('cold', 0):
        raise ValueError('composition primary query differs')
    if assay == 'delayed_prediction' and (kind, budget, horizon) != ('cold', 0, 16):
        raise ValueError('delayed primary query differs')
    if assay == 'factor_specificity' and ((kind, budget) not in (('cold', 0), ('moving', 95)) or horizon not in (16, 32)):
        raise ValueError('factor primary query differs')
    if profile['resolution'] not in (64, 128) or type(profile['evaluation_seed']) is not int or profile['evaluation_seed'] < 0:
        raise ValueError('prediction resolution or seed differs')
    if profile['compute_tier'] not in ('lite', 'standard') or profile['case_limit'] is not None:
        raise ValueError('complete matrix requires full planned cases and a declared compute tier')


def build_matrix(protocol, *, protocol_sha256):
    """Inspect candidate/frozen protocol metadata, with no implicit applicability.

    Each assay has profiles and profile_methods with identical profile IDs.
    Raw evidence uses [] because it is one fixed diagnostic, not a sixth learner.
    Formation random initialization applies only to learned slots. Control may
    split explicit and learned profiles but all slots share the physical task.
    """
    from .sealed_access import _sha
    from .sealed_formation import validate_profile as formation_profile
    from .sealed_control import validate_profile as control_profile, MAIN_CONDITIONS, REFERENCE_CONDITIONS
    from .sealed_recoverability import validate_profile as raw_profile
    if protocol.get('schema') != PROTOCOL_SCHEMA or protocol.get('status') not in ('CANDIDATE', 'FROZEN') or not _sha(protocol_sha256):
        raise ValueError('bound candidate or frozen protocol required')
    if protocol.get('source_fingerprint') != source_fingerprint():
        raise ValueError('matrix source differs from protocol')
    if set(protocol.get('assays', {})) != set(ASSAYS):
        raise ValueError('exact comprehensive eight-assay registration required')
    entries = protocol.get('method_slots', [])
    slots = {s['slot_id']: s for s in entries}
    if len(slots) != len(entries) or any(not isinstance(k, str) or not k for k in slots):
        raise ValueError('missing or duplicate method slot identity')
    learned = {k for k, s in slots.items() if s.get('kind') == 'learned'}
    explicit = {k for k, s in slots.items() if s.get('kind') == 'explicit'}
    if len(explicit) != 1 or learned | explicit != set(slots):
        raise ValueError('one explicit and four learned reference families required')
    seeds = {}
    for family in FAMILIES:
        values = [slots[k].get('seed') for k in learned if slots[k].get('family') == family]
        if not values or any(type(s) is not int for s in values) or len(set(values)) != len(values):
            raise ValueError('missing or duplicated learned family/seed slots')
        seeds[family] = sorted(values)
    if any(slots[k].get('family') not in FAMILIES for k in learned) or len({tuple(v) for v in seeds.values()}) != 1:
        raise ValueError('four learned families require identical registered seed sets')
    cells, control_groups, formation_roles, formation_pairs = [], {}, {}, {}
    formation_conditions = set()
    for assay in ASSAYS:
        spec = protocol['assays'][assay]
        profiles, mapping = spec.get('profiles'), spec.get('profile_methods')
        if not isinstance(profiles, dict) or not profiles or not isinstance(mapping, dict) or set(mapping) != set(profiles):
            raise ValueError('explicit profile_methods required for every profile: ' + assay)
        for profile_id in sorted(profiles):
            profile, selected = profiles[profile_id], mapping[profile_id]
            if not isinstance(profile_id, str) or not profile_id or not isinstance(selected, list) or len(set(selected)) != len(selected) or not set(selected) <= set(slots):
                raise ValueError('invalid profile identity or applicable method slots')
            if profile.get('stratum') not in protocol.get('populations', {}):
                raise ValueError('profile population not registered')
            if assay in PREDICTION_ASSAYS:
                validate_prediction_profile(profile, assay)
                if set(selected) != set(slots):
                    raise ValueError('prediction profile omits registered reference slots')
            elif assay == 'raw_recoverability':
                raw_profile(profile)
                if selected:
                    raise ValueError('raw evidence diagnostic cannot be repeated as learned-method cells')
            elif assay == 'formation':
                formation_profile(profile)
                formation_conditions.add(profile['feature_condition'])
                pair = digest({k: v for k, v in profile.items() if k not in ('feature_condition', 'readout_roles', 'reuse_roles', 'neural_source_checkpoint_role', 'compute_tier')})
                conditions = formation_pairs.setdefault(pair, set())
                if profile['feature_condition'] in conditions:
                    raise ValueError('duplicate formation condition at the same observation/probe budget')
                conditions.add(profile['feature_condition'])
                required = set(slots) if profile['feature_condition'] == 'trained' else learned
                if set(selected) != required:
                    raise ValueError('formation profile omits applicable slots or includes explicit random initialization')
                for key in selected:
                    if not set(profile['readout_roles'].values()) <= set(slots[key].get('required_roles', [])):
                        raise ValueError('formation readout roles not selected in method slot')
                    if not set(profile.get('reuse_roles',{}).values()) <= set(slots[key].get('required_roles', [])):
                        raise ValueError('formation reuse evidence not selected in method slot')
                    if slots[key]['kind']=='learned' and 'neural_source_checkpoint_role' in profile and profile['neural_source_checkpoint_role'] not in slots[key]['required_roles']:
                        raise ValueError('neural formation source checkpoint not selected')
                    for role, name in profile['readout_roles'].items():
                        fields = ('frames', 'resolution', 'feature_condition', 'feature_seed')
                        if role not in ('features', 'extraction'):
                            fields += ('probe_updates', 'probe_seed', 'decoders')
                        signature = digest({k: profile[k] for k in fields})
                        identity = (key, name)
                        if identity in formation_roles and formation_roles[identity] != signature:
                            raise ValueError('incompatible formation profiles reuse the same selected artifact role')
                        formation_roles[identity] = signature
            else:
                if not selected:
                    raise ValueError('empty control method assignment')
                for key in selected:
                    control_profile(profile, kind=slots[key]['kind'])
                    if slots[key]['kind'] == 'learned':
                        formation = protocol['assays']['formation']['profiles'].get(profile['formation_profile'])
                        if formation is None or formation['frames'] != 96 or formation['feature_condition'] != 'trained':
                            raise ValueError('control requires the trained 96-frame formation readout')
                        if key not in protocol['assays']['formation']['profile_methods'][profile['formation_profile']] or profile['prior_role'] not in slots[key].get('required_roles', []):
                            raise ValueError('control readout or calibration is not selected')
                    group = control_groups.setdefault((profile['stratum'], digest(profile['contract'])), {})
                    for condition in profile['conditions']:
                        pair = (key, condition)
                        if pair in group:
                            raise ValueError('control repeats a method/condition within the same physical task')
                        group[pair] = profile_id
            for slot in sorted(selected) if selected else [None]:
                cell = dict(assay=assay, profile=profile_id, method_slot=slot,
                    profile_sha256=digest(profile), runner=ROUTES[assay],
                    stratum=profile['stratum'], resolution=profile.get('resolution', 128),
                    method_family=None if slot is None else slots[slot].get('family', 'explicit'),
                    model_seed=None if slot is None else slots[slot].get('seed'))
                cell['cell_id'] = digest([protocol_sha256, cell])
                cells.append(cell)
    if formation_conditions != {'trained', 'random_initialization'}:
        raise ValueError('formation needs trained and random-initialization controls')
    if any(c != {'trained', 'random_initialization'} for c in formation_pairs.values()):
        raise ValueError('trained/random formation controls must match observation and probe budgets')
    for group in control_groups.values():
        required = {(s, c) for s in slots for c in MAIN_CONDITIONS} | {(s, c) for s in explicit for c in REFERENCE_CONDITIONS}
        if set(group) != required:
            raise ValueError('control task omits main reference methods or evaluator-only reference conditions')
    result = dict(schema=SCHEMA, protocol_sha256=protocol_sha256,
        protocol_status=protocol['status'], source_fingerprint=source_fingerprint(),
        evaluator_fingerprint=evaluator_fingerprint(), cells=cells, cell_count=len(cells),
        model_seeds=seeds, assay_counts={a: sum(c['assay'] == a for c in cells) for a in ASSAYS},
        test_read=False, scientific_acceptance=False,
        scope='execution coverage only; profile adequacy, frozen admission and statistical analysis remain required')
    result['matrix_sha256'] = digest(result)
    return result


def load_matrix(path, protocol_path):
    actual = json.loads(Path(path).read_text())
    expected = build_matrix(json.loads(Path(protocol_path).read_text()), protocol_sha256=stable_digest(protocol_path)['sha256'])
    if actual != expected:
        raise ValueError('execution matrix is stale or differs from exact protocol/source')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build_full_registry() if args.protocol is None else build_matrix(
        json.loads(args.protocol.read_text()), protocol_sha256=stable_digest(args.protocol)['sha256'])
    with args.output.open('x') as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), schema=value['schema'])))


if __name__ == '__main__':
    main()
