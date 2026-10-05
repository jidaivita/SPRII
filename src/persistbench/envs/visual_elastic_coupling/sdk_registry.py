"""Portable registry for implemented visual prediction assays.

This registry remains incubating until formal protocol and test acceptance.
Formation, raw-evidence certificates and closed-loop control retain their own
executed development entry points; registering these prediction assays does
not imply that the complete seven-capability release is finished.
"""
import argparse,hashlib,json
from pathlib import Path
from .sdk_bridge import ASSAYS,METRICS
from .training_protocol import source_fingerprint

CORE_FILES=('__init__.py','contracts.py','evaluator.py','metrics.py','registry.py','results.py')


def evaluator_fingerprint():
    digest=hashlib.sha256(source_fingerprint().encode());root=Path(__file__).parents[2]
    for name in CORE_FILES:digest.update(name.encode()+b'\0'+(root/name).read_bytes())
    return digest.hexdigest()


def build_registry():
    capabilities={'conditional_prediction':['learner_utility','system_specificity'],
        'history_composition':['organization'],'delayed_prediction':['delayed_value'],
        'factor_specificity':['system_specificity'],'predictive_transfer':['ood_transfer']}
    units={'position_mse':'m^2','velocity_mse':'(m/s)^2','center_position_mse':'m^2','relative_position_mse':'m^2','joint_standardized_mse':'dimensionless'}
    metrics={'vec_'+name:dict(implementation='vec_'+name,direction='lower',unit=units[name],version='1.1',output_key='joint_state_delta',
        scope='case',primary=True,per_system_aggregation='mean',suite_aggregation='macro',uncertainty='system_bootstrap') for name in METRICS}
    assays=[]
    for name,conditions in ASSAYS.items():
        assays.append(dict(assay_id='visual_elastic_coupling/'+name,environment_id='visual_elastic_coupling',protocol_version='1.1',
            task_type='prediction',interaction_mode='offline',capabilities=capabilities[name],
            controls=[dict(control_id=condition,kind='baseline' if condition=='null' else ('counterfactual' if condition.startswith(('wrong','factor_')) else 'condition')) for condition in conditions],
            metrics=list(metrics),group_split_keys=['physical_system'],output_spec=dict(output_type='prediction',required_keys=['joint_state_delta'],dtype='float64',shape=[8]),
            release_state='incubating',leaderboard=False,compute_tiers=['lite','standard']))
    return dict(schema='persistbench.registry.v0.2',environments=[dict(environment_id='visual_elastic_coupling',display_name='Visual Elastic Coupling',
        source_papers=['Paper Z / PersistBench','Paper A environment extension'],asset_kind='benchmark_native_visual_simulator',data_modality='simulated',
        source_ref='visual_elastic_coupling_v1_1',admission='core_candidate',fresh_test_required=True,
        status_note='Development integration of the coupled-sled visual upgrade; same physical family, not an independent domain. Formal seven-capability acceptance remains pending.')],
        sources=[dict(source_ref='visual_elastic_coupling_v1_1',source_kind='task_source_snapshot',immutable_ref='sha256:'+evaluator_fingerprint(),
            redistribution='unresolved',note='Executed task-owned code snapshot; packaging/license audit required before public release.')],
        metric_catalog=metrics,assays=assays,failure_evidence=[],compute_tiers=[
            dict(tier='lite',max_wall_seconds=600,max_cpu_cores=4,max_gpus=0,max_memory_gb=16,training_included=False,reference_hardware='development CPU inference'),
            dict(tier='standard',max_wall_seconds=86400,max_cpu_cores=16,max_gpus=1,max_memory_gb=64,training_included=False,reference_hardware='one L20 or CPU inference; training budget registered separately')])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('registry snapshot already exists')
    a.output.write_text(json.dumps(build_registry(),indent=2)+'\n')
    from persistbench.registry import BenchmarkRegistry
    print(json.dumps(BenchmarkRegistry.load(a.output).summary()),flush=True)


if __name__=='__main__':main()
