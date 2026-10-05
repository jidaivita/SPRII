"""Native cache API integration with synthetic arrays and the real 600/100 shape."""
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
import torch
from sprii_next.io import write,read,sha
from sprii_next.protocol import default_protocol


def test_native_cache_and_complete_endpoint():
    root=os.environ.get('SPRII_SPRING_NATIVE')
    if root is None:pytest.skip('optional Spring native source snapshot not bound')
    root=Path(root);sys.path[:0]=[str(root),str(root/'src'),str(root/'a_src'),str(root/'extension')]
    from persistbench.envs.visual_elastic_coupling.a_head_data import AHeadCasePlan
    from persistbench.envs.visual_elastic_coupling import a_head_features as af
    from persistbench.envs.visual_elastic_coupling.a_head_targets import extract_targets,_save_array
    from sprii_next.providers import SpringCache
    from sprii_next.engine import run_job
    with tempfile.TemporaryDirectory(prefix='sprii-spring-') as tmp:
        out=Path(tmp);episodes=[];counter=0
        for split,stratum,count in [('train','train',144),('validation','continuous_new_systems',64),
                                    ('validation','heldout_factorial_combinations',36),('validation','other_development',78)]:
            for i in range(count):
                sid=f'{split}:{stratum}:{i}';counter+=1
                for kind,number in (('forced',3),('cold',6),('moving',3)):
                    for rep in range(number):
                        episodes.append(dict(episode_key=f'{sid}:{kind}:{rep}',system_key=sid,split=split,stratum=stratum,
                            theta=[1+counter*.003,.5+counter*.002,2+counter*.01],kind=kind,replicate=rep,
                            requested_frames=128,raw_frames=128,anchor=None if kind=='forced' else 16))
        manifest={'episodes':episodes};write(out/'manifest.json',manifest)
        plan=AHeadCasePlan(manifest,seed=0,history_frames=96);donors=af._donors(plan);n=len(plan.base);d=len(donors)
        rng=np.random.default_rng(42)
        arrays=dict(query_embedding=rng.normal(size=(n,2,128)).astype(np.float32),future_actions=np.zeros((n,16,2),np.float32),
            query_support=np.ones((n,5),bool),donor_slot=np.zeros((d,128),np.float32),donor_support=np.ones(d,bool))
        arrays['donor_slot'][:,:64]=rng.normal(size=(d,64))
        features_dir=out/'features';features_dir.mkdir()
        files={name:_save_array(features_dir,name,value) for name,value in arrays.items()}
        snapshot='1'*64;model_hash='2'*64
        record=dict(schema=af.SCHEMA,status='COMPLETE',**af._identity(plan,model_hash),bank_snapshot_sha256=snapshot,
            variant='B3',native_context_dim=64,normalization_profile='train_history_running_statistics_pre_step_v1',
            query_episodes=[b['query_episode'] for b in plan.base],donor_episodes=donors,query_budgets=[0,1],horizons=[1,2,4,8,16],
            files=files,parsed_label_arrays=0,test_read=False)
        write(features_dir/'FEATURES.json',record)
        cache=af.AHeadFeatureCache(features_dir,plan,receipt_sha256=sha(features_dir/'FEATURES.json'),
            model_state_sha256=model_hash,bank_snapshot_sha256=snapshot)
        class SyntheticAccess:
            snapshot_sha256=snapshot
            def __init__(self):self.plan=plan;self.audit=[];self.snapshot={'content_sha256':'3'*64}
            def verify_all(self,workers=8):return {'status':'SYNTHETIC_FIXTURE_ONLY'}
            def target_row(self,i,purpose):
                self.audit.append(dict(kind='private_label',split=plan.base[i]['split'],purpose=purpose))
                return np.random.default_rng(i).normal(size=(5,8)),np.ones(5,bool)
        extract_targets(SyntheticAccess(),cache,out/'targets',workers=1)
        (out/'source.pt').write_bytes(b'synthetic fixture; not a trained source')
        write(out/'completion.json',{'fixture':True})
        descriptor=dict(environment='springworld',method='Both',source_seed=0,synthetic_fixture=True,native_files={},native_paths=[str(root),str(root/'src')],
            model_state_sha256=model_hash,source_completion=str(out/'completion.json'),source_completion_sha256=sha(out/'completion.json'))
        for k,p in dict(manifest=out/'manifest.json',features_receipt=features_dir/'FEATURES.json',
                        targets_receipt=out/'targets/SUPERVISION.json',checkpoint=out/'source.pt').items():
            descriptor[k]=str(p);descriptor[k+'_sha256']=sha(p)
        provider=SpringCache(descriptor)
        assert provider.donors('train')[0].shape==(432,64)
        torch.set_num_threads(1)
        job=dict(environment='springworld',stage='development',method='Both',source_seed=0,reader_seed=0,arm='decode')
        run_job(provider,default_protocol([descriptor]),job,out/'smoke',device='cpu',smoke_steps=1)
        result=read(out/'smoke/evaluation/RESULT.json')
        assert result['all_cases']==16020 and result['primary_cases']==600 and len(result['primary_system_mse'])==100
        assert read(out/'smoke/COMPLETE.json')['smoke'] is True
