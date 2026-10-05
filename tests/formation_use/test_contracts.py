import copy
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from sprii_next.decoder import RidgeDecoder,physical_coordinates
from sprii_next.model import Reader
from sprii_next.engine import jobs,context,run_job,job_name
from sprii_next.providers import Batch
from sprii_next.io import development_path,write,read,sha,digest,npz,code_hashes
from sprii_next.protocol import default_protocol
from sprii_next.statistics import assert_pairing,load_rows,verify_gate
from sprii_next.contrastive import symmetric_infonce,summarize_training
from sprii_next.effects import controlled_pairs
from sprii_next.geometry import source_geometry
from final_only.boundary import freeze,evaluator_ticket,ROLES,METHODS


@pytest.fixture
def scratch():
    with tempfile.TemporaryDirectory(prefix='sprii-check-') as d:yield Path(d)


def physics_data(n=100,offset=0):
    rng=np.random.default_rng(42+offset)
    logs=rng.normal(size=(n,3));p=np.zeros((n,64));p[:,:3]=logs
    return p.astype(np.float32),np.exp(logs),np.array([f's{offset+i}' for i in range(n)])


class SyntheticProvider:
    environment='fixture'
    descriptor={'environment':'fixture','method':'Both','source_seed':0}
    identity=digest(descriptor)
    normalization={'target_mean':[0.]*8,'target_scale':[1.]*8}
    def donors(self,split):
        p,t,ids=physics_data(80 if split=='train' else 12,0 if split=='train' else 100)
        return p,t,ids,np.array([f'{s}:donor' for s in ids])
    def batch(self,split,seed):
        p,t,ids,donors=self.donors(split);rng=np.random.default_rng(seed)
        q=rng.normal(size=(len(p),128)).astype(np.float32)
        target=np.column_stack((np.log(t),np.zeros((len(p),5)))).astype(np.float32)
        rows=[dict(system_id=s,query_id=s+':query',donor_id=d,query_episode=s+':q',donor_episode=d,
            donor_system_id=s,horizon=16,mass=float(v[0]),drag=float(v[1]),stiffness=float(v[2]),primary=True)
            for s,d,v in zip(ids,donors,t)]
        return Batch(q,p,t,np.zeros((len(p),16,2),np.float32),np.ones((len(p),16),np.float32),
            np.full(len(p),4,np.int64),target,rows)
    def training_batch(self,seed,step,size):return self.batch('train',seed+step)
    def evaluation_batches(self,size=256):yield self.batch('validation',0)


def test_exact_seed_grids():
    assert len(jobs('springworld'))==36
    assert len(jobs('pokeworld','pilot'))==12
    assert len(jobs('pokeworld','formal'))==24
    assert len(jobs('springworld','baseline'))==18
    assert {j['reader_seed'] for j in jobs('pokeworld','pilot')}=={0}
    assert {j['reader_seed'] for j in jobs('pokeworld','formal')}=={1,2}


def test_decoder_recovers_same_code_and_no_oracle_leak():
    p,t,ids=physics_data();decoder=RidgeDecoder.fit(p,t,ids,split='train',alpha=1e-4)
    assert np.max(np.abs(decoder.predict(p)-decoder.oracle(t)))<1e-4
    batch=SyntheticProvider().batch('validation',0)
    before=context(batch,'decode',decoder).copy();batch.theta*=17
    np.testing.assert_array_equal(before,context(batch,'decode',decoder))
    assert context(batch,'persistent',decoder) is batch.persistent
    with pytest.raises(PermissionError):RidgeDecoder.fit(p,t,ids,split='validation')
    with pytest.raises(ValueError):decoder.score(p,t,ids)
    restored=RidgeDecoder.from_record(json.loads(json.dumps(decoder.record())))
    np.testing.assert_array_equal(restored.predict(p),decoder.predict(p))


def test_common_initialization_and_interface_capacity():
    models={a:Reader(a,2) for a in ('null','persistent','decode','oracle')}
    assert models['decode'].architecture()==models['oracle'].architecture()
    assert models['decode'].architecture()['parameters']-models['persistent'].architecture()['parameters']==256
    for n,p in models['persistent'].named_parameters():
        for model in models.values():torch.testing.assert_close(p,dict(model.named_parameters())[n],rtol=0,atol=0)
    x=torch.randn(4,128);actions=torch.randn(4,16,2);mask=torch.ones(4,16);hi=torch.zeros(4,dtype=torch.long)
    # Oracle and Decode have identical functions before fitting when context is equal.
    t=torch.randn(4,3)
    torch.testing.assert_close(models['decode'](x,t,actions,mask,hi),models['oracle'](x,t,actions,mask,hi),rtol=0,atol=0)


def test_seal_path_rejected_including_symlinks(scratch):
    secret=scratch/'sealed';secret.mkdir();(secret/'payload.json').write_text('{}')
    with pytest.raises(PermissionError):development_path(secret/'payload.json')
    (scratch/'alias').symlink_to(secret,target_is_directory=True)
    with pytest.raises(PermissionError):development_path(scratch/'alias/payload.json')


def test_smoke_full_vectors_and_deterministic_pairing(scratch):
    torch.set_num_threads(1);p=SyntheticProvider();cfg=default_protocol([])
    completions=[]
    for arm in ('null','persistent','decode','oracle'):
        j=dict(environment='springworld',stage='development',method='Both',source_seed=0,reader_seed=0,arm=arm)
        output=scratch/arm;run_job(p,cfg,j,output,device='cpu',smoke_steps=2)
        c=read(output/'COMPLETE.json');completions.append(c)
        shard=read(output/'evaluation/RESULT.json')['shards'][0]
        with np.load(output/'evaluation'/shard['vectors']) as f:
            np.testing.assert_allclose(f['error_per_dimension'],(f['prediction_vector'].astype(float)-f['target_vector'])**2)
            np.testing.assert_array_equal(f['persistent_code'],p.batch('validation',0).persistent)
        with pytest.raises(ValueError,match='smoke'):load_rows(output)
    assert len({c['training_sequence_sha256'] for c in completions})==1


def test_pairing_rejects_altered_donor_and_physics():
    a=SyntheticProvider().batch('validation',0).rows;b=copy.deepcopy(a)
    b[0]['donor_id']='different'
    with pytest.raises(ValueError):assert_pairing(a,b)
    b=copy.deepcopy(a);b[0]['mass']*=2
    with pytest.raises(ValueError):assert_pairing(a,b)


def test_pilot_gate_is_nonstatistical_and_expansion_closed(scratch):
    summary=dict(stage='pilot',positive_slopes=2,artifacts={},protocol_sha256='x')
    write(scratch/'summary.json',summary)
    gate=dict(decision='go',trend_review='two directions and clear binned trend',binned_curve_nonflat=True,p_value_used=False,
              summary=str(scratch/'summary.json'),summary_sha256=sha(scratch/'summary.json'))
    write(scratch/'go.json',gate);verify_gate(scratch/'go.json','x')
    bad=dict(gate,p_value_used=True);write(scratch/'p-gate.json',bad)
    with pytest.raises(PermissionError):verify_gate(scratch/'p-gate.json')
    bad=dict(gate,binned_curve_nonflat=False);write(scratch/'flat.json',bad)
    with pytest.raises(PermissionError):verify_gate(scratch/'flat.json')
    job=jobs('pokeworld','formal')[0];provider=SyntheticProvider()
    provider.environment='pokeworld';provider.descriptor=dict(method='G1',source_seed=0)
    with pytest.raises(PermissionError):run_job(provider,default_protocol([]),job,scratch/'run',device='cpu',smoke_steps=1)


def test_similarity_really_float32_under_autocast():
    p=torch.randn(32,64,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        projected=p@torch.randn(64,64)
        loss,metrics=symmetric_infonce(projected,p,.1)
    assert metrics['similarity_dtype']=='torch.float32'
    loss.backward();assert p.grad.isfinite().all()
    identical=torch.ones(32,64)
    _,metrics=symmetric_infonce(identical,identical,.1)
    assert summarize_training([metrics]*100,10000)['status']=='COMPLETE'
    assert summarize_training([metrics]*100,10000)['outcome_threshold_applied'] is False
    orthogonal=torch.eye(16,64);p=torch.cat((orthogonal,orthogonal))
    _,metrics=symmetric_infonce(p,p,.1)
    assert summarize_training([metrics]*100,10000)['status']=='COMPLETE'
    assert summarize_training([metrics]*100,2)['status']=='SMOKE_ONLY'


def test_effect_pairs_never_relax_other_factors():
    theta=np.array([[1,1,2],[2,1,2],[1,3,4],[5,3,4]],float)
    pairs=controlled_pairs(theta,['a','b','c','d'],0)
    assert len(pairs)==2
    for i,j in pairs:np.testing.assert_array_equal(theta[i,1:],theta[j,1:])
    with pytest.raises(ValueError):controlled_pairs(theta,['a','b','c','d'],2)


def test_geometry_only_reads_source_and_is_invariant_to_reader():
    class GeometryProvider(SyntheticProvider):
        environment='springworld';descriptor={'method':'Structure'}
        def donors(self,split):
            p,t,ids,d=super().donors(split)
            return np.repeat(p,2,axis=0)+np.tile([0,.01],len(p))[:,None],np.repeat(t,2,axis=0),np.repeat(ids,2),np.array([f'{x}-{i}' for x in ids for i in (0,1)])
    result=source_geometry(GeometryProvider())
    assert result['reader_used'] is False and result['D_within']>0
    p=GeometryProvider();p.descriptor={'method':'Native'}
    with pytest.raises(ValueError):source_geometry(p)


def test_freeze_change_rejected_and_ticket_is_one_shot(scratch):
    role_files={}
    for role in ROLES:
        path=scratch/(role+'.json')
        value={'content':'synthetic fixture only'}
        if role=='baseline_qualification':value={'status':'COMPLETE','recipe_id':'fcrl_style_temporal_v1','test_read':False}
        if role=='native_evaluator_contract':value=dict(status='VALIDATED_ON_SYNTHETIC_DATA',native_sealed_authorization_required=True,
            methods=list(METHODS),statistical_unit='physical_system',one_shot_ledger=str(scratch/'ledger'))
        write(path,value);role_files[role]=[str(path)]
    receipt=scratch/'freeze.json';freeze(role_files,receipt);frozen_hash=sha(receipt)
    with evaluator_ticket(receipt,frozen_hash,scratch/'ledger') as value:assert value['status']=='FROZEN'
    with pytest.raises(FileExistsError):
        with evaluator_ticket(receipt,frozen_hash,scratch/'ledger'):pass
    (scratch/'claims.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='frozen role changed'):
        with evaluator_ticket(receipt,frozen_hash,scratch/'ledger'):pass
