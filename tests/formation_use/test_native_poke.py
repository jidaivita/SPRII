"""Optional native integration checks, all data/checkpoints synthetic.

Set SPRII_POKE_SOURCE to a read-only native source snapshot. Never points at a
data bank. No existing train/validation/test trajectories are opened here.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import asdict
import numpy as np
import pytest
from sprii_next.poke_simulator import replay,sensitivities
from sprii_next.io import write,sha,read
from sprii_next.providers import PokeCache


@pytest.fixture(scope='module')
def native():
    root=os.environ.get('SPRII_POKE_SOURCE')
    if root is None:pytest.skip('optional native source snapshot not bound')
    root=Path(root);path=root/'src/persistent_jepa/pokeworld.py'
    spec=importlib.util.spec_from_file_location('sprii_native_reference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return root,module


def test_fixed_action_replay_matches_native_trajectories(native):
    root,m=native;cfg=m.PokeConfig(train_systems=3,val_systems=3,test_systems=0)
    data=m.simulate_split(3,0,np.random.SeedSequence(20260919),cfg)
    s=data['states'].reshape(-1,64,8);a=data['actions'].reshape(-1,63,2)
    theta=np.repeat(np.column_stack((data['mass'],data['gamma'],data['stiffness'])),4,axis=0)
    for anchor in (0,24,47):
        out,contact=replay(s[:,anchor],a[:,anchor:anchor+16],theta,asdict(cfg))
        np.testing.assert_allclose(out,s[:,anchor+1:anchor+17],atol=5e-5,rtol=5e-5)
        np.testing.assert_array_equal(contact,data['contact'].reshape(-1,63)[:,anchor:anchor+16])
    sn,ag=sensitivities(s[:,24],a[:,24:40],theta,asdict(cfg),np.ones((5,8)),theta.std(0))
    assert sn.shape==(12,5,3) and np.isfinite(sn).all() and np.isfinite(ag).all()
    assert np.any(sn[:,:,0]>0) and np.any(sn[:,:,1]>0)


def test_native_encoder_export_and_provider_roundtrip(native):
    root,m=native
    with tempfile.TemporaryDirectory(prefix='sprii-native-') as d:
        out=Path(d);data_root=out/'data';data_root.mkdir()
        cfg=m.PokeConfig(train_systems=2,val_systems=2,test_systems=0)
        for split,offset,seed in (('train',0,1),('val',2,2)):
            data=m.simulate_split(2,offset,np.random.SeedSequence(seed),cfg)
            np.savez_compressed(data_root/(split+'.npz'),**data)
        write(data_root/'manifest.json',{'config':asdict(cfg),'fixture':True})
        script='''
import sys,torch
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/src')
from persistent_jepa.poke_model import PokeJEPA
torch.set_num_threads(1)
model=PokeJEPA('B2',history_length=24)
torch.save(dict(step=20000,model=model.state_dict(),config=dict(condition='G1',seed=0,family='refinement',test_read=False,model_variant='B2',history_length=24,synthetic_fixture=True)),sys.argv[2])
'''
        checkpoint=out/'fixture.pt'
        subprocess.run([sys.executable,'-c',script,str(root),str(checkpoint)],check=True,capture_output=True,text=True)
        files={str(root/'src/persistent_jepa'/n):sha(root/'src/persistent_jepa'/n) for n in ('poke_model.py','poke_torch.py','pokeworld.py')}
        files.update({str(data_root/n):sha(data_root/n) for n in ('manifest.json','train.npz','val.npz')})
        assets=dict(source_root=str(root),data_root=str(data_root),files=files,
            checkpoints={'G1_s0':dict(path=str(checkpoint),sha256=sha(checkpoint))},test_read=False)
        write(out/'assets.json',assets)
        command=[sys.executable,'-m','sprii_next','export-poke','--assets',str(out/'assets.json'),'--condition','G1','--seed','0',
            '--device','cpu','--batch-size','2','--output',str(out/'cache')]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        run=subprocess.run(command,capture_output=True,text=True,env=env,timeout=240)
        assert run.returncode==0,run.stderr
        source=PokeCache(read(out/'cache/SOURCE.json'))
        assert len(source.rows['validation'])==32
        assert source.data['train']['persistent'].shape==(32,64)
        assert sum(len(b.rows) for b in source.evaluation_batches())==160
        a=source.training_batch(0,7,16);b=source.training_batch(0,7,16)
        np.testing.assert_array_equal(a.persistent,b.persistent)
        assert a.rows==b.rows
        assert sha(checkpoint)==assets['checkpoints']['G1_s0']['sha256']
