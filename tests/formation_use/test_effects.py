import itertools
from pathlib import Path
import tempfile
import numpy as np
import torch
from sprii_next.providers import Batch
from sprii_next.model import Reader
from sprii_next.effects import effect_analysis,controlled_pairs
from sprii_next.io import write,read,sha


def test_controlled_pairs_keep_multiple_physical_distances():
    theta=np.array([[1,1,2],[2,1,2],[4,1,2]],float)
    pairs=controlled_pairs(theta,['a','b','c'],0)
    assert len(pairs)==3
    assert len({round(abs(np.log(theta[i,0]/theta[j,0])),8) for i,j in pairs})==2


def test_gated_effect_export_and_target_alignment():
    torch.set_num_threads(1)
    class Provider:
        environment='pokeworld';identity='synthetic'
        normalization={'target_scale':np.ones((5,8)).tolist()}
        def __init__(self):
            self.theta=np.array(list(itertools.product([1.,2.,4.],[.5,2.],[1000.,2000.])))
            n=len(self.theta);self.p=np.zeros((n,64),np.float32);self.p[:,:3]=np.log(self.theta)
            self.rows={'validation':[dict(system_id=f's{i}',query_id=f'q{i}',donor_id=f'd{i}',
                initial_state=[-.07,0.,0.,0.,.07,0.,.1,0.],mass=float(t[0]),drag=float(t[1]),stiffness=float(t[2])) for i,t in enumerate(self.theta)]}
        def donors(self,split):
            assert split=='validation'
            return self.p,self.theta,np.array([r['system_id'] for r in self.rows[split]]),np.array([r['donor_id'] for r in self.rows[split]])
        def batch(self,split,ix,hi):
            n=len(ix)
            return Batch(np.zeros((n,128),np.float32),self.p[ix],self.theta[ix],np.zeros((n,16,2),np.float32),np.ones((n,16),np.float32),
                hi,np.zeros((n,8),np.float32),[self.rows[split][i] for i in ix])
    with tempfile.TemporaryDirectory(prefix='sprii-effects-') as tmp:
        root=Path(tmp);run=root/'reader';run.mkdir();source=Provider()
        job=dict(arm='matched',reader_seed=0,method='G1',source_seed=0)
        write(run/'RUN.json',dict(job=job,protocol_sha256='synthetic',provider_sha256=source.identity))
        torch.save(dict(model=Reader('matched',0).state_dict()),run/'head.pt')
        write(run/'COMPLETE.json',dict(smoke=False,checkpoint_sha256=sha(run/'head.pt'),run_sha256=sha(run/'RUN.json')))
        write(root/'pilot.json',dict(stage='pilot',positive_slopes=2,artifacts={},protocol_sha256='synthetic'))
        write(root/'gate.json',dict(decision='go',trend_review='synthetic fixture',binned_curve_nonflat=True,p_value_used=False,
            summary=str(root/'pilot.json'),summary_sha256=sha(root/'pilot.json')))
        effect_analysis(source,run,root/'gate.json',root/'sensitivity',max_pairs=4,recipient_systems=3)
        result=read(root/'sensitivity/RESULT.json');assert result['target_alignment'] is False
        write(root/'review.json',dict(decision='run_target_alignment',reason='synthetic fixture',
            sensitivity_result=str(root/'sensitivity/RESULT.json'),sensitivity_result_sha256=sha(root/'sensitivity/RESULT.json')))
        cfg=dict(dt=.05,substeps=20,finger_mass=1.,finger_radius=.06,object_radius=.09,damping_ratio=.25,force_max=20.,arena_half_extent=1.,wall_restitution=.5)
        effect_analysis(source,run,root/'gate.json',root/'alignment',max_pairs=4,recipient_systems=3,
            target_alignment_review=root/'review.json',simulator_config=cfg)
        rows=read(root/'alignment/rows.json');assert len(rows)==24 and all('cosine' in r for r in rows)
        with np.load(root/'alignment/vectors.npz') as v:
            assert np.isfinite(v['true_counterfactual_effect']).all()
            np.testing.assert_allclose(v['prediction_effect'],v['prediction_j']-v['prediction_i'])
            np.testing.assert_allclose(v['error_per_dimension_i'],(v['prediction_i']-v['target_vector'])**2)
