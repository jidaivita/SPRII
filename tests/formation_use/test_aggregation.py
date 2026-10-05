"""Complete-grid statistics using synthetic, explicitly fabricated receipts."""
import copy
import tempfile
from pathlib import Path
import numpy as np
import pytest
from sprii_next.io import write,read,sha,npz
from sprii_next.engine import jobs,job_name
from sprii_next.statistics import pilot_summary,spring_summary,baseline_summary


def emit(root,job,poke=True):
    folder=root/job_name(job);(folder/'evaluation').mkdir(parents=True)
    rows=[]
    for system in range(3):
        for window in range(4):
            d=(window-1.5)+.1*system
            error=1.
            if job['arm']=='matched':error-=.2+(.03*d if job['method']=='G1' and job['source_seed']<2 else 0.)
            if job['arm']=='decode':error+=.1
            if job['arm']=='oracle':error-=.1
            rows.append(dict(system_id=f's{system}',query_id=f's{system}:w{window}',donor_id=f's{system}:d',donor_system_id=f's{system}',
                horizon=16,query_episode=f's{system}:q',donor_episode=f's{system}:d',mass=system+1.,drag=1.,stiffness=3.,
                aggregate_error=error,primary=True,sensitivity_dominance=d,split_weight=1/12))
    if not poke:
        # All five horizons are present in a full-mixture report.
        orig=copy.deepcopy(rows);rows=[]
        for h in (1,2,4,8,16):
            for r in orig:rows.append(dict(r,horizon=h,query_id=r['query_id']+f':h{h}',primary=h==16,split_weight=1/60))
    write(folder/'RUN.json',dict(job=job,smoke=False,protocol_sha256='synthetic',normalization={'unit':'fixture'},reader={'fixture_recipe':True}))
    (folder/'head.pt').write_bytes(b'explicitly synthetic aggregate fixture')
    write(folder/'PROBE.json',dict(fixture=True));npz(folder/'probe_vectors.npz',prediction_vector=np.zeros((3,3)))
    write(folder/'evaluation/rows.json',rows);npz(folder/'evaluation/vectors.npz',prediction_vector=np.zeros((len(rows),8)))
    write(folder/'evaluation/RESULT.json',dict(all_cases=len(rows),shards=[dict(rows='rows.json',rows_sha256=sha(folder/'evaluation/rows.json'),
        vectors='vectors.npz',vectors_sha256=sha(folder/'evaluation/vectors.npz'),count=len(rows))]))
    write(folder/'COMPLETE.json',dict(job=job,smoke=False,run_sha256=sha(folder/'RUN.json'),checkpoint_sha256=sha(folder/'head.pt'),
        evaluation_sha256=sha(folder/'evaluation/RESULT.json'),probe_sha256=sha(folder/'PROBE.json'),
        probe_vectors_sha256=sha(folder/'probe_vectors.npz'),training_sequence_sha256='same fixture sequence'))


def test_pilot_grid_trend_and_incomplete_denominator():
    with tempfile.TemporaryDirectory(prefix='sprii-aggregate-') as tmp:
        root=Path(tmp)
        for j in jobs('pokeworld','pilot'):emit(root,j)
        r=pilot_summary(root)
        assert r['positive_slopes']==2 and r['p_value_used_for_gate'] is False and r['automatic_go'] is False
        assert r['per_seed'][0]['slope']==pytest.approx(.03)
        (root/job_name(jobs('pokeworld','pilot')[0])/'COMPLETE.json').unlink()
        with pytest.raises(FileNotFoundError):pilot_summary(root)


def test_spring_paired_primary_and_secondary():
    with tempfile.TemporaryDirectory(prefix='sprii-aggregate-') as tmp:
        root=Path(tmp)
        for j in jobs('springworld'):emit(root,j,poke=False)
        r=spring_summary(root)
        assert r['complete_readers']==36
        assert r['primary_decode_minus_persistent']['mean']==pytest.approx(.1)
        assert r['primary_decode_minus_persistent']['unit']=='physical_system'
        assert len(r['secondary_horizon_mse']['decode'])==5
        for j in jobs('springworld','baseline'):emit(root,j,poke=False)
        baseline=baseline_summary(root)
        assert baseline['relative_gain']['value']==pytest.approx(0.)
        assert len(baseline['cells'])==36 and len(baseline['probes'])==6
