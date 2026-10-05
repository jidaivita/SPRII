import numpy as np

from paper_c.swimmer.personalization_diagnostic import (
    _candidate_spearman,
    _crossfit_center,
    _crossfit_choices,
    _double_crossfit_physical_choices,
    _rank_metrics,
    build_system_manifest,
)


def test_personalization_manifest_separates_source_local_and_balances_crossfit_halves():
    systems=np.arange(400,dtype=np.float64).reshape(80,5)
    first=build_system_manifest(systems,64,"selection","crossfit")
    second=build_system_manifest(systems,64,"selection","crossfit")
    assert first==second
    assert [row["local_system_index"] for row in first]==list(range(64))
    assert len({row["source_system_index"] for row in first})==64
    assert sum(row["system_crossfit_half"]=="A" for row in first)==32
    assert sum(row["system_crossfit_half"]=="B" for row in first)==32


def test_candidate_spearman_uses_named_candidate_axis():
    score=np.zeros((3,5,6,7),dtype=np.float64)
    for candidate in range(6):score[:,:,candidate,:]=candidate
    np.testing.assert_allclose(_candidate_spearman(score,score),1.0)


def test_rank_metrics_recover_perfect_split_half_reliability():
    score=np.zeros((4,5,6,7),dtype=np.float64)
    for candidate in range(6):score[:,:,candidate,:]=candidate
    result=_rank_metrics(score,score)
    assert result["spearman"]==1.0
    assert result["spearman_brown"]==1.0
    assert result["top1"]==1.0
    assert result["top2_overlap"]==1.0


def test_crossfit_population_choices_use_opposite_system_half():
    score=np.zeros((4,1,3,1),dtype=np.float64)
    score[:2,0,0,0]=10
    score[2:,0,2,0]=10
    halves=np.asarray(["A","A","B","B"])
    choices=_crossfit_choices(score,halves)
    assert np.all(choices[:2]==2)
    assert np.all(choices[2:]==0)


def test_nuisance_crossfit_never_evaluates_a_choice_on_its_selection_half():
    score=np.zeros((4,4,1,2,1),dtype=np.float64)
    score[:,0::2,0,0,0]=10
    score[:,1::2,0,1,0]=10
    halves=np.asarray(["A","A","B","B"])
    personal,_=_double_crossfit_physical_choices(score,halves)
    assert np.all(personal[:,0::2,0,0]==1)
    assert np.all(personal[:,1::2,0,0]==0)


def test_centering_uses_only_opposite_system_half():
    values=np.asarray([0.0,2.0,10.0,14.0])[:,None,None,None]
    halves=np.asarray(["A","A","B","B"])
    centered=_crossfit_center(values,halves).reshape(-1)
    np.testing.assert_allclose(centered,[-12.0,-10.0,9.0,13.0])
