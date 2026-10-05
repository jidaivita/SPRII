import json
from pathlib import Path

import numpy as np

from paper_c.swimmer.model import SwimmerModel
from paper_c.swimmer.s3_evaluate import _alignment, _grid_from_metadata, _selection, _validate_shared_query_targets
from paper_c.swimmer.waveforms import banks


ROOT = Path(__file__).resolve().parents[2]


def _config():
    return json.loads((ROOT / "configs/swimmer_external_assay_v1.json").read_text())


def test_swimmer_observation_and_rollout_are_deterministic():
    config = _config()
    model = SwimmerModel(config["model"])
    rng = np.random.default_rng(17)
    initial = model.sample_initial_state(rng, config["transient_initial_state"])
    history, _ = banks(0.4, config["model"]["timestep_s"])
    landmarks = np.asarray([10, 20, 30, 40])
    first = model.rollout(np.zeros(5), initial, history["j1_slow"], landmarks)
    second = model.rollout(np.zeros(5), initial, history["j1_slow"], landmarks)
    assert first.shape == (32,)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)


def test_initial_state_is_explicit_transient_nuisance():
    config = _config()
    model = SwimmerModel(config["model"])
    rng = np.random.default_rng(19)
    first_initial = model.sample_initial_state(rng, config["transient_initial_state"])
    second_initial = model.sample_initial_state(rng, config["transient_initial_state"])
    history, _ = banks(0.4, config["model"]["timestep_s"])
    landmarks = np.asarray([40])
    first = model.rollout(np.zeros(5), first_initial, history["quadrature"], landmarks)
    repeated = model.rollout(np.zeros(5), first_initial, history["quadrature"], landmarks)
    changed = model.rollout(np.zeros(5), second_initial, history["quadrature"], landmarks)
    np.testing.assert_allclose(first, repeated, atol=0, rtol=0)
    assert not np.allclose(first, changed)


def test_all_waveforms_have_equal_energy_and_respect_control_limit():
    config = _config()
    history, query = banks(1.2, config["model"]["timestep_s"])
    waveforms = [*history.values(), *query.values()]
    energies = np.asarray([np.mean(np.sum(value ** 2, axis=1)) for value in waveforms])
    np.testing.assert_allclose(energies, energies[0], rtol=1e-12, atol=1e-12)
    assert max(np.abs(value).max() for value in waveforms) <= config["model"]["control_limit"]


def test_s3_metadata_grid_preserves_unequal_semantic_axes_and_row_order():
    sizes = {"system": 2, "realization": 3, "history": 5, "candidate": 6, "query": 7}
    metadata = np.asarray([
        (s, r, h, e, q)
        for s in range(2) for r in range(3) for h in range(5)
        for q in range(7) for e in range(6)
    ], dtype=np.int64)
    values = np.asarray([10000*s + 1000*r + 100*h + 10*e + q for s, r, h, e, q in metadata])
    order = np.random.default_rng(23).permutation(len(metadata))
    grid = _grid_from_metadata(values[order], metadata[order], sizes, candidate=True)
    assert grid.shape == (2, 3, 5, 6, 7)
    assert grid[1, 2, 4, 5, 6] == 12456


def test_s3_selector_and_alignment_rank_only_candidate_axis():
    systems, histories, candidates, queries = 4, 5, 6, 7
    score = np.zeros((systems, histories, candidates, queries), dtype=np.float64)
    gain = np.zeros_like(score)
    for e in range(candidates):
        score[:, :, e, :] = e
        gain[:, :, e, :] = e
    selection, comparisons = _selection({
        "conditional_physical_value": score,
        "standalone_query_value": score,
        "trajectory_diversity": score,
        "action_diversity": score,
    }, gain, seed=29, reps=20)
    assert all(row["mean_regret"] == 0 for row in selection)
    assert all(row["mean"] == 0 for row in comparisons)
    assert _alignment(score, gain, seed=31, reps=20)["mean"] == 1.0


def test_s3_rejects_candidate_target_that_differs_from_shared_query_target():
    sizes = {"system": 1, "realization": 1, "history": 2, "candidate": 3, "query": 4}
    bmeta = np.asarray([(0, 0, h, q) for h in range(2) for q in range(4)], dtype=np.int64)
    cmeta = np.asarray([(0, 0, h, e, q) for h in range(2) for q in range(4) for e in range(3)], dtype=np.int64)
    baseline = np.asarray([[h, q] for _, _, h, q in bmeta], dtype=np.float32)
    candidate = np.asarray([[h, q] for _, _, h, _, q in cmeta], dtype=np.float32)
    _validate_shared_query_targets(baseline, candidate, bmeta, cmeta, sizes)
    candidate[-1, -1] += 1
    with np.testing.assert_raises(ValueError):
        _validate_shared_query_targets(baseline, candidate, bmeta, cmeta, sizes)


def test_s3_metadata_grid_rejects_duplicate_rows():
    sizes = {"system": 1, "realization": 1, "history": 1, "candidate": 2, "query": 1}
    metadata = np.asarray([(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)], dtype=np.int64)
    with np.testing.assert_raises(ValueError):
        _grid_from_metadata(np.asarray([1.0, 2.0]), metadata, sizes, candidate=True)
