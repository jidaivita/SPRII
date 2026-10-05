"""Synthetic two-host publication tests for orchestration, not new science."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from sprii_next.engine import jobs
from sprii_next.io import npz, sha, write


PACKAGE = Path(__file__).parents[2] / "benchmarks/formation_use"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalizer = load('finalizer', PACKAGE / 'scripts/finalize_spring_results.py')
emit = load('aggregation_fixture', Path(__file__).with_name('test_aggregation.py')).emit


def populate(publication):
    tasks = sorted(finalizer.EXPECTED_SOURCES)
    owners = {}
    for i in range(8):
        host = finalizer.HOSTS[i // 4]
        selected = tasks[i::8]
        for key in selected:
            owners[key] = host
        write(publication / host / 'lanes' / f'{host}_gpu{i % 4}.COMPLETE.json',
              dict(test_read=False, tasks=[dict(method=m, seed=s) for m, s in selected]))
    for job in [*jobs('springworld'), *jobs('springworld', 'baseline')]:
        emit(publication / owners[job['method'], job['source_seed']] / 'results/runs', job, poke=False)
    for method in finalizer.METHODS:
        for seed in range(3):
            directory = publication / owners[method, seed] / 'geometry' / f'{method}_s{seed}'
            npz(directory / 'source_vectors.npz', persistent_code=np.ones((6, 64)),
                physical_parameters=np.ones((6, 3)), system_id=np.repeat(['a', 'b', 'c'], 2),
                donor_id=np.array([f'd{i}' for i in range(6)]))
            write(directory / 'RESULT.json', dict(source=dict(method=method, source_seed=seed),
                reader_used=False, test_read=False, source_vectors_sha256=sha(directory / 'source_vectors.npz'),
                D_within=1., D_between=2., between_within_ratio=2., rho_m=.1, rho_gamma=.2,
                rho_k=.3, rho_k_over_m=.4, systems=3, histories=6))


def test_publication_waits_for_all_markers_and_complete_synthetic_aggregation(tmp_path):
    publication = tmp_path / 'published'
    lanes, pending = finalizer.published_lanes(publication)
    assert not lanes and len(pending) == 8
    partial = publication / 'new2/lanes/new2_gpu0.COMPLETE.json'
    partial.parent.mkdir(parents=True)
    partial.write_text('{')
    assert len(finalizer.published_lanes(publication)[1]) == 8
    partial.unlink()
    populate(publication)
    lanes, pending = finalizer.published_lanes(publication)
    assert len(lanes) == 8 and not pending
    result = finalizer.finalize(publication, tmp_path / 'report', PACKAGE, PACKAGE, lanes)
    assert result['spring_readers'] == 36 and result['baseline_readers'] == 18
    assert result['total_unique_readers'] == 54 and result['geometry_sources'] == 12
    assert result['baseline_comparison_cells'] == 36
    for report in result['reports'].values():
        assert sha(report['path']) == report['sha256']
    assert (tmp_path / 'report/staging/runs/springworld/development/Both/source0').is_symlink()


def test_duplicate_source_ownership_is_not_merged(tmp_path):
    populate(tmp_path)
    lanes, pending = finalizer.published_lanes(tmp_path)
    lanes[0]['tasks'].append(lanes[1]['tasks'][0])
    with pytest.raises(ValueError, match='duplicate source ownership'):
        finalizer.collect_results(tmp_path, lanes)
