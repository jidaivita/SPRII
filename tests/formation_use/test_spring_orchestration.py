"""Only checks introduced by incremental source / two-host execution."""
import importlib.util
from pathlib import Path
import pytest
from sprii_next.engine import jobs
from sprii_next.io import recipe_digest
from sprii_next.protocol import default_protocol

spec=importlib.util.spec_from_file_location('run_grid',Path(__file__).parents[2]/'benchmarks/formation_use/scripts/run_grid.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_two_hosts_cover_grid_once_and_ready_source_runs_early():
    grid=jobs('springworld')
    first=runner.selected_indices(grid,shard_index=0,shard_count=2)
    second=runner.selected_indices(grid,shard_index=1,shard_count=2)
    assert len(first)==len(second)==18 and not set(first)&set(second)
    assert sorted(first+second)==list(range(36))
    assert runner.selected_indices(grid,source_seeds=[1])==list(range(12,24))
    with pytest.raises(ValueError):runner.selected_indices(grid,job_indices=[0,0])


def test_sources_can_finish_separately_without_changing_reader_recipe():
    a=default_protocol([dict(source_seed=0)])
    b=default_protocol([dict(source_seed=1)])
    assert recipe_digest(a)==recipe_digest(b)
    b['reader']['steps']+=1
    assert recipe_digest(a)!=recipe_digest(b)
