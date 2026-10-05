# Quick start

Clone [jidaivita/SPRII](https://github.com/jidaivita/SPRII) or download its source ZIP.
With Python 3.10 or later, run these commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment on macOS/Linux with `source .venv/bin/activate`, or on
Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then install and run:

```sh
python -m pip install -e .
python examples/dclean_pairs.py
```

This small CPU example checks independent rollouts, disjoint physical-system
splits and collision-free wrong relations. It is a tutorial budget, not a paper
experiment.

For model, probe and preprocessing checks:

```sh
python -m pip install -e '.[train,dev]'
python examples/core_smoke.py
python -m pytest -q tests/core tests/formation_use
```

Some optional native integrations require the inputs documented in the
[SpringWorld guide](../benchmarks/springworld/README.md). Other frameworks,
MuJoCo and external datasets have benchmark-specific installation instructions.
Check the [verification scope](verification.md), [paper recipes](paper_recipes.md)
and [data sources](datasets.md) before a full-budget run.

To check the downloaded files against the source manifest:

```sh
python scripts/verify_release.py
```
