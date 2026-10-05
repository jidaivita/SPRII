# CoPhy adapters and evaluation

This subtree contains the CoPhy front-end and the research adapters for
Collision, Balls, and Blocktower. It is distributed under **GPL-3.0**, separately
from the root license. Upstream CoPhy is
[github.com/fabienbaradel/cophy](https://github.com/fabienbaradel/cophy),
with the audited baseline commit
`170fe463c9e80ddf0978110438a710cabb4997e2`. The
[official project page](https://projet.liris.cnrs.fr/cophy/) provides the dataset
and pretrained front-end resources. Dataset and weight redistribution is not
implied by the source-code license.

The included upstream-derived source has been modified for corrected input and
evaluation handling, explicit relation/parameter permissions, persistent and
transient role intervention, training/evaluation entrypoints and portability.
These modifications are distributed in September 2026; upstream attribution
and the full GPL-3.0 license are retained.

| Location | Purpose |
| --- | --- |
| `code/cf_learning`, `code/derendering`, `code/dataloaders` | Upstream-derived model, visual front-end and scene loaders |
| `code/cophy_adapter.py` | Typed AB/current-state/target separation, P/T swap, objectives and probes |
| `code/cophy_relations.py` | Same-physics pairing, balanced random controls and parameter references |
| `code/cophy_prepare_artifacts.py` | Data/split and relation artifacts |
| `code/cophy_evaluate.py` | Donor and physical-parameter evaluation |
| `code/latent_*`, `code/monolithic_jepa_*` | JEPA, CPC and RSSM adapters and readers |
| `code/cophy_complete_v7` | Complete learner-family training, readouts and final-test admission/evaluation |
| `discovery` | Source formation and query/readout implementations used by later continuation code |
| `code_profiles` | Scene-specific runtime corrections |
| `configs/cophy` at repository root | Versioned scientific protocol definitions |

Versioned directories preserve the algorithms and compatibility of the original
experiments. An earlier configuration is not automatically the final recipe
for all tables. Continuation entrypoints require compatible parent checkpoints
and source manifests; they are not independent from-scratch commands.

```bash
python -m pip install -r requirements/cophy.txt
export PYTHONPATH="$PWD/benchmarks/cophy/code:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python benchmarks/cophy/code/cophy_prepare_artifacts.py --help
python benchmarks/cophy/code/cf_learning/main.py --help
python benchmarks/cophy/code/cophy_evaluate.py --help
python -m pytest -q benchmarks/cophy/tests
```

Historical runtime entrypoints use `SPRII_COPHY_ROOT`, defaulting to
`runs/cophy`. Their runtime layout expects `source/`, prepared assets and
registered scene profiles beneath that root. `prepare_runtime.py` copies the
included source into this layout without modifying research data:

```bash
python benchmarks/cophy/prepare_runtime.py --output runs/cophy
export SPRII_COPHY_ROOT="$PWD/runs/cophy"
```

Supply data, fixed splits, front-end weights, and source/reader checkpoints as
required by each entrypoint. No experiment log, private execution receipt,
training dataset or model weight is included. The tests use synthetic arrays
and model fixtures to exercise objectives, parameter counts, gradients, pairing,
metrics and evaluation invariants. They do not constitute a full reproduction
of the original CoPhy paper or the reported training runs.
