# SpringWorld

`native/` contains the actual native source snapshot used by the source and
reader adapters. It includes the simulator, image observations, native model,
strict normalization adapter, feature extraction and physical target interface.
The nested source layout is preserved to satisfy native source-identity checks.

From the repository root:

```bash
export SPRII_SPRING_NATIVE="$PWD/benchmarks/springworld/native"
export SPRII_NATIVE_ROOT="$SPRII_SPRING_NATIVE"
export PYTHONPATH="$SPRII_SPRING_NATIVE:$SPRII_SPRING_NATIVE/src:$SPRII_SPRING_NATIVE/a_src:$SPRII_SPRING_NATIVE/extension:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m persistbench.envs.visual_elastic_coupling.research_bank --help
python -m sprii_next plan --environment springworld --stage development
python benchmarks/formation_use/scripts/train_spring_source.py --help
python -m sprii_next export-spring --help
python -m pytest -q tests/formation_use/test_native_spring.py
```

The research-bank generator provides a `--smoke` mode. Source training accepts
`--native-root`, `--bank`, `--method`, `--seed`, `--output`, and `--device`.
`Structure`, `Align`, `Cross`, and `Both` are distinct source recipes.
`NIGHT_POLICY.json` contains the source training specification and the original
bank snapshot hash. The source-repeat wrapper requires that exact bank. For a
newly generated bank, use the explicit native pretraining interface with the
new bank hash and a configuration file; do not overwrite the original hash to
misrepresent a regenerated bank as the original asset.

```bash
python -m persistbench.envs.visual_elastic_coupling.a_pretraining --help
python -m sprii_next export-spring --native-root "$SPRII_SPRING_NATIVE" \
  --completion runs/spring/both-seed0/COMPLETE.json --bank data/spring \
  --method Both --seed 0 --output runs/spring/cache/both-seed0
python -m sprii_next assemble --sources runs/spring/cache/both-seed0/SOURCE.json \
  --output runs/spring/protocol.json
python benchmarks/formation_use/scripts/run_grid.py --help
```

Completed source checkpoints, generated image banks and feature caches are
separate assets. The source release contains their generators and interfaces,
not those large artifacts. Reader grids require the corresponding completed
sources. The integration test uses synthetic arrays to validate the real
600-case/100-system endpoint; it does not train a new scientific source.

The native simulator package is Apache-2.0. See `native/src/persistbench/LICENSE`.

The legacy development-bank exporter accepts a caller-provided protocol file
through `SPRII_SOURCE_PLAN`; no machine-specific plan path is embedded.
