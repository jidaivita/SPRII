# D-Clean and core PokeWorld

These are the original state-based D-Clean and visual PokeWorld pipelines.
Additional factorized PokeWorld experiments have their own revision snapshot.
Run commands from the repository root after installing `requirements/core.txt`
and adding `src` to `PYTHONPATH`.

```bash
python benchmarks/core/scripts/generate_dclean.py --output data/dclean
python benchmarks/core/scripts/train.py \
  --data-root data/dclean --run-dir runs/dclean/b3-seed0 \
  --variant B3 --seed 0 --steps 20000 --lambda-p 1.0 --lambda-x 0.1
python benchmarks/core/scripts/evaluate.py --help

python benchmarks/core/scripts/generate_pokeworld.py --output data/pokeworld
python benchmarks/core/scripts/train_pokeworld.py --help
python benchmarks/core/scripts/evaluate_pokeworld.py --help
```

The D-Clean training command is a concrete executable example; it is not a
substitute for the selected configuration of every paper table. The parsers
expose objective weights, history, seed, device, relation map, and source
binding where supported. `--device cpu` permits CPU execution, although full
budgets are intended for accelerators.

D-Clean is an analytically integrated forced damped dynamical system. Physical
systems are split before rollouts are generated. PokeWorld is an independent
simulation implementation using the published interaction equations; its
configuration includes explicit reproduction choices. See the dataclasses in
`src/persistent_jepa/simulator.py` and `pokeworld.py` for every numerical value.

Probe and utility code is in `persistent_jepa.evaluation` and
`persistent_jepa.poke_evaluation`. Additional scripts cover fixed random
relations, nonoverlapping histories, counterfactual interventions, geometry,
OOD generation and evaluation, and raw-history GRU references. The Baxter and
RH20T adapters use external datasets and retain separate preparation and
evaluation commands. Download those datasets from their official sources.

Data generation writes dataset manifests and split files. Training writes new
run artifacts locally; no historical logs are part of this release. Test
selection checks are intentionally enforced by the evaluation code.
