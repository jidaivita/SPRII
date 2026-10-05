# Structured and inferred PokeWorld relations

These wrappers extend the archived factorized PokeWorld trainer. Run from the repository root; `SPRII_ROOT` may be set to another checkout/data root. Source modules are bound to `benchmarks/pokeworld/revision`, and the factorized bank is read from `data/pokeworld_factorized`. The original bank fingerprint is retained so that another generated population is not silently treated as the same experiment.

## Structured relation errors

`poke_structured.py` supports `true`, `uniform50`, `drag50`, and `mass50`. A training update contains 48 relation slots. The three 50% conditions use the same 24-correct/24-incorrect mask and preserve query, rollout, anchor and random-number budgets. The structured donor mappings change only drag or mass, hold the remaining factors fixed, and are bijections over the full bank. Each source runs 20,000 steps for seeds 0, 1 and 2.

```sh
python experiments/poke_controls/poke_structured.py --audit
python experiments/poke_controls/poke_structured.py \
  --mode drag50 --seed 0 --output runs/poke_structured/drag50_seed0
```

The full validation population is used for frozen geometry/probe readout; training donor substitutions are not reused as validation relations.

## Inferred relations

`poke_inferred.py` exposes `prepare`, `audit`, and `train`. It constructs a history-level nearest-neighbor graph using frozen representation features, with separate `inferred` and `random` graph arms. The teacher must be the corresponding frozen source checkpoint under `runs/pokeworld/refinement_split_sSEED`; it is an input dependency, not a newly fitted teacher hidden inside the evaluation. Ground-truth physical parameters are used for diagnostic match rates only, never graph construction or selection.

```sh
python experiments/poke_controls/poke_inferred.py prepare --seed 0
python experiments/poke_controls/poke_inferred.py audit --seed 0
python experiments/poke_controls/poke_inferred.py train --seed 0 \
  --mode inferred --output runs/poke_inferred/seed0
```

`poke_native_postrun.py` and `poke_native_postrun_generic.py` run the frozen geometry/probe readouts for these source models. Data, source weights and generated result files are not included in this code package. The earlier relation-quality bank and the new factorized bank are different populations; their measured match fractions cannot be combined into an exact common interpolation curve.

Effective per-arm, per-seed hyperparameters are included under `../../configs/paper/poke_controls/`. These are configuration files only; no training logs are redistributed.
