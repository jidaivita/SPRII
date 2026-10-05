# Hyperparameters and configuration

The [paper recipe guide](paper_recipes.md) lists the current manuscript budgets
and benchmark-specific settings. Parameters are also provided at the implementation
level rather than replaced by a single generic recipe. Preserve all overrides when reproducing an experiment.

| Component | Configuration source |
|---|---|
| Core simulator/model defaults | [Core defaults](../configs/core/defaults.json) and [model dataclasses](../src/persistent_jepa/model.py) |
| Spring native source | [Native configuration](../benchmarks/springworld/native/NIGHT_POLICY.json) and native training entrypoints |
| Frozen-source readers | [Reader protocol](../src/sprii_next/protocol.py) and [formation guide](../benchmarks/formation_use/README.md) |
| CoPhy | [CoPhy configurations](../configs/cophy/) and [adapter guide](../benchmarks/cophy/README.md) |
| Burgers/FHN | [NOD configurations](../benchmarks/nod/configs/) and [guide](../benchmarks/nod/README.md) |
| CaDM/GEPS/CoDA/Pendulum | [Baseline adapter guide](../benchmarks/baseline_adapters/README.md) |
| Poke revision | [PokeWorld revision guide](../benchmarks/pokeworld/revision/README.md) |
| Swimmer | [Swimmer configurations](../benchmarks/swimmer/configs/) |
| Overcooked | [Overcooked configurations](../benchmarks/overcooked/configs/) and [guide](../benchmarks/overcooked/README.md) |
| DALI/FCRL | [Selected recipes](../benchmarks/additional_baselines/paper_recipes.json) and [guide](../benchmarks/additional_baselines/README.md) |

[`cli_defaults.json`](cli_defaults.json) is an automatically extracted index of literal defaults
and expressions from the distributed argument parsers. It makes hidden parser
defaults discoverable, but **it is not an expanded paper-run configuration**:
CLI overrides, model dataclasses and benchmark-specific configuration still
apply. Source paths and line numbers in the index are relative to this archive.

The small example uses an explicitly reduced budget. Do not label example
settings or implementation defaults as the final paper recipe.

