# Formation and Use analyses

The `sprii_next` package exposes fixed-source physical probes, light geometry,
Decode/Oracle comparisons, donor interventions, paired effect analysis and
system-level aggregation. The CLI is deliberately limited to development data;
sealed evaluation remains separate in `final_only` and the native environment
protocols.

```bash
python -m sprii_next --help
python -m sprii_next plan --environment springworld --stage development
python -m sprii_next plan --environment pokeworld --stage pilot
python -m sprii_next.contrastive --help
```

Source descriptors bind source code, checkpoint, cache, data identities and
normalization. `assemble` creates the immutable protocol; `preflight` verifies
source identity and paired target vectors before fitting. `run` fits one job;
`run_grid.py` schedules an explicit grid. `geometry` uses persistent source
vectors only. `effects` evaluates fixed-query changes under donor substitutions.
`spring-report`, `baseline-report`, and `pilot-report` require the corresponding
complete paired grids.

Spring uses the native source in `benchmarks/springworld/native`. Revised
PokeWorld uses the separate revision source snapshot and an asset descriptor
for user-provided checkpoints and data. Absolute paths in a new local descriptor
refer only to that user's files and must be regenerated when assets move.

The Rel-InfoNCE implementation adapts the contrastive principle to the same
history encoder, P64 context and observation encoder. It is not a full
reproduction of an external FCRL pipeline. Its temperature is fixed at 0.07;
source seeds and optimization budget are specified in the implementation.
