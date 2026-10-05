# Interface guide

The release preserves existing package names and typed scientific interfaces.
There is no invented universal wrapper that silently changes a benchmark's
observations or model inputs.

| Interface | Implementation | Contract |
|---|---|---|
| D-Clean simulation | `persistent_jepa.simulator.DCleanConfig`, `DCleanDataset` | Separate state, action, physical-parameter and system-ID arrays per split |
| Relations | `persistent_jepa.sampling` | Physical system and rollout identities remain distinct |
| Model configuration | `persistent_jepa.model.ModelConfig` | Observation, transient, persistent and predictor dimensions |
| Paired model batches | `persistent_jepa.torch_data.PairedBatch` | Explicit histories, targets, future actions and masks |
| Training objective | `persistent_jepa.objective.compute_objective` | Native loss with explicit alignment/cross weights and supervised-reference branch |
| CoPhy P/T contract | `benchmarks/cophy/code/cophy_adapter.py` | Observation, visual input, targets and donor pairs are distinct objects |
| Source/reader protocol | `sprii_next.protocol`, `sprii_next.providers` | Frozen source identity, splits, cases and reader configuration |
| NOD/FHN | `benchmarks/nod/src` | Separate PDE model, solver, query coordinates and relation samplers |
| Swimmer | `benchmarks/swimmer/code/paper_c` | Physical generator, source predictor, persistent-update adapter and paired evaluation |

Consult [core.md](core.md) for tensor shapes. Benchmark READMEs identify the
actual training, evaluation and probe entrypoints and their additional inputs.
Use `--help` for the arguments supported by a particular program; installing
the lightweight root package does not install every optional framework.

