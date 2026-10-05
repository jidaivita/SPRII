# Reproducing SPRII experiments

The public source release accompanies [Shaping Persistent Representations from Independent Interactions](https://jidaivita.github.io/sprii/SPRII.pdf). The [project page](https://jidaivita.github.io/sprii/) introduces the method and settings; this guide helps select an executable path through the repository.

## 1. Run the CPU example

Follow the [quick start](quickstart.md) and run `python examples/dclean_pairs.py`. It generates a small D-Clean dataset and checks independent interaction pairs, disjoint system splits, and collision-free wrong relations. No external data or trained checkpoint is needed.

For a small learning-and-probe check, install the `train` dependencies and run `python examples/core_smoke.py`. These examples validate local setup and implementation behavior at tutorial budgets.

## 2. Choose a benchmark and an evidence question

| Question | What to evaluate | Starting points |
|---|---|---|
| Formation | Persistent-property readout and relation controls with the source frozen | [Core](../benchmarks/core/README.md), [formation/use](../benchmarks/formation_use/README.md), [PokeWorld](../benchmarks/pokeworld/revision/README.md) |
| Use | Substitute context while keeping the consuming predictor fixed | [SpringWorld](../benchmarks/springworld/README.md), [CoPhy](../benchmarks/cophy/README.md), [RH20T](../benchmarks/rh20t/README.md), [Swimmer](../benchmarks/swimmer/README.md) |
| Value | Task error under the specified reader, horizon, and information budget, paired with physical-property probes | [Spring postrun](../experiments/spring_postrun/README.md), [NOD fields](../benchmarks/nod/README.md), [Pendulum](../benchmarks/baseline_adapters/README.md), [Overcooked](../benchmarks/overcooked/README.md) |

Use the benchmark's own installation and execution guide. The [recipe index](paper_recipes.md) covers thirteen settings and links to their training entry points. Its scientific recipe text records the **2026-09-25 manuscript snapshot**; the current reading copy is the author manuscript linked above. Match a result to its manuscript revision, configuration and source before claiming reproduction.

## 3. Prepare the specified inputs

- Generated simulations: use the included generators and preserve the declared system splits and relation controls.
- External data: follow [datasets](datasets.md), the original access terms, and benchmark preprocessing instructions.
- External learners: use the pinned source revisions and the benchmark-specific preparation instructions. Dependencies fetched from upstream retain their upstream terms.
- Frozen-source analyses: supply the exact checkpoint, data split, and reader configuration required by that experiment. This source package does not include historical checkpoints or logs.

[Hyperparameters](hyperparameters.md) indexes released configurations and CLI defaults. [Interfaces](interfaces.md) describes inputs and outputs. A default-argument listing does not replace an executed-run configuration; retain all benchmark-specific overrides and continuation parents.

## 4. Preserve the comparison

Keep the documented source budget, checkpoint selection, data population, recipient inputs, and evaluation horizon. Pair downstream utility with the corresponding persistent-parameter or invariant probes. Keep representation readout, fixed-predictor interventions, and newly fitted readers distinct when interpreting results; [protocols](protocols.md) provides the full conventions.

## What this release verifies

Version 0.2.0 publishes the verified 0.1.2 implementation with new public documentation and citation metadata. No training implementation, numerical configuration, or scientific result is changed by this update. The release includes the original component tests and source provenance. It does not bundle external datasets, historical training logs, or model checkpoints, and it does not claim that all full-budget results have been rerun from this distribution.

See [verification](verification.md) for the checks actually run, including those inherited from earlier releases. Verify the downloaded source with:

```sh
python scripts/verify_release.py
```
