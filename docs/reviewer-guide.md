# Reviewer guide

[Latest author PDF](https://jidaivita.github.io/sprii/SPRII.pdf) · [Thirteen-setting code index](../README.md#explore-thirteen-settings)

SPRII asks what persistent context learns, how a fixed predictor uses it, and
when it improves a task. This guide maps those questions to the released
implementations and separates quick checks from result reproduction. In the
thirteen-setting index, each setting name opens its project-page detail section;
the adjacent links lead to the corresponding source, recipe, and inputs.

## Follow a paper question

| Paper question | What to inspect | How to run it and required inputs |
|---|---|---|
| **Training principle:** Align related contexts; Cross transfers context while preserving the native objective. | [Core objective](../src/persistent_jepa/objective.py), [relation sampling](../src/persistent_jepa/sampling.py), [controlled training](../benchmarks/core/README.md). | Start with the CPU pairing example below; `examples/core_smoke.py` checks finite objective/gradients and a physical-parameter probe after installing the `train` dependencies. These are synthetic setup checks. |
| **Formation:** relation quality and the shared factors change accessible persistent information. | [PokeWorld revision](../benchmarks/pokeworld/revision/README.md), its [frozen geometry/probe evaluator](../benchmarks/pokeworld/revision/scripts/evaluate_pokeworld_revision_geometry.py), and [Spring geometry](../src/sprii_next/geometry.py). | Follow the PokeWorld guide to generate the factorized bank and select its source snapshot. Probe evaluation needs the matching trained checkpoint and declared split; a preview image is not an evaluation bank. |
| **Use:** substituting persistent context changes an already-fixed predictor. | [Formation/use workflow](../benchmarks/formation_use/README.md), [paired effect analysis](../src/sprii_next/effects.py), [CoPhy learner routes](../benchmarks/cophy/README.md), and [Swimmer interventions](../benchmarks/swimmer/README.md#entry-points). | Bind the source, checkpoint, cache and normalization in a source descriptor; the guide's `assemble` and `preflight` stages validate it before reader/effect jobs. Preserve recipient inputs and fixed weights. These routes require their specified checkpoints and data. |
| **Value:** context benefit depends on the task, horizon, and readout. | [Spring frozen-reader evaluation](../experiments/spring_postrun/README.md), [reader protocol](../experiments/spring_postrun/reader/sprii_next/protocol.py), [Pendulum control](../benchmarks/baseline_adapters/README.md#pendulum), and [Overcooked evaluation/probes](../benchmarks/overcooked/README.md#evaluation-and-probes). | Spring's `export`, `fit`, `evaluate`, and `probe` stages require the matching native bank and completed source. Control experiments use their own data and dependencies. Report task error with the corresponding physical-property or invariant/function probes; distinguish new readers from fixed-predictor interventions. |
| **Breadth:** the principle is instantiated across thirteen settings and multiple learners. | The [thirteen-row index](../README.md#explore-thirteen-settings) gives each setting's code, recipe and inputs. [Recipes](paper_recipes.md) describe the separate learner budgets and continuation schedules. | Select a setting and learner before installing its dependencies. Shared directories do not merge scientific settings, and one learner's default budget does not specify another's experiment. |

## Run a small check first

From the repository root in a Python 3.10+ environment:

```sh
python -m pip install -e . pytest
python examples/dclean_pairs.py
python -m pytest -q tests/core/test_preflight.py tests/core/test_pokeworld_ood.py tests/core/test_pokeworld_interaction_mechanism.py
```

These CPU checks exercise simulation, independent interactions, disjoint system
splits, and relation controls. This exact test subset passed 14 tests during
public-release verification. It does not train a paper model.

For the small learning-and-probe example:

```sh
python -m pip install -e '.[train]'
python examples/core_smoke.py
```

That command uses synthetic data and an untrained model. Its earlier verification
record is in [verification](verification.md); the full Torch suite was not rerun
for the public documentation update.

## Reproduce a selected result

1. Open the setting's **Guide**, **Recipe**, and **Inputs** from the index. Preserve the source version, system split, relation rule, learner, budget, and checkpoint-selection procedure.
2. Prepare the specified data and dependencies. Controlled simulators are included; CoPhy, Baxter, RH20T and Burgers require their documented external resources. Overcooked uses a pinned upstream bootstrap; Swimmer has its own MuJoCo package. CaDM, GEPS and CoDA use separately fetched fixed revisions.
3. Choose either a new training run or a replay using the specified frozen inputs. Reader/intervention protocols may require historical checkpoints, cache identities and normalization records. Their identity checks should remain intact.
4. Keep Formation, Use and Value outputs distinct. Pair downstream utility with the relevant persistent-parameter or invariant probe, and preserve the documented resampling unit and evaluation population.

The release includes source, configurations, generators, tests and recipe guides.
It does **not** bundle historical checkpoints, external datasets or training logs.
The recipe text records the **2026-09-25 manuscript snapshot**; the latest PDF may
contain later manuscript revisions. The source map is a navigation aid, not a
claim that every number in the latest PDF has been rerun from this distribution.
For exact inputs and the checks actually performed, see [reproduction](reproduce.md),
[interfaces](interfaces.md), [datasets](datasets.md), and [verification](verification.md).
