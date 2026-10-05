<p align="center"><img src="assets/branding/logo.svg" width="760" alt="SPRII" /></p>

# Shaping Persistent Representations from Independent Interactions

**Ji Dai · Quan Fang · Junyu Gao · Rongfeng Guo · Haoyan Rong · Yiping Huang · Yongxi Li**

[Paper](https://arxiv.org/abs/2609.34604) · [Project page](https://jidaivita.github.io/sprii/) · [Code guide](docs/core.md) · [Reproduce](docs/reproduce.md) · [Citation](#citation)

SPRII uses relations between independent interactions to learn persistent context: information about a system that can be reused across changing states and actions. **Align** encourages contexts from related interactions to agree. **Cross** asks one interaction's context to help predict another's future. Both complement the learner's native prediction objective.

The experiments distinguish three questions: **Formation** — what persistent information can be read from the representation; **Use** — how that context changes a fixed predictor; and **Value** — when it improves a downstream task. The repository includes simulators, learner adaptations, physical-parameter probes, fixed-predictor interventions, and benchmark-specific training and evaluation programs.

## Get started

```sh
git clone https://github.com/jidaivita/SPRII.git
cd SPRII
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/dclean_pairs.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. The CPU example generates independent D-Clean interactions and checks disjoint system splits and wrong-relation controls. It is a small demonstration; paper experiments have their own budgets and inputs.

For learning, probes and implementation checks:

```sh
python -m pip install -e '.[train,dev]'
python examples/core_smoke.py
python -m pytest -q tests/core tests/formation_use
```

Start with the [reproduction guide](docs/reproduce.md) for the path from the CPU example to a selected benchmark. Some native integration tests need the source inputs in the [SpringWorld guide](benchmarks/springworld/README.md). Framework-specific dependencies and data are documented in each benchmark.

## Explore thirteen settings

<table>
<tr>
<td width="33%" align="center" valign="top"><a href="assets/environments/springworld.png"><img src="assets/environments/cards/springworld.svg" width="240" alt="SpringWorld: Physics demonstration" /></a><br/><a href="benchmarks/springworld/README.md">SpringWorld</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/pokeworld.png"><img src="assets/environments/cards/pokeworld.svg" width="240" alt="PokeWorld: History replay" /></a><br/><a href="benchmarks/pokeworld/revision/README.md">PokeWorld</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/dclean.png"><img src="assets/environments/cards/dclean.svg" width="240" alt="D-Clean: Recorded trajectory" /></a><br/><a href="benchmarks/core/README.md">D-Clean</a></td>
</tr>
<tr>
<td width="33%" align="center" valign="top"><a href="benchmarks/cophy/README.md"><strong>CoPhy Collision</strong></a><br/>Code and experiment guide</td>
<td width="33%" align="center" valign="top"><a href="benchmarks/cophy/README.md"><strong>CoPhy Balls</strong></a><br/>Code and experiment guide</td>
<td width="33%" align="center" valign="top"><a href="benchmarks/cophy/README.md"><strong>CoPhy Blocktower</strong></a><br/>Code and experiment guide</td>
</tr>
<tr>
<td width="33%" align="center" valign="top"><a href="assets/environments/burgers.png"><img src="assets/environments/cards/burgers.svg" width="240" alt="Burgers: Burgers field trajectory" /></a><br/><a href="benchmarks/nod/README.md">Burgers</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/fhn.png"><img src="assets/environments/cards/fhn.svg" width="240" alt="FHN: FitzHugh–Nagumo field evolution" /></a><br/><a href="benchmarks/nod/README.md">FHN</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/swimmer.png"><img src="assets/environments/cards/swimmer.svg" width="240" alt="Swimmer: Physics demonstration" /></a><br/><a href="benchmarks/swimmer/README.md">Swimmer</a></td>
</tr>
<tr>
<td width="33%" align="center" valign="top"><a href="assets/environments/overcooked.png"><img src="assets/environments/cards/overcooked.svg" width="240" alt="Overcooked: Overcooked environment replay" /></a><br/><a href="benchmarks/overcooked/README.md">Overcooked</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/baxter.png"><img src="assets/environments/cards/baxter.svg" width="240" alt="Baxter: Recorded tactile signals" /></a><br/><a href="docs/datasets.md">Baxter</a></td>
<td width="33%" align="center" valign="top"><a href="assets/environments/rh20t.png"><img src="assets/environments/cards/rh20t.svg" width="240" alt="RH20T: RH20T dataset example" /></a><br/><a href="benchmarks/rh20t/README.md">RH20T</a></td>
</tr>
<tr>
<td width="33%" align="center" valign="top"><a href="assets/environments/pendulum.png"><img src="assets/environments/cards/pendulum.svg" width="240" alt="Pendulum: Torque demonstration" /></a><br/><a href="benchmarks/baseline_adapters/README.md#pendulum">Pendulum</a></td>
</tr>
</table>

Ten environment previews and three CoPhy implementation links. Select an image
to view its preview, or an environment name to open the code guide.
See [image notes](docs/gallery.md) and
[media credits and licenses](docs/media-credits.md).

## Find an experiment

| Setting or component | Implementation and guide |
|---|---|
| D-Clean | [Core simulators and training](benchmarks/core/README.md) |
| PokeWorld | [Core](benchmarks/core/README.md), [revision source](benchmarks/pokeworld/revision/README.md) |
| SpringWorld | [Native source](benchmarks/springworld/README.md), [formation and use](benchmarks/formation_use/README.md) |
| CoPhy Collision, Balls, Blocktower | [CoPhy implementation and adapters](benchmarks/cophy/README.md) |
| Burgers / FHN | [NOD adaptation and FHN solver](benchmarks/nod/README.md) |
| Articulated Swimmer | [Simulation, source fitting and prospective intervention](benchmarks/swimmer/README.md) |
| Overcooked | [Cooperative control and native adaptation](benchmarks/overcooked/README.md) |
| Pendulum | [CaDM and comparison adapters](benchmarks/baseline_adapters/README.md) |
| Baxter / RH20T | [Real-data interfaces](benchmarks/core/README.md), [RH20T preprocessing](benchmarks/rh20t/README.md), [datasets](docs/datasets.md) |
| CaDM / GEPS / CoDA | [Benchmark adaptations](benchmarks/baseline_adapters/README.md) |
| DALI / FCRL | [Source, common readers and probes](benchmarks/additional_baselines/README.md) |

For scientific choices, see [protocols](docs/protocols.md), [paper recipes](docs/paper_recipes.md), [hyperparameters](docs/hyperparameters.md), and [interfaces](docs/interfaces.md). Separate source snapshots preserve the import layouts used by their experiments; the benchmark READMEs identify the relevant entry points.

## Release scope

Version **0.2.0** is the public source release accompanying [arXiv:2609.34604](https://arxiv.org/abs/2609.34604). It builds on the verified 0.1.2 source snapshot with public documentation, citation metadata, and stable project links. Training code, experimental configurations, and scientific results are unchanged by this publication update. The gallery keeps CoPhy as text-only implementation links.

The package contains code, configuration, tests, and source provenance. Generated simulation data can be recreated with the included generators. External datasets and dependencies are acquired through their official sources; trained checkpoints and historical training logs are not bundled. Some frozen-checkpoint analyses require those separately specified inputs. The [reproduction guide](docs/reproduce.md) explains this scope, and [verification](docs/verification.md) records the checks actually performed. The recipe document identifies its manuscript snapshot; this release does not claim a new rerun or a complete audit of every number in a later manuscript revision.

## Repository layout

```text
src/            Models, objectives, simulators and readers
benchmarks/     Environment-specific implementations and guides
experiments/    Preserved source and evaluation snapshots
configs/        Core defaults and protocol configuration
docs/           Reproduction, interfaces, protocols and verification
examples/       Small runnable demonstrations
tests/          Implementation and protocol checks
scripts/        Dependency preparation and release verification
third_party/    Pinned external sources and license notices
assets/         Project wordmark and environment previews
releases/       Source-file integrity manifest
```

## Citation

```bibtex
@article{dai2026sprii,
  title   = {Shaping Persistent Representations from Independent Interactions},
  author  = {Dai, Ji and Fang, Quan and Gao, Junyu and Guo, Rongfeng and Rong, Haoyan and Huang, Yiping and Li, Yongxi},
  journal = {arXiv preprint arXiv:2609.34604},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.34604},
  url     = {https://arxiv.org/abs/2609.34604}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

Original contributions are provided under the [MIT license](LICENSE), with component-specific exceptions. CoPhy-derived code retains GPL-3.0, persistbench retains Apache-2.0, and NOD-derived components retain their documented upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and [media credits](docs/media-credits.md) for the corresponding attribution, dependencies, and media terms.
