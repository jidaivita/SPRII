# Recipient-specific updates in Articulated Swimmer

This package contains the physical simulator, data generation, predictor and
adapter implementations, frozen-input interventions, and statistical evaluation
used for the Swimmer experiment. Articulated Swimmer is the custom three-link
MuJoCo system in this package; it is not Gymnasium Swimmer-v5.

The published comparison uses 512 prospective physical systems, one selected
context per system, six queries and six candidates. The same own-system or
cell-matched wrong-system persistent update passes through identical frozen
predictor and adapter weights. Three adapter checkpoints are averaged first,
then the 36 rows within each system. Confidence intervals resample physical
systems 4,000 times. The Persistent-JEPA base and the two Masked-GRU bases
(64101 and 64103) remain separate architectures, not three source seeds.

## Installation and a small verification

From the repository root, enter the Swimmer package before installing its
dependencies and running its checks:

```sh
cd benchmarks/swimmer
python -m pip install -e '.[test]'
python -m pytest tests/unit
python scripts/smoke_physics.py
```

Run the remaining commands in this guide from `benchmarks/swimmer`.

Four historical provenance tests are explicitly skipped because their old
protocol and machine-command fixtures are excluded; numerical and simulator
tests remain active. Full fitting and the complete prospective experiment were
not rerun for this release.

The existing `paper_c` Python namespace is retained for import compatibility.
It does not imply that every historical experiment in that research workspace
is included here. All included files are part of the Swimmer dependency closure.

## Entry points

| Operation | Implementation |
|---|---|
| MuJoCo dynamics and parameter sampling | `code/paper_c/swimmer/model.py` |
| Action waveforms | `code/paper_c/swimmer/waveforms.py` |
| Training and selection trajectories | `python -m paper_c.swimmer.s2_data --help` |
| Persistent-JEPA source fitting | `python -m paper_c.swimmer.s2_train --help` |
| Masked-GRU source fitting | `python -m paper_c.stage2.masked_gru_cross_architecture --help` |
| Persistent-update adapter fitting | `python -m paper_c.stage2.delta_gated_isolation --help` |
| Same-checkpoint prospective intervention | `python -m paper_c.extension.d1a_same_checkpoint_prospective --help` |

The JSON files under `configs/` specify the actual system, learner, adapter,
donor, normalization, and bootstrap settings. `d1a_same_checkpoint_prospective_v1.json`
is the final comparison specification. Internal paths are relative to the
directory supplied as `root` to the commands. Before a new freeze, set `execution.run_identity` and a timezone-aware
`calendar_stop` in the final comparison config, and export the same identity
as `SPRII_RUN_ID`. This preserves the run binding without a platform account.
Historical machine aliases have
been replaced by neutral worker labels; platform identifiers are removed.

## Exact-reproduction inputs and execution boundaries

This is a source release. It does not include training logs, model weights,
trajectory arrays, or historical authorization receipts. The archived
multi-stage orchestration deliberately verifies frozen checkpoints and
population provenance before evaluation. It is not a one-command demo, and it
must not be made to pass by inventing historical receipts. Exact numerical
reproduction requires the declared frozen checkpoints and training-only
normalizations, or a separately identified new training run.

The simulation and model modules are directly reusable. The source-generation
and training commands expose their positional inputs through `--help`.
The historical orchestration source is included to specify the actual
intervention and aggregation logic. Its freeze checks document the original
workflow; new experiments should create their own provenance rather than
representing their outputs as the original frozen result.

## Dependencies and license

The package's research code is covered by the release's MIT license. MuJoCo,
PyTorch, NumPy, SciPy, and pandas are installed as external dependencies and
retain their own licenses. No external simulator source, model weights, or
third-party dataset is re-licensed by this package.
