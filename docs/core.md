# Core method, simulators, and evaluation interfaces

This source release contains the method implementation, simulation code, data
preparation, training, evaluation, persistent-parameter probes, and intervention
analyses. Training output, experiment logs, private infrastructure configuration,
and historical execution receipts are not distributed. Running an experiment
creates its own configuration and integrity records.

## Source map

| Directory | Contents |
| --- | --- |
| `src/persistent_jepa` | D-Clean and PokeWorld simulators; encoder/predictor; SIGReg and VICReg; pairing; probes; Baxter and RH20T interfaces |
| `src/sprii_next` | Frozen-source readers, decode/inject intervention, physical-parameter probes, geometry, paired statistics, contrastive adaptation |
| `src/persistbench` | SpringWorld simulator, observation interface, data preparation, training contracts and evaluation |
| `benchmarks/core/scripts` | Original executable preparation, training, evaluation, and analysis entrypoints |
| `benchmarks/springworld/native` | Isolated SpringWorld source snapshot with its original package layout |
| `benchmarks/formation_use/scripts` | Spring source fitting, feature export and reader grids |
| `benchmarks/cophy` | CoPhy front-end, P/T adapters, multiple learner families, training, readouts, probes and final-test protocol |

The original Python package names are retained. The isolated Spring snapshot is
intentional: its source-binding checks compare imported modules with the source
files committed in a feature descriptor. Keep that snapshot together. Revised
PokeWorld experiments use the separate revision source snapshot; the core
PokeWorld entrypoints do not by themselves replace its factorized protocol.

## Installation

Python 3.10 or later is required by the core package. From the repository root:

```bash
python -m pip install -e '.[train,dev]'
python -m sprii_next --help
python -m pytest -q tests/core tests/formation_use
```

For script-based environments, `requirements/core.txt` is also provided;
set `PYTHONPATH` to the repository’s `src` directory in that setup.

For CoPhy, install `requirements/cophy.txt`. Install PyTorch and TorchVision
builds compatible with the desired CPU or CUDA platform. Dataset and model
weights must be acquired separately; a source release is not a weights release.
The minimum dependency versions express API requirements, not a claim that every
combination has been tested.

## Model and batch contracts

The common model configuration is in `persistent_jepa.model.ModelConfig`:
observation width 128, transient width 64, persistent width 64, Transformer width
192, four blocks, eight heads, dropout 0.1, horizon embedding width 32, and a
24-frame history. Individual benchmark adapters override this configuration.

D-Clean `PairedBatch` contains state history `[B,24,4]`, action history
`[B,23,2]`, targets `[B,3,4]`, future actions `[B,3,16,2]`, and action masks
`[B,3,16]`. The three prediction horizons are 1, 4, and 16. PokeWorld uses visual
current/previous observations, paired physical-system and rollout identifiers,
and the same masked action-horizon convention. Its legal anchors are from the
history length through frame 47: the temporal-difference channel also requires
the preceding frame.

SpringWorld's native source uses 96 observed frames. The strict visual adapter
validates image, action, and mask shapes and only updates running observation
statistics from training history. Future targets do not supply normalization
statistics. The native history batching code and its tests are included.

CoPhy separates `ABObservation`, `VisualInput`, `Targets`, and `DonorPairs`.
Physical labels enter supervised references and probes through explicit
interfaces. The P/T adapter divides the 32-dimensional code into 16 persistent
and 16 transient coordinates; donor injection replaces only the persistent
coordinates while retaining recipient transient state and current observation.
See `benchmarks/cophy/code/cophy_adapter.py` for the complete typed contract.

## Protocols and hyperparameters

Executable defaults are defined in the dataclasses and argument parsers. The
machine-readable snapshot `configs/core/defaults.json` records these defaults
and their source locations. Defaults alone do not establish a paper run: use
its benchmark-specific configuration and declared seed/selection protocol.

The Spring native source recipe is provided in
`benchmarks/springworld/native/NIGHT_POLICY.json`: 10,000 updates, 48 pairs per
batch, history 96, AdamW learning rate 0.0003, weight decay 0.05, warmup 500,
gradient clip 1, and CUDA bfloat16. Seeds are independent model, sampling and
stochastic fields. The source-repeat entrypoint sets all three to the requested
seed from 0, 1, and 2.

The frozen-source reader protocol in `sprii_next.protocol` uses 10,000 updates,
batch 256, AdamW learning rate 0.0003, weight decay 0.05, warmup 500, gradient
clip 1 and FP32. Sources remain frozen. Spring readers compare Null, Persistent,
Decode and Oracle with three source seeds and three reader seeds. Decode uses a
training-only ridge map from P64 to standardized log physical parameters;
Oracle uses the same parameter coordinates and moments. The primary Spring
endpoint is 600 cases from 100 physical systems. Aggregation preserves paired
case identities and averages by physical system.

All test boundaries are explicit. Development commands reject sealed-test
inputs. Model/decoder selection belongs to the declared training/validation
protocol. Sealed evaluation has a separate immutable admission interface. Do
not rename a development result into a test result or treat a smoke run as a
scientific result.

## Verification scope

Unit and synthetic integration tests exercise deterministic simulation, legal
pairing, target and mask support, objective gradients, source/reader contracts,
parameter probes, aggregation, and source-code identity checks. They do not
substitute for full training on each benchmark. Optional native-source tests
must be given the included source snapshot as documented in the SpringWorld
README. GPU training and historical checkpoints have not been rerun as part of
building this source archive.

## License boundaries

The repository root license applies to original code except where a more
specific license is included. `src/persistbench` and the native Spring
`src/persistbench` snapshot retain Apache-2.0. `benchmarks/cophy` retains
GPL-3.0 and its third-party origin. Public third-party attribution is preserved;
it does not identify the authors of this submission.
