# External baseline adapters

These are the actual model-facing adaptations used for CaDM, GEPS and CoDA comparisons. They retain distinct entrypoints and explicit budgets; they are not interchangeable baseline implementations. The `pilot` suffix is retained where a final runner imports its model/data components.

## Public source dependencies

Pinned upstream URLs, commits and license status are in `sources.json`. From the
repository root, install the adapter dependencies and fetch the immutable public
versions. Full third-party repositories are not included in this release:

```sh
python -m pip install -e '.[train]' h5py tqdm torchdiffeq einops
python scripts/fetch_external_sources.py cadm gym geps coda
```

Run commands from the repository root. Use Python 3.10+, NumPy, SciPy, h5py, PyTorch, torchdiffeq, einops, and the pinned upstream requirements when needed. CaDM Pendulum is a deterministic ensemble-size-one PyTorch adaptation; it does not require the original TensorFlow training stack. Its environment equivalence check executes the upstream Pendulum methods through dependency stubs.

## Pendulum

The observation is `(cos(theta), sin(theta), angular_velocity)`, action is normalized to `[-1,1]`, and context uses 10 observed transitions. Mass and length are hidden model parameters. Training uses 20 iterations × 10 episodes × 200 transitions = 40,000 interactions. Evaluation uses the fixed ID grid and four OOD groups, 10 episodes per group, totaling 10,000 interactions per method. CEM uses horizon 30, 200 candidates, 50 elites and five iterations.

```sh
python benchmarks/baseline_adapters/cadm_pendulum_engineering_gate.py validate \
  --output runs/pendulum/environment_check
python benchmarks/baseline_adapters/cadm_pendulum_online.py run \
  --method CaDM --profile formal --seed 0 \
  --output runs/pendulum/cadm_seed0 --device cuda:0
python benchmarks/baseline_adapters/cadm_pendulum_relation_components_v5.py run \
  --arm SPRIIAlign --profile formal --seed 0 --align-weight .003 \
  --relation-variant variance01 --output runs/pendulum/sprii_seed0 --device cuda:0
```

`cadm_pendulum_relation_confirmation_v4.py` retains the fixed initial Align recipe. `cadm_pendulum_relation_components_v5.py` exposes the variance-target/component recipe used by the later complete cohorts. They must be reported as separate complete recipes, without selecting different recipes per seed or OOD group. `cadm_pendulum_postrun_analysis.py` performs parameter probes and frozen donor substitutions on the common bank; `pendulum_id_control_development.py` defines development control evaluation. Seeds are 0, 1 and 2. Model widths, optimizer settings and fitting schedules are explicit in the source and the CLI reference.

## D-Clean

`cadm_dclean_source_v2.py` implements the CaDM adaptation and `cadm_dclean_common_reader_v2.py` supplies the shared reader and parameter-probe protocol. Their [shared D-Clean helpers](../missing_blocks/README.md) retain the data and reader contracts. A sample contains 24 state frames, 23 observed actions, and matched future actions; states have dimension 4 and actions dimension 2. Data normalization uses training examples only. The immutable generated D-Clean dataset must be supplied at `data/dclean`, or with `--data-root` where supported. The helpers preserve dataset fingerprint checks.

## Burgers: GEPS and CoDA

The NOD public-data setup is documented in `../nod/README.md`. GEPS uses the published Euler forecaster, code dimension 4, learning rate 0.01, four trajectories per update, and 50 code-only adaptation steps at 0.01. An external deterministic initialization repair zeros published uninitialized `Swish` context parameters; the upstream files themselves are unchanged. Formal training uses 5,000 updates and seeds 1234, 5678, 9012.

First create the required data binding without training:

```sh
python benchmarks/baseline_adapters/geps_burgers_pilot.py --mode inspect \
  --nod-source benchmarks/nod/src/nod_sprii \
  --geps-source benchmarks/baseline_adapters/third_party/geps_original \
  --data-root data/burgers_normalized/train \
  --output benchmarks/baseline_adapters/geps_pilot_retry1
```

Then `geps_formal_three_seed.py --seed SEED --gpu GPU --output OUTPUT` trains and evaluates a complete fixed-budget run. The formal script verifies the data-binding metadata produced by the inspect step. `geps_formal_mechanisms.py` implements post-training accessibility and frozen-code substitution.

CoDA uses the upstream grouped convolution/hypernetwork with a singleton spatial axis, RK4, hidden width 64, context width 2, learning rate 0.001, and code-only Adam adaptation for 50 steps at 0.001. The physical time scale is the NOD normalized 101-frame horizon. This is an explicit CoDA-to-Burgers adaptation: the public CoDA repository has no official Burgers configuration. `coda_burgers_components.py` loads the numerical upstream definitions; `coda_formal_three_seed.py` and `coda_postrun_mechanisms.py` implement the fixed training and diagnostics.

The frozen formal CoDA replay retains the original seed-1234 continuation/checkpoint guards. A fresh arbitrary run cannot satisfy the historical checkpoint fingerprint merely by renaming files. These source-replay dependencies, and the fixed decoder encoder experiments, are documented in `../../experiments/coda_frozen/README.md`. No weights, generated data, or historical logs are bundled.

## Outputs and verification

Source scripts naturally generate metrics, checkpoints and run metadata when executed. Those user-generated artifacts are not part of this source release. Input shape, trajectory pairing, equal-access controls, immutable decoder checks, and source parameter checks remain enabled. Syntax validation is not a claim that every GPU training recipe was rerun during packaging.
