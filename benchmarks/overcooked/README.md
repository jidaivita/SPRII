# Persistent representations in Overcooked

This source package contains the native paired-history learner, the original
A/VC and Random implementations, partner training and history generation,
fixed-policy evaluation, frozen representation probes, and external-history
interventions. The environment is the upstream `grounded_coord_simple` layout.
The final history-budget study uses 20 fixed training policies and two development
policies from one excluded source population. It is not the full upstream
80-policy benchmark, and the two reserved policies are not two independent
source seeds.

## Install and inspect

Run the following from `benchmarks/overcooked` using Python 3.11:

```sh
python -m pip install -r requirements.txt
python scripts/bootstrap.py --destination external/icrl4aht --variant matched
python scripts/smoke_numerics.py
```

The bootstrap verifies the pinned upstream archive, applies two engineering
changes (integer slice bounds and completion before timing), and installs our
additions into the upstream tree. For GPU training, install the compatible
`jax[cuda12]==0.5.3` distribution in the same environment. The native environment,
PPO, baseline model, and teammate generators come from the fixed upstream
revision in `upstream.json`. The bootstrap does not start training. To use the
Random control, create a separate tree with `--variant random`; do not mix its
training module with the matched variant.

The files under `variants/` are the actual source snapshots. All ten source
hashes in each released training configuration were checked against the
completed experiment's configuration. A/VC and Random have distinct snapshot
bindings. The portable wrappers only choose paths, inputs and devices.

## Reconstruct the data and train

The source population is seeds 4600, 4601, and 4700--4707; seed 4602 is reserved
for development. Each IPPO source uses 30 million requested transitions,
457 PPO updates, and five checkpoints. The actual population was initialized
with one CPU update and continued on GPU. The retained drivers preserve that
checkpoint transition:

Run this inspection in a subshell so the working directory remains
`benchmarks/overcooked` for the following commands:

```sh
(
  cd external/icrl4aht || exit
  python -m native_a.partner_run --help
  python -m native_a.gpu_partner_run --help
)
```

Keep each source under a configurable `partners/seed<seed>` directory. Use
`qualify_partners.py` to verify completed sources, require base return strictly
above 20, remove duplicate actual policy tensors, select 20 training policies
with `numpy.default_rng(0)`, and reserve checkpoints 3 and 4 from source 4602.
The selection fails rather than expanding the source pool or lowering the
threshold. Re-training can produce different eligible policies.

```sh
python scripts/qualify_partners.py --repo external/icrl4aht --partners data/partners --out data/run/dataset_inputs
python scripts/histories.py collect --repo external/icrl4aht --run-root data/run --task-index 0 --gpu 0
python scripts/histories.py pack --repo external/icrl4aht --run-root data/run
```

Run collection once for every task index 0--19, then pack. Each task requests
60 million transitions (228 updates, 59,768,832 actual transitions), records
1,024 environments with 100-step episode prefixes, and retains the original
quality-selected 128 histories. Packing preserves explicit episode boundaries;
it does not infer them from concatenated done flags. Put the frozen
`train_manifest.jsonl` next to the packed files in `data/run/dataset`.

```sh
python scripts/subsets.py --repo external/icrl4aht --data data/run/dataset --out data/run/subsets
python scripts/train.py --repo external/icrl4aht --config configs/VC100_seed4200.json --data data/run/dataset --allowlist data/run/subsets/100.json --out runs/VC100_seed4200
```

The fixed budgets are 25/50/100%, or 32/64/128 complete histories per partner.
The subsets are deterministic and nested, retaining the entire recorded time
range of each selected history. The shipped `splits/` files specify the original
identities and selection; generate newly bound allowlists when local paths or
data differ. Each A/VC run uses seeds 4200, 4201 or 4202, 20,000 optimizer updates,
1,024 rows per optimizer batch, and a 500-token budget: two separate 100-token
support episodes followed by a 300-token query. A denotes H+SPRII with mode
`I+VC`; VC denotes H+VC with mode `VC`. Both use relation coefficient 0.001
and Cross weight zero. The 25/50/100 suffix is the percentage of histories
retained. Random changes only the relation donor
through a separate checkpointed RNG and a different-identity permutation.
Its three runs contribute probes; returns were not part of its reported scope.

## Evaluation and probes

`python scripts/evaluate.py --help` describes the frozen-checkpoint evaluation.
It checks partner tensor identities against `qualification.json`, evaluates
20 fresh 100-step episodes for every partner with evaluation seed 920140, and
averages episodes 6--20 within partner before averaging partners. The 20 familiar
policies are primary; the two development policies are separate. The GPU path
uses the original CPU/GPU logit-and-action equivalence gate on training-panel
inputs, then keeps the environment and partner inference on CPU.

The original probe collector is `tools/probe_panel.py`; the sample-plan and
feature extraction functions are in `tools/frozen_probe.py`. They operate on
a newly bound plan containing local manifests, verified source isolation,
collector checkpoint identity and output paths. The common panel has 22
policies x 16 independent episodes, with reset seeds 52000--52007 for fitting
and 53000--53007 for testing. The fixed ego is source 4600 checkpoint 4.
Its common-input policy signatures contain 40 snippets x 8 steps x 6 actions,
using two snippets from each of the 20 training partners. A stale assertion in
the recovered collector was corrected from four to twenty partners to agree
with its existing loop, frozen plan and actual signature dimensions.

`python scripts/probe_arrays.py arrays.npz --out probe.json` applies the original
fixed ridge (0.001) classifier and functional readout. The NPZ keys are
`features` (176 x 32), `labels`, `contexts`, Boolean `fit`/`test`, `roles`
(`train` or `heldout`), and `targets` (22 x 1920). Identity classification sees
all 22 labels; the functional head and its standardizer fit only 20 training
identities. Familiar fresh-episode and heldout errors remain separate. The
frozen encoder receives zero updates. These are policy-function signatures,
not recovery of full neural-network weights.

`tools/history_intervention.py` retains the original matched/null/wrong external
history construction, same-query comparisons, deterministic donor schedule,
CPU/GPU gate, and 20-episode online chains. `tools/evaluate_support.py` retains
the original source/data binding and earlier matched/null-support evaluator.
All are source-level interfaces; the original weights, trajectories, operational
receipts, and training logs are not included. Exact historical numbers require
the original frozen inputs; a newly trained run must be identified separately.

## Validation and licensing

Validation covers Python compilation, pinned-upstream bootstrap and source
hashes, deterministic different-identity pairing, ridge probes, and same-query
history interventions. Full JAX training and the 7,920-episode evaluation have
not been rerun for this release. Linux with the declared JAX stack is required
for the original training and panel collector; the collector uses CPU affinity.

Our additions use the release's MIT license. ICRL4AHT's pinned `pyproject.toml`
declares the MIT classifier but its archive has no standalone LICENSE file.
It is therefore fetched from its original publisher as an external dependency;
we do not add a license text on the publisher's behalf. Upstream files,
transitive environments, and installed libraries retain their own licensing.
No logs, weights, private paths, credentials, or job identifiers are distributed.
