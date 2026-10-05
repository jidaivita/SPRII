# Additional baseline adaptations

This directory releases the actual DALI context/forward-model adaptations for D-Clean and CoPhy, and the FCRL paper implementation for D-Clean. The DALI adapters transfer released components to the common prediction task. They do not implement the complete Dreamer actor, critic, RSSM, or native control-return evaluation. FCRL is a paper adaptation, not an untouched upstream reproduction.

## Requirements and external sources

Use Python 3.10 or newer, PyTorch, NumPy, and SciPy. Training and the original common readers expect CUDA; the standalone component smoke check runs on CPU. The continuation tools use POSIX file locks and signals. From the repository root, install `requirements/core.txt` and install the project with `pip install -e .`.

DALI upstream is [frankroeder/DALI](https://github.com/frankroeder/DALI), pinned to commit `34374fbea258748b03e31977a68693ab040fab72`. Clone that version with its required `dreamerv3_compat` contents into this directory's `vendor/DALI`, or pass `--official-root`. The adapter validates the upstream `dreamerv3_compat/dreamerv3/nets.py` SHA-256 `0233a0d7aedce5f137a29a5ce3b49d100db1c0d1322d1c8b6238e66235539b9d`. The preserved [DALI license](DALI-LICENSE) and [Dreamer license](Dreamer-LICENSE) retain their upstream terms and attribution.

The shared D-Clean training/comparison protocol requires the released NOD
`code/MassSpring/model.py` identity even when fitting FCRL. Obtain it through the
[NOD data/source guide](../nod/README.md), which runs
`python scripts/fetch_external_sources.py nod` from the repository root. Set
`SPRII_NOD_MODEL` to that fetched file as shown below. The standalone CPU smoke
does not require this upstream source. `SPRII_NATIVE_MODEL` defaults to this
repository's `src/persistent_jepa/model.py`; its exact model identity is checked
by the shared reader.

## Layout and recipes

| Location | Purpose |
|---|---|
| `dclean/dali_context_torch.py` | PyTorch implementation of the transferred context and forward components |
| `dclean/dali_dclean_source.py` | Initial fixed-budget D-Clean DALI source fitting |
| `dclean/dali_dclean_common_reader.py` | Frozen-source common reader, cache, probes, and donor intervention |
| [dali_dclean_fixed40k.py](dclean/dali_dclean_fixed40k.py), [dali_dclean_fixed80k.py](dclean/dali_dclean_fixed80k.py), [dali_dclean_fixed160k.py](dclean/dali_dclean_fixed160k.py) | Preserved continuation and reader stages |
| [dali_dclean_80k_selection.py](dclean/dali_dclean_80k_selection.py), [dali_dclean_160k_selection.py](dclean/dali_dclean_160k_selection.py) | Selection-only budget checks with fixed parent identities |
| `shared/dclean_external.py` | D-Clean data/pairing contract, FCRL implementation, and NOD component adapter |
| `shared/dclean_panel.py` | Shared frozen-code reader, parameter probe, and fixed-donor intervention |
| `cophy/models.py`, `cophy/train_source.py` | Context8 from complete RGB-feature histories, source loss and fitting |
| `cophy/build_reader.py` | Exact transformation of the common CoPhy reader for the DALI role |
| `cophy/dclean_budget_sensitivity.py` | D-Clean continuation, selection evaluator, and exact resume smoke check |

[paper_recipes.json](paper_recipes.json) records selected budgets and hyperparameters. The current D-Clean DALI source ends at **160,000 updates**, after 20k/40k/80k stages. Sources use seeds 0/1/2; common readers use 20,000 updates and seeds 0/1/2. FCRL sources use 20,000 updates, latent dimension 50, and a transition-set mean-pooling encoder with an InfoNCE objective. CoPhy DALI uses 50 source epochs, batch 32, context dimension 8 padded to the common 128D representation, and a separate common reader. These source budgets are not all compute-matched.

## Running

Run commands below from `benchmarks/additional_baselines`, so the documented relative `shared/`, `data/`, and `runs/` arguments resolve consistently. Set `SPRII_DCLEAN_DATA` to the prepared D-Clean bank and `SPRII_DCLEAN_ROOT` to the desired comparison working directory (default `runs/dclean_controls`). The original data split and array hashes are enforced. See the [core benchmark](../core/README.md) for environment/data generation; an arbitrary new bank is not interchangeable with the frozen reported bank.

After fetching NOD, configure its model path from this benchmark directory:

```sh
export SPRII_NOD_MODEL="$PWD/../nod/third_party/nod_original/code/MassSpring/model.py"
```

```sh
python smoke.py
python shared/dclean_external.py --help
python dclean/dali_dclean_source.py --help
python dclean/dali_dclean_common_reader.py --help
python dclean/dali_dclean_fixed160k.py --help
```

`shared/dclean_external.py train --method FCRL --seed 0` fits an FCRL source using the declared data contract. The original full-data helper smoke also checks NOD and requires its upstream model and CUDA. `dali_dclean_source.py` accepts the input helper, official DALI source, seed, output, and update budget; use its explicit arguments. The reader runs `inspect`, `smoke`, `cache`, `fit`, `evaluate`, `probe`, and `finalize` phases on that frozen source. Preserve separate source and reader output directories for every seed.

The 40k/80k/160k tools are scientific continuation stages. They require completed ancestor checkpoints, selection decisions, normalizations, and the recorded output schema. Their checkpoint/selection hashes deliberately remain fixed. They are not fresh-run launchers, and the source archive does not ship historical weights, logs, or result receipts. Machine-specific queues and archive/monitor operations have been excluded; only source and reader entry points remain. Use the initial source and selection stages to understand the chain; do not bypass the ancestor checks or claim a fresh run is the original executed chain.

For CoPhy, prepare frozen visual features with the [CoPhy pipeline](../cophy/README.md), then invoke `cophy/train_source.py --features PATH --out PATH --epochs 50 --seed 0 --device cuda:0`. The feature cache must match the expected scene/frame/object layout. Generate its reader wrapper adjacent to the common reader, preserving sibling imports:

```sh
python cophy/build_reader.py \
  --core ../cophy/code/cophy_complete_v7/common/readout.py \
  --out ../cophy/code/cophy_complete_v7/common/dali_readout.py
```

Pass the generated reader the prepared base/features, DALI source checkpoint, this directory's `cophy/models.py`, and the prescribed source/reader budget. The transformation checks exact replacement counts and AST equality for the shared head, pairing and normalization code. Its parent hash is rebound to the published common reader after path anonymization; this is a release dependency binding, not a historical execution hash.

## Verification

The CPU smoke exercises real component forward/backward computation, finite DALI/FCRL losses, CoPhy context shape and zero padding, bitwise Adam/RNG continuation, and the selection evaluator's pair ordering and raw-MSE reduction. It reads no dataset or historical result. Public implementation-dependency hashes were rebound after path changes; dataset, checkpoint, and selection identities were retained. Full training and reported numerical results were not rerun as part of packaging.
