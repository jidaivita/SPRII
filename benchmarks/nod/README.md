# NOD, Burgers and FHN

This directory contains the clean NOD training and evaluation adaptation, SPRII relation objectives, Burgers pairing, and the Python FHN solver and matched hierarchical training implementation. The model and data interfaces remain separate from the core world-model package.

## Dependencies and public inputs

Use Python 3.10 or later, NumPy, SciPy, h5py, PyTorch 2.4 or later, and tqdm. FHN data generation can use NumPy on CPU or PyTorch on GPU. Full training requires substantially more memory and compute than the solver tests.

From the repository root, install the training dependencies and fetch the pinned
NOD source:

```sh
python -m pip install -e '.[train]' h5py tqdm
python scripts/fetch_external_sources.py nod
```

The NOD code is [Zenodo release 20406332](https://doi.org/10.5281/zenodo.20406332), archive SHA-256 `eacafe88ea61e3716a006e3123a49669284a23a61fec70873d9216738ba508e7`. It is attributed to **Zituo Chen** and provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The clean trainer, adapted loader, and FHN port build on that release; their changes are the train/validation separation, equal-access relation objectives, path normalization, standalone evaluation, and NumPy/PyTorch solver interface. The unchanged upstream neural-network modules are fetched rather than bundled. These upstream-derived files retain their CC BY attribution; the root MIT license does not replace it.

Obtain the Burgers archive from the [public dataset release](https://doi.org/10.5281/zenodo.20372988), also CC BY 4.0. Its SHA-256 is `c14dc62c1d5c2e8d432a5684df824950b3601d4e9c5122d45abeec2085178414`. After extracting it under `data/burgers_release`, normalize the public filenames without rewriting data:

```sh
python benchmarks/nod/src/nod_sprii/prepare_burgers_paths.py \
  --release-root data/burgers_release --out-root data/burgers_normalized
```

## Burgers interfaces and protocol

`src/nod_sprii/train_nod_clean.py` is the shared clean training loop. `train_sprii_clean.py` delegates to that same loop. The relation controls change `--lambda_align` and `--random_relation`; the base prediction route and observation access remain shared. The model predicts 101 frames at 401 spatial points, with 8,192 sampled `(t,x)` query points during training. `MODEL_DEFAULTS` contains the full branch/trunk widths and Fourier-feature settings.

Training cases are 0–39, validation cases 40–44, and final ID cases 45–49. The supplied public archive lacks the viscosity 0.003 truth shard; the common available evaluation manifest is used instead of pretending it is the full upstream set. `configs/burgers_formal_manifest.json` specifies seeds, budgets, relation controls, and the validation-only hyperparameter rule.

```sh
python benchmarks/nod/src/nod_sprii/train_nod_clean.py \
  --data_root data/burgers_normalized/train \
  --output_add_root data/burgers_normalized/train_add \
  --save_dir runs/burgers/native --seed 1234 --device cuda \
  --lambda_align 0
python benchmarks/nod/src/nod_sprii/evaluate_frozen.py \
  --checkpoint runs/burgers/native/best_eval.pth \
  --train-root data/burgers_normalized/train \
  --truth-root data/burgers_normalized/truth \
  --output runs/burgers/evaluation.json --deterministic-pairing
```

Use each entrypoint's `--help` for the full CLI. `equivalence_test.py` and `equivalence_real.py` check the zero-alignment path; `triple_data.py` defines equal-access A/B/C sampling.

## FHN generation and training

The current FHN comparison uses source seeds **42, 43 and 44**, with NOD ending
at **45,000** updates and SPRII at **50,000**. Start with the selected recipe in
[configs/fhn_selected_three_seed.json](configs/fhn_selected_three_seed.json)
and [the continuation guide](src/fhn_continuation/README.md). Seed 42 selected
the recipes using ID initial condition 5; seeds 43/44 repeat the selected stages
without further checkpoint selection. Final report recipients are 15, 24 and 45.

The following command only prints the selected stage commands. It does not load
data or start training; execution requires adding `--execute` after preparing
the data and upstream source:

```sh
python benchmarks/nod/src/fhn_continuation/replay_selected.py \
  --method sprii --seed 43 --data-dir data/fhn \
  --official-code benchmarks/nod/third_party/nod_original/code/DR2D \
  --output runs/fhn/sprii_seed43
```

### Generate the FHN data

The solver models a two-species diffusion-reaction field on a periodic 128×128 grid, uses RK4 with `dt=0.001`, runs 10,000 integration steps, and retains every 100th step. The default solver dtype is float64 and saved dtype is float32. The spatial stencil and random-field construction are covered by `src/fhn_python/test_solver.py`.

The FHN-only requirements below supplement the shared installation above; they
do not include the Burgers loader's h5py and tqdm dependencies.

```sh
python -m pip install -r benchmarks/nod/requirements-fhn.txt
python benchmarks/nod/src/fhn_python/test_solver.py
python benchmarks/nod/src/fhn_python/generate_fhn.py \
  --output-dir data/fhn --split all --initial-ids 50-89 \
  --seed 42 --backend torch --device cuda
```

Generate evaluation initial conditions separately using `--initial-ids 5,15,24,36,45` with the same solver and parameter grids. The historical [configs/fhn_minimal_closure.json](configs/fhn_minimal_closure.json) retains the train/interpolation/extrapolation parameter grids and the one-seed precursor. Use the selected three-seed recipe above for the current training schedule. ID 5 is reserved for selection; final ID targets are 15, 24 and 45, with conditioning ID 36. Do not use the CLI's broader diagnostic target list for the final report.

Six completed historical run records were checked during packaging; full-budget
training was not rerun for this distribution.

The current comparison also includes matched conditioning-history budgets K=1,2,4 and training-only parameter probes. It does not supply a matched Random relation control or the full fixed-predictor mechanism suite. `tune_train.py`, `select_validation.py`, and `evaluate.py` remain the shared lower-level implementation; the later LR-restart and multi-history/report-recipient code is included in `src/fhn_continuation/`.
