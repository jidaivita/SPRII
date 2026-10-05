# PokeWorld revision implementation

This is the source snapshot used by the factorized PokeWorld relation, geometry and control experiments. It has its own `persistent_jepa` package snapshot because the training/data interfaces are bound to these experiments. Use a fresh Python process and its `src` directory on `PYTHONPATH` rather than mixing it with a different version already imported into the same process.

- `scripts/prepare_pokeworld_factorized.py`: factorized train/validation bank generator, with fixed geometry validation.
- `scripts/train_pokeworld_revision.py`: independent-interaction objectives, relation quality controls and split controls.
- `scripts/evaluate_pokeworld_revision_geometry.py`: frozen parameter-accessibility and representation geometry evaluation.
- `scripts/evaluate_pokeworld_bridge.py`: learned prediction/bridge evaluation.
- `scripts/train_actual_input_certificate.py`: actual-input certificates.
- `src/persistent_jepa/pokeworld.py`: simulation and physical configuration.
- `src/persistent_jepa/poke_torch.py`: state/history tensors and relation samplers.
- `src/persistent_jepa/poke_model.py`: model and objective definitions.

The complete predeclared model, sampling, optimizer, control, geometry, ridge and bootstrap settings are in `PROTOCOL_MANIFEST.json`. Training geometry pairs are not reused as evaluation geometry pairs. Ridge selection uses training-system folds; bootstrap resampling units are physical systems.

```sh
PYTHONPATH=benchmarks/pokeworld/revision/src \
  python benchmarks/pokeworld/revision/scripts/prepare_pokeworld_factorized.py \
  --output data/pokeworld_factorized
PYTHONPATH=benchmarks/pokeworld/revision/src \
  python benchmarks/pokeworld/revision/scripts/train_pokeworld_revision.py --help
```

The structured-error and inferred-relation extensions are in `../../../experiments/poke_controls/`. Do not substitute a display animation for this simulator or its sampled training bank.
