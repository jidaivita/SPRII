# Shared D-Clean compatibility helpers

These modules provide the fixed D-Clean data and reader interfaces used by the
[CaDM source adapter](../baseline_adapters/cadm_dclean_source_v2.py) and
[CaDM common-reader adapter](../baseline_adapters/cadm_dclean_common_reader_v2.py).
The directory name is retained for import and source-binding compatibility.

| Module | Interface |
|---|---|
| [dclean_external.py](dclean_external.py) | Dataset fingerprints, training-only normalization, independent history pairing, source encoders and protocol definition |
| [dclean_panel.py](dclean_panel.py) | Frozen-source code cache, common prediction head, physical-parameter probes and donor interventions |

Start with the [baseline adapter guide](../baseline_adapters/README.md) for
dependencies, data paths and executable entry points. The source adapter's
`--data-helper` and the reader's `--data-helper` / `--panel-module` arguments
already select these modules by default. Dataset and checkpoint identity checks
remain part of the comparison contract.

The [additional DALI/FCRL adapters](../additional_baselines/README.md) retain a
separate `shared/` helper snapshot with their own path and source bindings.
Keep each adapter with its documented helper version.
