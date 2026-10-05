# External datasets and preparation

External data is obtained from the original publishers. This repository does
not mirror data archives, raw participant recordings or pretrained third-party
weights. The root code license does not replace the terms of those resources.
The dataset and media license statements below were rechecked on 2026-09-26.

## Baxter tactile hardness

The Baxter experiment uses the tactile grasp-peak dataset associated with
*Embedded real-time objects hardness classification for robotic grippers*.
The official [Zenodo V1.0 record, DOI 10.5281/zenodo.18246104](https://zenodo.org/records/18246104)
provides the source ZIP and links to the
[publisher's repository](https://github.com/cosmiclab-unige/Embedded-real-time-objects-hardness-classification-for-robotic-grippers).
Use that versioned archive, rather than a newly generated ZIP of the current
repository branch.

The official V1.0 record credits Youssef Amine, Christian Gianoglio and
Maurizio Valle and declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This source distribution links to the archive rather than mirroring the full
dataset. The included tactile preview is a derived visualization with its
source, attribution and transformations documented in [media credits](media-credits.md).
The dataset and its preview are not relicensed by the root code license.

The included SPRII loader registers only 80-by-16 tactile windows: cube and
cylinder configurations at shared hardness levels 0, 1 and 2. It uses 170 grasp
indices per configuration and a common split of 110 training, 30 validation and
30 confirmation indices. The 40-sample files are alternate windows and must
not be counted as independent interactions. These constraints are implemented
in `persistent_jepa.baxter_data` and `prepare_baxter_a1.py`.

After downloading the official archive as `data/baxter/source.zip` and
extracting it under `data/baxter/extracted`:

```bash
python benchmarks/core/scripts/prepare_baxter_a1.py \
  --source-archive data/baxter/source.zip \
  --extracted-root data/baxter/extracted \
  --output-dir data/baxter/prepared
python benchmarks/core/scripts/train_baxter_a1.py --help
python benchmarks/core/scripts/evaluate_baxter_a1.py --help
```

Preparation checks the frozen archive SHA-256
`a7d3782b29df55d46313ca0d3a9b1bd37a400fc15bcb4af12fc6072533f2643c`,
validates each input window and writes manifests plus training-only
normalization. The evidence concerns held-out grasp interactions from six
known configurations; it does not establish control performance or unseen
physical-configuration generalization.

## RH20T

Obtain RGB, low-dimensional robot signals and the corresponding calibration
files from the [official RH20T download page](https://rh20t.github.io/).
The [official RH20T API](https://github.com/rh20t/rh20t_api) documents video
extraction, timestamp alignment, coordinate conversion and scene loading.
Choose a consistent configuration and data-resolution release rather than
mixing archives across formats.

The [publisher's license statement](https://rh20t.github.io/) divides
the data by episode scene identifier: scenes `0001`–`0005` use CC BY-SA 4.0;
scenes `0006`–`0010` are described as CC BY-NC 4.0, but the linked license is
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Preserve
the ShareAlike condition of the linked license and consult the publisher if
that text/link discrepancy affects the intended use. The publisher excludes commercial
use of models trained on the non-commercial subset. Its API code is MIT, which
does not change these dataset terms. The publisher asks users to avoid sharing
sensitive participant imagery or audio and limits use to model training.

`persistent_jepa.rh20t_data.RH20TSplit` consumes per-episode NPZ caches together
with split, pairing and normalization manifests. Required cached fields include
`rgb_gray`, `ft_base_zeroed`, `tcp_base` and `gripper_command_width`. The original raw-data inventory, decoding, TCP sanity filtering and task-split
chain is included in [the RH20T preparation guide](../benchmarks/rh20t/README.md).
Install `requirements/rh20t.txt` for OpenCV. The additional metadata and
evaluation entrypoints include:

```bash
python benchmarks/core/scripts/prepare_rh20t_a2_metadata.py --help
python benchmarks/core/scripts/compute_rh20t_normalization.py --help
python benchmarks/core/scripts/prepare_rh20t_a2_evaluation_bank.py --help
python benchmarks/core/scripts/train_rh20t.py --help
python benchmarks/core/scripts/evaluate_rh20t_a2_condition.py --help
```

The metadata tool requires an eligible-episode list, split manifest and source
inventory. The normalization tool reads training episodes only. Evaluation-bank
construction preserves the registered task/episode relations and hard-negative
conditions. Follow the raw-data preparation guide before these commands. The fixed cfg1
preprocessing uses the registered camera, 96-by-96 grayscale observations,
causal command alignment and the 10-metre TCP corruption rule. It requires the
original eligible task population and fails if that population changes. Exact
published-run cache and split identities remain required for a frozen-checkpoint
comparison. A newly prepared corpus must be documented as a new corpus rather
than assigned the original hashes.

## CoPhy

The [official CoPhy repository](https://github.com/fabienbaradel/cophy) links the
224-by-224 dataset and official split archive on
[Zenodo record 3674790](https://zenodo.org/records/3674790). The repository's
published file links are [the CoPhy-224 data archive](https://zenodo.org/record/3674790/files/cophy_224.tar.gz?download=1)
and [the split archive](https://zenodo.org/record/3674790/files/splits.zip?download=1).
The [official project page](https://projet.liris.cnrs.fr/cophy/) provides the
benchmark description. These are upstream download links, not mirrored copies.

CoPhy source code is [GPL-3.0](https://github.com/fabienbaradel/cophy/blob/master/LICENSE),
retained in `benchmarks/cophy/LICENSE`. The dataset is a separately hosted
resource; this source release does not infer a dataset license from the code
license. Consult the hosting record's current terms. The hosting record could
not be fully retrieved during this release check, although the official
repository's archive links were verified. No replacement license is assigned.

Extract data retaining `CoPhy_224/ballsCF`, `blocktowerCF` and `collisionCF`.
Place the official split files under
`benchmarks/cophy/code/dataloaders/splits`. Obtain the corresponding pretrained
visual front-end files from the upstream repository's
[`ckpts` directory](https://github.com/fabienbaradel/cophy/tree/master/ckpts).
The registered source baseline is commit
`170fe463c9e80ddf0978110438a710cabb4997e2`.

```bash
python benchmarks/cophy/prepare_runtime.py --output runs/cophy
python benchmarks/cophy/code/cophy_prepare_artifacts.py --help
python benchmarks/cophy/code/derendering/extract_object_visual_properties.py --help
python benchmarks/cophy/code/cf_learning/main.py --help
python benchmarks/cophy/code/cophy_evaluate.py --help
```

Feature preparation verifies raw-field audits, archive/split identities,
front-end weights and scene-specific qualification inputs before producing
training artifacts. Supply those inputs explicitly; existing qualification
checks are not bypassed by the source release. The three registered scene
profiles are Balls with four objects, Collision with the normal split, and
Blocktower with three objects and the normal split. Keep these choices, fixed
splits and prediction dimensions aligned with the intended experiment.

The upstream generator covers the published Blocktower example workflow; it
must not be described as a complete generator for every scene. The included
adapters use the official data interfaces for all three scenes.
