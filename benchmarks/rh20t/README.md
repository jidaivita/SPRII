# RH20T raw-data preparation

The original cfg1 preprocessing chain is included in `preparation/`. Obtain the
matching RGB, low-dimensional and calibration archives from the
[official RH20T page](https://rh20t.github.io/), and merge the same
configuration's files into one extracted dataset directory. See
[external datasets](../../docs/datasets.md) for the data-license boundary.

Install `requirements/rh20t.txt`. `ffprobe` is optional for the inventory's video
header diagnostics; OpenCV is required for decoding. Raw files are read locally;
these scripts do not download data or send recordings to any service.

From the repository root:

```bash
python benchmarks/rh20t/preparation/rh20t_inventory_preflight.py \
  --dataset-root data/rh20t/cfg1 --output-dir data/rh20t/inventory
python benchmarks/rh20t/preparation/build_rh20t_cache_v1.py \
  --dataset-root data/rh20t/cfg1 --cache-root data/rh20t/cache \
  --output-dir data/rh20t/prepared_v1 --workers 8
python benchmarks/rh20t/preparation/refreeze_rh20t_after_tcp_audit.py \
  --eligible-manifest data/rh20t/prepared_v1/eligible_episode_manifest.json \
  --cache-root data/rh20t/cache/episodes \
  --output data/rh20t/prepared/eligible_episode_manifest.json
python benchmarks/rh20t/preparation/freeze_rh20t_manifests.py \
  --eligible-manifest data/rh20t/prepared/eligible_episode_manifest.json \
  --output-dir data/rh20t/prepared
python benchmarks/core/scripts/compute_rh20t_normalization.py \
  --cache-root data/rh20t/cache/episodes \
  --split-manifest data/rh20t/prepared/task_split_manifest.json \
  --output data/rh20t/prepared/normalization_stats.json
python benchmarks/rh20t/preparation/finalize_rh20t_data_freeze_v2.py \
  --prepared-dir data/rh20t/prepared
```

This is the fixed cfg1 protocol. Cache construction defaults to published camera
serial `750612070851`; `--camera-serial` explicitly selects another camera and
therefore defines a different corpus. Both official video layouts
`cam_<serial>/color.mp4` and `cam_<serial>/color/color.mp4` are accepted, with the
matching timestamp layout. The source code retains the original numerical
processing: 96-by-96 grayscale, camera timestamp alignment, duplicate-timestamp
resolution, quaternion sign continuity and normalization, and causal backward
as-of gripper commands. There is no interpolation from future action events.

Eligibility requires rating at least 2, at least 64 frames, complete finite
signals, exact video/timestamp frame counts and causal gripper coverage. The
second stage excludes nonfinite or physically corrupted base-frame TCP XYZ
coordinates exceeding 10 metres. It does not select on model outputs or add an
FT performance filter. The task-split implementation requires 124 eligible
tasks with at least four episodes each and preserves the fixed 74/25/25 split.
It fails rather than silently changing the protocol when the downloaded corpus
has a different eligible population.

The metadata/evaluation-bank tools under `benchmarks/core/scripts` consume these
prepared manifests. `persistent_jepa.rh20t_data` supplies the model loader,
relation sampler and normalized batches. Generated manifests record the
caller's own file paths and checksums; they are runtime artifacts, not bundled
historical logs.

```bash
python -m pytest -q tests/core/test_rh20t_preprocessing.py
```

The synthetic tests decode a small generated video, verify alignment and causal
action behavior, test corruption filtering, and verify task-disjoint splits and
within-split pairing. They do not use real participant data.
