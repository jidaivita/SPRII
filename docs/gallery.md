# Environment gallery

The README gives all thirteen settings an implementation, recipe and data link,
plus a visual preview. Selecting a preview opens that setting's video or
schematic on the [project page](https://jidaivita.github.io/sprii/).

The gallery uses the exact image selected by each project-page environment card.
Ten previews are PNGs; the three CoPhy previews are original vector task
schematics. Every image sits in the same warm-white 1280 × 720 SVG frame, with
its source aspect ratio preserved. PokeWorld uses the closer view shared with
the paper so its pusher and object remain visible in a small card; its video
poster separately shows both recorded histories.

| Preview | Content | Implementation |
|---|---|---|
| SpringWorld | Physics demonstration: force and release | [SpringWorld](../benchmarks/springworld/README.md) |
| PokeWorld | Recorded history, rendered for viewing on a light floor | [PokeWorld](../benchmarks/pokeworld/revision/README.md) |
| D-Clean | Recorded trajectory, velocity and applied forces | [Core](../benchmarks/core/README.md) |
| CoPhy Collision | Original contact schematic | [CoPhy](../benchmarks/cophy/README.md) |
| CoPhy Balls | Original four-ball schematic | [CoPhy](../benchmarks/cophy/README.md) |
| CoPhy Blocktower | Original stability and contact schematic | [CoPhy](../benchmarks/cophy/README.md) |
| Burgers | Released spatial profile and space–time field | [NOD](../benchmarks/nod/README.md) |
| FHN | Coupled reaction–diffusion fields | [NOD](../benchmarks/nod/README.md) |
| Swimmer | Physics demonstration: joint drive and reversal | [Swimmer](../benchmarks/swimmer/README.md) |
| Overcooked | Scripted cooperative cooking demonstration | [Overcooked](../benchmarks/overcooked/README.md) |
| Baxter | Recorded curves and a sixteen-channel tactile heatmap | [Baxter data](datasets.md) |
| RH20T | Official website demonstration: drawer manipulation | [RH20T](../benchmarks/rh20t/README.md) |
| Pendulum | Torque and angular-response demonstration | [Pendulum](../benchmarks/baseline_adapters/README.md#pendulum) |

These previews show the environments and recorded interactions, not learned-model
predictions. The light presentation of PokeWorld, Swimmer and Overcooked changes
only their human-facing rendering. Model observations, stored trajectories,
experimental code and numerical results are unchanged. Attribution and display
transformations are documented in [media credits](media-credits.md).

## Rebuild and verify

[`assets/environments/gallery.json`](../assets/environments/gallery.json)
records each source image's relative project-page path, public URL, SHA-256,
local copy hash and generated-card hash. Video-backed settings also record the
video and video-poster hashes separately. The source snapshot hash covers the
thirteen selected images and their associated video/poster versions. No local
absolute source paths are retained.

Source URLs preserve the page's content-version query strings. README card URLs
and the project logo also include their content hash so cached images cannot
retain an older display.

Rebuild the frames and README from the bundled images with Python's standard library:

```sh
python scripts/build_gallery.py
python scripts/build_gallery.py --check
```

Import a checked-out project page, using the image actually selected by each
setting's summary card:

```sh
python scripts/build_gallery.py --site-root path/to/project-page/sprii
python scripts/build_gallery.py --site-root path/to/project-page/sprii --check
```

The importer copies source bytes without recoloring or resampling, replaces old
previews, and rejects SVGs with scripts or external images. The CoPhy inputs
must be original vector schematics; historical benchmark-image imports are no
longer supported. The SVG wrappers embed their sources and request no remote
images. `--check` writes nothing and fails on mismatched source hashes, stale
cards, outdated README links or extra old preview files.

`ENVIRONMENTS` and `SETTING_DETAILS` in
[`scripts/build_gallery.py`](../scripts/build_gallery.py) define the setting
order and code links. Rebuilding retains thirteen index rows and thirteen
previews. The source code's release integrity is checked separately with
`python scripts/verify_release.py`.
