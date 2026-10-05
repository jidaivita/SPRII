# Environment gallery

The README covers thirteen settings with ten image previews and three text-only
CoPhy implementation links. Each image uses the same 720 × 480 SVG frame; its
thumbnail retains its aspect ratio. Transformations and attribution are described
in [media credits](media-credits.md). Select an image to inspect the preview, or
an environment name to open its implementation guide.

| Preview | Content | Implementation |
|---|---|---|
| SpringWorld | Physics demonstration: force and release | [SpringWorld](../benchmarks/springworld/README.md) |
| PokeWorld | History replay with applied actions | [PokeWorld](../benchmarks/pokeworld/revision/README.md) |
| D-Clean | Recorded trajectory: external force and drag | [Core](../benchmarks/core/README.md) |
| CoPhy Collision | Text-only implementation link | [CoPhy](../benchmarks/cophy/README.md) |
| CoPhy Balls | Text-only implementation link | [CoPhy](../benchmarks/cophy/README.md) |
| CoPhy Blocktower | Text-only implementation link | [CoPhy](../benchmarks/cophy/README.md) |
| Burgers | One-dimensional PDE field | [NOD](../benchmarks/nod/README.md) |
| FHN | Reaction–diffusion field | [NOD](../benchmarks/nod/README.md) |
| Swimmer | Physics demonstration: joint drive and reversal | [Swimmer](../benchmarks/swimmer/README.md) |
| Overcooked | Cooperative gridworld | [Overcooked](../benchmarks/overcooked/README.md) |
| Baxter | Recorded tactile signals | [Baxter data](datasets.md) |
| RH20T | Official website demonstration: drawer manipulation | [RH20T](../benchmarks/rh20t/README.md) |
| Pendulum | Torque demonstration | [Pendulum](../benchmarks/baseline_adapters/README.md#pendulum) |

The previews show physics demonstrations, field trajectories, and recorded
observations. RH20T uses the official website's drawer-manipulation demonstration.
The publishers, documented licenses, reuse boundaries and transformations are described
in [media credits](media-credits.md) and [data sources](datasets.md).
Preview images do not relicense the underlying external resources.

Additional [baseline adapters](../benchmarks/baseline_adapters/README.md) are also
included in the source release.

## Rebuild the layout

The shared registry version and SHA-256, thumbnail hashes, descriptive
identifiers, and code-guide mapping are in
[`assets/environments/gallery.json`](../assets/environments/gallery.json).
The ten retained PNGs are byte-for-byte copies of the registry's
`assets.thumbnail.png`. CoPhy entries contain only the environment name and
implementation-guide mapping; they have no image path or image hash.
From the repository root, verify those local copies and regenerate the frames
and README gallery with the Python standard library:

```sh
python scripts/build_gallery.py
```

To import a newer frozen shared media package, provide its registry explicitly:

```sh
python scripts/build_gallery.py --registry path/to/shared-media/registry.json
```

The importer validates the thirteen-environment mapping and verifies the ten
retained source thumbnails before copying them. It skips CoPhy media without
reading or copying those files; the three CoPhy entries remain text-only.
It records the registry version and hash without retaining local source paths.
The SVGs embed their local source images and make no requests to external image
hosts. The frames preserve the environment colors and source aspect ratios.
