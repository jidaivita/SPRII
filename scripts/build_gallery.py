"""Rebuild ten environment previews and three text-only CoPhy guide links."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


# Public filenames and implementation guides stay stable across media updates.
ENVIRONMENTS = (
    ("spring", "springworld", "SpringWorld", "benchmarks/springworld/README.md"),
    ("poke", "pokeworld", "PokeWorld", "benchmarks/pokeworld/revision/README.md"),
    ("dclean", "dclean", "D-Clean", "benchmarks/core/README.md"),
    ("cophy_collision", "cophy_collision", "CoPhy Collision", "benchmarks/cophy/README.md"),
    ("cophy_balls", "cophy_balls", "CoPhy Balls", "benchmarks/cophy/README.md"),
    ("cophy_blocktower", "cophy_blocktower", "CoPhy Blocktower", "benchmarks/cophy/README.md"),
    ("nod1d", "burgers", "Burgers", "benchmarks/nod/README.md"),
    ("nod2d", "fhn", "FHN", "benchmarks/nod/README.md"),
    ("swimmer", "swimmer", "Swimmer", "benchmarks/swimmer/README.md"),
    ("overcooked", "overcooked", "Overcooked", "benchmarks/overcooked/README.md"),
    ("baxter", "baxter", "Baxter", "docs/datasets.md"),
    ("rh20t", "rh20t", "RH20T", "benchmarks/rh20t/README.md"),
    ("pendulum", "pendulum", "Pendulum", "benchmarks/baseline_adapters/README.md#pendulum"),
)

# Keep implementation coverage without redistributing unverified upstream media.
TEXT_ONLY_ENVIRONMENTS = frozenset({
    "cophy_collision", "cophy_balls", "cophy_blocktower",
})


def text_only_item(environment_id: str, public_id: str, label: str, guide: str) -> dict:
    return {
        "id": public_id,
        "environment_id": environment_id,
        "label": label,
        "guide": guide,
        "kind": "Implementation guide",
        "presentation": "text",
    }


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Media paths must stay inside their source directory")
    return path


def import_registry(path: Path) -> tuple[dict, dict[str, bytes]]:
    raw = path.read_bytes()
    registry = json.loads(raw)
    if registry["schema_version"] != 1 or registry["status"] != "frozen":
        raise ValueError("Expected a frozen shared environment media registry")
    entries = registry["environments"]
    sources = {entry["id"]: entry for entry in entries}
    expected = {entry[0] for entry in ENVIRONMENTS}
    if len(entries) != len(expected) or set(sources) != expected:
        raise ValueError("The registry must contain all 13 gallery environments exactly once")
    manifest = {
        "schema_version": 2,
        "card_size": [720, 480],
        "source_registry": {"version": registry["version"], "sha256": sha256(raw)},
        "items": [],
    }
    images = {}
    for environment_id, public_id, label, guide in ENVIRONMENTS:
        if environment_id in TEXT_ONLY_ENVIRONMENTS:
            manifest["items"].append(text_only_item(environment_id, public_id, label, guide))
            continue
        source = sources[environment_id]
        thumbnail = source["assets"]["thumbnail.png"]
        data = local_file(path.parent, thumbnail["file"]).read_bytes()
        if sha256(data) != thumbnail["sha256"] or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Registry thumbnail verification failed: {environment_id}")
        image = public_id + ".png"
        images[image] = data
        manifest["items"].append({
            "id": public_id,
            "environment_id": environment_id,
            "label": label,
            "image": image,
            "guide": guide,
            "kind": source["caption"].split(" · ")[0],
            "caption": source["caption"],
            "thumbnail_sha256": thumbnail["sha256"],
        })
    return manifest, images


def readme_gallery(items: list[dict]) -> str:
    lines = ["## Explore thirteen settings", "", "<table>"]
    for index, item in enumerate(items):
        if index % 3 == 0:
            lines.append("<tr>")
        if item.get("presentation") == "text":
            lines.append(
                '<td width="33%" align="center" valign="top">'
                f'<a href="{item["guide"]}"><strong>{html.escape(item["label"])}</strong></a>'
                '<br/>Code and experiment guide</td>'
            )
        else:
            alt = html.escape(item["label"] + ": " + item["kind"], quote=True)
            lines.append(
                '<td width="33%" align="center" valign="top">'
                f'<a href="assets/environments/{item["image"]}">'
                f'<img src="assets/environments/cards/{item["id"]}.svg" '
                f'width="240" alt="{alt}" /></a><br/>'
                f'<a href="{item["guide"]}">{html.escape(item["label"])}</a></td>'
            )
        if index % 3 == 2 or index == len(items) - 1:
            lines.append("</tr>")
    lines.extend([
        "</table>", "",
        "Ten environment previews and three CoPhy implementation links. Select an image",
        "to view its preview, or an environment name to open the code guide.",
        "See [image notes](docs/gallery.md) and",
        "[media credits and licenses](docs/media-credits.md).", "", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path,
                        help="Import ten previews from a frozen shared media registry; keep CoPhy text-only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets" / "environments"
    if args.registry:
        manifest, images = import_registry(args.registry)
    else:
        manifest = json.loads((assets / "gallery.json").read_text())
        images = {item["image"]: local_file(assets, item["image"]).read_bytes()
                  for item in manifest["items"]
                  if item["environment_id"] not in TEXT_ONLY_ENVIRONMENTS}
    if [item["environment_id"] for item in manifest["items"]] != [row[0] for row in ENVIRONMENTS]:
        raise ValueError("Gallery environment mapping does not match the shared registry")
    if not manifest.get("source_registry", {}).get("sha256"):
        raise ValueError("Import the shared registry before rebuilding")
    for item in manifest["items"]:
        if item["environment_id"] in TEXT_ONLY_ENVIRONMENTS:
            if item.get("presentation") != "text" or "image" in item or "thumbnail_sha256" in item:
                raise ValueError(f'CoPhy entries must remain text-only: {item["id"]}')
        elif sha256(images[item["image"]]) != item["thumbnail_sha256"]:
            raise ValueError(f'Thumbnail hash mismatch: {item["id"]}')
        if not local_file(root, item["guide"].split("#")[0]).is_file():
            raise ValueError(f'Implementation guide is missing: {item["id"]}')
    readme_path = root / "README.md"
    readme = readme_path.read_text()
    start = readme.index("## Explore thirteen settings\n")
    end = readme.index("## Find an experiment\n", start)

    # Preserve the exact shared PNG bytes; framing never recolors or resamples them.
    if args.registry:
        for image, data in images.items():
            (assets / image).write_bytes(data)
        (assets / "gallery.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    width, height = manifest["card_size"]
    cards = assets / "cards"
    cards.mkdir(exist_ok=True)
    for item in manifest["items"]:
        if item.get("presentation") == "text":
            continue
        data = images[item["image"]]
        encoded = base64.b64encode(data).decode("ascii")
        title = html.escape(item["label"] + ": " + item["kind"])
        # Explicit fitted bounds also preserve aspect ratio in SVG renderers
        # that interpret embedded-image intrinsic dimensions differently.
        source_width = int.from_bytes(data[16:20], "big")
        source_height = int.from_bytes(data[20:24], "big")
        scale = min((width - 24) / source_width, (height - 24) / source_height)
        image_width, image_height = source_width * scale, source_height * scale
        image_x, image_y = (width - image_width) / 2, (height - image_height) / 2
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="title">\n'
            f'  <title id="title">{title}</title>\n'
            f'  <rect width="{width}" height="{height}" rx="12" fill="#f3f4f5"/>\n'
            f'  <image x="{image_x:.6f}" y="{image_y:.6f}" '
            f'width="{image_width:.6f}" height="{image_height:.6f}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'xlink:href="data:image/png;base64,{encoded}"/>\n'
            '</svg>\n'
        )
        (cards / (item["id"] + ".svg")).write_text(svg)
    readme_path.write_text(readme[:start] + readme_gallery(manifest["items"]) + readme[end:])
    print(f'Built {len(images)} image cards ({width} x {height}) and '
          f'{len(manifest["items"]) - len(images)} text-only guide links.')


if __name__ == "__main__":
    main()
