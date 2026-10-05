"""Build thirteen same-source environment previews and verify their project-page provenance."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

PROJECT_URL = "https://jidaivita.github.io/sprii/"
CARD_SIZE = (1280, 720)
CARD_PADDING = 16
PALETTE = {"canvas": "#F7F4EE", "border": "#C7C1B7"}
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

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

SETTING_DETAILS = {
    "springworld": ("benchmarks/formation_use/scripts/train_spring_source.py", "springworld", "benchmarks/springworld/README.md"),
    "pokeworld": ("benchmarks/pokeworld/revision/scripts/train_pokeworld_revision.py", "d-clean-and-pokeworld", "benchmarks/pokeworld/revision/scripts/prepare_pokeworld_factorized.py"),
    "dclean": ("benchmarks/core/scripts/train.py", "d-clean-and-pokeworld", "benchmarks/core/scripts/generate_dclean.py"),
    "cophy_collision": ("benchmarks/cophy/code/cophy_complete_v7/cpc/train.py", "cophy-balls-collision-and-blocktower", "docs/datasets.md#cophy"),
    "cophy_balls": ("benchmarks/cophy/code/cophy_complete_v7/rssm/train.py", "cophy-balls-collision-and-blocktower", "docs/datasets.md#cophy"),
    "cophy_blocktower": ("benchmarks/cophy/code/cophy_complete_v7/cpc/train.py", "cophy-balls-collision-and-blocktower", "docs/datasets.md#cophy"),
    "burgers": ("benchmarks/nod/src/nod_sprii/train_sprii_clean.py", "nod-burgers-and-fhn", "benchmarks/nod/README.md#dependencies-and-public-inputs"),
    "fhn": ("benchmarks/nod/src/fhn_minimal/train.py", "nod-burgers-and-fhn", "benchmarks/nod/README.md#generate-the-fhn-data"),
    "swimmer": ("benchmarks/swimmer/code/paper_c/swimmer/s2_train.py", "overcookedv2-articulated-swimmer-and-pendulum", "benchmarks/swimmer/README.md#entry-points"),
    "overcooked": ("benchmarks/overcooked/scripts/train.py", "overcookedv2-articulated-swimmer-and-pendulum", "benchmarks/overcooked/README.md#reconstruct-the-data-and-train"),
    "baxter": ("benchmarks/core/scripts/train_baxter_a1.py", "baxter-tactile-and-rh20t", "docs/datasets.md#baxter-tactile-hardness"),
    "rh20t": ("benchmarks/core/scripts/train_rh20t.py", "baxter-tactile-and-rh20t", "docs/datasets.md#rh20t"),
    "pendulum": ("benchmarks/baseline_adapters/cadm_pendulum_relation_components_v5.py", "overcookedv2-articulated-swimmer-and-pendulum", "benchmarks/baseline_adapters/README.md#pendulum"),
}

MEDIA = {
    "spring": ("Environment demonstration", "Force and release in SpringWorld"),
    "poke": ("Recorded histories", "PokeWorld histories with applied-action overlays"),
    "dclean": ("Recorded trajectory", "D-Clean motion, velocity and applied forces"),
    "cophy_collision": ("Original task schematic", "Contact under related object properties"),
    "cophy_balls": ("Original task schematic", "Related four-ball interactions"),
    "cophy_blocktower": ("Original task schematic", "Contact and stability in block towers"),
    "nod1d": ("Released numerical trajectory", "Burgers spatial profile and space-time field"),
    "nod2d": ("Solver demonstration", "Coupled FitzHugh-Nagumo fields"),
    "baxter": ("Recorded tactile data", "Sixteen-channel grasp recording"),
    "rh20t": ("Official website demonstration", "RH20T drawer manipulation"),
    "swimmer": ("Environment demonstration", "Joint drive and reversal in Swimmer"),
    "overcooked": ("Scripted environment demonstration", "Two scripted cooks prepare and deliver food"),
    "pendulum": ("Environment demonstration", "Pendulum torque and angular response"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Media paths must stay inside their source directory")
    return path


def image_size(data: bytes, suffix: str) -> tuple[float, float]:
    if suffix == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected a PNG poster")
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    svg = ET.fromstring(data)
    if svg.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("Expected an SVG task schematic")
    for element in svg.iter():
        if element.tag.rsplit("}", 1)[-1] in {"script", "foreignObject", "image"}:
            raise ValueError("Task schematics must contain original vector geometry only")
        for key, value in element.attrib.items():
            if key.rsplit("}", 1)[-1] == "href" and not value.startswith("#"):
                raise ValueError("External SVG resources are not permitted")
    _, _, width, height = map(float, svg.attrib["viewBox"].split())
    return width, height


def source_digest(items: list[dict]) -> str:
    record = [{"id": item["environment_id"], "asset": item["source_asset"],
               "sha256": item["source_sha256"], "poster_sha256": item.get("source_poster_sha256"),
               "video_sha256": item.get("source_video_sha256"), "url": item["source_url"],
               "poster_url": item.get("source_poster_url"), "video_url": item.get("source_video_url")}
              for item in items]
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())


class SummaryImages(HTMLParser):
    """Read the exact image selected by each project-page environment summary."""

    def __init__(self) -> None:
        super().__init__()
        self.environment = None
        self.in_summary = False
        self.images = {}
        self.posters = {}
        self.videos = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag == "details" and attrs.get("id", "").startswith("setting-"):
            self.environment = attrs["id"].removeprefix("setting-")
        elif tag == "summary":
            self.in_summary = True
        elif tag == "img" and self.in_summary and self.environment:
            if self.environment in self.images:
                raise ValueError(f"Multiple summary images for {self.environment}")
            self.images[self.environment] = attrs["src"]
        elif tag == "video" and self.environment:
            self.posters[self.environment] = attrs["poster"]
        elif tag == "source" and attrs.get("type") == "video/mp4" and self.environment:
            self.videos[self.environment] = attrs["src"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "summary":
            self.in_summary = False


def import_site(site: Path) -> tuple[dict, dict[str, bytes]]:
    items, images = [], {}
    summary = SummaryImages()
    summary.feed((site / "index.html").read_text())
    if set(summary.images) != {row[0] for row in ENVIRONMENTS}:
        raise ValueError("The project page must supply one summary image for each of the thirteen settings")
    for environment_id, public_id, label, guide in ENVIRONMENTS:
        schematic = environment_id.startswith("cophy_")
        source_url = summary.images[environment_id]
        relative = urlsplit(source_url).path
        filename = Path(relative).name
        expected_suffix = ".svg" if schematic else ".png"
        if Path(relative).suffix != expected_suffix:
            raise ValueError(f"Unexpected summary image format for {environment_id}")
        data = local_file(site, relative).read_bytes()
        image_size(data, Path(filename).suffix)
        image = public_id + Path(filename).suffix
        kind, caption = MEDIA[environment_id]
        item = {"id": public_id, "environment_id": environment_id, "label": label,
                "guide": guide, "kind": kind, "caption": caption, "image": image,
                "card": f"cards/{public_id}.svg", "project_url": f"{PROJECT_URL}#setting-{environment_id}",
                "source_asset": relative, "source_url": PROJECT_URL + source_url,
                "source_sha256": sha256(data), "image_sha256": sha256(data),
                "presentation": "Original project-page schematic" if schematic else "Exact project-page summary-image bytes"}
        if not schematic:
            poster_url = summary.posters[environment_id]
            poster_relative = urlsplit(poster_url).path
            item["source_poster"] = poster_relative
            item["source_poster_url"] = PROJECT_URL + poster_url
            item["source_poster_sha256"] = sha256(local_file(site, poster_relative).read_bytes())
            video_url = summary.videos[environment_id]
            video_relative = urlsplit(video_url).path
            item["source_video"] = video_relative
            item["source_video_url"] = PROJECT_URL + video_url
            item["source_video_sha256"] = sha256(local_file(site, video_relative).read_bytes())
        items.append(item)
        images[image] = data
    manifest = {"schema_version": 3, "card_size": list(CARD_SIZE), "card_padding": CARD_PADDING,
                "palette": PALETTE, "source": {"kind": "project-page-assets", "base_url": PROJECT_URL,
                "snapshot_sha256": source_digest(items)}, "items": items}
    return manifest, images


def card_svg(item: dict, data: bytes) -> bytes:
    width, height = CARD_SIZE
    sw, sh = image_size(data, Path(item["image"]).suffix)
    scale = min((width - 2 * CARD_PADDING) / sw, (height - 2 * CARD_PADDING) / sh)
    iw, ih = sw * scale, sh * scale
    x, y = (width - iw) / 2, (height - ih) / 2
    if item["image"].endswith(".svg"):
        nested = ET.fromstring(data)
        nested.attrib.update({"x": f"{x:.6f}", "y": f"{y:.6f}", "width": f"{iw:.6f}",
                              "height": f"{ih:.6f}", "preserveAspectRatio": "xMidYMid meet"})
        content = ET.tostring(nested, encoding="unicode")
    else:
        encoded = base64.b64encode(data).decode("ascii")
        content = (f'<image x="{x:.6f}" y="{y:.6f}" width="{iw:.6f}" height="{ih:.6f}" '
                   f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>')
    title = html.escape(item["label"] + ": " + item["kind"])
    return (f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="card-title">\n'
            f'<title id="card-title">{title}</title>\n'
            f'<rect width="{width}" height="{height}" rx="12" fill="{PALETTE["canvas"]}"/>\n'
            f'{content}\n</svg>\n').encode()


def readme_gallery(items: list[dict]) -> str:
    lines = ["## Explore thirteen settings", "",
             "Each setting links to its project-page demonstration or schematic, implementation, recipe, and data setup.", "",
             "| # | Setting | Code | Recipe | Data |", "|---|---|---|---|---|"]
    for index, item in enumerate(items, 1):
        code, recipe, data = SETTING_DETAILS[item["id"]]
        lines.append(f'| {index} | [{item["label"]}]({item["project_url"]}) '
                     f'| [Source]({code}) · [Guide]({item["guide"]}) '
                     f'| [Recipe](docs/paper_recipes.md#{recipe}) | [Inputs]({data}) |')
    lines.extend(["", "CoPhy Collision, Balls, and Blocktower are separate settings with shared",
                  "scene-selecting training entry points. Burgers and FHN, and Baxter and RH20T,",
                  "are listed separately even where they share a guide.", "",
                  "<details>", "<summary>Environment previews — all 13 settings</summary>", "",
                  "Select a preview to open its video or schematic and experiment details on the project page.",
                  "The previews share the page's source assets. They illustrate the settings; they are not learned-model predictions.",
                  "", "<table>"])
    for index, item in enumerate(items):
        if index % 2 == 0:
            lines.append("<tr>")
        alt = html.escape(item["label"] + ": " + item["kind"], quote=True)
        lines.append('<td width="50%" align="center" valign="top">'
                     f'<a href="{item["project_url"]}"><img src="assets/environments/{item["card"]}?v={item["card_sha256"][:12]}" '
                     f'width="320" height="180" alt="{alt}" /></a><br/>'
                     f'<a href="{item["project_url"]}">{html.escape(item["label"])}</a><br/>'
                     f'<sub>{html.escape(item["kind"])}</sub></td>')
        if index % 2 == 1 or index == len(items) - 1:
            if index == len(items) - 1 and index % 2 == 0:
                lines.append('<td width="50%"></td>')
            lines.append("</tr>")
    lines.extend(["</table>", "", "See [gallery notes](docs/gallery.md) and [media credits](docs/media-credits.md).",
                  "", "</details>", "", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, help="Import the current public project-page asset directory")
    parser.add_argument("--check", action="store_true", help="Verify bundled sources, cards and README without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets" / "environments"
    if args.site_root:
        manifest, images = import_site(args.site_root)
    else:
        manifest = json.loads((assets / "gallery.json").read_text())
        if manifest.get("schema_version") != 3:
            raise ValueError("Import project-page assets with --site-root before rebuilding this gallery")
        images = {item["image"]: local_file(assets, item["image"]).read_bytes() for item in manifest["items"]}
    if [item["environment_id"] for item in manifest["items"]] != [row[0] for row in ENVIRONMENTS]:
        raise ValueError("The gallery must contain the thirteen settings in the declared order")
    if source_digest(manifest["items"]) != manifest["source"]["snapshot_sha256"]:
        raise ValueError("Source snapshot hash does not match the declared assets")
    outputs = {}
    for item in manifest["items"]:
        data = images[item["image"]]
        if sha256(data) != item["source_sha256"] or sha256(data) != item["image_sha256"]:
            raise ValueError(f'Source image hash mismatch: {item["id"]}')
        code, _, input_guide = SETTING_DETAILS[item["id"]]
        for target in (item["guide"], code, input_guide):
            if not local_file(root, target.split("#")[0]).is_file():
                raise ValueError(f'Missing setting link: {target}')
        outputs[local_file(assets, item["image"])] = data
        card = card_svg(item, data)
        item["card_sha256"] = sha256(card)
        outputs[local_file(assets, item["card"])] = card
    outputs[assets / "gallery.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    readme_path = root / "README.md"
    readme = readme_path.read_text()
    logo_hash = sha256((root / "assets/branding/logo.svg").read_bytes())[:12]
    readme = re.sub(r'src="assets/branding/logo\.svg(?:\?[^\"]*)?"',
                    f'src="assets/branding/logo.svg?v={logo_hash}"', readme)
    start = readme.index("## Explore thirteen settings\n")
    end = readme.index("## Baselines and protocols\n", start)
    outputs[readme_path] = (readme[:start] + readme_gallery(manifest["items"]) + readme[end:]).encode()
    old_assets = set(assets.glob("*.png")) | set(assets.glob("*.svg")) | set((assets / "cards").glob("*.svg"))
    stale = old_assets - set(outputs)
    if args.check:
        changed = [str(path.relative_to(root)) for path, data in outputs.items()
                   if not path.is_file() or path.read_bytes() != data]
        if changed or stale:
            raise SystemExit("Gallery verification failed: " + ", ".join(changed + [str(p.relative_to(root)) for p in sorted(stale)]))
        print("Verified 13 source images, 13 deterministic cards, source hashes and project-page links.")
        return
    for path, data in outputs.items():
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(data)
    for path in stale:
        path.unlink()
    print("Built 13 same-source environment previews (1280 x 720) and their project-page links.")


if __name__ == "__main__":
    main()
