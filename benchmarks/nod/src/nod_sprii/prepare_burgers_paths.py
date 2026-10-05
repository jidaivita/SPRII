"""Build a writable path-normalized view of the immutable Zenodo Burgers release.

Only symlinks and JSON manifests are created. The immutable release is never edited.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_nu(name: str) -> float:
    m = re.search(r"viscous_nu_([^.]*(?:\.[^.]*)?)\.h5$", name)
    if not m:
        # Scientific notation names such as viscous_nu_1e-04.h5.
        m = re.search(r"viscous_nu_([^/]+)\.h5$", name)
    if not m:
        raise ValueError(f"cannot parse viscosity from {name}")
    return float(m.group(1))


def decimal_name(prefix: str, nu: float) -> str:
    return f"{prefix}_{nu:.4f}.h5"


def link_dir(src: Path, dst: Path, include_inviscid: bool = False) -> list[dict[str, str | float]]:
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for path in sorted(src.glob("viscous_nu_*.h5")):
        nu = parse_nu(path.name)
        out = dst / decimal_name("viscous_nu", nu)
        if out.exists() or out.is_symlink():
            out.unlink()
        out.symlink_to(path.resolve())
        rows.append({"source": str(path.resolve()), "normalized": str(out), "nu": nu})
    if include_inviscid:
        inv = src / "inviscid.h5"
        if inv.exists():
            out = dst / "inviscid_nu_0.0000.h5"
            if out.exists() or out.is_symlink():
                out.unlink()
            out.symlink_to(inv.resolve())
            rows.append({"source": str(inv.resolve()), "normalized": str(out), "nu": 0.0})
    if not rows:
        raise FileNotFoundError(f"no HDF5 shards found in {src}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    args = ap.parse_args()
    release = args.release_root / "data"
    out = args.out_root
    rows = {
        "train": link_dir(release / "train", out / "train"),
        "train_add": link_dir(release / "train_add", out / "train_add"),
        "truth": link_dir(release / "truth", out / "truth", include_inviscid=True),
    }
    manifest = {
        "release_root": str(args.release_root.resolve()),
        "normalized_root": str(out.resolve()),
        "immutable_inputs": rows,
        "missing_expected_truth": [0.003],
        "notes": "Symlink-only path normalization; source release is untouched.",
    }
    (out / "path_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
