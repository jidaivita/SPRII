"""Verify the source files listed in the release integrity manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'releases/manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid relative manifest path')
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise SystemExit('Source verification failed: ' + ', '.join(failures))
    print(f"Verified {len(manifest['files'])} source files for SPRII {manifest['version']}.")


if __name__ == '__main__':
    main()
