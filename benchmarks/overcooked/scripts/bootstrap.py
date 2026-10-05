"""Fetch the fixed upstream revision and overlay one research variant."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, default=Path('external/icrl4aht'))
p.add_argument('--variant', choices=('matched', 'random'), default='matched')
p.add_argument('--archive', type=Path, help='Optional previously downloaded, hash-verified upstream tarball')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
spec = json.loads((root / 'upstream.json').read_text())
if a.destination.exists():
    p.error('Destination must not already exist; use separate trees for the two variants.')
blob = a.archive.read_bytes() if a.archive else urllib.request.urlopen(spec['archive_url'], timeout=60).read()
if hashlib.sha256(blob).hexdigest() != spec['archive_sha256']:
    raise ValueError('Upstream archive hash differs from the pinned revision')
with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
    members = archive.getmembers()
    prefix = members[0].name.split('/')[0]
    for member in members:
        relative = Path(member.name).relative_to(prefix)
        if '..' in relative.parts or relative.is_absolute() or not (member.isdir() or member.isfile()):
            raise ValueError('Unsafe or unexpected archive member')
        target = a.destination / relative
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
subprocess.run(['patch', '--batch', '--forward', '-p1', '-d', str(a.destination.resolve()),
                '-i', str(root / 'patches/ippo_engineering.patch')], check=True)
shutil.copytree(root / 'variants' / a.variant / 'native_a', a.destination / 'native_a')
shutil.copy2(root / 'variants' / a.variant / 'compact_transport.py', a.destination / 'compact_transport.py')
print(f'Prepared fixed upstream revision with {a.variant} additions at {a.destination}')
