"""Fetch pinned public dependencies. Run explicitly; this script does not train."""
from __future__ import annotations
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'cadm': ('https://github.com/younggyoseo/CaDM.git', '38c11a58d959bfd597f9323e58f28b17f6bf4fd9', 'CaDM'),
    'geps': ('https://github.com/itsakk/geps.git', 'e9a865218ecffacb7007ac7d719f3741afcf8c02', 'geps_original'),
    'coda': ('https://github.com/yuan-yin/CoDA.git', '17b73521394f2a5986e5418c32ad2965c97cd8c0', 'coda_original'),
}

def fetch(name: str) -> None:
    if name == 'nod':
        url = 'https://zenodo.org/records/20406332/files/code.tar.gz?download=1'
        with urlopen(url, timeout=60) as response:
            content = response.read()
        expected = 'eacafe88ea61e3716a006e3123a49669284a23a61fec70873d9216738ba508e7'
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('NOD archive checksum changed; stop and inspect the upstream release.')
        target = ROOT / 'benchmarks/nod/third_party/nod_original'
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
            for entry in archive.getmembers():
                if not entry.isfile():
                    continue
                relative = Path(entry.name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unsafe archive member')
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(entry).read())
        local = ROOT / 'benchmarks/nod/src/nod_sprii/ngs'
        for filename in ('neuralnetworks.py', 'pygpt_network_PFFT_SDVAE.py', 'unet1d.py'):
            shutil.copy2(target / 'code/Burgers/ngs' / filename, local / filename)
    elif name == 'gym':
        target = ROOT / 'benchmarks/baseline_adapters/third_party/gym_0_16_pendulum.py'
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen('https://raw.githubusercontent.com/openai/gym/0.16.0/gym/envs/classic_control/pendulum.py', timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != 'c21df67111cc3ac2b5d55bff94dcac5af5ceb9cf582350d9d846f258033da221':
            raise RuntimeError('Pinned Gym source checksum changed')
        target.write_bytes(content)
    else:
        url, revision, folder = SOURCES[name]
        target = ROOT / 'benchmarks/baseline_adapters/third_party' / folder
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', url, str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError('Upstream revision mismatch')
    print(f'{name}: fetched pinned public source')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', choices=['nod', 'cadm', 'geps', 'coda', 'gym'])
    for source in parser.parse_args().sources:
        fetch(source)
