"""Create a fresh, portable CoPhy source layout without copying datasets or logs."""
from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    target = args.output
    if target.exists():
        raise FileExistsError('Choose a new runtime directory; existing data is never overwritten')
    target.mkdir(parents=True)
    shutil.copytree(source / 'code', target / 'source')
    shutil.copytree(source / 'discovery', target / 'discovery_v4_1')
    if (source / 'code_profiles').exists():
        shutil.copytree(source / 'code_profiles', target / 'code_profiles')
    shutil.copy2(source / 'LICENSE', target / 'LICENSE')
    print('Source layout created. Supply the benchmark assets before training.')


if __name__ == '__main__':
    main()
