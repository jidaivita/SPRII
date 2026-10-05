"""Isolated reader seed extension; donor and evaluation plans stay unchanged."""
import argparse
import hashlib
import sys
import types
from pathlib import Path

CORE_SHA = 'ca97ba4368cb2a86035e18d9866e8de967aa08906aee6e03ee79c36687c54c59'


def transform(source, seed, wrapper_sha):
    assert hashlib.sha256(source.encode()).hexdigest() == CORE_SHA
    assert seed in (0, 1, 2)
    changes = [
        ('torch.manual_seed(seedof(', 'torch.manual_seed(_reader_seedof(', 5),
        ('torch.manual_seed(991+epoch)', 'torch.manual_seed(991+epoch+1000003*READER_SEED)', 1),
        ('np.random.default_rng(771+epoch)', 'np.random.default_rng(771+epoch+1000003*READER_SEED)', 1),
        ('seed=0, selection_ids_sha256=', 'seed=READER_SEED, reader_wrapper_sha256=READER_WRAPPER_SHA, selection_ids_sha256=', 1),
    ]
    for old, new, count in changes:
        assert source.count(old) == count, old
        source = source.replace(old, new)
    # Helper only affects the five model initialization tags. Data.seedof callers are untouched.
    helper = ('\nREADER_SEED=' + repr(seed) + '\nREADER_WRAPPER_SHA=' + repr(wrapper_sha)
              + '\ndef _reader_seedof(s):\n'
              + "    return seedof(s if READER_SEED == 0 else s + ':reader-seed:' + str(READER_SEED))\n")
    # Inject after imports/definitions and before the CLI invocation, preserving future imports.
    marker = "if __name__ == '__main__':"
    if marker not in source:
        marker = "if __name__=='__main__':"
    assert source.count(marker) == 1
    return source.replace(marker, helper + '\n' + marker)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--core', required=True)
    parser.add_argument('--reader-seed', type=int, choices=(0, 1, 2), required=True)
    args, rest = parser.parse_known_args()
    path = Path(args.core)
    source = transform(path.read_text(), args.reader_seed, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    module = types.ModuleType('_seeded_reader')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path)] + rest
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
