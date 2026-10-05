"""Shared IO for the independent sealed-test stage; no work on import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

VERSION = 'cophy-v7-sealed-test-1'
SCENES = ('balls', 'collision', 'blocktower')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('An immutable sealed artifact already differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Changed frozen artifact: ' + str(path))
    return path


def load_core(path):
    path = Path(path).resolve()
    name = '_sealed_core_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_freeze(path, verify_all=True):
    value = read(path)
    if value.get('version') != VERSION or value.get('status') != 'FROZEN_FOR_FINAL_TEST':
        raise ValueError('A complete v7 final-test freeze is required before test IO')
    if value.get('main_source_epochs') != 100 or value.get('seed') != 0:
        raise ValueError('Only the fixed source100, seed0 primary test is registered')
    if verify_all:
        for item in value['bound_files']:
            checked(item)
    else:
        # Per-entry scoring rechecks its own source/head/probe below, without
        # hashing every other model's large training cache again.
        for item in value['bound_files']:
            if Path(item['path']).parent == Path(__file__).resolve().parent:
                checked(item)
    return value


def entry_by_id(freeze, ident):
    found = [e for e in freeze['learned_entries'] if e['id'] == ident]
    if len(found) != 1:
        raise ValueError('Unregistered or ambiguous final-test entry: ' + ident)
    return found[0]


def field_names(scene):
    return (['mass', 'friction', 'gravity_x', 'gravity_y'] if scene == 'blocktower'
            else ['mass', 'friction', 'restitution'])


def physical_values(record, scene):
    return list(record['raw_physical']) + (list(record['raw_gravity']) if scene == 'blocktower' else [])
