"""Content commitments and append-only result artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collections.abc import Mapping


def plain(value):
    if isinstance(value, Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)
        f.write('\n')


def npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def development_path(path):
    """Reject explicit seal paths before any stat/open. Metadata is not a public API."""
    p = Path(path)
    for part in p.parts:
        words = part.lower().replace('.', '_').replace('-', '_').split('_')
        if {'test', 'sealed', 'confirmation'} & set(words):
            raise PermissionError('development cannot consume sealed paths: ' + str(p))
    resolved = p.resolve()
    if resolved != p.absolute():
        for part in resolved.parts:
            if {'test', 'sealed', 'confirmation'} & set(part.lower().replace('.', '_').replace('-', '_').split('_')):
                raise PermissionError('development path resolves inside sealed data')
    return p


def checked(path, expected):
    path = development_path(path)
    if sha(path) != expected:
        raise ValueError('asset hash mismatch: ' + str(path))
    return path


def code_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: sha(p) for p in sorted(root.glob('*.py'))}


def tensor_state_digest(model):
    value=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        header=json.dumps([key,array.dtype.str,list(array.shape)],separators=(',',':')).encode()
        value.update(len(header).to_bytes(8,'big'));value.update(header);value.update(array.tobytes(order='C'))
    return value.hexdigest()


def load_protocol(path):
    cfg = read(development_path(path))
    if cfg.get('schema') != 'sprii-next.development.v1' or cfg.get('test_read') is not False:
        raise PermissionError('explicit development-only protocol required')
    if cfg.get('allowed_splits') != ['train', 'validation']:
        raise PermissionError('development split allowlist changed')
    if cfg.get('source_seeds') != [0, 1, 2] or cfg.get('reader_seeds') != [0, 1, 2]:
        raise ValueError('primary seed grid changed')
    if cfg.get('physics_coordinates') != 'train_standardized_log_m_log_gamma_log_k':
        raise ValueError('primary bottleneck coordinates must be explicit')
    keys=[(d['environment'],d['method'],d['source_seed']) for d in cfg['sources']]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate source descriptors')
    r=cfg['reader']
    if r['steps']<1 or r['batch_size']<1 or not 0<=r['warmup_steps']<r['steps']:
        raise ValueError('invalid fixed reader budget')
    return cfg
