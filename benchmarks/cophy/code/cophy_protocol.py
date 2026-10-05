"""Small, auditable v2 policy primitives. No experiment is launched here."""
import hashlib
import json
import math
from pathlib import Path


def category_index(value, support, tolerance=0.01):
    if not math.isfinite(float(value)):
        raise ValueError('Nonfinite confounder')
    matches = [i for i, x in enumerate(support) if abs(float(x)-float(value)) < tolerance]
    if len(matches) != 1:
        raise ValueError(f'Value {value} does not uniquely match declared support')
    return matches[0]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_preflight(path, release=None, require_release=False, stage='training'):
    p = Path(path)
    data = json.loads(p.read_text())
    if stage not in {'training', 'features'}:
        raise ValueError('Unknown preparation stage')
    required = ({'audit', 'splits', 'derenderer', 'protocol'} if stage == 'features' else
                {'audit', 'relation_index', 'sampler', 'random_rule',
                 'validation_correct', 'validation_wrong_any', 'validation_wrong_1', 'test_generator', 'protocol'})
    if data.get('status') != 'PASS' or not required <= data.get('artifacts', {}).keys():
        raise ValueError('Preflight is missing required audited artifacts')
    for item in data['artifacts'].values():
        target = Path(item['path'])
        if not target.is_absolute():
            target = p.parent / target
        if digest(target) != item['sha256']:
            raise ValueError(f'Artifact hash mismatch: {target}')
    if require_release:
        if not release:
            raise ValueError('Test remains sealed')
        seal = json.loads(Path(release).read_text())
        if (seal.get('status') != 'FROZEN_FOR_TEST' or
                seal.get('preflight_sha256') != digest(p)):
            raise ValueError('Test release is not bound to this preflight')
        for item in seal.get('artifacts',{}).values():
            target=Path(item['path'])
            if not target.is_absolute():target=Path(release).parent/target
            if digest(target)!=item['sha256']:raise ValueError('Changed frozen test-release artifact')
    return data


def verify_runtime_inputs(preflight_path, args, *, require_caches=True):
    """Bind actual data configuration to audited artifacts, before loading data.

    A feature-stage receipt intentionally does not require caches/manifests that
    feature extraction is about to create. Training uses the completed receipt.
    """
    p = Path(preflight_path)
    data = json.loads(p.read_text())
    profile = data.get('input_profile', {})
    if profile.get('dataset_dir'):
        profile = dict(profile, dataset_dir=str(Path(profile['dataset_dir']).resolve()))
    expected = {'scene': args.dataset_name, 'num_objects': args.num_objects,
                'type': args.type, 'dataset_dir': str(Path(args.dataset_dir).resolve())}
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError('Runtime dataset/configuration differs from the audited input profile')
    artifacts = data['artifacts']
    def path_for(name):
        item = artifacts.get(name)
        if item is None:
            raise ValueError(f'Missing input artifact: {name}')
        target = Path(item['path'])
        return (target if target.is_absolute() else p.parent/target).resolve()
    if digest(args.derendering_ckpt) != artifacts['derenderer']['sha256']:
        raise ValueError('Runtime visual frontend differs from the audited one')
    if require_caches and not args.train_from_rgb:
        prefix = {'balls': f'balls_{args.num_objects}',
                  'collision': f'collision_{args.type}',
                  'blocktower': f'blocktower_{args.num_objects}_{args.type}'}[args.dataset_name]
        for split in ('train', 'val'):
            actual = (Path(args.preextracted_obj_vis_prop_dir)/f'{prefix}_{split}_extracted_prop.pickle').resolve()
            if actual != path_for(f'cache_{split}'):
                raise ValueError('Runtime cache path differs from audited cache artifact')
    return {'preflight_sha256': digest(p), 'input_profile': profile}


def should_extend(native20, native25, a20, a25):
    vals = (native20, native25, a20, a25)
    if any(not math.isfinite(x) or x < 0 for x in vals):
        raise ValueError('Invalid validation loss')
    return any(old > 0 and (old-new)/old >= .01 for old, new in [(native20,native25),(a20,a25)])


def verify_adapter_binding(preflight_path):
    """The v3 trainer must not accept a v2 full-U or stale-code preflight."""
    data = json.loads(Path(preflight_path).read_text())
    if data.get('adapter_version') != 'cophy-pt16-v3':
        raise ValueError('Need a PT16 v3 data/code preflight; full-U v2 is superseded')
    root = Path(__file__).resolve().parent
    actual = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
    if data.get('code_sha256') != actual:
        raise ValueError('Preflight is not bound to the current complete adapter source')
    return data


def vicreg_focal(recipient, donor):
    import torch
    if recipient.shape != donor.shape or recipient.ndim != 2 or recipient.shape[1] != 16:
        raise ValueError('VICReg expects matching N x 16 persistent focal matrices')
    if len(recipient) < 2:
        return (recipient.sum()+donor.sum()) * 0, {'skipped': True}
    inv = (recipient-donor).square().mean()
    var = sum(torch.relu(1-torch.sqrt(x.var(dim=0, unbiased=True)+1e-4)).mean()
              for x in (recipient,donor)) / 2
    cov = recipient.new_zeros(())
    for x in (recipient,donor):
        x = x - x.mean(0)
        c = x.T @ x / (len(x)-1)
        cov = cov + (c.square().sum()-c.diag().square().sum()) / x.shape[1]
    return 25*inv+25*var+cov, {'skipped':False, 'invariance':inv,'variance':var,'covariance':cov}


def ridge_probe(train_x, train_y, eval_x, n_classes, alpha=1.):
    import numpy as np
    x, e = np.asarray(train_x, float), np.asarray(eval_x, float)
    y = np.asarray(train_y, int)
    if x.ndim != 2 or e.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Invalid probe data')
    if (y < 0).any() or (y >= n_classes).any():
        raise ValueError('Invalid class label')
    if not np.isfinite(x).all() or not np.isfinite(e).all():
        raise ValueError('Nonfinite representation supplied to probe')
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    x = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    e = np.column_stack(((e-mean)/scale, np.ones(len(e))))
    penalty = np.eye(x.shape[1])*alpha
    penalty[-1,-1] = 0
    w = np.linalg.solve(x.T@x+penalty, x.T@np.eye(n_classes)[y])
    scores=e@w
    if not np.isfinite(w).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite probe solution; do not turn NaN argmax into a class')
    return scores.argmax(1)


def paired_summary(native_correct, a_correct, native_wrong, a_wrong, a_null,
                   random_correct, groups, repeats=10000, seed=20260911):
    """Inputs are already recipient-level focal averages. Conditional on fixed models/donors."""
    import numpy as np
    xs = [np.asarray(x,float) for x in [native_correct,a_correct,native_wrong,a_wrong,a_null,random_correct]]
    if any(x.ndim != 1 or x.shape != xs[0].shape or not np.isfinite(x).all() for x in xs):
        raise ValueError('Need finite aligned per-recipient vectors')
    n,a,nw,aw,an,r = xs
    if not len(n) or len(groups) != len(n):
        raise ValueError('Missing aligned groups')
    values = np.stack([n-a, an-a, (aw-a)-(nw-n), r-a], axis=1)
    labels = list(dict.fromkeys(groups))
    # Preserve all recipients within each sampled family, rather than independent focal rows.
    indices = [np.array([i for i,g in enumerate(groups) if g == label]) for label in labels]
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats,4))
    for b in range(repeats):
        ix = np.concatenate([indices[j] for j in rng.integers(len(indices), size=len(indices))])
        samples[b] = values[ix].mean(0)
    names = ['reuse_gain','use_vs_null','dod_wrong_any','a_vs_random_correct']
    return {key:{'estimate':float(values[:,i].mean()),
                 'ci95':np.quantile(samples[:,i],[.025,.975]).tolist()}
            for i,key in enumerate(names)}
