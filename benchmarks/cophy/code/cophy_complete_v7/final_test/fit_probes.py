"""Export the existing ridge-alpha1 probes using TRAIN rows only, before freeze.

Source and head parameters remain frozen. No test path is accepted here.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from runtime import VERSION, artifact, checked, field_names, load_core, physical_values, read, sha, write


def fit(x, values, labels, path):
    x = np.asarray(x, np.float64); values = np.asarray(values, np.float64)
    labels = np.asarray(labels, np.int64)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(values).all():
        raise ValueError('Insufficient/nonfinite train probe observations')
    mean = x.mean(0); scale = x.std(0).clip(1e-8); z = (x-mean)/scale
    classes = [np.unique(labels[:, j]) for j in range(labels.shape[1])]
    onehot = np.concatenate([(labels[:, j, None] == c[None]).astype(float) for j, c in enumerate(classes)], 1)
    target = np.concatenate((values, onehot), 1); center = target.mean(0)
    weight = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ (target-center))
    arrays = dict(mean=mean, scale=scale, weight=weight, center=center,
                  raw_train_mean=values.mean(0), classes_count=np.asarray(len(classes)),
                  majority=np.asarray([c[np.argmax([(labels[:, j] == v).sum() for v in c])]
                                       for j, c in enumerate(classes)]))
    arrays.update({'classes_'+str(j): c for j, c in enumerate(classes)})
    tmp = Path(path).with_suffix('.pending.npz'); np.savez(tmp, **arrays); tmp.replace(path)
    return dict(rows=len(x), input_dims=x.shape[1], artifact=artifact(path), ridge_alpha=1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout', required=True); p.add_argument('--core', required=True)
    p.add_argument('--raw-train-relations', required=True); p.add_argument('--raw-train-sha256', required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', default='cpu');p.add_argument('--base')
    args = p.parse_args(); torch.set_num_threads(4)
    if (Path(args.readout)/'codes_complete.json').exists():
        from supervised_adapter import fit as fit_supervised
        fit_supervised(args);return
    core = load_core(args.core); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rd = Path(args.readout); encoding = read(rd/'encoding_complete.json'); binding = encoding['binding']
    if binding['readout_code_sha256'] != sha(args.core): raise ValueError('Use the exact bound readout implementation')
    raw = checked(dict(path=args.raw_train_relations, sha256=args.raw_train_sha256))
    records = read(raw)
    if any(r['split'] != 'train' for r in records): raise ValueError('Probe fitting accepts train records only')
    head_path = rd/'S3/learned/selected.pt'; ck = torch.load(head_path, map_location='cpu', weights_only=False)
    complete=read(rd/'S3/learned/complete.json')
    if binding['source_epochs']!=100 or complete.get('status')!='COMPLETE' or complete.get('epochs')!=100 or ck['config']['epochs'] != 100 or ck['config']['test_read'] is not False:
        raise ValueError('Source100 and head100 must be final before probe fitting')
    a = argparse.Namespace(out=str(rd), base=binding['base'], scene=binding['scene'], supports=3,
                           reference='learned', prepared=None, device=args.device)
    data = core.Data(a)
    head = core.Head(data.dims, data.det_dims, data.support_dims, data.horizon).to(args.device)
    head.load_state_dict(ck['model'], strict=True); head.eval(); head.requires_grad_(False)
    if ck['config']['code_sha256'] != sha(rd/'encoding_complete.json') or ck['config']['normalization_sha256'] != sha(rd/'normalization.json'):
        raise ValueError('Selected head/input binding differs')
    metadata = {(str(r['id']), int(r['slot'])): r for r in records}
    own_ids = list(map(str, read(rd/'cache/train/ids.json')))
    own_lut = {q: i for i, q in enumerate(own_ids)}
    own = np.load(rd/'cache/train/own_u.npy', mmap_mode='r')
    visible = np.load(rd/'cache/train/current_mask.npy', mmap_mode='r').any(1)
    datasets = {'own': dict(x=[], y=[], labels=[]), 'support_mean': dict(x=[], y=[], labels=[]),
                'memory': dict(x=[], y=[], labels=[])}
    def append(channel, vector, r):
        dst = datasets[channel]; dst['x'].append(vector); dst['y'].append(physical_values(r, a.scene)); dst['labels'].append(r['physical'])
    for r in records:
        ident, slot = str(r['id']), int(r['slot'])
        if r['in_C'] and ident in own_lut and visible[own_lut[ident], slot]: append('own', own[own_lut[ident], slot], r)
    plan = data.plan('train', 0); ids = data.data['train']['ids']
    for start in range(0, len(ids), 128):
        ix = np.arange(start, min(start+128, len(ids)))
        with torch.no_grad():
            u, _ = data.raw_context('train', ix, plan[ix], args.device)
            mean = torch.as_tensor(data.code_mean, device=args.device); scale = torch.as_tensor(data.code_scale, device=args.device)
            mask = torch.as_tensor(data.data['train']['mask'][ix], device=args.device, dtype=torch.float32)
            memory = head.support(((u-mean)/scale)*mask[:, :, None, None]).mean(2).cpu().numpy()
            average = u.mean(2).cpu().numpy()
        for row, slot in zip(*np.where(data.data['train']['mask'][ix] > 0)):
            r = metadata[(ids[ix[row]], int(slot))]
            append('support_mean', average[row, slot], r); append('memory', memory[row, slot], r)
    dependency = dict(version=VERSION, status='TRAIN_FIT_COMPLETE', scene=a.scene, source_epochs=binding['source_epochs'],
        method=binding['method'], estimator='train-standardized ridge alpha=1; numeric physical values plus category one-hot',
        fields=field_names(a.scene), fit_split='train', validation_used=False, test_read=False,
        source_optimizer_steps=0, head_optimizer_steps=0, train_plan_epoch=0,
        train_plan_sha256=__import__('hashlib').sha256(np.ascontiguousarray(plan).tobytes()).hexdigest(),
        readout=artifact(args.core), encoding=artifact(rd/'encoding_complete.json'), normalization=artifact(rd/'normalization.json'),
        source_checkpoint=dict(path=binding['checkpoint'], sha256=binding['checkpoint_sha256']),
        head_checkpoint=artifact(head_path), raw_train_relations=artifact(raw), implementation=artifact(__file__),
        runtime=artifact(Path(__file__).with_name('runtime.py')))
    if (out/'probe_fit.json').exists():
        old = read(out/'probe_fit.json')
        if {k:v for k,v in old.items() if k != 'channels'} != dependency: raise ValueError('Use another output for a changed fit')
        for channel in old['channels'].values(): checked(channel['artifact'])
        print('TRAIN_FIT_ALREADY_COMPLETE'); return
    dependency['channels'] = {name: fit(d['x'], d['y'], d['labels'], out/(name+'.npz')) for name, d in datasets.items()}
    dependency['channels']['own']['attribution'] = 'source complete OWN AB+query3 state, not P-only'
    dependency['channels']['support_mean']['attribution'] = 'mean of raw independent S3 complete states'
    dependency['channels']['memory']['attribution'] = 'after frozen supervised head support projection; not source-only information'
    write(out/'probe_fit.json', dependency, immutable=True)
    print('TRAIN_FIT_COMPLETE')


if __name__ == '__main__': main()
