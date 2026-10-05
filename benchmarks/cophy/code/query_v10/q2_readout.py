"""Two-frame query, unchanged frozen AB codes, score the shared CD[3:] future."""
import argparse
import hashlib
import json
from pathlib import Path
import types


def transformed(core):
    source = Path(core).read_text()
    patches = [
        ("q=x['pose'].copy(), det=x['detected'].copy()", "q=x['pose'][:, :2].copy(), det=x['detected'][:, :2].copy()"),
        ("(n, 3, self.slots, self.dims)", "(n, 2, self.slots, self.dims)"),
        ("for _ in range(self.horizon):", "for _ in range(self.horizon+1):"),
        ("return torch.stack(result, 1)", "return torch.stack(result, 1)[:, 1:]"),
        ("current_input='frozen official pose estimates for CD[0:3], detection and existing public type; no source g/T features',",
         "current_input='CD[0:2] only; predict t2 internally, score unchanged CD[3:]; no source g/T features', query_frames=2, scoring_start=3, q2_implementation_sha256=Q2_SHA,"),
    ]
    for before, after in patches:
        if source.count(before) != 1:
            raise ValueError('Parent source differs: '+before)
        source = source.replace(before, after)
    compile(source, str(core), 'exec')
    return source


def run(spec_path, device, check=False):
    spec = json.loads(Path(spec_path).read_text())
    core = Path(spec['core'])
    if hashlib.sha256(core.read_bytes()).hexdigest() != spec['core_sha256']:
        raise ValueError('Parent SHA mismatch')
    source = transformed(core)
    if check:
        print(json.dumps(dict(status='PASS', transformations=5, optimizer_steps=0)))
        return
    import torch
    torch.set_num_threads(2)
    out = Path(spec['out']); out.mkdir(parents=True, exist_ok=True)
    parent = Path(spec['parent_readout'])
    for name in ('codes_train.npz', 'codes_val.npz', 'codes_complete.json', 'probes.json'):
        src = parent/name; dst = out/name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError('Different source cache')
        else:
            dst.symlink_to(src)
    m = types.ModuleType('q2_supervised')
    m.__file__ = str(core)
    m.Q2_SHA = hashlib.sha256((source+Path(__file__).read_text()).encode()).hexdigest()
    exec(compile(source, str(core), 'exec'), m.__dict__)
    args = argparse.Namespace(out=str(out), base=spec['base'], scene=spec['scene'],
                              reference='learned', root=spec['root'], device=device,
                              supports=3, epochs=100)
    # Real-batch shape/gradient check; train() creates fresh weights afterwards.
    m.smoke(args)
    m.train(args)
    q2 = m.read(out/'S3/learned/results.json')
    q3 = m.read(parent/'S3/learned/results.json')
    if q2['matched']['ids'] != q3['matched']['ids']:
        raise ValueError('q2/q3 validation cohort mismatch')
    m.write(out/'q2_complete.json', dict(status='COMPLETE', scene=spec['scene'], role=spec['role'],
        source_budget=50, source_frozen=True, query_frames=2, supports=3, scoring_start=3,
        q2_mse=q2['matched']['mse'], q3_mse=q3['matched']['mse'],
        q3_results=str(parent/'S3/learned/results.json'), q3_sha256=m.sha(parent/'S3/learned/results.json'),
        source_probe=str(parent/'probes.json'), source_probe_sha256=m.sha(parent/'probes.json'),
        probe_reused_reason='Identical AB-only source checkpoint and history inputs; query branch is in the new head',
        implementation_sha256=m.Q2_SHA, test_read=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--check-only', action='store_true')
    a = p.parse_args(); run(a.spec, a.device, a.check_only)
