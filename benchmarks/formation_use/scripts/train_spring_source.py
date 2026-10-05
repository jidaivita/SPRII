"""Repeat the fixed native Spring source recipe for missing source seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--method', choices=['Both', 'Structure', 'Align', 'Cross'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    native = Path(args.native_root).resolve()
    sys.path[:0] = [str(native / p) for p in ('', 'src', 'a_src', 'extension')]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from persistbench.envs.visual_elastic_coupling.a_pretraining import PretrainingSpec, train_pretraining

    policy = json.loads((native / 'NIGHT_POLICY.json').read_text())
    values = dict(policy['training_spec'])
    assert values['steps'] == 10000 and values['history_frames'] == 96
    assert values['pairs_per_batch'] == 48 and policy['test_read'] is False
    values.update(model_seed=args.seed, sampling_seed=args.seed, stochastic_seed=args.seed)
    values['milestone_steps'] = tuple(values['milestone_steps'])
    snapshot = hashlib.sha256((Path(args.bank) / 'BANK_SNAPSHOT.json').read_bytes()).hexdigest()
    if snapshot != policy['bank_snapshot_sha256']:
        raise ValueError('recovered bank must match the original source recipe')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = train_pretraining(
        args.bank, args.output, name='Split' if args.method == 'Structure' else args.method,
        spec=PretrainingSpec(**values), bank_snapshot_sha256=snapshot,
        device=args.device, workers=8,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
