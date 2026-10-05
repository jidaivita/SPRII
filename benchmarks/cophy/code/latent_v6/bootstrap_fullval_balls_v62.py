"""Paired recipient bootstrap of already completed Balls full-validation MSEs."""

import os
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    run = Path(args.run)
    if run != ROOT / 'latent_v6_2_sigcal/weight02':
        raise ValueError('Only the selected source50/head100 cohort is allowed')
    rows = {}
    files = {}
    ids = None
    checkpoints = {}
    for method in ('Base', 'Both', 'Random-Both'):
        folder = run / 'fullval/balls' / method / 'S3'
        result = read(folder / 'results.json')
        freeze = read(folder / 'checkpoint_freeze.json')
        if result['status'] != 'COMPLETE' or result['full_validation_rows'] != 2000:
            raise ValueError('Full Balls evaluation incomplete')
        if freeze['head_budget'] != 100 or '/source50/' not in freeze['checkpoint']:
            raise ValueError('Do not mix source10 or a different head budget')
        if ids is None:
            ids = result['matched']['ids']
        elif ids != result['matched']['ids']:
            raise ValueError('Paired recipient order differs')
        rows[method] = np.asarray(result['matched']['per_recipient_mse'], dtype=np.float64)
        if rows[method].shape != (2000,) or not np.isfinite(rows[method]).all():
            raise ValueError('Invalid full recipient losses')
        checkpoints[method] = dict(selected_epoch=result['selected_epoch'],
                                   checkpoint_sha256=freeze['checkpoint_sha256'])
        for name in ('results.json', 'checkpoint_freeze.json'):
            files[str(folder / name)] = digest(folder / name)
    # Shared resampling draws preserve paired differences and covariance across
    # the two comparisons. Each recipient already aggregates its objects/frames.
    rng = np.random.default_rng(args.seed)
    sample_digest = hashlib.sha256()
    means = {method: [] for method in rows}
    for start in range(0, args.replicates, 100):
        count = min(100, args.replicates - start)
        indices = rng.integers(0, 2000, size=(count, 2000), dtype=np.int64)
        sample_digest.update(indices.tobytes())
        for method, values in rows.items():
            means[method].extend(values[indices].mean(1).tolist())
    means = {method: np.asarray(values) for method, values in means.items()}
    comparisons = {}
    for baseline in ('Base', 'Random-Both'):
        left, right = rows['Both'].mean(), rows[baseline].mean()
        delta = means['Both'] - means[baseline]
        reduction = 100 * (means[baseline] - means['Both']) / means[baseline]
        comparisons['Both-minus-' + baseline] = dict(
            both_mse=float(left), baseline_mse=float(right), mse_difference=float(left - right),
            mse_difference_ci95=np.quantile(delta, [.025, .975]).tolist(),
            relative_reduction_percent=float(100 * (right - left) / right),
            relative_reduction_ci95_percent=np.quantile(reduction, [.025, .975]).tolist())
    result = dict(status='COMPLETE', version='v6.2-balls-fixed-head-paired-bootstrap-1',
                  source_budget=50, head_budget=100, supports=3, recipients=2000,
                  resampling_unit='recipient episode, retaining all aggregated objects and frames',
                  replicates=args.replicates, seed=args.seed, interval='percentile 95%',
                  shared_draws_for_all_models=True, resampling_indices_sha256=sample_digest.hexdigest(),
                  ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                  relative_reduction_definition='100*(mean baseline MSE - mean Both MSE)/mean baseline MSE within each shared resample',
                  scope='uncertainty across validation recipients conditional on the fixed seed0 trained checkpoints; not cross-training-seed uncertainty',
                  test_read=False, optimizer_steps=0, gpu_used=False, checkpoints=checkpoints,
                  comparisons=comparisons, files=files, script_sha256=digest(__file__))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix('.pending.json')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(out)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'latent_v6_2_sigcal/weight02'))
    parser.add_argument('--out', default=str(ROOT / 'latent_v6_2_sigcal/weight02/fullval/balls/paired_bootstrap2000.json'))
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260912)
    main(parser.parse_args())
