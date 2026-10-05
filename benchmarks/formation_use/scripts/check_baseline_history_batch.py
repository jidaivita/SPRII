"""One real CPU batch: exact input equivalence and preparation timing, no training."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sprii_next.io import development_path, digest, read, write
from sprii_next.protocol import activate_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sweep', type=int, default=0)
    parser.add_argument('--batch-index', type=int, default=0)
    parser.add_argument('--output')
    args = parser.parse_args()
    activate_native(args.native_root)
    from native_training import NativePairSchedule, make_batch, visible
    from sprii_next.contrastive import PAIRS_PER_BATCH, make_history_batch
    torch.set_num_threads(1)
    bank = development_path(args.bank).resolve()
    schedule = NativePairSchedule(read(bank / 'MANIFEST.private.json'), seed=args.seed,
                                  pairs_per_batch=PAIRS_PER_BATCH)
    plan = schedule.sweep(args.sweep, 'Both')
    size = schedule.batch_pairs
    selected = plan['pairs'][args.batch_index * size:(args.batch_index + 1) * size]
    # Both timed calls see the same native public-episode cache. This isolates
    # preparation rather than crediting the second path for the first disk read.
    start = time.perf_counter()
    for branch in ('donor', 'recipient'):
        for pair in selected:
            row = schedule.rows[pair[branch + '_episode']]
            asset = row['assets']['128']
            visible(str(bank), asset['path'], asset['sha256'])
    warm_seconds = time.perf_counter() - start
    start = time.perf_counter()
    original, original_receipt = make_batch(schedule, plan, args.batch_index, bank)
    original_seconds = time.perf_counter() - start
    start = time.perf_counter()
    history_only, receipt = make_history_batch(schedule, plan, args.batch_index, bank)
    history_only_seconds = time.perf_counter() - start
    for name in ('history_images', 'history_actions'):
        reference = getattr(original, name).numpy()
        actual = getattr(history_only, name).numpy()
        assert reference.dtype == actual.dtype == np.float32
        np.testing.assert_array_equal(reference.view(np.uint32), actual.view(np.uint32))
    identity_keys = ('plan_sha256', 'batch_index', 'input_assets', 'pair_sha256', 'windows',
        'raw_image_frames_read', 'presented_history_frames', 'observed_action_intervals',
        'direction', 'resolution', 'physical_labels_read', 'test_read')
    assert {k: original_receipt[k] for k in identity_keys} == {k: receipt[k] for k in identity_keys}
    assert receipt['target_frames'] == receipt['future_targets_consumed'] == 0
    assert receipt['target_horizons'] == [] and not hasattr(history_only, 'target_images')
    # Training does this once inside the objective on its device. It is reported
    # separately here, rather than confusing CPU preparation with GPU execution.
    start = time.perf_counter()
    history_only.validate(96)
    validation_seconds = time.perf_counter() - start
    result = dict(status='PASS', seed=args.seed, sweep=args.sweep, batch_index=args.batch_index,
        pairs=size, history_images_bitwise_equal=True, history_actions_bitwise_equal=True,
        pair_identity_equal=True, pair_identity_sha256=digest(receipt['pair_sha256']),
        native_cache_warm_seconds=warm_seconds, original_preparation_seconds=original_seconds,
        history_only_preparation_seconds=history_only_seconds,
        cpu_preparation_speedup=original_seconds / history_only_seconds,
        separate_cpu_history_validation_seconds=validation_seconds,
        original_prepared_target_frames=original_receipt['target_frames'],
        history_only_prepared_target_frames=receipt['target_frames'],
        timing_scope='one cache-warmed CPU batch; not end-to-end GPU training throughput',
        optimizer_updates=0, test_read=False)
    if args.output:
        write(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
