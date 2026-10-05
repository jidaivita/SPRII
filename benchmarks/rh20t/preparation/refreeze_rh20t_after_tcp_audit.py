#!/usr/bin/env python3
"""Create eligibility v2 by removing objectively corrupted TCP trajectories."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TCP_XYZ_ABS_MAX_METRES = 10.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eligible-manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, CACHE, OUTPUT = args.eligible_manifest, args.cache_root, args.output
    source = json.loads(SOURCE.read_text())
    kept, rejected = [], []
    ft_maxima = []
    for index, item in enumerate(source['eligible'], 1):
        path = CACHE / f"{item['episode_id']}.npz"
        with np.load(path) as data:
            tcp = np.asarray(data['tcp_base'], dtype=np.float64)
            ft = np.asarray(data['ft_base_zeroed'], dtype=np.float64)
        xyz = tcp[:, :3]
        max_abs = float(np.max(np.abs(xyz)))
        ft_maxima.append(float(np.max(np.abs(ft))))
        if not np.isfinite(xyz).all() or max_abs > TCP_XYZ_ABS_MAX_METRES:
            rejected.append({
                **item,
                'eligible': False,
                'reason': 'tcp_xyz_corrupt_abs_gt_10m_or_nonfinite',
                'tcp_xyz_max_abs_metres': max_abs,
            })
        else:
            kept.append(item)
        if index % 500 == 0:
            print(json.dumps({'completed': index, 'total': len(source['eligible'])}), flush=True)
    counts = Counter(item['task_id'] for item in kept)
    if set(counts) != set(source['eligible_episodes_per_task']):
        raise RuntimeError('TCP filtering removed an entire task')
    if min(counts.values()) < 4:
        raise RuntimeError('TCP filtering leaves a task with fewer than four episodes')
    payload = {
        **source,
        'schema_version': 2,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'supersedes_manifest': str(SOURCE),
        'supersedes_manifest_sha256': sha256(SOURCE),
        'additional_eligibility_rule': {
            'name': 'tcp_xyz_physical_sanity',
            'definition': 'all cached base-frame TCP XYZ finite and max absolute coordinate <= 10 metres',
            'threshold_metres': TCP_XYZ_ABS_MAX_METRES,
            'rationale': 'removes objectively corrupted sensor/interpolation values before any model smoke or formal run',
            'model_outputs_or_validation_metrics_used': False,
        },
        'eligible_episode_count': len(kept),
        'excluded_episode_count': len(source['excluded']) + len(rejected),
        'eligible_task_count': len(counts),
        'eligible_episodes_per_task': dict(sorted(counts.items())),
        'eligible': kept,
        'excluded': source['excluded'] + rejected,
        'tcp_sanity_rejected': rejected,
        'ft_max_abs_diagnostic': {
            'maximum': max(ft_maxima),
            'p99': float(np.quantile(ft_maxima, 0.99)),
            'p999': float(np.quantile(ft_maxima, 0.999)),
            'note': 'diagnostic only; no FT filtering added',
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f'.{OUTPUT.name}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(OUTPUT)
    summary = {
        'output': str(OUTPUT),
        'sha256': sha256(OUTPUT),
        'eligible': len(kept),
        'rejected': len(rejected),
        'tasks': len(counts),
        'rejected_episode_ids': [item['episode_id'] for item in rejected],
        'ft_max_abs_diagnostic': payload['ft_max_abs_diagnostic'],
    }
    (OUTPUT.parent / 'TCP_SANITY_REFREEZE.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
