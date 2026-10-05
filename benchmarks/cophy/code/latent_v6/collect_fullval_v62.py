"""Read-only summary of fixed source50/head100 full-validation results."""

import os
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
METHODS = ('Base', 'Both', 'Cross', 'Align', 'Random-Both')


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def mean(values):
    return sum(values) / len(values) if values else None


def gain(base, value):
    return 100 * (base - value) / base if base else None


def collect(run, out):
    if str(run) != str(ROOT / 'latent_v6_2_sigcal/weight02'):
        raise ValueError('Only the selected weight02 source50 cohort is allowed')
    result = dict(version='v6.2-fullval-summary-1', run=str(run), collected_at=time.time(),
                  test_read=False, source_budget=50, head_budget=100, scenes={}, files={},
                  queue=read(run / 'fullval_tail_queue/status.json'))
    complete_groups = 0
    for scene, sizes in (('balls', (3,)), ('collision', (3, 8)), ('blocktower', (3,))):
        result['scenes'][scene] = {}
        for size in sizes:
            loaded = {}
            waiting = []
            for method in (*METHODS, 'Query-only', 'Known-parameters'):
                if method in METHODS:
                    path = run / 'fullval' / scene / method / f'S{size}/results.json'
                else:
                    ref = 'query' if method == 'Query-only' else 'known'
                    path = ROOT / 'latent_v6_2/fullval' / scene / 'references/S3' / ref / 'results.json'
                row = read(path)
                if row is None or row.get('status') != 'COMPLETE':
                    waiting.append(method)
                    continue
                if row.get('test_read') is not False or row.get('optimizer_steps') != 0:
                    raise ValueError('Unexpected evaluation scope')
                if row['scene'] != scene or (method in METHODS and row['supports'] != size):
                    raise ValueError('Wrong scene/support result')
                freeze = read(path.parent / 'checkpoint_freeze.json')
                if freeze['head_budget'] != 100 or (method in METHODS and '/source50/' not in freeze['checkpoint']):
                    raise ValueError('Do not mix source10 or unfinished heads')
                loaded[method] = row
                result['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            group = dict(status='COMPLETE' if not waiting else 'PENDING', waiting=waiting, methods={})
            expected_ids = None
            for method, row in loaded.items():
                ids = row['matched']['ids']
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise ValueError('Mismatched full-validation recipient order')
                item = dict(full_rows=len(ids), selected_epoch=row['selected_epoch'],
                            matched_mse=row['matched']['mse'],
                            original512_max_error=max(v['max_absolute_error'] for v in row['reproduction'].values()))
                for arm in ('null', 'wrong'):
                    if arm in row:
                        item[arm + '_mse'] = row[arm]['mse']
                if 'wrong' in row:
                    item.update(wrong_coverage=row['wrong_coverage'],
                                matched_on_wrong_cohort=row['matched_on_wrong_cohort']['mse'],
                                null_on_wrong_cohort=row['null_on_wrong_cohort']['mse'],
                                history_gain_percent=row['history_gain_percent'],
                                correct_vs_wrong_percent=row['correct_vs_wrong_percent'])
                group['methods'][method] = item
            if 'Base' in loaded:
                base = loaded['Base']['matched']['mse']
                group['improvement_over_base_percent'] = {m: gain(base, loaded[m]['matched']['mse'])
                    for m in METHODS[1:] if m in loaded}
            if not waiting:
                common = set(expected_ids)
                for method in METHODS:
                    common &= set(loaded[method]['wrong']['ids'])
                common_ids = [ident for ident in expected_ids if ident in common]
                common_table = {}
                for method, row in loaded.items():
                    item = {}
                    for arm in ('matched', 'null', 'wrong'):
                        if arm in row:
                            byid = dict(zip(row[arm]['ids'], row[arm]['per_recipient_mse']))
                            item[arm + '_mse'] = mean([byid[q] for q in common_ids])
                    common_table[method] = item
                group.update(common_wrong_rows=len(common_ids), common_wrong_table=common_table,
                             cohort_size=len(expected_ids))
                complete_groups += 1
            result['scenes'][scene][f'S{size}'] = group
            if not waiting:
                marker = run / 'fullval' / scene / f'S{size}' / 'complete.json'
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps(dict(status='COMPLETE', source_budget=50, head_budget=100,
                    supports=size, full_rows=len(expected_ids), methods=list(loaded),
                    test_read=False, optimizer_steps=0), indent=2) + '\n')
                if scene == 'balls':
                    # Natural-yield reservation ends after all five fixed heads
                    # and both references, not after the lower-priority legacy tail.
                    (run / 'fullval/balls/complete.json').write_text(marker.read_text())
    result.update(status='COMPLETE' if complete_groups == 4 else 'PARTIAL',
                  complete_scene_support_groups=complete_groups, expected_scene_support_groups=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.pending.json')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(out)
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'latent_v6_2_sigcal/weight02')
    parser.add_argument('--out', type=Path, default=ROOT / 'latent_v6_2_sigcal/weight02/fullval/summary.json')
    parser.add_argument('--watch-balls', action='store_true')
    parser.add_argument('--watch-all', action='store_true')
    parser.add_argument('--hours', type=float, default=4)
    args = parser.parse_args()
    deadline = time.time() + args.hours * 3600
    while True:
        result = collect(args.run, args.out)
        if ((args.watch_all and result['status'] == 'COMPLETE') or
            (args.watch_balls and not args.watch_all and result['scenes']['balls']['S3']['status'] == 'COMPLETE') or
            (not args.watch_balls and not args.watch_all)):
            break
        if time.time() >= deadline:
            raise TimeoutError('Full-validation watcher reached deadline before the requested cohort completed')
        time.sleep(15)
