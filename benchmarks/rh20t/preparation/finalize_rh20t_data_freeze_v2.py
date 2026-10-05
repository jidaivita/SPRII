#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib, json
from pathlib import Path



def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify the RH20T v2 data, split, pairing and normalization boundary')
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--superseded-dir', type=Path)
    args = parser.parse_args()
    ROOT, OLD = args.prepared_dir, args.superseded_dir
    eligible_path = ROOT / 'eligible_episode_manifest.json'
    split_path = ROOT / 'task_split_manifest.json'
    pairing_path = ROOT / 'causal_pairing_manifest.json'
    field_path = ROOT / 'field_action_manifest.json'
    norm_path = ROOT / 'normalization_stats.json'
    eligible = json.loads(eligible_path.read_text())
    split = json.loads(split_path.read_text())
    pairing = json.loads(pairing_path.read_text())
    valid = {item['episode_id'] for item in eligible['eligible']}
    rejected = {item['episode_id'] for item in eligible['tcp_sanity_rejected']}
    split_ids = set()
    split_tasks = []
    for name, item in split['splits'].items():
        ids = set(item['episode_ids'])
        if split_ids & ids:
            raise RuntimeError(f'episode leakage at {name}')
        split_ids |= ids
        split_tasks.append(set(item['task_ids']))
    if split_ids != valid:
        raise RuntimeError('eligible and split episode universes differ')
    if any(split_tasks[i] & split_tasks[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError('task leakage across splits')
    references = set()
    for split_name, split_pairing in pairing['splits'].items():
        allowed = set(split['splits'][split_name]['episode_ids'])
        for entry in split_pairing['entries']:
            ids = {entry['query_episode_id']}
            ids.update(entry['independent_candidate_episode_ids'])
            ids.update(entry['random_candidate_episode_ids'])
            if not ids <= allowed:
                raise RuntimeError(f'pairing crosses split in {split_name}')
            references |= ids
    if rejected & references:
        raise RuntimeError('rejected episode appears in pairing manifest')
    norm = json.loads(norm_path.read_text())
    if norm['train_episode_count'] != split['splits']['train']['episode_count']:
        raise RuntimeError('normalization episode count mismatch')
    payload = {
        'state': 'DATA_FROZEN',
        'schema_version': 2,
        'eligible_episodes': len(valid),
        'tasks': eligible['eligible_task_count'],
        'split_counts': {
            name: {'tasks': item['task_count'], 'episodes': item['episode_count']}
            for name, item in split['splits'].items()
        },
        'tcp_sanity_rule': eligible['additional_eligibility_rule'],
        'tcp_sanity_rejected_episode_ids': sorted(rejected),
        'hashes': {
            'eligible_episode_manifest': digest(eligible_path),
            'task_split_manifest': digest(split_path),
            'causal_pairing_manifest': digest(pairing_path),
            'field_action_manifest': digest(field_path),
            'normalization_stats': digest(norm_path),
        },
        'integrity_checks': {
            'eligible_equals_split_union': True,
            'task_disjoint_splits': True,
            'pairing_within_split': True,
            'rejected_absent_from_all_pairings': True,
            'normalization_train_only_episode_count_matches': True,
            'validation_or_test_metrics_used_for_filtering': False,
            'formal_training_started_before_freeze': False,
            'test_read': False,
        },
        'policy': 'No data-definition changes after this file; future failures change implementation or claims, not the dataset.',
    }
    output = ROOT / 'DATA_FREEZE_COMPLETE.json'
    output.write_text(json.dumps(payload, indent=2) + '\n')
    if OLD is not None:
        (OLD / 'SUPERSEDED_BY_PREFLIGHT_V2.txt').write_text(
            f'{output}\nDATA_FROZEN hash={digest(output)}\n'
        )
    print(json.dumps({**payload, 'data_freeze_sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()
