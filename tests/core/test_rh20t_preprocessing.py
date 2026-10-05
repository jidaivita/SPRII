"""Synthetic raw-data tests for alignment, decoding and fixed task splitting."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2', reason='install requirements/rh20t.txt for raw-video tests')
ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'benchmarks/rh20t/preparation'
spec = importlib.util.spec_from_file_location('rh20t_cache_release', PREP / 'build_rh20t_cache_v1.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def run_script(name, *args):
    return subprocess.run([sys.executable, str(PREP / name), *map(str, args)],
                          check=True, capture_output=True, text=True, timeout=60)


def make_scene(root, camera='camera_fixture', frames=64, nested=False):
    scene = root / 'task_0001_user_0001_scene_0001_cfg_0001'
    transformed = scene / 'transformed'
    transformed.mkdir(parents=True)
    video_dir = scene / f'cam_{camera}'
    if nested:
        video_dir = video_dir / 'color'
    video_dir.mkdir(parents=True)
    (scene / 'metadata.json').write_text(json.dumps({'rating': 3, 'action': 1, 'calib': 1, 'finish_time': frames * 100}))
    ts = np.arange(frames, dtype=np.int64) * 100
    np.save(video_dir / 'timestamps.npy', ts)
    video = cv2.VideoWriter(str(video_dir / 'color.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
    assert video.isOpened(), 'OpenCV test installation lacks mp4v support'
    for i in range(frames):
        video.write(np.full((32, 32, 3), i, np.uint8))
    video.release()
    np.save(transformed / 'force_torque_base.npy', {camera: [dict(timestamp=int(t), zeroed=np.full(6, i / 100)) for i, t in enumerate(ts)]})
    np.save(transformed / 'tcp_base.npy', {camera: [dict(timestamp=int(t), tcp=np.array([0., 0., i / 100, 1., 0., 0., 0.])) for i, t in enumerate(ts)]})
    np.save(transformed / 'gripper.npy', {camera: {0: {'gripper_command': [2., 0., 0]}, 100: {'gripper_command': [9., 0., 350]}}})
    return scene


@pytest.mark.parametrize('nested', [False, True])
def test_raw_video_to_cache_keeps_causal_command_and_shapes(tmp_path, nested):
    source = tmp_path / 'raw'
    scene = make_scene(source, nested=nested)
    result = cache.process_one((str(scene), str(tmp_path / 'cache'), str(source), 'camera_fixture'))
    assert result['eligible'], result
    with np.load(result['cache_path']) as data:
        assert data['rgb_gray'].shape == (64, 96, 96)
        assert data['ft_base_zeroed'].shape == (64, 6)
        assert data['tcp_base'].shape == (64, 7)
        np.testing.assert_array_equal(data['gripper_command_width'][:5, 0], [2, 2, 2, 2, 9])
        assert np.all(data['gripper_command_issue_ms'] <= data['timestamps_ms'])
        np.testing.assert_allclose(np.linalg.norm(data['tcp_base'][:, 3:7], axis=1), 1)


def test_alignment_collapses_duplicate_timestamp_and_quaternion_sign(tmp_path):
    (tmp_path / 'transformed').mkdir()
    np.save(tmp_path / 'transformed/tcp_base.npy', {'fixture': [
        {'timestamp': 0, 'tcp': [0, 0, 0, 1, 0, 0, 0]},
        {'timestamp': 100, 'tcp': [1, 0, 0, -1, 0, 0, 0]},
        {'timestamp': 100, 'tcp': [2, 0, 0, -1, 0, 0, 0]},
    ]})
    value = cache.aligned_branch(tmp_path, 'tcp_base.npy', 'tcp', np.array([0, 50, 100]), 'fixture')
    np.testing.assert_allclose(value[:, 0], [0, 1, 2])
    np.testing.assert_allclose(value[:, 3], [1, 1, 1])


def test_inventory_reads_public_scene_layout(tmp_path):
    source = tmp_path / 'raw'
    make_scene(source)
    out = tmp_path / 'inventory'
    run_script('rh20t_inventory_preflight.py', '--dataset-root', source, '--output-dir', out,
               '--sample-size', 0, '--probe-camera-count', 0)
    summary = json.loads((out / 'RH20T_CFG1_INVENTORY_SUMMARY.json').read_text())
    assert summary['episode_count'] == 1
    assert summary['camera_video_coverage']['camera_fixture'] == 1


def test_tcp_sanity_filter_preserves_valid_episodes(tmp_path):
    records = []
    for i in range(5):
        episode_id = f'episode_{i}'
        records.append({'episode_id': episode_id, 'task_id': 'task_0001', 'eligible': True})
        xyz = np.full((64, 7), 20. if i == 4 else 1., dtype=np.float32)
        np.savez(tmp_path / f'{episode_id}.npz', tcp_base=xyz, ft_base_zeroed=np.ones((64, 6)))
    source = tmp_path / 'eligible.json'
    source.write_text(json.dumps({'eligible': records, 'excluded': [], 'eligible_episodes_per_task': {'task_0001': 5}}))
    output = tmp_path / 'v2/eligible_episode_manifest.json'
    run_script('refreeze_rh20t_after_tcp_audit.py', '--eligible-manifest', source,
               '--cache-root', tmp_path, '--output', output)
    result = json.loads(output.read_text())
    assert len(result['eligible']) == 4
    assert result['tcp_sanity_rejected'][0]['episode_id'] == 'episode_4'
    assert result['additional_eligibility_rule']['model_outputs_or_validation_metrics_used'] is False


def test_fixed_task_split_is_disjoint_and_pairs_stay_in_split(tmp_path):
    records = [{'episode_id': f'task_{task:04d}_episode_{e}', 'task_id': f'task_{task:04d}', 'frames': 64}
               for task in range(124) for e in range(4)]
    source = tmp_path / 'eligible.json'
    source.write_text(json.dumps({'eligible': records, 'camera_serial': 'fixture'}))
    out = tmp_path / 'prepared'
    run_script('freeze_rh20t_manifests.py', '--eligible-manifest', source, '--output-dir', out)
    split = json.loads((out / 'task_split_manifest.json').read_text())['splits']
    pairing = json.loads((out / 'causal_pairing_manifest.json').read_text())['splits']
    assert [split[x]['task_count'] for x in ['train', 'validation', 'test']] == [74, 25, 25]
    sets = [set(split[x]['task_ids']) for x in ['train', 'validation', 'test']]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    for name in split:
        allowed = set(split[name]['episode_ids'])
        for entry in pairing[name]['entries']:
            assert set(entry['independent_candidate_episode_ids']) <= allowed
            assert set(entry['random_candidate_episode_ids']) <= allowed
            assert entry['query_episode_id'] not in entry['independent_candidate_episode_ids']
