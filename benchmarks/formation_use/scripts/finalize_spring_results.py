"""Wait for eight published lanes, then aggregate the fixed development results on CPU.

This file is orchestration only. It invokes the frozen scientific summarizers,
does not load research banks, and never starts inference or training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HOSTS = ('new2', 'new5')
METHODS = ('Structure', 'Align', 'Cross', 'Both')
EXPECTED_SOURCES = {(m, s) for m in (*METHODS, 'RelInfoNCE') for s in range(3)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def published_lanes(publish_root):
    """A lane receipt is published only after its final result copies finish."""
    records = []
    pending = []
    for host in HOSTS:
        for gpu in range(4):
            lane = f'{host}_gpu{gpu}'
            path = Path(publish_root) / host / 'lanes' / f'{lane}.COMPLETE.json'
            try:
                record = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # The small receipt itself can still be in the process of copying.
                pending.append(lane)
                continue
            if record.get('test_read') is not False or not record.get('tasks'):
                raise ValueError('invalid completed development lane: ' + str(path))
            records.append(dict(lane=lane, host=host, path=str(path), sha256=sha(path), tasks=record['tasks']))
    return records, pending


def collect_results(publish_root, lanes):
    owners = {}
    for lane in lanes:
        for task in lane['tasks']:
            key = (task['method'], task['seed'])
            if key in owners:
                raise ValueError('duplicate source ownership: ' + repr(key))
            owners[key] = lane['host']
    if set(owners) != EXPECTED_SOURCES:
        raise ValueError('published lane tasks do not cover the fixed 12 Spring + 3 baseline sources')

    def locate(relative, owner):
        candidates = [Path(publish_root) / host / relative for host in HOSTS
                      if (Path(publish_root) / host / relative).is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'exactly one published owner required for {relative}: {candidates}')
        expected = Path(publish_root) / owner / relative
        if candidates[0] != expected:
            raise ValueError('result directory differs from its completed lane owner: ' + str(relative))
        return candidates[0].resolve()

    readers = []
    geometries = []
    for (method, seed), host in sorted(owners.items()):
        if method in METHODS:
            relative = Path('geometry') / f'{method}_s{seed}'
            geometries.append(locate(relative, host) / 'RESULT.json')
        if method in ('Both', 'RelInfoNCE'):
            stage = 'development' if method == 'Both' else 'baseline'
            relative = Path('runs/springworld') / stage / method / f'source{seed}'
            readers.append((relative, locate(Path('results') / relative, host)))
    return readers, geometries


def finalize(publish_root, output_root, spring_code, baseline_code, lanes):
    readers, geometries = collect_results(publish_root, lanes)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    staging = output / 'staging'
    for relative, source in readers:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)
    write(output / 'INPUTS.json', dict(test_read=False, lanes=lanes,
        source_directories=[dict(relative=str(r), published=str(p)) for r, p in readers],
        geometry_results=list(map(str, geometries)),
        spring_code=str(Path(spring_code).resolve()), baseline_code=str(Path(baseline_code).resolve())))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def report(code, name, arguments):
        destination = output / f'{name}.json'
        command = [sys.executable, '-m', 'sprii_next', name.replace('_', '-'), *map(str, arguments), '--output', str(destination)]
        print(json.dumps(dict(event='AGGREGATE', command=command, code=str(code))), flush=True)
        with (output / f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return destination, json.loads(destination.read_text())

    spring_path, spring = report(spring_code, 'spring_report', ['--root', staging / 'runs'])
    geometry_path, geometry = report(spring_code, 'geometry_report', ['--results', *geometries])
    baseline_path, baseline = report(baseline_code, 'baseline_report', ['--root', staging / 'runs'])
    # The summarizers above require their complete registered grids and verify
    # scientific artifacts. This receipt describes counts, not a new analysis.
    result = dict(status='COMPLETE', test_read=False, completed_at=time.time(), completed_lanes=len(lanes),
        spring_readers=spring['complete_readers'], baseline_readers=18,
        total_unique_readers=spring['complete_readers'] + 18,
        geometry_sources=geometry['complete_sources'], baseline_comparison_cells=len(baseline['cells']),
        reports={name:dict(path=str(path), sha256=sha(path)) for name, path in
                 [('spring', spring_path), ('geometry', geometry_path), ('baseline', baseline_path)]},
        inputs_sha256=sha(output / 'INPUTS.json'), merged_runs=str(staging / 'runs'))
    write(output / 'COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--spring-code', required=True)
    parser.add_argument('--baseline-code', required=True)
    parser.add_argument('--wait', action='store_true', help='wait until all eight lanes publish completion receipts')
    parser.add_argument('--poll-seconds', type=int, default=60, choices=(30, 60))
    parser.add_argument('--check-only', action='store_true', help='inspect readiness and ownership without writes or aggregation')
    args = parser.parse_args()
    for code in (args.spring_code, args.baseline_code):
        if not (Path(code) / 'sprii_next/__main__.py').is_file():
            raise FileNotFoundError('frozen CLI not found: ' + code)
    last_pending = None
    while True:
        lanes, pending = published_lanes(args.publish_root)
        if not pending:
            break
        if pending != last_pending:
            print(json.dumps(dict(status='WAITING', completed_lanes=len(lanes), pending_lanes=pending, test_read=False)), flush=True)
            last_pending = pending
        if not args.wait or args.check_only:
            return 2
        time.sleep(args.poll_seconds)
    if args.check_only:
        readers, geometries = collect_results(args.publish_root, lanes)
        result = dict(status='READY', test_read=False, completed_lanes=len(lanes), reader_source_directories=len(readers),
                      geometry_sources=len(geometries), would_execute=['spring-report', 'geometry-report', 'baseline-report'])
    else:
        result = finalize(args.publish_root, args.output_root, args.spring_code, args.baseline_code, lanes)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
