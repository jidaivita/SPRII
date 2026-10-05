"""One serial task lane per GPU; start downstream work as soon as its source exits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--job-status-root', help='Directory containing optional per-job done.rc files; defaults to ROOT/jobs')
    parser.add_argument('--native-root', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--spring-code', required=True)
    parser.add_argument('--baseline-code', required=True)
    parser.add_argument('--publish-root', required=True)
    parser.add_argument('--geometry-only-script', help='Optional donor-only extractor for Structure/Align/Cross geometry')
    args = parser.parse_args()
    root = Path(args.root)
    tasks = json.loads(Path(args.plan).read_text())
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(code, command):
        command = list(map(str, command))
        print(json.dumps({'event': 'START', 'command': command, 'at': time.time()}), flush=True)
        subprocess.run([sys.executable, '-u', *command], cwd=code, env=env, check=True)

    def publish(path):
        destination = Path(args.publish_root) / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)

    for task in tasks:
        method, seed = task['method'], task['seed']
        baseline = method == 'RelInfoNCE'
        geometry_only = bool(args.geometry_only_script) and method in ('Structure', 'Align', 'Cross')
        code = Path(args.baseline_code if baseline else args.spring_code)
        source = root / 'sources' / f'{method}_s{seed}'
        cache = root / 'cache' / f'{method}_s{seed}'
        protocol = root / 'protocols' / f'{method}_s{seed}.json'
        if task.get('wait_handle'):
            exit_file = (Path(args.job_status_root) if args.job_status_root else root / 'jobs') / task['wait_handle'] / 'done.rc'
            print(json.dumps({'event': 'WAIT_SOURCE_EXIT', 'task': task}), flush=True)
            while not exit_file.exists():
                time.sleep(30)
            if int(exit_file.read_text().strip()) != 0:
                raise RuntimeError('source process failed: ' + task['wait_handle'])
        if task.get('train') and not (source / 'COMPLETE.json').exists():
            command = ['-m', 'sprii_next.contrastive', 'train'] if baseline else [root / 'train_spring_source.py', '--method', method]
            run(code, [*command, '--native-root', args.native_root, '--bank', args.bank,
                       '--seed', seed, '--output', source, '--device', 'cuda:0'])
        if not (source / 'COMPLETE.json').exists():
            raise FileNotFoundError(source / 'COMPLETE.json')
        if not geometry_only and not (cache / 'SOURCE.json').exists():
            command = ['-m', 'sprii_next.contrastive', 'export'] if baseline else ['-m', 'sprii_next', 'export-spring', '--method', method, '--seed', seed]
            run(code, [*command, '--native-root', args.native_root, '--bank', args.bank,
                       '--completion', source / 'COMPLETE.json', '--output', cache, '--device', 'cuda:0'])
        if not geometry_only and not protocol.exists():
            run(code, ['-m', 'sprii_next', 'assemble', '--sources', cache / 'SOURCE.json', '--output', protocol])
        if not baseline:
            geometry = root / 'geometry' / f'{method}_s{seed}'
            if not (geometry / 'RESULT.json').exists():
                if geometry_only:
                    run(code, [args.geometry_only_script, '--native-root', args.native_root,
                               '--bank', args.bank, '--completion', source / 'COMPLETE.json',
                               '--method', method, '--seed', seed, '--output', geometry,
                               '--device', 'cuda:0'])
                else:
                    run(code, ['-m', 'sprii_next', 'geometry', '--protocol', protocol,
                               '--method', method, '--seed', seed, '--output', geometry])
            publish(geometry)
        if method in ('Both', 'RelInfoNCE'):
            stage = 'baseline' if baseline else 'development'
            run(code, [code / 'scripts/run_grid.py', '--protocol', protocol,
                       '--output-root', root / 'results', '--environment', 'springworld',
                       '--stage', stage, '--source-seeds', seed, '--devices', 'cuda:0'])
            publish(root / 'results/runs/springworld' / stage / method / f'source{seed}')
        receipt = json.loads((source / 'COMPLETE.json').read_text())
        checkpoint = receipt.get('selected_checkpoint', Path(receipt.get('checkpoint', 'source.pt')).name)
        for name in ('COMPLETE.json', 'RUN.json', 'DIAGNOSTICS.json', checkpoint):
            if (source / name).is_file():
                publish(source / name)
        if protocol.exists():
            publish(protocol)
        print(json.dumps({'event': 'TASK_COMPLETE', 'method': method, 'seed': seed, 'test_read': False}), flush=True)
    complete = root / 'lanes' / (Path(args.plan).stem + '.COMPLETE.json')
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text(json.dumps({'tasks': tasks, 'test_read': False, 'completed_at': time.time()}, indent=2) + '\n')
    publish(complete)


if __name__ == '__main__':
    main()
