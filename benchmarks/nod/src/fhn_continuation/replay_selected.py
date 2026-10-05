"""Replay the frozen FHN stage recipe through the original training entrypoints.

This portable dispatcher replaces only machine queues/path binding. By default it
prints the commands; --execute is required to start the requested single run.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
NOD = SRC.parent


def commands(options):
    config = json.loads(options.recipe.read_text())
    previous = None
    result = []
    for index, stage in enumerate(config['recipes'][options.method]):
        output = options.output.resolve() / f'stage{index}'
        name = f'{options.method}_s{options.seed}_stage{index}'
        tuning = stage['tuning_options']
        restart = stage['restart_lr']
        if restart is not None:
            command = [sys.executable, '-u', str(HERE / 'fhn_lr_restart.py'), '--restart-lr', str(restart)]
        else:
            command = [sys.executable, '-u', '-m', 'fhn_minimal.tune_train' if tuning else 'fhn_minimal.train']
            if tuning:
                command += ['--horizon-weight-power', str(tuning.get('horizon_weight_power', 0))]
                if tuning.get('freeze_encoder'):
                    command += ['--freeze-encoder']
        fields = dict(stage['args'], seed=options.seed, data_dir=str(options.data_dir.resolve()),
                      official_code=str(options.official_code.resolve()), output_dir=str(output),
                      run_name=name, resume=str(previous) if previous else None,
                      max_steps=stage['end_step'], save_every=2500, log_every=250, cpu_threads=4,
                      device=options.device)
        for key, value in fields.items():
            if value is not None:
                command += ['--' + key.replace('_', '-'), str(value)]
        previous = output / f'{name}_step{stage["end_step"]:07d}.pth'
        result.append({'stage': index, 'end_step': stage['end_step'], 'command': command,
                       'checkpoint': str(previous)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['nod', 'sprii'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--official-code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, default=NOD / 'configs/fhn_selected_three_seed.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    plan = commands(options)
    print(json.dumps(plan, indent=2))
    if not options.execute:
        return
    if options.output.exists():
        raise FileExistsError('Use a fresh output directory; no implicit overwrite/resume.')
    options.output.mkdir(parents=True)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(SRC) + os.pathsep + environment.get('PYTHONPATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for stage in plan:
        subprocess.run(stage['command'], env=environment, check=True)
        if not Path(stage['checkpoint']).is_file():
            raise FileNotFoundError(stage['checkpoint'])


if __name__ == '__main__':
    main()
