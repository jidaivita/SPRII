"""Freeze exactly the32 already-planned CPC/RSSM heads as evaluation tails."""

import os
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy")))
VERSION = 'cophy-extensions-fullval32-v6.3-1'
CODE = ROOT / 'source/latent_extension_fullval_v6_3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    original = json.loads(args.extensions_manifest.read_text())
    if len(original['tasks']) != 116 or original['readout_heads'] != 32:
        raise ValueError('Expected the original116-task extension contract')
    heads = [row for row in original['tasks'] if row['kind'] == 'head']
    tasks = []
    common = ROOT / 'source/latent_v6/fullval_readout.py'
    guard = ROOT / 'source/latent_extension_v6_3/gpu_guard.py'
    local = lambda remote: args.code_root / Path(remote).relative_to(ROOT / 'source')
    bound = {str(p): sha(local(p)) for p in (
        common, guard, CODE / 'family_fullval.py', CODE / 'tail_worker.py', CODE / 'build_tasks.py')}
    reserve = ROOT / 'latent_v6_2_sigcal/weight02/fullval/balls/complete.json'
    for row in heads:
        command = row['commands'][0]
        value = lambda flag: command[command.index(flag) + 1]
        family, scene = row['family'], value('--scene')
        supports, readout = int(value('--supports')), Path(value('--out'))
        method = readout.parent.name
        if family not in ('CPC', 'RSSM') or method not in ('Base', 'Cross', 'Both', 'Random-Both'):
            raise ValueError('Unexpected original head identity')
        if value('--epochs') != '100' or value('--reference') != 'learned' or readout.name != 'source50':
            raise ValueError('Only planned fixed source50/head100 learned heads allowed')
        if supports not in (3, 8) or (supports == 8 and scene != 'collision'):
            raise ValueError('Unexpected support budget')
        core = Path(command[2])
        expected = original['code_sha256'][str(core)]
        if sha(local(core)) != expected:
            raise ValueError('Family core does not match the original manifest: ' + str(core))
        bound[str(core)] = expected
        prepared = ROOT / 'latent_v6_2/fullval_inputs' / scene
        out = ROOT / 'latent_extensions_v6_3/fullval' / scene / family / method / ('S' + str(supports))
        argv = [str(ROOT / 'venv/bin/python'), '-u', str(CODE / 'family_fullval.py'), 'evaluate',
                '--family', family, '--family-core', str(core), '--family-core-sha256', expected,
                '--common-eval', str(common), '--common-eval-sha256', bound[str(common)],
                '--scene', scene, '--prepared', str(prepared), '--readout', str(readout),
                '--supports', str(supports), '--device', '{device}', '--out', str(out)]
        headfolder = Path(row['marker']).parent
        tasks.append(dict(
            id=row['id'].replace('-head-', '-fullval-'), original_head_task=row['id'],
            family=family, scene=scene, method=method, supports=supports, argv=argv,
            marker=str(out / 'family_complete.json'),
            dependencies=[str(reserve), str(prepared / 'prepared.json'), row['marker']],
            files=[str(headfolder / name) for name in ('config.json', 'results.json', 'selected.pt', 'selected_validation.json')]
                  + [str(readout / 'codes_complete.json')],
            original_head_failure=str(ROOT / 'latent_extensions_v6_3/jobs' / row['id'] / 'failure.json'),
        ))
    expected_set = {(f, s, m, n) for f in ('CPC', 'RSSM')
                    for s in ('balls', 'collision', 'blocktower')
                    for m in ('Base', 'Cross', 'Both', 'Random-Both')
                    for n in ((3, 8) if s == 'collision' else (3,))}
    actual = {(t['family'], t['scene'], t['method'], t['supports']) for t in tasks}
    if actual != expected_set or len(tasks) != 32:
        raise ValueError('Evaluation matrix differs from24 S3 plus8 Collision S8')
    result = dict(
        version=VERSION, original_extensions_manifest=str(ROOT / 'latent_extensions_v6_3/manifest.json'),
        original_extensions_manifest_sha256=sha(args.extensions_manifest),
        code_sha256=bound, guard=str(guard), tasks=tasks, S3_evaluations=24,
        Collision_S8_evaluations=8, total=32, source_training_jobs=0, head_training_jobs=0,
        old512_reproduction_required=True, selected_checkpoint_unchanged=True,
        prepared_inputs_reused=True, first_priority_marker=str(reserve),
        test_read=False, optimizer_steps=0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and json.loads(args.out.read_text()) != result:
        raise ValueError('Existing32-task evaluation binding differs')
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='FROZEN', tasks=32, manifest=str(args.out),
                          manifest_sha256=sha(args.out), optimizer_steps=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extensions-manifest', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=ROOT / 'source')
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args())

