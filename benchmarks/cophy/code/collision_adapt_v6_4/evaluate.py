"""Evaluate fixed Collision heads through the unchanged common fullval engine.

Supports the original init-only readout and the independently implemented gated
step-memory readout. Code/cache versions are separate: the new head deliberately
consumes the original frozen P64 cache. No training or checkpoint selection.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

VERSION = 'collision-adaptation-v6.4-fixed-fullval-1'
OLD_CORE = 'latent-relation-v6.2-frozen-P64-pose-prefix-readout'
STEP_CORE = 'collision-v6.4-gated-step-memory-readout'
ALLOWED_CORES = {OLD_CORE: OLD_CORE, STEP_CORE: OLD_CORE}
COMMON_VERSION = 'latent-v6.2-fullval-fixed-P64-readout-1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('Different fixed evaluation binding: ' + str(path))
    if not Path(path).exists():
        write(path, value)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def load_cores(args):
    core_sha, common_sha = sha(args.readout_core), sha(args.common_eval)
    if args.readout_core_sha256 and core_sha != args.readout_core_sha256:
        raise ValueError('Readout implementation differs from requested hash')
    if args.common_eval_sha256 and common_sha != args.common_eval_sha256:
        raise ValueError('Common evaluator differs from requested hash')
    core = module('_collision_v64_readout_' + core_sha[:12], args.readout_core)
    if core.VERSION not in ALLOWED_CORES:
        raise ValueError('Only the original v6.2 or gated v6.4 readout is supported')
    expected_cache = ALLOWED_CORES[core.VERSION]
    if getattr(core, 'CACHE_VERSION', core.VERSION) != expected_cache:
        raise ValueError('Unexpected P64 cache format')
    # Dependency injection is local to this new process. No old module changes.
    sys.modules['readout'] = core
    common = module('_collision_v64_common_fullval', args.common_eval)
    if common.core is not core or common.VERSION != COMMON_VERSION:
        raise ValueError('Common engine did not bind the requested readout')
    for name in ('Data', 'Head', 'evaluate', 'immutable', 'sha'):
        if not hasattr(core, name):
            raise ValueError('Missing readout interface: ' + name)
    if not hasattr(common, 'FullData') or not callable(common.evaluate):
        raise ValueError('Missing common full-validation interface')
    identity = dict(version=VERSION, readout_version=core.VERSION,
        cache_version=expected_cache, readout_core=str(Path(args.readout_core).resolve()),
        readout_core_sha256=core_sha, common_eval=str(Path(args.common_eval).resolve()),
        common_eval_sha256=common_sha, common_eval_version=common.VERSION,
        wrapper=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        optimizer_steps=0, test_read=False)
    return core, common, identity


def prerequisites(args):
    folder = Path(args.readout); head = folder/'S3/learned'
    required = [folder/'codes_complete.json', folder/'codes_train.npz', folder/'codes_val.npz',
        head/'complete.json', head/'config.json', head/'selected.pt', head/'selected_validation.json',
        head/'results.json', Path(args.prepared)/'prepared.json']
    missing = [str(path) for path in required if not path.is_file()]
    for path in (folder/'codes_complete.json', head/'complete.json'):
        if path.is_file() and read(path).get('status') != 'COMPLETE':
            missing.append(str(path) + ' is not COMPLETE')
    if missing:
        write(Path(args.out)/'waiting.json', dict(status='WAITING', version=VERSION,
              dependencies=missing, optimizer_steps=0, test_read=False))
        print(json.dumps(dict(status='WAITING', dependencies=missing)), flush=True)
        return False
    return True


def head_identity(args, core, identity):
    import torch
    folder = Path(args.readout); head = folder/'S3/learned'
    config, complete = read(head/'config.json'), read(head/'complete.json')
    codes = read(folder/'codes_complete.json')
    prepared = read(Path(args.prepared)/'prepared.json')
    if complete.get('epochs') != 100 or config.get('epochs') != 100:
        raise ValueError('Requires the fixed100-epoch head budget')
    if config.get('version') != core.VERSION or config.get('scene') != 'collision' or config.get('reference') != 'learned' or config.get('supports') != 3:
        raise ValueError('Wrong readout identity or support budget')
    if config.get('test_read') is not False or codes.get('test_read') is not False:
        raise ValueError('Expected validation-only data provenance')
    if codes.get('version') != identity['cache_version'] or codes.get('representation') != 'P64':
        raise ValueError('Wrong frozen representation/cache version')
    if (prepared.get('version') != COMMON_VERSION or prepared.get('scene') != 'collision'
            or prepared.get('test_read') is not False or len(prepared['query_ids']) != 4000
            or len(prepared['selection_ids']) != 512):
        raise ValueError('Requires the unchanged full4000 validation preparation')
    ck = torch.load(head/'selected.pt', map_location='cpu', weights_only=False)
    if ck['config'] != config:
        raise ValueError('Checkpoint configuration differs from its receipt')
    gate = None
    if core.VERSION == STEP_CORE:
        if config.get('head_implementation_sha256') != identity['readout_core_sha256']:
            raise ValueError('Gated head differs from its actual training implementation')
        if config.get('memory_mode') not in ('init-only', 'per-step'):
            raise ValueError('Missing registered memory mode')
        gate = int(config['memory_mode'] == 'per-step')
        if config.get('memory_gate') != gate or complete.get('memory_mode') != config['memory_mode']:
            raise ValueError('Memory-mode receipts disagree')
        stored = ck['model'].get('memory_gate')
        branch = ck['model'].get('step_memory.weight')
        if stored is None or stored.numel() != 1 or float(stored) != gate:
            raise ValueError('Checkpoint does not contain the registered memory gate')
        if branch is None or tuple(branch.shape) != (128, 64) or not torch.isfinite(branch).all():
            raise ValueError('Missing/nonfinite per-step memory projection')
        # The common engine constructs the default Head and strict-loads the
        # full state. The saved buffer restores gate1; no constructor override.
        gate_loading = 'memory_gate restored by unchanged common strict load_state_dict'
    else:
        if 'memory_gate' in ck['model'] or 'step_memory.weight' in ck['model']:
            raise ValueError('New-format checkpoint supplied to the original readout')
        gate_loading = 'original init-only head, no added gate'
    identity.update(scene='collision', supports=3, reference='learned', representation='P64',
        head_budget_epochs=100, memory_mode=config.get('memory_mode', 'init-only'), memory_gate=gate,
        gate_loading=gate_loading, readout=str(folder.resolve()),
        selected_checkpoint_sha256=sha(head/'selected.pt'), selected_epoch=ck['epoch'],
        source_codes_sha256=sha(folder/'codes_complete.json'), source_checkpoint_sha256=codes['source_sha256'],
        prepared=str(Path(args.prepared).resolve()), prepared_sha256=sha(Path(args.prepared)/'prepared.json'),
        wrong_donor_domain=prepared['wrong_donor_domain'],
        cache_alias_targets={name:str((folder/name).resolve()) for name in ('codes_complete.json','codes_train.npz','codes_val.npz')})
    return identity


def main(args):
    core, common, identity = load_cores(args)
    if args.command == 'interface':
        print(json.dumps(dict(status='PASS_INTERFACE_ONLY', **identity,
              checkpoint_evaluated=False, cuda_context_created=False)), flush=True)
        return 0
    if not args.prepared or not args.readout or not args.out:
        raise ValueError('evaluate requires --prepared, --readout and --out')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'evaluate.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not prerequisites(args):
            return 75
        identity = head_identity(args, core, identity)
        immutable(out/'evaluation_binding.json', identity)
        # The unchanged engine gates full4000 on original512 Matched/Null/Wrong
        # reproduction and loads selected.pt strictly, including memory_gate.
        common.evaluate(args)
        result = read(out/'results.json'); done = read(out/'complete.json')
        freeze = read(out/'checkpoint_freeze.json')
        if result.get('status') != 'COMPLETE' or done.get('status') != 'COMPLETE':
            raise ValueError('Common evaluation did not complete')
        if (freeze['head_implementation_sha256'] != identity['readout_core_sha256']
                or freeze['evaluation_implementation_sha256'] != identity['common_eval_sha256']
                or freeze['checkpoint_sha256'] != identity['selected_checkpoint_sha256']):
            raise ValueError('Evaluation used a different fixed head or implementation')
        if result['selection_rows'] != 512 or result['full_validation_rows'] != 4000:
            raise ValueError('Wrong validation cohorts')
        for arm in ('selected', 'matched', 'null', 'wrong'):
            if not result['reproduction'].get(arm, {}).get('matched_ids_exact'):
                raise ValueError('Missing exact-ID original512 reproduction: ' + arm)
        final = dict(status='COMPLETE', **identity,
            original512_reproduction=result['reproduction'], full_validation_rows=4000,
            results=str(out/'results.json'), results_sha256=sha(out/'results.json'),
            common_complete_sha256=sha(out/'complete.json'), checkpoint_freeze_sha256=sha(out/'checkpoint_freeze.json'),
            evaluation_binding_sha256=sha(out/'evaluation_binding.json'),
            scope='fixed checkpoint and support plans; no reselection, optimizer, or changed Wrong donor domain')
        immutable(out/'evaluation_complete.json', final)
        core.emit('collision_adaptation_fullval_complete', readout_version=core.VERSION,
                  memory_mode=identity['memory_mode'], matched_mse=result['matched']['mse'],
                  rows=4000, optimizer_steps=0)
        return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('interface','evaluate'), nargs='?', default='evaluate')
    p.add_argument('--readout-core', required=True, type=Path)
    p.add_argument('--common-eval', required=True, type=Path)
    p.add_argument('--readout-core-sha256'); p.add_argument('--common-eval-sha256')
    p.add_argument('--prepared'); p.add_argument('--readout'); p.add_argument('--out')
    p.add_argument('--scene', choices=('collision',), default='collision')
    p.add_argument('--supports', choices=(3,), type=int, default=3)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(); args.reference = 'learned'
    raise SystemExit(main(args))
