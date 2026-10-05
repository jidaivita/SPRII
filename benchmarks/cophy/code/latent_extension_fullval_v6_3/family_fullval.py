"""Family-bound fixed-head evaluation using the unchanged common fullval engine.

Only evaluation is available. A CPU interface check imports the real family core,
but does not claim checkpoint acceptance. Real old512 reproduction remains inside
common.evaluate and must pass before any expanded-cohort scores are computed.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

VERSION = 'cophy-cpc-rssm-family-fullval-v6.3-1'
CORE_VERSIONS = {
    'CPC': 'cophy-cpc-v6.3-frozen-P64-pose-prefix-readout',
    'RSSM': 'cophy-rssm-v6.3-frozen-P64-pose-prefix-readout',
}
COMMON_VERSION = 'latent-v6.2-fullval-fixed-P64-readout-1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def immutable(path, data):
    if Path(path).exists() and read(path) != data:
        raise ValueError('Different family evaluation binding: ' + str(path))
    write(path, data)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_cores(args):
    if sha(args.family_core) != args.family_core_sha256:
        raise ValueError('Family readout differs from its bound training implementation')
    if sha(args.common_eval) != args.common_eval_sha256:
        raise ValueError('Common fullval engine changed')
    core = load_module('_fixed_family_' + args.family.lower(), args.family_core)
    if core.VERSION != CORE_VERSIONS[args.family]:
        raise ValueError('Family/version mismatch')
    # Explicit dependency injection in this isolated evaluator process only.
    # The original family and JEPA source files are never edited.
    sys.modules['readout'] = core
    common = load_module('_fixed_common_fullval', args.common_eval)
    if common.core is not core or common.VERSION != COMMON_VERSION:
        raise ValueError('Common evaluator did not load the requested family core')
    for name in ('Data', 'Head', 'evaluate', 'immutable', 'sha'):
        if not hasattr(core, name):
            raise ValueError('Missing family readout interface: ' + name)
    if not hasattr(common, 'FullData') or not callable(common.evaluate):
        raise ValueError('Missing common full-validation interface')
    return core, common


def main(args):
    core, common = load_cores(args)
    identity = dict(
        adapter_version=VERSION, family=args.family, family_core_version=core.VERSION,
        family_core=str(Path(args.family_core).resolve()),
        family_core_sha256=sha(args.family_core),
        common_eval=str(Path(args.common_eval).resolve()),
        common_eval_sha256=sha(args.common_eval), common_eval_version=common.VERSION,
        wrapper=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        test_read=False, optimizer_steps=0,
    )
    if args.command == 'interface':
        print(json.dumps(dict(status='PASS_INTERFACE_ONLY', **identity,
                              checkpoint_evaluated=False, cuda_context_created=False)),
              flush=True)
        return
    if not args.scene or not args.prepared or not args.readout or not args.out:
        raise ValueError('evaluate requires scene, prepared, readout, and out')
    if args.supports == 8 and args.scene != 'collision':
        raise ValueError('S8 is planned only for Collision')
    if Path(args.readout).name != 'source50':
        raise ValueError('This tail accepts only the fixed source50 readout directory')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'evaluate.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity.update(scene=args.scene, supports=args.supports,
                        representation='P64', reference='learned',
                        source_checkpoint_epoch=50, head_budget_epochs=100,
                        readout=str(Path(args.readout).resolve()),
                        prepared=str(Path(args.prepared).resolve()))
        codes = core.read(Path(args.readout) / 'codes_complete.json')
        if codes.get('version') != core.VERSION or codes.get('representation') != 'P64':
            raise ValueError('Wrong family P64 cache')
        identity['source_codes_sha256'] = core.sha(Path(args.readout) / 'codes_complete.json')
        prepared = core.read(Path(args.prepared) / 'prepared.json')
        if prepared.get('test_read') is not False:
            raise ValueError('Only committed validation preparation is allowed')
        identity['prepared_sha256'] = core.sha(Path(args.prepared) / 'prepared.json')
        identity['wrong_donor_domain'] = prepared['wrong_donor_domain']
        immutable(out / 'family_binding.json', identity)
        # This enforces head100, exact original config, unchanged selected.pt,
        # all bound P64 files, and old512 per-recipient Matched/Null/Wrong.
        common.evaluate(args)
        result = core.read(out / 'results.json')
        common_done = core.read(out / 'complete.json')
        freeze = core.read(out / 'checkpoint_freeze.json')
        if result.get('status') != 'COMPLETE' or common_done.get('status') != 'COMPLETE':
            raise ValueError('Common evaluator did not complete')
        if freeze['head_implementation_sha256'] != identity['family_core_sha256']:
            raise ValueError('Common result used a different family predictor')
        if result['selection_rows'] != 512:
            raise ValueError('Missing old512 reproduction cohort')
        for arm in ('selected', 'matched', 'null', 'wrong'):
            if arm not in result['reproduction'] or not result['reproduction'][arm]['matched_ids_exact']:
                raise ValueError('Missing exact-ID original reproduction: ' + arm)
        final = dict(
            status='COMPLETE', **identity, selected_epoch=result['selected_epoch'],
            full_validation_rows=result['full_validation_rows'],
            original512_reproduction=result['reproduction'],
            results=str(out / 'results.json'), results_sha256=core.sha(out / 'results.json'),
            common_complete_sha256=core.sha(out / 'complete.json'),
            checkpoint_freeze_sha256=core.sha(out / 'checkpoint_freeze.json'),
            family_binding_sha256=core.sha(out / 'family_binding.json'),
            scope='fixed trained head; no reselection, training, or changed Wrong domain',
        )
        immutable(out / 'family_complete.json', final)
        core.emit('family_fullval_complete', family=args.family, scene=args.scene,
                  supports=args.supports, rows=result['full_validation_rows'],
                  matched_mse=result['matched']['mse'], optimizer_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('interface', 'evaluate'))
    parser.add_argument('--family', choices=tuple(CORE_VERSIONS), required=True)
    parser.add_argument('--family-core', type=Path, required=True)
    parser.add_argument('--family-core-sha256', required=True)
    parser.add_argument('--common-eval', type=Path, required=True)
    parser.add_argument('--common-eval-sha256', required=True)
    parser.add_argument('--scene', choices=('balls', 'collision', 'blocktower'))
    parser.add_argument('--prepared')
    parser.add_argument('--readout')
    parser.add_argument('--supports', type=int, choices=(3, 8), default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out')
    args = parser.parse_args()
    args.reference = 'learned'
    main(args)
