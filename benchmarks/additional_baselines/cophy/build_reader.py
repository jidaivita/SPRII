"""Build an isolated, hash-bound DALI role wrapper around the common reader.

The Head class, sampler, source normalization, optimizer and prediction code
remain byte-for-byte identical. Gate20 is labeled 20, never disguised as100.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

CORE_SHA = 'ca97ba4368cb2a86035e18d9866e8de967aa08906aee6e03ee79c36687c54c59'


def transform(source):
    assert hashlib.sha256(source.encode()).hexdigest() == CORE_SHA
    replacements = [
        ("roles={'Native':", "roles={'DALI-context':{'DALI-context'},'Native':", 1),
        ('choices=(50,100,150)', 'choices=(5,50,100,150)', 1),
        ('choices=(100,)', 'choices=(20,100,)', 1),
        ('head_epochs=100', 'head_epochs=args.epochs', 1),
        ("complete.get('epochs')!=100", "complete.get('epochs')!=args.epochs", 1),
        ('head_budget=100', 'head_budget=args.epochs', 3),
        ("representation='complete legal state; query-conditioned; never AB-only P slice'",
         "representation='DALI scene context8 zero padded128; query3 remains in the common reader query path'", 1),
        ("current_input='shared pose/detection/public type plus current RGB-derived T64 or joint U; no future features'",
         "current_input='shared three-frame perceived pose/detection/public type; context branch uses AB history only'", 1),
        ("head_history_access='complete native or split state projected then mean; init only; dropout preserves current-only state'",
         "head_history_access='DALI global AB context8 padded128; fixed support projection then S3 mean; init only'", 1),
        ("null_semantics='zero all native history state then query; split zeroP plus current state; common train-only normalization'",
         "null_semantics='zero DALI context before common train-only normalization; perceived query3 unchanged'", 1),
        ("representation='complete OWN AB+query3 state; includes current information; not P-only formation'",
         "representation='DALI global OWN AB context8 padded128; current query does not enter this context probe'", 1),
        ("attribution=('frozen source with query context' if channel=='x' else",
         "attribution=('frozen DALI context component; no query information in code' if channel=='x' else", 1),
    ]
    changed = source
    for old, new, count in replacements:
        assert changed.count(old) == count, (old, changed.count(old), count)
        changed = changed.replace(old, new)
    # Computation in the downstream learner is unchanged. A full lossless AST
    # equality check protects against accidental head or pairing edits.
    before, after = ast.parse(source), ast.parse(changed)
    for name in ('Head', 'PairPlan', 'Data', 'scores', 'fit_normalization', 'score_head'):
        left = next((x for x in before.body if getattr(x, 'name', '') == name), None)
        right = next((x for x in after.body if getattr(x, 'name', '') == name), None)
        if left is not None:
            assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), name
    return changed


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--core', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); core = Path(a.core); dest = Path(a.out)
    text = transform(core.read_text()); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert dest.read_text() == text, 'Existing derived reader differs'
    else:
        dest.write_text(text)
    meta = {'status': 'COMPLETE', 'core': str(core), 'core_sha256': CORE_SHA,
            'derived_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'head_and_pairing_unchanged': True, 'source_role': 'DALI-context'}
    dest.with_suffix('.binding.json').write_text(json.dumps(meta, indent=2)+'\n')
