"""Apply the original frozen readout to locally generated feature/signature arrays."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from frozen_probe import linear_identity_probe, functional_invariant_probe, formation_summary

def evaluate(arrays):
    features, labels, contexts = [arrays[k] for k in ('features','labels','contexts')]
    fit, test, roles, targets = [arrays[k] for k in ('fit','test','roles','targets')]
    assert features.shape == (176, 32)
    identity, _, standardized = linear_identity_probe(features, labels, fit, test)
    heldout, head = functional_invariant_probe(features, labels, targets, fit, test, roles)
    results = {}
    for role in ('train', 'heldout'):
        mask = test & (roles == role)
        per = []
        for label in np.unique(labels[mask]):
            selected = mask & (labels == label)
            per.append({'identity_label':int(label),
                        'mse':float(np.mean((head['predictions'][selected] - targets[label])**2)),
                        'mean_baseline_mse':float(np.mean((head['intercept'] - targets[label])**2))})
        mean = float(np.mean([r['mse'] for r in per]))
        base = float(np.mean([r['mean_baseline_mse'] for r in per]))
        results[role] = {'macro_mse':mean,'mean_baseline_macro_mse':base,
                        'relative_error_reduction_vs_mean':1 - mean/base if base > 1e-12 else None,
                        'per_partner':per,
                        'formation':formation_summary(features,standardized,labels,contexts,mask)}
    return {'identity':identity, 'primary':results['train'],
            'heldout_development':results['heldout'],'original_heldout_protocol':heldout,
            'ridge':0.001,'frozen_encoder_updates':0,'parameter_recovery':False}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arrays', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with np.load(a.arrays, allow_pickle=False) as arrays:
        result = evaluate(arrays)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
