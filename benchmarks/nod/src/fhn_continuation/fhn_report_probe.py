"""Filter report recipients and refit the existing train-only probes."""
import sys, json, copy, argparse, hashlib
from pathlib import Path
import numpy as np
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.writing');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)
from fhn_minimal.data import SYSTEMS, CONDITION_FRAMES, load_trajectory
from fhn_minimal.evaluate import load_model, ridge_probe
from fhn_minimal.train import model_checksum

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint',type=Path);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
parser.add_argument('--data-dir',required=True,type=Path);parser.add_argument('--official-code',required=True,type=Path)
parser.add_argument('--device',default='cuda');options=parser.parse_args()
sys.path.insert(0,str(options.official_code))
from ngs.neuralnetworks import NGS_metaNet_Hier

checkpoint, source, output = options.checkpoint, options.source, options.output
d = json.loads(source.read_text())
assert d['checkpoint_sha256'] == sha(checkpoint)
assert d['K1_equivalence_pass'] and d['model_unchanged']
targets = {15, 24, 45}
d['cases'] = [r for r in d['cases'] if r['initial_id'] in targets]
d['latents'] = [r for r in d['latents'] if r['initial_id'] in targets]
assert {r['initial_id'] for r in d['cases']} == targets
torch.set_num_threads(4)
device = torch.device(options.device)
model = NGS_metaNet_Hier(64, 3, 2, 128).to(device)
load_model(model, checkpoint, device)
model.eval()
before = model_checksum(model)
zs, ys = [], []
data = options.data_dir
with torch.inference_mode():
    for system in SYSTEMS['train']:
        histories = torch.stack([load_trajectory(data, system, i)[list(CONDITION_FRAMES)] for i in range(50, 90)])
        z = torch.cat([model.conditioning_encoder(chunk.to(device)).cpu() for chunk in histories.split(8)]).numpy()
        zs.append(z)
        ys.extend([system] * 40)
zs, ys = np.stack(zs), np.asarray(ys)
probes, aggregates = {}, []
for K in [1, 2, 4]:
    train_z = np.mean([np.roll(zs, -j, axis=1) for j in range(K)], axis=0).reshape(-1, 2)
    probes[str(K)] = {}
    for split in SYSTEMS:
        rows = [r for r in d['latents'] if r['K'] == K and r['split'] == split]
        probes[str(K)][split] = ridge_probe(train_z, ys, np.asarray([r['z'] for r in rows]), np.asarray([[r['k'], r['beta']] for r in rows]))
        for h in [1, 5, 50]:
            cases = [r for r in d['cases'] if r['K'] == K and r['split'] == split and r['horizon'] == h]
            aggregates.append(dict(split=split, K=K, horizon=h, n=len(cases), **{m: float(np.mean([r[m] for r in cases])) for m in ['mse', 'l2_relative']}))
assert before == model_checksum(model)
d.update(probe=probes, aggregate=aggregates, source_json_sha256=sha(source), report_recipient_ids=sorted(targets))
d['protocol'] = dict(d['protocol'], targets=sorted(targets), development_recipient_excluded=True)
write(output, d)
np.savez_compressed(output.with_suffix('.probe_training.npz'), z=zs, y=ys)
print('FHN_REPORT_PROBE_COMPLETE', flush=True)
