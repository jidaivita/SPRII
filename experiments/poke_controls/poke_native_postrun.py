"""Frozen-checkpoint native latent forecasting on the common geometry population."""
import argparse, hashlib, json, pathlib, sys, time
ROOT = pathlib.Path(os.environ.get('SPRII_ROOT', '.'))
SRC = ROOT/'benchmarks/pokeworld/revision'
DATA = ROOT/'data/pokeworld_factorized'
sys.path.insert(0, str(SRC/'src'))
import numpy as np
import torch
from persistent_jepa.poke_model import PokeJEPA, poke_objective
from persistent_jepa.poke_torch import PokeSplit
from persistent_jepa.runtime import set_deterministic
from persistent_jepa.torch_data import HORIZONS

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p, d):
    q = p.with_suffix('.tmp'); q.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n'); q.replace(p)
def digest(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True); p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    stage, out = pathlib.Path(a.stage), pathlib.Path(a.output)
    out.mkdir(exist_ok=False)
    t = time.time(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    set_deterministic(20260903)
    complete = json.loads((stage/'COMPLETE.json').read_text())
    cp = stage/'checkpoints/step_020000.pt'; config = json.loads((stage/'config.json').read_text())
    assert complete['status'] == 'SOURCE_AND_GEOMETRY_COMPLETE'
    assert sha(cp) == complete['checkpoint_sha256']
    assert sha(stage/'GEOMETRY.json') == complete['geometry_sha256']
    assert sha(DATA/'manifest.json') == config['dataset_manifest_sha256']
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    assert payload['step'] == 20000 and payload['config'] == config
    geometry = json.loads((stage/'GEOMETRY.json').read_text())
    model = PokeJEPA(config['model_variant'], int(config['history_length'])).cuda().eval()
    model.load_state_dict(payload['model']); before = digest(model)
    train, val = PokeSplit(DATA, 'train'), PokeSplit(DATA, 'val')
    rng = np.random.default_rng(20260903)
    train_index = np.sort(rng.permutation(train.states.shape[0])[:200])
    val_index = np.sort(rng.permutation(val.states.shape[0])[:200])
    assert len(train_index) == len(val_index) == 200
    if a.smoke: val_index=val_index[:8]
    s = np.repeat(val_index, 16)
    r = np.tile(np.repeat(np.arange(4), 4), len(val_index))
    anchors = np.tile(np.array([24,32,40,47]), len(val_index)*4)
    manifest = dict(split='val', systems=val_index.tolist(), rollouts=[0,1,2,3], anchors=[24,32,40,47],
                    horizons=list(HORIZONS), windows=len(s), selection_seed=20260903,smoke=a.smoke,
                    train_permutation_consumed_before_validation=True,
                    bank_manifest_sha256=sha(DATA/'manifest.json'), val_data_sha256=sha(DATA/'val.npz'),
                    schema='poke-common-native-development-v1', test_read=False)
    write(out/'MANIFEST.json', manifest)
    assert geometry['anchors'] == manifest['anchors'] and geometry['rollouts'] == manifest['rollouts']
    # BatchNorm stays in eval mode: recipient predictions cannot see target frames.
    errors=[]; targets=[]; parity={}
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
    for start in range(0, len(s), a.batch_size):
        stop = min(start+a.batch_size, len(s))
        b = val._from_indices(s[start:stop],r[start:stop],anchors[start:stop]).to(torch.device('cuda'))
        hh, yy = model.encode_batch(b); _,_,context = model.codes(hh,b.history_actions)
        predictions=[]
        for hi,horizon in enumerate(HORIZONS):
            hi_tensor = torch.full((len(context),),hi,device='cuda',dtype=torch.long)
            predictions.append(model.predictor(context,b.future_actions[:,hi],b.action_masks[:,hi],hi_tensor))
        prediction=torch.stack(predictions,1)
        mse=(prediction-yy).square().mean(-1)
        assert torch.isfinite(prediction).all() and torch.isfinite(mse).all()
        if start == 0:
            # Direct check against the original training objective's three native losses.
            _, metrics=poke_objective(model,b,lambda x:x.new_zeros(()),sigreg_weight=0,lambda_p=0,lambda_x=0)
            for hi,horizon in enumerate(HORIZONS):
                torch.testing.assert_close(mse[:,hi].mean(),metrics[f'loss_h{horizon}'],rtol=2e-6,atol=1e-8)
            # Future observations may supply targets only, not the consuming history/code.
            original=b.target_current.clone(); previous=b.target_previous.clone()
            b.target_current.zero_(); b.target_previous.zero_()
            hh_masked,_=model.encode_batch(b); _,_,context_masked=model.codes(hh_masked,b.history_actions)
            assert torch.equal(hh,hh_masked) and torch.equal(context,context_masked)
            b.target_current.copy_(original); b.target_previous.copy_(previous)
            parity=dict(original_native_objective_losses_equal=True,target_frame_mask_preserves_context=True)
        errors.append(mse.double().cpu().numpy()); targets.append(yy.double().cpu().numpy())
    err=np.concatenate(errors); target=np.concatenate(targets)
    assert err.shape==(len(s),3) and digest(model)==before
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert sha(cp)==complete['checkpoint_sha256']
    np.savez_compressed(out/'WINDOWS.npz',system=s,rollout=r,anchor=anchors,native_mse=err,
                        target_second_moment=np.square(target).mean(2))
    rows={str(h):dict(mean_native_latent_mse=float(err[:,i].mean()),
                     target_variance=float(target[:,i].var(0).mean()),
                     target_second_moment=float(np.square(target[:,i]).mean())) for i,h in enumerate(HORIZONS)}
    write(out/'SUMMARY.json',dict(condition=config['relation_mode'],seed=config['seed'],rows=rows,
          interpretation='Native self-prediction in each frozen online encoder space; not physical-state MSE or control return. Target scales are separately reported.',
          checkpoint_sha256=sha(cp),manifest_sha256=sha(out/'MANIFEST.json'),model_and_buffers_unchanged=True,
          model_state_sha256=before,rng_unchanged=True,optimization_updates=0,new_environment_interactions=0,
          test_read=False,seconds=time.time()-t,parity=parity))
    write(out/'COMPLETE.json',dict(status='COMPLETE',files={f:sha(out/f) for f in ['SUMMARY.json','MANIFEST.json','WINDOWS.npz']},
          evaluator_sha256=sha(__file__),checkpoint_sha256=sha(cp),windows=len(s),horizons=list(HORIZONS),smoke=a.smoke))
    print(json.dumps(dict(status='COMPLETE',condition=config['relation_mode'],seed=config['seed'],seconds=time.time()-t)),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        if '--output' in sys.argv:
            out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
            if out.is_dir() and not (out/'COMPLETE.json').exists(): write(out/'FAILED.json',dict(error=repr(e)))
        raise
