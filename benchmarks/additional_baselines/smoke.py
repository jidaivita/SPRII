"""CPU checks for the released component adapters; no dataset is read."""
from pathlib import Path
import importlib.util
import sys
import torch

ROOT = Path(__file__).resolve().parent

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    sys.path.insert(0, str(ROOT / 'dclean'))
    api = load(ROOT / 'cophy/dclean_budget_sensitivity.py', '_budget_smoke')
    assert api.smoke()['cpu_adam_resume_bitwise_equal']
    source = load(ROOT / 'dclean/dali_dclean_source.py', '_dclean_source')
    norm = dict(state_mean=[0.] * 4, state_scale=[1.] * 4, action_scale=[1.] * 2)
    model = source.Model(norm)
    x, u = torch.randn(6, 24, 4), torch.randn(6, 23, 2)
    loss = model.objective(x, u, torch.tensor([1, 3, 7, 12, 19, 23]))
    loss.backward()
    assert torch.isfinite(loss) and model.encode(x, u).shape == (6, 8)
    helper = load(ROOT / 'shared/dclean_external.py', '_fcrl_smoke')
    helper.stats = lambda: norm
    fcrl = helper.Encoder('FCRL')
    loss = fcrl.objective(x, u, torch.randn(6, 32, 2), torch.randn(6, 3, 4))
    loss.backward()
    assert torch.isfinite(loss) and fcrl.encode(x, u).shape == (6, 50)
    api = load(ROOT / 'cophy/models.py', '_cophy_smoke')
    model = api.CoPhyDALI()
    features = torch.randn(2, 15, 4, 784)
    mask = torch.ones(2, 15, 4)
    loss = model.source_loss(features, mask, torch.tensor([3, 13]))
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    context = model.encode_joint(features, mask, features[:, :3], mask[:, :3])
    assert context.shape == (2, 4, 128) and torch.count_nonzero(context[..., 8:]) == 0
    print('PASS: exact Adam/RNG continuation, selection ordering, DALI/FCRL finite gradients, CoPhy interface and zero padding')

if __name__ == '__main__':
    main()
