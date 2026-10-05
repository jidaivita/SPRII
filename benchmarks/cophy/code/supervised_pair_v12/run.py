"""Frozen supervised representations: direct pair-memory access in the new head."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def run(path, device, smoke_only=False):
    import torch
    from torch import nn
    spec = json.loads(Path(path).read_text())
    loader = Path(spec['controlled'])
    if hashlib.sha256(loader.read_bytes()).hexdigest() != spec['controlled_sha256']:
        raise ValueError('Changed controlled reader')
    core = Path(spec['core'])
    if hashlib.sha256(core.read_bytes()).hexdigest() != spec['core_sha256']:
        raise ValueError('Changed source reader')
    imp = importlib.util.spec_from_file_location('pair_controlled', loader)
    ctl = importlib.util.module_from_spec(imp); imp.loader.exec_module(ctl)
    recipe = dict(initialization='v7', dropout=.1, memory='every_step', pair_memory=True)
    m = ctl.load_controlled(core, recipe)
    parent_class = m.Head

    class PairHead(parent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            old = self.message[0]
            new = nn.Linear(old.in_features+130, old.out_features)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            self.message[0] = new

        def forward(self, q, det, mask, support, missing=None):
            b, length, slots, dims = q.shape
            x = (q-self.xy_mean)/self.xy_scale
            dx = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:]-x[:, :-1]), 1)
            if det.ndim == 3: det = det[..., None]
            token = torch.cat((x, dx, det), -1).permute(0, 2, 1, 3).reshape(b*slots, length, -1)
            _, qh = self.query(token); qh = qh[0].reshape(b, slots, 64)
            if support is None:
                memory = self.missing.expand(b, slots, 64)
                available = torch.zeros(b, slots, 1, device=q.device)
            else:
                memory = self.support(support).mean(2)
                available = torch.ones(b, slots, 1, device=q.device)
                if missing is not None:
                    flag = missing[:, None, None]
                    memory = torch.where(flag, self.missing[None, None, :], memory)
                    available = available*(~flag)
            h = self.init(torch.cat((qh, memory, available), -1))
            position = x[:, -1]; velocity = (x[:, -1]-x[:, 0])/(length-1)
            pair = mask[:, :, None]*mask[:, None, :]*(1-torch.eye(slots, device=q.device)[None])
            denominator = pair.sum(2).clamp_min(1)[..., None]
            mi = memory[:, :, None].expand(b, slots, slots, 64)
            mj = memory[:, None, :].expand(b, slots, slots, 64)
            ai = available[:, :, None].expand(b, slots, slots, 1)
            aj = available[:, None, :].expand(b, slots, slots, 1)
            result = []
            for _ in range(self.horizon):
                hi = h[:, :, None].expand(b, slots, slots, self.width)
                hj = h[:, None, :].expand(b, slots, slots, self.width)
                rel = position[:, None]-position[:, :, None]
                rv = velocity[:, None]-velocity[:, :, None]
                msg = self.message(torch.cat((hi, hj, rel, rv, mi, mj, ai, aj), -1))
                agg = (msg*pair[..., None]).sum(2)/denominator
                step = torch.cat((agg, position, velocity, memory, available), -1)
                h = self.cell(step.reshape(b*slots, -1), h.reshape(b*slots, -1)).reshape(b, slots, -1)
                velocity = velocity+self.delta(h); position = position+velocity
                result.append(position*self.xy_scale+self.xy_mean)
            return torch.stack(result, 1)

    out = Path(spec['out']); out.mkdir(parents=True, exist_ok=True)
    parent = Path(spec['parent_readout'])
    for name in ['codes_train.npz', 'codes_val.npz', 'codes_complete.json', 'probes.json']:
        src, dst = parent/name, out/name
        if not src.exists(): raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            if dst.resolve() != src.resolve(): raise ValueError('Different source cache')
        else: dst.symlink_to(src)
    torch.set_num_threads(2)
    args = argparse.Namespace(out=str(out), base=spec['base'], scene=spec['scene'],
        reference='learned', root=spec['root'], device=device, supports=3, epochs=100)
    m.CONTROL_SHA = hashlib.sha256((m.CONTROL_SHA+Path(__file__).read_text()).encode()).hexdigest()
    data = m.Data(args)
    import numpy as np
    q, det, mask, support, y = data.batch('train', np.arange(4), 1, device)
    original = parent_class(data.dims, data.det_dims, data.support_dims, data.horizon).to(device)
    adapted = PairHead(data.dims, data.det_dims, data.support_dims, data.horizon).to(device)
    with torch.no_grad():
        torch.testing.assert_close(original(q, det, mask, support), adapted(q, det, mask, support), rtol=1e-5, atol=1e-6)
    loss = m.scores(adapted(q, det, mask, support), y, mask).mean(); loss.backward()
    gradient = adapted.message[0].weight.grad[:, -130:]
    if not torch.isfinite(gradient).all() or gradient.norm().item() == 0:
        raise ValueError('No useful pair-memory gradient')
    m.write(out/'pair_smoke.json', dict(status='PASS', initial_equivalence=True,
        added_gradient_norm=gradient.norm().item(), added_parameters=130*128, optimizer_steps=0))
    del original, adapted, data
    if smoke_only:
        return
    m.Head = PairHead
    m.train(args)
    m.write(out/'pair_complete.json', dict(status='COMPLETE', source_frozen=True,
        source_probe_sha256=m.sha(parent/'probes.json'), recipe=recipe, test_read=False,
        source_optimizer_steps=0, spec=spec, implementation_sha256=m.CONTROL_SHA))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--spec', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke-only', action='store_true')
    a=p.parse_args(); run(a.spec, a.device, a.smoke_only)
