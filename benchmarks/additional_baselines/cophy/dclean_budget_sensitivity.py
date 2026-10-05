"""One fixed DALI D-Clean budget sensitivity cell, selection100 only.

Source seed0 20000->40000 (or explicitly recorded scratch40000 if resume state
is incomplete), fresh reader0 20000. No hyperparameter/checkpoint selection,
no report-system evaluation, no automatic promotion or multi-seed launch.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'dclean'
SOURCE_SHA = '9338037016de08e56d5bdaa7bc0f6f734483e07d6c514f090896992e93441b5c'
COMPONENT_SHA = '069e15f406a4344ae0e61bf4fdbb0fb92dfccbe4514225a0faea452772752f32'
HELPER_SHA = '59c56a4caac12bcefcf3b6bca2a4353f913bd2b1f4aa6bc645dcb0c1f823821f'
PANEL_SHA = 'b64ac20c7a847703b783f1db5ad626499611122dccc3ee1dea0ff5b21b062ecc'
ANCESTOR_SHA = '77174fe1cebd8c641ad0876dcef428669f90275937926acafb42756f27c6fe15'
H = (1, 4, 16, 32)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.pending')
    q.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); q.replace(p)


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def restore(model, opt, checkpoint, prefix_rng):
    import torch
    model.load_state_dict(checkpoint['model'], strict=True)
    opt.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    prefix_rng.bit_generator.state = copy.deepcopy(checkpoint['numpy_prefix_rng'])


def inspect_rng_path(source, component):
    """Exact audited source has no dropout/no stochastic Torch training op."""
    import ast
    tree = ast.parse(source.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For) and ast.unparse(n.target) == 'step')
    calls = sorted({ast.unparse(n.func) for n in ast.walk(loop) if isinstance(n, ast.Call)})
    assert not any('rand' in s or 'dropout' in s for s in calls)
    assert 'prefix_rng.integers' in calls and 'helper.specs' in calls
    assert not any(isinstance(n, ast.Call) and ('dropout' in ast.unparse(n.func).lower() or
                   ast.unparse(n.func).startswith(('torch.rand', 'torch.bernoulli')))
                   for n in ast.walk(ast.parse(component.read_text())))
    return calls


def run(a):
    import numpy as np
    import torch
    out = Path(a.output).resolve()
    assert not out.exists(), 'Fresh independent output required; no implicit restart.'
    out.mkdir(parents=True)
    def expired(*_): raise TimeoutError('Fixed1200-second total budget exhausted')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, 1200)
    began = time.monotonic()
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        sys.path.insert(0, str(CODE))
        paths = [(CODE/'dali_dclean_source.py', SOURCE_SHA), (CODE/'dali_context_torch.py', COMPONENT_SHA),
                 (Path(a.helper), HELPER_SHA), (Path(a.panel), PANEL_SHA)]
        for path, digest in paths: assert sha(path) == digest, path
        helper = load(a.helper, 'dclean_external'); panel = load(a.panel, '_budget_original_panel')
        api = load(CODE/'dali_dclean_source.py', '_budget_source_api')
        original = Path(a.formal_root); src = original/'sources/seed0'
        old_checkpoint = src/'final.pt'; assert sha(old_checkpoint) == ANCESTOR_SHA
        done, config, summary = read(src/'COMPLETE.json'), read(src/'CONFIG.json'), read(src/'SUMMARY.json')
        assert done['status'] == 'COMPLETE' and read(src/'EXIT.json')['exit_code'] == 0
        assert done['summary_sha256'] == sha(src/'SUMMARY.json')
        assert config['source_seed'] == 0 and config['steps'] == summary['steps'] == 20000
        assert sha(src/'CONFIG.json') == summary['config_sha256']
        assert config['code_sha256'] == SOURCE_SHA and config['component_code_sha256'] == COMPONENT_SHA
        ck = torch.load(old_checkpoint, map_location='cpu', weights_only=False)
        assert ck['step'] == 20000 and ck['source_seed'] == 0 and ck['config_sha256'] == summary['config_sha256']
        required = ('model', 'optimizer', 'torch_rng', 'numpy_prefix_rng')
        missing = [k for k in required if k not in ck]
        resume = not missing
        protocol_path = helper.ROOT/'PROTOCOL.json'; protocol_sha = sha(protocol_path)
        protocol = helper.contract(); assert sha(protocol_path) == protocol_sha
        calls = inspect_rng_path(CODE/'dali_dclean_source.py', CODE/'dali_context_torch.py')
        plan = dict(source_seed=0, reader_seed=0, source_total_updates=40000, reader_updates=20000,
            source_mode='resume20000_to40000' if resume else 'scratch40000_missing_resume_state',
            missing_resume_fields=missing, source_ancestor=str(old_checkpoint), ancestor_sha256=ANCESTOR_SHA,
            source_new_updates=20000 if resume else 40000, optimizer='unchanged Adam lr1e-4 eps1e-8',
            source_selection='fixed40000 endpoint', reader_selection='fixed20000 endpoint',
            selected_systems=protocol['selection_ids'], excluded_report_systems=protocol['report_ids'],
            evaluation='selection100 only; exact same generated pairs for20k and40k',
            observation_files='Existing val.npz holds200 systems; only selection100 rows passed to encoding/evaluation',
            historical_report_already_seen=True, new_sealed_test=False, report_evaluation=False,
            scope='Training-budget sensitivity after historical report was already observed',
            total_timeout_seconds=1200, automatic_followup=False,
            review_trigger='H32 selection MSE at least10% lower and at least3/4 horizons lower; never automatic promotion',
            rng_restore=dict(torch_cpu=resume, numpy_prefix_generator=resume,
                cuda_snapshot='absent in ancestor; no CUDA-random training operations in audited path',
                global_python_numpy='not stored; unused by step-keyed data and independent prefix generator',
                deterministic_step_specs=True, audited_training_calls=calls),
            implementation={str(p): h for p,h in paths}, runner_sha256=sha(__file__))
        write(out/'PLAN.json', plan); write(out/'RUN.json', dict(pid=os.getpid(), start=time.time()))
        helper.seed_all(0); device = torch.device(a.device)
        assert str(device) == 'cuda:0', 'The unchanged common reader uses the root-owned masked CUDA0.'
        model = api.build_model(config, a.official_root).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
        prefix_rng = np.random.default_rng(202609250000)
        if resume: restore(model, opt, ck, prefix_rng)
        start = 20001 if resume else 1
        for group in opt.param_groups:
            assert group['lr'] == 1e-4 and group['eps'] == 1e-8 and group['weight_decay'] == 0
        if resume:
            assert {int(v['step']) for v in opt.state.values()} == {20000}
            assert all(torch.equal(v.cpu(), ck['model'][k]) for k,v in model.state_dict().items())
        new_source = out/'source40k'; new_source.mkdir()
        write(new_source/'CONFIG.json', dict(original=config, source_total_updates=40000, plan_sha256=sha(out/'PLAN.json')))
        model.train()
        with (new_source/'train.jsonl').open('x') as log:
            for step in range(start, 40001):
                batch = api.make_batch(helper.data('train'), helper.specs(0, step), device)
                lengths = torch.as_tensor(prefix_rng.integers(1,24,96), device=device)
                opt.zero_grad(set_to_none=True); loss = model.objective(*batch, lengths)
                assert torch.isfinite(loss); loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000, error_if_nonfinite=True); opt.step()
                if step == start or step % 100 == 0:
                    row = dict(step=step, loss=float(loss), gradient_norm=float(grad), seconds=time.monotonic()-began)
                    log.write(json.dumps(row)+'\n'); log.flush(); write(new_source/'PROGRESS.json', row)
        final = new_source/'final.pt'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=40000, source_seed=0,
                        torch_rng=torch.get_rng_state(), numpy_prefix_rng=prefix_rng.bit_generator.state,
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                        source_ancestor_sha256=ANCESTOR_SHA, plan_sha256=sha(out/'PLAN.json')), final)
        dev = api.dev_metrics(model, helper, device)
        write(new_source/'COMPLETE.json', dict(status='COMPLETE', step=40000, checkpoint_sha256=sha(final),
            initial20k_dev=summary['final_dev'], final40k_dev=dev, report_evaluation=False))
        model.eval().requires_grad_(False); source_digest = panel.digest(model)

        # Encode only training windows. Do not call original cache(), whose val
        # branch encodes all200 systems including the excluded report cohort.
        train = helper.data('train'); cache, norm = encode_windows(model, train, device)
        common = out/'common40k'; common.mkdir(); cache_dir=common/'cache/DALI_s0'; cache_dir.mkdir(parents=True)
        np.save(cache_dir/'train.npy', cache)
        info = dict(status='COMPLETE', normalization=norm, source=dict(checkpoint=str(final),sha256=sha(final)),
                    files={'train':sha(cache_dir/'train.npy')}, split='train only', source_frozen=True)
        write(cache_dir/'COMPLETE.json', info)
        panel.ROOT = common; panel.contract = lambda: plan
        write(common/'PROTOCOL.json', plan)
        original_load = panel.load_cache
        def train_only(method, seed, split):
            assert method == 'DALI' and seed == 0 and split == 'train'
            return original_load(method, seed, split)
        panel.load_cache = train_only
        panel.fit('DALI',0,0,'matched')
        new_reader = common/'readers/DALI_s0/r0'
        old_reader = original/'common/seed0/readers/DALI_s0/r0'
        assert read(new_reader/'COMPLETE.json')['step'] == read(old_reader/'COMPLETE.json')['step'] == 20000
        assert read(new_reader/'RUN.json')['initial_sha256'] == read(old_reader/'RUN.json')['initial_sha256']
        assert panel.digest(model) == source_digest
        val = helper.data('val'); selection = set(protocol['selection_ids']); excluded=set(protocol['report_ids'])
        indices = np.array([i for i,s in enumerate(val['system_ids']) if int(s) in selection])
        assert len(indices)==100 and selection.isdisjoint(excluded)
        subset={k: v[indices] for k,v in val.items()}; del val
        assert set(map(int,subset['system_ids'])) == selection
        sp = selection_specs(subset); keys = np.concatenate([sp,sp[:,[0,3,4,1,2]]])
        actual_ids = subset['system_ids'][keys[:,0]]
        assert len(keys)==1600 and not set(map(int,actual_ids)) & excluded
        write(out/'SELECTION_MANIFEST.json', dict(system_ids=list(map(int,actual_ids)), keys=keys.tolist(),
            report_ids_absent=True, seed=0, horizon=list(H), actual_pairs=1600))
        rows={}; arrays={}; probes={}
        for name, source, reader in (('source20k', old_checkpoint, old_reader),('source40k', final, new_reader)):
            if name=='source20k':
                source_model=api.build_model(config,a.official_root).to(device)
                source_model.load_state_dict(ck['model'],strict=True);source_model.eval().requires_grad_(False)
                old_cache=original/'common/seed0/cache/DALI_s0'
                old_info=read(old_cache/'COMPLETE.json')
                assert old_info['source']['sha256']==ANCESTOR_SHA and sha(old_cache/'train.npy')==old_info['files']['train']
                train_z=np.load(old_cache/'train.npy',mmap_mode='r'); source_norm=old_info['normalization']
            else: source_model=model; train_z=cache; source_norm=norm
            selected_z,_=encode_windows(source_model,subset,device,need_norm=False)
            reader_done=read(reader/'COMPLETE.json'); assert sha(reader/'final.pt')==reader_done['checkpoint_sha256']
            head=panel.Head().to(device).eval().requires_grad_(False)
            head_ck=torch.load(reader/'final.pt',map_location=device,weights_only=False)
            assert head_ck['step']==20000;head.load_state_dict(head_ck['model'],strict=True)
            risk=evaluate(head,selected_z,source_norm,subset,sp,device,panel)
            arrays[name]=risk
            macro=np.stack([risk[keys[:,0]==s].mean(0) for s in range(100)])
            rows[name]=dict(mean=macro.mean(0).tolist(),source_checkpoint_sha256=sha(source),
                           reader_checkpoint_sha256=sha(reader/'final.pt'),source_steps=20000 if name=='source20k' else 40000,
                           reader_steps=20000,cases=1600,systems=100,metric='raw state MSE')
            probes[name]=probe(train_z,selected_z,source_norm,train['gamma'],subset['gamma'])
        np.savez_compressed(out/'SELECTION_ERRORS.npz',keys=keys,system_ids=actual_ids,**arrays)
        reductions=1-np.array(rows['source40k']['mean'])/np.array(rows['source20k']['mean'])
        assert sha(old_checkpoint)==ANCESTOR_SHA and sha(protocol_path)==protocol_sha and panel.digest(model)==source_digest
        result=dict(status='COMPLETE',rows=rows,probe=probes,reduction_fraction=reductions.tolist(),
            worth_fixed_budget_three_seed_review=bool(reductions[-1]>=.10 and (reductions>0).sum()>=3),
            automatic_followup=False,selection_ids_only=True,report_evaluation=False,historical_report_already_seen=True,
            new_sealed_test=False,total_seconds=time.monotonic()-began,source_mode=plan['source_mode'],
            plan_sha256=sha(out/'PLAN.json'),manifest_sha256=sha(out/'SELECTION_MANIFEST.json'))
        write(out/'SUMMARY.json',result);write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json')))
        write(out/'EXIT.json',dict(exit_code=0,time=time.time())); print(json.dumps(result),flush=True)
    except BaseException:
        write(out/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write(out/'EXIT.json',dict(exit_code=1,time=time.time()));raise
    finally: signal.setitimer(signal.ITIMER_REAL,0)


def encode_windows(model, data, device, need_norm=True):
    import numpy as np
    import torch
    count=len(data['states']); rows=[]
    with torch.inference_mode():
        for start in range(0,count*72,256):
            ix=np.arange(start,min(start+256,count*72));s=ix//72;r=ix%72//9;t=23+ix%9
            x=torch.as_tensor(data['states'][s[:,None],r[:,None],t[:,None]+np.arange(-23,1)],device=device)
            u=torch.as_tensor(data['actions'][s[:,None],r[:,None],t[:,None]+np.arange(-23,0)],device=device)
            z=model.encode(x,u).cpu().numpy();assert z.shape==(len(ix),8) and np.isfinite(z).all();rows.append(z)
    z=np.concatenate(rows);norm=None
    if need_norm: norm=dict(mean=z.astype('float64').mean(0).tolist(),scale=z.astype('float64').std(0).clip(1e-8).tolist())
    return z.reshape(count,8,9,8),norm


def selection_specs(data):
    import numpy as np
    rows=[]
    for i, ident in enumerate(data['system_ids']):
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'sprii-dclean-eval-20260916:{int(ident)}'.encode()).digest()[:8],'little'))
        for _ in range(8):
            a=int(rng.integers(8));b=(a+int(rng.integers(1,8)))%8
            rows.append([i,a,int(rng.integers(23,32)),b,int(rng.integers(23,32))])
    return np.array(rows,dtype=int)


def evaluate(head, raw, norm, data, sp, device, panel):
    import numpy as np
    import torch
    errors=[]
    # All source codes are already derived solely from observed histories.
    with torch.inference_mode():
        for start in range(0,len(sp),48):
            a=sp[start:start+48];parts=[]
            for ri,ti in ((1,2),(3,4)):
                s,r,t=a[:,0,None],a[:,ri,None],a[:,ti,None]
                parts.append((data['states'][s,r,t][:,0],data['actions'][s,r,t+np.arange(32)],data['states'][s,r,t+np.array(H)]))
            x,u,y=[torch.as_tensor(np.concatenate([b[j] for b in parts]),device=device) for j in range(3)]
            z=panel.slot(raw,norm,a)
            errors.append((head(x,u,z)-y).square().mean(-1).cpu().numpy())
    # Original panel emits both ordered halves per batch. Reorder to the global
    # [all A recipients, all B recipients] manifest for this standalone analysis.
    first=[];second=[]
    for e in errors:
        assert len(e)%2==0;first.append(e[:len(e)//2]);second.append(e[len(e)//2:])
    result=np.concatenate(first+second);assert result.shape==(1600,4) and np.isfinite(result).all();return result


def probe(train, selected, norm, train_gamma, selected_gamma):
    import numpy as np
    from scipy.stats import spearmanr
    mu=np.asarray(norm['mean']);sd=np.asarray(norm['scale'])
    x=(train.reshape(-1,8)-mu)/sd;y=np.repeat(np.log(train_gamma),72).astype('float64');ym=y.mean()
    coef=np.linalg.solve(x.T@x+np.eye(8),x.T@(y-ym))
    v=(selected.reshape(-1,8)-mu)/sd;t=np.repeat(np.log(selected_gamma),72);pred=v@coef+ym
    rho=float(spearmanr(t,pred).statistic)
    return dict(r2=float(1-np.square(t-pred).sum()/np.square(t-t.mean()).sum()),spearman=rho if np.isfinite(rho) else None,
                spearman_defined=bool(np.isfinite(rho)),constant_prediction=bool(np.ptp(pred)==0),
                parameter='log_gamma',fit='train only ridge1',evaluation='selection100 only',labels_source_input=False)


def smoke():
    import numpy as np
    import torch
    sys.path.insert(0,str(CODE));from dali_context_torch import DALIContext
    torch.set_num_threads(1);torch.manual_seed(17)
    m=DALIContext(4,2,24);o=torch.optim.Adam(m.parameters(),lr=1e-4,eps=1e-8);r=np.random.default_rng(42)
    x=torch.randn(3,24,4);u=torch.randn(3,24,2)
    def step(m,o,r):
        lengths=torch.tensor(r.integers(1,24,len(x)));o.zero_grad();loss=m.prefix_loss(x,u,lengths);loss.backward();o.step()
        return float(loss),lengths
    step(m,o,r);ck=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state(),numpy_prefix_rng=r.bit_generator.state))
    ref_loss,ref_lengths=step(m,o,r);m2=DALIContext(4,2,24);o2=torch.optim.Adam(m2.parameters(),lr=1e-4,eps=1e-8);r2=np.random.default_rng(0)
    restore(m2,o2,ck,r2);loss,lengths=step(m2,o2,r2)
    assert torch.equal(ref_lengths,lengths) and loss==ref_loss
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
    data={'system_ids':np.arange(1000,1100)};sp=selection_specs(data);assert sp.shape==(800,5) and (sp[:,1]!=sp[:,3]).all()
    # Check the real standalone evaluator's batch/half ordering and raw-MSE
    # reduction against direct indexing, without any real dataset or model.
    from types import SimpleNamespace
    data['states']=np.random.default_rng(1).normal(size=(100,8,64,4)).astype('float32')
    data['actions']=np.zeros((100,8,64,2),dtype='float32')
    class ZeroHead(torch.nn.Module):
        def forward(self,x,u,z):return torch.zeros((len(x),4,4),dtype=x.dtype)
    dummy=SimpleNamespace(slot=lambda raw,norm,a:torch.zeros((2*len(a),64)))
    risk=evaluate(ZeroHead(),None,None,data,sp,torch.device('cpu'),dummy)
    keys=np.concatenate([sp,sp[:,[0,3,4,1,2]]])
    expected=np.stack([np.square(data['states'][s,r,t+np.array(H)]).mean(-1) for s,r,t,_,_ in keys])
    np.testing.assert_allclose(risk,expected,rtol=1e-6,atol=1e-7)
    calls=inspect_rng_path(CODE/'dali_dclean_source.py',CODE/'dali_context_torch.py')
    return dict(status='PASS',synthetic_only=True,cpu_adam_resume_bitwise_equal=True,prefix_rng_identical=True,
                all_parameters_identical=True,selection_pairs=1600,selection_evaluation_order_and_raw_mse=True,actual_dataset_access=False,
                actual_model_training=False,source_loop_calls=calls,runner_sha256=sha(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('smoke','run'));p.add_argument('--output')
    p.add_argument('--formal-root',default='./runs/dclean_formal')
    p.add_argument('--helper',default='shared/dclean_external.py')
    p.add_argument('--panel',default='shared/dclean_panel.py')
    p.add_argument('--official-root',default=str(HERE.parent/'vendor/DALI'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.phase=='smoke':print(json.dumps(smoke(),indent=2))
    else:
        assert a.output is not None
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');run(a)
