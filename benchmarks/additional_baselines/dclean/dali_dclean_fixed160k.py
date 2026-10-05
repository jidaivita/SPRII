#!/usr/bin/env python3
"""Fixed DALI source160k x common reader20k, three source x three reader seeds.

Independent follow-up to the completed source80k and selection-only seed0 gate.
No search, no new controls, no automatic partial restart. All hot outputs local.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import copy
import csv
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE_SHA = '9338037016de08e56d5bdaa7bc0f6f734483e07d6c514f090896992e93441b5c'
COMPONENT_SHA = '069e15f406a4344ae0e61bf4fdbb0fb92dfccbe4514225a0faea452772752f32'
READER_SHA = '02f99e3c5c96f422274530c597788971835c4776a68d05e73905005fe87cc7a1'
HELPER_SHA = '59c56a4caac12bcefcf3b6bca2a4353f913bd2b1f4aa6bc645dcb0c1f823821f'
PANEL_SHA = 'b64ac20c7a847703b783f1db5ad626499611122dccc3ee1dea0ff5b21b062ecc'
SENSITIVITY_SHA = '045b0dcab703a1b97efff50b5cc98e7a19a704b5b99c3e98b06f9e7d5b27334e'
ADOPT_SOURCE_SHA = '4a55a7e0d1b15aae67ceb003935031de600c856af5979dfdec46f8590483e7f3'
ADOPT_READER_SHA = '6d1fa0fdc11e5ca7ce4df098cadc3ddeacff4a1b4b845acf0d577140cdcc1e77'
FIXED80_SHA = 'eca8a2eb85d0a52b1f2a8c3ee861c2cc0be7b3c4ebcddf448695f6f26319754e'
GATE160_SHA = 'ec9aead1c0194100c2c66719996b36ee44a21fc0d2f3d5df28b72f61e9f3b031'
GATE_PLAN_SHA = 'cf926a2678d1e6d0babb50084663c10f25da457df99e99247b96f02cd2ce6bb2'
H = [1, 4, 16, 32]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.pending')
    q.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); q.replace(p)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def dependencies(a):
    expected = {HERE/'dali_dclean_fixed80k.py': FIXED80_SHA,
                HERE/'dali_dclean_160k_selection.py': GATE160_SHA,
                HERE/'dali_dclean_source.py': SOURCE_SHA,
                HERE/'dali_context_torch.py': COMPONENT_SHA,
                HERE/'dali_dclean_common_reader.py': READER_SHA,
                Path(a.helper): HELPER_SHA, Path(a.panel): PANEL_SHA,
                HERE.parent/'cophy/dclean_budget_sensitivity.py': SENSITIVITY_SHA}
    for p, h in expected.items(): assert sha(p) == h, (str(p), 'dependency SHA changed')
    assert sha(Path(a.official_root)/'dreamerv3_compat/dreamerv3/nets.py') == '0233a0d7aedce5f137a29a5ce3b49d100db1c0d1322d1c8b6238e66235539b9d'
    return {str(p): h for p, h in expected.items()}


def old_source(a, seed):
    previous = load(HERE/'dali_dclean_fixed80k.py','_fixed80_previous')
    assert sha(HERE/'dali_dclean_fixed80k.py') == FIXED80_SHA
    ns=argparse.Namespace(root=a.old_root,old_root='./runs/dclean_fixed40k_formal',original20k_root=a.original20k_root)
    assert previous.source_complete(ns,seed)
    p=Path(a.old_root)/'sources'/f'seed{seed}'
    return p,read(p/'CONFIG.json'),read(p/'SUMMARY.json')


def adoption(a):
    p=Path(a.selection_gate)
    s,d,plan=(read(p/n) for n in ('SUMMARY.json','COMPLETE.json','PLAN.json'))
    assert d['status']=='COMPLETE' and read(p/'EXIT.json')['exit_code']==0
    assert sha(p/'SUMMARY.json')==d['summary_sha256']
    assert s['plan_sha256']==sha(p/'PLAN.json')==GATE_PLAN_SHA
    assert plan['runner']['sha256']==GATE160_SHA
    assert s['decision']['worth_fixed_budget_three_seed_review'] is True
    assert s['decision']['mean_horizon_ratio']<=.95 and max(s['decision']['ratio160k_to80k'])<=1.10
    assert s['selection_ids_only'] is True and s['report_evaluation'] is False
    assert plan['source_seed']==plan['reader_seed']==0
    assert plan['source_total_updates']==160000 and plan['source_new_updates']==80000 and plan['reader_updates']==20000
    src,reader=p/'source160k',p/'common160k/readers/DALI_s0/r0'
    assert sha(src/'final.pt')==ADOPT_SOURCE_SHA==s['rows']['source160k']['source']['sha256']
    assert sha(reader/'final.pt')==ADOPT_READER_SHA==s['rows']['source160k']['reader']['sha256']
    assert read(src/'COMPLETE.json')['checkpoint_sha256']==ADOPT_SOURCE_SHA
    assert read(src/'COMPLETE.json')['step']==160000
    rd=read(reader/'COMPLETE.json')
    assert rd['step']==20000 and rd['checkpoint_sha256']==ADOPT_READER_SHA
    assert read(reader/'RUN.json')['steps']==20000
    assert plan['source_ancestor']['sha256']==sha(Path(a.old_root)/'sources/seed0/final.pt')
    return p,src,reader,plan


def source(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE)); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dependencies(a)
    root = Path(a.root); out = root/'sources'/f'seed{a.seed}'
    assert not out.exists(), 'No source overwrite or implicit partial restart'
    out.mkdir(parents=True)
    helper = load(a.helper, 'dclean_external')
    api = load(HERE/'dali_dclean_source.py', '_fixed160k_source')
    sensitivity = load(HERE.parent/'cophy/dclean_budget_sensitivity.py', '_fixed160k_resume')
    old, config, previous = old_source(a, a.seed)
    ck = torch.load(old/'final.pt', map_location='cpu', weights_only=False)
    previous_api=load(HERE/'dali_dclean_fixed80k.py','_fixed80_old_checkpoint')
    assert previous_api.validate_checkpoint(ck,config,previous)
    assert ck['step'] == 80000 and ck['source_seed'] == a.seed
    assert all(k in ck for k in ['model','optimizer','torch_rng','numpy_prefix_rng']), 'Incomplete source state; no scratch fallback'
    protocol_sha = sha(helper.ROOT/'PROTOCOL.json'); helper.contract()
    assert sha(helper.ROOT/'PROTOCOL.json') == protocol_sha
    sensitivity.inspect_rng_path(HERE/'dali_dclean_source.py', HERE/'dali_context_torch.py')
    config = copy.deepcopy(config)
    config.update(steps=160000, continuation_code_sha256=sha(__file__), ancestor_checkpoint_sha256=sha(old/'final.pt'),
                  ancestor_config_sha256=previous['config_sha256'], source_new_updates=80000,
                  selection='Fixed160000 endpoint following seed0 selection-only budget diagnostic; all3 seeds included',
                  rng_restore_scope='CPU Torch and independent prefix NumPy generator; step-keyed data. Original path has no CUDA-random operation.')
    if a.seed == 0:
        gate, adopted, _, _ = adoption(a)
        config.update(adopted_existing_source=True, adopted_plan_sha256=sha(gate/'PLAN.json'))
    else: config['adopted_existing_source'] = False
    write(out/'CONFIG.json', config); cfgsha = sha(out/'CONFIG.json')
    write(out/'RUN.json', dict(pid=os.getpid(), start=time.time(), config_sha256=cfgsha,
        total_updates=160000, updates_executed_this_stage=0 if a.seed==0 else 80000,
        source_seed=a.seed, ancestor=str(old/'final.pt'), ancestor_sha256=sha(old/'final.pt')))
    began = time.monotonic()
    helper.seed_all(a.seed); model = api.build_model(config,a.official_root).cuda()
    if a.seed == 0:
        shutil.copy2(adopted/'final.pt', out/'final.pt')
        assert sha(out/'final.pt') == ADOPT_SOURCE_SHA
        adopted_ck = torch.load(out/'final.pt', map_location='cpu', weights_only=False)
        assert adopted_ck['step'] == 160000 and adopted_ck['source_seed'] == 0
        assert adopted_ck['source_ancestor_sha256'] == config['ancestor_checkpoint_sha256']
        assert adopted_ck['plan_sha256'] == config['adopted_plan_sha256']
        assert {int(v['step']) for v in adopted_ck['optimizer']['state'].values()} == {160000}
        model.load_state_dict(adopted_ck['model'],strict=True)
        final_dev = read(adopted/'COMPLETE.json')['final160k_dev']
        shutil.copy2(adopted/'train.jsonl',out/'train_adopted.jsonl')
        write(out/'ADOPTED.json', dict(source_path=str(adopted), source_sha256=ADOPT_SOURCE_SHA,
            original_plan_sha256=config['adopted_plan_sha256'], original_config_sha256=sha(adopted/'CONFIG.json'),
            unchanged_checkpoint_bytes=True, normalized_manifest_only=True))
    else:
        opt = torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-8)
        rng = np.random.default_rng(202609250000+a.seed)
        sensitivity.restore(model,opt,ck,rng)
        assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items())
        assert {int(v['step']) for v in opt.state.values()} == {80000}
        for group in opt.param_groups:
            assert group['lr']==1e-4 and group['eps']==1e-8 and group['weight_decay']==0 and tuple(group['betas'])==(.9,.999)
        model.train()
        with (out/'train.jsonl').open('x') as log:
            for step in range(80001,160001):
                b=api.make_batch(helper.data('train'),helper.specs(a.seed,step),'cuda:0')
                lengths=torch.as_tensor(rng.integers(1,24,96),device='cuda:0')
                opt.zero_grad(set_to_none=True); loss=model.objective(*b,lengths)
                assert torch.isfinite(loss); loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1000,error_if_nonfinite=True); opt.step()
                if step==80001 or step%100==0:
                    row=dict(step=step,loss=float(loss),gradient_norm=float(grad),seconds=time.monotonic()-began)
                    log.write(json.dumps(row)+'\n');log.flush();write(out/'PROGRESS.json',row)
        assert {int(v['step']) for v in opt.state.values()} == {160000}
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=160000,source_seed=a.seed,
            config_sha256=cfgsha,torch_rng=torch.get_rng_state(),numpy_prefix_rng=rng.bit_generator.state,
            ancestor_checkpoint_sha256=config['ancestor_checkpoint_sha256']),out/'final.pt')
        final_dev=api.dev_metrics(model,helper,'cuda:0')
    assert sha(helper.ROOT/'PROTOCOL.json')==protocol_sha
    assert sha(old/'final.pt')==config['ancestor_checkpoint_sha256']
    summary=dict(status='COMPLETE',source_seed=a.seed,steps=160000,config_sha256=cfgsha,
        final_checkpoint_sha256=sha(out/'final.pt'),initial_dev=previous['final_dev'],final_dev=final_dev,
        training_seconds=time.monotonic()-began,adopted=a.seed==0,closed_loop=False,test_read=False)
    write(out/'SUMMARY.json',summary)
    write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json'),final_checkpoint_sha256=sha(out/'final.pt')))
    write(out/'EXIT.json',dict(exit_code=0,time=time.time()))


def source_complete(a, seed):
    p=Path(a.root)/'sources'/f'seed{seed}'
    if not (p/'COMPLETE.json').exists():return False
    cfg,s,d=(read(p/n) for n in ['CONFIG.json','SUMMARY.json','COMPLETE.json'])
    assert cfg['steps']==s['steps']==160000 and cfg['source_seed']==s['source_seed']==seed
    assert cfg['code_sha256']==SOURCE_SHA and cfg['continuation_code_sha256']==sha(__file__)
    assert sha(p/'CONFIG.json')==s['config_sha256']==read(p/'RUN.json')['config_sha256']
    assert sha(p/'SUMMARY.json')==d['summary_sha256'] and d['status']=='COMPLETE'
    assert sha(p/'final.pt')==d['final_checkpoint_sha256']==s['final_checkpoint_sha256']
    assert read(p/'EXIT.json')['exit_code']==0
    if seed==0:assert d['final_checkpoint_sha256']==ADOPT_SOURCE_SHA
    old,cfg20,_=old_source(a,seed)
    assert cfg['ancestor_checkpoint_sha256']==sha(old/'final.pt')
    for k in ['normalization','data_hashes','architecture','optimizer','history_states','history_actions','objective']:
        assert cfg[k]==cfg20[k],k
    return True


def install_reader160k(module):
    """Two narrow source-manifest adaptations; original reader fit/eval unchanged."""
    class Replace(ast.NodeTransformer):
        count_budget=0; count_checkpoint=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='source_steps':
                assert ast.unparse(node.value)=="1000 if args.profile == 'gate' else 20000"
                node.value=ast.Constant(160000);self.count_budget+=1
            return self.generic_visit(node)
        def visit_Assert(self,node):
            if ast.unparse(node.test)=="ck['step'] == source_steps and ck['config_sha256'] == source_summary['config_sha256']":
                node.test=ast.parse('_validate_160k_checkpoint(ck, source_config, source_summary)',mode='eval').body
                self.count_checkpoint+=1
            return self.generic_visit(node)
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.bootstrap))); transform=Replace();tree=transform.visit(tree)
    assert (transform.count_budget,transform.count_checkpoint)==(1,1)
    module._validate_160k_checkpoint=validate_checkpoint
    exec(compile(ast.fix_missing_locations(tree),'<fixed160k source manifest adapter>','exec'),module.__dict__)
    return dict(source_budget_replacements=1,checkpoint_manifest_replacements=1,
                fit_function_unchanged=True,evaluate_function_unchanged=True)


def validate_checkpoint(ck,cfg,summary):
    assert ck['step']==cfg['steps']==summary['steps']==160000 and ck['source_seed']==cfg['source_seed']
    if cfg['adopted_existing_source']:
        assert cfg['source_seed']==0 and ck['plan_sha256']==cfg['adopted_plan_sha256']
        assert ck['source_ancestor_sha256']==cfg['ancestor_checkpoint_sha256']
    else:
        assert ck['config_sha256']==summary['config_sha256']
        assert ck['ancestor_checkpoint_sha256']==cfg['ancestor_checkpoint_sha256']
    return True


def reader(a):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);dependencies(a)
    assert source_complete(a,a.seed)
    api=load(HERE/'dali_dclean_common_reader.py','_fixed160k_common')
    adaptation=install_reader160k(api)
    common=Path(a.root)/'common'/f'seed{a.seed}'
    assert not (common/f'COMPLETE_r{a.reader}.json').exists(), 'Completed reader must be reused by queue, not rerun'
    ns=argparse.Namespace(action='all',profile='formal',output=str(common),
        source_run=str(Path(a.root)/'sources'/f'seed{a.seed}'),seed=a.seed,reader=a.reader,
        data_helper=a.helper,panel_module=a.panel,pilot_module=str(HERE/'dali_dclean_source.py'),official_root=a.official_root)
    helper,panel,pilot,c=api.bootstrap(ns)
    api.real_interface_smoke(helper,panel,pilot,c)
    panel.cache('DALI',a.seed);api.verify_cache(helper,panel,c)
    reuse=None
    if a.seed==0 and a.reader==0:
        gate,_,original,_=adoption(a)
        oldcache=gate/'common160k/cache/DALI_s0';newcache=common/'cache/DALI_s0'
        assert sha(oldcache/'train.npy')==sha(newcache/'train.npy')
        assert read(oldcache/'COMPLETE.json')['normalization']==read(newcache/'COMPLETE.json')['normalization']
        copied=common/'readers/DALI_s0/r0';assert not copied.exists()
        shutil.copytree(original,copied)
        assert sha(copied/'final.pt')==ADOPT_READER_SHA
        assert read(copied/'RUN.json')['initial_sha256']==read(common/'INTERFACE_SMOKE_r0.json')['head_initial_sha256']
        oldrun=read(copied/'RUN.json')
        assert oldrun['source_cache_sha256']==sha(oldcache/'COMPLETE.json')
        ck=torch.load(copied/'final.pt',map_location='cpu',weights_only=False)
        assert ck['step']==20000 and ck['run_sha256']==sha(copied/'RUN.json')
        reuse=dict(path=str(original),checkpoint_sha256=ADOPT_READER_SHA,
            original_run_sha256=sha(original/'RUN.json'),original_protocol_sha256=oldrun['protocol_sha256'],
            original_cache_sha256=sha(oldcache/'COMPLETE.json'),new_cache_sha256=sha(newcache/'COMPLETE.json'),
            actual_training_array_sha256=sha(newcache/'train.npy'),unchanged_model_optimizer_and_run=True,
            original_training_normalization_equal=True,updates_executed_this_stage=0)
        write(common/'REUSED_READER_r0.json',reuse)
    panel.fit('DALI',a.seed,a.reader,'matched')
    panel.evaluate('DALI',a.seed,a.reader);panel.probe('DALI',a.seed)
    api.final_audit(helper,panel,c)
    summary=read(common/f'SUMMARY_r{a.reader}.json')
    summary.update(source_budget_comparison='DALI160k fixed source budget; original DALI20k and controls20k remain distinct. All readers20k. No equal FLOPs claim.',
        fixed160k_runner_sha256=sha(__file__),source_extension_adapter=adaptation,
        reused_reader=reuse,historical_validation=True,new_sealed_test=False)
    write(common/f'SUMMARY_r{a.reader}.json',summary)
    write(common/f'COMPLETE_r{a.reader}.json',dict(status='COMPLETE',summary_sha256=sha(common/f'SUMMARY_r{a.reader}.json')))
    write(common/f'EXIT_all_r{a.reader}.json',dict(exit_code=0,time=time.time(),code_sha256=sha(__file__)))


def cell_complete(a,seed,r):
    p=Path(a.root)/'common'/f'seed{seed}'
    if not (p/f'COMPLETE_r{r}.json').exists():return False
    s,d=read(p/f'SUMMARY_r{r}.json'),read(p/f'COMPLETE_r{r}.json')
    assert d['status']=='COMPLETE' and d['summary_sha256']==sha(p/f'SUMMARY_r{r}.json')
    assert s['source_seed']==seed and s['reader_seed']==r and s['source_steps']==160000 and s['reader_steps']==20000
    assert s['fixed160k_runner_sha256']==sha(__file__) and s['code_sha256']==READER_SHA
    assert s['same_keys_and_null_errors'] and s['cases']==1600 and s['report_systems']==100
    assert s['source_sha256']==sha(Path(a.root)/'sources'/f'seed{seed}'/'final.pt')
    assert s['protocol_sha256']==sha(p/'PROTOCOL.json')
    arrays=p/'results'/f'DALI_s{seed}_r{r}'/'per_case.npz'
    assert s['arrays_sha256']==sha(arrays)
    head=p/'readers'/f'DALI_s{seed}'/f'r{r}';hd=read(head/'COMPLETE.json')
    assert hd['step']==read(head/'RUN.json')['steps']==20000 and hd['checkpoint_sha256']==sha(head/'final.pt')
    assert read(p/f'EXIT_all_r{r}.json')['exit_code']==0
    return True


























def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['source','reader'])
    p.add_argument('--root',required=True)
    p.add_argument('--old-root',default='./runs/dclean_fixed80k_formal')
    p.add_argument('--original20k-root',default='./runs/dclean_formal')
    p.add_argument('--selection-gate',default='./runs/dclean160k_selection_seed0')
    p.add_argument('--official-root',default=str(HERE.parent/'vendor/DALI'))
    p.add_argument('--helper',default='shared/dclean_external.py')
    p.add_argument('--panel',default='shared/dclean_panel.py')
    p.add_argument('--controls-root',default='shared/dclean')
    p.add_argument('--cadm-root')
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--reader',type=int,choices=[0,1,2],default=0)
    p.add_argument('--gpu',type=int,choices=[3],default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--after-recovery')
    p.add_argument('--wait-seconds',type=int,default=14400)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--lock-root',default='runs/baseline_reference/gpu_locks')
    p.add_argument('--dali-lock-root',default='./locks')
    p.add_argument('--stage-timeout-seconds',type=int,default=1200)
    p.add_argument('--lane-timeout-seconds',type=int,default=3600)
    return p



if __name__ == "__main__":
    args = parser().parse_args()
    if args.action not in ("source", "reader"):
        raise ValueError("Only source and reader operations are distributed")
    {"source": source, "reader": reader}[args.action](args)
