#!/usr/bin/env python3
"""One fixed source0 40k->80k DALI budget gate, selection100 only.

Fresh original reader0/20k, immutable original40k comparison; no report/test
inference, no automatic promotion, no new controls or seed search.
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
ANCESTOR_SHA = '30e92f9f8f42b8b6d0d8e8fef3c36fd0df220503e3456f0ebd1a0deb3c5287cf'
READER40_SHA = '61a840d4f242289383d123e5cc237287aa1046a6ff726880446e9d2a48bb37bb'
DEPENDENCIES = {'dali_dclean_source.py': '9338037016de08e56d5bdaa7bc0f6f734483e07d6c514f090896992e93441b5c', 'dali_context_torch.py': '069e15f406a4344ae0e61bf4fdbb0fb92dfccbe4514225a0faea452772752f32', 'dali_dclean_fixed40k.py': '7ab16f06b9c20ed2832a4bb9adacdf1029a69f6b5d0ca19818c6bd7b97f6524b', '../cophy/dclean_budget_sensitivity.py': '045b0dcab703a1b97efff50b5cc98e7a19a704b5b99c3e98b06f9e7d5b27334e'}
HELPER_SHA = '59c56a4caac12bcefcf3b6bca2a4353f913bd2b1f4aa6bc645dcb0c1f823821f'
PANEL_SHA = 'b64ac20c7a847703b783f1db5ad626499611122dccc3ee1dea0ff5b21b062ecc'
H = [1, 4, 16, 32]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.pending.' + str(os.getpid()))
    q.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); q.replace(p)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def artifact(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def budget_decision(old_means, new_means):
    import numpy as np
    old, new = np.asarray(old_means, float), np.asarray(new_means, float)
    assert old.shape == new.shape == (4,) and np.isfinite(old).all() and np.isfinite(new).all()
    assert (old > 0).all() and (new >= 0).all()
    ratio = new / old
    return dict(ratio80k_to40k=ratio.tolist(), mean_horizon_ratio=float(ratio.mean()),
        all_horizons_ratio_at_most_1p10=bool((ratio <= 1.10).all()),
        worth_fixed_budget_three_seed_review=bool(ratio.mean() <= .95 and (ratio <= 1.10).all()))


def conditional_errors(api, panel, head, raw, norm, data, specs, device, condition):
    """Reuse the already-audited target/mask/order evaluator, change only donor code."""
    import numpy as np
    import torch
    assert condition in ('matched', 'wrong', 'zero', 'null')
    original = panel.slot
    if condition == 'wrong':
        def slot(z, n, s):
            donor = (s[:, 0] + 1) % len(data['system_ids'])
            assert np.all(donor != s[:, 0])
            return original(z, n, s, wrong=donor)
        panel.slot = slot
    elif condition in ('zero', 'null'):
        panel.slot = lambda z, n, s: torch.zeros((2 * len(s), 64), device=device)
    try: return api.evaluate(head, raw, norm, data, specs, device, panel)
    finally: panel.slot = original


def run(a):
    import numpy as np
    import torch
    out = Path(a.output).resolve()
    assert str(out).startswith('runtime/') and not out.exists(), 'Fresh independent local output required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(a.gpu), 'Root must launch in the assigned GPU mask'
    assert a.device == 'cuda:0'
    out.mkdir(parents=True); started = time.monotonic(); active_stage = 'validation'; stages = []
    def expired(*_): raise TimeoutError('Bounded80k gate time exhausted in ' + active_stage)
    signal.signal(signal.SIGALRM, expired)
    def stage(name, cap):
        nonlocal active_stage
        active_stage = name; remaining = 1800 - (time.monotonic() - started)
        assert remaining > 0
        signal.setitimer(signal.ITIMER_REAL, min(cap, remaining))
        stages.append(dict(stage=name, started_seconds=time.monotonic()-started,
                           stage_cap_seconds=cap, effective_cap_seconds=min(cap, remaining)))
        write(out/'STAGES.json', stages)
    stage('source', 900)
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1); sys.path.insert(0, str(HERE))
        for path, digest in DEPENDENCIES.items(): assert sha(HERE/path) == digest, path
        assert sha(a.helper) == HELPER_SHA and sha(a.panel) == PANEL_SHA
        helper = load(a.helper, 'dclean_external'); panel = load(a.panel, '_80k_original_panel')
        source_api = load(HERE/'dali_dclean_source.py', '_80k_source_api')
        previous_api = load(HERE/'dali_dclean_fixed40k.py', '_80k_40k_verifier')
        sensitivity = load(HERE/'../cophy/dclean_budget_sensitivity.py', '_80k_selection_api')
        previous = Path(a.formal_root); src = previous/'sources/seed0'; original_cp = src/'final.pt'
        assert previous_api.source_complete(argparse.Namespace(root=str(previous), old_root=a.original20k_root), 0)
        config, prior_summary = read(src/'CONFIG.json'), read(src/'SUMMARY.json')
        ck = torch.load(original_cp, map_location='cpu', weights_only=False)
        assert sha(original_cp) == ANCESTOR_SHA
        assert previous_api.validate_checkpoint(ck, config, prior_summary)
        assert ck['step'] == 40000 and ck['source_seed'] == 0
        assert all(k in ck for k in ('model', 'optimizer', 'torch_rng', 'numpy_prefix_rng'))
        old_reader = previous/'common/seed0/readers/DALI_s0/r0'
        assert sha(old_reader/'final.pt') == READER40_SHA == read(old_reader/'COMPLETE.json')['checkpoint_sha256']
        assert read(old_reader/'COMPLETE.json')['step'] == 20000
        protocol_sha = sha(helper.ROOT/'PROTOCOL.json'); protocol = helper.contract()
        assert sha(helper.ROOT/'PROTOCOL.json') == protocol_sha
        calls = sensitivity.inspect_rng_path(HERE/'dali_dclean_source.py', HERE/'dali_context_torch.py')
        plan = dict(source_seed=0, reader_seed=0, source_total_updates=80000, source_new_updates=40000,
            reader_updates=20000, source_ancestor=artifact(original_cp), reader40k=artifact(old_reader/'final.pt'),
            source_selection='fixed80000 final endpoint', reader_selection='fresh fixed20000 endpoint',
            source_optimizer='unchanged Adam1e-4 eps1e-8 clip1000', hyperparameters_changed=False,
            source40k_manifest=artifact(src/'CONFIG.json'), current_common_reader_sha256=PANEL_SHA,
            selected_systems=protocol['selection_ids'], excluded_report_systems=protocol['report_ids'],
            evaluation='same selection100,1600 recipient keys; H1/4/16/32 raw state MSE',
            threshold='mean(MSE80k/MSE40k across four H)<=.95 AND every H ratio<=1.10',
            total_seconds_cap=1800, phase_caps=dict(source=900, reader=600, evaluation=600),
            actual_timer='min(current phase cap, remaining total1800); caps are not additive',
            report_evaluation=False, new_sealed_test=False, historical_report_already_seen=True,
            automatic_followup=False, scope='released context/forward component; not native DALI closed-loop',
            rng_restore_scope='CPU Torch plus independent NumPy prefix generator; deterministic step-keyed data, no audited CUDA stochastic training op',
            audited_training_calls=calls, runner=artifact(__file__), dependencies=DEPENDENCIES)
        write(out/'PLAN.json',plan); write(out/'RUN.json',dict(pid=os.getpid(),gpu=a.gpu,time=time.time(),plan_sha256=sha(out/'PLAN.json')))
        helper.seed_all(0); device=torch.device(a.device)
        model=source_api.build_model(config,a.official_root).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-8); prefix=np.random.default_rng(202609250000)
        sensitivity.restore(model,opt,ck,prefix)
        assert {int(v['step']) for v in opt.state.values()} == {40000}
        assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items())
        for g in opt.param_groups:
            assert g['lr']==1e-4 and g['eps']==1e-8 and g['weight_decay']==0 and tuple(g['betas'])==(.9,.999)
        source=out/'source80k';source.mkdir()
        write(source/'CONFIG.json',dict(original=config,steps=80000,source_seed=0,plan_sha256=sha(out/'PLAN.json')))
        train_started=time.monotonic();model.train()
        with (source/'train.jsonl').open('x') as log:
            for step in range(40001,80001):
                batch=source_api.make_batch(helper.data('train'),helper.specs(0,step),device)
                lengths=torch.as_tensor(prefix.integers(1,24,96),device=device)
                opt.zero_grad(set_to_none=True);loss=model.objective(*batch,lengths)
                assert torch.isfinite(loss);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1000,error_if_nonfinite=True);opt.step()
                if step==40001 or step%100==0:
                    row=dict(step=step,loss=float(loss),gradient_norm=float(grad),seconds=time.monotonic()-train_started)
                    log.write(json.dumps(row)+'\n');log.flush();write(source/'PROGRESS.json',row)
        assert {int(v['step']) for v in opt.state.values()}=={80000}
        final=source/'final.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=80000,source_seed=0,
            torch_rng=torch.get_rng_state(),numpy_prefix_rng=prefix.bit_generator.state,
            source_ancestor_sha256=ANCESTOR_SHA,config_sha256=sha(source/'CONFIG.json'),plan_sha256=sha(out/'PLAN.json')),final)
        dev=source_api.dev_metrics(model,helper,device)
        write(source/'COMPLETE.json',dict(status='COMPLETE',step=80000,checkpoint_sha256=sha(final),
            initial40k_dev=prior_summary['final_dev'],final80k_dev=dev,report_evaluation=False,
            source_seconds=time.monotonic()-train_started))
        model.eval().requires_grad_(False); source_digest=panel.digest(model)
        stage('reader',600)
        train=helper.data('train');cache,norm=sensitivity.encode_windows(model,train,device)
        common=out/'common80k';cache_dir=common/'cache/DALI_s0';cache_dir.mkdir(parents=True)
        np.save(cache_dir/'train.npy',cache)
        write(cache_dir/'COMPLETE.json',dict(status='COMPLETE',normalization=norm,
            source=dict(checkpoint=str(final),sha256=sha(final)),files={'train':sha(cache_dir/'train.npy')},
            split='train only',source_frozen=True))
        panel.ROOT=common;panel.contract=lambda:plan;write(common/'PROTOCOL.json',plan)
        original_load=panel.load_cache
        def train_only(method,seed,split):
            assert method=='DALI' and seed==0 and split=='train'
            return original_load(method,seed,split)
        panel.load_cache=train_only
        panel.fit('DALI',0,0,'matched')
        new_reader=common/'readers/DALI_s0/r0'
        assert read(new_reader/'COMPLETE.json')['step']==20000
        assert read(new_reader/'RUN.json')['initial_sha256']==read(old_reader/'RUN.json')['initial_sha256']
        assert panel.digest(model)==source_digest
        stage('evaluation',600)
        val=helper.data('val');selected=set(protocol['selection_ids']);excluded=set(protocol['report_ids'])
        indices=np.array([i for i,v in enumerate(val['system_ids']) if int(v) in selected])
        assert len(indices)==100 and selected.isdisjoint(excluded)
        subset={k:v[indices] for k,v in val.items()};del val
        assert set(map(int,subset['system_ids']))==selected
        specs=sensitivity.selection_specs(subset);keys=np.concatenate([specs,specs[:,[0,3,4,1,2]]])
        actual_ids=subset['system_ids'][keys[:,0]]
        assert len(keys)==1600 and not set(map(int,actual_ids)) & excluded
        write(out/'SELECTION_MANIFEST.json',dict(keys=keys.tolist(),system_ids=actual_ids.tolist(),
            wrong_system_map={str(i):int((i+1)%100) for i in range(100)},source_seeds=[0],reader_seeds=[0],
            horizons=H,report_ids_absent=True,selection_only=True))
        rows={};arrays={};probes={};per_system={};sources_before={}
        old_cache=previous/'common/seed0/cache/DALI_s0';old_info=read(old_cache/'COMPLETE.json')
        assert old_info['source']['sha256']==ANCESTOR_SHA and sha(old_cache/'train.npy')==old_info['files']['train']
        null_dir=previous/'common/seed0/readers/null_shared/r0'
        assert sha(null_dir/'final.pt')==read(null_dir/'COMPLETE.json')['checkpoint_sha256']
        null_ck=torch.load(null_dir/'final.pt',map_location=device,weights_only=False)
        assert null_ck['step']==20000 and null_ck['initial_sha256']==read(old_reader/'RUN.json')['initial_sha256']
        null_head=panel.Head().to(device).eval().requires_grad_(False);null_head.load_state_dict(null_ck['model'],strict=True)
        for name,cp,reader in [('source40k',original_cp,old_reader),('source80k',final,new_reader)]:
            if name=='source40k':
                sm=source_api.build_model(config,a.official_root).to(device);sm.load_state_dict(ck['model'],strict=True)
                sm.eval().requires_grad_(False);train_z=np.load(old_cache/'train.npy',mmap_mode='r');sn=old_info['normalization']
            else:sm=model;train_z=cache;sn=norm
            before=panel.digest(sm);sources_before[name]=before
            z,_=sensitivity.encode_windows(sm,subset,device,need_norm=False)
            h=panel.Head().to(device).eval().requires_grad_(False); hc=torch.load(reader/'final.pt',map_location=device,weights_only=False)
            assert hc['step']==20000 and sha(reader/'final.pt')==read(reader/'COMPLETE.json')['checkpoint_sha256']
            h.load_state_dict(hc['model'],strict=True);hd=panel.digest(h)
            rows[name]=dict(source_steps=40000 if name=='source40k' else 80000,reader_steps=20000,
                source=artifact(cp),reader=artifact(reader/'final.pt'),metrics={},cases=1600,systems=100)
            for condition in ('matched','wrong','zero'):
                error=conditional_errors(sensitivity,panel,h,z,sn,subset,specs,device,condition)
                arrays[name+'_'+condition]=error
                macro=np.stack([error[keys[:,0]==s].mean(0) for s in range(100)])
                per_system[name+'_'+condition]=macro;rows[name]['metrics'][condition]=macro.mean(0).tolist()
            if 'null' not in arrays:
                nd=panel.digest(null_head)
                arrays['null']=conditional_errors(sensitivity,panel,null_head,z,sn,subset,specs,device,'null')
                assert panel.digest(null_head)==nd
            assert panel.digest(sm)==before and panel.digest(h)==hd
            probes[name]=sensitivity.probe(train_z,z,sn,train['gamma'],subset['gamma'])
            q=(z-np.asarray(sn['mean']))/np.asarray(sn['scale']);means=q.reshape(100,72,8).mean(1)
            within=float(((q.reshape(100,72,8)-means[:,None])**2).sum(-1).mean())
            between=float(((means-means.mean(0))**2).sum(-1).mean())
            probes[name]['geometry']=dict(within=within,between=between,ratio=between/max(within,1e-12),
                                         normalization='training latent statistics')
            probes[name]['latent_train_std']=np.asarray(train_z).reshape(-1,8).astype('float64').std(0).tolist()
        np.savez_compressed(out/'SELECTION_ERRORS.npz',keys=keys,system_ids=actual_ids,**arrays)
        decision=budget_decision(rows['source40k']['metrics']['matched'],rows['source80k']['metrics']['matched'])
        delta=per_system['source80k_matched']-per_system['source40k_matched']
        rng=np.random.default_rng(2026092580); ix=rng.integers(0,100,(2000,100))
        paired=dict(delta80k_minus40k=delta.mean(0).tolist(),
            paired_system_ci95=np.quantile(delta[ix].mean(1),[.025,.975],axis=0).T.tolist(),
            scope='selection100 systems, one source seed and one reader seed; descriptive post-budget diagnostic')
        assert sha(original_cp)==ANCESTOR_SHA and sha(old_reader/'final.pt')==READER40_SHA
        assert sha(helper.ROOT/'PROTOCOL.json')==protocol_sha and panel.digest(model)==source_digest
        write(out/'PROBE.json',probes)
        summary=dict(status='COMPLETE',rows=rows,decision=decision,paired=paired,
            fixed_donor_gap={n:(np.asarray(rows[n]['metrics']['wrong'])-np.asarray(rows[n]['metrics']['matched'])).tolist() for n in rows},
            null_reader=artifact(null_dir/'final.pt'),null_mean=arrays['null'].mean(0).tolist(),
            source_encoder_digests=sources_before,source_weights_unchanged_during_reader_and_evaluation=True,
            plan_sha256=sha(out/'PLAN.json'),arrays_sha256=sha(out/'SELECTION_ERRORS.npz'),
            manifest_sha256=sha(out/'SELECTION_MANIFEST.json'),probe_sha256=sha(out/'PROBE.json'),
            selection_ids_only=True,report_evaluation=False,new_sealed_test=False,automatic_followup=False,
            no_controls_trained=True,total_seconds=time.monotonic()-started,stages=stages)
        write(out/'SUMMARY.json',summary);write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json')))
        write(out/'EXIT.json',dict(exit_code=0,time=time.time()));print(json.dumps(summary),flush=True)
    except BaseException:
        write(out/'FAILED.json',dict(traceback=traceback.format_exc(),stage=active_stage,time=time.time()))
        write(out/'EXIT.json',dict(exit_code=1,time=time.time()));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--formal-root',default='./runs/dclean_fixed40k_formal')
    p.add_argument('--original20k-root',default='./runs/dclean_formal')
    p.add_argument('--helper',default='shared/dclean_external.py')
    p.add_argument('--panel',default='shared/dclean_panel.py')
    p.add_argument('--official-root',default=str(HERE.parent/'vendor/DALI'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args();assert a.gpu in range(4);run(a)
