#!/usr/bin/env python3
"""Finite frozen-CoDA encoder gate: shared train-only teacher, plain vs Cross.

Separate phases allow two authorized idle GPUs after shared preparation.
No automatic seed expansion; no test/OOD dataset is constructed.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

SOURCE_SHA = '69e21e714a8ee221cd2c59e0473aa77246f7a14195e36c324c6ef25b4b178be8'
TEACHER_SHA = '5365820fc1ab22b5ae68bccba67136922fc276cc7c9775cd6e3c84cfe39a37d8'
COMPONENT_SHA = 'acb8bec540220913ed7559c1e16af6e6536fa63e0d66f5f1e0d4d26e6900db3b'
DATA_API_SHA = '934fd3c75a15eb77d153685edbe94e7452f1b5cb72f5e2b6cb91a0e442c91d77'
POSTRUN_SHA = 'a448cd0b96fbda128bfab22aa657f51191e90dd2eebb81e677184807db62aaf1'
SEED = 1234
BUDGETS = (0, 5, 10, 50)
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.writing')
    q.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n'); q.replace(p)


def jsha(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def deadline(*_): raise TimeoutError('Finite phase wall-time reached; no implicit resume or extension')


def set_seed(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def bootstrap(a):
    root = a.code_root.resolve()
    sys.path[:0] = [str(root / 'geps_deps'), str(root)]
    import numpy as np
    import torch
    import coda_burgers_components as c
    import geps_burgers_pilot as base
    import geps_formal_three_seed as g
    import coda_frozen_encoder as impl
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert sha(c.__file__) == COMPONENT_SHA and sha(base.__file__) == DATA_API_SHA
    assert sha(g.__file__) == '82535b32b57b5779d12379e67a4d79b5267ed08b1b0db8bf5e9583ad01ce371a'
    assert sha(root / 'coda_postrun_mechanisms.py') == POSTRUN_SHA
    expected = {'utils.py':'c1a6daf23aaf3b01de5caadeac90cc6c36111b62c009f08d9debb5534182e983',
                'network.py':'e122604dc736011988fad1771db5b3c15cedf6d59da4acb6b365b756f8abd2dd',
                'ode_model.py':'9b855da11ae5b423ec3431eb2bf3b60f34439674d80e797fec95488b546a6957'}
    assert {n:sha(c.SOURCE/n) for n in expected} == expected
    src = root / 'coda_formal_three_seed/seed1234'
    checkpoint = src / 'latest.pt'; assert sha(checkpoint) == SOURCE_SHA
    assert read(src / 'EXIT.json')['exit_code'] == 0
    assert read(src / 'COMPLETE.json')['summary_sha256'] == sha(src / 'SUMMARY.json')
    # Source summary is read only for checkpoint provenance, not report metrics.
    assert read(src / 'SUMMARY.json')['checkpoint_sha256'] == SOURCE_SHA
    cfg = read(root / 'geps_pilot_retry1/CONFIG.json')
    args = argparse.Namespace(seed=SEED, nod_source=Path(cfg['nod_source']), data_root=Path(cfg['data_root']))
    assert sha(args.nod_source / 'ngs/utils.py') == g.LOADER_SHA
    data, meta = base.load_released_data(args)  # implementation reads train/eval only; cache_mode=none
    assert meta['train']['data_sha256'] == read(src / 'DATA_MANIFEST.json')['train']['data_sha256']
    assert tuple(data['train']['curves'].shape) == (360, 1, 401, 101)
    assert tuple(data['eval']['curves'].shape) == (45, 1, 401, 101)
    assert torch.equal(data['train']['t'], data['eval']['t']) and len(data['train']['t']) == 101
    set_seed()
    source = c.build(9, 'cuda:0')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['updates'] == 5000 and ck['config']['seed'] == SEED
    source.load_state_dict(ck['model'], strict=True); source.eval()
    for p in source.parameters(): p.requires_grad_(False)
    original_digest = impl.model_sha(source)
    torch.cuda.synchronize()
    gpu_pids = subprocess.check_output(['nvidia-smi','--id='+str(a.gpu),
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    # The entry check proved this GPU empty under the cooperative lock before
    # this process created an actual CUDA context. A stable singleton afterward
    # identifies the job even when nvidia-smi exposes a host PID in a container.
    nspid = [str(os.getpid())]
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('NSpid:'): nspid += line.split()[1:]
    pid_basis = 'proc_nspid' if set(gpu_pids)&set(nspid) else 'empty_locked_entry_then_single_cuda_process'
    gpu_pid_known = len(gpu_pids)==1
    context = dict(c=c, base=base, impl=impl, source=source, data=data, meta=meta,
                   gpu_pids=gpu_pids if gpu_pid_known else [],gpu_pid_mapping_known=gpu_pid_known,
                   gpu_pid_mapping_basis=pid_basis if gpu_pid_known else 'unknown',container_nspid=nspid,
                   original_digest=original_digest, checkpoint=checkpoint, code_root=root)
    return context


def data_pairs(bank):
    import torch
    pairs = []
    for env in range(9):
        ids = torch.where(bank['envs'] == env)[0].tolist(); assert len(ids) == 5
        assert sorted(int(bank['cases'][i]) for i in ids) == list(range(40, 45))
        pairs += [(i, ids[(j + 1) % 5]) for j, i in enumerate(ids)]
    assert len(pairs) == 45 and all(i != j for i, j in pairs)
    return pairs


def manifest(ctx):
    bank = ctx['data']['eval']; base = ctx['base']
    return [dict(recipient=i, donor=j, env=int(bank['envs'][i]),
                 pred_case=int(bank['cases'][i]), cond_case=int(bank['cases'][j]),
                 query_sha256=base.tensor_digest(bank['curves'][i]),
                 support_sha256=base.tensor_digest(bank['curves'][j])) for i, j in data_pairs(bank)]


def frozen_audit(ctx):
    assert ctx['impl'].model_sha(ctx['source']) == ctx['original_digest']
    assert sha(ctx['checkpoint']) == SOURCE_SHA
    assert all(p.grad is None and not p.requires_grad for p in ctx['source'].parameters())
    return dict(source_sha256=SOURCE_SHA, source_parameter_digest=ctx['original_digest'], unchanged=True)


def new_encoder(ctx, shared):
    import torch
    ck = torch.load(shared / 'shared_encoder.pt', map_location='cpu', weights_only=False)
    assert ck['updates'] == 500 and ck['source_sha256'] == SOURCE_SHA
    assert ck['teacher_sha256'] == TEACHER_SHA
    st = ck['stats']; impl = ctx['impl']
    enc = impl.HistoryEncoder(st['obs_mean'], st['obs_std'], st['code_mean'], st['code_std']).cuda()
    enc.load_state_dict(ck['model'], strict=True)
    return enc, ck


def evaluate(ctx, out, encoder=None):
    import numpy as np
    import torch
    c, impl = ctx['c'], ctx['impl']
    bank = ctx['data']['eval']; pairs = data_pairs(bank); mani = manifest(ctx)
    times = bank['t'].cuda(); records = []; time_rows = []
    owned = subprocess.check_output(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
       '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
    isolation_checks = [dict(batch_start=0,pids=owned)]
    isolated = ctx['gpu_pid_mapping_known'] and set(owned)==set(ctx['gpu_pids'])
    gpu_uuid = subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid','--format=csv,noheader,nounits'],text=True).strip()
    source_hash = impl.model_sha(ctx['source'])
    for start in range(0, 45, 8):
        block = pairs[start:start + 8]; r = [x[0] for x in block]; d = [x[1] for x in block]
        support = bank['curves'][d].cuda(); truth = bank['curves'][r].cuda()
        decoder = impl.FrozenDecoder(c, ctx['source'], len(block))
        enc_seconds = 0.
        with torch.no_grad():
            if encoder is None:
                initial = support.new_zeros((len(block), 2))
            else:
                encoder.eval(); encoder(support)  # fixed warmup, not an optimizer step
                torch.cuda.synchronize(); t0 = time.monotonic(); initial = encoder(support)
                torch.cuda.synchronize(); enc_seconds = time.monotonic() - t0
            decoder.predict(truth[..., 0], initial, times)  # full-path warmup on initial observation only
            torch.cuda.synchronize()
        z = torch.nn.Parameter(initial.detach().clone())
        optimizer = torch.optim.Adam([z], lr=.001)
        adapt_seconds = 0.
        def score(step):
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.monotonic()
                pred = decoder.predict(truth[..., 0], z, times)
                torch.cuda.synchronize(); pred_time = time.monotonic() - t0
                assert torch.isfinite(pred).all()
                assert torch.equal(pred[..., 0], truth[..., 0])
                err = (pred - truth).square().mean((1, 2)).cpu().numpy()
            for j, row in enumerate(mani[start:start + len(block)]):
                rec = dict(row, support_steps=step, mse_all101=float(err[j].mean()),
                           mse_future100=float(err[j, 1:].mean()), code=z[j].detach().cpu().tolist())
                rec.update({'h'+str(h): float(err[j, h]) for h in (1, 5, 50, 100)})
                records.append(rec)
            time_rows.append(dict(batch_start=start, batch_size=len(block), support_steps=step,
                encoder_seconds=enc_seconds, code_optimizer_seconds=adapt_seconds, forecast_seconds=pred_time,
                total_seconds=enc_seconds+adapt_seconds+pred_time))
        score(0)
        for step in range(50):
            # Identical per-batch/update teacher reset mask across zero/plain/Cross.
            np.random.seed(917000 + start * 100 + step)
            torch.cuda.synchronize(); t0 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pred = decoder.training_reconstruction(support, z, times, .95 * (.95 ** (step // 30)))
            loss = (pred - support).square().mean(); assert torch.isfinite(loss)
            loss.backward(); assert z.grad is not None and torch.isfinite(z.grad).all()
            optimizer.step(); torch.cuda.synchronize(); adapt_seconds += time.monotonic() - t0
            if step + 1 in BUDGETS: score(step + 1)
        decoder.audit()
        owned_now=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        isolation_checks.append(dict(batch_start=start,pids=owned_now))
        isolated=isolated and set(owned_now)==set(ctx['gpu_pids'])
        write(out / 'EVAL_PARTIAL.json', dict(completed_pairs=start + len(block), records=records, times=time_rows))
    assert impl.model_sha(ctx['source']) == source_hash
    result = {}
    for step in BUDGETS:
        rows = [x for x in records if x['support_steps'] == step]
        assert len(rows) == 45
        err = [x['mse_all101'] for x in rows]
        ts = [x for x in time_rows if x['support_steps'] == step]
        result[str(step)] = dict(n_pairs=45, native_batch8_mse=float(np.mean([np.mean(err[i:i+8]) for i in range(0,45,8)])),
            trajectory_mean_mse=float(np.mean(err)), future100=float(np.mean([x['mse_future100'] for x in rows])),
            horizons={str(h):float(np.mean([x['h'+str(h)] for x in rows])) for h in (1,5,50,100)},
            per_system={str(e):float(np.mean([x['mse_all101'] for x in rows if x['env']==e])) for e in range(9)},
            encoder_seconds=sum(x['encoder_seconds'] for x in ts),
            code_optimizer_seconds=sum(x['code_optimizer_seconds'] for x in ts),
            forecast_seconds=sum(x['forecast_seconds'] for x in ts), total_seconds=sum(x['total_seconds'] for x in ts))
    write(out / 'EVAL_PAIRS.json', records); write(out / 'EVAL_TIMES.json', time_rows)
    summary = dict(status='COMPLETE', budgets=result, manifest_sha256=jsha(mani),
        timing_foreign_processes_absent_at_all_checks=isolated,gpu_uuid=gpu_uuid,
        timing_isolation_checks=isolation_checks,gpu_pid_mapping_basis=ctx['gpu_pid_mapping_basis'],
        gpu_pid_mapping_known=ctx['gpu_pid_mapping_known'],source=frozen_audit(ctx),
        query_future_used_for_optimization=False, query_future_passed_to_forecaster=False,
        support_frames=101, native_batch_size=8, test_read=False, ood_read=False,
        inference_code_optimizer='Adam.001,50 maximum,teacher.95 decays after30; no code regularizer, same as existing CoDA')
    write(out / 'EVAL_SUMMARY.json', summary)
    return summary


def prepare(ctx, a):
    import numpy as np
    import torch
    out = a.output; impl = ctx['impl']; bank = ctx['data']['train']
    teacher_root = ctx['code_root'] / 'coda_postrun_mechanisms_three_seed/seed1234'
    teacher_path = teacher_root / 'PROBE_CODES.npz'; assert sha(teacher_path) == TEACHER_SHA
    assert read(teacher_root / 'EXIT.json')['exit_code'] == 0
    assert read(teacher_root / 'COMPLETE.json')['summary_sha256'] == sha(teacher_root / 'SUMMARY.json')
    ts = read(teacher_root / 'SUMMARY.json')
    assert ts['checkpoint_sha256'] == SOURCE_SHA and ts['n_training_histories'] == 360
    with np.load(teacher_path, allow_pickle=False) as z:
        teacher = torch.from_numpy(z['train'].copy()).float()
        cases = z['train_case'].copy()  # intentionally no id/id_nu/train_nu arrays accessed
    assert tuple(teacher.shape) == (360, 2) and torch.isfinite(teacher).all()
    np.testing.assert_array_equal(cases, bank['cases'].numpy())
    stats = dict(obs_mean=float(bank['curves'].mean()), obs_std=float(bank['curves'].std(unbiased=False)),
                 code_mean=teacher.mean(0).tolist(), code_std=teacher.std(0, unbiased=False).tolist())
    assert min(stats['code_std']) > 1e-8 and stats['obs_std'] > 1e-8
    set_seed()
    enc = impl.HistoryEncoder(**stats).cuda(); optimizer = torch.optim.Adam(enc.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(SEED)
    per_env = [torch.where(bank['envs'] == e)[0] for e in range(9)]
    assert all(len(x)==40 for x in per_env)
    orders = [torch.randperm(40, generator=rng) for _ in range(9)]; cursor = 0
    observations = bank['curves'].cuda(); target = teacher.cuda(); start = time.monotonic()
    initial = impl.model_sha(enc)
    for step in range(1, 501):
        if cursor == 40:
            orders = [torch.randperm(40, generator=rng) for _ in range(9)]; cursor = 0
        ix = torch.cat([ids[order[cursor:cursor+4]] for ids, order in zip(per_env, orders)]); cursor += 4
        optimizer.zero_grad(set_to_none=True)
        loss = ((enc(observations[ix]) - target[ix]) / enc.code_std).square().mean()
        assert torch.isfinite(loss); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)
        optimizer.step()
        if step % 25 == 0:
            with (out / 'PRETRAIN.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=step, normalized_teacher_mse=float(loss.detach()), seconds=time.monotonic()-start))+'\n')
    with torch.no_grad(): teacher_mse=float((enc(observations)-target).square().mean())
    state=dict(model=enc.state_dict(), optimizer=optimizer.state_dict(), updates=500, stats=stats,
               source_sha256=SOURCE_SHA, teacher_sha256=TEACHER_SHA, initial_encoder_digest=initial,
               data_sha256=ctx['meta']['train']['data_sha256'], rng=rng.get_state(),
               numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(state,out / 'shared_encoder.pt')
    write(out / 'PRETRAIN_COMPLETE.json',dict(updates=500,batch_size=36,histories_seen=18000,
          distinct_train_histories=360,teacher_mse=teacher_mse,seconds=time.monotonic()-start,
          checkpoint_sha256=sha(out/'shared_encoder.pt'),stats=stats,
          teacher_cache_sha256=TEACHER_SHA,teacher_cache_read_arrays=['train','train_case'],
          teacher_existing_cost=dict(histories=360,code_steps_each=50,total_independent_code_steps=18000,
             grouped_optimizer_steps=2250,recorded_adapt_seconds=ts['training_history_code_adapt_seconds']),
          source_optimizer_updates=0,train_only_normalization=True))
    write(out / 'SUMMARY.json',dict(status='COMPLETE',phase='shared500_train_only',
          shared_checkpoint_sha256=sha(out/'shared_encoder.pt'),source=frozen_audit(ctx)))


def train(ctx, a):
    import numpy as np
    import torch
    out=a.output; shared=a.shared.resolve(); impl=ctx['impl']; bank=ctx['data']['train']
    assert read(shared/'COMPLETE.json')['summary_sha256']==sha(shared/'SUMMARY.json')
    assert read(shared/'EXIT.json')['exit_code']==0
    assert read(shared/'SUMMARY.json')['shared_checkpoint_sha256']==sha(shared/'shared_encoder.pt')
    enc,ck=new_encoder(ctx,shared); assert ck['data_sha256']==ctx['meta']['train']['data_sha256']
    initial=impl.model_sha(enc); enc.train(); opt=torch.optim.Adam(enc.parameters(),lr=1e-4)
    observations=bank['curves'].cuda(); times=bank['t'].cuda()
    rng=torch.Generator().manual_seed(SEED+101); per_env=[torch.where(bank['envs']==e)[0] for e in range(9)]
    orders=[torch.randperm(40,generator=rng) for _ in range(9)]; cursor=0
    decoder=impl.FrozenDecoder(ctx['c'],ctx['source'],18); start=time.monotonic(); sample_hash=hashlib.sha256()
    hyper_constant=float(1e-6*torch.norm(ctx['source'].derivative.net_hyper.weight,dim=1).sum())
    cross=a.arm=='sprii_cross'; last=None
    for step in range(1,1001):
        if cursor==40:
            orders=[torch.randperm(40,generator=rng) for _ in range(9)]; cursor=0
        ia=torch.tensor([int(ids[order[cursor]]) for ids,order in zip(per_env,orders)])
        ib=torch.tensor([int(ids[order[cursor+1]]) for ids,order in zip(per_env,orders)]);cursor+=2
        assert torch.all(ia!=ib) and torch.equal(bank['envs'][ia],bank['envs'][ib])
        ix=torch.cat([ia,ib]); sample_hash.update(ix.numpy().tobytes()); truth=observations[ix]
        eps=.99*(.99**((step-1)//30)); opt.zero_grad(set_to_none=True)
        z=enc(truth);np.random.seed(510000+step)
        pred=decoder.training_reconstruction(truth,z,times,eps)
        self_mse=(pred-truth).square().mean(); reg=1e-4*z.square().sum()
        loss=self_mse+reg;assert torch.isfinite(loss);loss.backward()
        cross_mse=None
        if cross:
            # Recompute only the deterministic encoder graph; self and Cross share
            # exact 18 histories/decoder/TF mask. Sequential backward bounds memory.
            z_cross=enc(truth); swapped=torch.cat([z_cross[9:],z_cross[:9]])
            np.random.seed(510000+step)
            pred=decoder.training_reconstruction(truth,swapped,times,eps)
            cross_loss=(pred-truth).square().mean();assert torch.isfinite(cross_loss)
            cross_loss.backward();cross_mse=float(cross_loss.detach())
        assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)
        norm=float(torch.nn.utils.clip_grad_norm_(enc.parameters(),1.));opt.step();torch.cuda.synchronize()
        last=dict(update=step,self_mse=float(self_mse.detach()),cross_mse=cross_mse,
                  code_l2=float(reg.detach()),fixed_hyper_regularizer=hyper_constant,
                  gradient_norm_before_clip=norm,epsilon=eps,seconds=time.monotonic()-start)
        if step==1 or step%10==0:
            with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
            print(json.dumps(last),flush=True)
        if step==50:
            projected=(time.monotonic()-start)*20
            write(out/'THROUGHPUT50.json',dict(updates=50,projected_seconds_for1000=projected,
                   max_train_seconds=1800,continue_fixed1000=projected<=1800,
                   decoder_trajectory_forecasts=50*18*(2 if cross else 1),
                   no_claim_of_equal_training_compute=True))
            if projected>1800:raise TimeoutError('50-update gate projects beyond fixed30-minute arm cap; preserve partial only')
        if step%50==0:
            decoder.audit()
            torch.save(dict(model=enc.state_dict(),optimizer=opt.state_dict(),updates=step,stats=ck['stats'],
                source_sha256=SOURCE_SHA,teacher_sha256=TEACHER_SHA,shared_checkpoint_sha256=sha(shared/'shared_encoder.pt'),
                arm=a.arm,rng=rng.get_state(),sampler_orders=orders,sampler_cursor=cursor,
                data_sha256=ctx['meta']['train']['data_sha256']),out/'latest.pt.writing')
            (out/'latest.pt.writing').replace(out/'latest.pt')
    decoder.audit();assert impl.model_sha(enc)!=initial
    write(out/'TRAIN_COMPLETE.json',dict(updates=1000,last=last,initial_encoder_digest=initial,
          shared_checkpoint_sha256=sha(shared/'shared_encoder.pt'),checkpoint_sha256=sha(out/'latest.pt'),
          paired_history_stream_sha256=sample_hash.hexdigest(),histories_per_update=18,total_history_presentations=18000,
          decoder_trajectory_forecasts=18000*(2 if cross else 1),
          grouped_rk4_rhs_evaluations=400000*(2 if cross else 1),
          compute_note='Exact100 intervals x4 RK4 rhs calls per grouped forecast; not measured FLOPs',
          lambda_cross=int(cross),lambda_align=0,
          source_optimizer_updates=0,seconds=time.monotonic()-start,source=frozen_audit(ctx)))
    with torch.no_grad():
        final_train_codes=torch.cat([enc(observations[i:i+36]) for i in range(0,360,36)]).cpu()
    write(out/'TRAIN_CODE_DIAGNOSTICS.json',dict(mean=final_train_codes.mean(0).tolist(),
          std=final_train_codes.std(0,unbiased=False).tolist(),finite=bool(torch.isfinite(final_train_codes).all()),
          is_probe=False,train_histories_only=True))
    write(out/'SUMMARY.json',dict(status='COMPLETE',arm=a.arm,updates=1000,
          checkpoint_sha256=sha(out/'latest.pt'),source=frozen_audit(ctx)))


def summarize(a):
    out=a.output; paths={'zero':a.shared,'plain':a.plain,'sprii_cross':a.cross}; docs={}
    for arm,p in paths.items():
        assert read(p/'EXIT.json')['exit_code']==0
        assert read(p/'COMPLETE.json')['summary_sha256']==sha(p/'SUMMARY.json')
        docs[arm]=read(p/'SUMMARY.json')
    tplain=read(a.plain/'TRAIN_COMPLETE.json');tcross=read(a.cross/'TRAIN_COMPLETE.json')
    for k in ['initial_encoder_digest','shared_checkpoint_sha256','paired_history_stream_sha256','updates']:
        assert tplain[k]==tcross[k]
    ev={k:read(out/'evaluations'/k/'EVAL_SUMMARY.json') for k in docs}
    assert len({v['manifest_sha256'] for v in ev.values()})==1
    zero=ev['zero']['budgets']['50'];plain=ev['plain']['budgets']['0'];cross=ev['sprii_cross']['budgets']['0']
    reduction=1-cross['native_batch8_mse']/plain['native_batch8_mse']
    envwins=sum(cross['per_system'][str(e)]<plain['per_system'][str(e)] for e in range(9))
    c1=reduction>=.05 and envwins>=6
    speed=[]
    for budget in (0,5,10):
        row=ev['sprii_cross']['budgets'][str(budget)]
        qualifies=(row['native_batch8_mse']<=1.02*zero['native_batch8_mse'] and row['total_seconds']<zero['total_seconds']
                    and ev['sprii_cross']['timing_foreign_processes_absent_at_all_checks'] and ev['zero']['timing_foreign_processes_absent_at_all_checks'] and ev['sprii_cross']['gpu_uuid']==ev['zero']['gpu_uuid'])
        speed.append(dict(support_steps=budget,qualifies=qualifies,mse_ratio_to_zero50=row['native_batch8_mse']/zero['native_batch8_mse'],
                          time_ratio_to_zero50=row['total_seconds']/zero['total_seconds']))
    quality50=1-ev['sprii_cross']['budgets']['50']['native_batch8_mse']/zero['native_batch8_mse']
    c2=any(r['qualifies'] for r in speed) or quality50>=.05
    result=dict(status='COMPLETE',fixed_single_seed=SEED,gate_pass=bool(c1 and c2),
        plain_relative_improvement_code0=reduction,system_wins_code0=envwins,criterion1=c1,criterion2=c2,
        support_speed_candidates=speed,quality50_relative_improvement=quality50,
        automatic_confirmation_launched=False,statistical_significance_claim=False,
        method='SPRII-Cross only, lambda_x1 lambda_p0; frozen adapted CoDA decoder plus newly trained encoder',
        not_original_nod_sprii_result=True,not_untouched_coda_default=True,
        source_training_records=docs,development=ev,source_sha256=SOURCE_SHA,teacher_sha256=TEACHER_SHA,
        budget=dict(shared_encoder_pretrain=500,additional_encoder_updates_each=1000,source_updates=0,
                    preexisting_teacher_histories=360,preexisting_code_steps_each=50),
        test_read=False,ood_read=False)
    write(out/'SUMMARY.json',result)


def evaluate_all(ctx,a):
    import torch
    assert all([a.shared,a.plain,a.cross])
    models={}
    for arm,path in [('plain',a.plain),('sprii_cross',a.cross)]:
        assert read(path/'EXIT.json')['exit_code']==0
        assert read(path/'COMPLETE.json')['summary_sha256']==sha(path/'SUMMARY.json')
        assert read(path/'SUMMARY.json')['checkpoint_sha256']==sha(path/'latest.pt')
        state=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        assert state['updates']==1000 and state['arm']==arm and state['source_sha256']==SOURCE_SHA
        assert state['shared_checkpoint_sha256']==sha(a.shared/'shared_encoder.pt')
        encoder,_=new_encoder(ctx,a.shared);encoder.load_state_dict(state['model'],strict=True)
        models[arm]=encoder.eval()
    for arm,encoder in [('zero',None),('plain',models['plain']),('sprii_cross',models['sprii_cross'])]:
        dest=a.output/'evaluations'/arm;dest.mkdir(parents=True,exist_ok=False)
        signal.setitimer(signal.ITIMER_REAL,1800)
        evaluate(ctx,dest,encoder)
    summarize(a)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','evaluate'])
    p.add_argument('--code-root',type=Path,default=Path('benchmarks/baseline_adapters'))
    p.add_argument('--gpu',type=int);p.add_argument('--lock-dir',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shared',type=Path)
    p.add_argument('--arm',choices=['plain','sprii_cross']);p.add_argument('--plain',type=Path);p.add_argument('--cross',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    locks=[];os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    write(a.output/'RUN.json',dict(pid=os.getpid(),phase=a.phase,time=time.time(),runner_sha256=sha(__file__)))
    signal.signal(signal.SIGALRM,deadline)
    try:
        assert a.gpu in (0,1,2,3) and a.lock_dir is not None
        a.lock_dir.mkdir(parents=True,exist_ok=True)
        lock=(a.lock_dir/f'gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        signal.setitimer(signal.ITIMER_REAL,1800)
        ctx=bootstrap(a)
        write(a.output/'DATA_MANIFEST.json',ctx['meta']);write(a.output/'DEVELOPMENT_MANIFEST.json',manifest(ctx))
        write(a.output/'CONFIG.json',dict(seed=SEED,source_sha256=SOURCE_SHA,teacher_sha256=TEACHER_SHA,
            source_updates=0,decoder='CoDA original width64 code2 factor1 RK4, frozen',
            encoder='Conv1-16-32-64 stride2 k5/3/3 SiLU pool4x4 Linear1024-64-2',
            shared_pretrain_updates=500,each_arm_updates=1000,lambda_cross=1 if a.arm=='sprii_cross' else 0,lambda_align=0,
            mse_scale=1,train_cases=list(range(40)),development_cases=list(range(40,45)),test_read=False,ood_read=False,
            module_sha256=sha(HERE/'coda_frozen_encoder.py'),runner_sha256=sha(__file__),
            selection='fixed final update1000; no epoch selection',max_train_seconds_each=1800,
            common_data_budget=True,equal_training_flops=False))
        write(a.output/'SOURCE_SMOKE.json',ctx['impl'].smoke(ctx['c'],ctx['source'],'cuda:0'))
        set_seed()
        signal.setitimer(signal.ITIMER_REAL,1800)
        if a.phase=='prepare':prepare(ctx,a)
        elif a.phase=='train':
            assert a.arm and a.shared;train(ctx,a)
        else:evaluate_all(ctx,a)
        frozen_audit(ctx)
        write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json')))
        write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
    except BaseException:
        write(a.output/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
