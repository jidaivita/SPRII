#!/usr/bin/env python3
"""Finite CoDA component adaptation gate; development cases only, not a ranking."""
import argparse, fcntl, hashlib, importlib.util, json, os, random, signal, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'third_party/coda_original'
EXPECTED={'utils.py':'c1a6daf23aaf3b01de5caadeac90cc6c36111b62c009f08d9debb5534182e983',
          'network.py':'e122604dc736011988fad1771db5b3c15cedf6d59da4acb6b365b756f8abd2dd',
          'ode_model.py':'9b855da11ae5b423ec3431eb2bf3b60f34439674d80e797fec95488b546a6957'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def alarm(*_):raise TimeoutError('CoDA finite development gate phase exceeded 7200 seconds; preserve partial files.')
def main(a):
    locks=[]
    for directory in [HERE,HERE/'gpu_locks']:
        directory.mkdir(exist_ok=True);f=(directory/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    import subprocess
    assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    assert not a.output.exists();a.output.mkdir(parents=True)
    write(a.output/'RUN.json',dict(pid=os.getpid(),time=time.time(),runner_sha256=sha(__file__)))
    try:
        import numpy as np
        import torch
        import coda_burgers_components as c
        import geps_burgers_pilot as data_api
        torch.set_num_threads(1)
        assert sha(data_api.__file__)=='934fd3c75a15eb77d153685edbe94e7452f1b5cb72f5e2b6cb91a0e442c91d77'
        assert {n:sha(SOURCE/n) for n in EXPECTED}==EXPECTED
        smoke=c.smoke();write(a.output/'SMOKE.json',smoke)
        random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
        old=read(HERE/'geps_pilot_retry1/CONFIG.json')
        data_args=argparse.Namespace(nod_source=Path(old['nod_source']),data_root=Path(old['data_root']),seed=1234)
        data,manifest=data_api.load_released_data(data_args);write(a.output/'DATA_MANIFEST.json',manifest)
        bank=data['train'];envs=torch.unique(bank['envs']).tolist();assert envs==list(range(9))
        cfg=dict(seed=1234,updates=1000,model='CoDA published grouped 2D convolution on singleton spatial axis',
                 official_commit='17b73521394f2a5986e5418c32ad2965c97cd8c0',source_sha256=EXPECTED,
                 component_sha256=sha(c.__file__),runner_sha256=sha(__file__),state_c=1,hidden_c=64,code_c=2,
                 factor=1.0,method='rk4',lr=.001,l12m=1e-6,l2c=1e-4,grad_clip=1.,
                 teacher_forcing_initial=.99,teacher_forcing_multiplier=.99,teacher_forcing_every_updates=30,
                 observations_per_update=9,batch_per_environment=1,time_normalization=100,
                 adaptation_steps=50,adaptation_lr=.001,adaptation_initial_codes='zero',adaptation_teacher_forcing_initial=.95,
                 protocol='Development gate only; train cases0..39; development40..44; no test/OOD',
                 author_default_Burgers_exists=False,
                 adaptation_notes=['Original group Conv2d/HyperEnvNet/Forecaster numeric definitions executed verbatim.',
                    'One singleton spatial axis maps the 401-point 1D grid into the released convolution interface.',
                    'Factor1 uses the normalized Burgers time unit; gray-specific 5e-4 is not copied across physical units.',
                    'Finite1000-update gate, not the original120000-epoch protocol or convergence claim.',
                    '50-step code adaptation is a common finite observation-conditioned budget, not official convergence.'])
        write(a.output/'CONFIG.json',cfg)
        model=c.build(9,'cuda:0');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        per_env=[torch.where(bank['envs']==e)[0] for e in envs];assert all(len(x)==40 for x in per_env)
        generator=torch.Generator().manual_seed(1234);orders=[torch.randperm(40,generator=generator) for _ in envs];cursor=0
        times=bank['t'].cuda();epsilon=.99;start=time.monotonic();losses=[]
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,7200)
        for step in range(1,1001):
            if cursor==40:orders=[torch.randperm(40,generator=generator) for _ in envs];cursor=0
            ix=torch.tensor([int(ids[order[cursor]]) for ids,order in zip(per_env,orders)]);cursor+=1
            truth=bank['curves'][ix,0].unsqueeze(0).cuda();optimizer.zero_grad(set_to_none=True)
            output=c.forecast(model,truth,times,epsilon);mse=(output-truth).square().mean();reg=c.regularizer(model);loss=mse+reg
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(mse.detach()))
            if step%30==0:epsilon*=.99
            if step==1 or step%10==0:
                row=dict(update=step,mse=losses[-1],regularizer=float(reg.detach()),epsilon=epsilon,seconds=time.monotonic()-start)
                with (a.output/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step%250==0:
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=step,config=cfg,
                           sampler_generator=generator.get_state(),sampler_orders=orders,sampler_cursor=cursor,
                           numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                tmp=a.output/'latest.pt.writing';torch.save(state,tmp);tmp.replace(a.output/'latest.pt')
        signal.setitimer(signal.ITIMER_REAL,0)
        write(a.output/'TRAIN_COMPLETE.json',dict(status='COMPLETE',updates=1000,seconds=time.monotonic()-start,
              first20_mse=float(np.mean(losses[:20])),last20_mse=float(np.mean(losses[-20:])),checkpoint_sha256=sha(a.output/'latest.pt')))
        bank=data['eval'];pairs=[]
        for e in envs:
            ids=torch.where(bank['envs']==e)[0].tolist();assert len(ids)==5
            pairs += [(i,ids[(j+1)%5]) for j,i in enumerate(ids)]
        assert len(pairs)==45;records=[];codes=[];source=c.shared_digest(model)
        signal.setitimer(signal.ITIMER_REAL,7200);eval_start=time.monotonic()
        for start_pair in range(0,45,3):
            block=pairs[start_pair:start_pair+3];r=[x[0] for x in block];d=[x[1] for x in block]
            support=bank['curves'][d,0].unsqueeze(0).cuda();query=bank['curves'][r,0].unsqueeze(0).cuda()
            target=c.adapted(model,len(block));opt=torch.optim.Adam([target.derivative.codes],lr=.001);before=c.shared_digest(target)
            with torch.no_grad():initial=float((c.forecast(target,support,times)-support).square().mean())
            begin=time.monotonic()
            for adapt_step in range(50):
                opt.zero_grad(set_to_none=True);loss=(c.forecast(target,support,times,epsilon=.95*(.95**(adapt_step//30)))-support).square().mean()
                assert torch.isfinite(loss);loss.backward();assert torch.isfinite(target.derivative.codes.grad).all();opt.step()
            with torch.no_grad():
                final=float((c.forecast(target,support,times)-support).square().mean())
                x=torch.zeros_like(query);x[...,0]=query[...,0];prediction=c.forecast(target,x,times)
                assert torch.isfinite(prediction).all() and torch.equal(prediction[...,0],query[...,0])
                assert c.shared_digest(target)==before==source and c.shared_digest(model)==source
                error=(prediction-query).square()
                for j,(recipient,donor) in enumerate(block):
                    records.append(dict(recipient=recipient,donor=donor,nu_id=int(bank['envs'][recipient]),
                        pred_case=int(bank['cases'][recipient]),cond_case=int(bank['cases'][donor]),
                        mse_all101=float(error[0,j].mean()),mse_future100=float(error[0,j,...,1:].mean()),
                        support_mse_initial=initial,support_mse_final=final,adapt_seconds_batch=time.monotonic()-begin))
                    codes.append(target.derivative.codes[j].cpu().tolist())
            write(a.output/'DEVELOPMENT_PARTIAL.json',dict(completed_pairs=len(records),records=records,codes=codes))
        signal.setitimer(signal.ITIMER_REAL,0)
        result=dict(status='COMPLETE',training_updates=1000,development_pairs=45,code_steps_per_support=50,
                    mse_all101=float(np.mean([r['mse_all101'] for r in records])),
                    mse_future100=float(np.mean([r['mse_future100'] for r in records])),
                    records=records,codes=codes,eval_seconds=time.monotonic()-eval_start,checkpoint_sha256=sha(a.output/'latest.pt'),
                    test_read=False,ood_read=False,finite_budget_gate=True,not_paper_ranking=True)
        write(a.output/'SUMMARY.json',result);write(a.output/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(a.output/'SUMMARY.json')))
        write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
    except BaseException:
        write(a.output/'FAILED.json',dict(traceback=traceback.format_exc()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
