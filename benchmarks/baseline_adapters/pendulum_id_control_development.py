"""Prospective common ID control bank; frozen source models and real rewards."""
import argparse, fcntl, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE_SHA='ac458b321d0d3c78c4ab6d7bff193568706d633f680891fc133a32cf9166d9ad'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(state):
    h=hashlib.sha256()
    for name,x in sorted(state.items()):h.update(name.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main(a):
    assert not a.output.exists();locks=[]
    for d in [HERE,HERE/'gpu_locks']:
        f=(d/f'gpu{a.gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert sha(HERE/'cadm_pendulum_online.py')==BASE_SHA
    spec=importlib.util.spec_from_file_location('unchanged_pendulum_for_ID',HERE/'cadm_pendulum_online.py');b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    a.output.mkdir(parents=True)
    write(a.output/'RUN.json',dict(pid=os.getpid(),gpu=a.gpu,time=time.time(),runner_sha256=sha(__file__)))
    try:
        s=read(a.source/'SUMMARY.json');assert read(a.source/'EXIT_run.json')['exit_code']==0 and s['status']=='COMPLETE'
        assert read(a.source/'COMPLETE.json')['summary_sha256']==sha(a.source/'SUMMARY.json')
        cfg=read(a.source/'CONFIG.json');assert cfg['seed']==0 and cfg['profile']=='formal' and cfg['method']!='Vanilla'
        checkpoint=a.source/'checkpoint_iter20.pt';assert sha(checkpoint)==s['final_checkpoint_sha256']
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['config_sha256']==sha(a.source/'CONFIG.json')
        model=b.Model('CaDM').to('cuda:0');model.load_state_dict(ck['model'],strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        before=digest(model.state_dict())
        bank=[b.manifest(700,index,10,'ID',evaluation=True) for index in range(5)]
        # Source training seeds were0/1/2; this identifier names reset draws, not another training seed.
        assert len({x['seed'] for batch in bank for x in batch})==50
        manifest_hash=hashlib.sha256(json.dumps(bank,sort_keys=True).encode()).hexdigest()
        write(a.output/'PLAN.json',dict(source=str(a.source),source_training_seed=0,reset_bank_id=700,episodes=50,episode_steps=200,
            baseline_sha256=BASE_SHA,checkpoint_sha256=sha(checkpoint),model_state_sha256=before,reset_manifest=bank,manifest_sha256=manifest_hash,
            selection_metric='Mean actual ID environment return, larger is better; all eligible recipes use the same fixed resets and CEM seeds.',
            planning='Unchanged native CEM: horizon30/population200/elites50/iterations5; fresh planner state per independent ten-episode batch.',
            training_updates=0,additional_development_evaluation_interactions=10000,new_sealed_test=False,
            reason='Prospective control-oriented development after a mismatch between historical ID forecast selection and actual control outcomes.'))
        returns=[];success=[];records=[];start=time.monotonic()
        for index,rows in enumerate(bank):
            paths,seconds=b.sample(model,rows,'cuda:0',600000+index,random_action=False)
            b.save_paths(a.output,f'ID_batch{index}',paths)
            assert [p['metadata'] for p in paths]==rows
            values=[float(p['reward'].astype('float64').sum()) for p in paths]
            assert np.isfinite(values).all();returns+=values;success += [bool(p['success']) for p in paths]
            records.append(dict(batch=index,returns=values,seconds=seconds))
            write(a.output/'PARTIAL.json',dict(episodes=len(returns),returns=returns,batches=records))
            print(json.dumps(dict(batch=index,episodes=len(returns),mean_return=float(np.mean(returns)),seconds=time.monotonic()-start)),flush=True)
        assert len(returns)==50 and digest(model.state_dict())==before and sha(checkpoint)==s['final_checkpoint_sha256']
        write(a.output/'SUMMARY.json',dict(status='COMPLETE',name=a.name,source=str(a.source),source_training_seed=0,returns=returns,
            mean_return=float(np.mean(returns)),episode_sd=float(np.std(returns,ddof=1)),success_rate=float(np.mean(success)),
            episodes=50,evaluation_interactions=10000,training_interactions=0,optimizer_updates=0,
            manifest_sha256=manifest_hash,checkpoint_sha256=sha(checkpoint),model_unchanged=True,source_summary_sha256=sha(a.source/'SUMMARY.json'),
            seconds=time.monotonic()-start,scope='New common ID development controls on frozen existing models; not new training seeds or a sealed final test.'))
        write(a.output/'COMPLETE.json',dict(status='COMPLETE',files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}))
        write(a.output/'EXIT.json',dict(exit_code=0,time=time.time()))
    except BaseException:
        write(a.output/'FAILED.json',dict(traceback=traceback.format_exc()));write(a.output/'EXIT.json',dict(exit_code=1,time=time.time()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name',required=True);main(p.parse_args())
