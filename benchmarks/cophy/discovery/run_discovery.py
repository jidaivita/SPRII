"""One resumable Balls pilot; never resumes or signals old dispatchers."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from xep_discovery import VERSION, digest, emit, read, write


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'controller.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=Path(__file__).with_name('xep_discovery.py')
    logs=out/'logs';logs.mkdir(exist_ok=True)
    record={'version':VERSION,'pid':os.getpid(),'started_at':time.time(),
            'code_sha256':digest(code),'test_read':False,'old_dispatchers_resumed':False,
            'stages':[],'status':'RUNNING'}

    def stage(name,tasks):
        entry={'stage':name,'started_at':time.time(),'status':'RUNNING','jobs':[]}
        record['stages'].append(entry);record['current_stage']=name
        handles=[]
        for label,args,gpu in tasks:
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                     PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
            log=open(logs/(label+'.log'),'a',buffering=1)
            command=[sys.executable,str(code),*args,'--out',str(out)]
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            job={'label':label,'pid':proc.pid,'gpu':gpu,'status':'RUNNING','log':str(logs/(label+'.log'))}
            entry['jobs'].append(job);handles.append((proc,log,job))
        write(out/'controller_status.json',record);emit('stage_started',stage=name,jobs=entry['jobs'])
        while handles:
            for proc,log,job in handles[:]:
                rc=proc.poll()
                if rc is not None:
                    job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc,finished_at=time.time())
                    log.close();handles.remove((proc,log,job));write(out/'controller_status.json',record)
            if handles:time.sleep(5)
        entry['status']='COMPLETE' if all(j['exit_code']==0 for j in entry['jobs']) else 'FAILED'
        entry['finished_at']=time.time();write(out/'controller_status.json',record)
        if entry['status']=='FAILED':raise RuntimeError('Failed stage '+name+'; see individual logs')

    try:
        stage('manifest',[('manifest',['manifest'],None)])
        stage('visual_prefix',[(f'prefix_{i}',['prefix','--shard',str(i),'--shards','4'],i) for i in range(4)])
        stage('frozen_support',[(f'encode_{m}',['encode','--method',m],i) for i,m in enumerate(['Native','A'])])
        stage('merge',[('merge',['merge','--shards','4'],None)])
        stage('real_batch',[('smoke',['smoke'],None)])
        stage('head_training',[(f'train_{m}',['train','--method',m,'--device','cuda:0','--epochs','10'],i)
              for i,m in enumerate(['Native-U','A-U','A-P','Known-parameters'])]+
              [('train_Query-only',['train','--method','Query-only','--device','cpu','--epochs','10'],None)])
        record['status']='COMPLETE'
        record['summary']={m:read(out/'runs'/m/'complete.json') for m in ['Native-U','A-U','A-P','Known-parameters','Query-only']}
        emit('pilot_complete',summary=record['summary'])
    except Exception as exc:
        record.update(status='FAILED',error=str(exc));raise
    finally:
        record['finished_at']=time.time();write(out/'controller_status.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
