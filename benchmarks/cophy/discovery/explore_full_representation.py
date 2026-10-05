"""Data-directed follow-up: full A representation in all support conditions."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
import xep_discovery as base
from advantage_explore import OUT

OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'full_representation.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'status':'WAITING_FOR_FIRST_QUEUE','reason':'At60 standard A-U leads Native by7.356%, A-P is tied','test_read':False,'jobs':[]}
base.write(OUT/'full_representation_followup.json',status)
while True:
    original=base.read(OUT/'dispatch_status.json')
    if original['status']=='FAILED':raise RuntimeError('Initial queue failed; inspect before followup')
    if original['status']=='COMPLETE':break
    time.sleep(5)
live=[]
for gpu,setting in zip([0,2,3],['standard_s1','standard_s5','far_s3']):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    log=open(OUT/'logs'/f'{setting}_A-U.log','a',buffering=1)
    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('advantage_explore.py')),'train','--setting',setting,'--method','A-U','--epochs','60'],env=env,stdout=log,stderr=subprocess.STDOUT)
    job={'setting':setting,'method':'A-U','gpu':gpu,'pid':proc.pid,'status':'RUNNING'}
    status['jobs'].append(job);live.append((proc,log,job))
status['status']='RUNNING';base.write(OUT/'full_representation_followup.json',status)
while live:
    for proc,log,job in live[:]:
        rc=proc.poll()
        if rc is not None:
            job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);log.close();live.remove((proc,log,job))
            base.write(OUT/'full_representation_followup.json',status)
    if live:time.sleep(3)
status['status']='COMPLETE' if all(j['exit_code']==0 for j in status['jobs']) else 'FAILED'
base.write(OUT/'full_representation_followup.json',status)
subprocess.run([sys.executable,str(Path(__file__).with_name('advantage_explore.py')),'collect'],check=True)
raise SystemExit(0 if status['status']=='COMPLETE' else 1)
