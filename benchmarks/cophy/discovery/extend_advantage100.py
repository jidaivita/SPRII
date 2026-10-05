"""Continue the standard and best exploratory condition with its Native control."""
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import xep_discovery as base
from advantage_explore import OUT,collect

lock=open(OUT/'extend100.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record={'status':'WAITING_FOR_60','test_read':False,'waves':[]}
base.write(OUT/'extension100.json',record)
while True:
    state=base.read(OUT/'full_representation_followup.json')
    if state['status']=='FAILED':raise RuntimeError('Prior stage failed')
    if state['status']=='COMPLETE':break
    time.sleep(5)
collect();shutil.copy2(OUT/'result_snapshot.json',OUT/'result_60epochs.json')
snapshot=base.read(OUT/'result_60epochs.json');rank=[]
for setting,rows in snapshot['settings'].items():
    native=rows['Native-U']['mse']
    for method in ['A-U','A-P']:
        rank.append(((native-rows[method]['mse'])/native,setting,method))
gain,best_setting,best_method=max(rank)
record.update(status='RUNNING',selected_setting=best_setting,selected_method_at60=best_method,gain_at60=gain)
waves=[[('standard_s3',m,gpu) for m,gpu in [('Native-U',0),('A-U',1),('A-P',2),('Known-parameters',3),('Query-only',0)]]]
if best_setting!='standard_s3':waves.append([(best_setting,m,gpu) for m,gpu in [('Native-U',0),('A-U',2),('A-P',3)]])
for tasks in waves:
    jobs=[];record['waves'].append(jobs);live=[]
    for setting,method,gpu in tasks:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        log=open(OUT/'logs'/f'{setting}_{method}_100.log','a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('advantage_explore.py')),'train','--setting',setting,'--method',method,'--epochs','100'],env=env,stdout=log,stderr=subprocess.STDOUT)
        job={'setting':setting,'method':method,'gpu':gpu,'pid':proc.pid,'status':'RUNNING'};jobs.append(job);live.append((proc,log,job))
    base.write(OUT/'extension100.json',record)
    while live:
        for proc,log,job in live[:]:
            rc=proc.poll()
            if rc is not None:
                job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);log.close();live.remove((proc,log,job))
                base.write(OUT/'extension100.json',record)
        if live:time.sleep(3)
    if any(j['exit_code']!=0 for j in jobs):
        record['status']='FAILED';base.write(OUT/'extension100.json',record);raise RuntimeError('Continuation failed')
collect();shutil.copy2(OUT/'result_snapshot.json',OUT/'result_after100.json')
random_root=base.ROOT/'xep_random_reference_v4_3'
if (random_root/'result_snapshot.json').exists():
    shutil.copy2(random_root/'result_snapshot.json',random_root/'result_60epochs.json')
    live=[];record['random_jobs']=[]
    for method,gpu in [('Random-U',0),('Random-P',2)]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        log=open(random_root/'logs'/f'standard_s3_{method}_100.log','a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('random_reference.py')),'train','--setting','standard_s3','--method',method,'--epochs','100'],env=env,stdout=log,stderr=subprocess.STDOUT)
        job={'method':method,'gpu':gpu,'pid':proc.pid,'status':'RUNNING'};record['random_jobs'].append(job);live.append((proc,log,job))
    base.write(OUT/'extension100.json',record)
    while live:
        for proc,log,job in live[:]:
            rc=proc.poll()
            if rc is not None:
                job.update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);log.close();live.remove((proc,log,job))
                base.write(OUT/'extension100.json',record)
        if live:time.sleep(3)
    if any(j['exit_code']!=0 for j in record['random_jobs']):
        record['status']='FAILED';base.write(OUT/'extension100.json',record);raise RuntimeError('Random100 failed')
    subprocess.run([sys.executable,str(Path(__file__).with_name('random_reference.py')),'collect'],check=True)
    shutil.copy2(random_root/'result_snapshot.json',random_root/'result_after100.json')
record['status']='COMPLETE';base.write(OUT/'extension100.json',record)
