"""Continue the four completed GPU heads without changing their data or weights."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from xep_discovery import read,write

root=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/xep_discovery_balls_v4_1'))
lock=open(root/'continue20.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
code=Path(__file__).with_name('xep_discovery.py')
methods=['Native-U','A-U','A-P','Known-parameters']
for method in methods:
    if read(root/'runs'/method/'complete.json')['epochs']<10:raise RuntimeError('First budget incomplete')
record={'target_epochs':20,'status':'RUNNING','started_at':time.time(),'test_read':False,
        'reason':'Training and validation still improve within first ten epochs; resume existing states','jobs':[]}
processes=[]
for gpu,method in enumerate(methods):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    log=open(root/'logs'/('continue20_'+method+'.log'),'a',buffering=1)
    p=subprocess.Popen([sys.executable,'-u',str(code),'train','--out',str(root),'--method',method,'--device','cuda:0','--epochs','20'],
                       env=env,stdout=log,stderr=subprocess.STDOUT)
    job={'method':method,'gpu':gpu,'pid':p.pid,'status':'RUNNING'};record['jobs'].append(job)
    processes.append((p,log,job))
write(root/'continuation20.json',record)
while processes:
    for p,log,job in processes[:]:
        rc=p.poll()
        if rc is not None:
            job.update(exit_code=rc,status='COMPLETE' if rc==0 else 'FAILED');log.close();processes.remove((p,log,job))
            write(root/'continuation20.json',record)
    if processes:time.sleep(2)
record['status']='COMPLETE' if all(j['exit_code']==0 for j in record['jobs']) else 'FAILED'
record['finished_at']=time.time();write(root/'continuation20.json',record)
raise SystemExit(0 if record['status']=='COMPLETE' else 1)
