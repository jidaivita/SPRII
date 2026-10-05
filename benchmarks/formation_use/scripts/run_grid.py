"""Finite multi-device grid runner. Existing complete jobs are verified, never overwritten."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from queue import Queue,Empty
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sprii_next.io import load_protocol,read,write,digest,code_hashes
from sprii_next.engine import jobs,job_name
from sprii_next.statistics import load_rows,verify_gate
from sprii_next.protocol import preflight


def selected_indices(grid,source_seeds=None,job_indices=None,shard_index=0,shard_count=1):
    if shard_count<1 or not 0<=shard_index<shard_count:raise ValueError('invalid shard index/count')
    if source_seeds is not None and (not source_seeds or not set(source_seeds)<=set(range(3))):
        raise ValueError('source seeds must be selected from 0, 1, 2')
    candidates=list(range(len(grid))) if job_indices is None else list(job_indices)
    if len(set(candidates))!=len(candidates) or any(i<0 or i>=len(grid) for i in candidates):
        raise ValueError('job indices must be unique registered indices')
    candidates=[i for i in candidates if source_seeds is None or grid[i]['source_seed'] in source_seeds]
    chosen=candidates[shard_index::shard_count]
    if not chosen:raise ValueError('job selection is empty')
    return chosen


def main():
    p=argparse.ArgumentParser()
    for key in ('protocol','output-root','environment','stage'):p.add_argument('--'+key,required=True)
    p.add_argument('--devices',nargs='+',default=['cuda:0']);p.add_argument('--gate')
    p.add_argument('--source-seeds',nargs='+',type=int,help='run ready sources without waiting for other seeds')
    p.add_argument('--job-indices',nargs='+',type=int,help='explicit indices in the full registered grid')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args();cfg=load_protocol(a.protocol);grid=jobs(a.environment,a.stage)
    indices=selected_indices(grid,a.source_seeds,a.job_indices,a.shard_index,a.shard_count)
    if len(a.devices)!=len(set(a.devices)):raise ValueError('each device must be listed once')
    if a.stage=='formal':
        if a.gate is None:raise PermissionError('formal readers require pilot go')
        verify_gate(a.gate,digest(cfg))
    root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    report=preflight(cfg,a.environment,a.stage,[grid[i] for i in indices])
    attempt=root/'launches'/str(time.time_ns());attempt.mkdir(parents=True)
    write(attempt/'PREFLIGHT.json',report)
    write(attempt/'JOBS.json',[dict(job_index=i,**grid[i]) for i in indices])
    queue=Queue()
    for index in indices:queue.put(index)
    def worker(device):
        while True:
            try:index=queue.get_nowait()
            except Empty:return
            job=grid[index];out=root/'runs'/job_name(job)
            if (out/'COMPLETE.json').exists():
                _,complete,run=load_rows(out)
                if complete['job']!=job or run['protocol_sha256']!=digest(cfg) or run['code_sha256']!=code_hashes():
                    raise ValueError('completed job belongs to a different implementation/protocol')
                continue
            if out.exists():raise FileExistsError('incomplete attempt preserved; use a separately recorded new root: '+str(out))
            command=[sys.executable,'-m','sprii_next','run','--protocol',str(Path(a.protocol).resolve()),'--root',str(root),
                '--environment',a.environment,'--stage',a.stage,'--job-index',str(index),'--device',device]
            if a.gate:command+=['--gate',str(Path(a.gate).resolve())]
            env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            print(f"START registered_index={index} {job_name(job)} on {device}",flush=True)
            with (attempt/f'job{index:03d}.log').open('x') as log:
                subprocess.run(command,env=env,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f"COMPLETE registered_index={index} {job_name(job)}",flush=True)
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        futures=[pool.submit(worker,device) for device in a.devices]
        for future in futures:future.result()
    write(attempt/'COMPLETE.json',dict(jobs=len(indices),job_indices=indices,protocol_sha256=digest(cfg),test_read=False))


if __name__=='__main__':main()
