"""Generate the committed test population only after formal selection freezes.

The sole physical generator remains research_bank.generate. This entry adds
authorization, exact preallocated slots, immutable statistics and content
admission; it does not change the mechanics, camera or action templates.
"""
import argparse,json,multiprocessing,time,os
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .sealed_access import SealedAuthorization
from .sealed_bank import SCHEMA,capture_content
from .dataset_snapshot import stable_digest
from .planned_support import planned_jobs


def generation_jobs(authorization,populations):
    if not isinstance(authorization,SealedAuthorization):raise PermissionError('frozen formal selection required before generation')
    authorization.revalidate()
    if set(populations)!=set(authorization.protocol['populations']):raise ValueError('test population matrix differs from frozen protocol')
    jobs=[]
    for name,entry in populations.items():
        namespace=authorization.validate_population(name,entry['plan'],entry['systems'])
        jobs.extend(planned_jobs(entry['plan'],entry['systems'],namespace))
    if len({j['episode_key'] for j in jobs})!=len(jobs):raise ValueError('episode identity reused across sealed populations')
    return jobs


def generate_bank(output,authorization,populations,*,statistics_path,workers=32):
    # All permission and selection checks precede output creation or importing
    # the physical runtime. No candidate/development override exists here.
    jobs=generation_jobs(authorization,populations);statistics_path=Path(statistics_path)
    expected=authorization.protocol['training_bank']['target_statistics_sha256']
    if stable_digest(statistics_path)['sha256']!=expected:raise ValueError('original training statistics commitment differs')
    statistics_bytes=statistics_path.read_bytes()
    if json.loads(statistics_bytes).get('source_split')!='train':raise ValueError('non-training normalization forbidden')
    if not isinstance(workers,int) or workers<1:raise ValueError('positive generation worker budget required')
    if version('mujoco')!=authorization.protocol['render_runtime']['mujoco_version'] or os.environ.get('MUJOCO_GL')!=authorization.protocol['render_runtime']['backend']:
        raise ValueError('render runtime differs from frozen generation protocol')
    output=Path(output)
    if output.exists():raise ValueError('sealed attempt exists; do not overwrite or resample it')
    output.mkdir(parents=True);began=time.monotonic()
    config=dict(schema='vec.sealed-bank.v1',test_generated=True,test_read=False,job_count=len(jobs),
        protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256,
        source_fingerprint=authorization.protocol['source_fingerprint'],resolutions=[64,128],
        physics_config=authorization.protocol['physics_config'],render_runtime=authorization.protocol['render_runtime'],
        state='GENERATING_SEALED',episode_policy='exact committed independent slots; all failed attempts retained; no resampling',
        admission_scope='generation/content completeness only; assay support and scientific effects are evaluated separately')
    (output/'BANK_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'PLANS.private.json').write_text(json.dumps(populations,indent=2)+'\n')
    (output/'TRAIN_TARGET_STATISTICS.json').write_bytes(statistics_bytes)
    from .research_bank import generate
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(generate,(job,str(output))):job['episode_key'] for job in jobs}
        with (output/'progress.private.jsonl').open('x') as progress:
            for future in as_completed(futures):
                try:
                    row=future.result();rows.append(row);event=dict(episode_key=row['episode_key'],failure=row['failure'])
                except Exception as exc:event=dict(episode_key=futures[future],error=repr(exc));errors.append(event)
                progress.write(json.dumps(event)+'\n');progress.flush()
                if (len(rows)+len(errors))%128==0:print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),execution_errors=len(errors))),flush=True)
    # Manifest storage order has no role in grouping: the original plans bind
    # generation order and statistical blocks independently of this sorting.
    rows.sort(key=lambda r:r['episode_key'])
    (output/'MANIFEST.private.json').write_text(json.dumps(dict(schema='vec.sealed-episodes.v1',episodes=rows),indent=2)+'\n')
    report=dict(status='GENERATED_SEALED' if not errors else 'FAILURES_RETAINED',episodes=len(rows),execution_errors=errors,
        physical_failures=[dict(episode_key=r['episode_key'],failure=r['failure'],raw_frames=r['raw_frames']) for r in rows if r['failure']],
        seconds=time.monotonic()-began,test_generated=True,test_read=False)
    (output/'BANK_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors:raise RuntimeError('sealed generation has retained execution failures; no admission emitted')
    authorization.revalidate()
    if stable_digest(statistics_path)['sha256']!=expected or (output/'TRAIN_TARGET_STATISTICS.json').read_bytes()!=statistics_bytes:
        raise ValueError('training statistics changed during generation')
    content=capture_content(output,authorization)
    admission=dict(schema=SCHEMA,status='PASS',protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256,
        content=content,scope='all planned episode attempts and bytes, including failed physical prefixes; not a claim every assay has sufficient support',test_read=False)
    admission_path=output/'BANK_ADMISSION.json';admission_path.write_text(json.dumps(admission,indent=2)+'\n')
    commitment=dict(schema='vec.seal-commitment.v1',bank_admission_sha256=stable_digest(admission_path)['sha256'],
        content_sha256=content['content_sha256'],protocol_sha256=authorization.protocol_sha256,selection_sha256=authorization.selection_sha256,
        episodes=len(rows),physical_failures=len(report['physical_failures']),test_generated=True,test_read=False)
    (output/'SEAL_COMMITMENT.json').write_text(json.dumps(commitment,indent=2)+'\n');return commitment


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--selection-sha256',required=True)
    p.add_argument('--populations',type=Path,required=True);p.add_argument('--training-statistics',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();authorization=SealedAuthorization(a.protocol,a.selection,protocol_sha256=a.protocol_sha256,selection_sha256=a.selection_sha256)
    commitment=generate_bank(a.output,authorization,json.loads(a.populations.read_text()),statistics_path=a.training_statistics,workers=a.workers)
    print(json.dumps(commitment),flush=True)


if __name__=='__main__':main()
