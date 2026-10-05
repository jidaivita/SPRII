"""Three-condition control assay for each frozen neural historical prior.

Shared online physical inference is explicit and identical across families;
these results measure utility of their retained history under this controller,
not autonomous learned-control proficiency.
"""
import argparse,hashlib,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .control_development import boundary_sample,run_case
from .prediction_assay import donor_assignment
from .training_protocol import source_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--probes',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--replicates',type=int,default=1);p.add_argument('--full',action='store_true');a=p.parse_args()
    from .pixel_training import TrainingBank
    if a.output.exists():raise ValueError('learned control attempt exists')
    if a.replicates not in (1,2) or a.workers<1:raise ValueError('invalid control budget')
    bank=TrainingBank(a.bank);keys=bank.selection_keys if a.full else boundary_sample(bank)
    calibration=json.loads(a.calibration.read_text())
    if calibration['bank_manifest_sha256']!=bank.manifest_sha256:raise ValueError('control calibration/evaluation bank mismatch')
    prior_spec={key:str(getattr(a,key)) for key in ('checkpoint','probes','calibration','model_source')}
    jobs=[];assignments=[];conditions=('query_only_online','matched_online','wrong_online')
    for rep in range(a.replicates):
        wrong=donor_assignment(keys,991055+rep);assignments.append({key[1]:keys[wrong[i]][1] for i,key in enumerate(keys)})
        for i,key in enumerate(keys):
            query=[r for r in bank.by_system[key]['cold'] if r['template']=='pulse_050'][rep]
            matched=bank.eligible(key,'forced',96)[rep];mismatched=bank.eligible(keys[wrong[i]],'forced',96)[rep]
            for condition in conditions:
                jobs.append((str(a.bank),str(a.output),bank.systems[key],query,mismatched if condition=='wrong_online' else matched,rep,condition,len(jobs),prior_spec))
    a.output.mkdir(parents=True);config=dict(schema='vec.learned-control-development.v1.1',status='DEVELOPMENT_NOT_FROZEN',family=calibration['family'],
        source_fingerprint=source_fingerprint(),bank_manifest_sha256=bank.manifest_sha256,systems=len(keys),cases=len(jobs),replicates=a.replicates,
        checkpoint_sha256=calibration['checkpoint_sha256'],mlp_sha256=calibration['mlp_sha256'],
        calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest(),conditions=conditions,assignments=assignments,
        adapter='learned persistent history + frozen supervised physical readout + calibrated virtual measurement; common explicit online visual fit and centerPD',
        physical_privileges='train labels for representation prediction and probe fitting; validation for selection/calibration; no private query theta/state supplied to policy',
        control_contract='same task, initial episodes, donor pairing, fit steps, force cap and metrics as explicit control_development',
        population='all100 main validation systems' if a.full else '8 result-blind corner-nearest systems',formal_results=False,test_read=False)
    (a.output/'CONTROL_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');rows=[];errors=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,job):job[-2] for job in jobs}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:errors.append(dict(index=futures[future],error=repr(exc)))
            print(json.dumps(dict(completed=len(rows)+len(errors),total=len(jobs),errors=len(errors))),flush=True)
    (a.output/'CONTROL_REPORT.json').write_text(json.dumps(dict(config=config,cases=rows,errors=errors,seconds=time.monotonic()-start),indent=2)+'\n')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
