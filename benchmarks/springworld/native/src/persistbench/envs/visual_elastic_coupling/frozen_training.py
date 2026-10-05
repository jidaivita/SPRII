"""Formal entry: verify all input bytes before and after a training attempt.

The common developmental trainer is reused without changing its learning code.
A TRAIN_REPORT alone is insufficient for formal admission: the completion
receipt here must PASS and bind the frozen protocol, bank and model artifacts.
"""
import argparse,json
from pathlib import Path
from .dataset_snapshot import verify
from .training_protocol import file_digest


def run(args):
    from .pixel_training import train
    if args.output.exists():raise ValueError('formal attempt already exists')
    args.stage='formal';args.formal_entry='frozen_training'
    before=dict(snapshot=file_digest(args.bank/'BANK_SNAPSHOT.json'),protocol=file_digest(args.protocol),
                admission=file_digest(args.data_admission))
    snapshot=json.loads((args.bank/'BANK_SNAPSHOT.json').read_text())
    try:
        train(args)  # verify_formal re-hashes every declared private/public file.
        after=dict(snapshot=file_digest(args.bank/'BANK_SNAPSHOT.json'),protocol=file_digest(args.protocol),
                   admission=file_digest(args.data_admission))
        if before!=after:raise ValueError('formal sidecar/protocol/admission changed during training')
        final=verify(args.bank,snapshot)
        training=json.loads((args.output/'TRAIN_CONFIG.json').read_text())
        report=json.loads((args.output/'TRAIN_REPORT.json').read_text())
        if report.get('status')!='EXECUTED' or report.get('stage')!='formal':raise ValueError('training did not complete formally')
        for name,digest in report['checkpoints'].items():
            if file_digest(args.output/name)!=digest:raise ValueError('final checkpoint digest changed')
        result=dict(schema='vec.formal-training-completion.v1.1',status='PASS',input_bindings=training['formal_bindings'],
            protocol_sha256=before['protocol'],data_admission_sha256=before['admission'],after_run_content_check=final,
            checkpoints=report['checkpoints'],train_report_sha256=file_digest(args.output/'TRAIN_REPORT.json'),test_read=False)
    except Exception as exc:
        if args.output.exists():
            (args.output/'FORMAL_COMPLETION.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc),test_read=False),indent=2)+'\n')
        raise
    (args.output/'FORMAL_COMPLETION.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('gru','transformer','transition_deepsets','causal_tcn'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resolution',type=int,default=128);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data-admission',type=Path,required=True)
    p.add_argument('--max-updates',type=int,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--accumulation',type=int,default=1);p.add_argument('--learning-rate',type=float,required=True)
    p.add_argument('--validate-every',type=int,required=True);p.add_argument('--validation-batches',type=int,required=True)
    args=p.parse_args()
    if min(args.max_updates,args.batch_size,args.accumulation,args.validate_every,args.validation_batches)<1:
        raise ValueError('positive formal training budgets required')
    run(args)


if __name__=='__main__':main()
