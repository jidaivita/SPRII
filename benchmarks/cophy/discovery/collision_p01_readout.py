"""One existing lower-alignment source, with the same Collision downstream task."""
import fcntl
import torch
import collision_xep as base

BASE=base.ROOT/'xep_discovery_collision_v4_4'
OUT=base.ROOT/'xep_collision_p01_v4_4'

def main():
    OUT.mkdir(exist_ok=True)
    lock=open(OUT/'dispatch.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'RUNNING','test_read':False,'source_variant':'A-p01',
        'reason':'Default source A-U still trails Native near 60-head-epoch pilot; reuse existing lambda_p=0.1 source without new pretraining.',
        'default_random_is_not_weight_matched':True,'base_manifest_sha256':base.digest(BASE/'manifest.json'),
        'wrapper_sha256':base.digest(__file__)}
    base.write(OUT/'status.json',status)
    try:
        if not (OUT/'manifest.json').exists():
            m=base.read(BASE/'manifest.json')
            ckpt=base.ROOT/'runs_followup/collision/candidates/p01/A/model_state_dict.pt'
            state=torch.load(ckpt,map_location='cpu',weights_only=False);c=state['run_config']
            assert c['method']=='A' and c['lambda_p']==0.1 and c['lambda_x']==0.1
            assert c['data_binding']['preflight_sha256']==m['preflight_sha256']
            m['source_models']={'A':{'path':str(ckpt),'sha256':base.digest(ckpt),'epoch':state['epoch'],'source_run_config':c}}
            m['source_variant']='A-p01';m['base_manifest_sha256']=status['base_manifest_sha256']
            base.write(OUT/'manifest.json',m)
            del state
        for split in ['train','val']:
            for prefix in ['input','target','parameters']:
                dest=OUT/f'{prefix}_{split}.npz'
                if not dest.exists():dest.symlink_to(BASE/dest.name)
        base.encode_source(OUT,'A')
        for budget in [20,60]:
            base.train(OUT,'A-U','cuda:0',budget)
            base.write(OUT/f'result_{budget}epochs.json',{'method':'A-p01-U','budget':budget,
                'test_read':False,'selected':base.read(OUT/'runs/A-U/selected_validation.json')})
        status['status']='COMPLETE';status['result']=base.read(OUT/'runs/A-U/selected_validation.json')
    except Exception as exc:
        status.update(status='FAILED',error=str(exc));raise
    finally:
        base.write(OUT/'status.json',status)

if __name__=='__main__':main()
