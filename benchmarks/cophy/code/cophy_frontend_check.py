"""Two deterministic real train videos through the actual cached and online paths."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cophy_prepare_artifacts import SPECS, checked_cache, frozen_write
from cophy_protocol import digest, verify_preflight, verify_adapter_binding


@torch.no_grad()
def run(root, scene, preflight, output, device):
    from cf_learning.model import CoPhyNet, extract_pose_ab_c
    from dataloaders.utils import get_rgb, get_pose_3D
    verify_preflight(preflight, stage='features')
    bound=verify_adapter_binding(preflight);spec=SPECS[scene];root=Path(root)
    splits=json.loads(Path(bound['artifacts']['splits']['path']).read_text())
    ids=splits['train']['ids'];cache_path=root/'features'/f"{spec['cache']}_train_extracted_prop.pickle"
    cache=checked_cache(cache_path,ids,spec)
    torch.manual_seed(20260911)
    net=CoPhyNet(spec['slots']).to(device).eval()
    checkpoint=Path(bound['artifacts']['derenderer']['path'])
    net.derendering.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    records=[]
    for ident in ids[:2]:
        folder=Path(bound['input_profile']['dataset_dir'])
        if scene!='collision':folder=folder/str(spec['num_objects'])
        folder=folder/ident
        ab=torch.from_numpy(get_rgb(str(folder/'ab'))).unsqueeze(0).to(device)
        c=torch.from_numpy(get_rgb(str(folder/'cd'),max_frames=1)).unsqueeze(0).to(device)
        pa,pc,xa,xc=extract_pose_ab_c(net.derendering,ab,c)
        cached=[torch.from_numpy(cache[ident][k]).unsqueeze(0).to(device) for k in
                ['presence_ab','presence_c','pose_ab','pose_c']]
        for expected,actual in zip([pa,pc,xa,xc],cached):
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        online=net(ab,c)[0]
        stored=net(None,None,cached[0],cached[2],cached[1],cached[3])[0]
        torch.testing.assert_close(stored,online,rtol=1e-5,atol=1e-6)
        # C is a single initial observation, not a future-dependent qualification.
        gt=np.load(folder/'cd/states.npy',mmap_mode='r',allow_pickle=False)[0,:,:3]
        active=np.abs(gt).sum(-1)>0
        records.append({'id':ident,'max_prediction_difference':float((stored-online).abs().max()),
            'true_presence_C':active.tolist(),'predicted_presence_C':pc[0].tolist(),
            'C_pose_rmse_active':float(np.sqrt(((xc[0,0].cpu().numpy()-gt)[active]**2).mean()))})
    return frozen_write(output,{'status':'PASS','scene':scene,'split':'train',
        'feature_preflight_sha256':digest(preflight),'derenderer_sha256':digest(checkpoint),
        'cache_train_sha256':digest(cache_path),'examples':records,'device':str(device),
        'predictor_untrained_path_check_only':True,'test_read':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--scene',required=True,choices=SPECS)
    parser.add_argument('--preflight',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    print(run(args.root,args.scene,args.preflight,args.output,args.device))
