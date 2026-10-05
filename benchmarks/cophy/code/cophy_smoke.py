"""The single bounded real-data smoke required before a scene's formal runs."""
import argparse
import itertools
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from cophy_adapter import PTCoPhy
from cophy_protocol import digest, verify_preflight, verify_adapter_binding
from cophy_relations import RelationIndex, PairProvider, read_artifact
from cophy_training import train_adapter_epoch
from cophy_prepare_artifacts import SPECS, frozen_write


class FirstBatches:
    def __init__(self, loader, batches):
        self.loader=loader;self.dataset=loader.dataset;self.batches=batches
    def __iter__(self):return itertools.islice(self.loader,self.batches)


def run(args):
    from cf_learning.main import get_dataloaders
    from cf_learning.model import CoPhyNet
    verify_preflight(args.preflight);bound=verify_adapter_binding(args.preflight)
    profile=bound['input_profile'];spec=SPECS[profile['scene']];device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Formal smoke requires the real GPU path')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'receipt.json').exists():raise ValueError('Smoke already recorded; do not rerun silently')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    cache=Path(bound['artifacts']['cache_train']['path']).parent
    loader,_,_,dims=get_dataloaders(profile['scene'],profile['dataset_dir'],
        {'batch_size':32,'num_workers':args.workers,'pin_memory':True},profile['num_objects'],profile['type'],
        preextracted_obj_vis_prop_dir=str(cache),train_from_rgb=False,sampler_seed=0)
    if len(loader)<200:raise ValueError('Primary training split has fewer than the registered 200 batches')
    method='A' if 'A' in bound['allowed_methods'] else 'Native'
    backbone=CoPhyNet(spec['slots'])
    backbone.derendering.load_state_dict(torch.load(bound['artifacts']['derenderer']['path'],map_location='cpu',weights_only=True),strict=True)
    model=PTCoPhy(backbone,method).to(device)
    provider=(PairProvider(RelationIndex(read_artifact(args.preflight,'relation_index'),loader.dataset.list_ex,profile['scene']),loader.dataset,0)
              if method=='A' else None)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    stats=train_adapter_epoch(model,device,FirstBatches(loader,200),optimizer,str(output/'smoke.jsonl'),
        epoch=1,pair_provider=provider,lambda_x=.1,lambda_p=1.,dims=dims)
    if stats['updates']!=200:raise ValueError('Smoke did not finish its registered batch count')
    # Health and real timing only. No checkpoint is saved or carried into training.
    return frozen_write(output/'receipt.json',{'status':'PASS','scene':profile['scene'],
        'preflight_sha256':digest(args.preflight),'method':method,'updates':stats['updates'],
        'paired':stats['paired'],'seconds':stats['seconds'],'peak_memory_bytes':stats.get('peak_memory_bytes'),
        'no_weights_exported':True,'formal_training_started':False,'test_read':False,
        'all_training_seeds_reinitialize_model_and_optimizer':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(args.threads);print(run(args))
