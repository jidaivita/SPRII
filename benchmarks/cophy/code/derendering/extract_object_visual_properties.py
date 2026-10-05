"""
Given a pre-trained model it extract the object properties (presence, bounding boxes and pose 3D)

# debugging:
ipython derendering/extract_object_visual_properties.py --sanity_check

"""

from dataloaders.dataset_collision import Collision_CF
from dataloaders.dataset_blocktower import Blocktower_CF
from dataloaders.dataset_balls import Balls_CF
from derendering.model import DeRendering
from torch import optim
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
import os
from tqdm import *
from random import choice
import torch.nn.functional as F
import time
from dataloaders.utils import *
import pickle as pkl


def extract_object_visual_properties(model, device, loader, sanity_check=False):
  from cf_learning.model import extract_pose_ab_c
  model.eval()
  cache = {}
  for input in loader:
    ab = input['rgb_ab'].to(device)
    c = input['rgb_cd'][:, :1].to(device)
    pa, pc, xa, xc = extract_pose_ab_c(model, ab, c)
    for i, example_id in enumerate(input['id']):
      cache[example_id] = {
        'cache_version': 'ab_c_float32_v2',
        'presence_ab': pa[i].cpu().numpy().astype(np.float32),
        'presence_c': pc[i].cpu().numpy().astype(np.float32),
        'pose_ab': xa[i].cpu().numpy().astype(np.float32),
        'pose_c': xc[i].cpu().numpy().astype(np.float32),
      }
  return cache


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader, split, type,
    num_objects):
  # choice of dataset
  if dataset_name == 'balls':
    dataset = Balls_CF(num_balls=num_objects,
                       root_dir=dataset_dir,
                       split=split,
                       is_rgb=True, only_cd=False)
    fn = f"balls_{num_objects}_{split}"
  elif dataset_name == 'collision':
    dataset = Collision_CF(type=type,
                           root_dir=dataset_dir,
                           split=split,
                           is_rgb=True, only_cd=False)
    fn = f"collision_{type}_{split}"
  elif dataset_name == 'blocktower':
    dataset = Blocktower_CF(type=type,
                            num_blocks=num_objects,
                            root_dir=dataset_dir,
                            split=split,
                            is_rgb=True, only_cd=False)
    fn = f"blocktower_{num_objects}_{type}_{split}"
  else:
    raise NameError('Unkown dataset name.')

  # loader
  dataset.rgb_c_only = True
  kwargs_loader['batch_size'] = 1
  loader = DataLoader(dataset, **kwargs_loader)

  return loader, fn


def main(args):
  from cophy_protocol import verify_preflight, verify_adapter_binding, verify_runtime_inputs
  verify_preflight(args.preflight_receipt, release=args.sealed_release,
                   require_release=(args.split == 'test'), stage='features')
  verify_adapter_binding(args.preflight_receipt)
  verify_runtime_inputs(args.preflight_receipt, args, require_caches=False)
  # kwargs
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  kwargs_loader = {}
  if device.type == 'cuda':
    kwargs_loader.update({'num_workers': args.workers, 'pin_memory': True})

  # datasets and loaders
  loader, fn = get_dataloaders(args.dataset_name,
                               args.dataset_dir,
                               kwargs_loader,
                               args.split,
                               args.type,
                               args.num_objects)

  # model and optim
  model = DeRendering(num_objects=loader.dataset.num_objects).to(device)

  # load the derendering module
  pretrained_dict = torch.load(args.derendering_ckpt, map_location='cpu', weights_only=True)
  pretrained_dict = {k: v for k, v in pretrained_dict.items()}
  model_dict = model.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                     k in model_dict}
  model.load_state_dict(pretrained_dict, strict=True)

  # extract
  dict_id2object_properties = extract_object_visual_properties(model, device, loader, args.sanity_check)

  # save
  os.makedirs(args.out_dir, exist_ok=True)
  out_fn = os.path.join(args.out_dir, f"{fn}_extracted_prop.pickle")
  with open(out_fn, 'wb') as f:
    pkl.dump(dict_id2object_properties, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Training of the derendering module.')
  parser.add_argument('--dataset_dir',
                      default='data/cophy/blocktowerCF',
                      type=str,
                      help='Location of the data.')
  parser.add_argument('--derendering_ckpt',
                      # default='checkpoints/cophy/ballsCF/model_state_dict.pt',
                      default='./ckpts/derendering/blocktowerCF/model_state_dict.pt',
                      type=str,
                      help='Location of the pre-trained derendering module.')
  parser.add_argument('--out_dir',
                      default='./preextracted_object_visual_properties',
                      type=str,
                      help='Location of the out dir.')
  parser.add_argument('--dataset_name',
                      default='blocktower',
                      type=str,
                      help='Which dataset to take (balls, collision, blocktower).')
  parser.add_argument('--workers', default=8, type=int, help='Workers.')
  parser.add_argument('--num_objects',
                      default=4,
                      type=int,
                      help='Number of objects for training.')
  parser.add_argument('--type',
                      default='normal',
                      type=str,
                      help='Type of train/val/test split.')
  parser.add_argument('--split',
                      default='val',
                      type=str,
                      help='Type of train/val/test split.')
  parser.add_argument('--sanity_check', dest='sanity_check', action='store_true')
  parser.add_argument('--no_sanity_check', dest='sanity_check', action='store_false')
  parser.set_defaults(feature=False)
  parser.add_argument('--preflight_receipt', required=True)
  parser.add_argument('--sealed_release', default=None)
  args = parser.parse_args()

  main(args)
