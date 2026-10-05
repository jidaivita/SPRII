"""
Counterfactual learning

# debugging:
ipython cf_learning/main.py

"""

from dataloaders.dataset_collision import Collision_CF
from dataloaders.dataset_blocktower import Blocktower_CF
from dataloaders.dataset_balls import Balls_CF
from cf_learning.model import CoPhyNet, CopyC
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
import random
from dataloaders.utils import *


def get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
               gt_pose_d, gt_stab_d, gt_presence_cd,
               w_stab=1., w_pose=1.):
    mask = gt_presence_cd[:, None, :].expand_as(pred_stab_d)
    if mask.sum().item() == 0:
        raise ValueError("No valid target objects in batch")
    stationary = F.binary_cross_entropy_with_logits(pred_stab_d, gt_stab_d, reduction='none')
    loss_stab = (stationary * mask).sum() / mask.sum()
    loss_3d = (((pred_pose_d - gt_pose_d)**2).mean(-1) * mask).sum() / mask.sum()
    return w_stab * loss_stab + w_pose * loss_3d, (loss_stab, loss_3d)


def get_acc_stab(pred, gt):
    acc = 1. - torch.mean(torch.abs((pred > 0).float() - gt))
    return acc


def train_one_epoch(model, device, loader, optimizer,
                    log_file,
                    print_freq=50, D=3,
                    is_rgb=False):
    model.train()

    end = time.time()
    list_acc_stab, list_mse_3d = [], []
    loader.dataset.is_rgb = is_rgb
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end
        if is_rgb:
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:,:1].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(rgb_ab, rgb_c)
        else:
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )


        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]
        gt_presence_cd = input['presence_cd'].to(device)
        gt_stab_d = input['stab_cd'][:, 1:].to(device)

        # loss
        loss, _ = get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
                             gt_pose_d, gt_stab_d, gt_presence_cd,
                             w_stab=1., w_pose=10.)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % print_freq == 0:
            # metrics
            mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D).mean()
            acc_stab = 100. * get_acc_stab(pred_stab_d, gt_stab_d)
            list_mse_3d.append(mse_3d.item())
            list_acc_stab.append(acc_stab.item())

            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"Loss = {loss:.4f} "
                  f"Acc_stab = {np.mean(list_acc_stab):.2f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )

    # append to log file
    with open(log_file, "a+") as f:
        f.write(f"Acc_presence={np.mean(list_acc_stab):.2f} "
                f"MSE_3d={np.mean(list_mse_3d):.6f}\n")


def get_mse(pred, gt, presence, D=3):
    T = pred.shape[1]
    dist = ((pred[:, :, :, :D] - gt[:, :, :, :D]) ** 2).mean(-1) * presence.unsqueeze(1)  # (B,T,K)
    count = presence.sum(1) * T
    if (count <= 0).any():
        raise ValueError("Zero predicted presence: register coverage failure, do not report zero MSE")
    mse = dist.sum((1, 2)) / count
    return mse


@torch.no_grad()
def validate(model, device, loader, log_dir, log_file, print_freq=100, D=3, is_rgb=False):
    model.eval()

    end = time.time()
    list_mse_3d = []
    loader.dataset.is_rgb = is_rgb
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end

        # pred
        if is_rgb:
            # from RGB
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:, :1].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(rgb_ab, rgb_c)
        else:
            #fro preextracted visual object properties
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )
        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]

        # metrics
        mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D)
        list_mse_3d.extend(mse_3d.detach().cpu().tolist())

        if i % print_freq == 0:
            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )
    # append to log file
    to_write = f"MSE_3d={np.mean(list_mse_3d):.6f}\n"
    print(f"\n***Results: {to_write}***\n")
    with open(log_file, "a+") as f:
        f.write(to_write)
    return float(np.mean(list_mse_3d))


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader, num_objects=3, type='normal',
                    preextracted_obj_vis_prop_dir='',
                    train_from_rgb=False,
                    evaluate_on_test_only=False, sampler_seed=0):
    # choice of dataset
    if dataset_name == 'balls':
        if not evaluate_on_test_only:
            train_dataset = Balls_CF(num_balls=num_objects,
                                     root_dir=dataset_dir,
                                     split='train',
                                     is_rgb=train_from_rgb,
                                     only_cd=False,
                                     use_preextracted_object_properties=not train_from_rgb,
                                     preextracted_object_properties_dir=preextracted_obj_vis_prop_dir, #'data/cophy/extracted_object_properties/ballCF'
                                     )
            val_dataset = Balls_CF(num_balls=num_objects,
                                   root_dir=dataset_dir,
                                   split='val',
                                   is_rgb=train_from_rgb,
                                   only_cd=False,
                                   use_preextracted_object_properties=not train_from_rgb,
                                   preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                   )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = None
        if evaluate_on_test_only:
            test_dataset = Balls_CF(num_balls=num_objects,
                                    root_dir=dataset_dir,
                                    split='test',
                                    is_rgb=True,
                                    only_cd=False,
                                    use_preextracted_object_properties=False,
                                    preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                    )
        D = 2
    elif dataset_name == 'collision':
        if not evaluate_on_test_only:
            train_dataset = Collision_CF(type=type,
                                         root_dir=dataset_dir,
                                         split='train',
                                         is_rgb=train_from_rgb,
                                         only_cd=False,
                                         use_preextracted_object_properties=not train_from_rgb,
                                         preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                         )
            val_dataset = Collision_CF(type=type,
                                       root_dir=dataset_dir,
                                       split='val',
                                       is_rgb=train_from_rgb,
                                       only_cd=False,
                                       use_preextracted_object_properties=not train_from_rgb,
                                       preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                       )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = None
        if evaluate_on_test_only:
            test_dataset = Collision_CF(type=type,
                                        root_dir=dataset_dir,
                                        split='test',
                                        is_rgb=True,
                                        only_cd=False,
                                        use_preextracted_object_properties=False,
                                        preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                        )
        D = 3
    elif dataset_name == 'blocktower':
        if not evaluate_on_test_only:
            train_dataset = Blocktower_CF(type=type,
                                          num_blocks=num_objects,
                                          root_dir=dataset_dir,
                                          split='train',
                                          is_rgb=train_from_rgb,
                                          only_cd=False,
                                          use_preextracted_object_properties=not train_from_rgb,
                                          preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                          )
            val_dataset = Blocktower_CF(type=type,
                                        num_blocks=num_objects,
                                        root_dir=dataset_dir,
                                        split='val',
                                        is_rgb=train_from_rgb,
                                        only_cd=False,
                                        use_preextracted_object_properties=not train_from_rgb,
                                        preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                        )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = None
        if evaluate_on_test_only:
            test_dataset = Blocktower_CF(type=type,
                                         num_blocks=num_objects,
                                         root_dir=dataset_dir,
                                         split='test',
                                         is_rgb=True,
                                         only_cd=False,
                                         use_preextracted_object_properties=False,
                                         preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                         )
        D = 3
    else:
        raise NameError('Unkown dataset name.')

    # loader
    if not evaluate_on_test_only:
        train_generator = torch.Generator().manual_seed(sampler_seed)
        train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **kwargs_loader)
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        val_loader = DataLoader(val_dataset, generator=torch.Generator().manual_seed(sampler_seed+1), **kwargs_loader_val)
        return train_loader, val_loader, None, D
    else:
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        test_loader = DataLoader(test_dataset, **kwargs_loader_val)
        return None, None, test_loader, D


def get_trainable_params(model):
    """ get list of parameters to train of a network """
    trainable_params = []
    for name_c, child in model.named_children():
        for name_p, param in child.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

    return trainable_params


def main(args):
    import json
    from pathlib import Path
    from cophy_adapter import PTCoPhy, ADAPTER_VERSION
    from cophy_protocol import verify_preflight, verify_adapter_binding, verify_runtime_inputs, digest
    from cophy_relations import RelationIndex, PairProvider, ParameterFeatures, read_artifact, artifact_path
    from cophy_training import train_adapter_epoch, validate_adapter

    verify_preflight(args.preflight_receipt, release=args.sealed_release if args.evaluate else None,
                     require_release=args.evaluate)
    preflight = verify_adapter_binding(args.preflight_receipt)
    if args.method not in preflight.get('allowed_methods', []):
        raise ValueError('Method is not eligible under the audited scene/gravity branch')
    data_binding = verify_runtime_inputs(args.preflight_receipt, args, require_caches=not args.evaluate)
    if preflight.get('scene') != args.dataset_name:
        raise ValueError('Scene disagrees with the audited preflight')
    if digest(args.derendering_ckpt) != digest(artifact_path(args.preflight_receipt, 'derenderer')):
        raise ValueError('Visual checkpoint differs from the audited cache frontend')
    if args.method in {'A', 'Random'} and args.train_from_rgb and not args.evaluate:
        raise ValueError('Relation training requires the audited AB+C cache profile')
    if args.model == 'copy_c' and not args.evaluate:
        raise ValueError('Copy C is evaluation-only')
    if args.model not in {'copy_c', 'cophynet'}:
        raise ValueError('Unknown model')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs_loader = {'batch_size': args.batch_size}
    if device.type == 'cuda':
        kwargs_loader.update(num_workers=args.workers, pin_memory=True)
    train_loader, val_loader, test_loader, D = get_dataloaders(
        args.dataset_name, args.dataset_dir, kwargs_loader, args.num_objects, args.type,
        preextracted_obj_vis_prop_dir=args.preextracted_obj_vis_prop_dir,
        train_from_rgb=args.train_from_rgb, evaluate_on_test_only=args.evaluate,
        sampler_seed=args.seed)
    shape_dataset = test_loader.dataset if args.evaluate else train_loader.dataset
    for loader in (train_loader, val_loader, test_loader):
        if loader is not None:
            loader.dataset.rgb_c_only = True

    parameter_stores = {}
    parameter_schema = None
    if args.method == 'Param-known' and args.model != 'copy_c':
        for split in (['test'] if args.evaluate else ['train', 'val']):
            source = args.sealed_release if split == 'test' else args.preflight_receipt
            store = ParameterFeatures(read_artifact(source, f'parameters_{split}'),
                                      args.dataset_name, split)
            parameter_stores[split] = store
            if parameter_schema is not None and store.schema != parameter_schema:
                raise ValueError('Parameter feature schema/normalization changed across splits')
            parameter_schema = store.schema
        audit = read_artifact(args.preflight_receipt, 'audit')
        fields = [field['name'] for field in next(iter(parameter_stores.values())).fields]
        if fields != audit.get('parameter_fields'):
            raise ValueError('Parameter reference omits or reorders audited fields')
    backbone = (CopyC if args.model == 'copy_c' else CoPhyNet)(num_objects=shape_dataset.num_objects)
    for parameter in backbone.derendering.parameters():
        parameter.requires_grad_(False)
    backbone.derendering.load_state_dict(torch.load(args.derendering_ckpt, map_location='cpu', weights_only=True), strict=True)
    backbone.derendering.eval()
    model = (backbone if args.model == 'copy_c' else
             PTCoPhy(backbone, args.method,
                     len(next(iter(parameter_stores.values())).fields) if parameter_stores else None)).to(device)

    run_config = dict(adapter_version=ADAPTER_VERSION, method=args.method, model=args.model,
                      scene=args.dataset_name, num_objects=args.num_objects, split_type=args.type,
                      seed=args.seed, batch_size=args.batch_size, lambda_x=args.lambda_x, lambda_p=args.lambda_p,
                      parameter_schema=parameter_schema, code_sha256=preflight['code_sha256'],
                      protocol_sha256=preflight['artifacts']['protocol']['sha256'], data_binding=data_binding)
    os.makedirs(args.log_dir, exist_ok=True)
    config_path = Path(args.log_dir)/'run_config.json'
    if not args.evaluate and config_path.exists() and not args.resume:
        raise ValueError('Run directory exists; resume explicitly or use a new directory')

    def check_checkpoint(state):
        if state.get('run_config') != run_config:
            raise ValueError('Checkpoint belongs to another adapter/method/configuration; do not mix full-U and PT16')

    if args.evaluate:
        if args.model != 'copy_c':
            seal=json.loads(Path(args.sealed_release).read_text())
            if digest(args.pretrained_ckpt) not in seal.get('checkpoint_sha256',[]):
                raise ValueError('Test checkpoint was not selected before release')
            state = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
            check_checkpoint(state)
            model.load_state_dict(state['model'], strict=True)
            validate_adapter(model, device, test_loader, str(Path(args.log_dir)/'test.txt'),
                             dims=D, is_rgb=True, parameter_store=parameter_stores.get('test'))
        else:
            validate(model, device, test_loader, args.log_dir, str(Path(args.log_dir)/'test.txt'), D=D, is_rgb=True)
        return

    pair_provider = None
    if args.method in {'A', 'Random'}:
        index = RelationIndex(read_artifact(args.preflight_receipt, 'relation_index'),
                              train_loader.dataset.list_ex, args.dataset_name)
        pair_provider = PairProvider(index, train_loader.dataset, args.seed)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    best_score, start_epoch = float('inf'), 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        check_checkpoint(state)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best_score = state['epoch']+1, state['best_score']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        train_loader.generator.set_state(state['loader_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    config_path.write_text(json.dumps(run_config, indent=2)+'\n')
    (Path(args.log_dir)/'model_cost.json').write_text(json.dumps({
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'all_parameters': sum(p.numel() for p in model.parameters()),
        'extra_donor_encoder': args.method in {'A', 'Random'},
    }, indent=2)+'\n')
    for epoch in range(start_epoch, args.epochs+1):
        train_adapter_epoch(model, device, train_loader, optimizer, str(Path(args.log_dir)/'train.jsonl'),
                            epoch=epoch, pair_provider=pair_provider,
                            parameter_store=parameter_stores.get('train'),
                            lambda_x=args.lambda_x, lambda_p=args.lambda_p,
                            is_rgb=args.train_from_rgb, dims=D)
        score = validate_adapter(model, device, val_loader, str(Path(args.log_dir)/'val.txt'),
                                 dims=D, is_rgb=args.train_from_rgb,
                                 parameter_store=parameter_stores.get('val'), epoch=epoch)
        if epoch in {10, 15, 20, 25, 30, 35, 40, 45, 50}:
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'run_config': run_config}
            torch.save(checkpoint, Path(args.log_dir)/f'epoch_{epoch}.pt')
            if score < best_score:
                best_score = score
                torch.save(checkpoint, Path(args.log_dir)/'model_state_dict.pt')
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'run_config': run_config, 'best_score': best_score,
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'loader_rng': train_loader.generator.get_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
                   Path(args.log_dir)/'latest_resume.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the derendering module.')
    parser.add_argument('--dataset_dir',
                        # default='data/cophy/ballsCF',
                        # default='data/cophy/collisionCF',
                        default='data/cophy/blocktowerCF',
                        type=str,
                        help='Location of the data.')
    parser.add_argument('--num_objects',
                        default=3,
                        type=int,
                        help='Number of objects for training.')
    parser.add_argument('--type',
                        default='normal',
                        type=str,
                        help='Type of train/val/test split.')
    parser.add_argument('--derendering_ckpt',
                        # default='checkpoints/cophy/ballsCF/model_state_dict.pt',
                        default='checkpoints/cophy/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--log_dir',
                        default='runs/cophy/cf_learning',
                        type=str,
                        help='Location of the log dir.')
    parser.add_argument('--dataset_name',
                        # default='balls',
                        # default='collision',
                        default='blocktower',
                        type=str,
                        help='Which dataset to take (balls, collision, blocktower).')
    parser.add_argument('--model',
                        # default='copy_c',
                        # default='copy_b',
                        default='cophynet',
                        type=str,
                        help='Model name to use.')
    parser.add_argument('--method', choices=['Native', 'A', 'Random', 'Param-known'], default='Native')
    parser.add_argument('--lambda_x', type=float, default=.1)
    parser.add_argument('--lambda_p', type=float, default=1.)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Workers.')
    parser.add_argument('--epochs', default=25, type=int, help='Num epochs.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--preflight_receipt', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sealed_release', default=None)
    parser.add_argument('--train-from-rgb', dest='train_from_rgb', action='store_true')
    parser.add_argument('--no-train-from-rgb', dest='train_from_rgb', action='store_false')
    parser.set_defaults(train_from_rgb=False)
    parser.add_argument('--pretrained_ckpt',
                        default='checkpoints/cophy/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--preextracted_obj_vis_prop_dir',
                        default='',
                        type=str,
                        help='Location of the pre-extracted object visual properties.')

    args = parser.parse_args()

    main(args)
