# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model.")
    parser.add_argument('--load_from', default=None, help='load from pretrained weights and do not freeze')
    parser.add_argument('--partial_load', action='store_true',
                        help='partially load weight for N=100')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--saliency_query', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_memory', action='store_true',
                        help="Do not use memory in mask head")
    parser.add_argument('--slim', action='store_true',
                        help="Use low resolution input image")
    parser.add_argument('--no_attentive_fusion', dest='attentive_fusion', action='store_false',
                        help="Disable attentive fusion module")
    parser.add_argument('--no_refine_block', dest='refine_block', action='store_false',
                        help="Disable attentive fusion module")
    parser.add_argument('--detr_mask_head', action='store_true',
                        help="Use detr mask head")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=1, type=float,
                        help="mask l1 coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=1, type=float,
                        help="mask dice coefficient in the matching cost")
    parser.add_argument('--mask_matcher', action='store_true',
                        help="Use mask matcher")
    parser.add_argument('--set_iou_thres', default=1, type=float,
                        help='use iou to assign gt mask to one or more predictions before matching, '
                             'set 1 to not assign')

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--coco_len', type=int, default=0)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--visualize_type', type=str, default='box',
                        help="Output of visualize results, choose from [box, seg, both]")
    parser.add_argument('--record_grad', action='store_true',
                        help="Record gradients during training. Will be slower.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks or args.dataset_file == 'sis'
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    # freeze weights
    if (args.frozen_weights is not None or args.load_from is not None) \
            and args.dataset_file == 'sis':
        if args.num_queries == 100 and not args.partial_load:  # freeze backbone and transformer
            unfreeze_layers = ('class_embed', 'bbox_embed')
        else:  # freeze backbone and transformer encoder
            unfreeze_layers = ('class_embed', 'bbox_embed', 'transformer.decoder', 'query_embed')
        if args.masks:
            unfreeze_layers = tuple('detr.' + l for l in unfreeze_layers)
            unfreeze_layers += ('cfm', 'mask_head')
            # print('Mask unfreeze layers:', unfreeze_layers)
        if args.frozen_weights is not None:
            for n, p in model.named_parameters():
                if not n.startswith(unfreeze_layers):
                    p.requires_grad_(False)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    smaller_layers = ('backbone',)
    if args.load_from is not None and args.dataset_file == 'sis':
        smaller_layers = unfreeze_layers
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if not n.startswith(smaller_layers) and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if n.startswith(smaller_layers) and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    if args.eval and args.dataset_file == 'sis':
        dataset_val = build_dataset(image_set='test', args=args)
    else:
        dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:  # for coco and sis
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None or args.load_from is not None:
        weights_path = args.frozen_weights if args.frozen_weights is not None \
            else args.load_from
        checkpoint = torch.load(weights_path, map_location='cpu')
        if args.dataset_file == 'sis' and not args.masks:
            del_keys = []
            for k in checkpoint['model'].keys():
                if k.startswith(unfreeze_layers):
                    del_keys.append(k)
            for k in del_keys:
                del checkpoint['model'][k]
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if args.dataset_file == 'sis' and args.masks:
            del_keys = []
            unfreeze_layers_og = tuple(l.replace('detr.', '') for l in unfreeze_layers)
            print(unfreeze_layers_og)
            for k in checkpoint['model'].keys():
                if k.startswith(unfreeze_layers_og):
                    del_keys.append(k)
            for k in del_keys:
                del checkpoint['model'][k]
            # for k in checkpoint['model'].keys():
            #     print('Load key:', k)
            model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    best_val_map = 0
    start_time = time.time()
    writer = None
    if args.record_grad:
        writer = SummaryWriter(log_dir=args.output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, writer=writer)
        lr_scheduler.step()

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if not args.masks:
                cur_val = test_stats['coco_eval_bbox'][0]
            else:
                cur_val = test_stats['coco_eval_masks'][0]
            if cur_val > best_val_map:  # save best result
                best_val_map = cur_val
                checkpoint_paths.append(output_dir / f'checkpoint_val_best.pth')
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    if args.record_grad:
        writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OQTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
