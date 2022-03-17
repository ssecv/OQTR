# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from engine import evaluate
from models import build_model
from run_utils import get_args_parser
from datasets import build_dataset, get_coco_api_from_dataset


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:  # for coco and sis
        base_ds = get_coco_api_from_dataset(dataset_val)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                          data_loader_val, base_ds, device, args.output_dir)
    if args.output_dir:
        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
