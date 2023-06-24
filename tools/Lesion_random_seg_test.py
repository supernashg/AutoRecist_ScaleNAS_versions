# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import random
import logging
import time
import timeit
import ast
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import _init_paths
import models
import dataset
from config import cfg
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.seg_function import validate_seg as validate
from core.oneshot_function import calib_bn_seg as calib_bn
from utils.utils import get_model_summary
from utils.utils import create_logger, FullModel
from dataset.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--bn_calib',
                        action='store_true')
    parser.add_argument('--save_name',
                        type=str,
                        default='oneshot')
    parser.add_argument('--sample_num',
                        type=int,
                        default=100)
    parser.add_argument('--fusion_percentage',
                        type=float,
                        default=0.5)
    parser.add_argument('--depth_list',
                        type=str,
                        default=[2, 3, 4, 5])
    args = parser.parse_args()
    update_config(cfg, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'test')

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    # build model
    model = eval('models.' + cfg.MODEL.NAME +
                 '.get_seg_model')(cfg)

    input_shape = (1, cfg.LESION.SLICE_NUM, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])

    dump_input = torch.rand(
        (1, cfg.LESION.SLICE_NUM, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
    )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if cfg.TEST.MODEL_FILE:
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')

    model.init_weights(model_state_file)
    # logger.info('=> loading model from {}'.format(model_state_file))

    # pretrained_dict = torch.load(model_state_file, map_location='cpu')
    # model_dict = model.state_dict()

    # model_keys = set(model_dict.keys())
    # pretrained_keys = set(pretrained_dict.keys())
    # missing_keys = model_keys - pretrained_keys
    # logger.warn('Missing keys in pretrained_dict: {}'.format(missing_keys))

    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)


    # prepare data
    test_roidb, test_ratio_list, test_ratio_index = combined_roidb_for_training(
            cfg.VAL.DATASETS, cfg.VAL.PROPOSAL_FILES)

    test_dataset = RoiDataLoader(
        test_roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)

    test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    if cfg.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                     thres=cfg.LOSS.OHEMTHRES,
                                     min_kept=cfg.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights.cuda())
    else:
        criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights.cuda())

    gpus = list(cfg.GPUS)
    model = FullModel(model, criterion)

    model = nn.DataParallel(model, device_ids=gpus).cuda()

    fusion_percentage = args.fusion_percentage
    depth_list = ast.literal_eval(args.depth_list)
    depth_list_str = ''.join(str(v) for v in depth_list)
    filepath = './eval_results/' + f'seg_{args.save_name}_f{fusion_percentage}_d{depth_list_str}.txt'

    for i in range(args.sample_num):
        seed = random.randint(0, 1000000)
        logger.info('Use fusion_percentage {}, depth_list {}, random seed {} '
                    .format(fusion_percentage, depth_list, seed))
        random.seed(seed)
        model.module.set_fusion_percentage(fusion_percentage)
        model.module.set_depth_list(depth_list)
        subnet_setting = model.module.sample_active_subnet()

        # logger.info(pprint.pformat(subnet_setting))
        save_mask_dir = f'{final_output_dir}_f{fusion_percentage}_d{depth_list_str}_{seed}'
        if not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
        mask_npy = os.path.join(save_mask_dir, 'mask.npy')
        np.save(mask_npy, subnet_setting)
        if args.bn_calib:
            calib_bn(cfg, model, seed)
        params, flops, details = get_model_summary(model.module.model, dump_input.cuda())
        logger.info('INFO: profile from scalenet_seg: Model {}: params {:.1f} M, flops {:.1f} G (with input size {})'.
                    format(cfg.MODEL.NAME, params / 1e6, flops, input_shape))

        start = timeit.default_timer()

        valid_loss, mean_IoU, IoU_array = validate(cfg,
                                                   testloader, model, writer_dict, device=None)

        msg = 'valid_loss: {: 4.4f} mean_IoU: {: 4.4f}, IoU_array: {}'.format(valid_loss, mean_IoU, IoU_array)
        logging.info(msg)
        logger.info('Model seed {}: params {:.2f} M, flops {:.2f} G AP {})'
                    .format(seed, params / 1e6, flops, mean_IoU))


        end = timeit.default_timer()
        logger.info('Mins: %d' % np.int((end - start) / 60))
        # subnet_setting_one_line = '; '.join(str(e) for e in subnet_setting)

        with open(filepath, "a") as f:
            val_log = 'Subnet with fusion_percentage {0} depth_list {1} seed {2}' \
                      ' Params(M) {3:.2f} FLOPs(G) {4:.2f}  AP {5:.4f}' \
                .format(fusion_percentage, depth_list, seed,
                        params / 1e6, flops, mean_IoU)
            f.write(val_log + '\n')


if __name__ == '__main__':
    main()