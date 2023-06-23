# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os

import logging
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
from core.seg_function import validate_seg_wo_loss as validate
from core.oneshot_function import calib_bn_seg as calib_bn
from utils.utils import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

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
    parser.add_argument('--mask_path',
                        help='the path of a mask.npy',
                        default=None,
                        type=str)
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

    input_shape = (1, 3, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])

    dump_input = torch.rand(
        (1, 3, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
    )

    if cfg.TEST.MODEL_FILE:
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()

    model_keys = set(model_dict.keys())
    pretrained_keys = set(pretrained_dict.keys())
    missing_keys = model_keys - pretrained_keys
    logger.warn('Missing keys in pretrained_dict: {}'.format(missing_keys))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)


    # prepare data
    test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
    test_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        root=cfg.DATASET.ROOT,
        list_path=cfg.DATASET.TEST_SET,
        num_samples=None,
        num_classes=cfg.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=cfg.TRAIN.IGNORE_LABEL,
        base_size=cfg.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    gpus = list(cfg.GPUS)

    model = nn.DataParallel(model, device_ids=gpus).cuda()

    if os.path.exists(args.mask_path):
        masks = np.load(args.mask_path, allow_pickle=True)
        model.module.set_active_subnet(masks)
        logger.info('=> setting mask from {}'.format(args.mask_path))
        logger.info(masks)
    else:
        raise NotImplementedError

    if args.bn_calib:
        calib_bn(cfg, model, 0, masks)
    params, flops, details = get_model_summary(model.module, dump_input.cuda())
    logger.info('INFO: profile from scalenet_seg: Model {}: params {:.1f} M, flops {:.1f} G (with input size {})'.
                format(cfg.MODEL.NAME, params / 1e6, flops, input_shape))

    mean_IoU, IoU_array = validate(cfg, testloader, model, writer_dict, device=None)

    msg = 'mean_IoU: {: 4.4f}, IoU_array: {}'.format(mean_IoU, IoU_array)
    logging.info(msg)



if __name__ == '__main__':
    main()