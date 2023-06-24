# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os
import logging
import fcntl
import time

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
    parser.add_argument('--mask_path',
                        help='the path of a mask.npy',
                        default=None,
                        type=str)
    parser.add_argument('--evo_file',
                        help='evo_file has masks to be validated',
                        default='./evo_files/lesion_evo_file.txt',
                        type=str)
    parser.add_argument('--masks_dir',
                        help='dir of masks',
                        default='./evo_files/masks',
                        type=str)
    args = parser.parse_args()
    update_config(cfg, args)

    return args

def lockfile(f):
    try:
        fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except:
        return False


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'test')

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))
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

    gpus = list(cfg.GPUS)

    model = nn.DataParallel(model, device_ids=gpus).cuda()

    evo_file = args.evo_file
    index_line = -1
    while True:  # wait for new task LOOP-0
        while True:  # LOOP-1
            if not os.path.exists(evo_file):
                time.sleep(1)
                continue
            line_state = None
            with open(evo_file, "r+") as f:
                if not lockfile(f):
                    time.sleep(10)
                    logger.info('Waiting for loading...')
                    continue  # re-open
                lines = f.readlines()
                for lines_ind in range(0, len(lines)):  # update line_state LOOP-2
                    line = lines[lines_ind]
                    _line = line.split()
                    if len(_line) == 1:
                        mask_id = _line[0]
                        line_state = lines[lines_ind - 1]
                        lines[lines_ind] = f"{mask_id} Evaluating...\n"
                        f.seek(0, 0)
                        f.writelines(lines)
                        index_line = lines_ind
                        break
                if line_state is None:  # no arch
                    continue  # re-open LOOP-1
                else:
                    break  # LOOP-1
        BASE_DIR = args.masks_dir
        mask_path = os.path.join(f'{BASE_DIR}/{mask_id}.npy')

        if os.path.exists(mask_path):
            masks = np.load(mask_path, allow_pickle=True)
            model.module.set_active_subnet(masks)
            logger.info('=> setting mask from {}'.format(mask_path))
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

        while True:  # wait for file to write back
            with open(evo_file, "r+") as f:
                if not lockfile(f):
                    time.sleep(1)
                    logger.info('Waiting for writing...')
                    continue
                lines = f.readlines()
                lines[index_line] = f'{mask_id} {params / 1e6} {flops} {mean_IoU}\n'
                f.seek(0, 0)
                f.writelines(lines)
                break

        if len(lines) > 10000:
            break


if __name__ == '__main__':
    main()