
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

from dataset.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader

from PIL import Image
import torch.nn.functional as F
from utils.utils import get_confusion_matrix

import time

def parse_args(l):
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
    args = parser.parse_args(l)
    update_config(cfg, args)

    return args

experiment_name = 'Lesion_Q5_9Slices_scalenet_seg_test'
mask_name = 'f0.5_d34_215856.npy'
arglist = ['--cfg', '../experiments/lesion_Q5/%s.yaml'%experiment_name ,  
           '--mask_path', '../evo_files/masks/%s.npy'%mask_name,  
           'TEST.MODEL_FILE', '../outputlog/Lesion/superscalenet_seg/Lesion_Q5_9Slices_superscalenet_1/data_patch_train/best.pth',
           'DATASET.ROOT','',
           'TRAIN.USE_FLIPPED',False]


args = parse_args(arglist)

model = eval('models.' + cfg.MODEL.NAME +
             '.get_seg_model')(cfg)


if cfg.TEST.MODEL_FILE:
    model_state_file = cfg.TEST.MODEL_FILE
else:
    raise NotImplementedError
    model_state_file = os.path.join(final_output_dir,
                                    'final_state.pth')
# logger.info('=> loading model from {}'.format(model_state_file))

pretrained_dict = torch.load(model_state_file)

D2= {}
for key in pretrained_dict.keys():
    if key[:6] == 'model.':
        new_key = key[6:]
        D2[new_key] = pretrained_dict[key]
    else:
        # print(key)
        D2[key] = pretrained_dict[key]

pretrained_dict = D2      
model_dict = model.state_dict()

model_keys = set(model_dict.keys())
pretrained_keys = set(pretrained_dict.keys())
missing_keys = model_keys - pretrained_keys

print('Missing keys in pretrained_dict: {}'.format(missing_keys))

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)


torch.save(model.state_dict(), 'best_resave_no_module.pth', _use_new_zipfile_serialization=False)