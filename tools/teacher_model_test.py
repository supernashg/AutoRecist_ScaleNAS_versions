
# coding: utf-8


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



# In[4]:

# arglist = ['--cfg', '../experiments/cityscapes/scalenet_seg_w32_test.yaml' ,  
#            '--bn_calib',
#            '--mask_path', '../experiments/searched_masks/cityscapes/seg_w32_S1.npy' ,  
#            'TEST.MODEL_FILE', '../models/pytorch/seg_cityscapes/superscalenet_seg_w32.pth',
#            'DATASET.ROOT','../data/']

experiment_name = 'Lesion_train_teacher_model_lr1e-4_bs16x4_slice3_WL1'
arglist = ['--cfg', '../experiments/lesion/%s.yaml'%experiment_name ,  
           '--bn_calib',
           'TEST.MODEL_FILE', '../output/Lesion/superscalenet_seg/%s/data_patch_train/best.pth'%experiment_name,
           'DATASET.ROOT','../data/']

args = parse_args(arglist)





# In[6]:

('PDS_AMGEN_20020408_22Cat_test',)
('PDS_Q2_A&C_22Cat_train',)



logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'valtest')

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
    input_shape
)

if cfg.TEST.MODEL_FILE:
    model_state_file = cfg.TEST.MODEL_FILE
else:
    model_state_file = os.path.join(final_output_dir,
                                    'final_state.pth')
logger.info('=> loading model from {}'.format(model_state_file))

pretrained_dict = torch.load(model_state_file)

D2= {}
for key in pretrained_dict.keys():
    if key[:6] == 'model.':
        new_key = key[6:]
        D2[new_key] = pretrained_dict[key]
    else:
        print(key)
        D2[key] = pretrained_dict[key]

pretrained_dict = D2      
model_dict = model.state_dict()

model_keys = set(model_dict.keys())
pretrained_keys = set(pretrained_dict.keys())
missing_keys = model_keys - pretrained_keys
logger.warn('Missing keys in pretrained_dict: {}'.format(missing_keys))

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)


# prepare data
test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])




import cv2
from PIL import Image
import torch.nn.functional as F
from utils.utils import get_confusion_matrix
def convert_name(name):
    new = name.replace('/','_')
    return new

def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
    
# def save_pred(preds, sv_path, name):

#     preds = preds.cpu().numpy().copy()
#     preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
#     for i in range(preds.shape[0]):
#         cv2.imwrite(os.path.join(sv_path, convert_name(name[i])) , preds[i])

        
def save_pred( preds, sv_path, name):
    palette = get_palette(256)
    preds = preds.cpu().numpy().copy()
    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    for i in range(preds.shape[0]):
        pred = preds[i]
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, convert_name(name[i]) ))
        

def testval_lesion(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=True, device = None):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            size = label.size()
            if device is None:
                image = image.cuda()
                label = label.long().cuda()
            else:
                image = image.to(device)
                label = label.long().to(device)

            pred = model(image)
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]),
                                  mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


# In[16]:

from dataset.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch

test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])

test_roidb, test_ratio_list, test_ratio_index = combined_roidb_for_training(
        ('PDS_AMGEN_20020408_22Cat_test',) , cfg.VAL.PROPOSAL_FILES)

test_dataset = RoiDataLoader(
    test_roidb,
    cfg.MODEL.NUM_CLASSES,
    training=True)


testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
    shuffle=False,
    num_workers=cfg.WORKERS,
    pin_memory=True,
    sampler=None)



gpus = list(cfg.GPUS)
logger.info('GPU list is {}'.format(gpus))

# model = nn.DataParallel(model, device_ids=gpus).cuda()

# if args.mask_path and os.path.exists(args.mask_path):
#     masks = np.load(args.mask_path, allow_pickle=True)
#     model.module.set_active_subnet(masks)
#     logger.info('=> setting mask from {}'.format(args.mask_path))
#     logger.info(masks)
# else:
#     masks=None
#     logger.info('No model mask')

model.set_fusion_percentage(1.0)
depth = cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS[0]
model.set_depth_list([depth])
subnet_setting = model.sample_active_subnet()

# if args.bn_calib:
#     calib_bn(cfg, model, 0, masks)
# params, flops, details = get_model_summary(model.module, dump_input.cuda())
# logger.info('INFO: profile from hrnet: Model {}: params {:.1f} M, flops {:.1f} G (with input size {})'.
#             format(cfg.MODEL.NAME, params / 1e6, flops, input_shape))
           


# In[30]:


mean_IoU, IoU_array, pixel_acc, mean_acc = testval_lesion(cfg, 
                                                  test_dataset, 
                                                  testloader, 
                                                  model.cuda(),
                                                  device=None)

msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f},    Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
   pixel_acc, mean_acc)
logging.info(msg)
logging.info(IoU_array)

