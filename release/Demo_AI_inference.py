
# coding: utf-8

# In[1]:

get_ipython().system('rm ./cache/inference_gt_roidb.pkl')


# In[2]:



# In[1]:

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

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
t = time.time()


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
                    os.makedirs(sv_path)
                save_pred(pred, sv_path, name)

            if index % 100 == 0:
                # logging.info('processing: %d batches' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                # logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


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




experiment_name = 'Lesion_Q5_scalenet_seg_test'
mask_name = 'mask_1514'
arglist = ['--cfg', '../experiments/lesion_Q5/%s.yaml'%experiment_name ,  
           '--mask_path', '../evo_files/masks/%s.npy'%mask_name,  
           'TEST.MODEL_FILE', '../output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth',
           'DATASET.ROOT','',
           'TRAIN.USE_FLIPPED',False]

args = parse_args(arglist)

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
# logger.warn('Missing keys in pretrained_dict: {}'.format(missing_keys))

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)


# do stuff
elapsed = time.time() - t
print('Current time cost is {} sec'.format(elapsed) )


# prepare data
test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])

# manully select from below.
# ('PDS_AMGEN_20020408_22Cat_test',)
# ('PDS_Q2_A&C_22Cat_train',)
# ('PDS_CUIMC_22Cat_test',)

test_roidb, test_ratio_list, test_ratio_index = combined_roidb_for_training(
        ('inference',) , cfg.VAL.PROPOSAL_FILES)

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
# logger.info('GPU list is {}'.format(gpus))

model = nn.DataParallel(model, device_ids=gpus).cuda()

if args.mask_path and os.path.exists(args.mask_path):
    masks = np.load(args.mask_path, allow_pickle=True)
    model.module.set_active_subnet(masks)
    # logger.info('=> setting mask from {}'.format(args.mask_path))
    # logger.info(masks)
else:
    masks=None
    logger.info('No model mask')


           



mean_IoU, IoU_array, pixel_acc, mean_acc = testval_lesion(cfg, 
                                                  test_dataset, 
                                                  testloader, 
                                                  model.cuda(),
                                                  sv_dir=mask_name, 
                                                  device=None)

# msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f},    Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
#    pixel_acc, mean_acc)
# logging.info(msg)
# logging.info(IoU_array)

# do stuff
elapsed = time.time() - t
print('Current time cost is {} sec'.format(elapsed) )





