
# coding: utf-8

# This notebook is to load predicted mask for hard-disk and calculate Segmentation evaluation metics.
# predicted mask is the output of ScaleNASv2 Test and save predicted mask.ipynb
# 
# cp -v /mnt/fast-data/mjc/AutoRECIST/Codes/ScaleNAS/ScaleNASv1/tools/utils_test.py .
# 
# gt box are loaded from /cache/*gt_roidb.pkl
# gt segmentation are loaded from Hao's Raw files
# 
# predictions are all_boxes and all_segms which both are loaded from mask_1988 png files.




# # This file is for segmetation metrics evaluation in 3D
# Edited by Jingchen around 06/20/2021
# This file is after ScaleNAS test which save predition into png images.
# This file load png images as predicted contours in 2D
# load cache pkl as gold-standard contours in 2D
# The stack 2D based on dicom-header to get 3D
# Evaluate 3D metics of dice, IoU, over-segmetation, and under-segmetation
# 
get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import logging
import numpy as np
import _init_paths

from config import cfg
from config import update_config
from utils_test import *

import cv2
from PIL import Image
import torch.nn.functional as F
from utils.utils import get_confusion_matrix
def convert_name(name):
    new = name.replace('/','_')
    return new



site_list_liver_lung_LNs = [2,4,6,8,9,10,11,14,17] 
site_list_LNs = [2,4,6,10,11,14,17] 

site_list = site_list_liver_lung_LNs

cache_path = './cache/'
name = 'inference'
# name = 'lesion_train'

cache_filepath = os.path.join(cache_path, name+'_gt_roidb.pkl')
# print('Loading cached gt_roidb from %s', cache_filepath)
with open(cache_filepath, 'rb') as fp:
    cached_roidb = pickle.load(fp)
    
roidb = cached_roidb


from utils_test import __get_annotation__

sv_dir = mask_name = 'mask_1988'
sv_path = os.path.join(sv_dir, 'test_val_results')

all_boxes = [ [ np.zeros((0,5),dtype="float32") for _ in range(len(roidb)) ] for _ in range( cfg.DATASET.NUM_CLASSES) ]
all_segms = [ [ [] for _ in range(len(roidb)) ] for _ in range( cfg.DATASET.NUM_CLASSES) ]

for i in range(len(roidb)):

    one = roidb[i]
    onename = one['image']
    if not os.path.exists( os.path.join( sv_path, convert_name(onename) ) ):
        print(os.path.join(sv_path, convert_name(onename) ) , 'not exists!')
    pred_im = Image.open(os.path.join( sv_path, convert_name(onename) ))
    pred = np.array(pred_im)
    for j in range(cfg.DATASET.NUM_CLASSES):
        mask = np.asarray( pred==j , dtype=np.uint8)
        if np.sum(mask > 0) <= 3 :
            continue
        segmentation, bbox, area = __get_annotation__(mask , xywh = False , bbox_score=True)
        if segmentation and bbox:
            all_segms[j][i] = segmentation
            all_boxes[j][i] = bbox


D_CT = {}
for i , aroidb in enumerate(roidb):
    dicom_path , png_name = os.path.split(aroidb['image'])
    slice_no , _= os.path.splitext(png_name)
    slice_no = int(slice_no)
    if slice_no != aroidb['slice_no']:
        print('following slice numbers are not consistence.')
        print(dicom_path,slice_no,aroidb['slice_no'])

    segmentations = {}
    bboxes = {}
    for j in site_list:
        segmentations[j] = all_segms[j][i]
        bboxes[j] = all_boxes[j][i]

    if dicom_path not in D_CT:
        D_CT[dicom_path] = {}
        D_CT[dicom_path][slice_no] = [aroidb , bboxes , segmentations]
    else:
        D_CT[dicom_path][slice_no] = [aroidb , bboxes , segmentations]





SHOW_LABEL = True
SHOW_BOX = False
SHOW_MASK = True
SHOW_UNION_MASK= False
SHOW_MASK_LABEL = False
SHOW_GT_MASK = False
from utils_metrics_3d import *


def get_proper_CT_windowing(segmentations):

    if 8 in segmentations and segmentations[8] :
        return -160 , 240
    if 7 in segmentations and segmentations[7] :
        return -160 , 240    
    if 9 in segmentations and segmentations[9] :
        return -1250 , 250
    return None

keys = list(D_CT.keys())

def compute_colors_for_labels(i):
    D = {
        1 : [1, 127, 31] , # abdomen
        2 : [64, 255, 64] , #  abdomen LN
        3 : [255, 64, 255] , # adrenal
        4 : [64, 255, 64] , # axillary LN
        5 : [5, 125, 155] , # bone
        6 : [64, 255, 64] , #  inguinal LN
        7 : [7, 124, 217] , # kidney
        8 : [8, 251, 248] , # liver
        9 : [255, 64, 64] , # lung
        10 : [64, 255, 64] , # mediastinum LN
        11 : [64, 255, 64] , # neck LN
        12 : [12, 249, 117] , # ovary
        13 : [13, 121, 148] , # pancreas
        14 : [64, 255, 64] , # pelvic LN
        15 : [15, 120, 210] , # pelvis
        16 : [16, 247, 241] , # pleural
        17 : [64, 255, 64] , #  retroperitoneal LN
        18 : [18, 246, 48] , # soft tissue
        19 : [19, 118, 79] , # spleen
        20 : [20, 245, 110] , # stomach
        21 : [21, 117, 141] , # thyroid
        }
    return D[i]

for k in keys:
    if 'COU-AA-302' in k:
        site_list = site_list_LNs
    else:
        site_list = site_list_liver_lung_LNs
    
    oneCT = remove_single_slice_segms(D_CT[k])
    savepath = os.path.join( SAVE_PATH , 'CTs_%s/%s'%(name,convert_name_compact(k)) )

    print('Image results are saving into {}'.format(savepath))

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    for s in oneCT:
        aroidb , bboxes , segmentations = oneCT[s]

        image_path = os.path.join(aroidb['image'])
    
        CT_windowing = get_proper_CT_windowing(segmentations)
        if CT_windowing:
            HU1, HU2 = CT_windowing
        else:
            [HU1, HU2 ] = aroidb['windows']
            

        image = load_image(image_path, HU1, HU2)
        height,width = image.shape
    #     plt.imshow(image)
        image = np.dstack((image,image,image))
        
        if SHOW_MASK:
            for j in site_list:
                contours = segmentations[j]
                colors = compute_colors_for_labels(j)
                label = ix2labelname(j)
                for c in contours:
                    c = np.reshape(c,(-1,2))
                    if c.shape[0]:
                        image = cv2.drawContours(image, [np.int64( c  )], -1, colors, 1)
                        if SHOW_MASK_LABEL:
                            x,y,_,_ =ploy2boxes(c)
                            template = "{}"
                            if len(label)>=9:
                                label=label[:3]+label[-3:]
                
                            s = template.format(label)
                            cv2.putText(
                                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, colors, 1
                            )  
        

        cv2.imwrite( os.path.join(savepath, convert_name_compact( aroidb['image'] ) ) , image )





