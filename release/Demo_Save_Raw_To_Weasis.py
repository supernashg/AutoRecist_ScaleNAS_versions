
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
import cv2
from PIL import Image

import sys
sys.path.append('/mnt/fast-disk1/mjc/utils_codes/')
sys.path.append('/mnt/fast-disk1/mjc/utils_codes/read_weasis_raw_v0.96/')
import weasis_raw_data_api as wr
from utils_test import *
from utils_test import __get_annotation__
from utils_metrics_3d import *



def convert_name(name):
    new = name.replace('/','_')
    return new

HEIGHT , WIDTH = 512, 512
def get_pred_vol(oneCT , site_list , D_z_index, union_mask = True):
    slice_no_list =list ( oneCT.keys() )

    V = D_z_index.values()
    shape_z = np.max(list(V)) + 1
    vol_shape = (shape_z , HEIGHT , WIDTH )
    height = vol_shape[1]
    width = vol_shape[2]

    if len(slice_no_list):
        slice_no_list.sort()
        vol_gt = np.zeros(vol_shape, dtype = bool)
        vol_pred = np.zeros(vol_shape, dtype = bool)

        for s in slice_no_list:
            aroidb , bboxes , segmentations = oneCT[s]

            ix = [a for a,b in enumerate(aroidb['gt_classes']) if int(b) in site_list]
            contours = [ aroidb['segms'][int(kk)] for kk in ix ]
            
            for c in contours:
                if len(c): #gt
                    new = polys_to_mask(c , height , width)
                    vol_gt[D_z_index[s]][new>0] = 1 


            for j in site_list:
                contours = segmentations[j]
                if union_mask:
                    #contour should be numpy.array here. list cause error of no attribute 'flatten'
                    cc = [ contour.flatten().tolist() for contour in contours if len(contour)!=0]
                    contours = union_ploys(cc , height, width)

                for c in contours:#union pred
                    if len(c)>=6:
                        new = polys_to_mask([c] , height , width) 
                        vol_pred[D_z_index[s]][new>0] = 1 
                    elif len(c):
                        print('len pred contour is %d'%len(c))
    return vol_pred

from skimage import measure
def seperate_vol(vol_pred , reduceFP = False):
    # vol_dict = seperate_vol(vol_pred)
    connectivity = None #Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
                        #Accepted values are ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used.
    
    labels_pred=measure.label(vol_pred,connectivity=connectivity)
    l_pred,c_pred = np.unique(labels_pred , return_counts=True)


    ix2 = np.logical_and(l_pred,c_pred>=6) #commit by Jingchen to remove 'single dots' of AI outputs
    l_pred = l_pred[ix2] #background pixels are labeled as 0, so we exclude them
    c_pred = c_pred[ix2]

    if reduceFP:
        ix2 = l_pred>0
        for i, p in enumerate(l_pred):
            z = np.where(labels_pred == p)[0]
            if len( set(z) )<=1:
                ix2[i]=False

        l_pred = l_pred[ix2] #background pixels are labeled as 0, so we exclude them
        c_pred = c_pred[ix2]


    vol_dict = {}

    for p in l_pred:
        vp = labels_pred == p
        vol_dict[p] = vp

    return vol_dict

def saved_png_to_boxes_segms(roidb , sv_dir , NUM_CLASSES = cfg.DATASET.NUM_CLASSES):
    all_boxes = [ [ np.zeros((0,5),dtype="float32") for _ in range(len(roidb)) ] for _ in range( NUM_CLASSES) ]
    all_segms = [ [ [] for _ in range(len(roidb)) ] for _ in range( NUM_CLASSES) ]

    for i in range(len(roidb)):

        one = roidb[i]
        onename = one['image']
        if not os.path.exists( os.path.join( sv_path, convert_name(onename) ) ):
            print(os.path.join(sv_path, convert_name(onename) ) , 'not exists!')
        pred_im = Image.open(os.path.join( sv_path, convert_name(onename) ))
        pred = np.array(pred_im)
        for j in range(NUM_CLASSES):
            mask = np.asarray( pred==j , dtype=np.uint8)
            if np.sum(mask > 0) <= 3 :
                continue
            segmentation, bbox, area = __get_annotation__(mask , xywh = False , bbox_score=True)
            if segmentation and bbox:
                all_segms[j][i] = segmentation
                all_boxes[j][i] = bbox
    return all_boxes, all_segms


def boxes_segms_to_CTs(all_boxes,all_segms):
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
    return D_CT


site_list_liver_lung_LNs = [2,4,6,8,9,10,11,14,17] 
site_list_LNs = [2,4,6,10,11,14,17] 

site_list = [8]
user_id = 'jm4669'

cache_path = './cache/'
name = 'inference'
sv_dir = mask_name
sv_path = os.path.join(sv_dir, 'test_val_results')

SAVE_PATH = '/mnt/fast-disk1/refine_gt/'
SAVE_NAME = 'ScaleNAS3Slices_525pts'

cache_filepath = os.path.join(cache_path, name+'_gt_roidb.pkl')
print('Loading cached gt_roidb from %s', cache_filepath)
with open(cache_filepath, 'rb') as fp:
    roidb = pickle.load(fp)
    

all_boxes, all_segms = saved_png_to_boxes_segms(roidb , sv_dir , cfg.DATASET.NUM_CLASSES)
D_CT = boxes_segms_to_CTs(all_boxes, all_segms)



Metrics_vol = []
keys = list(D_CT.keys())
for k in keys:

    image_series_path = k.replace('/Pngs/' , '/Inputs/')    

    df_image = get_dicom_header_df( image_series_path )
    instanceNumber_list = df_image.index.to_list()
    D_z_index = instanceNumber2Matrix_z_index(instanceNumber_list)


    oneCT = remove_single_slice_segms(D_CT[k])
    vol_pred = get_pred_vol(oneCT , site_list , D_z_index, union_mask = False)
    



    image_series = wr.dicom_header(image_series_path)
    if len(image_series):
        height = image_series[0].Rows 
        width = image_series[0].Columns 
    else:
        print('ERROR image_series has no len' , image_series_path)
    assert(len(image_series) == vol_pred.shape[0])

    vol2 = vol_pred
    vol2[vol2>1]=1
    vol_dict = seperate_vol(vol2)
    
    for tumor_index in vol_dict:
        mask_volume = vol_dict[tumor_index]
        weasis_raw_data = wr.create(image_series, mask_volume)

        file_folder = os.path.join(SAVE_PATH , SAVE_NAME)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        file_name = wr.unique(image_series, tumor_index, user_id)
        file_name = os.path.join(file_folder,file_name)
        wr.write(weasis_raw_data, file_name)

        Metrics_vol.append( [image_series_path , file_name , user_id ])

    print('finished of ', k )


df_metrics = pd.DataFrame(Metrics_vol, 
                          columns = ['Image File Path','Contour File Path','Uni']) 
df_metrics.to_csv('%s.csv'%SAVE_NAME , index=False)



