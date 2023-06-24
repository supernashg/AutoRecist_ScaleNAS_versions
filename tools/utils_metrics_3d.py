
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import collections
# from tqdm import tqdm_notebook as tqdm
from datetime import datetime

from math import ceil, floor
import cv2
import sys
from utils_test import polys_to_mask,mask_to_ploys, union_ploys, polys_to_boxes, ploy2boxes ,union_ploys_to_mask
# from sklearn.model_selection import ShuffleSplit

def window_image(img, window_center,window_width, intercept, slope):
    
#     window_center,window_width = 50 ,100
    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)-1
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return np.zeros(img.shape)-1
    return 2*(img - mi) / (ma - mi) - 1

def getName(s):
    ix1 = s.rfind('/')
    ix2 = s.rfind('.')
    return s[ix1:ix2]


def _read(path, desired_size = (512,512)):
    """Will be used in DataGenerator"""

    try:
        data = pydicom.read_file(path)
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        
        image_windowed = window_image(image, window_center, window_width, intercept, slope)
        img = normalize_minmax(image_windowed)

    except:
        img = np.zeros(desired_size[:2])-1
    
    if img.shape[:2] != desired_size[:2]:
        print("image shape is not desired size. Interpolation is done.")
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    
    
    return img





    

def InstanceNumber2file_name(df_image, num):
    return df_image.loc[num,'ImageName']

def InstanceNumber2data_element(df_image, num, label):
    return df_image.loc[num , label]

def get_SliceThickness(df_image):
    L = df_image['ImagePositionPatient_2'].tolist()
    L.sort()
    thick = list( np.diff(L) )
    return float( max(set(thick), key=thick.count) )

def InstanceNumber2windows_min_max(df_image,num):
    WL = InstanceNumber2data_element(df_image,num,'WindowCenter')
    WW = InstanceNumber2data_element(df_image,num,'WindowWidth')
    minHU = int( WL-WW/2 )
    maxHU = minHU + int(WW)
    return [minHU , maxHU]


class ASerial:
    P=-1
    D=-1
    S=-1
    name = ''
    def __init__(self, path_str):
        self.path = path_str
        self.getP()
        self.getD()
        self.getS()
        self.convert_path()
        
    def getP(self, target = 'DeepLesion_', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.P = int(ss)
        
    def getD(self, target = '/D', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.D = int(ss)
        
    def getS(self, target = '/S', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.S = int(ss)
        
    def convert_path(self):
        self.name = '%06d_%02d_%02d'%(self.P, self.D, self.S)


def instanceNumber2Matrix_z_index(instanceNumber_list):
    # D_z_index = instanceNumber2Matrix_z_index(instanceNumber_list)
    instanceNumber_list.sort()
    D_z_index = {}
    for k , num in enumerate(instanceNumber_list):
        D_z_index[num] = k
    return D_z_index

def initialize_mask_vol(weasis_raw_data , D_z_index):
    V = D_z_index.values()
    shape_z = np.max(list(V)) + 1
    mask_vol = np.zeros((shape_z , weasis_raw_data.height , weasis_raw_data.width ) , dtype=np.uint8 )
    return mask_vol

def get_gt_and_pred_vols(oneCT, site_list , vol_shape , D_z_index, union_mask=True):
    slice_no_list =list ( oneCT.keys() )
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
            if not contours:
                continue

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
    return vol_gt, vol_pred

def cal_seg_metrics( vol_gt , vol_pred):
    intersection = np.sum( np.logical_and(vol_gt,vol_pred) )
    union = np.sum( np.logical_or(vol_gt,vol_pred) )
    area_pred = np.sum( vol_pred )
    area_gt = np.sum(vol_gt)
    iou_score = intersection / union
    dice_score = 2*intersection /(area_pred+area_gt)
    over_seg = (area_pred - intersection) / area_gt
    under_seg = (area_gt - intersection) / area_gt
    return [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union]
'''
# def vols_seg_results(vol_gt , vol_pred , CTname='abc' , gt_keep_largest=None , pred_keep_largest=None):

#     connectivity = 2
#     from skimage import measure,color
#     labels_gt=measure.label(vol_gt,connectivity=connectivity)
#     l_gt,c_gt = np.unique(labels_gt , return_counts=True)
#     labels_pred=measure.label(vol_pred,connectivity=connectivity)
#     l_pred,c_pred = np.unique(labels_pred , return_counts=True)

#     ix = l_gt>0
#     l_gt = l_gt[ix] #background pixels are labeled as 0, so we exclude them
#     c_gt = c_gt[ix]

#     ix2 = l_pred>0
#     l_pred = l_pred[ix2] #background pixels are labeled as 0, so we exclude them
#     c_pred = c_pred[ix2]

#     if gt_keep_largest:
#         l_gt_ix = c_gt.argsort()[-gt_keep_largest:][::-1]
#         l_gt = l_gt[l_gt_ix]

#     if pred_keep_largest:
#         l_pred_ix = c_pred.argsort()[-pred_keep_largest:][::-1]
#         l_pred = l_pred[l_pred_ix]


#     Metrics = []
#     MissedLesions = []
#     for g in l_gt:
#         vg = labels_gt == g
#         VPs = np.zeros(labels_pred.shape,dtype=bool)
#         merge = 0
#         for p in l_pred:
#             vp = labels_pred == p
#             intersection1 = np.sum( np.logical_and(vg, vp) )
#             if intersection1 > 0:
#                 VPs = np.logical_or(VPs, vp)
#                 merge+=1
                
#         [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = cal_seg_metrics(vg , VPs)
#         if intersection>0:
#             Metrics.append( [ CTname, g, merge,len(l_gt),len(l_pred),
#                     iou_score, dice_score, over_seg, under_seg, 
#                     area_gt, area_pred, intersection, union] )
#         else:
#             MissedLesions.append( [ CTname, g, merge,len(l_gt),len(l_pred),
#                     iou_score, dice_score, over_seg, under_seg, 
#                     area_gt, area_pred, intersection, union] )
            

#     return Metrics, MissedLesions
'''

def vols_seg_results(vol_gt , vol_pred , CTname='abc' , gt_keep_largest=None , pred_keep_largest=None , pred_keep_best=True):
    ## This version add 

    # print('vols_seg_results is New Version that include both hit and miss cases')

    connectivity = 2
    from skimage import measure,color
    labels_gt=measure.label(vol_gt,connectivity=connectivity)
    l_gt,c_gt = np.unique(labels_gt , return_counts=True)
    labels_pred=measure.label(vol_pred,connectivity=connectivity)
    l_pred,c_pred = np.unique(labels_pred , return_counts=True)

    ix = l_gt>0
    l_gt = l_gt[ix] #background pixels are labeled as 0, so we exclude them
    c_gt = c_gt[ix]

    ix2 = l_pred>0
    l_pred = l_pred[ix2] #background pixels are labeled as 0, so we exclude them
    c_pred = c_pred[ix2]

    if gt_keep_largest:
        l_gt_ix = c_gt.argsort()[-gt_keep_largest:][::-1]
        l_gt = l_gt[l_gt_ix]

    if pred_keep_largest:
        l_pred_ix = c_pred.argsort()[-pred_keep_largest:][::-1]
        l_pred = l_pred[l_pred_ix]

    Metrics = []
    if pred_keep_best:
        merge=0

        for g in l_gt:
            vg = labels_gt == g

            vp_zeros = np.zeros(labels_pred.shape,dtype=bool)
            [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = cal_seg_metrics(vg , vp_zeros)

            best_dice = 0
            best_vp_results = [ CTname, g, merge,len(l_gt),len(l_pred),
                                    iou_score, dice_score, over_seg, under_seg, 
                                    area_gt, area_pred, intersection, union]

            for p in l_pred:
                vp = labels_pred == p
                [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = cal_seg_metrics(vg , vp)
                if dice_score > best_dice:
                    best_vp_results = [ CTname, g, merge,len(l_gt),len(l_pred),
                                    iou_score, dice_score, over_seg, under_seg, 
                                    area_gt, area_pred, intersection, union]
                    best_dice = dice_score

            Metrics.append( best_vp_results )


    else:
        for g in l_gt:
            vg = labels_gt == g
            VPs = np.zeros(labels_pred.shape,dtype=bool)
            merge = 0
            for p in l_pred:
                vp = labels_pred == p
                intersection1 = np.sum( np.logical_and(vg, vp) )
                if intersection1 > 0:
                    VPs = np.logical_or(VPs, vp)
                    merge+=1
                    
            [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = cal_seg_metrics(vg , VPs)
            
            Metrics.append( [ CTname, g, merge,len(l_gt),len(l_pred),
                    iou_score, dice_score, over_seg, under_seg, 
                    area_gt, area_pred, intersection, union] )
            
    return Metrics



#done 07/19 5:25 JM

