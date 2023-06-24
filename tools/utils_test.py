from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os.path as osp
import sys

import json
import logging
import numpy as np
import os
import uuid
import pdb

# from utils.myio import save_object
import utils.boxes as box_utils
from scipy import misc
import cv2
# from utils.myio import read_json

from scipy import interpolate
from six.moves import cPickle as pickle

from config import cfg

from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

def __get_annotation__(mask, image=None , xywh = True, bbox_score=False):
    if cv2.__version__[0] =='3':
        _,contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif cv2.__version__[0] =='4':
        contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        print('cv2 version is not support.')

    segmentations = []
    bboxes = []
    area = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        contour_size = contour.size
        if  contour_size >= 6:
            segmentations.append(contour.flatten().tolist())
            [x, y, w, h] = cv2.boundingRect(contour)
            if xywh:
                bbox = [x, y, w, h]
            else:
                bbox = [x, y, x+w-1, y+h-1]

            if bbox_score:
                bbox = bbox+[1.]
            bboxes.append( bbox )
            # if contour_size/2 <= 10:
            #     print( 'only %d points'%(contour_size/2) )
        else:
            # print( 'contour is too small: only %d points may cause errors'%contour_size )
            pass
    if len(segmentations)==0:
        return None, None , None

    # RLEs = cocomask.frPyObjects(segmentations, mask.shape[0], mask.shape[1])
    # RLE = cocomask.merge(RLEs)
    # # RLE = cocomask.encode(np.asfortranarray(mask))
    # area = cocomask.area(RLE)
    # [x, y, w, h] = cv2.boundingRect(mask)

    return segmentations, bboxes, int(area)


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range = None):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    roidb, dataset, start, end, total_num_images = \
    get_roidb_and_dataset(dataset_name, proposal_file, ind_range = None)
    """
    from dataset.json_dataset import JsonDataset

    if cfg.DATA_SOURCE == 'coco':
        dataset = JsonDataset(dataset_name)
    elif cfg.DATA_SOURCE == 'mammo':
        dataset = MammoDataset(dataset_name)
    elif cfg.DATA_SOURCE == 'lesion':
        dataset = LesionDataset(dataset_name)
    if proposal_file and cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        if cfg.DATA_SOURCE == 'coco':
            roidb = dataset.get_roidb(gt=True)
        elif cfg.DATA_SOURCE == 'mammo':
            roidb = dataset.get_roidb(
                gt=True,
                proposal_file='',
                crowd_filter_thresh=0)
        #elif cfg.DATA_SOURCE == 'lesion':
        #    roidb = dataset.get_roidb(
        #       gt=True)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end
        
    for one in roidb:
        one['need_crop'] = False
    return roidb, dataset, start, end, total_num_images

def if_overlap(predict, label, cutoff=.1):
    x1, y1, w1, h1 = predict
    x2, y2, w2, h2 = label
    predict_area = w1 * h1
    roi_area = w2 * h2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx > 0 and dy > 0:
        inter_area = dx * dy
    else:
        return False
    return inter_area * 1.0/roi_area > cutoff or inter_area * 1.0/predict_area > cutoff

# ========================================
# all below added by lizihao, for eval_FROC()
def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps


def num_true_positive(boxes, gts, num_box, iou_th):
    # only count once if one gt is hit multiple times
    hit = np.zeros((gts.shape[0],), dtype=np.bool)
    scores = boxes[:, -1]
    boxes = boxes[scores.argsort()[::-1], :4]

    for i, box1 in enumerate(boxes):
        if i == num_box: break
        overlaps = IOU(box1, gts)
        hit = np.logical_or(hit, overlaps >= iou_th)

    tp = np.count_nonzero(hit)

    return tp


def recall_all(boxes_all, gts_all, num_box, iou_th):
    # Compute the recall at num_box candidates per image
    nCls = len(boxes_all)
    nImg = len(boxes_all[0])
    recs = np.zeros((nCls, len(num_box)))
    nGt = np.zeros((nCls,), dtype=np.float)

    for cls in range(nCls):
        for i in range(nImg):
            nGt[cls] += gts_all[cls][i].shape[0]
            for n in range(len(num_box)):
                tp = num_true_positive(boxes_all[cls][i], gts_all[cls][i], num_box[n], iou_th)
                recs[cls, n] += tp
    recs /= nGt
    return recs


def FROC(boxes_all, gts_all, iou_th):
    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    # img_idxs_ori : array([   0.,    0.,    0., ..., 4830., 4830., 4830.])
    img_idxs_ori = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs_ori[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    no_lesion = 0
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if overlaps.shape[0] == 0:
            no_lesion += 1
            nMiss += 1
            if 0 and no_lesion<=10: #debug
                print(overlaps)
                print(boxes_cat[i, :])
                print(gts_all[img_idxs[i]])
                print('-'*40)
                
        elif overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    print('FROC:FP in no-lesion-images: ', no_lesion)
    return sens, fp_per_img, np.sort(scores)[::-1]

def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
    # compute the sensitivity at avgFP (average FP per image)
    sens, fp_per_img, _ = FROC(boxes_all, gts_all, iou_th)
    max_fp = fp_per_img[-1]
    f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
    if(avgFP[-1] < max_fp):
        valid_avgFP_end_idx = len(avgFP)
    else:
        valid_avgFP_end_idx = np.argwhere(np.array(avgFP) > max_fp)[0][0]
    valid_avgFP = np.hstack((avgFP[:valid_avgFP_end_idx], max_fp))
    res = f(valid_avgFP)
    return res,valid_avgFP



def get_gt_boxes(roidb, cls = None):
    gt_boxes = [[] for _ in range(len(roidb))]
    for i, entry in enumerate(roidb):
        if cls is None:

            gt_boxes[i] = roidb[i]['boxes']
        else:
            ix = np.where(roidb[i]['gt_classes'] == cls)
            gt_boxes[i] = roidb[i]['boxes'][ix]
    return gt_boxes

def get_gt_segms(roidb):
    gt = [[] for _ in range(len(roidb))]
    for i, entry in enumerate(roidb):
        if cls is None:
            gt[i] = roidb[i]['segms']
        else:
            ix = np.where(roidb[i]['gt_classes'] == cls)
            gt[i] = [ roidb[i]['segms'][j] for j in ix ]
    return gt

def eval_FROC(dataset, all_boxes, avgFP=[0.5,1,2,3,4,8,16,32,64], iou_th=0.5):
    # all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    # only one class for lesion dataset.
    # all_boxes[1][image] = N X 5

    #dataset = JsonDataset(dataset)
    roidb = dataset.get_roidb(gt = True)
    gt_boxes = get_gt_boxes(roidb)

    result, valid_avgFP = sens_at_FP(all_boxes[1], gt_boxes, avgFP, iou_th)
    print('='*40)
    for recall,fp in zip(result,valid_avgFP):
        print('Recall@%.1f=%.2f%%' % (fp, recall*100))
    #TODO: when num of valid_avgFP < 6,is FROC correct?
    print('Mean FROC is %.2f'% np.mean(np.array(result[:6])*100))
    print('='*40)


import cv2
def load_slices(dir, slice_idxs):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    for slice_idx in slice_idxs:
        fn = '%03d.png' % slice_idx
        path = os.path.join(dir_in, dir, fn)
        im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, 'error reading %s' % path

        # the 16-bit png file has a intensity bias of 32768
        ims.append((im.astype(np.int32) - 32768).astype(np.int16))
    return ims

def windowing(im,HU1=-500,HU2=500):
    im = np.float32(im)
    im[im>HU2]=HU2
    im[im<HU1]=HU1
    im = (im-HU1)/(HU2-HU1)
    return im
def window_image(img, window_center,window_width, intercept, slope):
    
#     window_center,window_width = 50 ,100
    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def load_imread(path , HU1=-500,HU2=500 ):
    im = cv2.imread(path,-1)
    assert im is not None, 'error reading %s' % path
    im = (im.astype(np.int32) - 32768).astype(np.int16)
    im = windowing(im,HU1,HU2)
    im = (im*255).astype(np.uint8)
    return im


def load_image(path , HU1=-500,HU2=500 ):
    if '.dcm' in path:
        print('dicom load')
        im = dicom_read(path)
        im = ( (im+1)*255/2 ).astype(np.uint8)
    else:
        im = load_imread(path, HU1 , HU2)
    return im

def compute_colors_for_labels(i):
    palette = np.array( [2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1] )
    colors = i * palette
    colors = (colors % 255).astype("uint8").tolist()
    return colors


newcats = [{'supercategory': 'DeepLesion', 'id': 1, 'name': 'abdomen'},
       {'supercategory': 'DeepLN', 'id': 2, 'name': 'abdomen LN'},
       {'supercategory': 'DeepLesion', 'id': 3, 'name': 'adrenal'},
       {'supercategory': 'DeepLN', 'id': 4, 'name': 'axillary LN'},
       {'supercategory': 'DeepLesion', 'id': 5, 'name': 'bone'},
       {'supercategory': 'DeepLN', 'id': 6, 'name': 'inguinal LN'},
       {'supercategory': 'DeepLesion', 'id': 7, 'name': 'kidney'},
       {'supercategory': 'DeepLesion', 'id': 8, 'name': 'liver'},
       {'supercategory': 'DeepLesion', 'id': 9, 'name': 'lung'},
       {'supercategory': 'DeepLN', 'id': 10, 'name': 'mediastinum LN'},
       {'supercategory': 'DeepLN', 'id': 11, 'name': 'neck LN'},
       {'supercategory': 'DeepLesion', 'id': 12, 'name': 'ovary'},
       {'supercategory': 'DeepLesion', 'id': 13, 'name': 'pancreas'},
       {'supercategory': 'DeepLN', 'id': 14, 'name': 'pelvic LN'},
       {'supercategory': 'DeepLesion', 'id': 15, 'name': 'pelvis'},
       {'supercategory': 'DeepLesion', 'id': 16, 'name': 'pleural'},
       {'supercategory': 'DeepLN', 'id': 17, 'name': 'retroperitoneal LN'},
       {'supercategory': 'DeepLesion', 'id': 18, 'name': 'soft tissue'},
       {'supercategory': 'DeepLesion', 'id': 19, 'name': 'spleen'},
       {'supercategory': 'DeepLesion', 'id': 20, 'name': 'stomach'},
       {'supercategory': 'DeepLesion', 'id': 21, 'name': 'thyroid'} ]
D_cls = {}
for d in newcats:
    id_ = d['id']
    name_ = d['name']
    D_cls[id_] = name_
def ix2labelname(ix):
    return D_cls[ix]

def overlay_class_names(self, image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [self.CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image

import pycocotools.mask as mask_util
def polys_to_mask(polygons, height, width):
    """
    Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask
def mask_to_ploys(mask):
    # height,width = mask.shape
    contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        contour_size = contour.size
        if  contour_size >= 0:
            segmentation.append(contour.flatten().tolist())
            if contour_size/2 <= 4:
                print( 'only %d points'%(contour_size/2) )
        else:
            print( 'contour is too small: only %d points may cause errors'%contour_size )
    return segmentation
def union_ploys(ploys , height=512 , width=512):
    assert(type(ploys) == list)
    if len(ploys) == 0:
        return ploys
    RLEs = mask_util.frPyObjects(ploys, height , width)
    RLE = mask_util.merge(RLEs)
    mask = mask_util.decode(RLE)
    new_ploys = mask_to_ploys(mask)
    return new_ploys

def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys

def ploy2boxes(p):
    x0 = np.min(p[:,0])
    x1 = np.max(p[:,0])
    y0 = np.min(p[:,1])
    y1 = np.max(p[:,1])
    return [x0,y0,x1,y1]


def union_ploys_to_mask(ploys , height=512 , width=512):
    assert(type(ploys) == list)
    if len(ploys) == 0:
        return ploys
    RLEs = mask_util.frPyObjects(ploys, height , width)
    RLE = mask_util.merge(RLEs)
    mask = np.array(mask_util.decode(RLE), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask

# def get_gt_and_pred_vols(oneCT,site_list):
#     slice_no_list =list ( oneCT.keys() )


#     if len(slice_no_list):
#         slice_no_list.sort()
#         slice_no_0 = slice_no_list[0]


#         height = oneCT[slice_no_0][0]['height']
#         width = oneCT[slice_no_0][0]['width']
#         vol_shape = (slice_no_list[-1] - slice_no_0 +1 , height , width )
#         vol_gt = np.zeros(vol_shape, dtype = bool)
#         vol_pred = np.zeros(vol_shape, dtype = bool)

#         for s in slice_no_list:
#             aroidb , bboxes , segmentations = oneCT[s]

#             ix = [a for a,b in enumerate(aroidb['gt_classes']) if int(b) in site_list]
#             contours = [ aroidb['segms'][int(kk)] for kk in ix ]
#             if not contours:
#                 continue

#             for c in contours:
#                 if len(c): #gt
#                     vol_gt[s - slice_no_0] = polys_to_mask(c , height , width) 


#             for j in site_list:
#                 contours = segmentations[j]
#                 cc = [ contour.flatten().tolist() for contour in contours if len(contour)!=0]
#                 contours = union_ploys(cc , height, width)

#                 for c in contours:#union pred
#                     if len(c)>=6:
#                         vol_pred[s - slice_no_0] = polys_to_mask([c] , height , width) 
#                     elif len(c):
#                         print('len pred contour is %d'%len(c))
#     return vol_gt, vol_pred



# def cal_seg_metrics( vol_gt , vol_pred):
#     intersection = np.sum( np.logical_and(vol_gt,vol_pred) )
#     union = np.sum( np.logical_or(vol_gt,vol_pred) )
#     area_pred = np.sum( vol_pred )
#     area_gt = np.sum(vol_gt)
#     iou_score = intersection / union
#     dice_score = 2*intersection /(area_pred+area_gt)
#     over_seg = (area_pred - intersection) / area_gt
#     under_seg = (area_gt - intersection) / area_gt
#     return [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union]


# def vols_seg_results_wrong(vol_gt , vol_pred , CTname='abc'):

#     connectivity = 2
#     from skimage import measure,color
#     labels_gt=measure.label(vol_gt,connectivity=connectivity)
#     l_gt,c_gt = np.unique(labels_gt , return_counts=True)
#     labels_pred=measure.label(vol_pred,connectivity=connectivity)
#     l_pred,c_pred = np.unique(labels_pred , return_counts=True)
#     l_gt = l_gt[l_gt>0]
#     l_pred = l_pred[l_pred>0]


#     Metrics = []
#     for g in l_gt:
#         vg = labels_gt == g
#         for p in l_pred:
#             vp = labels_pred == p
            
#             [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = \
#             cal_seg_metrics(vg , vp)
#             if intersection>0:
#                 Metrics.append( [ CTname, g, p,len(l_gt),len(l_pred),
#                         iou_score, dice_score, over_seg, under_seg, 
#                         area_gt, area_pred, intersection, union] )

#     return Metrics

# def vols_seg_results(vol_gt , vol_pred , CTname='abc'):

#     connectivity = 2
#     from skimage import measure,color
#     labels_gt=measure.label(vol_gt,connectivity=connectivity)
#     l_gt,c_gt = np.unique(labels_gt , return_counts=True)
#     labels_pred=measure.label(vol_pred,connectivity=connectivity)
#     l_pred,c_pred = np.unique(labels_pred , return_counts=True)
#     l_gt = l_gt[l_gt>0]
#     l_pred = l_pred[l_pred>0]


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
                
#         [iou_score, dice_score, over_seg, under_seg, area_gt, area_pred, intersection, union] = \
#         cal_seg_metrics(vg , VPs)
#         if intersection>0:
#             Metrics.append( [ CTname, g, merge,len(l_gt),len(l_pred),
#                     iou_score, dice_score, over_seg, under_seg, 
#                     area_gt, area_pred, intersection, union] )
#         else:
#             MissedLesions.append( [ CTname, g, merge,len(l_gt),len(l_pred),
#                     iou_score, dice_score, over_seg, under_seg, 
#                     area_gt, area_pred, intersection, union] )
            

#     return Metrics, MissedLesions
