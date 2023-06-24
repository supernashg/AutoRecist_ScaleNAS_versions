# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.MASK_PATH = ''
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
# multiscale training params
_C.MODEL.MULTI_IMAGE_SIZE = []
_C.MODEL.MULTI_HEATMAP_SIZE = []

# OCR params
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False


# segmentation
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000

# segmentation OCR
_C.LOSS.BALANCE_WEIGHTS = [1]

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.INPUT_SIZE = 256
# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False


# segmentation related params
_C.DATASET.NUM_CLASSES = 22
_C.DATASET.EXTRA_TRAIN_SET = ''

# train
_C.TRAIN = CN()

# for OCR
_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

# for segmentation
_C.TRAIN.IMAGE_SIZE = [512, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.NUM_SAMPLES = 0
_C.TRAIN.EXTRA_EPOCH = 0

# BigNAS
_C.TRAIN.SANDWICH_RULE = False
_C.TRAIN.GROUP_SAMPLING = False
_C.TRAIN.FUSION_OPTIONS = [0.5]
_C.TRAIN.DEPTH_OPTIONS = [2, 3, 4, 5]

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.MINMAL_LR = 0.00001
_C.TRAIN.LR_SCHEDULER_TYPE = 'Step'

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.SUBNET_NUM = 1

# testing
_C.TEST = CN()

# segmentation
_C.TEST.IMAGE_SIZE = [512, 542]
_C.TEST.BASE_SIZE = 2048
_C.TEST.MULTI_SCALE = False
_C.TEST.NUM_SAMPLES = 0
_C.TEST.CENTER_CROP_TEST = False

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.JSON = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

# Jingchen added for CUMC Lesion Data
# /lib/roi_data/loader.py
_C.MODEL.LR_VIEW_ON = False
_C.MODEL.GIF_ON = False
_C.MODEL.LRASY_MAHA_ON = False






_C.TRAIN.VIS_ANCHOR_DIR = './anchor_vis/'
_C.TRAIN.VIS_ANCHOR = False

_C.TRAIN.IMS_PER_BATCH = 1
_C.TRAIN.ASPECT_GROUPING = True
# random cropping the images with new bounding box
_C.TRAIN.ONLINE_RANDOM_CROPPING = False
# probability of whether to crop or not
_C.TRAIN.ONLINE_RANDOM_CROPPING_PROBABILITY = 1.0
# threshold of iop, to keep the truncated mass
_C.TRAIN.ONLINE_RANDOM_CROPPING_IOP_THRESHOLD = 0.9
# range of cropped bounding box
_C.TRAIN.ONLINE_RANDOM_CROPPING_HEIGHT_MIN = 500
_C.TRAIN.ONLINE_RANDOM_CROPPING_HEIGHT_MAX = 2000
_C.TRAIN.ONLINE_RANDOM_CROPPING_WIDTH_MIN = 500
_C.TRAIN.ONLINE_RANDOM_CROPPING_WIDTH_MAX = 1000

# Crop images that have too small or too large aspect ratio
_C.TRAIN.ASPECT_CROPPING = False
_C.TRAIN.ASPECT_HI = 2
_C.TRAIN.ASPECT_LO = 0.5

_C.TRAIN.IGNORE_ON = False

_C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
_C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
_C.TRAIN.BG_THRESH_HI = 0.5
_C.TRAIN.BG_THRESH_LO = 0.0

# ---------------------------------------------------------------------------- #
# RPN training options
# ---------------------------------------------------------------------------- #

# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IOU >= thresh ==> positive RPN
# example)
_C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IOU < thresh ==> negative RPN
# example)
_C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Target fraction of foreground (positive) examples per RPN minibatch
_C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of RPN examples per image
_C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256

# NMS threshold used on RPN proposals (used during end-to-end training with RPN)
_C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring RPN proposals to keep before applying NMS (per image)
# When FPN is used, this is *per FPN level* (not total)
_C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring RPN proposals to keep after applying NMS (per image)
# This is the total number of RPN proposals produced (for both FPN and non-FPN
# cases)
_C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.TRAIN.RPN_STRADDLE_THRESH = 0

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale; not scale used during training or inference)
_C.TRAIN.RPN_MIN_SIZE = 0

# Filter proposals that are inside of crowd regions by CROWD_FILTER_THRESH
# "Inside" is measured as: proposal-with-crowd intersection area divided by
# proposal area
_C.TRAIN.CROWD_FILTER_THRESH = 0.7

# Ignore ground-truth objects with area < this threshold
_C.TRAIN.GT_MIN_AREA = -1

# /lib/roi_data/minibatch.py
_C.RPN = CN()
_C.RPN.RPN_ON = True

_C.TRAIN.SCALES = (512,)

_C.RETINANET = CN()
_C.RETINANET.RETINANET_ON = False

_C.DATA_SOURCE = 'coco'

_C.TRAIN.AUGMENTATION = False
 #        if _C.TRAIN.AUGMENTATION:
#             transform_cv = cv_transforms.Compose([
#                 cv_transforms.ColorJitter(brightness=0.5,
#                 contrast=0.25, gamma=0.5)])

_C.PIXEL_MEANS = [[[102.9801, 115.9465, 122.7717]]]

_C.TRAIN.MAX_SIZE = 800

# /lib/utils/blob.py
_C.FPN = CN()

# FPN is enabled if True
_C.FPN.FPN_ON = True

# Channel dimension of the FPN feature levels
_C.FPN.DIM = 256

# Initialize the lateral connections to output zero if True
_C.FPN.ZERO_INIT_LATERAL = False

# Stride of the coarsest FPN level
# This is needed so the input can be padded properly
_C.FPN.COARSEST_STRIDE = 32

#
# FPN may be used for just RPN, just object detection, or both
#

# Use FPN for RoI transform for object detection if True
_C.FPN.MULTILEVEL_ROIS = True
# Hyperparameters for the RoI-to-FPN level mapping heuristic
_C.FPN.ROI_CANONICAL_SCALE = 224  # s0
_C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
# Coarsest level of the FPN pyramid
_C.FPN.ROI_MAX_LEVEL = 5
# Finest level of the FPN pyramid
_C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
_C.FPN.MULTILEVEL_RPN = True
# Coarsest level of the FPN pyramid
_C.FPN.RPN_MAX_LEVEL = 6
# Finest level of the FPN pyramid
_C.FPN.RPN_MIN_LEVEL = 2
# FPN RPN anchor aspect ratios
_C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this size on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
_C.FPN.RPN_ANCHOR_START_SIZE = 16
# [Infered Value] Scale for RPN_POST_NMS_TOP_N.
# Automatically infered in training, fixed to 1 in testing.
_C.FPN.RPN_COLLECT_SCALE = 1
# Use extra FPN levels, as done in the RetinaNet paper
_C.FPN.EXTRA_CONV_LEVELS = False
# Use GroupNorm in the FPN-specific layers (lateral, etc.)
_C.FPN.USE_GN = False

# lib/utils/ImageIO.py:
# Whether implemente histogram equalization
_C.HIST_EQ = False
# set to true to make im and other_im do hist-eq at the same position, i.e. symmentry
_C.HIST_EQ_SYM = False
_C.A_HIST_EQ = False
# 
_C.TRAIN.PADDING = 128
# Set to True to randomly perturb the bbox of other images during training
_C.TRAIN.AUG_LRV_BBOX = False

# For deeplesion dataset. This parameter is used on 'https://github.com/rsummers11/CADLab/tree/master/lesion_detector_3DCE' 
_C.WINDOWING = [-1024, 3071]
# ---------------------------------------------------------------------------- #
# DeepLesion options
# ---------------------------------------------------------------------------- #
_C.LESION = CN()
_C.LESION.USE_3DCE_FROC = True

# For 2D network, whether use three slices or a single slices without normalization
#_C.LESION.THREE_SLICES = False
#_C.LESION.USE_NORMED_CT = False

_C.LESION.SLICE_INTERVAL = 2.0
_C.LESION.LESION_ENABLED = True

# For 3D DeepLesion Input
_C.LESION.USE_3D_INPUT = False
_C.LESION.SLICE_NUM = 3
_C.LESION.NO_DEPTH_PAD = True
_C.LESION.DEBUG = False

# For 3DCE architecture
_C.LESION.USE_3DCE = False
# Symbol M in the paper,can be 1,3,5,7,9...
_C.LESION.NUM_IMAGES_3DCE = 1

_C.LESION.MULTI_MODALITY = True
# For multi modality,concat blobs before RPN
_C.LESION.CONCAT_BEFORE_RPN = True
# For multi modality,sum blobs before RPN
_C.LESION.SUM_BEFORE_RPN = False
_C.LESION.GIF_BEFORE_RPN = False
_C.LESION.FUSION_BEFORE_RPN = False
_C.LESION.WITHOUT_SHARE = False

# For Position supervise
_C.LESION.USE_POSITION = True
_C.LESION.SHALLOW_POSITION = False
_C.LESION.POSITION_RCNN = False
_C.LESION.MM_POS= False
_C.LESION.MM_POS_CHANNEL= False
_C.LESION.POS_CONCAT_RCNN= False
# Using specific window for each sample
_C.LESION.USE_SPECIFIC_WINDOWS = False
# Only use samples with maximum num of same window (-175,275)
_C.LESION.USE_ONE_WINDOW = False

# Multi modality input
_C.LESION.USE_MULTI_WINDOWS = False
# Added for implementing multi-context roi-pooling RCNN
_C.LESION.MM_TEST= False


_C.TRAIN.DATASETS = ('PDS_Q2_A&C_22Cat_train',)
_C.TRAIN.PROPOSAL_FILES = ()
_C.TRAIN.USE_ADJACENT_LAYER = False
_C.TRAIN.USE_FLIPPED = True
_C.TRAIN.USE_Z_FLIPPED = False
_C.TRAIN.BBOX_THRESH = 0.5

_C.VAL = CN()
_C.VAL.DATASETS = ('PDS_AMGEN_20020408_22Cat_test',)
_C.VAL.PROPOSAL_FILES = ()

_C.MODEL.NUM_CLASSES = 22
_C.MODEL.KEYPOINTS_ON = False
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False
_C.MODEL.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

_C.EXCLUDE_LAYERS = []#['last_layer.3.weight' ,'last_layer.3.bias']

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # if args.modelDir:
    if hasattr(args, 'modelDir'):
        if len(args.modelDir) > 0:
            cfg.OUTPUT_DIR = args.modelDir

    # if args.logDir:
    if hasattr(args, 'logDir'):
        cfg.LOG_DIR = args.logDir

    # if args.dataDir:
    if hasattr(args, 'dataDir'):
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    print(_C)
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

