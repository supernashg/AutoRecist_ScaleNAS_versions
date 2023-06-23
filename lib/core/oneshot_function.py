# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import copy
import random
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import dataset

logger = logging.getLogger(__name__)


def calib_bn(cfg, model, seed, masks=None):
    logger.info('prepare to set running statistics')

    assert len(cfg.GPUS), 'calib_bn DO NOT support multi-node mode, '
    calib_loader = builf_calib_dataload(cfg)

    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    random.seed(seed)
    # subnet_setting = forward_model.module.sample_active_subnet()
    if masks is not None:
        forward_model.module.set_active_subnet(masks)
    else:
        subnet_setting = forward_model.module.sample_active_subnet()
    # logger.info(pprint.pformat(subnet_setting))

    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):

            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    with torch.no_grad():
        for i, (input, _, _, _) in enumerate(calib_loader):
            _ = forward_model(input)

    logger.info('calculate bn done')

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    return model

def builf_calib_dataload(cfg):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Using train dataset as calibration dataset
    # TODO: split subdata size for coco
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    n_samples = 118287
    g = torch.Generator()
    g.manual_seed(937162211)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    n_images = cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS) * 8
    chosen_indexes = rand_indexes[:n_images]

    calib_dataset = train_dataset
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        sampler=sub_sampler,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    return calib_loader

def calib_bn_seg(cfg, model, seed, masks=None):
    logger.info('prepare to set running statistics')

    assert len(cfg.GPUS), 'calib_bn DO NOT support multi-node mode, '

    calib_loader = builf_calib_dataload_seg(cfg)

    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    random.seed(seed)
    # subnet_setting = forward_model.module.sample_active_subnet()
    if masks is not None:
        forward_model.module.set_active_subnet(masks)
    else:
        subnet_setting = forward_model.module.sample_active_subnet()

    # logger.info(pprint.pformat(subnet_setting))
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):

            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(calib_loader):
            images = images
            labels = labels.long()
            try:
                _, _ = forward_model(images, labels)
            except:
                _ = forward_model(images)

    logger.info('calculate bn done')

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    return model

def builf_calib_dataload_seg(cfg):

    crop_size = (cfg.TRAIN.IMAGE_SIZE[1], cfg.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        root=cfg.DATASET.ROOT,
        list_path=cfg.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=cfg.DATASET.NUM_CLASSES,
        multi_scale=cfg.TRAIN.MULTI_SCALE,
        flip=cfg.TRAIN.FLIP,
        ignore_label=cfg.TRAIN.IGNORE_LABEL,
        base_size=cfg.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        downsample_rate=cfg.TRAIN.DOWNSAMPLERATE,
        scale_factor=cfg.TRAIN.SCALE_FACTOR)

    n_samples = 2912
    g = torch.Generator()
    g.manual_seed(937162211)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    n_images = cfg.TRAIN.BATCH_SIZE_PER_GPU * 8
    chosen_indexes = rand_indexes[:n_images]

    calib_dataset = train_dataset
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        sampler=sub_sampler,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    return calib_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
