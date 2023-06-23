# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import random
import timeit
import logging

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import dataset
from config import cfg
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.seg_function import train_seg as train
from core.seg_function import validate_seg as validate
from core.seg_function import validate_seg_wo_loss as validate_wo_loss
from utils.utils import get_model_summary
from utils.utils import create_logger, FullModel, get_rank


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--manual_seed',
                        default=2147483659,
                        type=int,
                        help='the default is a random prime number')
    parser.add_argument('--oneshot_train',
                        action='store_true',
                        help='oneshot train mode')
    parser.add_argument('--use_kd',
                        action='store_true',
                        help='using kd mode')
    parser.add_argument('--kd_ratio',
                        type=float,
                        default=0.5,
                        help='KD ratio')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    # set a random seed for reproducibility
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    gpus = list(cfg.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    model = eval('models.' + cfg.MODEL.NAME +
                 '.get_seg_model')(cfg)

    if cfg.MODEL.NAME == 'superscalenet_seg':
        if cfg.MODEL.MASK_PATH is '':
            subnet_setting = model.sample_active_subnet()
        else:
            subnet_setting = np.load(cfg.MODEL.MASK_PATH, allow_pickle=True)
            model.set_active_subnet(subnet_setting)
            logger.info('=> setting mask from {}'.format(cfg.MODEL.MASK_PATH))
        logger.info(pprint.pformat(subnet_setting))

    if args.local_rank == 0:
        # provide the summary of model
        dump_input = torch.rand(
            (1, 3, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)


    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data

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

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    if cfg.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            root=cfg.DATASET.ROOT,
            list_path=cfg.DATASET.EXTRA_TRAIN_SET,
            num_samples=None,
            num_classes=cfg.DATASET.NUM_CLASSES,
            multi_scale=cfg.TRAIN.MULTI_SCALE,
            flip=cfg.TRAIN.FLIP,
            ignore_label=cfg.TRAIN.IGNORE_LABEL,
            base_size=cfg.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=cfg.TRAIN.DOWNSAMPLERATE,
            scale_factor=cfg.TRAIN.SCALE_FACTOR)

        if distributed:
            extra_train_sampler = DistributedSampler(extra_train_dataset)
        else:
            extra_train_sampler = None

        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=cfg.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)

    test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
    test_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        root=cfg.DATASET.ROOT,
        list_path=cfg.DATASET.TEST_SET,
        num_samples=cfg.TEST.NUM_SAMPLES,
        num_classes=cfg.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=cfg.TRAIN.IGNORE_LABEL,
        base_size=cfg.TEST.BASE_SIZE,
        crop_size=test_size,
        center_crop_test=cfg.TEST.CENTER_CROP_TEST,
        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    teacher_model = None

    # criterion
    if cfg.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                     thres=cfg.LOSS.OHEMTHRES,
                                     min_kept=cfg.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights.cuda())
    else:
        criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights.cuda())
    if args.use_kd:

        teacher_model = get_teacher_model(cfg, args)
        # valid_loss, mean_IoU, IoU_array = validate_wo_loss(cfg,
        #                                            valid_loader, teacher_model, writer_dict, device)
        mean_IoU, IoU_array = validate_wo_loss(cfg, valid_loader, teacher_model, writer_dict, device)
        if args.local_rank == 0:
            logger.info('Use kd ratio {}'.format(args.kd_ratio))
            msg = 'mean_IoU: {: 4.4f}'.format(mean_IoU)
            logging.info(msg)
            logging.info(IoU_array)

        teacher_model.eval()

    model = FullModel(model, criterion)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.lol
    # optimizer
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': cfg.TRAIN.LR}],
                                    lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WD,
                                    nesterov=cfg.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         cfg.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    if cfg.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {} best_mIoU {})"
                        .format(checkpoint['epoch'], best_mIoU))

    start = timeit.default_timer()
    end_epoch = cfg.TRAIN.END_EPOCH + cfg.TRAIN.EXTRA_EPOCH
    num_iters = cfg.TRAIN.END_EPOCH * epoch_iters
    extra_iters = cfg.TRAIN.EXTRA_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):

        if epoch >= cfg.TRAIN.END_EPOCH:
            train(cfg, epoch - cfg.TRAIN.END_EPOCH,
                      cfg.TRAIN.EXTRA_EPOCH, epoch_iters,
                      cfg.TRAIN.EXTRA_LR, extra_iters,
                      extra_trainloader, optimizer, model,
                      writer_dict, device, args.oneshot_train,
                      teacher_model, args.kd_ratio)
        else:
            train(cfg, epoch, cfg.TRAIN.END_EPOCH,
                      epoch_iters, cfg.TRAIN.LR, num_iters,
                      trainloader, optimizer, model,
                      writer_dict, device, args.oneshot_train,
                      teacher_model, args.kd_ratio)

        valid_loss, mean_IoU, IoU_array = validate(cfg,
                                                   valid_loader, model, writer_dict, device)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int((end-start)/3600))
                logger.info('Done')


def get_teacher_model(cfg, args):

    teacher_model = eval('models.' + cfg.MODEL.NAME + '.get_seg_model')(
        cfg, is_train=True)
    ## doesn't use FullModel for teacher as we only forward pass
    teacher_model.set_fusion_percentage(1.0)
    depth = cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS[0]
    teacher_model.set_depth_list([depth])

    subnet_setting = teacher_model.sample_active_subnet()
    if args.local_rank == 0:
        logging.info('###################Teacher setting##############\n{}\n'
                     '###################Teacher setting##############'.format(subnet_setting))

    teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    device = torch.device('cuda:{}'.format(args.local_rank))
    teacher_model = teacher_model.to(device)
    teacher_model = nn.parallel.DistributedDataParallel(
        teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return teacher_model

if __name__ == '__main__':
    main()
