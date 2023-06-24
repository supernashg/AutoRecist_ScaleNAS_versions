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

from dataset.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch



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
                        default=0,
                        help='KD ratio')
    parser.add_argument('--no_last_layer',
                        action='store_true',
                        help='pretrain weights do not use for last layer')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)
    
    cfg.defrost()
    if hasattr(args, 'no_last_layer'):
        cfg.EXCLUDE_LAYERS = ['last_layer.3.weight' ,'last_layer.3.bias']
    cfg.freeze()


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

    model.set_fusion_percentage(1.0)
    depth = cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS[0]
    model.set_depth_list([depth])
    subnet_setting = model.sample_active_subnet()
    logging.info('###################Teacher setting##############\n{}\n'
                 '###################Teacher setting##############'.format(subnet_setting))


    if args.local_rank == 0:
        # provide the summary of model
        dump_input = torch.rand(
            (1, cfg.LESION.SLICE_NUM, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    train_roidb, train_ratio_list, train_ratio_index = combined_roidb_for_training(
            cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)

    train_dataset = RoiDataLoader(
        train_roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    

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


    test_size = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])

    test_roidb, test_ratio_list, test_ratio_index = combined_roidb_for_training(
            cfg.VAL.DATASETS, cfg.VAL.PROPOSAL_FILES)

    test_dataset = RoiDataLoader(
        test_roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)

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
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)


    model = FullModel(model, criterion)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
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
            raise ValueError('No extra_trainloader')
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

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))

            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            
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



if __name__ == '__main__':
    main()
