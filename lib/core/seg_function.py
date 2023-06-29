# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import random
import timeit

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.utils import Seg_AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank

logger = logging.getLogger(__name__)

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train_seg(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, writer_dict, device=None,
              oneshot_train=False, teacher_model=None, kd_ratio=1.0):
    # Training
    model.train()
    batch_time = Seg_AverageMeter()
    ave_loss = Seg_AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    fusion_percentage_options = config.TRAIN.FUSION_OPTIONS
    depth_list_options = config.TRAIN.DEPTH_OPTIONS
    # get group nums, SANDWICH_RULE: 9 + 2
    group_nums = len(fusion_percentage_options) * len(depth_list_options)
    if config.TRAIN.SANDWICH_RULE:
        group_nums += 2
    if rank == 0:
        logging.info('Use group_sampling {}, Use sandwich_rule {}'
                     .format(config.TRAIN.GROUP_SAMPLING, config.TRAIN.SANDWICH_RULE))
        logging.info('Group_nums {}, Fusion_percentage_options {}, Depth_list_options {}'
                     .format(group_nums, fusion_percentage_options, depth_list_options))

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        if device is None:
            images = images.cuda()
            labels = labels.long().cuda()
        else:
            images = images.to(device)
            labels = labels.long().to(device)
        group_idx = i_iter % group_nums
        subnet_msg = 'Use Normal train mode'
        if oneshot_train:
            if config.TRAIN.GROUP_SAMPLING:
                if config.TRAIN.SANDWICH_RULE:
                    if group_idx == 0:
                        model.module.set_fusion_percentage(0.1)
                        model.module.set_depth_list([2])
                        subnet_msg = 'Use oneshot train mode, current sample smallest model'
                    elif group_idx == group_nums - 1:
                        model.module.set_fusion_percentage(1)
                        model.module.set_depth_list([5])
                        subnet_msg = 'Use oneshot train mode, current sample largest model'
                    else:
                        model.module.set_fusion_percentage(fusion_percentage_options[group_idx % 9 // 3])
                        model.module.set_depth_list(depth_list_options[group_idx % 3])
                        subnet_msg = 'Use oneshot train mode, current sample f {} d {}' \
                            .format(fusion_percentage_options[group_idx % 9 // 3],
                                    depth_list_options[group_idx % 3])
                else:
                    model.module.set_fusion_percentage(fusion_percentage_options[group_idx % 9 // 3])
                    model.module.set_depth_list(depth_list_options[group_idx % 3])

            # if i % config.PRINT_FREQ == 0:
            #     logger.info('Using fusion {} depth {}'.format(fusion_percentage_options[i % 9 // 3],
            #                                                   depth_list_options[i % 3]))
            subnet_seed = int('%d%.3d%.3d' % (epoch * len(trainloader) + i_iter, 0, 0))
            random.seed(subnet_seed)
            subnet_setting = model.module.sample_active_subnet()
            # logger.info('Manual_seed {}'.format(subnet_seed))
            # logger.info(subnet_setting)
        # compute output

        losses, output = model(images, labels)
        loss = losses.mean()

        if teacher_model is not None:
            with torch.no_grad():
                soft_logits = teacher_model(images).detach()
            kd_loss = F.mse_loss(output, soft_logits)
            # logger.info('Task loss {} KD loss {}'.format(loss, kd_loss))
            loss += kd_ratio * kd_loss
            # logger.info('Final loss {}'.format(loss))


        reduced_loss = reduce_tensor(loss)
        model.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Time {} Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}'.format(time.strftime('%Y-%m-%d-%H-%M'),
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), lr, print_loss)
            logging.info(msg)
            logging.info(subnet_msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate_seg(config, testloader, model, writer_dict, device=None):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = Seg_AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            if device is None:
                image = image.cuda()
                label = label.long().cuda()
            else:
                image = image.to(device)
                label = label.long().to(device)

            losses, pred = model(image, label)
            pred = F.upsample(input=pred, size=(
                size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
    if device is None:
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    else:
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[8]
    print_loss = ave_loss.average() / world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array

def validate_seg_wo_loss(config, testloader, model, writer_dict, device=None):
    rank = get_rank()
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            if device is None:
                image = image.cuda()
                label = label.long().cuda()
            else:
                image = image.to(device)
                label = label.long().to(device)

            pred = model(image)
            pred = F.upsample(input=pred, size=(
                size[-2], size[-1]), mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()

    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[8]

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

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
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array[8]
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[8]

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]),
                                  mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

def validate_seg_batchmodels(config, testloader, models, writer_dict, device=None):
    rank = get_rank()

    CMs = []
    mIoUs = []
    for model in models:
        model.eval()    
        confusion_matrix = np.zeros(
            (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
        CMs.append(confusion_matrix)

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            if device is None:
                image = image.cuda()
                label = label.long().cuda()
            else:
                image = image.to(device)
                label = label.long().to(device)

            for j , model in enumerate(models):

                losses, pred = model(image, label)

                pred = F.upsample(input=pred, size=(
                    size[-2], size[-1]), mode='bilinear')

                CMs[j] += get_confusion_matrix(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)      


    for confusion_matrix in CMs:
        if device is None:
            confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        else:
            confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)

        confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array[8]
        mIoUs.append(mean_IoU)


    return mIoUs



"""
                 start = timeit.default_timer()

                losses, pred = model(image, label)
                end_loss = timeit.default_timer()

                pred = F.upsample(input=pred, size=(
                    size[-2], size[-1]), mode='bilinear')
                end_pred = timeit.default_timer()

                CMs[j] += get_confusion_matrix(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)      

                end_cm = timeit.default_timer()
                logger.info('millisec: %f: %f: %f' %( 
                    (end_loss - start)*1000 , (end_pred - end_loss)*1000 , (end_cm - end_pred)*1000 ))   

"""
