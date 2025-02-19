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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.utils import AverageMeter
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
import datetime

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

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, oneshot_train=False, teacher_model=None, kd_ratio=1.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # fusion_percentage_options = [0.5, 0.2, 0.8, 1.0]
    fusion_percentage_options = [0.2, 0.5, 0.8]
    depth_list_options = [[2, 3], [3, 4], [4, 5]]
    # get group nums, SANDWICH_RULE: 9 + 2
    group_nums = len(fusion_percentage_options) * len(depth_list_options)
    if config.TRAIN.SANDWICH_RULE:
        group_nums += 2
    for i, (input, target, target_weight, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        # get group idx, 0: the min model, 10: the max model
        group_idx = i % group_nums
        if oneshot_train:
            if config.TRAIN.GROUP_SAMPLING:
                if config.TRAIN.SANDWICH_RULE:
                    if group_idx == 0:
                        model.module.set_fusion_percentage(0)
                        model.module.set_depth_list([2])
                        # logger.info('Sample smallest model')
                    elif group_idx == group_nums - 1:
                        model.module.set_fusion_percentage(1)
                        model.module.set_depth_list([5])
                        # logger.info('Sample largest model')
                    else:
                        model.module.set_fusion_percentage(fusion_percentage_options[group_idx % 9 // 3])
                        model.module.set_depth_list(depth_list_options[group_idx % 3])
                        # logger.info('Sample f {} d {}'.format(fusion_percentage_options[group_idx % 9 // 3],
                        #                                       depth_list_options[group_idx % 3]))

                else:
                    model.module.set_fusion_percentage(fusion_percentage_options[group_idx % 9 // 3])
                    model.module.set_depth_list(depth_list_options[group_idx % 3])
                    # logger.info('Sample f {} d {}'.format(fusion_percentage_options[group_idx % 9 // 3],
                    #                                       depth_list_options[group_idx % 3]))

            # if i % config.PRINT_FREQ == 0:
            #     logger.info('Using fusion {} depth {}'.format(fusion_percentage_options[i % 9 // 3],
            #                                                   depth_list_options[i % 3]))
            subnet_seed = int('%d%.3d%.3d' % (epoch * len(train_loader) + i, 0, 0))
            random.seed(subnet_seed)
            subnet_setting = model.module.sample_active_subnet()
            # logger.info('Manual_seed {}'.format(subnet_seed))
            # logger.info(subnet_setting)
        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        if teacher_model is not None:
            with torch.no_grad():
                soft_logits = teacher_model(input).detach()
            # TODO: support other type of kd loss (only MSE loss is supported now)
            kd_loss = F.mse_loss(output, soft_logits)
            # logger.info('Task loss {} KD loss {}'.format(loss, kd_loss))
            loss += kd_ratio * kd_loss
            # logger.info('Final loss {}'.format(loss))

        # compute gradient and do update step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                        target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Local time:{3}\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), now_time, batch_time=batch_time,
                    speed=input.size(0)/batch_time.val,
                    data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                            prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
