# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time
import numpy as np
import fcntl



def parse_args():
    parser = argparse.ArgumentParser(description='Evolutionary algorithm')
    # general
    parser.add_argument('--generation',
                        default=20,
                        type=int)
    parser.add_argument('--population',
                        default=100,
                        type=int)
    parser.add_argument('--cross_over_prob',
                        default=0.25,
                        type=float)
    parser.add_argument('--mutation_prob',
                        default=0.5,
                        type=float)

    parser.add_argument('--evo_file',
                        help='evo_file has masks to be validated',
                        default='./evo_files/lesion_evo_file.txt',
                        type=str)
    parser.add_argument('--masks_dir',
                        help='dir of masks',
                        default='./evo_files/masks',
                        type=str)

    args = parser.parse_args()
    return args

def lockfile(f):
    try:
        fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except:
        return False

def select_masks(masks, mask_num=100):
    assert len(masks) >= mask_num
    masks = sorted(masks, key=lambda x: x[2])
    new_masks = []
    while len(new_masks) < mask_num:
        new_masks.append(masks[0])
        for i in range(1, len(masks)):
            if masks[i][3] > new_masks[-1][3] and len(new_masks) < mask_num:
                new_masks.append(masks[i])
        masks = [item for item in masks if item not in new_masks]
        masks = [item for item in masks if item not in new_masks]
    return new_masks


def do_cross_over(mask_base, mask_evo, prob=0.25):

    assert len(mask_base) == len(mask_evo)

    for module_id in range(len(mask_base) - 1):
        if random.random() < prob:
            # print('replacing module {}'.format(module_id))
            mask_base[module_id] = mask_evo[module_id]

    return mask_base

def do_mutation(mask_base, prob=0.5):

    for m, sub_mask in enumerate(mask_base[:-1]):
        depth = sub_mask['d']
        fusion = sub_mask['f']
        for i, sub_fusion in enumerate(fusion):
            for j, sub_sub_fusion in enumerate(sub_fusion):
                if j < depth[i][0]:
                    for k, single_fusion in enumerate(sub_sub_fusion):
                        if k != i and random.random() < prob:
                            mask_base[m]['f'][i][j][k] = 1 - single_fusion

    return mask_base


def main():
    args = parse_args()

    evo_file = args.evo_file
    index_line = -1
    while True:  # wait for new task LOOP-0
        while True:  # LOOP-1
            if not os.path.exists(evo_file):
                time.sleep(1)
                continue
            with open(evo_file, "r+") as f:
                if not lockfile(f):
                    time.sleep(10)
                    print('Evo waiting for loading...')
                    continue  # re-open
                lines = f.readlines()
                masks_w_ap = []
                for lines_ind in range(0, len(lines)):
                    line = lines[lines_ind]
                    _line = line.split()
                    if len(_line) == 4:
                        id = _line[0]
                        params = float(_line[1])
                        flops = float(_line[2])
                        ap = float(_line[3])
                        masks_w_ap.append([id, params, flops, ap])
            if len(masks_w_ap) % args.population != 0:  # no arch
                print('Cur masks {} Waiting for validation results'.format(len(masks_w_ap)))
                time.sleep(60)
                continue  # re-open LOOP-1
            else:
                break  # LOOP-1

        cur_masks_num = len(masks_w_ap)
        masks_w_ap_top = select_masks(masks_w_ap, args.population)
        print(masks_w_ap_top)
        BASE_DIR = args.masks_dir
        masks_list = []
        for m, _, _, _ in masks_w_ap_top:
            mask_path = os.path.join(f'{BASE_DIR}/{m}.npy')
            masks_list.append(np.load(mask_path, allow_pickle=True))
        with open(evo_file, "a") as f:
            for cur_i, i in enumerate(range(cur_masks_num, cur_masks_num + args.population)):
                mask_base = masks_list[cur_i]
                random_i = random.randint(0, args.population - 1)
                mask_evo = masks_list[random_i]
                mask_base = do_cross_over(mask_base, mask_evo, args.cross_over_prob)
                mask_base = do_mutation(mask_base, args.mutation_prob)
                new_mask_path = os.path.join(f'{BASE_DIR}/mask_{i}.npy')
                np.save(new_mask_path, mask_base)
                f.writelines(f'mask_{i}\n')
        exit()

if __name__ == '__main__':
    main()
