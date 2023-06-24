#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=0 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=1 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=1 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=2 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=2 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=3 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=3 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion_Q5/Lesion_Q5_scalenet_seg_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_superscalenet_base/data_patch_train/best.pth &
