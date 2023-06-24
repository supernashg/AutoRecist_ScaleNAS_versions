#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion/Lesion_scalenet_seg_w32_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_superscalenet_lr1e-3_bs16x4_slice3_WL1/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=2 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion/Lesion_scalenet_seg_w32_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_superscalenet_lr1e-3_bs16x4_slice3_WL1/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=3 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion/Lesion_scalenet_seg_w32_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_superscalenet_lr1e-3_bs16x4_slice3_WL1/data_patch_train/best.pth &

CUDA_VISIBLE_DEVICES=3 python -u tools/Lesion_evo_seg_test.py \
  --cfg experiments/lesion/Lesion_scalenet_seg_w32_test.yaml \
  --evo_file evo_files/lesion_evo_file.txt \
  TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_superscalenet_lr1e-3_bs16x4_slice3_WL1/data_patch_train/best.pth &
