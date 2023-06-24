#!/bin/bash

for f in 0.2 0.35 0.5 0.65 0.8
do
  for d in [2,3] [3,4] [4,5]
  do
    python -u tools/Lesion_random_seg_test.py \
     --cfg experiments/lesion/Lesion_scalenet_seg_w32_test.yaml \
     --save_name superscalenet_lesion \
     --sample_num 67 \
     --fusion_percentage $f \
     --depth_list $d  \
     TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_superscalenet_lr1e-3_bs16x4_slice3_WL1/data_patch_train/best.pth
  done
done

