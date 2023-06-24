#!/bin/bash

for f in 0.8
do
  for d in [2,3] 
  do
    CUDA_VISIBLE_DEVICES=0 python -u tools/Lesion_random_seg_test.py \
     --cfg experiments/lesion_Q5/Lesion_Q5_9Slices_scalenet_seg_test.yaml \
     --save_name superscalenet_lesion \
     --sample_num 67 \
     --fusion_percentage $f \
     --depth_list $d  \
     TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_9Slices_superscalenet/data_patch_train/best.pth &
    
    sleep 10
  done
done


