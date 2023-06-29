#!/bin/bash



for f in 0.35
do
  for d in [4,5]
  do
    CUDA_VISIBLE_DEVICES=3 python -u tools/Lesion_random_seg_test_batchmodels.py \
     --cfg experiments/lesion_Q5/Lesion_Q5_9Slices_scalenet_seg_test.yaml \
     --save_name superscalenet_lesion \
     --sample_num 15 \
     --fusion_percentage $f \
     --depth_list $d  \
     --batchmodels 5 \
     TEST.MODEL_FILE output/Lesion/superscalenet_seg/Lesion_Q5_9Slices_superscalenet/data_patch_train/best.pth &
    
    sleep 20
  done
done

