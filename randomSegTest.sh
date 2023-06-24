#!/bin/bash
for f in 0.2 0.8
do
  for d in [2,3] 
  do
    python -u tools/random_seg_test.py \
     --cfg experiments/cityscapes/scalenet_seg_w32_test.yaml \
     --save_name superscalenet_seg_w32 \
     --sample_num 67 \
     --bn_calib \
     --fusion_percentage $f \
     --depth_list $d  \
     TEST.MODEL_FILE models/pytorch/seg_cityscapes/superscalenet_seg_w32.pth &
  done
done