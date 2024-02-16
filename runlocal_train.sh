python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  --seeds 10 \
  --include_logicano \
  --logicano_select absolute \
  --num_logicano 10 \
  --logicano_loss focal \
  --equal_train_normal_logicano \
  --use_seg_network \
  --train_steps 10 \
  --use_l1_loss \

#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors