python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  \
  --train_steps 40 \
  --seeds 10 \
  \
  --include_logicano \
  --logicano_select percent \
  --percent_logicano 0.1 \
  \
  --equal_train_normal_logicano \
  \
  # --use_seg_network \
  # --equal_train_normal_logicano \
  # --logicano_loss focal \
  # --use_l1_loss \
  # --include_logicano \
  # --train_steps 10 \

#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors