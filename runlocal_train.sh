python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset pushpins \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  \
  --train_steps 400 \
  --seeds 10 \
  \
  --include_logicano \
  --logicano_select percent \
  --percent_logicano 1.0 \
  # \
  # --use_lid_score \
  # --trained_folder outputs/folder_baseline/output_20240131_213923_16_56 \
  
#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors