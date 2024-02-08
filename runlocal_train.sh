python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "haha" \
  --seeds 10 \
  --include_logicano \
  --logicano_select absolute \
  --num_logicano 10 \
  --logicano_loss focal \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  --loss_on_resize \
  --geo_augment \
  # --iters_stg2 20 \
 
#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors