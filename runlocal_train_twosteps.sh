python -u efficientad_twosteps.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "haha" \
  --seeds 20 \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  --logicano_select percent \
  --percent_logicano 1.0 \
  --lr_stg2 0.000004 \
  --loss_on_resize \
  # --iters_stg2 20 \
 
#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors