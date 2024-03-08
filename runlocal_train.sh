python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
  \
  --train_steps 12 \
  --seeds 10 \
  \
  --use_masked_conv \
  --pos_masked_conv d7 \
  --w_loss_masked_conv 50.0 \
  --note "masked conv" \
  
  # \
  # --lid_score_eval \
  # --trained_folder outputs/folder_baseline/output_20240131_213923_16_56 \
  
#  breakfast_box
#  juice_bottle
#  pushpins
#  screw_bag
#  splicing_connectors