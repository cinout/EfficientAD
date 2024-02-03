python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "haha" \
  --seeds 10 \
  --include_logicano \
  --logicano_select absolute \
  --num_logicano 10 \
  # --iters_stg2 20 \
  # --stg1_ckpt outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]/trainings/mvtec_loco/breakfast_box \
 