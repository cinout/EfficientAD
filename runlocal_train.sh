python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --analysis_heatmap \
  --pretrained_network mae_vit \
  --mae_vit_teacher \
  --image_size_mae_vit_teacher 512 \
  --note "mae_vit_as_teacher" \
  # --seeds 10 20 30 \