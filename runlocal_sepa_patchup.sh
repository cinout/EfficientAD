python -u efficientad_separatebranches_patchup.py \
  --dataset mvtec_loco \
  --subdataset pushpins \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "separate branches [vit for logical]" \
  --logical_teacher vit \
  --logical_teacher_image_size 1024 \
  --output_dir "outputs/folder_sepa_vit/output_20231018_233435_67_25_[pp]" \