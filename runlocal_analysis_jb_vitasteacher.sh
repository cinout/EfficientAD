python -u efficientad_analysis.py \
  --dataset mvtec_loco \
  --subdataset juice_bottle \
  --pretrained_network vit \
  --vit_teacher \
  --image_size_vit_teacher 512 \
  --note "vit as teacher" \
  --ana_id vitasteacher \
  --pth_folder "outputs/folder_vitteacher_sz512/output_20231013_122206_46_21_[jb]/trainings/mvtec_loco/juice_bottle/" \