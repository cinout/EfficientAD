python -u efficientad_analysis.py \
  --dataset mvtec_loco \
  --subdataset juice_bottle \
  --pretrained_network vit \
  --image_size_vit_teacher 512 \
  --weights pretrained_pdn/pretrained_pdn_vit_lastlayer_bs8_sz512/teacher_small_state_vit_b16.pth \
  --note "vit pretrained PDN" \
  --ana_id vitpretrainedPDN \
  --pth_folder "outputs/folder_vit_bs8_sz512/output_20231013_172041_67_22_[jb]/trainings/mvtec_loco/juice_bottle/" \