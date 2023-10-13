python -u efficientad_analysis.py \
  --dataset mvtec_loco \
  --subdataset juice_bottle \
  --pretrained_network wide_resnet101_2 \
  --weights pretrained_pdn/pretrained_pdn_wide_resnet101_2/teacher_small.pth \
  --note "wideresnet pretrained PDN" \
  --ana_id respretrainedPDN \