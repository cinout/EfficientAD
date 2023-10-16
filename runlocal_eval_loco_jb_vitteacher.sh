folder_name="folder_vitteacher_sz512"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "juice_bottle" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/tiff/" \
  --output_dir "./outputs/${folder_name}/tiff/" \
  --folder_name "${folder_name}" \