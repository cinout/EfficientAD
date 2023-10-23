timestamp="20231013_122206_46_21_[jb]"
folder_name="folder_vitteacher_sz512"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "juice_bottle" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/${folder_name}/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \
  --folder_name "${folder_name}" \