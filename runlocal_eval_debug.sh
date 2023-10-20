timestamp="20231018_231623_21_46_[bb]"
folder_name="folder_sepa_vit"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "breakfast_box" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/${folder_name}/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \
  --folder_name "${folder_name}" \