timestamp="20230530_195518_[bb]"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py --object_name "breakfast_box" --dataset_base_dir "./datasets/loco/" --anomaly_maps_dir "./output_${timestamp}/anomaly_maps/mvtec_loco/" --output_dir "./output_${timestamp}/metrics/mvtec_loco/"