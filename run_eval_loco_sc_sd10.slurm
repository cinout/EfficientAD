#!/bin/bash

#SBATCH --partition=sapphire

#SBATCH --job-name="eval_sc"
#SBATCH --account=punim1623
#SBATCH --time=0-01:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate eff

timestamp="20240222_224806_27_49_sd10_[sc]"
folder_name="folder_baseline_lid"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "splicing_connectors" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/${folder_name}/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \
  --folder_name "${folder_name}" \
  --generate_visual \


##Log this job's resource usage stats###
my-job-stats -a -n -s
##