#!/bin/bash

#SBATCH --partition=cascade

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
conda activate ds2

timestamp="20230926_100404_58_80_[sc]"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "splicing_connectors" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \

## FIXME: update timestamp
##Log this job's resource usage stats###
my-job-stats -a -n -s
##