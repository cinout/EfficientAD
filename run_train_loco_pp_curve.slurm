#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_pp"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate eff

set -e
set -x


python -u efficientad.py \
  --dataset mvtec_loco \
  --subdataset pushpins \
  --stg1_ckpt outputs/folder_baseline/output_20240131_215528_43_52_sd10_[pp]/trainings/mvtec_loco/pushpins \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  \
  --train_steps 70000 \
  --seeds 10 \
  \
  --trained_folder outputs/ckpts_isofor/output_20240313_165808_50_22 \
  --lid_on_history \
  #--note "try iso_forest alone" \


##Log this job's resource usage stats###
my-job-stats -a -n -s
##