#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_sb"
#SBATCH --account=punim1623
#SBATCH --time=0-21:00:00

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
  --subdataset screw_bag \
  --stg1_ckpt outputs/folder_baseline/output_20240131_215545_7_68_sd10_[sb]/trainings/mvtec_loco/screw_bag \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  \
  --train_steps 70000 \
  --seeds 10 20 30 \
  \
  --include_logicano \
  --logicano_select percent \
  --percent_logicano 0.6 \
  \
  --note "60% logicano, equal" \


##Log this job's resource usage stats###
my-job-stats -a -n -s
##