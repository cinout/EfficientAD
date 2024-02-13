#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_sc"
#SBATCH --account=punim1623
#SBATCH --time=0-07:00:00

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

python -u efficientad_twosteps.py \
  --dataset mvtec_loco \
  --subdataset splicing_connectors \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "twostage" \
  --seeds 10 20 30 \
  --stg1_ckpt outputs/folder_baseline/output_20240131_215545_79_5_sd10_[sc]/trainings/mvtec_loco/splicing_connectors \
  --logicano_select percent \
  --percent_logicano 1.0 \
  --lr_stg2 0.000004 \
  --loss_on_resize \


##Log this job's resource usage stats###
my-job-stats -a -n -s
##