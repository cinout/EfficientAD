#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_jb"
#SBATCH --account=punim1623
#SBATCH --time=1-10:00:00

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
  --subdataset juice_bottle \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "include_logicano, loss=focal, iter=120000, geo_aug" \
  --seeds 10 20 30 \
  --include_logicano \
  --logicano_select absolute \
  --num_logicano 10 \
  --logicano_loss focal \
  --stg1_ckpt outputs/folder_baseline/output_20240131_215528_9_92_sd10_[jb]/trainings/mvtec_loco/juice_bottle \
  --train_steps 120000 \
  --loss_on_resize \
  --geo_augment \
  --use_rotate_flip \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##