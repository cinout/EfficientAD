#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_sb"
#SBATCH --account=punim1623
#SBATCH --time=0-00:30:00

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
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  \
  --train_steps 70000 \
  --seeds 10 20 30 \
  \
  --trained_folder outputs/ckpts_lid_sepa/output_20240228_002744_34_12 \
  --note "[], lid_sepa" \
  --lid_score_eval \
  --lid_eval_onwhat separate_mean \


##Log this job's resource usage stats###
my-job-stats -a -n -s
##