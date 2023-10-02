#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="train_sc"
#SBATCH --account=punim1623
#SBATCH --time=0-05:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate ds2

set -e
set -x

python -u efficientad.py \
  --pretrained_network vit \
  --weights pretrained_pdn/pretrained_pdn_20231002_204347_4_81/teacher_small_state_vit_b16_ft_splicing_connectors.pth \
  --dataset mvtec_loco \
  --subdataset splicing_connectors \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --note "[pretrained] vit_ft"
# FIXME: update job-name, weights, pretrained_network, subdataset, note
##Log this job's resource usage stats###
my-job-stats -a -n -s
##