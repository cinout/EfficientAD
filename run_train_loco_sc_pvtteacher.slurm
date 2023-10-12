#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="train_sc"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00

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

python -u efficientad_tmp.py \
  --pretrained_network pvt2_b2li \
  --dataset mvtec_loco \
  --subdataset splicing_connectors \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --image_size_pvt2_teacher 1024 \
  --pvt2_teacher \
  --pvt2_stage3 \
  --output_dir "./outputs/output_20231012_005841_44_33_[sc]" \
  --note "pvt as teacher"
# FIXME: update job-name, weights, pretrained_network, subdataset, note
##Log this job's resource usage stats###
my-job-stats -a -n -s
##