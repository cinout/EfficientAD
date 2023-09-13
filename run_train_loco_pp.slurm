#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="train_pp"
#SBATCH --account=punim1623
#SBATCH --time=0-05:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate efficient

set -e
set -x

python -u efficientad.py --dataset mvtec_loco --subdataset pushpins --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train --note "add code book to encoder"

##Log this job's resource usage stats###
my-job-stats -a -n -s
##