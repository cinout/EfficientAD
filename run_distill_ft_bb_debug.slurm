#!/bin/bash

#SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="ds_ft_bb"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

export WORLD_SIZE=1  #### update world size: nodes x ntasks-per-node
export MASTER_PORT=12342

echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate ds2

set -e
set -x

python -u pretraining_after_ft.py \
  --network "vit" \
  --batch_size 2  \
  --extractor_input_size 512 \
  --subdataset breakfast_box \
  --ft_folder finetuned_vit_20231001_112430_27_40 \
  --note "distill after ft, bb" \
# FIXME: update gpu_size, batch_size
##Log this job's resource usage stats###
my-job-stats -a -n -s
##