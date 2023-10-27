#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="pvt_patch"
#SBATCH --account=punim1623
#SBATCH --time=0-10:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

export WORLD_SIZE=2  #### update world size: nodes x ntasks-per-node
export MASTER_PORT=12340

echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate ds2

set -e
set -x

srun python -u pretraining.py \
  --network "mae_vit" \
  --batch_size 8  \
  --extractor_input_size 512 \
  --note "mae_vit, bs=8, size=512" \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##