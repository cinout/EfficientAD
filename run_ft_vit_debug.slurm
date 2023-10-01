#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="ft_vit"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00
### TODO: update time

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

srun python -u finetune.py \
  --epochs 2 \
  --note "finetune vit on mvtec_loco" \
### TODO: update epochs, note

##Log this job's resource usage stats###
my-job-stats -a -n -s
##