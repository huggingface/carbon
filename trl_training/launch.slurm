#!/bin/bash
#SBATCH --job-name=carbon
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/leandro/logs/carbon/smol-%x-%j.out
#SBATCH --error=/fsx/leandro/logs/carbon/smol-%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --qos=high

module load cuda/12.9
set -x -e

source ~/.bashrc
conda activate carbon

# Distributed config
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=6000

# Force crash on NCCL issues
export NCCL_ASYNC_ERROR_HANDLING=1

export LAUNCHER="accelerate launch \
    --config_file accelerate_config.yaml \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    "

export CMD="pretrain.py"

srun --wait=60 --kill-on-bad-exit=1 bash -c "$LAUNCHER $CMD"