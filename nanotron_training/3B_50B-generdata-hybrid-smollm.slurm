#!/bin/bash
#SBATCH --job-name=3B-gener-hybrid-smollm3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/loubna/projects_v2/smollm3/nanotron/ablations/logs/carbon/3B_ablations/hybrid_tokenizer/3B-50B-generdata-hybrid-smollm3-%j.out
#SBATCH --qos=high
#SBATCH --begin=now+0minutes
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=loubna@huggingface.co

set -x
TRAINER_PYTHON_FILE=/fsx/loubna/projects_v2/smollm3/nanotron/run_train.py
nvidia-smi
echo "/!\\ please verify that you use the right env"
which python3
echo "START TIME: $(date)"

export PRINT_FILES=1
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export WANDB_ENTITY="huggingface"

export FORCE_UPDATE_CACHE_S3=1
export TMPDIR=/scratch
export CUDA_DEVICE_MAX_CONNECTIONS="1"

# Hybrid BP settings (must match config + tokenizer dna_config.json)
export ENABLE_HYBRID_BP=1
export HYBRID_BP_DNA_KMER_START_ID=${HYBRID_BP_DNA_KMER_START_ID:-128259}
export HYBRID_BP_DNA_KMER_END_ID=${HYBRID_BP_DNA_KMER_END_ID:-132481}
export HYBRID_BP_K=${HYBRID_BP_K:-6}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${HYBRID_BP_TOKENIZER_PATH:-}" ]]; then
    python "$SCRIPT_DIR/../verify_token_mask_sample.py" \
        --tokenizer_path "${HYBRID_BP_TOKENIZER_PATH}" \
        --k "${HYBRID_BP_K}"
fi

module load cuda/12.4
source /fsx/loubna/.venv-2-6-cu124/bin/activate
echo "PATH: $PATH"
echo "Environment variables:"
printenv | sort
echo "Python path: $(which python)"
echo "Torchrun path: $(which torchrun)"
python -m pip freeze
nvcc --version

echo go "$COUNT_NODE"
echo "$HOSTNAMES"

CMD=" \
    $TRAINER_PYTHON_FILE \
    --config-file nanotron_training/3B_50B-generdata-hybrid-smollm.yaml
    "

export LAUNCHER="torchrun \
    --nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep "$sleep_time"

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
